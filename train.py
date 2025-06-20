import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from models.gnn_fraud_detection import NodeGNN, EdgeGNN

def evaluate_model(node_model, edge_model, data):
    """Evaluates the model on the test set and returns accuracy."""
    node_model.eval()
    edge_model.eval()
    
    with torch.no_grad():
        node_logits = node_model(data.x, data.edge_index)
        edge_logits = edge_model(data.x, data.edge_index, data.edge_attr)
        
        node_pred = node_logits.argmax(dim=1)
        edge_pred = edge_logits.argmax(dim=1)

        node_mask = data.node_test_mask
        edge_mask = data.edge_test_mask

        node_correct = (node_pred[node_mask] == data.y_node[node_mask]).sum()
        node_acc = int(node_correct) / int(node_mask.sum()) if int(node_mask.sum()) > 0 else 0
        
        edge_correct = (edge_pred[edge_mask] == data.y_edge[edge_mask]).sum()
        edge_acc = int(edge_correct) / int(edge_mask.sum()) if int(edge_mask.sum()) > 0 else 0
        
        return node_acc, edge_acc


def train(epochs=100, lr=0.01):
    """
    Loads processed data, splits it, trains GNN models, evaluates,
    and saves the trained model weights.
    """
    print("Loading processed data...")
    nodes_df = pd.read_csv('data/processed_nodes_synthetic.csv')
    edges_df = pd.read_csv('data/processed_edges_synthetic.csv')

    print("Building graph and creating train/test splits...")
    G = nx.from_pandas_edgelist(
        edges_df,
        source='Sender_UPI_ID',
        target='Receiver_UPI_ID',
        edge_attr=True,
        create_using=nx.DiGraph
    )
    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    
    suspicious_dict = nodes_df.set_index('UPI_ID')['is_suspicious'].to_dict()
    nx.set_node_attributes(G, {n: suspicious_dict.get(n, 0) for n in G.nodes()}, 'is_suspicious')

    edge_index = torch.tensor([[node_id_map[u], node_id_map[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
    
    # --- Feature Extraction ---
    # Node features: degrees and engineered stats (scaled)
    in_degrees = torch.tensor([G.in_degree(n) for n in G.nodes()], dtype=torch.float).view(-1, 1)
    out_degrees = torch.tensor([G.out_degree(n) for n in G.nodes()], dtype=torch.float).view(-1, 1)
    
    # Get engineered features from nodes_df and align them with the graph's node order
    node_feature_cols = [
        'sent_mean', 'sent_median', 'sent_std', 
        'received_mean', 'received_median', 'received_std', 'account_age_days'
    ]
    engineered_features_df = nodes_df.set_index('UPI_ID').loc[[n for n in G.nodes()]]
    engineered_features = torch.tensor(engineered_features_df[node_feature_cols].values, dtype=torch.float)

    # Combine all node features and scale them
    node_features_combined = torch.cat([in_degrees, out_degrees, engineered_features], dim=1)
    node_scaler = StandardScaler()
    node_features = torch.tensor(node_scaler.fit_transform(node_features_combined), dtype=torch.float)

    # Re-order edges_df to match the graph's internal edge order before creating tensors
    edge_list_for_ordering = list(G.edges(data=True))
    edge_order_df = pd.DataFrame(
        [(u, v, d['Amount_INR'], d['time_since_last_tx_seconds']) for u, v, d in edge_list_for_ordering],
        columns=['Sender_UPI_ID', 'Receiver_UPI_ID', 'Amount_INR', 'time_since_last_tx_seconds']
    )

    # Edge features: amount and time delta (scaled)
    amounts = torch.tensor(edge_order_df['Amount_INR'].values, dtype=torch.float).view(-1, 1)
    time_deltas = torch.tensor(edge_order_df['time_since_last_tx_seconds'].values, dtype=torch.float).view(-1, 1)
    
    edge_features_combined = torch.cat([amounts, time_deltas], dim=1)
    edge_scaler = StandardScaler()
    edge_features = torch.tensor(edge_scaler.fit_transform(edge_features_combined), dtype=torch.float)

    # Get ground truth labels directly from the final graph object to ensure consistency
    node_labels = torch.tensor([G.nodes[node]['is_suspicious'] for node in G.nodes()], dtype=torch.long)
    edge_labels = torch.tensor([d['is_fraud'] for u, v, d in G.edges(data=True)], dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    data.y_node = node_labels
    data.y_edge = edge_labels
    
    # --- 2. Create Train/Test Masks ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes > 0:
        node_indices = np.arange(num_nodes)
        _, node_test_indices = train_test_split(node_indices, test_size=0.2, random_state=42, stratify=node_labels)
        node_train_mask = torch.ones(num_nodes, dtype=torch.bool)
        node_train_mask[node_test_indices] = 0
        data.node_train_mask = node_train_mask
        data.node_test_mask = ~node_train_mask
    else:
        data.node_train_mask = torch.empty(0, dtype=torch.bool)
        data.node_test_mask = torch.empty(0, dtype=torch.bool)

    if num_edges > 0:
        edge_indices = np.arange(num_edges)
        if len(np.unique(edge_labels)) > 1:
            _, edge_test_indices = train_test_split(edge_indices, test_size=0.2, random_state=42, stratify=edge_labels)
        else: # Cannot stratify with only one class
            _, edge_test_indices = train_test_split(edge_indices, test_size=0.2, random_state=42)
        edge_train_mask = torch.ones(num_edges, dtype=torch.bool)
        edge_train_mask[edge_test_indices] = 0
        data.edge_train_mask = edge_train_mask
        data.edge_test_mask = ~edge_train_mask
    else:
        data.edge_train_mask = torch.empty(0, dtype=torch.bool)
        data.edge_test_mask = torch.empty(0, dtype=torch.bool)

    print("Initializing models...")
    node_model = NodeGNN(in_channels=data.x.shape[1], hidden_channels=16, out_channels=2)
    edge_model = EdgeGNN(node_in=data.x.shape[1], edge_in=data.edge_attr.shape[1], hidden=16, out=2)
    optimizer = torch.optim.Adam(list(node_model.parameters()) + list(edge_model.parameters()), lr=lr)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        node_model.train()
        edge_model.train()
        optimizer.zero_grad()

        node_logits = node_model(data.x, data.edge_index)
        edge_logits = edge_model(data.x, data.edge_index, data.edge_attr)
        
        if data.node_train_mask.sum() > 0:
            loss_node = F.cross_entropy(node_logits[data.node_train_mask], data.y_node[data.node_train_mask])
        else:
            loss_node = torch.tensor(0.0)

        if data.edge_train_mask.sum() > 0:
            loss_edge = F.cross_entropy(edge_logits[data.edge_train_mask], data.y_edge[data.edge_train_mask])
        else:
            loss_edge = torch.tensor(0.0)
            
        total_loss = loss_node + loss_edge
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f}')

    print("\nTraining finished. Evaluating on test set...")
    node_accuracy, edge_accuracy = evaluate_model(node_model, edge_model, data)
    print(f"  -> Node Classification Accuracy: {node_accuracy*100:.2f}%")
    print(f"  -> Edge Classification Accuracy: {edge_accuracy*100:.2f}%")

    print("\nSaving models...")
    torch.save(node_model.state_dict(), 'models/node_gnn_model.pth')
    torch.save(edge_model.state_dict(), 'models/edge_gnn_model.pth')
    print("Models saved successfully.")

if __name__ == '__main__':
    train() 