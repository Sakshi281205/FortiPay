"""
GNN-Powered Fraud Detection for UPI Transactions
------------------------------------------------
- Node & Edge Classification using PyTorch Geometric
- Feature engineering: in/out degree, transaction velocity, temporal patterns, etc.
- Fuzzy VPA matching for spoof detection
- Outputs risk scores and fraud predictions
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import degree
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance
import networkx as nx
import numpy as np

# -----------------------------
# 1. Feature Engineering Utils
# -----------------------------
def compute_graph_features(G: nx.DiGraph):
    """
    Compute node and edge features for the transaction graph.
    Features: in/out degree, transaction velocity, temporal patterns, etc.
    """
    node_features = []
    node_id_map = {n: i for i, n in enumerate(G.nodes())}
    for node in G.nodes(data=True):
        n = node[0]
        attr = node[1]
        in_deg = G.in_degree(n)
        out_deg = G.out_degree(n)
        tx_times = [G.edges[e]['timestamp'] for e in G.in_edges(n)]
        tx_velocity = np.std(tx_times) if len(tx_times) > 1 else 0
        account_age = attr.get('account_age', 0)
        is_verified = int(attr.get('is_verified', False))
        node_features.append([
            in_deg, out_deg, tx_velocity, account_age, is_verified
        ])
    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_features = []
    edge_index = [[], []]
    for u, v, attr in G.edges(data=True):
        amount = attr.get('amount', 0)
        timestamp = attr.get('timestamp', 0)
        psp_flag = int(attr.get('psp_flag', 0))
        tx_type = attr.get('type', 0)
        edge_features.append([
            amount, timestamp, psp_flag, tx_type
        ])
        edge_index[0].append(node_id_map[u])
        edge_index[1].append(node_id_map[v])
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return node_features, edge_features, edge_index, node_id_map

# -----------------------------
# 2. Fuzzy VPA Matching
# -----------------------------
def fuzzy_vpa_score(vpa, known_vpas, threshold=85):
    """
    Returns True if VPA is similar to any known VPA above threshold (spoof detection).
    """
    for known in known_vpas:
        fuzz_score = fuzz.ratio(vpa, known)
        lev_score = levenshtein_distance(vpa, known)
        if fuzz_score >= threshold or lev_score <= 2:
            return True
    return False

# -----------------------------
# 3. GNN Models
# -----------------------------
class NodeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EdgeGNN(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden, out):
        super().__init__()
        self.node_encoder = torch.nn.Linear(node_in, hidden)
        self.edge_encoder = torch.nn.Linear(edge_in, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, out)
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.node_encoder(x))
        edge_attr = F.relu(self.edge_encoder(edge_attr))
        x = self.conv1(x, edge_index)
        # Edge classification: aggregate node and edge features
        edge_logits = self.classifier((x[edge_index[0]] + x[edge_index[1]] + edge_attr) / 3)
        return edge_logits

# -----------------------------
# 4. Main Pipeline
# -----------------------------
def predict_fraud(G: nx.DiGraph, known_vpas=None, node_model=None, edge_model=None):
    """
    Takes a NetworkX DiGraph of transactions and outputs risk scores and fraud predictions.
    known_vpas: list of trusted merchant VPAs for spoof detection
    node_model, edge_model: pretrained GNN models (if None, random init)
    """
    if known_vpas is None:
        known_vpas = []
    node_features, edge_features, edge_index, node_id_map = compute_graph_features(G)
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    # Node classification (e.g., suspicious account)
    if node_model is None:
        node_model = NodeGNN(node_features.shape[1], 16, 2)  # 2: normal/suspicious
    node_logits = node_model(data.x, data.edge_index)
    node_probs = torch.sigmoid(node_logits)
    node_preds = (node_probs[:, 1] > 0.8).int()  # threshold for suspicious

    # Edge classification (e.g., fraudulent transaction)
    if edge_model is None:
        edge_model = EdgeGNN(node_features.shape[1], edge_features.shape[1], 16, 2)
    edge_logits = edge_model(data.x, data.edge_index, data.edge_attr)
    edge_probs = torch.sigmoid(edge_logits)
    edge_preds = (edge_probs[:, 1] > 0.8).int()

    # Fuzzy VPA spoof detection
    vpa_spoof_flags = {}
    for node, idx in node_id_map.items():
        vpa = G.nodes[node].get('vpa', '')
        vpa_spoof_flags[node] = fuzzy_vpa_score(vpa, known_vpas)

    # Risk scores (0-1) for each transaction (edge)
    risk_scores = edge_probs[:, 1].detach().numpy()
    fraud_edges = [e for i, e in enumerate(G.edges()) if edge_preds[i]]
    suspicious_nodes = [n for i, n in enumerate(G.nodes()) if node_preds[i]]
    spoofed_vpas = [n for n, flag in vpa_spoof_flags.items() if flag]

    return {
        'risk_scores': risk_scores,
        'fraud_edges': fraud_edges,
        'suspicious_nodes': suspicious_nodes,
        'spoofed_vpas': spoofed_vpas
    }

# -----------------------------
# 5. Example Usage
# -----------------------------
if __name__ == "__main__":
    # Example: create a toy transaction graph
    G = nx.DiGraph()
    G.add_node('A', vpa='merchant@upi', account_age=2, is_verified=True)
    G.add_node('B', vpa='user1@upi', account_age=0.1, is_verified=False)
    G.add_node('C', vpa='user2@upi', account_age=0.2, is_verified=False)
    G.add_edge('B', 'A', amount=1000, timestamp=1, psp_flag=0, type=1)
    G.add_edge('C', 'A', amount=500, timestamp=2, psp_flag=0, type=1)
    G.add_edge('A', 'B', amount=100, timestamp=3, psp_flag=1, type=2)
    known_vpas = ['merchant@upi']
    result = predict_fraud(G, known_vpas)
    print(result) 