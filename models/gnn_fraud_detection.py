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
# 5. Use Case Specific Detectors
# -----------------------------
def detect_qr_swap_fraud(G: nx.DiGraph, shopkeeper_node: str, current_timestamp: int, lookback_days=30, suspicion_days=2):
    """
    Detects potential QR code swap fraud for a specific shopkeeper.

    Args:
        G (nx.DiGraph): The full transaction graph. Edges need a 'timestamp' attribute.
        shopkeeper_node (str): The node ID of the legitimate shopkeeper.
        current_timestamp (int): The current unix timestamp for time-window calculations.
        lookback_days (int): How many days back to look to identify regular customers.
        suspicion_days (int): The most recent number of days to check for the fraud pattern.

    Returns:
        A dictionary with the suspected fraudster and the affected customers, or None.
    """
    lookback_window_start = current_timestamp - (lookback_days * 86400)
    suspicion_window_start = current_timestamp - (suspicion_days * 86400)

    # 1. Identify regular customers from the lookback period (before the suspicion window)
    regular_customers = set()
    for u, v, attr in G.in_edges(shopkeeper_node, data=True):
        if lookback_window_start <= attr['timestamp'] < suspicion_window_start:
            regular_customers.add(u)

    if not regular_customers:
        print(f"No regular customers found for {shopkeeper_node} in the lookback period.")
        return None

    # 2. Find regulars who have stopped transacting with the shopkeeper recently
    lapsed_regulars = set()
    for customer in regular_customers:
        recent_tx = any(
            suspicion_window_start <= attr['timestamp']
            for u, v, attr in G.out_edges(customer, data=True)
            if v == shopkeeper_node
        )
        if not recent_tx:
            lapsed_regulars.add(customer)

    if not lapsed_regulars:
        print("All regular customers have continued to transact. No signs of QR swap.")
        return None

    # 3. Find who these lapsed regulars are paying now
    potential_suspects = {}
    for customer in lapsed_regulars:
        for u, v, attr in G.out_edges(customer, data=True):
            # Check if the transaction is recent and not to the original shopkeeper
            if v != shopkeeper_node and suspicion_window_start <= attr['timestamp']:
                # Check for no prior history with this new payee
                has_prior_history = any(
                    attr_hist['timestamp'] < suspicion_window_start
                    for _, _, attr_hist in G.out_edges(customer, data=True)
                    if v == v
                )
                if not has_prior_history:
                    potential_suspects.setdefault(v, set()).add(customer)

    # 4. Identify the most likely suspect
    if not potential_suspects:
        print("Lapsed regulars are not transacting with any single new entity.")
        return None

    # The suspect is the one who received funds from the most lapsed regulars
    suspect_node, paid_by = max(potential_suspects.items(), key=lambda item: len(item[1]))

    # Rule: at least 2 regulars must have paid the suspect to be considered a pattern
    if len(paid_by) < 2:
        print("No single new entity is receiving payments from multiple lapsed regulars.")
        return None

    return {
        'shopkeeper_node': shopkeeper_node,
        'suspect_node': suspect_node,
        'affected_customers': list(paid_by),
        'explanation': f"Potential QR Swap: {len(paid_by)} regular customers of '{shopkeeper_node}' have stopped paying them and recently paid new account '{suspect_node}', with whom they have no prior history."
    }

# -----------------------------
# 6. Example Usage
# -----------------------------
if __name__ == "__main__":
    # --- Original GNN prediction example ---
    G_gnn = nx.DiGraph()
    G_gnn.add_node('A', vpa='merchant@upi', account_age=2, is_verified=True)
    G_gnn.add_node('B', vpa='user1@upi', account_age=0.1, is_verified=False)
    G_gnn.add_node('C', vpa='user2@upi', account_age=0.2, is_verified=False)
    G_gnn.add_edge('B', 'A', amount=1000, timestamp=1667281800, psp_flag=0, type=1)
    G_gnn.add_edge('C', 'A', amount=500, timestamp=1667281801, psp_flag=0, type=1)
    G_gnn.add_edge('A', 'B', amount=100, timestamp=1667281802, psp_flag=1, type=2)
    known_vpas = ['merchant@upi']

    print("--- GNN Fraud Prediction ---")
    result = predict_fraud(G_gnn, known_vpas)
    print(result)
    print("-" * 20)

    # --- QR Swap Fraud Example ---
    G_qr = nx.DiGraph()
    shopkeeper = 'Shopkeeper_Real_VPA'
    fraudster = 'Fraudster_New_VPA'
    regulars = [f'Regular_{i}' for i in range(5)]
    current_time = 1672531200 # Jan 1, 2023

    # Regulars paid shopkeeper in the past (e.g., 15 days ago)
    for r in regulars:
        G_qr.add_edge(r, shopkeeper, timestamp=current_time - 15 * 86400)

    # Now, regulars pay the fraudster instead (e.g., 1 day ago)
    for r in regulars:
        G_qr.add_edge(r, fraudster, timestamp=current_time - 1 * 86400)
    
    # One regular also pays the shopkeeper (to show it can handle imperfect patterns)
    G_qr.add_edge('Regular_0', shopkeeper, timestamp=current_time - 1 * 86400)

    print("\n--- QR Swap Fraud Detection ---")
    qr_fraud_result = detect_qr_swap_fraud(
        G_qr,
        shopkeeper_node=shopkeeper,
        current_timestamp=current_time
    )
    if qr_fraud_result:
        print(qr_fraud_result['explanation'])
        # print(qr_fraud_result) 