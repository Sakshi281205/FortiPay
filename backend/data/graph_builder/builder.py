import networkx as nx
from .schema import NODE_ATTRIBUTES, EDGE_ATTRIBUTES

def build_graph(df):
    """Build a directed transaction graph from the DataFrame."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        # Add nodes with attributes
        if not G.has_node(row['VPA_from']):
            G.add_node(row['VPA_from'], account_age=row['account_age_from'])
        if not G.has_node(row['VPA_to']):
            G.add_node(row['VPA_to'], account_age=row['account_age_to'])
        # Add edge with transaction attributes
        G.add_edge(
            row['VPA_from'],
            row['VPA_to'],
            amount=row['amount'],
            timestamp=row['timestamp'],
            PSP=row['PSP'],
            transaction_type=row['transaction_type']
        )
    return G 