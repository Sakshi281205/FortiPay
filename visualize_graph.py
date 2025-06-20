import pandas as pd
import networkx as nx
from pyvis.network import Network
from datetime import datetime, timedelta

def create_user_centric_visualization(
    focus_user_upi: str,
    time_window_days: int = 30,
    nodes_path='data/processed_nodes_synthetic.csv',
    edges_path='data/processed_edges_synthetic.csv',
    output_path='user_centric_graph.html'
):
    """
    Creates an interactive, user-centric HTML graph visualization for a specific
    user within a given time window.
    """
    print(f"Loading data to generate a graph for user '{focus_user_upi}'...")
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    edges_df['Unix_Timestamp'] = pd.to_datetime(edges_df['Unix_Timestamp'], unit='s')
    # Convert Timestamp to string to prevent JSON serialization error
    edges_df['Timestamp'] = edges_df['Unix_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # --- 1. Filter Data for Time Window ---
    print(f"Filtering transactions for the last {time_window_days} days...")
    end_date = pd.to_datetime(edges_df['Timestamp']).max()
    start_date = end_date - timedelta(days=time_window_days)
    
    time_filtered_edges = edges_df[pd.to_datetime(edges_df['Timestamp']) >= start_date]

    # --- 2. Create User-Centric "Ego" Graph ---
    print("Building user-centric graph...")
    
    # Find all transactions involving the focus user
    user_edges = time_filtered_edges[
        (time_filtered_edges['Sender_UPI_ID'] == focus_user_upi) |
        (time_filtered_edges['Receiver_UPI_ID'] == focus_user_upi)
    ]

    if user_edges.empty:
        print(f"No transactions found for user '{focus_user_upi}' in the last {time_window_days} days.")
        return

    # Build the graph from these specific edges
    G = nx.from_pandas_edgelist(
        user_edges,
        source='Sender_UPI_ID',
        target='Receiver_UPI_ID',
        edge_attr=['Amount_INR', 'is_fraud', 'Timestamp'],
        create_using=nx.DiGraph
    )
    
    suspicious_dict = nodes_df.set_index('UPI_ID')['is_suspicious'].to_dict()
    nx.set_node_attributes(G, {n: suspicious_dict.get(n, 0) for n in G.nodes()}, 'is_suspicious')

    # --- 3. Create and Customize PyVis Graph ---
    print("Creating PyVis interactive graph...")
    net = Network(height='800px', width='100%', notebook=False, directed=True)
    net.from_nx(G)

    for node in net.nodes:
        is_suspicious = G.nodes[node['id']].get('is_suspicious', 0)
        
        # Highlight the focus node
        if node['id'] == focus_user_upi:
            node['color'] = '#f1c40f' # Gold
            node['size'] = 30
        elif is_suspicious:
            node['color'] = 'red'
            node['size'] = 20
        else:
            node['color'] = '#3498db'
            node['size'] = 15
        
        node['title'] = f"UPI ID: {node['id']}<br>Suspicious: {'Yes' if is_suspicious else 'No'}"

    for edge in net.edges:
        try:
            edge_data = G.get_edge_data(edge['from'], edge['to'])
            is_fraud = edge_data.get('is_fraud', 0) if isinstance(edge_data, dict) else edge_data[0].get('is_fraud', 0)
        except (KeyError, AttributeError):
            is_fraud = 0

        if is_fraud:
            edge['color'] = '#e74c3c'
            edge['width'] = 3
        else:
            edge['color'] = '#bdc3c7'
            edge['width'] = 1
        
        # Add transaction amount to hover title
        amount = G.get_edge_data(edge['from'], edge['to']).get('Amount_INR', 'N/A')
        edge['title'] = f"Amount: {amount:.2f} INR"


    net.set_options("""
    var options = {
      "physics": { "solver": "repulsion", "repulsion": { "nodeDistance": 200 } },
      "interaction": { "hover": true, "navigationButtons": true }
    }
    """)
    
    final_output_path = f"{focus_user_upi.split('@')[0]}_graph.html"
    net.save_graph(final_output_path)
    print(f"\nSuccessfully created interactive graph: '{final_output_path}'")
    print("You can now open this HTML file in your web browser.")

if __name__ == "__main__":
    # Example: Find a suspicious user from our data to visualize
    nodes = pd.read_csv('data/processed_nodes_synthetic.csv')
    suspicious_users = nodes[nodes['is_suspicious'] == 1]
    
    if not suspicious_users.empty:
        example_user = suspicious_users['UPI_ID'].iloc[0]
        create_user_centric_visualization(focus_user_upi=example_user, time_window_days=90)
    else:
        print("No suspicious users found to generate an example visualization.") 