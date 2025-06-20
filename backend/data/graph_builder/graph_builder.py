import pandas as pd
import networkx as nx
from datetime import datetime

# Example schema for expected columns in the CSV:
# VPA_from, VPA_to, amount, timestamp, PSP, transaction_type, account_age_from, account_age_to

REQUIRED_COLUMNS = [
    'VPA_from', 'VPA_to', 'amount', 'timestamp', 'PSP', 'transaction_type', 'account_age_from', 'account_age_to'
]

def load_data(csv_path):
    """Load raw transaction data from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

def clean_data(df):
    """Clean and preprocess the transaction data."""
    # Drop rows with missing required fields
    df = df.dropna(subset=REQUIRED_COLUMNS)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # Remove rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    # Convert amount to float
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    # Standardize VPAs to lowercase
    df['VPA_from'] = df['VPA_from'].str.lower()
    df['VPA_to'] = df['VPA_to'].str.lower()
    return df

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

# Example usage:
# df = load_data('transactions.csv')
# df_clean = clean_data(df)
# G = build_graph(df_clean) 