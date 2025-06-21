import pandas as pd
import numpy as np

def prepare_data(input_path='transactions_synthetic.csv', output_dir='data/'):
    """
    Loads raw transaction data from the synthetic dataset, cleans it, 
    identifies suspicious nodes based on fraud labels, and saves 
    processed node and edge files.
    """
    print(f"Loading raw data from '{input_path}'...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please ensure the dataset exists.")
        return

    # --- 1. Data Cleaning and Preprocessing ---
    print("Cleaning and preprocessing data...")
    # The synthetic data already has the correct columns, but we'll ensure consistency.
    
    # Convert timestamp to datetime and then to unix for calculations
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Unix_Timestamp'] = df['Timestamp'].astype(np.int64) // 10**9

    # Filter for successful transactions (all are success in synthetic, but good practice)
    df = df[df['Status'] == 'SUCCESS'].copy()

    # Sort by user and time to calculate time deltas
    df = df.sort_values(by=['Sender_UPI_ID', 'Unix_Timestamp'])

    # Calculate time since sender's last transaction
    df['time_since_last_tx_seconds'] = df.groupby('Sender_UPI_ID')['Unix_Timestamp'].diff().fillna(0)

    # --- 2. Identify Suspicious Nodes from Fraud Labels ---
    print("Identifying suspicious nodes from existing fraud labels...")
    
    # A node is suspicious if it was involved in any fraudulent transaction
    fraudulent_transactions = df[df['is_fraud'] == 1]
    suspicious_senders = fraudulent_transactions['Sender_UPI_ID']
    suspicious_receivers = fraudulent_transactions['Receiver_UPI_ID']
    suspicious_nodes = set(pd.concat([suspicious_senders, suspicious_receivers]).unique())
    
    print(f"Identified {len(suspicious_nodes)} suspicious nodes from {fraudulent_transactions.shape[0]} fraudulent transactions.")

    # --- 3. Create Node and Edge Lists with Enhanced Features ---
    print("Creating final node and edge lists with enhanced features...")

    # Edges are the transactions
    edges_df = df[[
        'Sender_UPI_ID', 'Receiver_UPI_ID', 'Amount_INR', 'Unix_Timestamp', 'is_fraud', 'time_since_last_tx_seconds'
    ]]

    # Get all unique user UPIs
    all_upi_ids = pd.concat([df['Sender_UPI_ID'], df['Receiver_UPI_ID']]).unique()
    nodes_df = pd.DataFrame({'UPI_ID': all_upi_ids})
    
    # --- Feature Engineering ---
    # Calculate transaction statistics for each user
    sender_stats = df.groupby('Sender_UPI_ID')['Amount_INR'].agg(['mean', 'median', 'std']).rename(columns=lambda x: f'sent_{x}')
    receiver_stats = df.groupby('Receiver_UPI_ID')['Amount_INR'].agg(['mean', 'median', 'std']).rename(columns=lambda x: f'received_{x}')
    
    # Account age (time of first transaction)
    first_tx_time = df.groupby('Sender_UPI_ID')['Unix_Timestamp'].min()
    account_age = (df['Unix_Timestamp'].max() - first_tx_time) / (3600*24) # in days
    account_age = account_age.rename('account_age_days')

    # --- Bust-Out Fraud Feature Engineering ---
    # Historical transaction frequency (transactions per day)
    tx_counts = df.groupby('Sender_UPI_ID').size()
    historical_tx_frequency = tx_counts / (account_age.loc[tx_counts.index] + 1) # Add 1 to avoid division by zero
    historical_tx_frequency = historical_tx_frequency.rename('historical_tx_frequency')

    # Inflow spike: Compare latest day's inflow to historical average
    df['date'] = df['Timestamp'].dt.date
    daily_inflow = df.groupby(['Receiver_UPI_ID', 'date'])['Amount_INR'].sum().reset_index()
    historical_avg_inflow = daily_inflow.groupby('Receiver_UPI_ID')['Amount_INR'].mean().rename('historical_avg_daily_inflow')
    
    # Correctly get the last inflow amount per user without creating a multi-index
    latest_day_inflow = daily_inflow.groupby('Receiver_UPI_ID').last()['Amount_INR']
    inflow_spike_ratio = (latest_day_inflow / (historical_avg_inflow + 1e-6)).rename('inflow_spike_ratio')

    # Fund dispersal velocity (average time to send money after receiving)
    # This is a complex feature, approximating by comparing send/receive timestamps
    received_times = df.groupby('Receiver_UPI_ID')['Unix_Timestamp'].min()
    sent_times = df.groupby('Sender_UPI_ID')['Unix_Timestamp'].min()
    
    # Ensure we are subtracting correctly and handle cases where users only send or receive
    dispersal_time = (sent_times - received_times).rename('fund_dispersal_velocity_seconds')

    # Merge features into the nodes dataframe
    nodes_df = nodes_df.merge(sender_stats, left_on='UPI_ID', right_index=True, how='left')
    nodes_df = nodes_df.merge(receiver_stats, left_on='UPI_ID', right_index=True, how='left')
    nodes_df = nodes_df.merge(account_age, left_on='UPI_ID', right_index=True, how='left')
    nodes_df = nodes_df.merge(historical_tx_frequency, left_on='UPI_ID', right_index=True, how='left')
    nodes_df = nodes_df.merge(inflow_spike_ratio, left_on='UPI_ID', right_index=True, how='left')
    nodes_df = nodes_df.merge(dispersal_time, left_on='UPI_ID', right_index=True, how='left')
    nodes_df.fillna(0, inplace=True) # Fill NaNs for users who only send or only receive

    # Add suspicious flag
    nodes_df['is_suspicious'] = nodes_df['UPI_ID'].isin(suspicious_nodes).astype(int)

    # --- 4. Save Processed Data ---
    nodes_path = f"{output_dir}processed_nodes_synthetic.csv"
    edges_path = f"{output_dir}processed_edges_synthetic.csv"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    print(f"Successfully saved processed synthetic nodes to {nodes_path}")
    print(f"Successfully saved processed synthetic edges to {edges_path}")

if __name__ == "__main__":
    prepare_data() 