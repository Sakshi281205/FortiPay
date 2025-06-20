import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta

def generate_dataset(num_users=2000, num_transactions=20000, fraud_ratio=0.05):
    """
    Generates a large synthetic dataset of UPI transactions with various fraud patterns.

    Args:
        num_users (int): The number of unique users to create.
        num_transactions (int): The total number of transactions to generate.
        fraud_ratio (float): The approximate percentage of transactions that should be fraudulent.
    """
    print(f"Starting dataset synthesis: {num_users} users, {num_transactions} transactions...")

    # --- 1. User Generation ---
    users = pd.DataFrame({
        'user_id': range(num_users),
        'upi_id': [f'user{i}@bank' for i in range(num_users)],
        'name': [f'User {i}' for i in range(num_users)]
    })
    
    transactions = []
    fraud_flags = [] # 0 for normal, 1 for fraud

    # --- 2. Fraud Injection Functions ---
    def inject_money_muling(user_pool, start_time):
        """A -> B -> C chain"""
        mule_chain = random.sample(user_pool, 3)
        t1_time = start_time + timedelta(minutes=random.randint(1, 10))
        t2_time = t1_time + timedelta(minutes=random.randint(1, 10))
        t3_time = t2_time + timedelta(minutes=random.randint(1, 10))
        
        tx1 = (mule_chain[0], mule_chain[1], round(random.uniform(1000, 5000), 2), t1_time)
        tx2 = (mule_chain[1], mule_chain[2], round(random.uniform(900, 4900), 2), t2_time)
        return [tx1, tx2], {mule_chain[0], mule_chain[1], mule_chain[2]}

    def inject_fan_in_out(user_pool, start_time):
        """Multiple users pay one mule, who then pays a final destination."""
        mule = random.choice(user_pool)
        destination = random.choice([u for u in user_pool if u != mule])
        num_senders = random.randint(5, 15)
        senders = random.sample([u for u in user_pool if u not in [mule, destination]], num_senders)
        
        txs = []
        total_amount = 0
        for sender in senders:
            amount = round(random.uniform(100, 500), 2)
            total_amount += amount
            tx_time = start_time + timedelta(minutes=random.randint(1, 30))
            txs.append((sender, mule, amount, tx_time))
        
        # Fan-out transaction
        txs.append((mule, destination, total_amount * 0.95, tx_time + timedelta(minutes=5)))
        return txs, set(senders + [mule, destination])

    # --- 3. Generate Transactions ---
    num_fraud_tx = int(num_transactions * fraud_ratio)
    fraud_tx_count = 0
    
    # Generate fraud transactions first
    print(f"Generating ~{num_fraud_tx} fraudulent transactions...")
    user_list = list(users['upi_id'])
    start_date = datetime(2023, 1, 1)
    
    while fraud_tx_count < num_fraud_tx:
        current_time = start_date + timedelta(hours=random.randint(fraud_tx_count, fraud_tx_count + 100))
        fraud_type = random.choice(['mule', 'fan'])
        
        if fraud_type == 'mule':
            new_txs, _ = inject_money_muling(user_list, current_time)
        else:
            new_txs, _ = inject_fan_in_out(user_list, current_time)
            
        transactions.extend(new_txs)
        fraud_flags.extend([1] * len(new_txs))
        fraud_tx_count += len(new_txs)

    # Generate normal transactions
    print("Generating normal transactions...")
    num_normal_tx = num_transactions - len(transactions)
    for i in range(num_normal_tx):
        sender, receiver = random.sample(user_list, 2)
        amount = round(random.uniform(50, 2000), 2)
        tx_time = start_date + timedelta(hours=random.randint(i, i + 100))
        transactions.append((sender, receiver, amount, tx_time))
        fraud_flags.append(0)

    # --- 4. Final DataFrame Assembly ---
    print("Assembling final DataFrame...")
    df = pd.DataFrame(transactions, columns=['Sender_UPI_ID', 'Receiver_UPI_ID', 'Amount_INR', 'Timestamp'])
    df['is_fraud'] = fraud_flags

    # Add other necessary columns
    df['Transaction_ID'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['Status'] = 'SUCCESS'
    
    # Add names
    user_name_map = users.set_index('upi_id')['name'].to_dict()
    df['Sender_Name'] = df['Sender_UPI_ID'].map(user_name_map)
    df['Receiver_Name'] = df['Receiver_UPI_ID'].map(user_name_map)
    
    # Reorder and shuffle
    df = df[[
        'Transaction_ID', 'Timestamp', 'Sender_Name', 'Sender_UPI_ID',
        'Receiver_Name', 'Receiver_UPI_ID', 'Amount_INR', 'Status', 'is_fraud'
    ]]
    df = df.sample(frac=1).reset_index(drop=True)

    # --- 5. Save to CSV ---
    output_filename = 'transactions_synthetic.csv'
    df.to_csv(output_filename, index=False)
    print(f"Successfully generated synthetic dataset at '{output_filename}'")
    print(f"Total transactions: {len(df)}, Fraudulent: {df['is_fraud'].sum()}")


if __name__ == "__main__":
    generate_dataset() 