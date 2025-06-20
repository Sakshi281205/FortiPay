import pandas as pd

REQUIRED_COLUMNS = [
    'VPA_from', 'VPA_to', 'amount', 'timestamp', 'PSP', 'transaction_type', 'account_age_from', 'account_age_to'
]

def load_data(csv_path):
    """Load raw transaction data from a CSV file."""
    df = pd.read_csv(csv_path)
    return df

def clean_data(df):
    """Clean and preprocess the transaction data."""
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    df['VPA_from'] = df['VPA_from'].str.lower()
    df['VPA_to'] = df['VPA_to'].str.lower()
    return df

def preprocess_data(df):
    """Optional: further feature engineering or normalization."""
    # Example: add hour of transaction
    df['hour'] = df['timestamp'].dt.hour
    return df 