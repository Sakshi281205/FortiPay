# Node and edge schema definitions for the transaction graph

NODE_ATTRIBUTES = [
    'VPA', 'account_age'
]

EDGE_ATTRIBUTES = [
    'amount', 'timestamp', 'PSP', 'transaction_type'
]

# Example schema dictionaries (for reference)
NODE_SCHEMA = {
    'VPA': str,
    'account_age': float
}

EDGE_SCHEMA = {
    'amount': float,
    'timestamp': 'datetime64[ns]',
    'PSP': str,
    'transaction_type': str
} 