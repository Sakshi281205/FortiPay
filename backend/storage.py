import uuid
from fraud import predict_fraud

TRANSACTIONS = {}
RISK_SCORES = {}

def save_transaction(transaction):
    tx_id = str(uuid.uuid4())
    TRANSACTIONS[tx_id] = transaction
    risk_score, prediction = predict_fraud(transaction)
    RISK_SCORES[tx_id] = (risk_score, prediction)
    return tx_id

def get_transaction(tx_id):
    return TRANSACTIONS.get(tx_id)

def get_risk_score(tx_id):
    return RISK_SCORES.get(tx_id, (None, None)) 