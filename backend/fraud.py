import random

def predict_fraud(transaction):
    # For demo: random risk score and prediction
    risk_score = round(random.uniform(0, 1), 2)
    prediction = 'fraud' if risk_score > 0.8 else 'legit'
    return risk_score, prediction

def explain_alert(transaction, risk_score, prediction):
    if prediction == 'fraud':
        return f"Transaction flagged: risk score {risk_score}. Pattern matches known fraud scenario."
    else:
        return "No suspicious activity detected." 