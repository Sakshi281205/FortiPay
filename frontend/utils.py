import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

def detect_star_fraud(df, threshold=0.8):
    """Detect star-shaped fraud patterns"""
    # Group by recipient VPA and count unique senders
    recipient_counts = df.groupby('VPA_to')['VPA_from'].nunique()
    star_suspects = recipient_counts[recipient_counts > 5].index
    
    star_frauds = []
    for suspect in star_suspects:
        suspect_transactions = df[df['VPA_to'] == suspect]
        if suspect_transactions['risk_score'].mean() > threshold:
            star_frauds.append({
                'center_vpa': suspect,
                'sender_count': len(suspect_transactions['VPA_from'].unique()),
                'avg_risk': suspect_transactions['risk_score'].mean(),
                'total_amount': suspect_transactions['amount'].sum()
            })
    
    return pd.DataFrame(star_frauds)

def detect_cycle_fraud(df, cycle_length=3, threshold=0.7):
    """Detect cycle fraud patterns"""
    cycles = []
    
    # Create graph from transactions
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['VPA_from'], row['VPA_to'], 
                  amount=row['amount'], 
                  timestamp=row['timestamp'])
    
    # Find cycles
    try:
        cycles_found = list(nx.simple_cycles(G))
        for cycle in cycles_found:
            if len(cycle) == cycle_length:
                cycle_transactions = []
                for i in range(len(cycle)):
                    from_vpa = cycle[i]
                    to_vpa = cycle[(i + 1) % len(cycle)]
                    edge_data = G.get_edge_data(from_vpa, to_vpa)
                    if edge_data:
                        cycle_transactions.append({
                            'from_vpa': from_vpa,
                            'to_vpa': to_vpa,
                            'amount': edge_data['amount'],
                            'timestamp': edge_data['timestamp']
                        })
                
                if cycle_transactions:
                    cycles.append({
                        'cycle_nodes': cycle,
                        'transaction_count': len(cycle_transactions),
                        'total_amount': sum(t['amount'] for t in cycle_transactions)
                    })
    except:
        pass
    
    return cycles

def calculate_risk_metrics(df):
    """Calculate comprehensive risk metrics"""
    metrics = {
        'total_transactions': len(df),
        'high_risk_count': len(df[df['risk_score'] > 0.8]),
        'medium_risk_count': len(df[(df['risk_score'] > 0.5) & (df['risk_score'] <= 0.8)]),
        'low_risk_count': len(df[df['risk_score'] <= 0.5]),
        'avg_risk_score': df['risk_score'].mean(),
        'avg_confidence': df['confidence'].mean(),
        'fraud_count': len(df[df['fraud_type'] != 'normal']),
        'total_amount': df['amount'].sum(),
        'high_value_fraud_amount': df[df['fraud_type'] == 'high_value_fraud']['amount'].sum()
    }
    
    return metrics

def generate_time_series_data(df, freq='D'):
    """Generate time series data for fraud analysis"""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
    
    daily_stats = df_copy.groupby('date').agg({
        'transaction_id': 'count',
        'risk_score': 'mean',
        'amount': 'sum',
        'fraud_type': lambda x: (x != 'normal').sum()
    }).reset_index()
    
    daily_stats.columns = ['date', 'transaction_count', 'avg_risk_score', 'total_amount', 'fraud_count']
    
    return daily_stats

def create_alert_summary(df):
    """Create summary of alerts by severity"""
    alerts = {
        'critical': df[(df['risk_score'] > 0.9) & (df['confidence'] > 0.9)],
        'high': df[(df['risk_score'] > 0.8) & (df['confidence'] > 0.8)],
        'medium': df[(df['risk_score'] > 0.6) & (df['confidence'] > 0.7)],
        'low': df[(df['risk_score'] > 0.4) & (df['confidence'] > 0.6)]
    }
    
    summary = {}
    for severity, alert_df in alerts.items():
        summary[severity] = {
            'count': len(alert_df),
            'total_amount': alert_df['amount'].sum(),
            'avg_risk': alert_df['risk_score'].mean(),
            'avg_confidence': alert_df['confidence'].mean()
        }
    
    return summary

def format_currency(amount):
    """Format amount as Indian currency"""
    return f"₹{amount:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1f}%"

def get_fraud_explanation(fraud_type, risk_score, confidence):
    """Generate human-readable fraud explanation"""
    explanations = {
        'star_fraud': {
            'pattern': 'Star-shaped transaction pattern',
            'description': 'One account receiving funds from multiple sources',
            'risk_factors': ['Money laundering', 'Fake merchant scams', 'Account takeover'],
            'recommendations': ['Block central account', 'Investigate senders', 'Review account history']
        },
        'cycle_fraud': {
            'pattern': 'Circular transaction pattern',
            'description': 'Funds moving in a circular pattern (A→B→C→A)',
            'risk_factors': ['Money laundering', 'Transaction layering', 'Artificial flow'],
            'recommendations': ['Block all accounts in cycle', 'Investigate source of funds', 'Report to authorities']
        },
        'high_value_fraud': {
            'pattern': 'Unusually high transaction amount',
            'description': 'Transaction amount significantly higher than normal',
            'risk_factors': ['Account takeover', 'Unauthorized access', 'Social engineering'],
            'recommendations': ['Verify user identity', 'Check account activity', 'Contact account holder']
        }
    }
    
    base_explanation = explanations.get(fraud_type, {
        'pattern': 'Suspicious transaction pattern',
        'description': 'Transaction flagged by fraud detection system',
        'risk_factors': ['Unknown risk factors'],
        'recommendations': ['Investigate further', 'Monitor account activity']
    })
    
    return {
        **base_explanation,
        'risk_score': risk_score,
        'confidence': confidence,
        'severity': 'Critical' if risk_score > 0.9 else 'High' if risk_score > 0.8 else 'Medium' if risk_score > 0.6 else 'Low'
    } 