import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# Page configuration
st.set_page_config(
    page_title="FortiPay - Advanced UPI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        border-left: 6px solid #d32f2f;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        border-left: 6px solid #f57c00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #ffd54f, #ffc107);
        border-left: 6px solid #ff8f00;
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background: linear-gradient(135deg, #81c784, #4caf50);
        border-left: 6px solid #388e3c;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-explanation {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: #e9ecef;
        border-radius: 0.25rem;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 0.25rem;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'selected_fraud' not in st.session_state:
    st.session_state.selected_fraud = None

def generate_sample_data():
    """Generate comprehensive sample data for demonstration"""
    np.random.seed(42)
    
    # Generate transaction data
    n_transactions = 1000
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    data = {
        'transaction_id': [f'TX{i:06d}' for i in range(n_transactions)],
        'timestamp': np.random.choice(dates, n_transactions),
        'VPA_from': [f'user{i%50}@upi' for i in range(n_transactions)],
        'VPA_to': [f'user{(i+1)%50}@upi' for i in range(n_transactions)],
        'amount': np.random.randint(100, 50000, n_transactions),
        'PSP': np.random.choice(['GooglePay', 'PhonePe', 'Paytm', 'BHIM'], n_transactions),
        'transaction_type': np.random.choice(['transfer', 'collect', 'merchant'], n_transactions),
        'risk_score': np.random.beta(2, 5, n_transactions),  # Skewed towards lower risk
        'confidence': np.random.uniform(0.6, 1.0, n_transactions),
        'fraud_type': ['normal'] * n_transactions
    }
    
    df = pd.DataFrame(data)
    
    # Introduce fraud patterns
    fraud_indices = np.random.choice(n_transactions, 150, replace=False)
    
    # Star-shaped fraud (one account receiving from many)
    star_fraud = fraud_indices[:50]
    df.loc[star_fraud[0], 'fraud_type'] = 'star_fraud_center'
    for i in star_fraud[1:]:
        df.loc[i, 'fraud_type'] = 'star_fraud'
        df.loc[i, 'VPA_to'] = df.loc[star_fraud[0], 'VPA_to']
        df.loc[i, 'risk_score'] = np.random.uniform(0.8, 1.0)
        df.loc[i, 'confidence'] = np.random.uniform(0.85, 1.0)
    
    # Cycle fraud (A->B->C->A)
    cycle_fraud = fraud_indices[50:100]
    for i in range(0, len(cycle_fraud), 3):
        if i+2 < len(cycle_fraud):
            df.loc[cycle_fraud[i:i+3], 'fraud_type'] = 'cycle_fraud'
            df.loc[cycle_fraud[i:i+3], 'risk_score'] = np.random.uniform(0.7, 0.95, 3)
            df.loc[cycle_fraud[i:i+3], 'confidence'] = np.random.uniform(0.75, 0.95, 3)
    
    # High-value fraud
    high_value_fraud = fraud_indices[100:150]
    df.loc[high_value_fraud, 'fraud_type'] = 'high_value_fraud'
    df.loc[high_value_fraud, 'amount'] = np.random.randint(10000, 100000, len(high_value_fraud))
    df.loc[high_value_fraud, 'risk_score'] = np.random.uniform(0.9, 1.0, len(high_value_fraud))
    df.loc[high_value_fraud, 'confidence'] = np.random.uniform(0.9, 1.0, len(high_value_fraud))
    
    return df

def create_confusion_matrix():
    """Create confusion matrix data"""
    # Sample confusion matrix data
    cm_data = {
        'Actual': ['Fraud', 'Fraud', 'Normal', 'Normal'],
        'Predicted': ['Fraud', 'Normal', 'Fraud', 'Normal'],
        'Count': [85, 15, 8, 892]  # TP, FN, FP, TN
    }
    return pd.DataFrame(cm_data)

def create_time_series_fraud():
    """Create time series fraud data"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    fraud_counts = np.random.poisson(5, 30)  # Average 5 frauds per day
    normal_counts = np.random.poisson(30, 30)  # Average 30 normal transactions per day
    
    return pd.DataFrame({
        'date': dates,
        'fraud_count': fraud_counts,
        'normal_count': normal_counts,
        'total_count': fraud_counts + normal_counts
    })

def create_fraud_network(fraud_type="star"):
    """Create network graph for fraud visualization"""
    G = nx.DiGraph()
    
    if fraud_type == "star":
        # Star-shaped fraud: one central node receiving from many
        center_node = "fraud_center@upi"
        G.add_node(center_node, risk_score=0.95, fraud_type="center")
        
        for i in range(10):
            sender = f"sender{i}@upi"
            G.add_node(sender, risk_score=0.8 + i*0.02, fraud_type="sender")
            G.add_edge(sender, center_node, amount=1000 + i*500, confidence=0.85 + i*0.01)
    
    elif fraud_type == "cycle":
        # Cycle fraud: A->B->C->A
        nodes = ["cycle_a@upi", "cycle_b@upi", "cycle_c@upi"]
        for node in nodes:
            G.add_node(node, risk_score=0.85, fraud_type="cycle")
        
        G.add_edge("cycle_a@upi", "cycle_b@upi", amount=5000, confidence=0.9)
        G.add_edge("cycle_b@upi", "cycle_c@upi", amount=5000, confidence=0.9)
        G.add_edge("cycle_c@upi", "cycle_a@upi", amount=5000, confidence=0.9)
    
    return G

def plot_network_graph(G, title="Fraud Network"):
    """Create interactive network visualization"""
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.extend([f"Amount: ₹{edge[2].get('amount', 'N/A')}<br>Confidence: {edge[2].get('confidence', 'N/A'):.2f}"] * 3)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#ff6b6b'),
        hoverinfo='text',
        hovertext=edge_text,
        mode='lines')

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        risk_score = G.nodes[node].get('risk_score', 0)
        fraud_type = G.nodes[node].get('fraud_type', 'normal')
        node_text.append(f"VPA: {node}<br>Risk: {risk_score:.2f}<br>Type: {fraud_type}")
        node_colors.append(risk_score)
        node_sizes.append(20 if fraud_type == 'center' else 15)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='RdYlBu_r',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title='Risk Score',
                xanchor='left',
                titleside='right'
            )
        ))

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def show_dashboard():
    """Main dashboard overview"""
    st.markdown('<h1 class="main-header">🛡️ FortiPay Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Get sample data
    df = generate_sample_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Transactions</h3>
            <h2>{total_transactions:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk = len(df[df['risk_score'] > 0.8])
        st.markdown(f"""
        <div class="metric-card">
            <h3>High Risk Transactions</h3>
            <h2>{high_risk}</h2>
            <p>Risk Score > 0.8</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fraud_count = len(df[df['fraud_type'] != 'normal'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>Detected Frauds</h3>
            <h2>{fraud_count}</h2>
            <p>Various patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_confidence = df['confidence'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Confidence</h3>
            <h2>{avg_confidence:.2f}</h2>
            <p>Model confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main alert section
    st.subheader("🚨 Critical Alerts")
    
    # Get critical alerts
    critical_frauds = df[(df['risk_score'] > 0.9) & (df['confidence'] > 0.9)].head(5)
    
    for _, fraud in critical_frauds.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class="alert-critical">
                <h4>🚨 CRITICAL FRAUD ALERT</h4>
                <p><strong>Transaction ID:</strong> {fraud['transaction_id']}</p>
                <p><strong>Risk Score:</strong> {fraud['risk_score']:.3f}</p>
                <p><strong>Confidence:</strong> {fraud['confidence']:.3f}</p>
                <p><strong>Amount:</strong> ₹{fraud['amount']:,}</p>
                <p><strong>Type:</strong> {fraud['fraud_type'].replace('_', ' ').title()}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button(f"Investigate {fraud['transaction_id']}", key=f"investigate_{fraud['transaction_id']}"):
                st.session_state.selected_fraud = fraud
                st.rerun()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud Distribution by Type")
        fraud_counts = df[df['fraud_type'] != 'normal']['fraud_type'].value_counts()
        fig = px.pie(values=fraud_counts.values, names=fraud_counts.index,
                     title="Fraud Pattern Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(df, x='risk_score', nbins=20, 
                          title="Transaction Risk Score Distribution",
                          color_discrete_sequence=['#ff6b6b'])
        st.plotly_chart(fig, use_container_width=True)

def show_fraud_analysis():
    """Detailed fraud analysis page"""
    st.title("🔍 Fraud Analysis & Detection")
    
    df = generate_sample_data()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        fraud_type_filter = st.selectbox("Fraud Type", ['All'] + list(df['fraud_type'].unique()))
    with col2:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7)
    with col3:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
    
    # Apply filters
    filtered_df = df.copy()
    if fraud_type_filter != 'All':
        filtered_df = filtered_df[filtered_df['fraud_type'] == fraud_type_filter]
    filtered_df = filtered_df[
        (filtered_df['risk_score'] >= risk_threshold) &
        (filtered_df['confidence'] >= confidence_threshold)
    ]
    
    # Fraud patterns visualization
    st.subheader("Fraud Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Star-Shaped Fraud Detection")
        star_fraud = filtered_df[filtered_df['fraud_type'].str.contains('star', na=False)]
        if not star_fraud.empty:
            fig = px.scatter(star_fraud, x='amount', y='risk_score', 
                           color='confidence', size='confidence',
                           title="Star-Shaped Fraud Pattern",
                           hover_data=['VPA_from', 'VPA_to'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No star-shaped fraud detected in current filters")
    
    with col2:
        st.subheader("Cycle Fraud Detection")
        cycle_fraud = filtered_df[filtered_df['fraud_type'].str.contains('cycle', na=False)]
        if not cycle_fraud.empty:
            fig = px.scatter(cycle_fraud, x='amount', y='risk_score',
                           color='confidence', size='confidence',
                           title="Cycle Fraud Pattern",
                           hover_data=['VPA_from', 'VPA_to'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cycle fraud detected in current filters")
    
    # Time series analysis
    st.subheader("Fraud Trends Over Time")
    time_data = create_time_series_fraud()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data['date'], y=time_data['fraud_count'],
                            mode='lines+markers', name='Fraud Count',
                            line=dict(color='#ff6b6b', width=3)))
    fig.add_trace(go.Scatter(x=time_data['date'], y=time_data['normal_count'],
                            mode='lines+markers', name='Normal Count',
                            line=dict(color='#4caf50', width=3)))
    fig.update_layout(title="Daily Transaction Counts", xaxis_title="Date", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

def show_confusion_matrix():
    """Confusion matrix and model performance"""
    st.title("📊 Model Performance & Confusion Matrix")
    
    cm_data = create_confusion_matrix()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        
        # Create confusion matrix heatmap
        cm_values = np.array([[85, 15], [8, 892]])  # TP, FN, FP, TN
        fig = px.imshow(cm_values,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Fraud', 'Normal'],
                       y=['Fraud', 'Normal'],
                       text_auto=True,
                       color_continuous_scale='Reds',
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        
        # Calculate metrics
        tp, fn, fp, tn = 85, 15, 8, 892
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        st.metric("Accuracy", f"{accuracy:.3f}")
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1-Score", f"{f1_score:.3f}")
    
    # Detailed breakdown
    st.subheader("Detailed Classification Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>True Positives</h3>
            <h2>85</h2>
            <p>Frauds correctly identified</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>False Negatives</h3>
            <h2>15</h2>
            <p>Frauds missed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>False Positives</h3>
            <h2>8</h2>
            <p>Normal transactions flagged as fraud</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>True Negatives</h3>
            <h2>892</h2>
            <p>Normal transactions correctly identified</p>
        </div>
        """, unsafe_allow_html=True)

def show_fraud_investigation():
    """Detailed fraud investigation with network visualization"""
    st.title("🔍 Fraud Investigation & Network Analysis")
    
    if st.session_state.selected_fraud is not None:
        fraud = st.session_state.selected_fraud
        
        st.markdown(f"""
        <div class="fraud-explanation">
            <h3>🚨 Fraud Investigation: {fraud['transaction_id']}</h3>
            <p><strong>Risk Score:</strong> {fraud['risk_score']:.3f}</p>
            <p><strong>Confidence:</strong> {fraud['confidence']:.3f}</p>
            <p><strong>Amount:</strong> ₹{fraud['amount']:,}</p>
            <p><strong>Fraud Type:</strong> {fraud['fraud_type'].replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence visualization
        st.subheader("Model Confidence")
        confidence_percentage = fraud['confidence'] * 100
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percentage}%; background: {'#ff6b6b' if confidence_percentage > 90 else '#ffa726' if confidence_percentage > 70 else '#4caf50'};"></div>
        </div>
        <p>Confidence: {confidence_percentage:.1f}%</p>
        """, unsafe_allow_html=True)
        
        # Fraud explanation
        st.subheader("Fraud Detection Evidence")
        
        if 'star' in fraud['fraud_type']:
            st.markdown("""
            **Star-Shaped Fraud Pattern Detected:**
            - One central account receiving funds from multiple sources
            - Typical of money laundering or fake merchant scams
            - High risk due to unusual transaction pattern
            """)
            
            # Show star fraud network
            G = create_fraud_network("star")
            fig = plot_network_graph(G, "Star-Shaped Fraud Network")
            st.plotly_chart(fig, use_container_width=True)
            
        elif 'cycle' in fraud['fraud_type']:
            st.markdown("""
            **Cycle Fraud Pattern Detected:**
            - Circular transaction pattern (A→B→C→A)
            - Indicates money laundering or transaction layering
            - High risk due to artificial transaction flow
            """)
            
            # Show cycle fraud network
            G = create_fraud_network("cycle")
            fig = plot_network_graph(G, "Cycle Fraud Network")
            st.plotly_chart(fig, use_container_width=True)
            
        elif 'high_value' in fraud['fraud_type']:
            st.markdown("""
            **High-Value Fraud Detected:**
            - Unusually large transaction amount
            - May indicate account takeover or unauthorized access
            - High risk due to significant financial impact
            """)
        
        # Recommended actions
        st.subheader("Recommended Actions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚫 Block Transaction", type="primary"):
                st.success("Transaction blocked successfully!")
        with col2:
            if st.button("🔍 Investigate Further"):
                st.info("Investigation initiated. Analysts will review.")
        with col3:
            if st.button("📞 Contact User"):
                st.warning("User notification sent.")
        
        # Clear selection
        if st.button("Clear Selection"):
            st.session_state.selected_fraud = None
            st.rerun()
    
    else:
        st.info("Select a fraud alert from the dashboard to investigate.")

def show_transaction_graph():
    """Transaction graph visualization page"""
    st.title("🕸️ Transaction Graph Analysis")
    
    # Get sample data
    df = generate_sample_data()
    
    # Filters
    st.sidebar.subheader("Graph Filters")
    
    col1, col2 = st.columns(2)
    with col1:
        max_nodes = st.sidebar.slider("Max Nodes to Display", 10, 100, 50)
        risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5)
    with col2:
        selected_psp = st.sidebar.multiselect("PSP Filter", df['PSP'].unique(), default=df['PSP'].unique())
        graph_type = st.sidebar.selectbox("Graph Type", ["All Transactions", "High Risk Only", "Fraud Patterns"])
    
    # Apply filters
    filtered_df = df[
        (df['PSP'].isin(selected_psp)) &
        (df['risk_score'] >= risk_threshold)
    ]
    
    if graph_type == "High Risk Only":
        filtered_df = filtered_df[filtered_df['risk_score'] > 0.8]
    elif graph_type == "Fraud Patterns":
        filtered_df = filtered_df[filtered_df['fraud_type'] != 'normal']
    
    # Limit nodes for performance
    if len(filtered_df) > max_nodes:
        filtered_df = filtered_df.head(max_nodes)
    
    # Create transaction graph
    if not filtered_df.empty:
        G = create_transaction_graph(filtered_df)
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes (VPAs)", G.number_of_nodes())
        with col2:
            st.metric("Edges (Transactions)", G.number_of_edges())
        with col3:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            st.metric("Avg Degree", f"{avg_degree:.1f}")
        with col4:
            density = nx.density(G)
            st.metric("Graph Density", f"{density:.3f}")
        
        # Network visualization
        st.subheader("Transaction Network Graph")
        
        # Graph layout options
        layout_option = st.selectbox("Graph Layout", ["Spring", "Circular", "Random", "Shell"])
        
        if layout_option == "Spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout_option == "Circular":
            pos = nx.circular_layout(G)
        elif layout_option == "Random":
            pos = nx.random_layout(G)
        else:  # Shell
            pos = nx.shell_layout(G)
        
        # Create interactive network graph
        fig = plot_network_graph(G, "Transaction Network")
        st.plotly_chart(fig, use_container_width=True, height=600)
        
        # Graph analysis
        st.subheader("Graph Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Centrality analysis
            st.subheader("Top Central VPAs")
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                
                centrality_df = pd.DataFrame(top_central, columns=['VPA', 'Centrality'])
                st.dataframe(centrality_df, use_container_width=True)
        
        with col2:
            # Connected components
            st.subheader("Connected Components")
            components = list(nx.connected_components(G.to_undirected()))
            st.write(f"Number of connected components: {len(components)}")
            
            if components:
                largest_component = max(components, key=len)
                st.write(f"Largest component size: {len(largest_component)}")
                
                # Show nodes in largest component
                if len(largest_component) <= 20:
                    st.write("Nodes in largest component:")
                    for node in largest_component:
                        st.write(f"- {node}")
        
        # Transaction details table
        st.subheader("Transaction Details")
        
        # Add graph metrics to dataframe
        display_df = filtered_df.copy()
        display_df['degree'] = display_df['VPA_from'].map(dict(G.degree()))
        display_df['centrality'] = display_df['VPA_from'].map(nx.degree_centrality(G))
        
        # Select columns to display
        columns_to_show = ['transaction_id', 'VPA_from', 'VPA_to', 'amount', 'risk_score', 'confidence', 'fraud_type', 'PSP', 'degree', 'centrality']
        available_columns = [col for col in columns_to_show if col in display_df.columns]
        
        st.dataframe(display_df[available_columns], use_container_width=True)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Transaction Data as CSV",
            data=csv,
            file_name="transaction_graph_data.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No transactions match the current filters. Try adjusting the filters.")
    
    # Graph insights
    st.subheader("🔍 Graph Insights")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Network Patterns:**
            - **Star patterns**: One VPA receiving from many (potential fraud)
            - **Cycle patterns**: Circular transactions (money laundering)
            - **Isolated nodes**: Single transactions (normal behavior)
            - **Clusters**: Groups of connected VPAs (business relationships)
            """)
        
        with col2:
            st.markdown("""
            **Risk Indicators:**
            - **High centrality**: VPAs with many connections
            - **High degree**: VPAs involved in many transactions
            - **Isolated high-risk**: Suspicious single transactions
            - **Connected high-risk**: Potential fraud networks
            """)

def create_transaction_graph(df):
    """Create a NetworkX graph from transaction data"""
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        # Add nodes with attributes
        if not G.has_node(row['VPA_from']):
            G.add_node(row['VPA_from'], 
                      risk_score=row['risk_score'],
                      fraud_type=row['fraud_type'],
                      account_type='sender')
        
        if not G.has_node(row['VPA_to']):
            G.add_node(row['VPA_to'], 
                      risk_score=row['risk_score'],
                      fraud_type=row['fraud_type'],
                      account_type='receiver')
        
        # Add edge with transaction attributes
        G.add_edge(row['VPA_from'], row['VPA_to'], 
                  amount=row['amount'],
                  timestamp=row['timestamp'],
                  PSP=row['PSP'],
                  transaction_type=row['transaction_type'],
                  risk_score=row['risk_score'],
                  confidence=row['confidence'])
    
    return G

def main():
    # Sidebar navigation
    st.sidebar.title("🛡️ FortiPay")
    
    # Demo login (for demonstration purposes)
    if not st.session_state.auth_token:
        st.sidebar.subheader("Demo Login")
        if st.sidebar.button("Login as Analyst"):
            st.session_state.auth_token = "demo_token"
            st.success("Logged in successfully!")
            st.rerun()
        return
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Fraud Analysis", "Transaction Graph", "Model Performance", "Fraud Investigation"]
    )
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.auth_token = None
        st.session_state.selected_fraud = None
        st.rerun()
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Fraud Analysis":
        show_fraud_analysis()
    elif page == "Transaction Graph":
        show_transaction_graph()
    elif page == "Model Performance":
        show_confusion_matrix()
    elif page == "Fraud Investigation":
        show_fraud_investigation()

if __name__ == "__main__":
    main() 