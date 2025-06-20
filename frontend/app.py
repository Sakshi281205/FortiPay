import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="FortiPay - Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .login-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 400px;
        width: 100%;
    }
    
    .dashboard-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .alert-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .alert-critical {
        border-left-color: #ff4757;
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.1), rgba(255, 71, 87, 0.05));
    }
    
    .alert-high {
        border-left-color: #ffa502;
        background: linear-gradient(135deg, rgba(255, 165, 2, 0.1), rgba(255, 165, 2, 0.05));
    }
    
    .fraud-details {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for fraud details
if 'selected_fraud' not in st.session_state:
    st.session_state.selected_fraud = None

def show_login_page():
    """Professional login page like Google"""
    st.markdown("""
    <div class="login-container">
        <div class="login-card">
            <div style="font-size: 3rem; margin-bottom: 20px;">🛡️</div>
            <h1 style="font-size: 2rem; color: #333; margin-bottom: 30px; font-weight: 300;">FortiPay</h1>
            <p style="color: #666; margin-bottom: 30px;">Enterprise Fraud Detection Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); 
                        border-radius: 20px; padding: 40px; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);">
            """, unsafe_allow_html=True)
            
            st.markdown("<h2 style='text-align: center; color: #333; margin-bottom: 30px;'>Data Analyst Login</h2>", unsafe_allow_html=True)
            
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Sign In", key="login_button", use_container_width=True):
                    if username == "analyst" and password == "fortipay123":
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.current_page = 'dashboard'
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Use: analyst / fortipay123")
            
            st.markdown("</div>", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample transaction data with fraud patterns"""
    np.random.seed(42)
    random.seed(42)  # for reproducibility
    n_transactions = 1000
    
    # Generate VPAs
    vpas = [f"user{i}@upi" for i in range(1, 101)]
    
    # Generate transaction data
    data = []
    for i in range(n_transactions):
        vpa_from = random.choice(vpas)
        vpa_to = random.choice(vpas)
        
        # Ensure different sender and receiver
        while vpa_to == vpa_from:
            vpa_to = random.choice(vpas)
        
        # Generate amount (mostly small, some large)
        if random.random() < 0.1:  # 10% high value
            amount = random.randint(5000, 50000)
        else:
            amount = random.randint(10, 2000)
        
        # Generate timestamp
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Generate fraud patterns
        fraud_prob = random.random()
        if fraud_prob < 0.1:  # 10% star fraud
            fraud_type = 'star_fraud_center'
            risk_score = random.uniform(0.8, 1.0)
        elif fraud_prob < 0.2:  # 10% cycle fraud
            fraud_type = 'cycle_fraud'
            risk_score = random.uniform(0.7, 0.95)
        elif fraud_prob < 0.25:  # 5% high value fraud
            fraud_type = 'high_value_fraud'
            risk_score = random.uniform(0.6, 0.9)
        else:  # 75% normal
            fraud_type = 'normal'
            risk_score = random.uniform(0.0, 0.3)
        
        data.append({
            'VPA_from': vpa_from,
            'VPA_to': vpa_to,
            'amount': amount,
            'timestamp': timestamp,
            'PSP': random.choice(['Google Pay', 'PhonePe', 'Paytm', 'BHIM']),
            'transaction_type': random.choice(['UPI', 'QR', 'Collect']),
            'fraud_type': fraud_type,
            'risk_score': risk_score,
            'confidence': random.uniform(0.7, 1.0)
        })
    
    return pd.DataFrame(data)

def show_dashboard():
    """Main dashboard with overview metrics and charts"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🏠 FortiPay Dashboard</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Real-time UPI Fraud Detection & Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Add transaction_id if it doesn't exist
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = [f'TX{i:06d}' for i in range(len(df))]
    
    # Top row: Key metrics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(df)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        fraud_transactions = len(df[df['fraud_type'] != 'normal'])
        fraud_rate = (fraud_transactions / total_transactions) * 100
        st.metric("Fraud Detected", f"{fraud_transactions:,}", f"{fraud_rate:.1f}%")
    
    with col3:
        avg_risk = df['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    
    with col4:
        total_amount = df['amount'].sum()
        st.metric("Total Volume", f"₹{total_amount:,.0f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle row: Charts
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 Fraud Detection Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Confusion Matrix Pie Chart
        fraud_types = df['fraud_type'].value_counts()
        fig = px.pie(
            values=fraud_types.values,
            names=fraud_types.index,
            title="Transaction Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Score Distribution
        fig = px.histogram(
            df, x='risk_score', nbins=20,
            title="Risk Score Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # PSP Distribution
        psp_counts = df['PSP'].value_counts()
        fig = px.bar(
            x=psp_counts.index,
            y=psp_counts.values,
            title="Transactions by PSP",
            color_discrete_sequence=['#ff4757']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom row: Time-based Bar Chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 Transaction Trends Over Time")
    
    # Calculate daily counts for normal and fraud transactions
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby(['date', 'fraud_type']).size().unstack(fill_value=0)
    
    if 'normal' not in daily_counts.columns:
        daily_counts['normal'] = 0
        
    fraud_cols = [col for col in daily_counts.columns if col != 'normal']
    daily_counts['fraud'] = daily_counts[fraud_cols].sum(axis=1)
    
    time_data = daily_counts[['normal', 'fraud']].reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=time_data['date'],
        y=time_data['normal'],
        name='Normal Transactions',
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        x=time_data['date'],
        y=time_data['fraud'],
        name='Fraud Transactions',
        marker_color='#ff4757'
    ))
    
    fig.update_layout(
        title="Transaction Volume: Normal vs Fraud",
        xaxis_title="Date",
        yaxis_title="Count",
        barmode='stack',
        height=400,
        legend=dict(x=0.01, y=0.99, bordercolor='Gainsboro', borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Right side: Alerts
    st.sidebar.markdown("""
    <div style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); 
                border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #333; margin-bottom: 20px;">🚨 Live Alerts</h3>
    </div>
    """, unsafe_allow_html=True)
    
    critical_frauds = df[(df['risk_score'] > 0.9) & (df['confidence'] > 0.9)].head(10)
    
    for _, fraud in critical_frauds.iterrows():
        alert_class = "alert-critical" if fraud['risk_score'] > 0.95 else "alert-high"
        
        st.sidebar.markdown(f"""
        <div class="alert-card {alert_class}">
            <h4 style="margin: 0 0 10px 0; color: #333;">🚨 {fraud['fraud_type'].replace('_', ' ').title()}</h4>
            <p style="margin: 5px 0; color: #666;">ID: {fraud['transaction_id']}</p>
            <p style="margin: 5px 0; color: #666;">Amount: ₹{fraud['amount']:,}</p>
            <p style="margin: 5px 0; color: #666;">Risk: {fraud['risk_score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button(f"Investigate {fraud['transaction_id']}", key=f"investigate_{fraud['transaction_id']}"):
            st.session_state.selected_fraud = fraud
            st.session_state.current_page = 'fraud_details'
            st.rerun()
    
    if st.sidebar.button("Logout", key="logout"):
        st.session_state.authenticated = False
        st.session_state.current_page = 'login'
        st.rerun()

def show_fraud_details():
    """
    Shows the local graph and explanation for a single selected fraud transaction.
    """
    if 'selected_fraud' not in st.session_state or st.session_state.selected_fraud is None:
        st.error("No transaction selected. Please go back to Fraud Analysis and click 'Investigate'.")
        if st.button("← Back to Fraud Analysis"):
            st.session_state.current_page = 'fraud_analysis'
            st.rerun()
        return

    fraud = st.session_state.selected_fraud
    
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 style='color: #333;'>Investigation: Transaction {fraud.get('transaction_id', 'N/A')}</h1>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Fraud Analysis"):
        st.session_state.current_page = 'fraud_analysis'
        st.rerun()

    # --- Key Details & Explanation ---
    st.markdown('<div class="fraud-details">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Risk Score", f"{fraud['risk_score']:.2f}")
        st.metric("Amount", f"₹{fraud['amount']:,}")
        st.write(f"**From:** {fraud['VPA_from']}")
        st.write(f"**To:** {fraud['VPA_to']}")
    with col2:
        st.subheader("📝 Why is this transaction suspicious?")
        if 'star' in fraud['fraud_type']:
            explanation = "This transaction is part of a **star-shaped fraud pattern**: one account is receiving funds from many sources. This is suspicious for money laundering or account takeover."
        elif 'cycle' in fraud['fraud_type']:
            explanation = "This transaction is part of a **cycle fraud pattern**: funds are moving in a loop. This is often used for money laundering or to obscure fund origins."
        elif 'high_value' in fraud['fraud_type']:
            explanation = "This is a **high-value transaction** that is much larger than typical amounts, which is a common sign of fraud or account takeover."
        else:
            explanation = "This transaction is flagged due to a high-risk score based on our GNN model's analysis of its features and connections."
        st.info(explanation)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Local Graph Visualization ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader(f"🔗 Local Transaction Graph for {fraud['VPA_to']}")
    
    # Build a 1-hop neighborhood graph around the transaction's VPAs
    df = generate_sample_data()
    involved_vpas = {fraud['VPA_from'], fraud['VPA_to']}
    # Find neighbors
    neighbors = set(df[df['VPA_from'].isin(involved_vpas)]['VPA_to']) | set(df[df['VPA_to'].isin(involved_vpas)]['VPA_from'])
    neighborhood_vpas = involved_vpas | neighbors
    
    local_df = df[df['VPA_from'].isin(neighborhood_vpas) | df['VPA_to'].isin(neighborhood_vpas)]
    
    if not local_df.empty:
        local_G = create_fraud_network_graph(local_df)
        fig = create_interactive_network_graph(local_G, 'Spring', fraud['fraud_type'])
        st.plotly_chart(fig, use_container_width=True, height=500)
    else:
        st.warning("Could not generate a local graph for this transaction.")
    st.markdown('</div>', unsafe_allow_html=True)

def show_sidebar():
    """Sidebar navigation"""
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>🔒 FortiPay</h2>
        <p>UPI Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.markdown("### 📊 Navigation")
    
    if st.sidebar.button("🏠 Dashboard", use_container_width=True):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    if st.sidebar.button("🔍 Fraud Analysis", use_container_width=True):
        st.session_state.current_page = 'fraud_analysis'
        st.rerun()
    
    if st.sidebar.button("📈 Model Performance", use_container_width=True):
        st.session_state.current_page = 'model_performance'
        st.rerun()
    
    if st.sidebar.button("🔍 Fraud Investigation", use_container_width=True):
        st.session_state.current_page = 'fraud_details'
        st.rerun()
    
    if st.sidebar.button("🕸️ Transaction Graph", use_container_width=True):
        st.session_state.current_page = 'transaction_graph'
        st.rerun()
    
    # User info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 User Info")
    st.sidebar.write(f"**User:** {st.session_state.get('username', 'Admin')}")
    st.sidebar.write(f"**Role:** Security Analyst")
    
    # Logout
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_page = 'login'
        st.rerun()

def show_fraud_analysis():
    """
    A redesigned, more intuitive fraud analysis page.
    """
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🔍 Advanced Fraud Analysis</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Interactive Detection & Investigation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("ℹ️ Use the filters below to investigate suspicious transactions. Scroll to the bottom for a summary dashboard.")

    df = generate_sample_data()
    
    # --- Filters ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Analysis Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        fraud_type = st.selectbox("Fraud Pattern", ["All Patterns", "Star Fraud", "Cycle Fraud", "High Value Fraud"])
    with col2:
        risk_threshold = st.slider("Risk Score Threshold", 0.0, 1.0, 0.7)
    with col3:
        max_transactions = st.slider("Max Transactions to Display", 10, 100, 25)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Filter Logic ---
    if fraud_type == "Star Fraud":
        filtered_df = df[df['fraud_type'].str.contains('star', na=False)]
    elif fraud_type == "Cycle Fraud":
        filtered_df = df[df['fraud_type'].str.contains('cycle', na=False)]
    elif fraud_type == "High Value Fraud":
        filtered_df = df[df['fraud_type'].str.contains('high_value', na=False)]
    else:
        filtered_df = df[df['fraud_type'] != 'normal']
    
    filtered_df = filtered_df[filtered_df['risk_score'] >= risk_threshold].sort_values('risk_score', ascending=False).head(max_transactions)

    # --- High-Risk Transactions Table ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🚨 High-Risk Transactions Detected")

    if not filtered_df.empty:
        # Add an 'Investigate' column with buttons
        cols = st.columns((1, 2, 2, 1, 1, 1, 1.5))
        headers = ["ID", "From", "To", "Amount", "Risk", "Type", "Action"]
        for col, header in zip(cols, headers):
            col.write(f"**{header}**")
        
        for idx, row in filtered_df.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns((1, 2, 2, 1, 1, 1, 1.5))
            with col1:
                st.write(row.get('transaction_id', f'TX{idx}'))
            with col2:
                st.write(row['VPA_from'])
            with col3:
                st.write(row['VPA_to'])
            with col4:
                st.write(f"₹{row['amount']:,}")
            with col5:
                st.write(f"{row['risk_score']:.2f}")
            with col6:
                st.write(row['fraud_type'].replace('_', ' ').title())
            with col7:
                if st.button("🔍 Investigate", key=f"investigate_{idx}"):
                    st.session_state.selected_fraud = row
                    st.session_state.current_page = 'fraud_details'
                    st.rerun()
    else:
        st.warning("No transactions match the current filter settings. Try lowering the risk threshold.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- GNN Visualization of Filtered Transactions ---
    if not filtered_df.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🕸️ GNN Visualization of Filtered Transactions")
        G = create_fraud_network_graph(filtered_df)
        fig = create_interactive_network_graph(G, 'Spring', fraud_type)
        st.plotly_chart(fig, use_container_width=True, height=600)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Summary Dashboard (at the bottom) ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📝 Fraud Pattern Summary Dashboard")
    # ... (summary logic remains the same) ...
    st.markdown('</div>', unsafe_allow_html=True)

def create_fraud_network_graph(df):
    """Create a NetworkX graph for fraud analysis"""
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

def create_interactive_network_graph(G, layout_type, fraud_type):
    """Create interactive network visualization"""
    if layout_type == "Spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.extend([f"Amount: ₹{edge[2].get('amount', 'N/A')}<br>Risk: {edge[2].get('risk_score', 'N/A'):.2f}<br>PSP: {edge[2].get('PSP', 'N/A')}"] * 3)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#ff4757'),
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
        account_type = G.nodes[node].get('account_type', 'unknown')
        
        node_text.append(f"VPA: {node}<br>Risk: {risk_score:.2f}<br>Type: {fraud_type}<br>Role: {account_type}")
        node_colors.append(risk_score)
        
        # Size nodes based on importance
        if 'star_fraud_center' in fraud_type:
            node_sizes.append(25)
        elif 'cycle' in fraud_type:
            node_sizes.append(20)
        else:
            node_sizes.append(15)

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
                       title=f'{fraud_type} Network Analysis',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
    }
    
    /* Dashboard header */
    .dashboard-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Fraud details section */
    .fraud-details {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Alert cards */
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff4757 0%, #c44569 100%);
        border-left: 5px solid #ff3838;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        border-left: 5px solid #ff6348;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Login page styling */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .login-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        max-width: 400px;
        width: 100%;
    }
    
    .login-logo {
        font-size: 3rem;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Streamlit default styling overrides */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSidebar {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application logic"""
    # Set page config once at the beginning
    st.set_page_config(
        page_title="FortiPay - Enterprise Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'
    if 'username' not in st.session_state:
        st.session_state.username = ''
    
    # Show login page if not authenticated
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Show sidebar navigation
    show_sidebar()
    
    # Show appropriate page based on current_page
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'model_performance':
        show_model_performance()
    elif st.session_state.current_page == 'fraud_details':
        show_fraud_details()
    elif st.session_state.current_page == 'transaction_graph':
        show_transaction_graph()
    elif st.session_state.current_page == 'fraud_analysis':
        show_fraud_analysis()

if __name__ == "__main__":
    main() 