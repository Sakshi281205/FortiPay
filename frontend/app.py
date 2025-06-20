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
    initial_sidebar_state="collapsed"
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
    """Detailed fraud analysis"""
    if st.session_state.selected_fraud is None:
        st.error("No fraud selected")
        return
    
    fraud = st.session_state.selected_fraud
    
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🔍 Fraud Investigation Details</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Comprehensive Analysis & Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    # --- SUMMARY BOX ---
    st.markdown(f"""
    <div class="dashboard-header">
        <h2 style='color: #ff4757;'>🚨 Fraud Summary</h2>
        <p><b>Risk Score:</b> {fraud['risk_score']:.2f} &nbsp; | &nbsp; <b>Type:</b> {fraud['fraud_type'].replace('_', ' ').title()} &nbsp; | &nbsp; <b>Amount:</b> ₹{fraud['amount']:,}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fraud details
    st.markdown(f"""
    <div class="fraud-details">
        <h2 style="color: #333; margin-bottom: 20px;">Transaction: {fraud['transaction_id']}</h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4 style="color: #667eea;">Transaction Details</h4>
                <p><strong>From:</strong> {fraud['VPA_from']}</p>
                <p><strong>To:</strong> {fraud['VPA_to']}</p>
                <p><strong>Amount:</strong> ₹{fraud['amount']:,}</p>
                <p><strong>PSP:</strong> {fraud['PSP']}</p>
            </div>
            <div>
                <h4 style="color: #667eea;">Risk Assessment</h4>
                <p><strong>Risk Score:</strong> {fraud['risk_score']:.3f}</p>
                <p><strong>Confidence:</strong> {fraud['confidence']:.3f}</p>
                <p><strong>Fraud Type:</strong> {fraud['fraud_type'].replace('_', ' ').title()}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk visualization
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Risk Score Visualization")
    
    risk_percentage = fraud['risk_score'] * 100
    confidence_percentage = fraud['confidence'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Risk Score</h4>
            <div style="background: #e9ecef; border-radius: 10px; height: 25px; margin: 10px 0;">
                <div style="width: {risk_percentage}%; height: 100%; border-radius: 10px; 
                            background: {'#ff4757' if risk_percentage > 90 else '#ffa502' if risk_percentage > 70 else '#2ed573'};"></div>
            </div>
            <p style="font-size: 1.5rem; font-weight: bold;">{risk_percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Confidence Score</h4>
            <div style="background: #e9ecef; border-radius: 10px; height: 25px; margin: 10px 0;">
                <div style="width: {confidence_percentage}%; height: 100%; border-radius: 10px; 
                            background: {'#2ed573' if confidence_percentage > 90 else '#ffa502' if confidence_percentage > 70 else '#ff4757'};"></div>
            </div>
            <p style="font-size: 1.5rem; font-weight: bold;">{confidence_percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- EXPLANATION ---
    st.markdown('<div class="fraud-details">', unsafe_allow_html=True)
    st.subheader("📝 Why is this flagged as fraud?")
    if 'star' in fraud['fraud_type']:
        explanation = """
        This transaction is part of a **star-shaped fraud pattern**: one account is receiving funds from many sources. This is suspicious for money laundering, fake merchant scams, or account takeover.
        """
    elif 'cycle' in fraud['fraud_type']:
        explanation = """
        This transaction is part of a **cycle fraud pattern**: funds are moving in a loop between accounts. This is often used for money laundering or to obscure fund origins.
        """
    elif 'high_value' in fraud['fraud_type']:
        explanation = """
        This is a **high value transaction** that is much larger than typical amounts, which is a common sign of fraud, account takeover, or social engineering.
        """
    else:
        explanation = "This transaction is flagged due to unusual risk factors."
    st.info(explanation)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown('<div class="fraud-details">', unsafe_allow_html=True)
    st.subheader("⚡ Recommended Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚫 Block Transaction", type="primary", use_container_width=True):
            st.success("Transaction blocked successfully!")
    
    with col2:
        if st.button("📞 Alert Customer", use_container_width=True):
            st.info("Customer notification sent!")
    
    with col3:
        if st.button("🔍 Investigate Further", use_container_width=True):
            st.warning("Investigation initiated.")
    
    with col4:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Report generated and saved.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LOCAL GRAPH ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔗 Local Transaction Graph")
    # Show a local subgraph for this fraud (1-hop neighborhood)
    df = generate_sample_data()  # Use the same function to get the data
    # Find all transactions involving this VPA (from or to)
    vpa = fraud['VPA_to']
    local_df = df[(df['VPA_from'] == vpa) | (df['VPA_to'] == vpa)]
    local_G = create_fraud_network_graph(local_df)
    fig = create_interactive_network_graph(local_G, 'Spring', fraud['fraud_type'])
    st.plotly_chart(fig, use_container_width=True, height=400)
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
    """Comprehensive fraud analysis with GNN graphs"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🔍 Advanced Fraud Analysis</h1>
        <p style="color: #666; margin: 5px 0 0 0;">GNN-Based Pattern Detection & Network Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = generate_sample_data()
    
    # Filters
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Analysis Filters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fraud_type = st.selectbox("Fraud Pattern", ["All Patterns", "Star Fraud", "Cycle Fraud", "High Value Fraud"])
    with col2:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7)
    with col3:
        max_nodes = st.slider("Max Graph Nodes", 10, 100, 30)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data based on selection
    if fraud_type == "Star Fraud":
        filtered_df = df[df['fraud_type'].astype(str).str.contains('star', na=False)]
    elif fraud_type == "Cycle Fraud":
        filtered_df = df[df['fraud_type'].astype(str).str.contains('cycle', na=False)]
    elif fraud_type == "High Value Fraud":
        filtered_df = df[df['fraud_type'].astype(str).str.contains('high_value', na=False)]
    else:
        filtered_df = df[df['fraud_type'] != 'normal']
    
    filtered_df = filtered_df[filtered_df['risk_score'] >= risk_threshold]
    
    if len(filtered_df) > max_nodes:
        filtered_df = filtered_df.head(max_nodes)
    
    if not filtered_df.empty:
        # Create GNN graph
        G = create_fraud_network_graph(filtered_df)
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes (VPAs)", G.number_of_nodes())
        with col2:
            st.metric("Edges (Transactions)", G.number_of_edges())
        with col3:
            degrees = dict(G.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            st.metric("Avg Degree", f"{avg_degree:.1f}")
        with col4:
            density = nx.density(G)
            st.metric("Graph Density", f"{density:.3f}")
        
        # Network visualization
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🕸️ GNN Transaction Network Graph")
        
        # Graph layout options
        layout_option = st.selectbox("Graph Layout", ["Spring", "Circular", "Shell", "Random"])
        
        fig = create_interactive_network_graph(G, layout_option, fraud_type)
        st.plotly_chart(fig, use_container_width=True, height=600)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pattern analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Centrality analysis
            st.subheader("🔍 Centrality Analysis")
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                
                centrality_df = pd.DataFrame(top_central, columns=['VPA', 'Centrality Score'])
                st.dataframe(centrality_df, use_container_width=True)
                
                # Highlight suspicious central nodes
                high_centrality = [node for node, score in top_central if score > 0.5]
                if high_centrality:
                    st.warning(f"⚠️ High centrality VPAs detected: {', '.join(high_centrality[:3])}")
        
        with col2:
            # Connected components
            st.subheader("🔗 Network Components")
            components = list(nx.connected_components(G.to_undirected()))
            st.write(f"Number of connected components: {len(components)}")
            
            if components:
                largest_component = max(components, key=len)
                st.write(f"Largest component size: {len(largest_component)}")
                
                # Detect star patterns
                star_centers = detect_star_patterns(G)
                if star_centers:
                    st.error(f"🚨 Star fraud centers detected: {', '.join(star_centers[:3])}")
                
                # Detect cycles
                cycles = detect_cycle_patterns(G)
                if cycles:
                    st.error(f"🔄 Cycle fraud patterns detected: {len(cycles)} cycles found")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed fraud explanations
        st.markdown('<div class="fraud-details">', unsafe_allow_html=True)
        st.subheader("🔍 Fraud Pattern Explanations")
        
        if fraud_type == "Star Fraud" or fraud_type == "All Patterns":
            st.markdown("""
            **⭐ Star Fraud Pattern Analysis:**
            
            Star-shaped fraud occurs when one central account receives funds from multiple sources. 
            This pattern is highly suspicious because:
            
            • **Money Laundering**: Attempting to obscure the source of funds by funneling through multiple accounts
            • **Fake Merchant Scams**: Fraudulent merchants collecting payments from multiple victims
            • **Account Takeover**: Compromised account being used as a collection point
            • **Structuring**: Breaking large amounts into smaller transactions to avoid detection
            
            **Risk Indicators:**
            - High in-degree (many incoming transactions)
            - Low out-degree (few outgoing transactions)
            - Unusual transaction timing patterns
            - Multiple unique senders to one recipient
            """)
        
        if fraud_type == "Cycle Fraud" or fraud_type == "All Patterns":
            st.markdown("""
            **🔄 Cycle Fraud Pattern Analysis:**
            
            Cycle fraud involves funds moving in circular patterns (A→B→C→A). This indicates:
            
            • **Money Laundering**: Artificial transaction flow to obscure fund origins
            • **Transaction Layering**: Multiple hops to make tracing difficult
            • **Structuring**: Breaking large amounts into smaller transactions
            • **Smurfing**: Using multiple accounts to avoid detection thresholds
            
            **Risk Indicators:**
            - Circular transaction flow
            - Similar amounts in cycle
            - Rapid transaction timing
            - Artificial transaction patterns
            - Multiple accounts involved in short time
            """)
        
        if fraud_type == "High Value Fraud" or fraud_type == "All Patterns":
            st.markdown("""
            **💰 High Value Fraud Analysis:**
            
            High-value fraud involves unusually large transactions that exceed normal patterns:
            
            • **Account Takeover**: Unauthorized access to account for large transfers
            • **Social Engineering**: Victim tricked into making large transfers
            • **Unauthorized Access**: Compromised credentials used for large transactions
            • **Business Email Compromise**: Fraudulent requests for large payments
            
            **Risk Indicators:**
            - Amount significantly higher than account history
            - Unusual transaction timing
            - High-risk recipient account
            - Suspicious transaction context
            - New or recently created recipient accounts
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Alert and action section
        st.markdown('<div class="fraud-details">', unsafe_allow_html=True)
        st.subheader("🚨 Fraud Alerts & Actions")
        
        # Generate alerts based on patterns
        alerts = generate_fraud_alerts(G, filtered_df)
        
        for alert in alerts:
            alert_class = "alert-critical" if alert['severity'] == 'Critical' else "alert-high"
            
            st.markdown(f"""
            <div class="alert-card {alert_class}">
                <h4 style="margin: 0 0 10px 0; color: #333;">🚨 {alert['title']}</h4>
                <p style="margin: 5px 0; color: #666;">{alert['description']}</p>
                <p style="margin: 5px 0; color: #666;"><strong>Risk Score:</strong> {alert['risk_score']:.3f}</p>
                <p style="margin: 5px 0; color: #666;"><strong>Affected VPAs:</strong> {alert['affected_vpas']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(f"🚫 Block {alert['type']}", key=f"block_{alert['id']}"):
                    st.success(f"Blocked {alert['type']} transactions!")
            with col2:
                if st.button(f"📞 Alert Users", key=f"alert_{alert['id']}"):
                    st.info(f"Alerted users for {alert['type']}!")
            with col3:
                if st.button(f"🔍 Investigate", key=f"investigate_{alert['id']}"):
                    st.warning(f"Investigation initiated for {alert['type']}!")
            with col4:
                if st.button(f"📊 Report", key=f"report_{alert['id']}"):
                    st.info(f"Report generated for {alert['type']}!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- SUMMARY DASHBOARD ---
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📝 Fraud Pattern Summary Dashboard")
        
        # Calculate summary stats from the original dataset, not just filtered
        df_original = generate_sample_data()
        total_star = len(df_original[df_original['fraud_type'].astype(str).str.contains('star', na=False)])
        total_cycle = len(df_original[df_original['fraud_type'].astype(str).str.contains('cycle', na=False)])
        total_high_value = len(df_original[df_original['fraud_type'].astype(str).str.contains('high_value', na=False)])
        
        # Get top risky from filtered data if available, otherwise from original
        if not filtered_df.empty:
            top_risky = filtered_df.sort_values('risk_score', ascending=False).head(5)
            most_common = filtered_df['fraud_type'].value_counts().idxmax() if not filtered_df.empty else 'None'
        else:
            top_risky = df_original.sort_values('risk_score', ascending=False).head(5)
            most_common = df_original['fraud_type'].value_counts().idxmax() if not df_original.empty else 'None'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Star Patterns", total_star)
        with col2:
            st.metric("Cycle Patterns", total_cycle)
        with col3:
            st.metric("High Value Patterns", total_high_value)
        
        st.markdown("**Top 5 Risky VPAs:**")
        if not top_risky.empty:
            st.table(top_risky[['VPA_from', 'VPA_to', 'risk_score', 'fraud_type']])
        else:
            st.write("No risky VPAs detected.")
        
        st.markdown(f"**Most Common Fraud Type:** {most_common.replace('_', ' ').title()}")
        
        # English summary
        summary_text = f"""
        **Summary of Detected Patterns:**
        - {total_star} star-shaped fraud patterns detected (multiple senders to one receiver).
        - {total_cycle} cycle fraud patterns detected (circular fund movement).
        - {total_high_value} high value frauds detected (unusually large transactions).
        - The most common fraud type is **{most_common.replace('_', ' ').title()}**.
        
        **Current Filter Results:** {len(filtered_df)} transactions match your current filters.
        """
        st.info(summary_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning("No fraud patterns detected with current filters. Try adjusting the risk threshold or fraud type.")
        
        # Show summary even when no patterns detected
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📝 Overall Fraud Pattern Summary")
        
        df_original = generate_sample_data()
        total_star = len(df_original[df_original['fraud_type'].astype(str).str.contains('star', na=False)])
        total_cycle = len(df_original[df_original['fraud_type'].astype(str).str.contains('cycle', na=False)])
        total_high_value = len(df_original[df_original['fraud_type'].astype(str).str.contains('high_value', na=False)])
        top_risky = df_original.sort_values('risk_score', ascending=False).head(5)
        most_common = df_original['fraud_type'].value_counts().idxmax() if not df_original.empty else 'None'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Star Patterns", total_star)
        with col2:
            st.metric("Cycle Patterns", total_cycle)
        with col3:
            st.metric("High Value Patterns", total_high_value)
        
        st.markdown("**Top 5 Risky VPAs:**")
        if not top_risky.empty:
            st.table(top_risky[['VPA_from', 'VPA_to', 'risk_score', 'fraud_type']])
        
        st.markdown(f"**Most Common Fraud Type:** {most_common.replace('_', ' ').title()}")
        
        summary_text = f"""
        **Overall Summary:**
        - {total_star} star-shaped fraud patterns detected (multiple senders to one receiver).
        - {total_cycle} cycle fraud patterns detected (circular fund movement).
        - {total_high_value} high value frauds detected (unusually large transactions).
        - The most common fraud type is **{most_common.replace('_', ' ').title()}**.
        
        **Note:** No transactions match your current filters. Try lowering the risk threshold or selecting "All Patterns".
        """
        st.info(summary_text)
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

def detect_star_patterns(G):
    """Detect star-shaped fraud patterns"""
    star_centers = []
    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Star pattern: high in-degree, low out-degree
        if in_degree > 3 and out_degree < 2:
            star_centers.append(node)
    
    return star_centers

def detect_cycle_patterns(G):
    """Detect cycle fraud patterns"""
    try:
        cycles = list(nx.simple_cycles(G))
        return [cycle for cycle in cycles if len(cycle) >= 3]
    except:
        return []

def generate_fraud_alerts(G, df):
    """Generate fraud alerts based on patterns"""
    alerts = []
    alert_id = 0
    
    # Star fraud alerts
    star_centers = detect_star_patterns(G)
    for center in star_centers:
        alert_id += 1
        center_data = df[df['VPA_to'] == center]
        alerts.append({
            'id': alert_id,
            'type': 'Star Fraud',
            'title': f'Star Fraud Center Detected',
            'description': f'Account {center} receiving from {len(center_data)} different sources',
            'risk_score': center_data['risk_score'].mean(),
            'severity': 'Critical' if center_data['risk_score'].mean() > 0.9 else 'High',
            'affected_vpas': f'{center} + {len(center_data)} senders'
        })
    
    # Cycle fraud alerts
    cycles = detect_cycle_patterns(G)
    for i, cycle in enumerate(cycles[:3]):  # Limit to 3 cycles
        alert_id += 1
        cycle_data = df[
            (df['VPA_from'].isin(cycle)) & 
            (df['VPA_to'].isin(cycle))
        ]
        alerts.append({
            'id': alert_id,
            'type': 'Cycle Fraud',
            'title': f'Cycle Fraud Pattern {i+1}',
            'description': f'Circular transaction pattern: {" → ".join(cycle)}',
            'risk_score': cycle_data['risk_score'].mean() if not cycle_data.empty else 0.8,
            'severity': 'Critical',
            'affected_vpas': ', '.join(cycle)
        })
    
    # High value fraud alerts
    high_value = df[df['amount'] > 10000]
    if not high_value.empty:
        alert_id += 1
        alerts.append({
            'id': alert_id,
            'type': 'High Value Fraud',
            'title': 'High Value Transactions Detected',
            'description': f'{len(high_value)} transactions above ₹10,000',
            'risk_score': high_value['risk_score'].mean(),
            'severity': 'High',
            'affected_vpas': f'{len(high_value)} transactions'
        })
    
    return alerts

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