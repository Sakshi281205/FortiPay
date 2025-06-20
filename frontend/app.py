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
    
    # Add sidebar toggle button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("📋 Toggle Sidebar", help="Click to show/hide the navigation sidebar"):
            st.info("Use the hamburger menu (☰) in the top left corner to toggle the sidebar")
    
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
    
    time_data = df.groupby(df['timestamp'].dt.date).agg({
        'transaction_id': 'count',
        'fraud_type': lambda x: (x != 'normal').sum()
    }).reset_index()
    
    time_data.columns = ['date', 'transaction_count', 'fraud_count']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=time_data['date'],
        y=time_data['transaction_count'],
        name='Total Transactions',
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        x=time_data['date'],
        y=time_data['fraud_count'],
        name='Fraud Transactions',
        marker_color='#ff4757'
    ))
    
    fig.update_layout(
        title="Transaction Volume vs Fraud Detection",
        xaxis_title="Date",
        yaxis_title="Count",
        barmode='group',
        height=400
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
    """Comprehensive fraud investigation with detailed analysis"""
    if st.session_state.selected_fraud is None:
        st.error("No transaction selected. Please go back to Fraud Analysis and click 'Investigate'.")
        if st.button("← Back to Fraud Analysis"):
            st.session_state.current_page = 'fraud_analysis'
            st.rerun()
        return

    fraud = st.session_state.selected_fraud
    
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 style='color: #333;'>🔍 Transaction Investigation</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Detailed Analysis: {fraud.get('transaction_id', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back to Analysis", use_container_width=True):
            st.session_state.current_page = 'fraud_analysis'
            st.rerun()
    with col3:
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()

    # --- Transaction Overview Card ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📋 Transaction Details")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_level = "🔴 High Risk" if fraud['risk_score'] > 0.8 else "🟡 Medium Risk" if fraud['risk_score'] > 0.4 else "🟢 Low Risk"
        st.metric("Risk Score", f"{fraud['risk_score']:.3f}", delta=risk_level)
    with col2:
        st.metric("Amount", f"₹{fraud['amount']:,}")
    with col3:
        st.metric("Confidence", f"{fraud['confidence']:.3f}")
    with col4:
        fraud_type_display = str(fraud['fraud_type']).replace('_', ' ').title()
        st.metric("Fraud Type", fraud_type_display)
    
    # Transaction details in a nice format
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sender Details:**")
        st.info(f"**VPA:** {fraud['VPA_from']}")
        st.info(f"**PSP:** {fraud['PSP']}")
    with col2:
        st.markdown("**Receiver Details:**")
        st.info(f"**VPA:** {fraud['VPA_to']}")
        st.info(f"**Transaction Type:** {fraud['transaction_type']}")
    
    st.markdown(f"**Timestamp:** {fraud['timestamp']}")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Risk Score Visualization ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Risk Assessment Visualization")
    
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

    # --- Fraud Pattern Explanation ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🚨 Why is this transaction flagged?")
    
    if 'star' in fraud['fraud_type']:
        explanation = """
        **🔴 Star Fraud Pattern Detected**
        
        This transaction is part of a **star-shaped fraud pattern** where one central account (hub) is receiving funds from multiple sources (spokes). 
        
        **Why it's suspicious:**
        - **Money Laundering**: Centralizing funds from multiple sources to obscure origins
        - **Account Takeover**: Multiple small transactions to test account access
        - **Fake Merchant Scams**: Fraudulent merchants collecting payments from multiple victims
        - **Structuring**: Breaking large amounts into smaller transactions to avoid detection
        
        **Risk Factors:**
        - High in-degree (many incoming transactions to one account)
        - Low out-degree (few outgoing transactions from the central account)
        - Unusual transaction timing patterns
        - Multiple unique senders to one recipient
        
        **Recommended Action:** Monitor the receiver account for unusual activity patterns.
        """
    elif 'cycle' in fraud['fraud_type']:
        explanation = """
        **🔄 Cycle Fraud Pattern Detected**
        
        This transaction is part of a **cycle fraud pattern** where funds move in a circular path between accounts.
        
        **Why it's suspicious:**
        - **Money Laundering**: Obscuring the origin of funds through artificial transaction flow
        - **Wash Trading**: Creating fake transaction volume to manipulate metrics
        - **Transaction Layering**: Multiple hops to make tracing difficult
        - **Structuring**: Breaking large amounts into smaller transactions
        
        **Risk Factors:**
        - Circular transaction flow (A→B→C→A)
        - Similar amounts in cycle transactions
        - Rapid transaction timing
        - Artificial transaction patterns
        - Multiple accounts involved in short time
        
        **Recommended Action:** Investigate the entire transaction cycle for money laundering.
        """
    elif 'high_value' in fraud['fraud_type']:
        explanation = """
        **💰 High-Value Transaction Alert**
        
        This is a **high-value transaction** that significantly exceeds typical transaction amounts.
        
        **Why it's suspicious:**
        - **Account Takeover**: Large unauthorized transfers from compromised accounts
        - **Social Engineering**: Victim tricked into making large transfers
        - **Money Laundering**: Moving large sums quickly to obscure origins
        - **Business Email Compromise**: Fraudulent requests for large payments
        
        **Risk Factors:**
        - Amount significantly higher than account history
        - Unusual transaction timing
        - High-risk recipient account
        - Suspicious transaction context
        - New or recently created recipient accounts
        
        **Recommended Action:** Verify the transaction with both parties immediately.
        """
    else:
        explanation = """
        **⚠️ High-Risk Transaction**
        
        This transaction has been flagged by our AI model due to suspicious patterns in its features and network connections.
        
        **Risk Factors:**
        - Unusual transaction timing patterns
        - Suspicious account behavior
        - Network connection patterns
        - Amount patterns inconsistent with account history
        - High-risk recipient or sender patterns
        
        **Recommended Action:** Review transaction details and account history.
        """
    
    st.markdown(explanation)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Focused Transaction Graph ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader(f"🔗 Focused Fraud Pattern Analysis")
    
    # Explain what the focused graph shows
    st.markdown("""
    **What this graph shows:**
    - 🔴 **Red nodes and edges**: The specific fraud transaction and accounts directly involved
    - 🟠 **Orange nodes and edges**: Suspicious connections that explain why this transaction was flagged
    - The graph focuses only on transactions that are part of the detected fraud pattern, not all account activity
    """)
    
    # Get the full dataset to build the focused graph
    df = generate_sample_data()
    
    # Create a focused graph showing this specific transaction and its immediate connections
    focused_df = create_focused_transaction_graph(df, fraud)
    
    if not focused_df.empty:
        # Create the network graph
        G = create_focused_network_graph(focused_df, fraud)
        fig = create_interactive_network_graph(G, 'Spring', fraud['fraud_type'])
        
        # Add custom title
        fig.update_layout(
            title=f"Focused Analysis: Transaction {fraud.get('transaction_id', 'N/A')}<br><sub>Showing fraud pattern and suspicious connections only</sub>",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True, height=600)
        
        # Show graph statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud-Related Accounts", len([n for n in G.nodes() if G.nodes[n].get('is_fraud_node', False)]))
        with col2:
            st.metric("Suspicious Connections", len([e for e in G.edges(data=True) if not e[2].get('is_fraud_edge', False)]))
        with col3:
            st.metric("Pattern Transactions", len(focused_df))
        
        # Explain the fraud pattern
        st.markdown("**🔍 Pattern Analysis:**")
        if 'star' in fraud['fraud_type']:
            st.info("""
            **Star Pattern Detected**: This transaction is part of a star-shaped fraud pattern where multiple accounts 
            send money to one central account. The red nodes show the central account and the fraud transaction, 
            while orange nodes show other suspicious incoming transactions to the central account.
            """)
        elif 'cycle' in fraud['fraud_type']:
            st.info("""
            **Cycle Pattern Detected**: This transaction is part of a circular money flow pattern. The red nodes show 
            the accounts directly involved in the fraud transaction, while orange nodes show other transactions 
            that form the cycle or are suspiciously connected.
            """)
        elif 'high_value' in fraud['fraud_type']:
            st.info("""
            **High-Value Pattern Detected**: This transaction involves an unusually large amount. The red nodes show 
            the accounts in the fraud transaction, while orange nodes show other high-value or suspicious 
            transactions involving these accounts.
            """)
        else:
            st.info("""
            **Suspicious Pattern Detected**: This transaction shows unusual behavior patterns. The red nodes show 
            the accounts directly involved, while orange nodes show other suspicious transactions that 
            help explain why this transaction was flagged.
            """)
        
        # Edge weight analysis
        st.markdown("**🔗 Connection Details:**")
        edge_weights = []
        for edge in G.edges(data=True):
            edge_data = edge[2]
            edge_weights.append({
                'From': edge[0],
                'To': edge[1],
                'Amount': f"₹{edge_data.get('amount', 0):,}",
                'Risk Score': f"{edge_data.get('risk_score', 0):.3f}",
                'Connection Type': "🔴 Fraud Transaction" if edge_data.get('is_fraud_edge', False) else "🟠 Suspicious"
            })
        
        if edge_weights:
            edge_df = pd.DataFrame(edge_weights)
            st.dataframe(edge_df, use_container_width=True)
    else:
        st.warning("Could not generate a focused graph for this transaction.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Fraud Alerts and Recommendations ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🚨 Fraud Alerts & Recommendations")
    
    # Generate specific alerts for this transaction
    alerts = generate_transaction_alerts(fraud, df)
    
    for alert in alerts:
        alert_class = "alert-critical" if alert['severity'] == 'Critical' else "alert-high"
        
        st.markdown(f"""
        <div class="alert-card {alert_class}">
            <h4 style="margin: 0 0 10px 0; color: #333;">🚨 {alert['title']}</h4>
            <p style="margin: 5px 0; color: #666;">{alert['description']}</p>
            <p style="margin: 5px 0; color: #666;"><strong>Risk Score:</strong> {alert['risk_score']:.3f}</p>
            <p style="margin: 5px 0; color: #666;"><strong>Confidence:</strong> {alert['confidence']:.3f}</p>
            <p style="margin: 5px 0; color: #666;"><strong>Affected VPAs:</strong> {alert['affected_vpas']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Action Recommendations ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎯 Recommended Actions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Immediate Actions:**")
        st.markdown("- 🚨 Flag account for monitoring")
        st.markdown("- 📞 Contact account holder")
        st.markdown("- 🔒 Temporarily freeze if high risk")
        st.markdown("- 📋 Document investigation findings")
    
    with col2:
        st.markdown("**Investigation Steps:**")
        st.markdown("- 📊 Review account history")
        st.markdown("- 🔍 Check related transactions")
        st.markdown("- 📋 Document findings")
        st.markdown("- 🔗 Analyze network connections")
    
    with col3:
        st.markdown("**Prevention:**")
        st.markdown("- ⚠️ Set up alerts for similar patterns")
        st.markdown("- 🔄 Monitor account activity")
        st.markdown("- 📈 Update risk models")
        st.markdown("- 🛡️ Implement additional security measures")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Action Buttons ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("⚡ Take Action")
    
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
    
    # Add transaction_id if it doesn't exist
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = [f'TX{i:06d}' for i in range(len(df))]
    
    # Filters
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Analysis Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fraud_type = st.selectbox("Fraud Pattern", ["All Fraud Patterns", "All Transactions", "Star Fraud", "Cycle Fraud", "High Value Fraud", "Normal Transactions"])
    with col2:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5)
    with col3:
        max_nodes = st.slider("Max Graph Nodes", 10, 200, 50)
    with col4:
        layout_option = st.selectbox("Graph Layout", ["Spring", "Circular", "Shell", "Random", "Kamada-Kawai"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data based on selection
    if fraud_type == "Star Fraud":
        filtered_df = df[df['fraud_type'].astype(str).str.contains('star', na=False)]
    elif fraud_type == "Cycle Fraud":
        filtered_df = df[df['fraud_type'].astype(str).str.contains('cycle', na=False)]
    elif fraud_type == "High Value Fraud":
        filtered_df = df[df['fraud_type'].astype(str).str.contains('high_value', na=False)]
    elif fraud_type == "Normal Transactions":
        filtered_df = df[df['fraud_type'] == 'normal']
    elif fraud_type == "All Fraud Patterns":
        filtered_df = df[df['fraud_type'] != 'normal']
    else:  # "All Transactions"
        filtered_df = df.copy()
    
    filtered_df = filtered_df[filtered_df['risk_score'] >= risk_threshold]
    
    if len(filtered_df) > max_nodes:
        filtered_df = filtered_df.head(max_nodes)
    
    # Show transaction table with investigation buttons
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader(f"📋 Transactions ({len(filtered_df)} found)")
    
    if not filtered_df.empty:
        # Display transactions in a table with investigation buttons
        for idx, row in filtered_df.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 1, 1, 1, 1, 1])
            with col1:
                st.write(f"**{row['VPA_from']}**")
            with col2:
                st.write(f"**{row['VPA_to']}**")
            with col3:
                st.write(f"₹{row['amount']:,}")
            with col4:
                risk_color = "🔴" if row['risk_score'] > 0.8 else "🟡" if row['risk_score'] > 0.6 else "🟢"
                st.write(f"{risk_color} {row['risk_score']:.2f}")
            with col5:
                st.write(f"{row['confidence']:.2f}")
            with col6:
                fraud_type_display = str(row['fraud_type']).replace('_', ' ').title()
                st.write(fraud_type_display)
            with col7:
                if st.button("🔍 Investigate", key=f"investigate_{row.get('transaction_id', idx)}"):
                    st.session_state.selected_fraud = row
                    st.session_state.current_page = 'fraud_details'
                    st.rerun()
            st.markdown("---")
    else:
        st.warning("No transactions match the current filter settings.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
        
        fig = create_interactive_network_graph(G, layout_option, fraud_type)
        st.plotly_chart(fig, use_container_width=True, height=600)
        st.markdown('</div>', unsafe_allow_html=True)

def create_focused_transaction_graph(df, fraud_transaction):
    """Create a focused dataset showing the specific fraud transaction and its suspicious connections"""
    # Get the VPAs involved in the fraud transaction
    fraud_vpa_from = fraud_transaction['VPA_from']
    fraud_vpa_to = fraud_transaction['VPA_to']
    fraud_amount = fraud_transaction['amount']
    fraud_type = fraud_transaction['fraud_type']
    
    # Start with the fraud transaction itself
    focused_df = df[
        (df['VPA_from'] == fraud_vpa_from) & 
        (df['VPA_to'] == fraud_vpa_to) &
        (df['amount'] == fraud_amount)
    ].copy()
    
    # Add suspicious connections based on fraud type
    if 'star' in fraud_type:
        # For star fraud, show the central account (receiver) and its multiple incoming transactions
        central_account = fraud_vpa_to
        star_transactions = df[
            (df['VPA_to'] == central_account) & 
            (df['risk_score'] > 0.5)  # Only high-risk incoming transactions
        ].copy()
        focused_df = pd.concat([focused_df, star_transactions], ignore_index=True)
        
    elif 'cycle' in fraud_type:
        # For cycle fraud, show the transaction cycle
        # Find transactions that form a cycle with the fraud transaction
        cycle_transactions = df[
            ((df['VPA_from'] == fraud_vpa_to) & (df['risk_score'] > 0.5)) |  # Outgoing from receiver
            ((df['VPA_to'] == fraud_vpa_from) & (df['risk_score'] > 0.5))    # Incoming to sender
        ].copy()
        focused_df = pd.concat([focused_df, cycle_transactions], ignore_index=True)
        
    elif 'high_value' in fraud_type:
        # For high-value fraud, show other high-value transactions involving these accounts
        high_value_transactions = df[
            ((df['VPA_from'] == fraud_vpa_from) | (df['VPA_to'] == fraud_vpa_from) |
             (df['VPA_from'] == fraud_vpa_to) | (df['VPA_to'] == fraud_vpa_to)) &
            (df['amount'] > fraud_amount * 0.5) &  # Transactions with similar high amounts
            (df['risk_score'] > 0.4)  # Only risky transactions
        ].copy()
        focused_df = pd.concat([focused_df, high_value_transactions], ignore_index=True)
        
    else:
        # For other fraud types, show recent suspicious transactions involving these accounts
        recent_suspicious = df[
            ((df['VPA_from'] == fraud_vpa_from) | (df['VPA_to'] == fraud_vpa_from) |
             (df['VPA_from'] == fraud_vpa_to) | (df['VPA_to'] == fraud_vpa_to)) &
            (df['risk_score'] > 0.6) &  # Only high-risk transactions
            (df.index != focused_df.index[0] if not focused_df.empty else True)  # Exclude the main fraud transaction
        ].copy()
        focused_df = pd.concat([focused_df, recent_suspicious], ignore_index=True)
    
    # Add a flag to highlight the specific fraud transaction
    focused_df['is_fraud_transaction'] = (
        (focused_df['VPA_from'] == fraud_vpa_from) & 
        (focused_df['VPA_to'] == fraud_vpa_to) &
        (focused_df['amount'] == fraud_amount)
    )
    
    # Remove duplicates and limit to most relevant transactions
    focused_df = focused_df.drop_duplicates().head(20)  # Limit to 20 most relevant transactions
    
    return focused_df

def create_focused_network_graph(df, fraud_transaction):
    """Create a NetworkX graph focused on the specific fraud transaction and its suspicious connections"""
    G = nx.DiGraph()
    
    # Get the fraud transaction details
    fraud_vpa_from = fraud_transaction['VPA_from']
    fraud_vpa_to = fraud_transaction['VPA_to']
    fraud_amount = fraud_transaction['amount']
    fraud_type = fraud_transaction['fraud_type']
    
    # Add nodes and edges from the focused dataset
    for _, row in df.iterrows():
        # Add nodes with attributes
        if not G.has_node(row['VPA_from']):
            is_fraud_node = (row['VPA_from'] in [fraud_vpa_from, fraud_vpa_to])
            G.add_node(row['VPA_from'], 
                      risk_score=row['risk_score'],
                      fraud_type=row['fraud_type'],
                      account_type='sender',
                      is_fraud_node=is_fraud_node,
                      node_type='fraud_related' if is_fraud_node else 'suspicious')
        
        if not G.has_node(row['VPA_to']):
            is_fraud_node = (row['VPA_to'] in [fraud_vpa_from, fraud_vpa_to])
            G.add_node(row['VPA_to'], 
                      risk_score=row['risk_score'],
                      fraud_type=row['fraud_type'],
                      account_type='receiver',
                      is_fraud_node=is_fraud_node,
                      node_type='fraud_related' if is_fraud_node else 'suspicious')
        
        # Add edge with transaction attributes
        is_fraud_edge = (
            row['VPA_from'] == fraud_vpa_from and 
            row['VPA_to'] == fraud_vpa_to and
            row['amount'] == fraud_amount
        )
        
        # Determine edge type based on fraud pattern
        edge_type = 'fraud_transaction' if is_fraud_edge else 'suspicious_connection'
        
        G.add_edge(row['VPA_from'], row['VPA_to'], 
                  amount=row['amount'],
                  timestamp=row['timestamp'],
                  PSP=row['PSP'],
                  transaction_type=row['transaction_type'],
                  risk_score=row['risk_score'],
                  confidence=row['confidence'],
                  is_fraud_edge=is_fraud_edge,
                  edge_type=edge_type)
    
    return G

def generate_transaction_alerts(fraud_transaction, df):
    """Generate specific alerts for a transaction"""
    alerts = []
    
    # High risk score alert
    if fraud_transaction['risk_score'] > 0.8:
        alerts.append({
            'title': 'Critical Risk Score',
            'description': f'Transaction has extremely high risk score of {fraud_transaction["risk_score"]:.3f}',
            'risk_score': fraud_transaction['risk_score'],
            'confidence': fraud_transaction['confidence'],
            'severity': 'Critical',
            'affected_vpas': f"{fraud_transaction['VPA_from']} → {fraud_transaction['VPA_to']}"
        })
    
    # High value alert
    if fraud_transaction['amount'] > 10000:
        alerts.append({
            'title': 'High Value Transaction',
            'description': f'Transaction amount ₹{fraud_transaction["amount"]:,} exceeds normal limits',
            'risk_score': fraud_transaction['risk_score'],
            'confidence': fraud_transaction['confidence'],
            'severity': 'High',
            'affected_vpas': f"{fraud_transaction['VPA_from']} → {fraud_transaction['VPA_to']}"
        })
    
    # Fraud pattern alert
    if 'star' in fraud_transaction['fraud_type']:
        alerts.append({
            'title': 'Star Fraud Pattern',
            'description': 'Transaction is part of a star-shaped fraud pattern',
            'risk_score': fraud_transaction['risk_score'],
            'confidence': fraud_transaction['confidence'],
            'severity': 'Critical',
            'affected_vpas': f"{fraud_transaction['VPA_from']} → {fraud_transaction['VPA_to']}"
        })
    elif 'cycle' in fraud_transaction['fraud_type']:
        alerts.append({
            'title': 'Cycle Fraud Pattern',
            'description': 'Transaction is part of a cycle fraud pattern',
            'risk_score': fraud_transaction['risk_score'],
            'confidence': fraud_transaction['confidence'],
            'severity': 'Critical',
            'affected_vpas': f"{fraud_transaction['VPA_from']} → {fraud_transaction['VPA_to']}"
        })
    elif 'high_value' in fraud_transaction['fraud_type']:
        alerts.append({
            'title': 'High Value Fraud',
            'description': 'Transaction flagged as high-value fraud',
            'risk_score': fraud_transaction['risk_score'],
            'confidence': fraud_transaction['confidence'],
            'severity': 'High',
            'affected_vpas': f"{fraud_transaction['VPA_from']} → {fraud_transaction['VPA_to']}"
        })
    
    # Low confidence alert
    if fraud_transaction['confidence'] < 0.7:
        alerts.append({
            'title': 'Low Confidence Score',
            'description': f'Low confidence score of {fraud_transaction["confidence"]:.3f} indicates uncertain prediction',
            'risk_score': fraud_transaction['risk_score'],
            'confidence': fraud_transaction['confidence'],
            'severity': 'High',
            'affected_vpas': f"{fraud_transaction['VPA_from']} → {fraud_transaction['VPA_to']}"
        })
    
    return alerts

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
    """Create interactive network visualization with focused fraud highlighting"""
    if layout_type == "Spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Shell":
        pos = nx.shell_layout(G)
    elif layout_type == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # Create edges with different styles for fraud vs suspicious connections
    fraud_edge_x = []
    fraud_edge_y = []
    fraud_edge_text = []
    suspicious_edge_x = []
    suspicious_edge_y = []
    suspicious_edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_data = edge[2]
        
        edge_text = f"Amount: ₹{edge_data.get('amount', 'N/A')}<br>Risk: {edge_data.get('risk_score', 'N/A'):.2f}<br>PSP: {edge_data.get('PSP', 'N/A')}<br>Type: {edge_data.get('edge_type', 'N/A')}"
        
        if edge_data.get('is_fraud_edge', False):
            fraud_edge_x.extend([x0, x1, None])
            fraud_edge_y.extend([y0, y1, None])
            fraud_edge_text.extend([edge_text] * 3)
        else:
            suspicious_edge_x.extend([x0, x1, None])
            suspicious_edge_y.extend([y0, y1, None])
            suspicious_edge_text.extend([edge_text] * 3)

    # Fraud transaction edge (thick red)
    fraud_edge_trace = go.Scatter(
        x=fraud_edge_x, y=fraud_edge_y,
        line=dict(width=4, color='#ff0000'),
        hoverinfo='text',
        hovertext=fraud_edge_text,
        mode='lines',
        name='Fraud Transaction')

    # Suspicious connection edges (thin orange)
    suspicious_edge_trace = go.Scatter(
        x=suspicious_edge_x, y=suspicious_edge_y,
        line=dict(width=2, color='#ff6b35'),
        hoverinfo='text',
        hovertext=suspicious_edge_text,
        mode='lines',
        name='Suspicious Connections')

    # Create nodes with different styles for fraud-related vs suspicious
    fraud_node_x = []
    fraud_node_y = []
    fraud_node_text = []
    suspicious_node_x = []
    suspicious_node_y = []
    suspicious_node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_data = G.nodes[node]
        risk_score = node_data.get('risk_score', 0)
        fraud_type = node_data.get('fraud_type', 'normal')
        account_type = node_data.get('account_type', 'unknown')
        node_type = node_data.get('node_type', 'normal')
        
        node_text = f"VPA: {node}<br>Risk: {risk_score:.2f}<br>Type: {fraud_type}<br>Role: {account_type}<br>Node Type: {node_type}"
        
        if node_data.get('is_fraud_node', False):
            fraud_node_x.append(x)
            fraud_node_y.append(y)
            fraud_node_text.append(node_text)
        else:
            suspicious_node_x.append(x)
            suspicious_node_y.append(y)
            suspicious_node_text.append(node_text)

    # Fraud-related nodes (large red circles)
    fraud_node_trace = go.Scatter(
        x=fraud_node_x, y=fraud_node_y,
        mode='markers+text',
        hoverinfo='text',
        text=fraud_node_text,
        textposition="top center",
        marker=dict(
            size=25,
            color='#ff0000',
            line=dict(width=2, color='#cc0000')
        ),
        name='Fraud-Related Accounts')

    # Suspicious nodes (medium orange circles)
    suspicious_node_trace = go.Scatter(
        x=suspicious_node_x, y=suspicious_node_y,
        mode='markers+text',
        hoverinfo='text',
        text=suspicious_node_text,
        textposition="top center",
        marker=dict(
            size=15,
            color='#ff6b35',
            line=dict(width=1, color='#e55a2b')
        ),
        name='Suspicious Accounts')

    fig = go.Figure(data=[fraud_edge_trace, suspicious_edge_trace, fraud_node_trace, suspicious_node_trace],
                   layout=go.Layout(
                       title=f'Focused Fraud Analysis: {fraud_type.replace("_", " ").title()}<br><sub>Red = Fraud Transaction, Orange = Suspicious Connections</sub>',
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=60),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       legend=dict(
                           x=0.02,
                           y=0.98,
                           bgcolor='rgba(255, 255, 255, 0.8)',
                           bordercolor='rgba(0, 0, 0, 0.2)',
                           borderwidth=1
                       ))
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

def show_model_performance():
    """Model performance and metrics page"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">📈 Model Performance</h1>
        <p style="color: #666; margin: 5px 0 0 0;">GNN Model Metrics & Performance Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎯 Model Accuracy Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", "94.2%", delta="+2.1%")
    with col2:
        st.metric("Precision", "91.8%", delta="+1.5%")
    with col3:
        st.metric("Recall", "96.5%", delta="+3.2%")
    with col4:
        st.metric("F1-Score", "94.1%", delta="+2.3%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance over time
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Performance Trends")
    
    # Generate sample performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
    accuracy_data = [92.1, 93.2, 93.8, 94.1, 94.5, 94.2, 94.8, 95.1, 94.9, 94.7, 94.3, 94.2]
    precision_data = [89.5, 90.2, 91.1, 91.5, 91.8, 91.2, 91.9, 92.1, 91.8, 91.5, 91.2, 91.8]
    recall_data = [94.2, 95.1, 95.8, 96.1, 96.5, 96.2, 96.8, 97.1, 96.9, 96.7, 96.3, 96.5]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=accuracy_data, mode='lines+markers', name='Accuracy', line=dict(color='#667eea', width=3)))
    fig.add_trace(go.Scatter(x=dates, y=precision_data, mode='lines+markers', name='Precision', line=dict(color='#ff6b6b', width=3)))
    fig.add_trace(go.Scatter(x=dates, y=recall_data, mode='lines+markers', name='Recall', line=dict(color='#2ed573', width=3)))
    
    fig.update_layout(
        title='Model Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Percentage (%)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fraud pattern detection accuracy
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔍 Fraud Pattern Detection Accuracy")
    
    pattern_data = {
        'Pattern Type': ['Star Fraud', 'Cycle Fraud', 'High Value', 'Account Takeover', 'Money Laundering'],
        'Detection Rate': [96.8, 94.2, 92.5, 95.1, 93.7],
        'False Positives': [2.1, 3.2, 4.1, 2.8, 3.5],
        'Response Time (ms)': [45, 52, 38, 41, 48]
    }
    
    pattern_df = pd.DataFrame(pattern_data)
    st.dataframe(pattern_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model configuration
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Architecture:**")
        st.markdown("- Graph Neural Network (GNN)")
        st.markdown("- 3 Graph Convolutional Layers")
        st.markdown("- 128 Hidden Dimensions")
        st.markdown("- Dropout Rate: 0.3")
        st.markdown("- Learning Rate: 0.001")
        
    with col2:
        st.markdown("**Training Parameters:**")
        st.markdown("- Batch Size: 32")
        st.markdown("- Epochs: 100")
        st.markdown("- Optimizer: Adam")
        st.markdown("- Loss Function: Cross-Entropy")
        st.markdown("- Validation Split: 20%")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_transaction_graph():
    """Global transaction graph visualization"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🕸️ Transaction Network Graph</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Global Transaction Network Visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Graph controls
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Graph Controls")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        layout_type = st.selectbox("Layout Algorithm", ["Spring", "Circular", "Shell", "Kamada-Kawai", "Random"])
    with col2:
        max_nodes = st.slider("Max Nodes to Display", 50, 500, 200)
    with col3:
        risk_threshold = st.slider("Risk Score Threshold", 0.0, 1.0, 0.5, 0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Filter data based on risk threshold
    filtered_df = df[df['risk_score'] >= risk_threshold].head(max_nodes)
    
    if not filtered_df.empty:
        # Create network graph
        G = create_fraud_network_graph(filtered_df)
        fig = create_interactive_network_graph(G, layout_type, 'Global Network')
        
        # Update title
        fig.update_layout(
            title=f'Global Transaction Network (Risk ≥ {risk_threshold})<br><sub>Showing {len(filtered_df)} high-risk transactions</sub>',
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True, height=700)
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", len(G.nodes()))
        with col2:
            st.metric("Total Edges", len(G.edges()))
        with col3:
            st.metric("High Risk Nodes", len([n for n in G.nodes() if G.nodes[n].get('risk_score', 0) > 0.8]))
        with col4:
            st.metric("Average Risk", f"{sum(G.nodes[n].get('risk_score', 0) for n in G.nodes()) / len(G.nodes()):.3f}")
        
        # Node details table
        st.markdown("**📋 Node Details:**")
        node_details = []
        for node in list(G.nodes())[:20]:  # Show first 20 nodes
            node_data = G.nodes[node]
            node_details.append({
                'VPA': node,
                'Risk Score': f"{node_data.get('risk_score', 0):.3f}",
                'Fraud Type': node_data.get('fraud_type', 'normal'),
                'Account Type': node_data.get('account_type', 'unknown')
            })
        
        if node_details:
            node_df = pd.DataFrame(node_details)
            st.dataframe(node_df, use_container_width=True)
    else:
        st.warning("No transactions found with the selected risk threshold.")

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