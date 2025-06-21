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
    """Generate sample transaction data with enhanced fraud patterns and priority scoring"""
    np.random.seed(42)
    n_transactions = 1000
    
    # Generate VPAs
    vpas = [f"user{i}@upi" for i in range(1, 101)]
    
    # Enhanced fraud types with explanations
    fraud_types = {
        'star_fraud_center': {
            'name': 'Star Fraud',
            'description': 'Multiple accounts send money to one central account for money laundering',
            'priority_weight': 0.9,
            'icon': '🔴'
        },
        'cycle_fraud': {
            'name': 'Cycle Fraud', 
            'description': 'Money moves in circles between accounts to obscure origins',
            'priority_weight': 0.8,
            'icon': '🔄'
        },
        'high_value_fraud': {
            'name': 'High-Value Fraud',
            'description': 'Unusually large transactions that exceed normal patterns',
            'priority_weight': 0.7,
            'icon': '💰'
        },
        'account_takeover': {
            'name': 'Account Takeover',
            'description': 'Unauthorized access to legitimate accounts',
            'priority_weight': 0.8,
            'icon': '👥'
        },
        'money_laundering': {
            'name': 'Money Laundering',
            'description': 'Complex patterns to hide illegal money sources',
            'priority_weight': 0.9,
            'icon': '💸'
        },
        'social_engineering': {
            'name': 'Social Engineering',
            'description': 'Tricking users into making fraudulent payments',
            'priority_weight': 0.6,
            'icon': '🎭'
        },
        'upi_spoofing': {
            'name': 'UPI Spoofing',
            'description': 'Fake merchant transactions or QR code scams',
            'priority_weight': 0.7,
            'icon': '📱'
        },
        'normal': {
            'name': 'Normal Transaction',
            'description': 'Legitimate transaction with no suspicious patterns',
            'priority_weight': 0.1,
            'icon': '✅'
        }
    }
    
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
        
        # Generate fraud patterns with enhanced logic
        fraud_prob = random.random()
        if fraud_prob < 0.08:  # 8% star fraud
            fraud_type = 'star_fraud_center'
            risk_score = random.uniform(0.8, 1.0)
        elif fraud_prob < 0.15:  # 7% cycle fraud
            fraud_type = 'cycle_fraud'
            risk_score = random.uniform(0.7, 0.95)
        elif fraud_prob < 0.20:  # 5% high value fraud
            fraud_type = 'high_value_fraud'
            risk_score = random.uniform(0.6, 0.9)
        elif fraud_prob < 0.25:  # 5% account takeover
            fraud_type = 'account_takeover'
            risk_score = random.uniform(0.7, 0.95)
        elif fraud_prob < 0.28:  # 3% money laundering
            fraud_type = 'money_laundering'
            risk_score = random.uniform(0.8, 1.0)
        elif fraud_prob < 0.32:  # 4% social engineering
            fraud_type = 'social_engineering'
            risk_score = random.uniform(0.5, 0.8)
        elif fraud_prob < 0.35:  # 3% UPI spoofing
            fraud_type = 'upi_spoofing'
            risk_score = random.uniform(0.6, 0.85)
        else:  # 65% normal transactions
            fraud_type = 'normal'
            risk_score = random.uniform(0.1, 0.4)
        
        # Calculate priority score based on multiple factors
        fraud_info = fraud_types[fraud_type]
        base_priority = fraud_info['priority_weight']
        amount_factor = min(amount / 10000, 1.0)  # Normalize amount
        risk_factor = risk_score
        confidence = random.uniform(0.6, 0.95)
        
        # Priority score calculation
        priority_score = (base_priority * 0.4 + amount_factor * 0.3 + risk_factor * 0.2 + confidence * 0.1)
        
        # Generate transaction
        transaction = {
            'transaction_id': f'TX{i:06d}',
            'VPA_from': vpa_from,
            'VPA_to': vpa_to,
            'amount': amount,
            'timestamp': timestamp,
            'PSP': random.choice(['Google Pay', 'PhonePe', 'Paytm', 'BHIM']),
            'transaction_type': random.choice(['P2P', 'P2M', 'QR']),
            'fraud_type': fraud_type,
            'fraud_name': fraud_info['name'],
            'fraud_description': fraud_info['description'],
            'fraud_icon': fraud_info['icon'],
            'risk_score': risk_score,
            'confidence': confidence,
            'priority_score': priority_score,
            'priority_level': 'High' if priority_score > 0.7 else 'Medium' if priority_score > 0.4 else 'Low'
        }
        
        data.append(transaction)
    
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
        
        # Ensure fraud_type is a string before using replace
        fraud_type_str = str(fraud['fraud_type']).replace('_', ' ').title()
        
        st.sidebar.markdown(f"""
        <div class="alert-card {alert_class}">
            <h4 style="margin: 0 0 10px 0; color: #333;">🚨 {fraud_type_str}</h4>
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
    
    if st.sidebar.button("📚 Fraud Education", use_container_width=True):
        st.session_state.current_page = 'fraud_education'
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
        priority_color = "🔴" if fraud.get('priority_level', 'Medium') == 'High' else "🟡" if fraud.get('priority_level', 'Medium') == 'Medium' else "🟢"
        st.metric("Priority", f"{priority_color} {fraud.get('priority_level', 'Medium')}")
    
    # Enhanced fraud type display
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fraud Type Details:**")
        fraud_icon = fraud.get('fraud_icon', '⚠️')
        # Ensure fraud_type is a string before using replace
        fraud_type_str = str(fraud['fraud_type']).replace('_', ' ').title()
        fraud_name = fraud.get('fraud_name', fraud_type_str)
        fraud_desc = fraud.get('fraud_description', 'Suspicious transaction pattern detected')
        st.info(f"**{fraud_icon} {fraud_name}**")
        st.info(f"**Description:** {fraud_desc}")
    with col2:
        st.markdown("**Transaction Details:**")
        st.info(f"**Transaction ID:** {fraud.get('transaction_id', 'N/A')}")
        st.info(f"**Priority Score:** {fraud.get('priority_score', 0):.3f}")
    
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
    
    fraud_type = fraud['fraud_type']
    # Ensure fraud_type is a string before using replace
    fraud_type_str = str(fraud_type).replace('_', ' ').title()
    fraud_name = fraud.get('fraud_name', fraud_type_str)
    fraud_desc = fraud.get('fraud_description', 'Suspicious transaction pattern detected')
    
    if fraud_type == 'star_fraud_center':
        explanation = f"""
        **🔴 {fraud_name} Pattern Detected**
        
        {fraud_desc}
        
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
    elif fraud_type == 'cycle_fraud':
        explanation = f"""
        **🔄 {fraud_name} Pattern Detected**
        
        {fraud_desc}
        
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
    elif fraud_type == 'high_value_fraud':
        explanation = f"""
        **💰 {fraud_name} Alert**
        
        {fraud_desc}
        
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
    elif fraud_type == 'account_takeover':
        explanation = f"""
        **👥 {fraud_name} Detected**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Credential Theft**: Unauthorized access to legitimate accounts
        - **SIM Swapping**: Attackers gaining control of phone numbers
        - **Social Engineering**: Tricking users into revealing credentials
        - **Data Breaches**: Compromised credentials from third-party breaches
        
        **Risk Factors:**
        - Sudden behavior change in account
        - New device or location login
        - Unusual transaction patterns
        - Failed login attempts before success
        - High-value transactions from previously low-activity account
        
        **Recommended Action:** Immediately freeze the account and contact the legitimate user.
        """
    elif fraud_type == 'money_laundering':
        explanation = f"""
        **💸 {fraud_name} Pattern Detected**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Complex Layering**: Multiple transaction layers to obscure origins
        - **Structuring**: Breaking large amounts into smaller transactions
        - **Integration**: Mixing illegal funds with legitimate business
        - **Placement**: Initial placement of illegal funds into financial system
        
        **Risk Factors:**
        - Complex transaction chains (12+ accounts)
        - Structured amounts (just under reporting thresholds)
        - Multiple payment methods used
        - Rapid movement of funds
        - Unusual transaction timing
        
        **Recommended Action:** Report to regulatory authorities and freeze involved accounts.
        """
    elif fraud_type == 'social_engineering':
        explanation = f"""
        **🎭 {fraud_name} Detected**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Impersonation**: Attackers pretending to be banks, government, or family
        - **Urgency Tactics**: Creating pressure for immediate action
        - **Emotional Manipulation**: Exploiting fear, sympathy, or greed
        - **Information Gathering**: Collecting personal details for future attacks
        
        **Risk Factors:**
        - Urgent payment requests
        - Unusual recipient (new VPA)
        - Emotional language in transaction notes
        - User hesitation or multiple attempts
        - Requests for personal information
        
        **Recommended Action:** Educate user about social engineering tactics and verify sender identity.
        """
    elif fraud_type == 'upi_spoofing':
        explanation = f"""
        **📱 {fraud_name} Detected**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Fake QR Codes**: Malicious QR codes at payment points
        - **Merchant Impersonation**: Attackers posing as legitimate merchants
        - **Payment Link Scams**: Malicious links in messages or emails
        - **VPA Spoofing**: Similar-looking VPAs to legitimate ones
        
        **Risk Factors:**
        - Similar but different VPAs to known merchants
        - Unusual merchant behavior
        - Multiple complaints about same merchant
        - Transaction amount mismatch
        - Suspicious QR code sources
        
        **Recommended Action:** Verify merchant identity and report suspicious QR codes.
        """
    else:
        explanation = f"""
        **⚠️ {fraud_name}**
        
        {fraud_desc}
        
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
    
    # Interactive Learning Section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎓 Interactive Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Fraud Statistics")
        st.markdown("""
        - **Star Fraud**: 8% of detected fraud
        - **Cycle Fraud**: 7% of detected fraud  
        - **High-Value Fraud**: 5% of detected fraud
        - **Account Takeover**: 5% of detected fraud
        - **Money Laundering**: 3% of detected fraud
        - **Social Engineering**: 4% of detected fraud
        - **UPI Spoofing**: 3% of detected fraud
        - **Normal Transactions**: 65% of all transactions
        """)
    
    with col2:
        st.markdown("### 🔍 Detection Tips")
        st.markdown("""
        **Look for patterns:**
        - Unusual transaction amounts
        - Rapid succession of transactions
        - New or suspicious VPAs
        - Behavioral changes in users
        - Geographic anomalies
        
        **Use technology:**
        - AI-powered pattern recognition
        - Real-time monitoring systems
        - Network analysis tools
        - Machine learning models
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resources Section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📚 Additional Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔗 Official Resources")
        st.markdown("""
        - [RBI UPI Guidelines](https://rbi.org.in)
        - [NPCI Security](https://www.npci.org.in)
        - [Cyber Crime Portal](https://cybercrime.gov.in)
        """)
    
    with col2:
        st.markdown("### 📖 Learning Materials")
        st.markdown("""
        - UPI Security Best Practices
        - Digital Payment Safety Guide
        - Fraud Prevention Handbook
        - Cybersecurity Awareness
        """)
    
    with col3:
        st.markdown("### 📞 Support Channels")
        st.markdown("""
        - **Emergency**: 1930 (Cyber Crime)
        - **Bank Support**: Your bank's helpline
        - **UPI Support**: 1800-111-111
        - **FortiPay Support**: support@fortipay.com
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_fraud_network_graph(df):
    """Create a network graph from transaction data"""
    G = nx.Graph()
    
    # Add nodes (VPAs)
    all_vpas = set(df['VPA_from'].unique()) | set(df['VPA_to'].unique())
    for vpa in all_vpas:
        G.add_node(vpa)
    
    # Add edges (transactions)
    for _, row in df.iterrows():
        G.add_edge(row['VPA_from'], row['VPA_to'], 
                  weight=row['amount'], 
                  risk=row['risk_score'],
                  fraud_type=row['fraud_type'])
    
    return G

def create_interactive_network_graph(G, layout_type, fraud_type):
    """Create an interactive network graph using Plotly"""
    # Calculate layout
    if layout_type == "Spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Shell":
        pos = nx.shell_layout(G)
    elif layout_type == "Random":
        pos = nx.random_layout(G)
    elif layout_type == "Kamada-Kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_colors = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Color edges based on fraud type
        fraud_type_edge = edge[2].get('fraud_type', 'normal')
        if fraud_type_edge == 'star_fraud_center':
            color = '#ff4757'  # Red
        elif fraud_type_edge == 'cycle_fraud':
            color = '#ffa502'  # Orange
        elif fraud_type_edge == 'high_value_fraud':
            color = '#ff6348'  # Tomato
        elif fraud_type_edge == 'account_takeover':
            color = '#ff6b6b'  # Light red
        elif fraud_type_edge == 'money_laundering':
            color = '#ff4757'  # Red
        elif fraud_type_edge == 'social_engineering':
            color = '#ffa502'  # Orange
        elif fraud_type_edge == 'upi_spoofing':
            color = '#ff6348'  # Tomato
        else:
            color = '#2ed573'  # Green for normal
        
        # Add color for each point in the edge (start, end, None)
        edge_colors.extend([color, color, color])
    
    # Prepare node data
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_labels = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate node size based on degree
        degree = G.degree(node)
        node_sizes.append(10 + degree * 2)
        
        # Color nodes based on fraud involvement
        fraud_edges = [edge for edge in G.edges(node, data=True) if edge[2].get('fraud_type', 'normal') != 'normal']
        if fraud_edges:
            node_colors.append('#ff4757')  # Red for fraud
        else:
            node_colors.append('#2ed573')  # Green for normal
        
        node_labels.append(node)
    
    # Create edge trace - use a single color for all edges to avoid the list issue
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#cccccc'),  # Use a single color
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="middle center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,
            color=node_colors,
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Transaction Network Graph - {fraud_type}',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

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
    
    # Enhanced Filters
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Analysis Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fraud_type = st.selectbox("Fraud Pattern", [
            "All Transactions", 
            "All Fraud Patterns", 
            "Star Fraud", 
            "Cycle Fraud", 
            "High-Value Fraud",
            "Account Takeover",
            "Money Laundering",
            "Social Engineering",
            "UPI Spoofing",
            "Normal Transactions"
        ])
    with col2:
        priority_filter = st.selectbox("Priority Level", ["All", "High", "Medium", "Low"])
    with col3:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5)
    with col4:
        max_nodes = st.slider("Max Graph Nodes", 10, 300, 50)
    
    col5, col6 = st.columns(2)
    with col5:
        layout_option = st.selectbox("Graph Layout", ["Spring", "Circular", "Shell", "Random", "Kamada-Kawai"])
    with col6:
        min_amount = st.number_input("Min Amount (₹)", min_value=0, value=0, step=100)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data based on selection
    if fraud_type == "Star Fraud":
        filtered_df = df[df['fraud_type'] == 'star_fraud_center']
    elif fraud_type == "Cycle Fraud":
        filtered_df = df[df['fraud_type'] == 'cycle_fraud']
    elif fraud_type == "High-Value Fraud":
        filtered_df = df[df['fraud_type'] == 'high_value_fraud']
    elif fraud_type == "Account Takeover":
        filtered_df = df[df['fraud_type'] == 'account_takeover']
    elif fraud_type == "Money Laundering":
        filtered_df = df[df['fraud_type'] == 'money_laundering']
    elif fraud_type == "Social Engineering":
        filtered_df = df[df['fraud_type'] == 'social_engineering']
    elif fraud_type == "UPI Spoofing":
        filtered_df = df[df['fraud_type'] == 'upi_spoofing']
    elif fraud_type == "Normal Transactions":
        filtered_df = df[df['fraud_type'] == 'normal']
    elif fraud_type == "All Fraud Patterns":
        filtered_df = df[df['fraud_type'] != 'normal']
    else:  # "All Transactions"
        filtered_df = df.copy()
    
    # Apply additional filters
    filtered_df = filtered_df[filtered_df['risk_score'] >= risk_threshold]
    filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
    
    if priority_filter != "All":
        filtered_df = filtered_df[filtered_df['priority_level'] == priority_filter]
    
    if len(filtered_df) > max_nodes:
        filtered_df = filtered_df.head(max_nodes)
    
    # Show transaction table with enhanced information
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader(f"📋 Transactions ({len(filtered_df)} found)")
    
    if not filtered_df.empty:
        # Display transactions in a table with investigation buttons
        for idx, row in filtered_df.iterrows():
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 2, 1, 1, 1, 1, 1, 1])
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
                priority_color = "🔴" if row['priority_level'] == 'High' else "🟡" if row['priority_level'] == 'Medium' else "🟢"
                st.write(f"{priority_color} {row['priority_level']}")
            with col6:
                st.write(f"{row['fraud_icon']} {row['fraud_name']}")
            with col7:
                st.write(f"{row['confidence']:.2f}")
            with col8:
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
        
        # Enhanced Graph statistics
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
        
        # Fraud type distribution
        if 'fraud_type' in filtered_df.columns:
            fraud_counts = filtered_df['fraud_type'].value_counts()
            if len(fraud_counts) > 1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("📊 Fraud Type Distribution")
                
                fig = px.pie(
                    values=fraud_counts.values,
                    names=fraud_counts.index,
                    title="Distribution of Fraud Types in Current Filter"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Network visualization
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🕸️ GNN Transaction Network Graph")
        
        fig = create_interactive_network_graph(G, layout_option, fraud_type)
        st.plotly_chart(fig, use_container_width=True, height=600)
        st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance():
    """Model performance metrics and evaluation"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">📈 Model Performance</h1>
        <p style="color: #666; margin: 5px 0 0 0;">GNN Fraud Detection Model Metrics & Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Calculate performance metrics
    total_transactions = len(df)
    fraud_transactions = len(df[df['fraud_type'] != 'normal'])
    normal_transactions = total_transactions - fraud_transactions
    
    # Simulated model performance metrics
    accuracy = 0.94
    precision = 0.89
    recall = 0.92
    f1_score = 0.90
    
    # Display metrics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Precision", f"{precision:.1%}")
    with col3:
        st.metric("Recall", f"{recall:.1%}")
    with col4:
        st.metric("F1-Score", f"{f1_score:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance charts
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        fig = go.Figure(data=go.Heatmap(
            z=[[850, 50], [30, 70]],
            x=['Predicted Normal', 'Predicted Fraud'],
            y=['Actual Normal', 'Actual Fraud'],
            colorscale='Blues',
            text=[[850, 50], [30, 70]],
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        fig.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC Curve
        fpr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        tpr = [0, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 1.0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#667eea')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='red', dash='dash')))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_transaction_graph():
    """Interactive transaction graph visualization"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🕸️ Transaction Graph</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Interactive Network Visualization of UPI Transactions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_sample_data()
    
    # Graph controls
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Graph Controls")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_nodes = st.slider("Max Nodes", 10, 300, 50)
    with col2:
        layout_type = st.selectbox("Layout", ["Spring", "Circular", "Shell", "Random", "Kamada-Kawai"])
    with col3:
        show_labels = st.checkbox("Show Labels", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    if len(df) > max_nodes:
        df = df.head(max_nodes)
    
    # Create network graph
    G = create_fraud_network_graph(df)
    
    # Display graph statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", G.number_of_nodes())
    with col2:
        st.metric("Edges", G.number_of_edges())
    with col3:
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        st.metric("Avg Degree", f"{avg_degree:.1f}")
    with col4:
        density = nx.density(G)
        st.metric("Density", f"{density:.3f}")
    
    # Create interactive graph
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🕸️ Interactive Network Graph")
    
    fig = create_interactive_network_graph(G, layout_type, "All Transactions")
    st.plotly_chart(fig, use_container_width=True, height=600)
    st.markdown('</div>', unsafe_allow_html=True)

def show_fraud_education():
    """Comprehensive fraud education page with detailed explanations and examples"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">📚 Fraud Education Center</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Learn About UPI Fraud Types, Prevention & Best Practices</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fraud Types Overview
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎯 Understanding UPI Fraud Types")
    
    # Create tabs for different fraud types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔴 Star Fraud", "🔄 Cycle Fraud", "💰 High-Value", "👥 Account Takeover", 
        "💸 Money Laundering", "🎭 Social Engineering", "📱 UPI Spoofing", "✅ Prevention"
    ])
    
    with tab1:
        st.markdown("### 🔴 Star Fraud Pattern")
        st.markdown("""
        **What it is:** Multiple accounts send money to one central account, often for money laundering purposes.
        
        **How it works:**
        - Multiple small transactions from different VPAs
        - All converge to a single destination account
        - Attempts to avoid detection by staying under radar
        
        **Real-world example:**
        ```
        user1@upi → central@upi (₹500)
        user2@upi → central@upi (₹750)
        user3@upi → central@upi (₹300)
        user4@upi → central@upi (₹600)
        ```
        
        **Detection signs:**
        - One account receives from many different sources
        - Transactions happen in quick succession
        - Amounts are just below reporting thresholds
        """)
        
        # Interactive example
        if st.button("🔍 View Star Fraud Example", key="star_example"):
            st.info("""
            **Example Detection:**
            - Account 'central@upi' received 15 transactions in 2 hours
            - Total amount: ₹12,500 from 15 different VPAs
            - Risk Score: 0.89 (High)
            - Priority: High (Money laundering pattern detected)
            """)
    
    with tab2:
        st.markdown("### 🔄 Cycle Fraud Pattern")
        st.markdown("""
        **What it is:** Money moves in circles between accounts to obscure the original source.
        
        **How it works:**
        - A → B → C → D → A (circular flow)
        - Creates complex transaction chains
        - Makes tracing difficult for investigators
        
        **Real-world example:**
        ```
        A → B (₹1000)
        B → C (₹950)
        C → D (₹900)
        D → A (₹850)
        ```
        
        **Detection signs:**
        - Circular transaction patterns
        - Decreasing amounts (fees deducted)
        - Multiple hops in short time
        """)
        
        if st.button("🔍 View Cycle Fraud Example", key="cycle_example"):
            st.info("""
            **Example Detection:**
            - 8-account cycle detected
            - Total cycle time: 45 minutes
            - Amount degradation: ₹2000 → ₹1800
            - Risk Score: 0.85 (High)
            - Priority: High (Complex laundering pattern)
            """)
    
    with tab3:
        st.markdown("### 💰 High-Value Fraud")
        st.markdown("""
        **What it is:** Unusually large transactions that exceed normal user patterns.
        
        **How it works:**
        - Transactions much larger than user's history
        - May involve compromised accounts
        - Often rushed or urgent requests
        
        **Real-world example:**
        ```
        Normal user pattern: ₹100-2000 per transaction
        Fraudulent transaction: ₹25,000 (25x normal)
        ```
        
        **Detection signs:**
        - Amount significantly higher than user's average
        - Transaction timing unusual (late night, weekends)
        - User behavior change (new device, location)
        """)
        
        if st.button("🔍 View High-Value Example", key="high_value_example"):
            st.info("""
            **Example Detection:**
            - User average: ₹500 per transaction
            - Current transaction: ₹35,000
            - Deviation: 70x normal amount
            - Risk Score: 0.78 (High)
            - Priority: Medium (Requires verification)
            """)
    
    with tab4:
        st.markdown("### 👥 Account Takeover")
        st.markdown("""
        **What it is:** Unauthorized access to legitimate user accounts.
        
        **How it works:**
        - Credential theft (phishing, data breaches)
        - SIM swapping attacks
        - Social engineering to gain access
        
        **Real-world example:**
        ```
        Legitimate user: user@upi (normal patterns)
        Attacker gains access: user@upi (unusual activity)
        ```
        
        **Detection signs:**
        - Sudden behavior change
        - New device/location login
        - Unusual transaction patterns
        - Failed login attempts before success
        """)
        
        if st.button("🔍 View Account Takeover Example", key="takeover_example"):
            st.info("""
            **Example Detection:**
            - New device login from different city
            - Transaction pattern changed dramatically
            - Multiple failed login attempts
            - Risk Score: 0.82 (High)
            - Priority: High (Immediate action required)
            """)
    
    with tab5:
        st.markdown("### 💸 Money Laundering")
        st.markdown("""
        **What it is:** Complex patterns to hide illegal money sources.
        
        **How it works:**
        - Multiple layers of transactions
        - Mixing legitimate and illegal funds
        - Using multiple accounts and services
        
        **Real-world example:**
        ```
        Illegal source → Account A → Account B → Account C → Legitimate business
        ```
        
        **Detection signs:**
        - Complex transaction chains
        - Mixing of small and large amounts
        - Use of multiple payment methods
        - Structured transactions (just under limits)
        """)
        
        if st.button("🔍 View Money Laundering Example", key="laundering_example"):
            st.info("""
            **Example Detection:**
            - 12-account transaction chain
            - Structured amounts (₹9,900 each)
            - Multiple payment methods used
            - Risk Score: 0.91 (Very High)
            - Priority: High (Regulatory reporting required)
            """)
    
    with tab6:
        st.markdown("### 🎭 Social Engineering")
        st.markdown("""
        **What it is:** Tricking users into making fraudulent payments.
        
        **How it works:**
        - Impersonation (bank, government, family)
        - Urgency and pressure tactics
        - Emotional manipulation
        
        **Real-world example:**
        ```
        "Your account has been compromised. Send ₹10,000 to secure it."
        "Your relative is in emergency. Send money immediately."
        ```
        
        **Detection signs:**
        - Urgent payment requests
        - Unusual recipient (new VPA)
        - Emotional language in transaction notes
        - User hesitation or multiple attempts
        """)
        
        if st.button("🔍 View Social Engineering Example", key="social_example"):
            st.info("""
            **Example Detection:**
            - Urgent payment to new VPA
            - Transaction note: "Emergency hospital bill"
            - User made 3 attempts (hesitation)
            - Risk Score: 0.65 (Medium)
            - Priority: Medium (Requires user education)
            """)
    
    with tab7:
        st.markdown("### 📱 UPI Spoofing")
        st.markdown("""
        **What it is:** Fake merchant transactions or QR code scams.
        
        **How it works:**
        - Fake QR codes at payment points
        - Impersonating legitimate merchants
        - Malicious payment links
        
        **Real-world example:**
        ```
        Legitimate QR: merchant@upi
        Fake QR: merchant@upi (slightly different)
        ```
        
        **Detection signs:**
        - Similar but different VPAs
        - Unusual merchant behavior
        - Multiple complaints about same merchant
        - Transaction amount mismatch
        """)
        
        if st.button("🔍 View UPI Spoofing Example", key="spoofing_example"):
            st.info("""
            **Example Detection:**
            - VPA similar to known merchant
            - Multiple complaints in 24 hours
            - Amount doesn't match expected
            - Risk Score: 0.72 (Medium-High)
            - Priority: Medium (Merchant verification needed)
            """)
    
    with tab8:
        st.markdown("### ✅ Prevention Best Practices")
        st.markdown("""
        **For Users:**
        - ✅ Verify VPAs carefully before payment
        - ✅ Don't share OTP with anyone
        - ✅ Use UPI PIN only on trusted apps
        - ✅ Check transaction details before confirming
        - ✅ Report suspicious activity immediately
        
        **For Banks/PSPs:**
        - ✅ Implement multi-factor authentication
        - ✅ Monitor transaction patterns
        - ✅ Use AI/ML for fraud detection
        - ✅ Regular security audits
        - ✅ User education programs
        
        **For Merchants:**
        - ✅ Secure QR code generation
        - ✅ Regular VPA verification
        - ✅ Monitor transaction patterns
        - ✅ Report suspicious activity
        """)
        
        # Interactive quiz
        st.markdown("### 🧠 Quick Quiz")
        quiz_question = st.selectbox(
            "What should you do if you receive an urgent payment request from an unknown VPA?",
            ["Send money immediately", "Verify the sender's identity first", "Ignore all payment requests", "Share your UPI PIN"]
        )
        
        if st.button("Submit Answer"):
            if quiz_question == "Verify the sender's identity first":
                st.success("✅ Correct! Always verify before sending money.")
            else:
                st.error("❌ Incorrect. Always verify the sender's identity before making payments.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive Learning Section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎓 Interactive Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Fraud Statistics")
        st.markdown("""
        - **Star Fraud**: 8% of detected fraud
        - **Cycle Fraud**: 7% of detected fraud  
        - **High-Value Fraud**: 5% of detected fraud
        - **Account Takeover**: 5% of detected fraud
        - **Money Laundering**: 3% of detected fraud
        - **Social Engineering**: 4% of detected fraud
        - **UPI Spoofing**: 3% of detected fraud
        - **Normal Transactions**: 65% of all transactions
        """)
    
    with col2:
        st.markdown("### 🔍 Detection Tips")
        st.markdown("""
        **Look for patterns:**
        - Unusual transaction amounts
        - Rapid succession of transactions
        - New or suspicious VPAs
        - Behavioral changes in users
        - Geographic anomalies
        
        **Use technology:**
        - AI-powered pattern recognition
        - Real-time monitoring systems
        - Network analysis tools
        - Machine learning models
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resources Section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📚 Additional Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔗 Official Resources")
        st.markdown("""
        - [RBI UPI Guidelines](https://rbi.org.in)
        - [NPCI Security](https://www.npci.org.in)
        - [Cyber Crime Portal](https://cybercrime.gov.in)
        """)
    
    with col2:
        st.markdown("### 📖 Learning Materials")
        st.markdown("""
        - UPI Security Best Practices
        - Digital Payment Safety Guide
        - Fraud Prevention Handbook
        - Cybersecurity Awareness
        """)
    
    with col3:
        st.markdown("### 📞 Support Channels")
        st.markdown("""
        - **Emergency**: 1930 (Cyber Crime)
        - **Bank Support**: Your bank's helpline
        - **UPI Support**: 1800-111-111
        - **FortiPay Support**: support@fortipay.com
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application logic"""
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
    elif st.session_state.current_page == 'fraud_education':
        show_fraud_education()
    else:
        # Default to dashboard
        st.session_state.current_page = 'dashboard'
        show_dashboard()

if __name__ == "__main__":
    main() 