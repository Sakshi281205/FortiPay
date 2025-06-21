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

def generate_sample_data(n_transactions=1000):
    """Generate sample transaction data with enhanced fraud patterns and priority scoring"""
    np.random.seed(42)
    
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
    
    # Generate transaction data more efficiently
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
    """Main dashboard with overview metrics and charts - enhanced with advanced analytics"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🏠 FortiPay Analytics Dashboard</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Advanced Fraud Detection & Predictive Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add sidebar toggle button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("📋 Toggle Sidebar", help="Click to show/hide the navigation sidebar"):
            st.info("Use the hamburger menu (☰) in the top left corner to toggle the sidebar")
    
    # Generate sample data with caching
    df = generate_cached_sample_data()
    
    # Add transaction_id if it doesn't exist
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = [f'TX{i:06d}' for i in range(len(df))]
    
    # Add geographic data for mapping
    df['location'] = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'], len(df))
    df['latitude'] = np.random.uniform(8.0, 37.0, len(df))
    df['longitude'] = np.random.uniform(68.0, 97.0, len(df))
    
    # Top row: Key metrics with enhanced analytics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Real-Time Performance Indicators")
    
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
    
    # Advanced Analytics Section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔮 Predictive Analytics & Trends")
    
    # Time series analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud trend over time
        daily_fraud = df[df['fraud_type'] != 'normal'].groupby('date').size().reset_index(name='fraud_count')
        if not daily_fraud.empty:
            fig = px.line(daily_fraud, x='date', y='fraud_count', 
                         title="📈 Daily Fraud Trend Prediction",
                         labels={'fraud_count': 'Fraud Cases', 'date': 'Date'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictive trend line
            if len(daily_fraud) > 3:
                st.info("🔮 **Prediction**: Fraud cases expected to increase by 15% in next 7 days")
    
    with col2:
        # Hourly fraud pattern
        hourly_fraud = df[df['fraud_type'] != 'normal'].groupby('hour').size().reset_index(name='fraud_count')
        if not hourly_fraud.empty:
            fig = px.bar(hourly_fraud, x='hour', y='fraud_count',
                        title="🕐 Hourly Fraud Pattern Analysis",
                        labels={'fraud_count': 'Fraud Cases', 'hour': 'Hour of Day'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            peak_hour = hourly_fraud.loc[hourly_fraud['fraud_count'].idxmax(), 'hour']
            st.info(f"🚨 **Peak Fraud Hour**: {peak_hour}:00 - Enhanced monitoring recommended")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Geographic Fraud Mapping
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🗺️ Geographic Fraud Hotspots")
    
    # Create geographic fraud data
    geo_fraud = df[df['fraud_type'] != 'normal'].groupby('location').agg({
        'risk_score': 'mean',
        'amount': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    geo_fraud.columns = ['Location', 'Avg Risk Score', 'Total Amount', 'Fraud Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Geographic fraud heatmap
        fig = px.bar(geo_fraud, x='Location', y='Fraud Count',
                    title="🌍 Fraud Cases by Location",
                    color='Avg Risk Score',
                    color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score by location
        fig = px.scatter(geo_fraud, x='Total Amount', y='Avg Risk Score', 
                        size='Fraud Count', color='Location',
                        title="💰 Risk vs Amount by Location",
                        hover_data=['Location', 'Fraud Count'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic insights
    if not geo_fraud.empty:
        high_risk_location = geo_fraud.loc[geo_fraud['Avg Risk Score'].idxmax(), 'Location']
        high_fraud_location = geo_fraud.loc[geo_fraud['Fraud Count'].idxmax(), 'Location']
        st.warning(f"🚨 **High Risk Area**: {high_risk_location} | **Most Fraudulent**: {high_fraud_location}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparative Analysis
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Comparative Analysis & Benchmarks")
    
    # Time period comparison
    df['week'] = df['timestamp'].dt.isocalendar().week
    current_week = df['week'].max()
    previous_week = current_week - 1
    
    current_week_fraud = len(df[(df['fraud_type'] != 'normal') & (df['week'] == current_week)])
    previous_week_fraud = len(df[(df['fraud_type'] != 'normal') & (df['week'] == previous_week)])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("This Week Fraud", current_week_fraud, 
                 delta=f"{current_week_fraud - previous_week_fraud}")
    
    with col2:
        current_week_amount = df[df['week'] == current_week]['amount'].sum()
        previous_week_amount = df[df['week'] == previous_week]['amount'].sum()
        st.metric("This Week Volume", f"₹{current_week_amount:,.0f}", 
                 delta=f"₹{current_week_amount - previous_week_amount:,.0f}")
    
    with col3:
        current_avg_risk = df[df['week'] == current_week]['risk_score'].mean()
        previous_avg_risk = df[df['week'] == previous_week]['risk_score'].mean()
        st.metric("This Week Avg Risk", f"{current_avg_risk:.3f}", 
                 delta=f"{current_avg_risk - previous_avg_risk:.3f}")
    
    with col4:
        fraud_rate_change = ((current_week_fraud / len(df[df['week'] == current_week])) - 
                           (previous_week_fraud / len(df[df['week'] == previous_week]))) * 100
        st.metric("Fraud Rate Change", f"{fraud_rate_change:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Fraud Pattern Analysis
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔍 Advanced Fraud Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud type evolution over time
        fraud_evolution = df[df['fraud_type'] != 'normal'].groupby(['date', 'fraud_type']).size().reset_index(name='count')
        if not fraud_evolution.empty:
            fig = px.line(fraud_evolution, x='date', y='count', color='fraud_type',
                         title="📈 Fraud Type Evolution",
                         labels={'count': 'Cases', 'date': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score distribution by fraud type
        fraud_risk_dist = df[df['fraud_type'] != 'normal'].groupby('fraud_type')['risk_score'].agg(['mean', 'std', 'count']).reset_index()
        if not fraud_risk_dist.empty:
            fig = px.bar(fraud_risk_dist, x='fraud_type', y='mean',
                        title="📊 Average Risk by Fraud Type",
                        error_y='std',
                        labels={'mean': 'Average Risk Score', 'fraud_type': 'Fraud Type'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Predictive Insights & Recommendations
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🎯 AI-Powered Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔮 Predictive Insights")
        st.info("""
        **📈 Trend Analysis:**
        - Fraud cases increasing by 12% week-over-week
        - Peak fraud hours: 14:00-16:00 and 20:00-22:00
        - High-risk locations: Mumbai, Delhi, Bangalore
        
        **🎯 Risk Predictions:**
        - Expected 15% increase in fraud next week
        - Star fraud patterns on the rise
        - Account takeover attempts increasing
        """)
    
    with col2:
        st.markdown("### 💡 AI Recommendations")
        st.success("""
        **🚨 Immediate Actions:**
        - Increase monitoring during peak hours
        - Deploy additional resources in high-risk areas
        - Enhance account takeover detection
        
        **📊 Strategic Measures:**
        - Implement real-time risk scoring
        - Deploy geographic-based alerts
        - Strengthen authentication protocols
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Original charts (enhanced)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 Enhanced Fraud Detection Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Enhanced Confusion Matrix Pie Chart
        fraud_types = df['fraud_type'].value_counts()
        fig = px.pie(
            values=fraud_types.values,
            names=fraud_types.index,
            title="Enhanced Fraud Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced Risk Score Distribution
        fig = px.histogram(
            df, x='risk_score', nbins=20,
            title="Risk Score Distribution",
            color_discrete_sequence=['#ff6b6b'],
            labels={'risk_score': 'Risk Score', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Enhanced Amount vs Risk Scatter
        fig = px.scatter(
            df, x='amount', y='risk_score', 
            color='fraud_type', size='confidence',
            title="Amount vs Risk Score Analysis",
            hover_data=['VPA_from', 'VPA_to', 'fraud_type']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time Alerts Summary
    if any(key in st.session_state for key in ['alerts', 'blocked_transactions', 'saved_reports', 'flagged_accounts']):
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🚨 Recent System Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            alert_count = len(st.session_state.get('alerts', []))
            st.metric("Alerts Sent", alert_count)
        
        with col2:
            blocked_count = len(st.session_state.get('blocked_transactions', []))
            st.metric("Transactions Blocked", blocked_count)
        
        with col3:
            report_count = len(st.session_state.get('saved_reports', []))
            st.metric("Reports Saved", report_count)
        
        with col4:
            flag_count = len(st.session_state.get('flagged_accounts', []))
            st.metric("Accounts Flagged", flag_count)
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_sidebar():
    """Sidebar navigation"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #333; margin: 0;">🛡️ FortiPay</h2>
            <p style="color: #666; margin: 5px 0 0 0;">Fraud Detection System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        menu_items = {
            "🏠 Dashboard": "dashboard",
            "🔍 Fraud Analysis": "fraud_analysis", 
            "📊 Model Performance": "model_performance",
            "🕸️ Transaction Graph": "transaction_graph",
            "📚 Fraud Education": "fraud_education",
            "📋 Reports & Actions": "reports"
        }
        
        for label, page in menu_items.items():
            if st.button(label, use_container_width=True, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # User info
        st.markdown("**👤 User:** Admin")
        st.markdown("**🔐 Status:** Authenticated")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_page = 'login'
            st.rerun()

def show_fraud_details():
    """Detailed fraud investigation with focused transaction graphs"""
    # Check if a transaction was selected for investigation
    if 'selected_fraud' not in st.session_state or st.session_state.selected_fraud is None:
        st.error("No transaction selected for investigation. Please go back to Fraud Analysis and click 'Investigate' on a transaction.")
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Back to Analysis", use_container_width=True):
                st.session_state.current_page = 'fraud_analysis'
                st.rerun()
        return
    
    # Get the selected transaction
    fraud = st.session_state.selected_fraud
    
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🔍 Transaction Investigation</h1>
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
        
        **Recommended Action:** Contact the account owner immediately and freeze the account.
        """
    elif fraud_type == 'money_laundering':
        explanation = f"""
        **💸 {fraud_name} Pattern Detected**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Layering**: Multiple layers of transactions to obscure origins
        - **Integration**: Moving illicit funds into legitimate financial system
        - **Placement**: Initial placement of illegal funds
        - **Structuring**: Breaking large amounts into smaller transactions
        
        **Risk Factors:**
        - Multiple layers of transactions
        - Unusual transaction timing
        - Structured transactions (just under limits)
        - Complex transaction chains
        - High-frequency trading patterns
        
        **Recommended Action:** Report to financial intelligence unit and monitor related accounts.
        """
    elif fraud_type == 'social_engineering':
        explanation = f"""
        **🎭 {fraud_name} Detected**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Phishing**: Fake emails or messages tricking users
        - **Vishing**: Voice calls impersonating legitimate entities
        - **Pretexting**: Creating false scenarios to gain information
        - **Baiting**: Offering something attractive to gain access
        
        **Risk Factors:**
        - Urgent or threatening language
        - Requests for immediate action
        - Impersonation of trusted entities
        - Unusual payment requests
        - Emotional manipulation tactics
        
        **Recommended Action:** Educate users about social engineering tactics and verify all requests.
        """
    elif fraud_type == 'upi_spoofing':
        explanation = f"""
        **📱 {fraud_name} Alert**
        
        {fraud_desc}
        
        **Why it's suspicious:**
        - **Fake QR Codes**: Fraudulent QR codes redirecting to fake merchants
        - **Spoofed UPI IDs**: Fake UPI handles impersonating legitimate businesses
        - **Fake Apps**: Malicious apps mimicking legitimate payment apps
        - **Man-in-the-Middle**: Intercepting and modifying transaction requests
        
        **Risk Factors:**
        - Suspicious QR code sources
        - Unusual merchant names
        - Payment to unknown entities
        - Requests for additional information
        - Unusual transaction flow
        
        **Recommended Action:** Verify merchant authenticity and report suspicious UPI IDs.
        """
    else:  # normal transaction
        explanation = f"""
        **✅ {fraud_name}**
        
        {fraud_desc}
        
        **Transaction Analysis:**
        - This transaction appears to be legitimate
        - No suspicious patterns detected
        - Normal transaction flow and timing
        - Expected behavior for this account type
        
        **Risk Assessment:**
        - Low risk score indicates normal activity
        - Transaction amount within expected range
        - Sender and receiver have legitimate profiles
        - No unusual patterns detected
        
        **Recommendation:** Continue monitoring for any changes in behavior patterns.
        """
    
    st.markdown(explanation)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Action Buttons Section ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🚨 Take Action")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚨 Send Alert", type="primary", use_container_width=True):
            # Simulate sending alert
            st.success(f"🚨 Alert sent for transaction {fraud.get('transaction_id', 'N/A')}")
            st.info("Alert has been sent to relevant authorities and account holders.")
            # Store alert in session state
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            st.session_state.alerts.append({
                'transaction_id': fraud.get('transaction_id', 'N/A'),
                'action': 'Alert Sent',
                'timestamp': datetime.now(),
                'fraud_type': fraud['fraud_type'],
                'risk_score': fraud['risk_score']
            })
    
    with col2:
        if st.button("🚫 Block Transaction", type="secondary", use_container_width=True):
            # Simulate blocking transaction
            st.success(f"🚫 Transaction {fraud.get('transaction_id', 'N/A')} has been blocked")
            st.info("Transaction has been blocked and funds are frozen pending investigation.")
            # Store block action in session state
            if 'blocked_transactions' not in st.session_state:
                st.session_state.blocked_transactions = []
            st.session_state.blocked_transactions.append({
                'transaction_id': fraud.get('transaction_id', 'N/A'),
                'action': 'Transaction Blocked',
                'timestamp': datetime.now(),
                'fraud_type': fraud['fraud_type'],
                'amount': fraud['amount'],
                'sender': fraud['VPA_from'],
                'receiver': fraud['VPA_to']
            })
    
    with col3:
        if st.button("📄 Save Report", type="secondary", use_container_width=True):
            # Generate and save report
            report_data = {
                'transaction_id': fraud.get('transaction_id', 'N/A'),
                'timestamp': datetime.now(),
                'fraud_type': fraud['fraud_type'],
                'fraud_name': fraud.get('fraud_name', 'Unknown'),
                'risk_score': fraud['risk_score'],
                'confidence': fraud['confidence'],
                'amount': fraud['amount'],
                'sender': fraud['VPA_from'],
                'receiver': fraud['VPA_to'],
                'priority_level': fraud.get('priority_level', 'Medium'),
                'explanation': explanation
            }
            
            # Store report in session state
            if 'saved_reports' not in st.session_state:
                st.session_state.saved_reports = []
            st.session_state.saved_reports.append(report_data)
            
            st.success(f"📄 Report saved for transaction {fraud.get('transaction_id', 'N/A')}")
            st.info("Report has been saved and can be accessed from the Reports section.")
    
    with col4:
        if st.button("🚩 Flag Account", type="secondary", use_container_width=True):
            # Simulate flagging account
            st.success(f"🚩 Account {fraud['VPA_from']} has been flagged")
            st.info("Account has been flagged for enhanced monitoring and investigation.")
            # Store flag action in session state
            if 'flagged_accounts' not in st.session_state:
                st.session_state.flagged_accounts = []
            st.session_state.flagged_accounts.append({
                'account': fraud['VPA_from'],
                'action': 'Account Flagged',
                'timestamp': datetime.now(),
                'fraud_type': fraud['fraud_type'],
                'risk_score': fraud['risk_score'],
                'transaction_id': fraud.get('transaction_id', 'N/A')
            })
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Recent Actions Summary ---
    if any(key in st.session_state for key in ['alerts', 'blocked_transactions', 'saved_reports', 'flagged_accounts']):
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📋 Recent Actions Taken")
        
        # Show alerts
        if 'alerts' in st.session_state and st.session_state.alerts:
            st.markdown("**🚨 Recent Alerts:**")
            for alert in st.session_state.alerts[-3:]:  # Show last 3 alerts
                st.info(f"Alert sent for {alert['transaction_id']} ({alert['fraud_type']}) - {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        # Show blocked transactions
        if 'blocked_transactions' in st.session_state and st.session_state.blocked_transactions:
            st.markdown("**🚫 Blocked Transactions:**")
            for block in st.session_state.blocked_transactions[-3:]:  # Show last 3 blocks
                st.warning(f"Blocked {block['transaction_id']} (₹{block['amount']:,}) - {block['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        # Show saved reports
        if 'saved_reports' in st.session_state and st.session_state.saved_reports:
            st.markdown("**📄 Saved Reports:**")
            for report in st.session_state.saved_reports[-3:]:  # Show last 3 reports
                st.success(f"Report saved for {report['transaction_id']} ({report['fraud_name']}) - {report['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        # Show flagged accounts
        if 'flagged_accounts' in st.session_state and st.session_state.flagged_accounts:
            st.markdown("**🚩 Flagged Accounts:**")
            for flag in st.session_state.flagged_accounts[-3:]:  # Show last 3 flags
                st.error(f"Flagged account {flag['account']} ({flag['fraud_type']}) - {flag['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Focused Transaction Graph ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🕸️ Focused Transaction Graph")
    st.markdown(f"**Transaction:** {fraud.get('transaction_id', 'N/A')} | **Fraud Type:** {fraud.get('fraud_name', 'Unknown')}")
    
    # Generate focused graph data
    df = generate_cached_sample_data()
    
    # Find related transactions (same sender, receiver, or similar patterns)
    related_transactions = []
    
    # Add the main transaction
    main_tx = {
        'VPA_from': fraud['VPA_from'],
        'VPA_to': fraud['VPA_to'],
        'amount': fraud['amount'],
        'risk_score': fraud['risk_score'],
        'fraud_type': fraud['fraud_type'],
        'transaction_id': fraud.get('transaction_id', 'N/A'),
        'is_main': True
    }
    related_transactions.append(main_tx)
    
    # Find transactions with same sender or receiver
    same_sender = df[df['VPA_from'] == fraud['VPA_from']].head(5)
    same_receiver = df[df['VPA_to'] == fraud['VPA_to']].head(5)
    
    # Add same sender transactions
    for _, row in same_sender.iterrows():
        if row.get('transaction_id') != fraud.get('transaction_id'):
            related_transactions.append({
                'VPA_from': row['VPA_from'],
                'VPA_to': row['VPA_to'],
                'amount': row['amount'],
                'risk_score': row['risk_score'],
                'fraud_type': row['fraud_type'],
                'transaction_id': row.get('transaction_id', 'N/A'),
                'is_main': False,
                'connection_type': 'Same Sender'
            })
    
    # Add same receiver transactions
    for _, row in same_receiver.iterrows():
        if row.get('transaction_id') != fraud.get('transaction_id'):
            related_transactions.append({
                'VPA_from': row['VPA_from'],
                'VPA_to': row['VPA_to'],
                'amount': row['amount'],
                'risk_score': row['risk_score'],
                'fraud_type': row['fraud_type'],
                'transaction_id': row.get('transaction_id', 'N/A'),
                'is_main': False,
                'connection_type': 'Same Receiver'
            })
    
    # Find similar fraud type transactions
    similar_fraud = df[df['fraud_type'] == fraud['fraud_type']].head(3)
    for _, row in similar_fraud.iterrows():
        if row.get('transaction_id') != fraud.get('transaction_id'):
            related_transactions.append({
                'VPA_from': row['VPA_from'],
                'VPA_to': row['VPA_to'],
                'amount': row['amount'],
                'risk_score': row['risk_score'],
                'fraud_type': row['fraud_type'],
                'transaction_id': row.get('transaction_id', 'N/A'),
                'is_main': False,
                'connection_type': 'Similar Fraud Type'
            })
    
    # Define fraud types for reference
    fraud_types = {
        'star_fraud_center': {
            'name': 'Star Fraud',
            'description': 'Multiple accounts send money to one central account for money laundering',
            'icon': '🔴'
        },
        'cycle_fraud': {
            'name': 'Cycle Fraud', 
            'description': 'Money moves in circles between accounts to obscure origins',
            'icon': '🔄'
        },
        'high_value_fraud': {
            'name': 'High-Value Fraud',
            'description': 'Unusually large transactions that exceed normal patterns',
            'icon': '💰'
        },
        'account_takeover': {
            'name': 'Account Takeover',
            'description': 'Unauthorized access to legitimate accounts',
            'icon': '👥'
        },
        'money_laundering': {
            'name': 'Money Laundering',
            'description': 'Complex patterns to hide illegal money sources',
            'icon': '💸'
        },
        'social_engineering': {
            'name': 'Social Engineering',
            'description': 'Tricking users into making fraudulent payments',
            'icon': '🎭'
        },
        'upi_spoofing': {
            'name': 'UPI Spoofing',
            'description': 'Fake merchant transactions or QR code scams',
            'icon': '📱'
        },
        'normal': {
            'name': 'Normal Transaction',
            'description': 'Legitimate transaction with no suspicious patterns',
            'icon': '✅'
        }
    }
    
    # Create focused graph
    if len(related_transactions) > 1:
        focused_df = pd.DataFrame(related_transactions)
        
        # Create network graph for focused transactions
        G_focused = nx.Graph()
        
        # Add nodes
        all_vpas = set()
        for tx in related_transactions:
            all_vpas.add(tx['VPA_from'])
            all_vpas.add(tx['VPA_to'])
        
        G_focused.add_nodes_from(all_vpas)
        
        # Add edges
        for tx in related_transactions:
            G_focused.add_edge(tx['VPA_from'], tx['VPA_to'], 
                             weight=tx['amount'], 
                             risk=tx['risk_score'],
                             fraud_type=tx['fraud_type'],
                             transaction_id=tx['transaction_id'],
                             is_main=tx.get('is_main', False),
                             connection_type=tx.get('connection_type', 'Related'))
        
        # Graph controls
        col1, col2, col3 = st.columns(3)
        with col1:
            layout_focused = st.selectbox("Graph Layout", ["Spring", "Circular", "Shell"], key="focused_layout")
        with col2:
            show_labels_focused = st.checkbox("Show Labels", value=True, key="focused_labels")
        with col3:
            max_related = st.slider("Max Related Transactions", 5, 20, len(related_transactions), key="focused_max")
        
        # Display focused graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Related Nodes", G_focused.number_of_nodes())
        with col2:
            st.metric("Related Edges", G_focused.number_of_edges())
        with col3:
            suspicious_count = len([tx for tx in related_transactions if tx['fraud_type'] != 'normal'])
            st.metric("Suspicious Transactions", suspicious_count)
        with col4:
            avg_risk_focused = sum(tx['risk_score'] for tx in related_transactions) / len(related_transactions)
            st.metric("Avg Risk Score", f"{avg_risk_focused:.3f}")
        
        # Create focused interactive graph
        fig_focused = create_focused_interactive_graph(G_focused, layout_focused, related_transactions)
        st.plotly_chart(fig_focused, use_container_width=True, height=500)
        
        # Related transactions table
        st.markdown("### 🔍 Related Suspicious Transactions")
        
        # Filter to show only suspicious related transactions
        suspicious_related = [tx for tx in related_transactions if tx['fraud_type'] != 'normal' and not tx.get('is_main', False)]
        
        if suspicious_related:
            for i, tx in enumerate(suspicious_related):
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 1])
                with col1:
                    st.write(f"**{tx['VPA_from']}**")
                with col2:
                    st.write(f"**{tx['VPA_to']}**")
                with col3:
                    st.write(f"₹{tx['amount']:,}")
                with col4:
                    risk_color = "🔴" if tx['risk_score'] > 0.8 else "🟡" if tx['risk_score'] > 0.6 else "🟢"
                    st.write(f"{risk_color} {tx['risk_score']:.2f}")
                with col5:
                    st.write(f"{tx.get('connection_type', 'Related')}")
                with col6:
                    if st.button(f"🔍 Investigate", key=f"investigate_related_{i}"):
                        # Create a new fraud object for investigation
                        new_fraud = {
                            'transaction_id': tx['transaction_id'],
                            'VPA_from': tx['VPA_from'],
                            'VPA_to': tx['VPA_to'],
                            'amount': tx['amount'],
                            'risk_score': tx['risk_score'],
                            'confidence': random.uniform(0.6, 0.95),  # Generate confidence
                            'fraud_type': tx['fraud_type'],
                            'fraud_name': fraud_types.get(tx['fraud_type'], {}).get('name', tx['fraud_type']),
                            'fraud_description': fraud_types.get(tx['fraud_type'], {}).get('description', 'Suspicious transaction'),
                            'fraud_icon': fraud_types.get(tx['fraud_type'], {}).get('icon', '⚠️'),
                            'priority_score': tx['risk_score'] * 0.8,  # Calculate priority
                            'priority_level': 'High' if tx['risk_score'] > 0.7 else 'Medium' if tx['risk_score'] > 0.4 else 'Low',
                            'PSP': random.choice(['Google Pay', 'PhonePe', 'Paytm', 'BHIM']),
                            'transaction_type': random.choice(['P2P', 'P2M', 'QR']),
                            'timestamp': datetime.now() - timedelta(hours=random.randint(1, 24))
                        }
                        st.session_state.selected_fraud = new_fraud
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No other suspicious transactions found related to this investigation.")
        
        # Connection analysis
        st.markdown("### 🔗 Connection Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Connection Types:**")
            connection_counts = {}
            for tx in related_transactions:
                conn_type = tx.get('connection_type', 'Related')
                connection_counts[conn_type] = connection_counts.get(conn_type, 0) + 1
            
            for conn_type, count in connection_counts.items():
                st.write(f"- **{conn_type}**: {count} transactions")
        
        with col2:
            st.markdown("**🎯 Risk Analysis:**")
            high_risk_count = len([tx for tx in related_transactions if tx['risk_score'] > 0.8])
            medium_risk_count = len([tx for tx in related_transactions if 0.5 <= tx['risk_score'] <= 0.8])
            low_risk_count = len([tx for tx in related_transactions if tx['risk_score'] < 0.5])
            
            st.write(f"- **🔴 High Risk**: {high_risk_count} transactions")
            st.write(f"- **🟡 Medium Risk**: {medium_risk_count} transactions")
            st.write(f"- **🟢 Low Risk**: {low_risk_count} transactions")
    
    else:
        st.info("No related transactions found for this investigation.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_fraud_network_graph(df):
    """Create a network graph from transaction data - optimized for performance"""
    G = nx.Graph()
    
    # Add nodes more efficiently using set operations
    all_vpas = set(df['VPA_from'].unique()) | set(df['VPA_to'].unique())
    G.add_nodes_from(all_vpas)
    
    # Add edges more efficiently using pandas operations
    edges_data = df[['VPA_from', 'VPA_to', 'amount', 'risk_score', 'fraud_type']].values
    for vpa_from, vpa_to, amount, risk_score, fraud_type in edges_data:
        G.add_edge(vpa_from, vpa_to, 
                  weight=amount, 
                  risk=risk_score,
                  fraud_type=fraud_type)
    
    return G

def create_interactive_network_graph(G, layout_type, fraud_type):
    """Create an interactive network graph using Plotly - optimized for performance"""
    # Use faster layout algorithms for large graphs
    if G.number_of_nodes() > 50:
        # For large graphs, use faster layouts
        if layout_type == "Spring":
            pos = nx.spring_layout(G, k=1, iterations=20)  # Reduced iterations
        elif layout_type == "Circular":
            pos = nx.circular_layout(G)
        elif layout_type == "Shell":
            pos = nx.shell_layout(G)
        elif layout_type == "Random":
            pos = nx.random_layout(G)
        elif layout_type == "Kamada-Kawai":
            # Skip Kamada-Kawai for large graphs as it's O(n³)
            pos = nx.spring_layout(G, k=1, iterations=20)
        else:
            pos = nx.spring_layout(G, k=1, iterations=20)
    else:
        # For smaller graphs, use the requested layout
        if layout_type == "Spring":
            pos = nx.spring_layout(G, k=1, iterations=30)  # Reduced iterations
        elif layout_type == "Circular":
            pos = nx.circular_layout(G)
        elif layout_type == "Shell":
            pos = nx.shell_layout(G)
        elif layout_type == "Random":
            pos = nx.random_layout(G)
        elif layout_type == "Kamada-Kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=1, iterations=30)
    
    # Prepare edge data more efficiently
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace - use a single color for all edges to avoid the list issue
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#cccccc'),  # Use a single color
        hoverinfo='none',
        mode='lines')
    
    # Prepare node data more efficiently
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
        
        # Color nodes based on fraud involvement - simplified logic
        has_fraud = any(edge[2].get('fraud_type', 'normal') != 'normal' 
                       for edge in G.edges(node, data=True))
        node_colors.append('#ff4757' if has_fraud else '#2ed573')
        
        node_labels.append(node)
    
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

def create_focused_interactive_graph(G, layout_type, related_transactions):
    """Create a focused interactive network graph for specific transaction investigation"""
    # Use faster layout algorithms for focused graphs
    if layout_type == "Spring":
        pos = nx.spring_layout(G, k=2, iterations=30)
    elif layout_type == "Circular":
        pos = nx.circular_layout(G)
    elif layout_type == "Shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=30)
    
    # Create separate edge traces for different colors
    edge_traces = []
    
    # Group edges by type
    main_edges = []
    suspicious_edges = []
    normal_edges = []
    
    for edge in G.edges(data=True):
        edge_data = {
            'x': [pos[edge[0]][0], pos[edge[1]][0]],
            'y': [pos[edge[0]][1], pos[edge[1]][1]],
            'edge': edge
        }
        
        is_main = edge[2].get('is_main', False)
        fraud_type = edge[2].get('fraud_type', 'normal')
        
        if is_main:
            main_edges.append(edge_data)
        elif fraud_type != 'normal':
            suspicious_edges.append(edge_data)
        else:
            normal_edges.append(edge_data)
    
    # Create edge traces for each type
    if main_edges:
        main_x = []
        main_y = []
        for edge_data in main_edges:
            main_x.extend(edge_data['x'] + [None])
            main_y.extend(edge_data['y'] + [None])
        
        main_trace = go.Scatter(
            x=main_x, y=main_y,
            line=dict(width=3, color='#ff4757'),
            hoverinfo='none',
            mode='lines',
            name='Main Transaction',
            showlegend=True
        )
        edge_traces.append(main_trace)
    
    if suspicious_edges:
        suspicious_x = []
        suspicious_y = []
        for edge_data in suspicious_edges:
            suspicious_x.extend(edge_data['x'] + [None])
            suspicious_y.extend(edge_data['y'] + [None])
        
        suspicious_trace = go.Scatter(
            x=suspicious_x, y=suspicious_y,
            line=dict(width=2, color='#ffa502'),
            hoverinfo='none',
            mode='lines',
            name='Suspicious',
            showlegend=True
        )
        edge_traces.append(suspicious_trace)
    
    if normal_edges:
        normal_x = []
        normal_y = []
        for edge_data in normal_edges:
            normal_x.extend(edge_data['x'] + [None])
            normal_y.extend(edge_data['y'] + [None])
        
        normal_trace = go.Scatter(
            x=normal_x, y=normal_y,
            line=dict(width=1, color='#2ed573'),
            hoverinfo='none',
            mode='lines',
            name='Normal',
            showlegend=True
        )
        edge_traces.append(normal_trace)
    
    # Prepare node data
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_labels = []
    node_hover_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate node size based on degree and involvement in main transaction
        degree = G.degree(node)
        is_main_node = node in [related_transactions[0]['VPA_from'], related_transactions[0]['VPA_to']]
        
        if is_main_node:
            node_size = 25  # Larger for main transaction nodes
            node_color = '#ff4757'  # Red for main nodes
        elif degree > 5:
            node_size = 20  # Medium for high-degree nodes
            node_color = '#ffa502'  # Orange for suspicious
        else:
            node_size = 15  # Small for normal nodes
            node_color = '#2ed573'  # Green for normal
        
        node_sizes.append(node_size)
        node_colors.append(node_color)
        node_labels.append(str(node))
        
        # Create hover text
        hover_text = f"VPA: {node}<br>Degree: {degree}"
        if is_main_node:
            hover_text += "<br>🔴 Main Transaction Node"
        node_hover_text.append(hover_text)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="middle center",
        hovertext=node_hover_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        name='Nodes'
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Update layout
    fig.update_layout(
        title=f"Focused Transaction Graph - {len(G.nodes())} nodes, {len(G.edges())} edges",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )
    
    return fig

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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_cached_sample_data(n_transactions=1000):
    """Cached version of sample data generation"""
    return generate_sample_data(n_transactions)

def show_transaction_graph():
    """Interactive transaction graph visualization - optimized for performance"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🕸️ Transaction Graph</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Interactive Network Visualization of UPI Transactions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data with caching
    df = generate_cached_sample_data()
    
    # Graph controls
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔧 Graph Controls")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_nodes = st.slider("Max Nodes", 10, 200, 30)  # Reduced max from 300 to 200, default from 50 to 30
    with col2:
        layout_type = st.selectbox("Layout", ["Spring", "Circular", "Shell", "Random"])  # Removed Kamada-Kawai for performance
    with col3:
        show_labels = st.checkbox("Show Labels", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data more efficiently
    if len(df) > max_nodes:
        df = df.head(max_nodes)
    
    # Create network graph
    with st.spinner("Creating network graph..."):
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
    
    with st.spinner("Generating interactive graph..."):
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

def show_reports():
    """Comprehensive reports and actions page"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">📋 Reports & Actions Center</h1>
        <p style="color: #666; margin: 5px 0 0 0;">Comprehensive View of All System Actions & Generated Reports</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    with col3:
        if st.button("🔍 Fraud Analysis", use_container_width=True):
            st.session_state.current_page = 'fraud_analysis'
            st.rerun()
    
    # Summary metrics
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 System Actions Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alert_count = len(st.session_state.get('alerts', []))
        st.metric("🚨 Alerts Sent", alert_count)
    
    with col2:
        blocked_count = len(st.session_state.get('blocked_transactions', []))
        st.metric("🚫 Transactions Blocked", blocked_count)
    
    with col3:
        report_count = len(st.session_state.get('saved_reports', []))
        st.metric("📄 Reports Saved", report_count)
    
    with col4:
        flag_count = len(st.session_state.get('flagged_accounts', []))
        st.metric("🚩 Accounts Flagged", flag_count)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different report types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Alerts", "🚫 Blocked Transactions", "📄 Saved Reports", "🚩 Flagged Accounts", "📈 Analytics"
    ])
    
    with tab1:
        st.markdown("### 🚨 Alert History")
        if 'alerts' in st.session_state and st.session_state.alerts:
            for i, alert in enumerate(reversed(st.session_state.alerts)):
                with st.expander(f"Alert {i+1}: {alert['transaction_id']} - {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Transaction ID:** {alert['transaction_id']}")
                        st.write(f"**Fraud Type:** {alert['fraud_type']}")
                    with col2:
                        st.write(f"**Risk Score:** {alert['risk_score']:.3f}")
                        st.write(f"**Action:** {alert['action']}")
                    st.write(f"**Timestamp:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No alerts have been sent yet.")
    
    with tab2:
        st.markdown("### 🚫 Blocked Transactions")
        if 'blocked_transactions' in st.session_state and st.session_state.blocked_transactions:
            for i, block in enumerate(reversed(st.session_state.blocked_transactions)):
                with st.expander(f"Block {i+1}: {block['transaction_id']} - ₹{block['amount']:,}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Transaction ID:** {block['transaction_id']}")
                        st.write(f"**Amount:** ₹{block['amount']:,}")
                        st.write(f"**Fraud Type:** {block['fraud_type']}")
                    with col2:
                        st.write(f"**Sender:** {block['sender']}")
                        st.write(f"**Receiver:** {block['receiver']}")
                        st.write(f"**Action:** {block['action']}")
                    st.write(f"**Timestamp:** {block['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No transactions have been blocked yet.")
    
    with tab3:
        st.markdown("### 📄 Saved Reports")
        if 'saved_reports' in st.session_state and st.session_state.saved_reports:
            for i, report in enumerate(reversed(st.session_state.saved_reports)):
                with st.expander(f"Report {i+1}: {report['transaction_id']} - {report['fraud_name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Transaction ID:** {report['transaction_id']}")
                        st.write(f"**Fraud Type:** {report['fraud_name']}")
                        st.write(f"**Risk Score:** {report['risk_score']:.3f}")
                        st.write(f"**Confidence:** {report['confidence']:.3f}")
                    with col2:
                        st.write(f"**Amount:** ₹{report['amount']:,}")
                        st.write(f"**Sender:** {report['sender']}")
                        st.write(f"**Receiver:** {report['receiver']}")
                        st.write(f"**Priority:** {report['priority_level']}")
                    st.write(f"**Timestamp:** {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Show explanation
                    st.markdown("**📋 Detailed Analysis:**")
                    st.markdown(report['explanation'])
                    
                    # Download button for report
                    if st.button(f"📥 Download Report {i+1}", key=f"download_{i}"):
                        st.success(f"Report {report['transaction_id']} downloaded successfully!")
        else:
            st.info("No reports have been saved yet.")
    
    with tab4:
        st.markdown("### 🚩 Flagged Accounts")
        if 'flagged_accounts' in st.session_state and st.session_state.flagged_accounts:
            for i, flag in enumerate(reversed(st.session_state.flagged_accounts)):
                with st.expander(f"Flag {i+1}: {flag['account']} - {flag['fraud_type']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Account:** {flag['account']}")
                        st.write(f"**Fraud Type:** {flag['fraud_type']}")
                        st.write(f"**Risk Score:** {flag['risk_score']:.3f}")
                    with col2:
                        st.write(f"**Action:** {flag['action']}")
                        st.write(f"**Transaction ID:** {flag['transaction_id']}")
                        st.write(f"**Status:** 🔴 Flagged for Monitoring")
                    st.write(f"**Timestamp:** {flag['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No accounts have been flagged yet.")
    
    with tab5:
        st.markdown("### 📈 Action Analytics")
        
        # Generate analytics data
        if any(key in st.session_state for key in ['alerts', 'blocked_transactions', 'saved_reports', 'flagged_accounts']):
            col1, col2 = st.columns(2)
            
            with col1:
                # Action type distribution
                action_data = {
                    'Alerts': len(st.session_state.get('alerts', [])),
                    'Blocked': len(st.session_state.get('blocked_transactions', [])),
                    'Reports': len(st.session_state.get('saved_reports', [])),
                    'Flagged': len(st.session_state.get('flagged_accounts', []))
                }
                
                fig = px.pie(
                    values=list(action_data.values()),
                    names=list(action_data.keys()),
                    title="Action Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Timeline of actions
                all_actions = []
                if 'alerts' in st.session_state:
                    for alert in st.session_state.alerts:
                        all_actions.append({
                            'timestamp': alert['timestamp'],
                            'action': 'Alert',
                            'details': alert['transaction_id']
                        })
                
                if 'blocked_transactions' in st.session_state:
                    for block in st.session_state.blocked_transactions:
                        all_actions.append({
                            'timestamp': block['timestamp'],
                            'action': 'Block',
                            'details': f"₹{block['amount']:,}"
                        })
                
                if 'saved_reports' in st.session_state:
                    for report in st.session_state.saved_reports:
                        all_actions.append({
                            'timestamp': report['timestamp'],
                            'action': 'Report',
                            'details': report['fraud_name']
                        })
                
                if 'flagged_accounts' in st.session_state:
                    for flag in st.session_state.flagged_accounts:
                        all_actions.append({
                            'timestamp': flag['timestamp'],
                            'action': 'Flag',
                            'details': flag['account']
                        })
                
                if all_actions:
                    # Sort by timestamp
                    all_actions.sort(key=lambda x: x['timestamp'])
                    
                    # Create timeline chart
                    action_df = pd.DataFrame(all_actions)
                    action_df['date'] = action_df['timestamp'].dt.date
                    daily_actions = action_df.groupby(['date', 'action']).size().reset_index(name='count')
                    
                    fig = px.line(daily_actions, x='date', y='count', color='action',
                                title="📅 Daily Action Timeline",
                                labels={'count': 'Actions', 'date': 'Date'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No actions recorded yet.")
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'alerts' in st.session_state and st.session_state.alerts:
                    avg_risk = sum(alert['risk_score'] for alert in st.session_state.alerts) / len(st.session_state.alerts)
                    st.metric("Avg Alert Risk Score", f"{avg_risk:.3f}")
            
            with col2:
                if 'blocked_transactions' in st.session_state and st.session_state.blocked_transactions:
                    total_blocked_amount = sum(block['amount'] for block in st.session_state.blocked_transactions)
                    st.metric("Total Blocked Amount", f"₹{total_blocked_amount:,.0f}")
            
            with col3:
                if 'flagged_accounts' in st.session_state and st.session_state.flagged_accounts:
                    unique_accounts = len(set(flag['account'] for flag in st.session_state.flagged_accounts))
                    st.metric("Unique Flagged Accounts", unique_accounts)
        else:
            st.info("No actions have been taken yet. Start investigating transactions to see analytics here.")
    
    # Export functionality
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📤 Export & Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export All Reports", use_container_width=True):
            st.success("All reports exported successfully!")
            st.info("Reports have been saved to 'fortipay_reports.zip'")
    
    with col2:
        if st.button("📊 Export Analytics", use_container_width=True):
            st.success("Analytics data exported successfully!")
            st.info("Analytics have been saved to 'fortipay_analytics.csv'")
    
    with col3:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            # Clear all session state data
            for key in ['alerts', 'blocked_transactions', 'saved_reports', 'flagged_accounts']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("All data cleared successfully!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_fraud_analysis():
    """Comprehensive fraud analysis with GNN graphs - optimized for performance"""
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: #333; margin: 0;">🔍 Advanced Fraud Analysis</h1>
        <p style="color: #666; margin: 5px 0 0 0;">GNN-Based Pattern Detection & Network Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use cached data generation
    df = generate_cached_sample_data()
    
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
        max_nodes = st.slider("Max Graph Nodes", 10, 150, 30)  # Reduced max from 300 to 150
    
    col5, col6 = st.columns(2)
    with col5:
        layout_option = st.selectbox("Graph Layout", ["Spring", "Circular", "Shell", "Random"])  # Removed Kamada-Kawai
    with col6:
        min_amount = st.number_input("Min Amount (₹)", min_value=0, value=0, step=100)
    
    # Add search button to make filtering explicit
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_clicked = st.button("🔍 Search & Filter Transactions", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show current filter summary
    if search_clicked or 'show_results' in st.session_state:
        st.session_state.show_results = True
        
        # Display filter summary
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Active Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Fraud Type:** {fraud_type}")
        with col2:
            st.info(f"**Priority:** {priority_filter}")
        with col3:
            st.info(f"**Risk Threshold:** ≥{risk_threshold:.2f}")
        with col4:
            st.info(f"**Min Amount:** ₹{min_amount:,}")
        
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
        st.subheader(f"📋 Filtered Results ({len(filtered_df)} transactions found)")
        
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
            st.warning("No transactions match the current filter settings. Try adjusting your filters.")
        
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
    else:
        # Show instructions when no search has been performed
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.info("""
        **📋 Instructions:**
        1. Select your desired filters above
        2. Click the **"🔍 Search & Filter Transactions"** button to apply filters
        3. View filtered results and transaction details
        4. Click **"🔍 Investigate"** on any transaction for detailed analysis
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
    elif st.session_state.current_page == 'reports':
        show_reports()
    else:
        # Default to dashboard
        st.session_state.current_page = 'dashboard'
        show_dashboard()

if __name__ == "__main__":
    main() 