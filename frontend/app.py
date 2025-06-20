import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Assuming our scripts are importable
from visualize_graph import create_user_centric_visualization

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

# --- Data Loading Functions ---
@st.cache_data
def load_processed_data():
    """Loads the processed nodes and edges from our pipeline."""
    nodes_path = 'data/processed_nodes_synthetic.csv'
    edges_path = 'data/processed_edges_synthetic.csv'
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        st.error("Processed data not found! Please run the data preparation pipeline first.")
        return None, None
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    return nodes_df, edges_df

# --- UI Pages ---
def show_dashboard():
    """Display the main summary dashboard using our real data."""
    st.header("Overall Fraud Landscape")
    
    nodes_df, edges_df = load_processed_data()
    if nodes_df is None or edges_df is None:
        return

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type':'domain'}, {'type':'xy'}]],
        subplot_titles=('User Activity Breakdown', 'Avg. Transaction Amount')
    )

    suspicious_counts = nodes_df['is_suspicious'].value_counts()
    pie_labels = ['Normal Users', 'Suspicious Users']
    pie_values = [suspicious_counts.get(0, 0), suspicious_counts.get(1, 0)]

    avg_amounts = edges_df.groupby('is_fraud')['Amount_INR'].mean()
    bar_labels = ['Normal Transactions', 'Fraudulent Transactions']
    bar_values = [avg_amounts.get(0, 0), avg_amounts.get(1, 0)]

    fig.add_trace(go.Pie(
        labels=pie_labels, values=pie_values, hole=.3,
        marker_colors=['#3498db', '#e74c3c'], textinfo='percent+label'
    ), 1, 1)

    fig.add_trace(go.Bar(
        x=bar_labels, y=bar_values,
        marker=dict(color=['#3498db', '#e74c3c']),
        text=[f'₹{v:.2f}' for v in bar_values], textposition='auto'
    ), 1, 2)
    
    fig.update_layout(
        title_text='<b>Fraud Detection Summary Dashboard</b>',
        title_x=0.5, showlegend=False, yaxis_title="Amount (INR)"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_fraud_investigation():
    """UI for the User 360° View, connected to our backend."""
    st.header("🕵️‍♂️ User 360° View")
    
    nodes_df, _ = load_processed_data()
    if nodes_df is None:
        return
        
    user_list = sorted(nodes_df['UPI_ID'].unique().tolist())
    
    focus_user = st.selectbox("Select a User UPI to Investigate:", user_list)
    time_window = st.slider("Select Time Window (Days):", 1, 90, 30)

    if st.button("Generate User Graph"):
        if focus_user:
            with st.spinner("Generating interactive graph..."):
                html_path = create_user_centric_visualization(
                    focus_user_upi=focus_user,
                    time_window_days=time_window
                )
                if html_path and os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    components.html(source_code, height=800, scrolling=True)
                else:
                    st.warning("No transaction data found for this user in the selected time window.")
        else:
            st.warning("Please select a user.")

def main():
    st.sidebar.title("FortiPay 🛡️")
    st.sidebar.success("Logged in as: Fraud Analyst")

    menu = {
        "📊 Dashboard": show_dashboard,
        "🕵️‍♂️ User 360° View": show_fraud_investigation,
    }
    choice = st.sidebar.radio("Navigation", list(menu.keys()))
    menu[choice]()

if __name__ == "__main__":
    main() 