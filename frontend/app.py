import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Assuming our scripts are importable
from visualize_graph import create_user_centric_visualization

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
                    st.components.v1.html(source_code, height=800, scrolling=True)
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