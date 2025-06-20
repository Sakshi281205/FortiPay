import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd

def metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{value}</h2>
        {f'<p style="color: {"green" if delta_color == "normal" else "red"}">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def alert_card(alert_data, alert_type="info"):
    """Create a styled alert card"""
    color_map = {
        "high": "#f44336",
        "medium": "#ff9800", 
        "low": "#4caf50",
        "info": "#2196f3"
    }
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color_map.get(alert_type, color_map['info'])};">
        <h4>{alert_data.get('title', 'Alert')}</h4>
        <p>{alert_data.get('description', '')}</p>
        <p><strong>Risk Score:</strong> {alert_data.get('risk_score', 'N/A')}</p>
        <p><strong>Time:</strong> {alert_data.get('timestamp', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

def create_network_graph(G, title="Transaction Network"):
    """Create an interactive network graph visualization"""
    pos = nx.spring_layout(G)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.extend([f"Amount: ₹{edge[2].get('amount', 'N/A')}<br>PSP: {edge[2].get('PSP', 'N/A')}"] * 3)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        hovertext=edge_text,
        mode='lines')

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        risk_score = G.nodes[node].get('risk_score', 0)
        node_text.append(f"VPA: {node}<br>Risk: {risk_score:.2f}")
        node_colors.append(risk_score)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='RdYlBu_r',
            size=15,
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

def create_time_series_chart(df, x_col, y_col, title="Time Series"):
    """Create a time series chart"""
    fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(xaxis_title="Time", yaxis_title=y_col)
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, title="Scatter Plot"):
    """Create a scatter plot"""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
    return fig

def create_pie_chart(df, values_col, names_col, title="Distribution"):
    """Create a pie chart"""
    fig = px.pie(df, values=values_col, names=names_col, title=title)
    return fig

def create_histogram(df, x_col, title="Distribution"):
    """Create a histogram"""
    fig = px.histogram(df, x=x_col, title=title)
    return fig

def filter_sidebar(df):
    """Create a filter sidebar for data filtering"""
    st.sidebar.subheader("Filters")
    
    # PSP Filter
    psp_options = ['All'] + list(df['PSP'].unique()) if 'PSP' in df.columns else ['All']
    selected_psp = st.sidebar.selectbox("Payment Service Provider", psp_options)
    
    # Risk Score Filter
    if 'risk_score' in df.columns:
        risk_range = st.sidebar.slider("Risk Score Range", 
                                     float(df['risk_score'].min()), 
                                     float(df['risk_score'].max()), 
                                     (float(df['risk_score'].min()), float(df['risk_score'].max())))
    
    # Date Filter
    if 'timestamp' in df.columns:
        date_range = st.sidebar.date_input("Date Range", 
                                          value=(df['timestamp'].min().date(), df['timestamp'].max().date()))
    
    # Amount Filter
    if 'amount' in df.columns:
        amount_range = st.sidebar.slider("Amount Range (₹)", 
                                       int(df['amount'].min()), 
                                       int(df['amount'].max()), 
                                       (int(df['amount'].min()), int(df['amount'].max())))
    
    return {
        'psp': selected_psp,
        'risk_range': risk_range if 'risk_score' in df.columns else None,
        'date_range': date_range if 'timestamp' in df.columns else None,
        'amount_range': amount_range if 'amount' in df.columns else None
    }

def apply_filters(df, filters):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    if filters['psp'] and filters['psp'] != 'All':
        filtered_df = filtered_df[filtered_df['PSP'] == filters['psp']]
    
    if filters['risk_range']:
        filtered_df = filtered_df[
            (filtered_df['risk_score'] >= filters['risk_range'][0]) &
            (filtered_df['risk_score'] <= filters['risk_range'][1])
        ]
    
    if filters['date_range']:
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= filters['date_range'][0]) &
            (filtered_df['timestamp'].dt.date <= filters['date_range'][1])
        ]
    
    if filters['amount_range']:
        filtered_df = filtered_df[
            (filtered_df['amount'] >= filters['amount_range'][0]) &
            (filtered_df['amount'] <= filters['amount_range'][1])
        ]
    
    return filtered_df 