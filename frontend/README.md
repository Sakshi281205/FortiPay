# FortiPay Frontend Dashboard

A comprehensive web-based dashboard for UPI fraud detection and analysis, built with Streamlit and Plotly.

## Features

### 🛡️ Fraud Detection Dashboard
- **Real-time Transaction Monitoring**: Live tracking of UPI transactions with risk scoring
- **Fraud Pattern Recognition**: Detection of star-shaped, cycle-based, and high-value fraud patterns
- **Interactive Network Visualization**: Graph-based visualization of transaction networks
- **Confidence Scoring**: Model confidence metrics for each fraud detection

### 📊 Advanced Analytics
- **Confusion Matrix**: Model performance visualization with precision, recall, and F1-score
- **Time Series Analysis**: Fraud trends over time with customizable date ranges
- **Risk Distribution**: Histograms and pie charts showing fraud pattern distribution
- **Alert Management**: Categorized alerts by severity (Critical, High, Medium, Low)

### 🔍 Investigation Tools
- **Detailed Fraud Analysis**: Step-by-step explanation of why transactions are flagged
- **Network Graph Analysis**: Interactive visualization of fraud networks
- **Evidence Presentation**: Clear evidence and reasoning for fraud detection
- **Action Recommendations**: Suggested actions for each fraud type

### 🎯 Guardian Mode
- **Enhanced Security**: Stricter fraud detection thresholds
- **Immediate Alerts**: Real-time notifications for critical fraud attempts
- **Auto-blocking**: Automatic blocking of high-risk transactions
- **User Confirmation**: Required confirmation for suspicious transactions

## Installation

1. **Install Dependencies**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

2. **Start the Backend Server** (if not already running):
   ```bash
   cd ../backend
   python server.py
   ```

3. **Run the Dashboard**:
   ```bash
   cd frontend
   streamlit run app.py
   ```

## Usage

### Dashboard Navigation

1. **Dashboard**: Overview with key metrics and critical alerts
2. **Fraud Analysis**: Detailed fraud pattern analysis with filters
3. **Model Performance**: Confusion matrix and performance metrics
4. **Fraud Investigation**: Detailed investigation of selected fraud cases

### Key Features

#### Alert System
- **Critical Alerts**: Risk score > 0.9, Confidence > 0.9
- **High Alerts**: Risk score > 0.8, Confidence > 0.8
- **Medium Alerts**: Risk score > 0.6, Confidence > 0.7
- **Low Alerts**: Risk score > 0.4, Confidence > 0.6

#### Fraud Patterns Detected
1. **Star-Shaped Fraud**: One account receiving from multiple sources
2. **Cycle Fraud**: Circular transaction patterns (A→B→C→A)
3. **High-Value Fraud**: Unusually large transaction amounts

#### Interactive Features
- **Network Graphs**: Click on nodes to see transaction details
- **Time Filters**: Filter data by date ranges
- **Risk Thresholds**: Adjustable risk and confidence thresholds
- **Real-time Updates**: Live data updates (when connected to backend)

## Configuration

### API Configuration
- Default API URL: `http://localhost:8000`
- Can be modified in the Settings page

### Guardian Mode Settings
- **Risk Threshold**: Minimum risk score for blocking (default: 0.5)
- **Confirmation Required**: Require user confirmation for all transactions
- **Auto-block**: Automatically block high-risk transactions
- **Notification Frequency**: Choose alert frequency

## Data Visualization

### Charts and Graphs
1. **Pie Charts**: Fraud type distribution
2. **Histograms**: Risk score distribution
3. **Scatter Plots**: Risk vs Amount analysis
4. **Time Series**: Daily fraud trends
5. **Network Graphs**: Transaction relationship visualization
6. **Confusion Matrix**: Model performance heatmap

### Color Coding
- **Red**: Critical/High risk
- **Orange**: Medium risk
- **Yellow**: Low risk
- **Green**: Normal transactions

## Troubleshooting

### Common Issues

1. **Backend Connection Error**:
   - Ensure backend server is running on port 8000
   - Check API URL in settings

2. **Import Errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Visualization Issues**:
   - Clear browser cache
   - Check if Plotly is properly installed

### Performance Tips
- Use filters to reduce data load
- Limit network graph size for better performance
- Refresh dashboard periodically for real-time updates

## Security Features

- **JWT Authentication**: Secure login system
- **Session Management**: Automatic logout on inactivity
- **Data Encryption**: Secure transmission of sensitive data
- **Access Control**: Role-based access to different features

## Contributing

1. Follow the existing code structure
2. Add proper documentation for new features
3. Test thoroughly before submitting
4. Update requirements.txt for new dependencies

## Support

For technical support or feature requests, please contact the development team or create an issue in the repository. 