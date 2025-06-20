# FortiPay: Graph-Based Real-Time UPI Fraud Detection

## Overview
FortiPay is a real-time fraud detection system for UPI transactions, leveraging Graph Neural Networks (GNNs) to identify suspicious patterns, relational anomalies, and behavioral outliers in payment flows. The system is designed for banks, UPI apps, and analysts, providing explainable alerts and a user-friendly dashboard.

## Features
- Graph-based modeling of UPI transactions (nodes: VPAs/accounts, edges: transactions)
- GNN-powered fraud detection (node/edge classification)
- Real-time risk scoring and explainable alerts
- Guardian Mode for vulnerable users
- Analyst dashboard for fraud tracing and visualization
- Secure API for integration with UPI apps

## Project Structure
```
FortiPay/
│
├── backend/         # API, security, GNN model
├── frontend/        # Dashboard code (Streamlit, D3.js, etc.)
├── data/            # Sample data, scripts
├── models/          # Trained models, GNN scripts
├── docs/            # Architecture, diagrams
└── README.md
```

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/pixelPanda123/FortiPay.git
   cd FortiPay
   ```
2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the backend API:**
   ```bash
   cd backend
   # Instructions in backend/README.md
   ```
4. **Run the frontend dashboard:**
   ```bash
   cd frontend
   streamlit run app.py
   ```

## Contribution Guidelines
- Use feature branches for new work (e.g., `feature/graph-builder-data-pipeline`)
- Submit pull requests to `main` with clear descriptions
- Write tests and documentation for new features
- Follow code style and security best practices

## Contacts
- Project Lead: [Your Name]
- GitHub: https://github.com/pixelPanda123/FortiPay

---
For detailed architecture and technical mapping, see `docs/`. 