# 🛡️ Credit Card Fraud Detection System

A real-time machine learning system for detecting fraudulent credit card transactions using advanced ensemble methods and explainable AI.

## 🎯 Project Overview

This project implements an end-to-end fraud detection pipeline that:
- Handles highly imbalanced datasets (0.17% fraud rate)
- Achieves 98%+ ROC-AUC score with XGBoost
- Provides real-time risk scoring via REST API
- Includes model explainability using SHAP values
- Features an interactive Streamlit dashboard

**Built for:** AI/ML Engineer position in Banking sector

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **ROC-AUC Score** | 0.9831 |
| **Precision(0)** | 100% |
| **Precision(1)** | 87% |
| **Recall(0)** | 100% |
| **Recall(1)** | 84% |
| **F1-Score** | 0.85 |
| **Cost Reduction** | 82.4% vs baseline |

---

## 🏗️ Architecture

```
Data Pipeline → Feature Engineering → SMOTE Balancing → 
XGBoost Model → Real-time API → Dashboard Visualization
```

### Key Components:
1. **Data Processing**: Handles 284K+ transactions with temporal features
2. **Model**: XGBoost with class weighting + SMOTE oversampling
3. **Explainability**: SHAP values for regulatory compliance
4. **Deployment**: Streamlit dashboard + Flask API

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 2GB+ RAM
- Kaggle dataset downloaded

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv fraud_env
source fraud_env/bin/activate  # Windows: fraud_env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/ folder
```

### Training the Model

```bash
# Option 1: Jupyter Notebook (recommended for exploration)
jupyter notebook notebooks/01_eda_and_modeling.ipynb

# Option 2: Python script (for reproducibility)
python src/train_model.py
```

### Running the Dashboard

```bash
streamlit run src/app.py
```

Open browser at `http://localhost:8501`

---

## 📁 Project Structure

```
fraud_detection_project/
├── data/
│   └── creditcard.csv              # Kaggle dataset
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # Exploration & experiments
├── src/
│   ├── train_model.py              # Model training pipeline
│   ├── app.py                      # Streamlit dashboard
│   └── utils.py                    # Helper functions
├── models/
│   ├── fraud_detection_model.pkl   # Trained XGBoost model
│   └── feature_names.pkl           # Feature metadata
├── requirements.txt
└── README.md