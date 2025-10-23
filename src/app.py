'''
Fraud Detection Dashboard
Run with: streamlit run src/app.py
'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_detection_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

# Title
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("---")

try:
    model, feature_names = load_model()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Single Transaction", "Batch Predictions", "Model Info"])
    
    # ============ PAGE 1: Single Transaction ============
    if page == "Single Transaction":
        st.header("🔍 Single Transaction Risk Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
        
        with col2:
            v14 = st.number_input("V14 (PCA Feature)", value=0.0)
            v10 = st.number_input("V10 (PCA Feature)", value=0.0)
        
        with col3:
            v4 = st.number_input("V4 (PCA Feature)", value=0.0)
            v12 = st.number_input("V12 (PCA Feature)", value=0.0)
            
        with col4:
            v20 = st.number_input("V20 (PCA Feature)", value=0.0)
            v19 = st.number_input("V19 (PCA Feature)", value=0.0)
        
        st.info("The V1-V28 values come from PCA dimensionality reduction applied to safeguard user identities and sensitive information.")
        st.info("Dataset Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)")

        
        if st.button("Analyze Transaction", type="primary"):

            features = {
                'Amount': amount,
                'Hour': hour,
                'Amount_log': np.log1p(amount),
                'Day': 0,  # Default
            }
            
            # Add V1-V28 features (simplified - use actual values)
            for i in range(1, 29):
                features[f'V{i}'] = locals().get(f'v{i}', 0.0)
            
            # Create DataFrame
            input_df = pd.DataFrame([features])
            
            # Ensure column order matches training
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            
            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE")
            
            with col2:
                st.metric("Fraud Probability", f"{probability*100:.2f}%")
            
            with col3:
                if probability < 0.3:
                    risk = "🟢 LOW"
                elif probability < 0.7:
                    risk = "🟡 MEDIUM"
                else:
                    risk = "🔴 HIGH"
                st.metric("Risk Level", risk)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Risk Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)
    
    # ============ PAGE 2: Batch Predictions ============
    elif page == "Batch Predictions":
        st.header("📊 Batch Transaction Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} transactions")
            st.dataframe(df.head())
            
            if st.button("Run Fraud Detection"):
                with st.spinner("Analyzing transactions..."):
                    # Prepare features (simplified - add proper feature engineering)
                    X = df.drop(['Class'], axis=1, errors='ignore')
                    X = X.reindex(columns=feature_names, fill_value=0)
                    
                    # Predict
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]
                    
                    # Add results
                    df['Prediction'] = predictions
                    df['Fraud_Probability'] = probabilities
                    df['Risk_Level'] = pd.cut(
                        probabilities, 
                        bins=[0, 0.3, 0.7, 1.0], 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Transactions", len(df))
                    col2.metric("Flagged as Fraud", predictions.sum())
                    col3.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")
                    col4.metric("Avg Risk Score", f"{probabilities.mean()*100:.2f}%")
                    
                    # Visualizations
                    fig = px.histogram(
                        df, 
                        x='Fraud_Probability', 
                        color='Prediction',
                        title="Fraud Probability Distribution",
                        labels={'Prediction': 'Is Fraud'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show flagged transactions
                    st.subheader("🚨 High-Risk Transactions")
                    high_risk = df[df['Risk_Level'] == 'High'].sort_values('Fraud_Probability', ascending=False)
                    st.dataframe(high_risk)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "fraud_detection_results.csv",
                        "text/csv"
                    )
    
    # ============ PAGE 3: Model Info ============
    else:
        st.header("📈 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model Type:** {type(model).__name__}")
            st.write(f"**Number of Features:** {len(feature_names)}")
            st.write(f"**Training Date:** 2024")
            
            st.subheader("Performance Metrics")
            st.write("**ROC-AUC Score:** 0.98")
            st.write("**Precision:** 0.89")
            st.write("**Recall:** 0.82")
            st.write("**F1-Score:** 0.85")
        
        with col2:
            st.subheader("Key Features")
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 10 Most Important Features'
                )
                st.plotly_chart(fig)
        
        st.subheader("Business Impact")
        st.info("""
        **Cost-Benefit Analysis:**
        - False Positive Cost: $10 (customer friction)
        - False Negative Cost: $100 (fraud loss)
        - Total Estimated Savings: $XX,XXX per month
        """)

except FileNotFoundError:
    st.error("❌ Model not found!")
    st.info("Make sure you're running from the project root directory")