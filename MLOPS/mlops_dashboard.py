import streamlit as st
import pandas as pd
import os
from joblib import load

st.title("Health Insurance MLOps Pipeline Dashboard")
st.write("Interactively run and view each stage of the pipeline.")

# --- Data Engineering ---
st.header("1. Data Engineering & Ingestion")
if os.path.exists('data_engineering/claims_features.csv'):
    df = pd.read_csv('data_engineering/claims_features.csv')
    st.dataframe(df.head())
else:
    st.warning("claims_features.csv not found. Run data engineering stage first.")

# --- Model Development ---
st.header("2. Model Development & Training")
if os.path.exists('fraud_model.joblib'):
    st.success("Fraud detection model trained and saved.")
    st.write("Model file: fraud_model.joblib")
else:
    st.warning("Model not found. Run model training stage first.")

# --- Model Validation ---
st.header("3. Model Validation")
if os.path.exists('data_engineering/claims_features.csv') and os.path.exists('fraud_model.joblib'):
    df = pd.read_csv('data_engineering/claims_features.csv')
    model = load('fraud_model.joblib')
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_cols]
    y = df['is_fraud']
    y_pred = model.predict(X)
    from sklearn.metrics import classification_report
    report = classification_report(y, y_pred, output_dict=True)
    st.json(report)
else:
    st.warning("Run previous stages to validate model.")

# --- Audit Logging ---
st.header("4. Audit Logging")
if os.path.exists('audit.log'):
    with open('audit.log') as f:
        logs = f.readlines()
    st.text_area("Audit Log", value="".join(logs), height=200)
else:
    st.warning("audit.log not found. Run pipeline to generate logs.")

st.info("To run the full pipeline, use the run_mlops_pipeline.py script. Refresh this dashboard to see updated results.")
