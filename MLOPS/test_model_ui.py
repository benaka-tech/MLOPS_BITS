import streamlit as st
import pandas as pd
from joblib import load
import os

st.title("Test Health Insurance Fraud Detection Model")

if not os.path.exists('fraud_model.joblib'):
    st.error("Model file 'fraud_model.joblib' not found. Please train the model first.")
    st.stop()

model = load('fraud_model.joblib')

st.header("Enter Claim Details")
claim_amount = st.number_input("Claim Amount", min_value=0.0, value=1000.0)
num_claims = st.number_input("Number of Claims", min_value=1, value=1)
total_claim_amount = claim_amount * num_claims

input_dict = {
    'claim_amount': claim_amount,
    'num_claims': num_claims,
    'total_claim_amount': total_claim_amount
}

input_df = pd.DataFrame([input_dict])

if st.button("Test Model"):
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")