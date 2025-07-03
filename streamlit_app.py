import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction data below to check if it's fraudulent.")

# Input fields
def get_user_input():
    scaled_amount = st.number_input("scaled_amount")
    scaled_time = st.number_input("scaled_time")

    v_features = {}
    for i in range(1, 29):
        v_features[f"V{i}"] = st.number_input(f"V{i}")

    # Combine all into a single row DataFrame
    features = {
        "scaled_amount": scaled_amount,
        "scaled_time": scaled_time,
        **v_features
    }
    return pd.DataFrame([features])

# Get input
input_df = get_user_input()

# Predict button
if st.button("Predict Fraud"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Transaction Looks Legit (Confidence: {prob:.2f})")
