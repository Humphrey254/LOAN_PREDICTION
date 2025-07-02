import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('loan_approval_model.h5')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.markdown("Enter applicant loan details to check if the loan is likely to be **Approved** or **Rejected**.")

# Input form (matches your dataset)
loan_id = st.number_input("Loan ID", min_value=0, step=1)
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
education_encoded = 1 if education == 'Graduate' else 0

self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
self_employed_encoded = 1 if self_employed == 'Yes' else 0

income_annum = st.number_input("Annual Income (KES)", min_value=0, step=100000)
loan_amount = st.number_input("Loan Amount Requested (KES)", min_value=0, step=100000)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=1)
cibil_score = st.number_input("CIBIL Score (300–900)", min_value=300, max_value=900, step=1)

residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=100000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=100000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, step=100000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, step=100000)

# Predict button
if st.button("🔍 Predict Loan Approval"):
    input_data = np.array([[
        loan_id,
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    decision = "✅ Approved" if prediction >= 0.5 else "❌ Rejected"

    st.subheader("Prediction Result:")
    st.success(f"Loan Status: {decision}")
    st.markdown(f"**Confidence Score:** {prediction:.2f}")
