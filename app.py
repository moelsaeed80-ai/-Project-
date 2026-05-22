import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Credit Risk App", page_icon="💳")

st.title("💳 Credit Risk Classification App")

st.subheader("Customer Information")

# numeric inputs فقط
duration = st.number_input("Duration")
credit_amount = st.number_input("Credit Amount")
installment_commitment = st.number_input("Installment Commitment")
residence_since = st.number_input("Residence Since")
age = st.number_input("Age")
existing_credits = st.number_input("Existing Credits")
num_dependents = st.number_input("Number of Dependents")

if st.button("Predict"):

    # 🔥 default values للـ categorical (حل مؤقت احترافي)
    input_data = pd.DataFrame([[
        "A14",          # checking_status
        duration,
        "A34",          # credit_history
        "A40",          # purpose
        credit_amount,
        "A65",          # savings_status
        "A75",          # employment
        installment_commitment,
        "A93",          # personal_status
        "A101",         # other_parties
        residence_since,
        "A121",         # property_magnitude
        age,
        "A143",         # other_payment_plans
        "A152",         # housing
        existing_credits,
        "A173",         # job
        num_dependents,
        "A192",         # own_telephone
        "A201"          # foreign_worker
    ]], columns=[
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker"
    ])

    # encoding
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # scaling
    input_scaled = scaler.transform(input_data)

    # prediction
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.success("Good Credit Risk 👍")
    else:
        st.error("Bad Credit Risk ⚠️")

    st.metric("Confidence", f"{prob:.2%}")
