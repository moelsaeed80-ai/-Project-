import streamlit as st
import pandas as pd
import joblib

# =====================
# LOAD MODELS
# =====================
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Credit Risk App", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Classification App")
st.markdown("Predict Good or Bad Credit Risk")

st.divider()

# =====================
# INPUTS
# =====================
st.subheader("Customer Information")

# categorical (SAFE from encoders)
checking_status = st.selectbox("Checking Status", encoders["checking_status"].classes_)
credit_history = st.selectbox("Credit History", encoders["credit_history"].classes_)
purpose = st.selectbox("Purpose", encoders["purpose"].classes_)
savings_status = st.selectbox("Savings Status", encoders["savings_status"].classes_)
employment = st.selectbox("Employment", encoders["employment"].classes_)
personal_status = st.selectbox("Personal Status", encoders["personal_status"].classes_)
other_parties = st.selectbox("Other Parties", encoders["other_parties"].classes_)
property_magnitude = st.selectbox("Property Magnitude", encoders["property_magnitude"].classes_)
other_payment_plans = st.selectbox("Other Payment Plans", encoders["other_payment_plans"].classes_)
housing = st.selectbox("Housing", encoders["housing"].classes_)
job = st.selectbox("Job", encoders["job"].classes_)
own_telephone = st.selectbox("Own Telephone", encoders["own_telephone"].classes_)
foreign_worker = st.selectbox("Foreign Worker", encoders["foreign_worker"].classes_)

# numeric
duration = st.number_input("Duration")
credit_amount = st.number_input("Credit Amount")
installment_commitment = st.number_input("Installment Commitment")
residence_since = st.number_input("Residence Since")
age = st.number_input("Age")
existing_credits = st.number_input("Existing Credits")
num_dependents = st.number_input("Number of Dependents")

# =====================
# PREDICTION
# =====================
if st.button("🔍 Predict"):

    input_data = pd.DataFrame([[
        checking_status,
        duration,
        credit_history,
        purpose,
        credit_amount,
        savings_status,
        employment,
        installment_commitment,
        personal_status,
        other_parties,
        residence_since,
        property_magnitude,
        age,
        other_payment_plans,
        housing,
        existing_credits,
        job,
        num_dependents,
        own_telephone,
        foreign_worker
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

    # =====================
    # ENCODE
    # =====================
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # =====================
    # SCALE
    # =====================
    input_scaled = scaler.transform(input_data)

    # =====================
    # PREDICT
    # =====================
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # =====================
    # OUTPUT
    # =====================
    if pred == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("❌ Bad Credit Risk")

    st.metric("Confidence", f"{prob:.2%}")
    st.progress(float(prob))
