import streamlit as st
import pandas as pd
import joblib

# =====================
# Load saved artifacts
# =====================
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")
columns = joblib.load("columns.pkl")

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="Credit Risk App",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Classification App")
st.markdown("Predict whether a customer is **Good** or **Bad** credit risk")

st.divider()

# =====================
# INPUT SECTION
# =====================
st.subheader("Customer Information")

input_dict = {}

for col in columns:

    # categorical
    if col in encoders:
        input_dict[col] = st.selectbox(col, encoders[col].classes_)
    else:
        input_dict[col] = st.number_input(col)

st.divider()

# =====================
# PREDICTION
# =====================
if st.button("🔍 Predict Credit Risk"):

    # build dataframe in correct order
    input_data = pd.DataFrame([input_dict])[columns]

    # encode categorical
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # scale numeric
    input_scaled = scaler.transform(input_data)

    # predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # =====================
    # OUTPUT
    # =====================
    st.subheader("Result")

    if pred == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("❌ Bad Credit Risk")

    st.metric("Confidence (Good Risk Probability)", f"{prob:.2%}")

    st.progress(float(prob))
