
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on their profile and service usage.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

# Prepare input as DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet],
    "OnlineSecurity": [online_security],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "MonthlyCharges": [monthly_charges],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
})

# Encode categorical columns
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    result = target_encoder.inverse_transform(prediction)[0]

    if result == "Yes":
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is NOT likely to churn.")
