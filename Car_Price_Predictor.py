import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# 🎨 BRAND THEME (Lavania Style)
# ------------------------------

def local_css():
    css = """
    <style>
        body {
            background-color: #f9fafb;
            font-family: 'Segoe UI', sans-serif;
        }

        .title {
            font-weight: 800 !important;
            color: #064e3b;
            text-align: center;
            font-size: 48px !important;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #444;
            margin-bottom: 25px;
        }

        .stButton > button {
            background-color: #047857 !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px 20px !important;
            font-size: 18px !important;
            border: none;
        }

        .stButton > button:hover {
            background-color: #065f46 !important;
            color: #ffffff !important;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.07);
            margin-bottom: 30px;
        }

        .prediction-box {
            background: #ecfdf5;
            border-left: 6px solid #10b981;
            padding: 25px;
            border-radius: 12px;
            font-size: 22px;
            font-weight: 600;
            color: #064e3b;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

local_css()

# ------------------------------
# Load trained regression model
# ------------------------------
model_path = "lr_model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except:
    st.error("❌ Model file not found. Please upload lr_model.pkl to your repository.")
    st.stop()

# ------------------------------
# UI HEADER
# ------------------------------
st.markdown("<h1 class='title'>🚗 Used Car Price Predictor (South Africa)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Estimate the resale value of a car in Rands (R)</p>", unsafe_allow_html=True)
st.write("")

# ------------------------------
# Input Section
# ------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    present_price_rands = st.number_input("Present Price (R)", min_value=0, step=5000)
    kms_driven = st.number_input("Kms Driven", min_value=0, step=500)
    age = st.slider("Age of the Car (Years)", min_value=0, max_value=25, value=5)

    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Convert Rands → Lakhs
# ------------------------------
present_price_lakhs = present_price_rands / 100000

# ------------------------------
# Encode categorical inputs
# ------------------------------
fuel_map = {
    "Petrol": [1, 0, 0],
    "Diesel": [0, 1, 0],
    "CNG": [0, 0, 1],
}

trans_map = {
    "Manual": [1, 0],
    "Automatic": [0, 1],
}

seller_map = {
    "Dealer": [1, 0],
    "Individual": [0, 1],
}

fuel_petrol, fuel_diesel, fuel_cng = fuel_map[fuel_type]
trans_manual, trans_auto = trans_map[transmission]
seller_dealer, seller_individual = seller_map[seller_type]

# Input DataFrame
input_data = pd.DataFrame({
    "Present_Price": [present_price_lakhs],
    "Kms_Driven": [kms_driven],
    "Car_Age": [age],
    "Fuel_Type_CNG": [fuel_cng],
    "Fuel_Type_Diesel": [fuel_diesel],
    "Fuel_Type_Petrol": [fuel_petrol],
    "Transmission_Manual": [trans_manual],
    "Transmission_Automatic": [trans_auto],
    "Seller_Type_Dealer": [seller_dealer],
    "Seller_Type_Individual": [seller_individual]
})

# ------------------------------
# Predict Button
# ------------------------------
predict_btn = st.button("Predict Selling Price")

if predict_btn:
    try:
        # Ensure correct feature alignment
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            for col in expected_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_for_model = input_data[expected_cols]
        else:
            input_for_model = input_data

        # Predict in lakhs
        pred_lakhs = model.predict(input_for_model)[0]

        # SAFETY CORRECTION — Avoid negative car prices
        pred_lakhs = max(pred_lakhs, 0.05)  # minimum 0.05 lakhs (R 5,000)

        # Convert to Rands
        pred_rands = pred_lakhs * 100000
        pred_rands_fmt = f"R {pred_rands:,.2f}"

        st.markdown(
            f"""
            <div class='prediction-box'>
                💰 Estimated Selling Price: <strong>{pred_rands_fmt}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error making prediction: {e}")
