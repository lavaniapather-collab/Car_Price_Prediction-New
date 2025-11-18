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

# --------------------------------
# Load trained regression model
# --------------------------------
model_path = "lr_model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except:
    st.error("❌ Model file not found. Please upload lr_model.pkl to your repository.")
    st.stop()

# --------------------------------
# UI HEADER
# --------------------------------
st.markdown("<h1 class='title'>🚗 Used Car Price Predictor (South Africa)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Estimate the resale value of a car in Rands (R)</p>", unsafe_allow_html=True)
st.write("")

# --------------------------------
# INPUT SECTION (Card layout)
# --------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Updated for Rands
    present_price_rands = st.number_input("Present Price (R)", min_value=0, step=5000)
    kms_driven = st.number_input("Kms Driven", min_value=0, step=500)
    age = st.slider("Age of the Car (Years)", min_value=0, max_value=25, value=5)

    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# Convert Rands → Lakhs (model expects lakhs)
# --------------------------------
# 1 Lakh = 100,000 Rands
present_price_lakhs = present_price_rands / 100000  

# --------------------------------
# Encode categorical features
# --------------------------------
fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_cng = 1 if fuel_type == "CNG" else 0

trans_manual = 1 if transmission == "Manual" else 0
trans_auto = 1 if transmission == "Automatic" else 0

seller_dealer = 1 if seller_type == "Dealer" else 0
seller_individual = 1 if seller_type == "Individual" else 0

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

# --------------------------------
# PREDICTION BUTTON
# --------------------------------
predict_btn = st.button("Predict Selling Price")

if predict_btn:
    try:
        # 🔍 Align features with what the model was trained on
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            
            # Add any missing columns with default 0
            for col in expected_cols:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Keep only the expected columns, in the correct order
            input_for_model = input_data[expected_cols]
        else:
            # Fallback if model doesn't store feature names
            input_for_model = input_data

        # Make prediction in lakhs
        pred_lakhs = model.predict(input_for_model)[0]

        # Convert lakhs → Rands
        pred_rands = pred_lakhs * 100000
        pred_rands_fmt = f"R {pred_rands:,.2f}"  # Format with commas

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