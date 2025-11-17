import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the details below to estimate the resale price of a car.")

# --------------------------------------------------------------------
# 1. AUTO-FIND THE MODEL FILE ANYWHERE IN STREAMLIT CLOUD CONTAINER
# --------------------------------------------------------------------
def find_model_file(filename="lr_model.pkl"):
    for root, dirs, files in os.walk(os.getcwd()):
        if filename in files:
            return os.path.join(root, filename)
    return None

model_path = find_model_file()

if model_path is None:
    st.error("❌ Model file 'lr_model.pkl' not found in the app environment.")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

# --------------------------------------------------------------------
# 2. USER INPUTS
# --------------------------------------------------------------------
present_price = st.number_input(
    "Present Price (₹ Lakhs)", 
    min_value=0.0, 
    max_value=100.0, 
    format="%.2f"
)

kms_driven = st.number_input(
    "Kms Driven", 
    min_value=0, 
    max_value=300000, 
    step=500
)

car_age = st.slider("Age of the Car (Years)", 0, 25, 5)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

# --------------------------------------------------------------------
# 3. MANUAL ONE-HOT ENCODING (MATCH YOUR TRAINING DATA)
# --------------------------------------------------------------------
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_cng = 1 if fuel_type == "CNG" else 0

transmission_manual = 1 if transmission == "Manual" else 0
transmission_auto = 1 if transmission == "Automatic" else 0

seller_type_individual = 1 if seller_type == "Individual" else 0

# --------------------------------------------------------------------
# 4. CREATE INPUT DATAFRAME
#    Must match EXACT columns used during training
# --------------------------------------------------------------------
input_data = pd.DataFrame({
    "Present_Price": [present_price],
    "Kms_Driven": [kms_driven],
    "Car_Age": [car_age],
    "Fuel_Type_CNG": [fuel_type_cng],
    "Fuel_Type_Diesel": [fuel_type_diesel],
    "Fuel_Type_Petrol": [fuel_type_petrol],
    "Transmission_Automatic": [transmission_auto],
    "Transmission_Manual": [transmission_manual],
    "Seller_Type_Individual": [seller_type_individual]
})

# --------------------------------------------------------------------
# 5. RUN PREDICTION
# --------------------------------------------------------------------
if st.button("Predict Selling Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction error: {e}")
