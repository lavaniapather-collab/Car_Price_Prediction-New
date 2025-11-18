import streamlit as st
import numpy as np
import pickle

# Load trained linear regression model
model = pickle.load(open('lr_model.pkl', 'rb'))
# Streamlit UI
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
 
st.title("🚗 Used Car Price Prediction")
st.markdown("Enter the details of the car below to predict its price.")
 
# Input fields
kms = st.number_input("Kilometers Driven", min_value=0, max_value=300000, step=1000)
age = st.number_input("Age of the Car (in years)", min_value=0, max_value=25, step=1)
oprice = st.number_input("Original Price (Rs.)", min_value=500000, max_value=5000000, step=10000)
 
fuel_type = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol","Other"])
transmission = st.radio("Transmission", ["Manual", "Automatic"])
 
#condition
if fuel_type == 'Petrol':
    fuel = [0.0, 0.0, 1.0]
elif fuel_type == 'Diesel':
    fuel = [0.0, 1.0, 0.0]
elif fuel_type == 'CNG':
    fuel = [1.0, 0.0, 0.0]
else:
    fuel = [0.0, 0.0, 0.0]  # in case of "Other"
 
if transmission == 'Automatic':
    transmission_vals = [1.0, 0.0]
else:
    transmission_vals = [0.0, 1.0]

# Prediction
if st.button("🔮 Predict Price"):
    data = np.array([[oprice, kms, age, 
                  fuel[0], fuel[1], fuel[2], 
                  transmission_vals[0], transmission_vals[1]]])

    result = np.round(model.predict(data))
    st.success(f"Predicted Car Price: ₹ {result[0]:,.0f}")
 