import  pandas as pd  
df = pd.read_csv('car_cleaned_data.csv')
df.head()
df.columns
X=df[['Present_Price', 'Kms_Driven',
       'Car_Age', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
       'Fuel_Type_Petrol', 'Transmission_Automatic', 'Transmission_Manual']]
y=df['Selling_Price']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

import pickle
with open('lr_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("lr_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("🚘 Car Price Predictor")
st.markdown("### Predict the resale price of a car based on its features")

# Collect user inputs
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, format="%.2f")
kms_driven = st.number_input("Kms Driven", min_value=0, step=1000)
car_age = st.slider("Car Age (Years)", 0, 20, 5)
fuel_type = st.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG"))
transmission = st.selectbox("Transmission Type", ("Manual", "Automatic"))
seller_type = st.selectbox("Seller Type", ("Dealer", "Individual"))

# Encode categorical variables manually
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_cng = 1 if fuel_type == "CNG" else 0
transmission_manual = 1 if transmission == "Manual" else 0
transmission_auto = 1 if transmission == "Automatic" else 0

# Prepare the input data for the model
input_data = pd.DataFrame({
    "Present_Price": [present_price],
    "Kms_Driven": [kms_driven],
    "Car_Age": [car_age],
    "Fuel_Type_CNG": [fuel_type_cng],
    "Fuel_Type_Diesel": [fuel_type_diesel],
    "Fuel_Type_Petrol": [fuel_type_petrol],
    "Transmission_Automatic": [transmission_auto],
    "Transmission_Manual": [transmission_manual]
})

# Prediction
if st.button("🔮 Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {predicted_price * 100000:,.2f}")