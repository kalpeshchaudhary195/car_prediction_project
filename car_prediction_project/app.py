import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime


model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


df = pd.read_csv("car data.csv")
st.title("🚗 Used Car Price Prediction App")


car_name = st.selectbox("Select Car", df["Car_Name"].unique())
year = st.number_input("Manufacturing Year", 2000, 2024)
kms = st.number_input("Kilometers Driven", 500, 300000)
owner = st.selectbox("Number of Owners", [0, 1, 2, 3])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Petrol/CNG"])  # Added Petrol/CNG
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
location = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Pune", "Ahmedabad"])
condition_score = st.slider("Car Condition Score", 0.5, 1.2, 1.0)


if st.button("Predict Price"):

   
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    
    if "Year" in input_df.columns:
        input_df["Year"] = year
    if "Kms_Driven" in input_df.columns:
        input_df["Kms_Driven"] = kms
    if "Owner" in input_df.columns:
        input_df["Owner"] = owner

    
    if fuel == "Petrol/CNG":
        for f in ["Petrol", "CNG"]:
            col = f"Fuel_Type_{f}"
            if col in input_df.columns:
                input_df[col] = 1
    else:
        fuel_col = f"Fuel_Type_{fuel}"
        if fuel_col in input_df.columns:
            input_df[fuel_col] = 1

    
    seller_col = f"Seller_Type_{seller}"
    if seller_col in input_df.columns:
        input_df[seller_col] = 1

    trans_col = f"Transmission_{transmission}"
    if trans_col in input_df.columns:
        input_df[trans_col] = 1

    loc_col = f"Location_{location}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1

    car_col = f"Car_Name_{car_name}"
    if car_col in input_df.columns:
        input_df[car_col] = 1

    
    scaled = scaler.transform(input_df)
    price = model.predict(scaled)[0]

    
    current_year = datetime.now().year
    car_age = current_year - year

    if car_age <= 1:
        age_multiplier = 1.2   
    elif car_age <= 3:
        age_multiplier = 1.1   
    elif car_age <= 5:
        age_multiplier = 1.0   
    elif car_age <= 10:
        age_multiplier = 0.9   
    else:
        age_multiplier = 0.8  

    final_price = price * condition_score * age_multiplier

    st.success(f"Estimated Price: ₹ {final_price:.3f} Lakhs")
