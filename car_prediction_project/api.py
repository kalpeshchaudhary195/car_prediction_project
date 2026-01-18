from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

@app.post("/predict")
def predict_price(data: dict):

    df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    df["Year"] = data["year"]
    df["Kms_Driven"] = data["kms"]
    df["Owner"] = data["owner"]

    df[f"Fuel_Type_{data['fuel']}"] = 1
    df[f"Seller_Type_{data['seller']}"] = 1
    df[f"Transmission_{data['transmission']}"] = 1
    df[f"Location_{data['location']}"] = 1

    scaled = scaler.transform(df)
    price = model.predict(scaled)[0]

    return {"Predicted Price (Lakhs)": round(price, 2)}
