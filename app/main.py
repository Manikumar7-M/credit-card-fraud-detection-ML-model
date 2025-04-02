import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
with open("fraud_model_xgb.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the feature order (Ensure this matches the order during training)
FEATURE_ORDER = [
    "merchant_name", "category", "transaction_amount", "gender", "city", "state", "zip",
    "client_latitude", "client_longitude", "city_population", "job", "unix_time",
    "merchant_latitude", "merchant_longitude", "transaction_hour", "transaction_month",
    "age", "transaction_longitude_distance", "transaction_lattitude_distance",
    "transaction_distance", "age_intervals"
]

# Define request body format
class Transaction(BaseModel):
    merchant_name: int
    category: int
    transaction_amount: float
    gender: int
    city: int
    state: int
    zip: int
    client_latitude: float
    client_longitude: float
    city_population: int
    job: int
    unix_time: int
    merchant_latitude: float
    merchant_longitude: float
    transaction_hour: int
    transaction_month: int
    age: int
    transaction_longitude_distance: float
    transaction_lattitude_distance: float
    transaction_distance: float
    age_intervals: int

# API Endpoint: Health Check
@app.get("/")
def health_check():
    return {"status": "API is running"}

# API Endpoint: Predict Fraud
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Convert input data to DataFrame
    transaction_data = pd.DataFrame([transaction.dict()])

    # Ensure correct feature order
    transaction_data = transaction_data[FEATURE_ORDER]

    # Debug: Check feature counts
    print(f"Model trained with {scaler.n_features_in_} features")
    print(f"Received transaction with {transaction_data.shape[1]} features")

    # Scale the transaction
    transaction_scaled = scaler.transform(transaction_data)

    # Predict fraud
    prediction = model.predict(transaction_scaled)
    fraud_probability = model.predict_proba(transaction_scaled)[:, 1][0]

    # Return response
    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "prediction": "FRAUD" if prediction[0] == 1 else "NOT FRAUD"
    }

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
