from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model
with open("fraud_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define the FastAPI app
app = FastAPI()

# Define request schema (ensure the features match model training data)
class Transaction(BaseModel):
    Unnamed_0: int
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

@app.get("/")
def home():
    return {"message": "Fraud Detection API is Running"}

@app.post("/predict/")
def predict(transaction: Transaction):
    # Convert input data into a DataFrame
    data_dict = transaction.dict()
    data_df = pd.DataFrame([data_dict])

    # Ensure correct feature order
    feature_order = ['Unnamed_0', 'merchant_name', 'category', 'transaction_amount', 
                     'gender', 'city', 'state', 'zip', 'client_latitude', 
                     'client_longitude', 'city_population', 'job', 'unix_time', 
                     'merchant_latitude', 'merchant_longitude', 'transaction_hour', 
                     'transaction_month', 'age', 'transaction_longitude_distance', 
                     'transaction_lattitude_distance', 'transaction_distance', 
                     'age_intervals']
    
    data_df = data_df[feature_order]

    # Make a prediction
    prediction = model.predict(data_df)[0]
    result = "Fraud" if prediction == 1 else "Not Fraud"
    
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
