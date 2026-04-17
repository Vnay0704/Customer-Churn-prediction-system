from fastapi import FastAPI
import sys
import os

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict_api(data: dict):
    return predict(data)