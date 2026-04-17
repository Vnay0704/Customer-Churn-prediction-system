Overview

This project is an end-to-end machine learning system that predicts customer churn using telecom data. It goes beyond model training by implementing a complete production-ready inference pipeline, exposed via a REST API and containerized using Docker.

Key Features

Built a churn prediction model using XGBoost
Performed EDA and feature engineering to identify churn patterns
Implemented one-hot encoding for categorical features
Designed a robust inference pipeline with feature alignment using joblib
Tuned classification threshold (0.4) to improve recall for churn class
Exposed predictions via FastAPI REST API
Containerized using Docker for portability and deployment


Tech Stack

Python
Pandas, NumPy
Scikit-learn, XGBoost
FastAPI
Docker
Uvicorn, Joblib


Project Structure

customer-churn-project/
│
├── api/                # FastAPI application
│   └── app.py
│
├── src/                # ML inference logic
│   └── predict.py
│
├── models/             # Saved model and features
│   ├── xgb_model.pkl
│   └── features.pkl
│
├── Dockerfile          # Container configuration
├── requirements.txt
└── output.ipynb        # EDA & model training


How It Works

User Input (JSON)
        ↓
FastAPI (/predict)
        ↓
Preprocessing (encoding + feature alignment)
        ↓
XGBoost Model
        ↓
Prediction + Probability


API Usage

Endpoint
POST /predict


Sample Request

{
  "Gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 80,
  "TotalCharges": 400,
  "HighRisk": 1
}


Sample Response

{
  "churn_probability": 0.1667,
  "prediction": 0
}


Docker Setup

Build Image
docker build -t churn-api .


Run Container
docker run -p 8000:8000 churn-api


Access API
http://localhost:8000/docs


Key Learnings
Importance of consistent preprocessing during inference
Handling feature mismatch using reindexing
Building end-to-end ML systems (not just models)
Deploying ML models using FastAPI + Docker


Future Improvements

Deploy on AWS EC2 or cloud platform
Add monitoring and logging
Build frontend UI
Implement CI/CD pipeline