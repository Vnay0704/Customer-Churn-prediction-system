import joblib
import pandas as pd

model = joblib.load("models/xgb_model.pkl")
features = joblib.load("models/features.pkl")

THRESHOLD = 0.4

def predict(input_dict):   # ✅ EXACT NAME
    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)

    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[:,1][0]

    prediction = int(prob > THRESHOLD)

    return {
        "churn_probability": float(prob),
        "prediction": prediction
    }