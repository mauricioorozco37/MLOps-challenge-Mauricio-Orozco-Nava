from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="California Housing Predictor")

with open("models/best_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"message": "API de predicción de precios de viviendas"}

@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}

@app.post("/predict")
def predict(features: Features):
    X = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                   features.AveBedrms, features.Population, features.AveOccup,
                   features.Latitude, features.Longitude]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return {"prediction": round(pred, 4), "price_usd": f"${pred * 100000:,.0f}"}