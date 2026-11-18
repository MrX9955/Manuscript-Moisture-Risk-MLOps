
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Manuscript Moisture Risk Prediction API",
    description="Predicts moisture risk in rare manuscript storage rooms",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "moisture_risk_model.pkl"
)
CONFIG_PATH = os.path.join(
    BASE_DIR, "..", "models", "model_config.json"
)

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

class StorageInput(BaseModel):
    Relative_Humidity: float = Field(ge=0, le=100)
    Room_Temperature: float = Field(ge=-10, le=60)
    Wall_Surface_Temperature: float = Field(ge=-10, le=60)
    Door_Openings_Per_Hour: float = Field(ge=0, le=100)
    Ventilation_Duration: float = Field(ge=0, le=60)
    Dehumidifier_Operating_Hours: float = Field(ge=0, le=24)
    Room_Occupancy: float = Field(ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: StorageInput):
    try:
        a = 17.27
        b = 237.7

        alpha = (
            (a * data.Room_Temperature) /
            (b + data.Room_Temperature)
        ) + np.log(data.Relative_Humidity / 100)

        dew_point = (b * alpha) / (a - alpha)

        condensation_risk_index = max(
            0, dew_point - data.Wall_Surface_Temperature
        )

        ventilation_effectiveness_ratio = min(
            1, data.Ventilation_Duration / 60
        )

        input_data = pd.DataFrame([{
            "Relative_Humidity": data.Relative_Humidity,
            "Room_Temperature": data.Room_Temperature,
            "Wall_Surface_Temperature": data.Wall_Surface_Temperature,
            "Door_Openings_Per_Hour": data.Door_Openings_Per_Hour,
            "Ventilation_Duration": data.Ventilation_Duration,
            "Dehumidifier_Operating_Hours": data.Dehumidifier_Operating_Hours,
            "Room_Occupancy": data.Room_Occupancy,
            "Dew_Point": dew_point,
            "condensation_risk_index": condensation_risk_index,
            "ventilation_effectiveness_ratio": ventilation_effectiveness_ratio
        }])

        input_data = input_data[config["feature_columns"]]
        prediction = int(model.predict(input_data)[0])
        result = config["target_classes"][prediction]

        return {
            "prediction": result,
            "message": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
