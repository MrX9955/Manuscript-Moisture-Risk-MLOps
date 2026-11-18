
from fastapi.testclient import TestClient
import sys
sys.path.append("api")

from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_valid_prediction():
    payload = {
        "Relative_Humidity": 70,
        "Room_Temperature": 22,
        "Wall_Surface_Temperature": 19,
        "Door_Openings_Per_Hour": 5,
        "Ventilation_Duration": 15,
        "Dehumidifier_Operating_Hours": 2,
        "Room_Occupancy": 3
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "Moisture Risk",
        "Storage Conditions Acceptable"
    ]

def test_invalid_humidity():
    payload = {
        "Relative_Humidity": 120,
        "Room_Temperature": 22,
        "Wall_Surface_Temperature": 19,
        "Door_Openings_Per_Hour": 5,
        "Ventilation_Duration": 15,
        "Dehumidifier_Operating_Hours": 2,
        "Room_Occupancy": 3
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_temperature():
    payload = {
        "Relative_Humidity": 70,
        "Room_Temperature": 80,
        "Wall_Surface_Temperature": 19,
        "Door_Openings_Per_Hour": 5,
        "Ventilation_Duration": 15,
        "Dehumidifier_Operating_Hours": 2,
        "Room_Occupancy": 3
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
