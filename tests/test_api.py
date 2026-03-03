import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Mock the model loading before importing the app
mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

mock_feature_cols = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_3h", "volt_std_3h", "rotate_mean_3h", "rotate_std_3h",
    "pressure_mean_3h", "pressure_std_3h", "vibration_mean_3h", "vibration_std_3h",
    "volt_mean_24h", "volt_std_24h", "rotate_mean_24h", "rotate_std_24h",
    "pressure_mean_24h", "pressure_std_24h", "vibration_mean_24h", "vibration_std_24h",
    "error_error1", "error_error2", "error_error3", "error_error4", "error_error5",
    "days_since_comp1", "days_since_comp2", "days_since_comp3", "days_since_comp4",
    "age", "model_model1", "model_model2", "model_model3", "model_model4"
]

with patch("joblib.load", side_effect=[mock_model, mock_feature_cols]):
    from fastapi.testclient import TestClient
    from app.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "volt": 170.0, "rotate": 450.0,
    "pressure": 100.0, "vibration": 40.0,
    "volt_mean_3h": 169.5, "volt_std_3h": 1.2,
    "rotate_mean_3h": 448.0, "rotate_std_3h": 3.1,
    "pressure_mean_3h": 99.8, "pressure_std_3h": 0.5,
    "vibration_mean_3h": 39.9, "vibration_std_3h": 0.8,
    "volt_mean_24h": 168.0, "volt_std_24h": 2.1,
    "rotate_mean_24h": 445.0, "rotate_std_24h": 5.0,
    "pressure_mean_24h": 100.2, "pressure_std_24h": 1.1,
    "vibration_mean_24h": 40.1, "vibration_std_24h": 1.3,
    "error_error1": 0, "error_error2": 1,
    "error_error3": 0, "error_error4": 0, "error_error5": 0,
    "days_since_comp1": 45.0, "days_since_comp2": 12.0,
    "days_since_comp3": 30.0, "days_since_comp4": 60.0,
    "age": 15.0,
    "model_model1": 0, "model_model2": 0,
    "model_model3": 1, "model_model4": 0
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "n_features" in data

def test_predict_valid_input():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "failure_predicted" in data
    assert "failure_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def test_predict_missing_field():
    response = client.post("/predict", json={"volt": 170.0})
    assert response.status_code == 422

def test_predict_with_machine_id():
    response = client.post("/predict?machine_id=42", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["machine_id"] == 42