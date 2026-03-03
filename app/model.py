import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR   = Path("models")
MODEL_PATH   = MODELS_DIR / "lgbm_final.pkl"
FEATURES_PATH = MODELS_DIR / "feature_cols.pkl"
MODEL_VERSION = "1.0.0"

PREDICTIQ_ENV = os.getenv("PREDICTIQ_ENV", "local")

if PREDICTIQ_ENV == "ci":
    # CI mode: use a dummy model that returns constant low risk
    class DummyModel:
        def predict_proba(self, X):
            # Always return [no-failure, failure] probabilities
            return np.array([[0.99, 0.01]] * len(X))

    model = DummyModel()
    feature_cols = ["dummy"]  # will be overridden by tests using the API schema
else:
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)


def get_risk_level(probability: float) -> tuple[str, str]:
    if probability < 0.25:
        return "LOW", "No immediate action required. Continue routine monitoring."
    elif probability < 0.50:
        return "MEDIUM", "Schedule inspection within 48 hours."
    elif probability < 0.75:
        return "HIGH", "Schedule maintenance within 24 hours."
    else:
        return "CRITICAL", "Immediate maintenance required. Risk of imminent failure."

def run_inference(input_data: dict, machine_id: int | None = None) -> dict:
    if PREDICTIQ_ENV == "ci":
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame([input_data])[feature_cols]

    prob = model.predict_proba(df)[0, 1]
    predicted = bool(prob >= 0.5)
    risk_level, recommendation = get_risk_level(prob)
    
    return {
        "machine_id"         : machine_id,
        "failure_predicted"  : predicted,
        "failure_probability": round(float(prob), 4),
        "risk_level"         : risk_level,
        "recommendation"     : recommendation,
        "model_version"      : MODEL_VERSION
    }