from pydantic import BaseModel, Field
from typing import Optional

class SensorInput(BaseModel):
    # Raw sensor readings
    volt           : float = Field(..., description="Voltage reading")
    rotate         : float = Field(..., description="Rotation speed")
    pressure       : float = Field(..., description="Pressure reading")
    vibration      : float = Field(..., description="Vibration level")

    # Rolling means 3h
    volt_mean_3h      : float
    volt_std_3h       : float
    rotate_mean_3h    : float
    rotate_std_3h     : float
    pressure_mean_3h  : float
    pressure_std_3h   : float
    vibration_mean_3h : float
    vibration_std_3h  : float

    # Rolling means 24h
    volt_mean_24h      : float
    volt_std_24h       : float
    rotate_mean_24h    : float
    rotate_std_24h     : float
    pressure_mean_24h  : float
    pressure_std_24h   : float
    vibration_mean_24h : float
    vibration_std_24h  : float

    # Error counts (last 24h)
    error_error1 : float = 0.0
    error_error2 : float = 0.0
    error_error3 : float = 0.0
    error_error4 : float = 0.0
    error_error5 : float = 0.0

    # Maintenance history
    days_since_comp1 : float = 9999.0
    days_since_comp2 : float = 9999.0
    days_since_comp3 : float = 9999.0
    days_since_comp4 : float = 9999.0

    # Machine metadata
    age           : float = Field(..., description="Machine age in years")
    model_model1  : int = 0
    model_model2  : int = 0
    model_model3  : int = 0
    model_model4  : int = 0

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    machine_id        : Optional[int]
    failure_predicted : bool
    failure_probability: float
    risk_level        : str   # LOW, MEDIUM, HIGH, CRITICAL
    recommendation    : str
    model_version     : str