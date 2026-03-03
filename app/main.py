import time
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.schemas import SensorInput, PredictionResponse
from app.model import run_inference, MODEL_VERSION, feature_cols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"PredictIQ API starting up — Model version: {MODEL_VERSION}")
    logger.info(f"Model expects {len(feature_cols)} features")
    yield
    logger.info("PredictIQ API shutting down")

app = FastAPI(
    title="PredictIQ — Predictive Maintenance API",
    description="Real-time failure prediction for industrial equipment",
    version=MODEL_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status"       : "healthy",
        "model_version": MODEL_VERSION,
        "features"     : len(feature_cols)
    }

@app.get("/model-info")
def model_info():
    return {
        "model_version" : MODEL_VERSION,
        "model_type"    : "LightGBM (Tuned with Optuna)",
        "n_features"    : len(feature_cols),
        "feature_names" : feature_cols,
        "training_data" : "Microsoft Azure Predictive Maintenance Dataset",
        "tuning"        : "50 Optuna trials, 3-fold CV",
        "metrics"       : {
            "roc_auc"  : 1.0000,
            "f1_score" : 0.9946,
            "precision": 0.9907,
            "recall"   : 0.9985
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: SensorInput,
    machine_id: int = Query(default=None, description="Optional machine ID")
):
    start = time.time()
    
    try:
        result = run_inference(payload.model_dump(), machine_id=machine_id)
        latency = (time.time() - start) * 1000
        
        logger.info(
            f"Prediction | machine_id={machine_id} | "
            f"risk={result['risk_level']} | "
            f"prob={result['failure_probability']} | "
            f"latency={latency:.1f}ms"
        )
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payloads: list[SensorInput]):
    results = []
    for i, payload in enumerate(payloads):
        result = run_inference(payload.model_dump(), machine_id=i)
        results.append(result)
    return results
