# monitoring/monitor_drift.py
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

DATA_DIR = Path("data")
BASELINE_PATH = DATA_DIR / "master_features.csv"  # Your training features
MLFLOW_EXPERIMENT = "predictiq-monitoring"

def psi(baseline: pd.Series, current: pd.Series, buckets=10) -> float:
    """Population Stability Index - measures distribution shift"""
    def scale_range(input_data, min_val, max_val):
        return (input_data - min_val) / (max_val - min_val)

    scaled_baseline = scale_range(baseline, baseline.min(), baseline.max())
    scaled_current = scale_range(current, current.min(), current.max())

    breaks = np.arange(0, 10, 1.0 / buckets)
    groups_baseline = pd.cut(scaled_baseline, bins=breaks, labels=False, include_lowest=True)
    groups_current = pd.cut(scaled_current, bins=breaks, labels=False, include_lowest=True)

    # Get value counts and fill missing bins with 0
    hist_baseline = groups_baseline.value_counts().reindex(range(buckets), fill_value=0).sort_index()
    hist_current = groups_current.value_counts().reindex(range(buckets), fill_value=0).sort_index()

    pbi = hist_baseline / hist_baseline.sum()
    pci = hist_current / hist_current.sum()

    # Avoid log(0) by adding small epsilon
    pbi = pbi.replace(0, 1e-15)
    pci = pci.replace(0, 1e-15)

    psi = np.sum((pbi - pci) * np.log(pbi / pci))
    return psi


def load_baseline():
    return pd.read_csv(BASELINE_PATH)

def get_recent_data(days=7):
    # Simulate production data - in real setup, query from database
    df = pd.read_csv(BASELINE_PATH)
    return df.sample(n=int(len(df) * days / 365), replace=True)  # 7 days worth

def detect_drift(baseline_df: pd.DataFrame, recent_df: pd.DataFrame):
    results = {}
    
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        psi_score = psi(baseline_df[col], recent_df[col])
        ks_stat, ks_pval = ks_2samp(baseline_df[col], recent_df[col])
        
        results[col] = {
            "psi": psi_score,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval
        }
    
    # Aggregate metrics
    avg_psi = np.mean([v["psi"] for v in results.values()])
    num_high_drift = sum(1 for v in results.values() if v["psi"] > 0.1)
    
    return {
        "drift_summary": {
            "avg_psi": avg_psi,
            "high_drift_features": num_high_drift,
            "needs_retraining": avg_psi > 0.1
        },
        "feature_details": results
    }

def run_monitoring():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=f"daily-drift-check-{pd.Timestamp.now().strftime('%Y%m%d')}"):
        baseline = load_baseline()
        recent = get_recent_data()
        
        drift_results = detect_drift(baseline, recent)
        
        # Log metrics
        summary = drift_results["drift_summary"]
        mlflow.log_metric("avg_psi", summary["avg_psi"])
        mlflow.log_metric("high_drift_features", summary["high_drift_features"])
        mlflow.log_metric("needs_retraining", int(summary["needs_retraining"]))
        
        # Log feature-level drift
        mlflow.log_dict(drift_results["feature_details"], "feature_drift.json")
        
        print(f"Avg PSI: {summary['avg_psi']:.4f}")
        print(f"High drift features: {summary['high_drift_features']}")
        if summary["needs_retraining"]:
            print("🚨 RETRAINING TRIGGERED")
        else:
            print("✅ Model stable")

if __name__ == "__main__":
    run_monitoring()
