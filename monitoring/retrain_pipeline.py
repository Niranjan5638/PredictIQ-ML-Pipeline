import mlflow
import mlflow.sklearn
import joblib
import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR  = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MLFLOW_EXPERIMENT = "predictiq-retraining"

# ── Load & preprocess data ──
df = pd.read_csv(DATA_DIR / "masterfeatures_small.csv")

# Keep only numeric columns for LightGBM
NUMERIC_COLS = df.select_dtypes(include=['number']).columns.tolist()
FEATURE_COLS = [c for c in NUMERIC_COLS if c not in ['label']]
X = df[FEATURE_COLS]
y = df["label"]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"✅ Loaded {len(FEATURE_COLS)} numeric features")
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Failure rate: {y.mean():.2%}")

def objective(trial):
    params = {
        "objective":        "binary",
        "metric":           "auc",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 10, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val)

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
    )

    pred = model.predict(X_val)
    return roc_auc_score(y_val, pred)


def run_retraining_pipeline():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="automated-retrain-v1.1"):
        print("🚀 Starting automated retraining with Optuna (25 trials)...")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25, show_progress_bar=True)

        best_params = study.best_params
        best_auc    = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_auc", best_auc)

        print(f"\n✅ Best Val AUC: {best_auc:.4f}")
        print(f"Best params: {best_params}")

        # Train final model on ALL data
        full_data  = lgb.Dataset(X, label=y)
        final_model = lgb.train(
            {**best_params, "objective": "binary", "verbosity": -1},
            full_data,
            num_boost_round=200
        )

        # Save model + feature columns
        model_path    = MODELS_DIR / "lgbm_final_v1.1.pkl"
        features_path = MODELS_DIR / "feature_cols_v1.1.pkl"

        joblib.dump(final_model, model_path)
        joblib.dump(FEATURE_COLS, features_path)

        mlflow.log_param("model_version", "v1.1")
        mlflow.log_artifact(str(model_path))

        print(f"\n💾 Model saved  → {model_path}")
        print(f"💾 Features saved → {features_path}")
        print("🎉 Retraining complete! New model is ready.")


if __name__ == "__main__":
    run_retraining_pipeline()
