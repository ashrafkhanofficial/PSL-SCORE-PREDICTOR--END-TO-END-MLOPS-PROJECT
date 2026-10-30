"""
Main entry point for the PSL Score Predictor pipeline.
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    RAW_DATA_FILE_PATH,
    PROCESSED_DATA_FILE_PATH,
    MODEL_FILE_PATH,
    PARAMS_FILE_PATH
)
from src.data.data_ingestion import load_raw_data, validate_data, save_processed_data
from src.features.feature_engineering import extract_features_and_target
from src.models.train_model import train_model, evaluate_model, save_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_params() -> dict:
    """Load hyperparameters and pipeline settings from params.yaml."""
    try:
        with open(PARAMS_FILE_PATH, 'r') as f:
            params = yaml.safe_load(f)
        return params or {}
    except Exception as e:
        logger.warning(f"Failed to read params.yaml ({e}), using default parameters.")
        return {'data': {'test_size': 0.2, 'random_state': 42}, 'model': {'n_estimators': 300}}


def run_pipeline():
    """Execute end-to-end MLOps pipeline."""
    print("=" * 60)
    print("PSL Score Predictor: End-to-End Training Pipeline")
    print("=" * 60)

    # 1. Load Parameters
    params = load_params()
    data_params = params.get('data', {})
    model_params = params.get('model', {})
    test_size = data_params.get('test_size', 0.2)
    random_state = data_params.get('random_state', 42)

    # 2. Data Ingestion & Validation
    print("\n[Stage 1/4] Ingesting and validating raw match data...")
    df_raw = load_raw_data(str(RAW_DATA_FILE_PATH))
    df_validated = validate_data(df_raw)
    print(f"-> Ingested {len(df_validated):,} ball-by-ball records across {len(df_validated.columns)} columns.")

    # 3. Feature Engineering
    print("\n[Stage 2/4] Engineering features and targets...")
    X, y = extract_features_and_target(df_validated)
    print(f"-> Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    # Save processed feature sample for inspection/tracking
    processed_df = X.copy()
    processed_df['target'] = y
    save_processed_data(processed_df, str(PROCESSED_DATA_FILE_PATH))
    print(f"-> Processed dataset saved to: {PROCESSED_DATA_FILE_PATH}")

    # 4. Model Training
    print("\n[Stage 3/4] Training XGBoost ColumnTransformer Pipeline...")
    model, X_train, X_test, y_train, y_test = train_model(
        X, y,
        test_size=test_size,
        random_state=random_state,
        model_params=model_params
    )
    print("-> Pipeline fitted successfully.")

    # 5. Model Evaluation
    print("\n[Stage 4/4] Evaluating model performance...")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    print("=" * 60)
    print("Evaluation Results:")
    print(f"  Test R2 Score:   {metrics['test_r2']:.4f}")
    print(f"  Test MAE:        {metrics['test_mae']:.2f} runs")
    print(f"  Test RMSE:       {metrics['test_rmse']:.2f} runs")
    print(f"  Train R2 Score:  {metrics['train_r2']:.4f}")
    print(f"  Train MAE:       {metrics['train_mae']:.2f} runs")
    print("=" * 60)

    # 6. Save Model Artifact
    save_model(model, str(MODEL_FILE_PATH))
    print(f"\nTrained pipeline saved to: {MODEL_FILE_PATH}")
    print("Pipeline execution completed successfully!\n")
    return metrics


if __name__ == '__main__':
    run_pipeline()
