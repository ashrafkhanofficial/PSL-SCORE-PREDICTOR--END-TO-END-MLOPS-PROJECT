"""
Module for training and evaluating machine learning models.
"""

import sys
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

from src.exception import MyException
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_FEATURES = ["venue", "batting_team", "bowling_team"]
NUMERICAL_FEATURES = ["balls_remaining", "current_score", "wickets_remaining", "crr"]


def build_pipeline(
    n_estimators: int = 500, learning_rate: float = 0.1, max_depth: int = 8, random_state: int = 42
) -> Pipeline:
    """
    Build an end-to-end preprocessing and regression pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )

    regressor = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])

    return pipeline


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: Dict[str, Any] = None,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train an end-to-end XGBoost regression pipeline.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        model_params: Optional hyperparameters dictionary

    Returns:
        Tuple of (pipeline, X_train, X_test, y_train, y_test)
    """
    logger.info("Starting model training...")
    try:
        if model_params is None:
            model_params = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Split data into train: {X_train.shape} and test: {X_test.shape}")

        pipeline = build_pipeline(
            n_estimators=model_params.get("n_estimators", 500),
            learning_rate=model_params.get("learning_rate", 0.1),
            max_depth=model_params.get("max_depth", 8),
            random_state=random_state,
        )

        logger.info("Fitting model pipeline...")
        pipeline.fit(X_train, y_train)
        logger.info("Model pipeline fitted successfully")

        return pipeline, X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error("Error occurred during model training")
        raise MyException(e, sys) from e


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate model performance on both training and test sets.
    """
    logger.info("Evaluating model performance...")
    try:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        metrics = {
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
        }

        logger.info(
            f"Evaluation complete. Test MAE: {metrics['test_mae']:.2f}, Test R2: {metrics['test_r2']:.4f}"
        )
        return metrics

    except Exception as e:
        logger.error("Error occurred during model evaluation")
        raise MyException(e, sys) from e


def save_model(model: Pipeline, filepath: str) -> None:
    """
    Save trained model pipeline to disk.
    """
    logger.info(f"Saving model to {filepath}")
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully at {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}")
        raise MyException(e, sys) from e


def load_model(filepath: str) -> Pipeline:
    """
    Load model pipeline from disk.
    """
    logger.info(f"Loading model from {filepath}")
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}")
        raise MyException(e, sys) from e
