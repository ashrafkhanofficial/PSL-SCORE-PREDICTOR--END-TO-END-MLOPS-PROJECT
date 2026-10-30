"""
Unit tests for model pipeline construction, training, evaluation, and serialization.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from src.models.train_model import (
    build_pipeline,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


@pytest.fixture
def mock_training_data():
    """Generate synthetic PSL feature DataFrame and target Series."""
    np.random.seed(42)
    n_samples = 100
    venues = ["Lahore", "Karachi", "Rawalpindi", "Dubai"]
    teams = [
        "Peshawar Zalmi",
        "Islamabad United",
        "Lahore Qalandars",
        "Karachi Kings",
        "Quetta Gladiators",
        "Multan Sultans",
    ]

    X = pd.DataFrame(
        {
            "venue": np.random.choice(venues, n_samples),
            "batting_team": np.random.choice(teams, n_samples),
            "bowling_team": np.random.choice(teams, n_samples),
            "balls_remaining": np.random.randint(1, 120, n_samples),
            "current_score": np.random.randint(10, 180, n_samples),
            "wickets_remaining": np.random.randint(1, 10, n_samples),
            "crr": np.random.uniform(5.0, 11.0, n_samples),
        }
    )

    # Realistic target
    y = pd.Series(
        X["current_score"]
        + X["crr"] * (X["balls_remaining"] / 6.0)
        + np.random.normal(0, 10, n_samples)
    )
    return X, y


class TestModelPipeline:
    """Tests for model training, evaluation, and serialization."""

    def test_build_pipeline(self):
        """Verify pipeline contains preprocessor and model steps."""
        pipeline = build_pipeline(n_estimators=10, max_depth=3)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_train_and_evaluate_model(self, mock_training_data):
        """Verify pipeline trains and generates evaluation metrics."""
        X, y = mock_training_data
        model_params = {"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1}

        pipeline, X_train, X_test, y_train, y_test = train_model(
            X, y, test_size=0.2, random_state=42, model_params=model_params
        )

        assert isinstance(pipeline, Pipeline)
        assert len(X_train) == 80
        assert len(X_test) == 20

        # Predictions
        preds = pipeline.predict(X_test)
        assert len(preds) == len(X_test)

        # Evaluation metrics
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        assert "test_r2" in metrics
        assert "test_mae" in metrics
        assert "test_rmse" in metrics
        assert metrics["test_mae"] >= 0

    def test_save_and_load_model(self, mock_training_data, tmp_path):
        """Verify pipeline serializes and deserializes accurately."""
        X, y = mock_training_data
        pipeline, _, X_test, _, _ = train_model(
            X, y, test_size=0.2, random_state=42, model_params={"n_estimators": 10}
        )

        test_filepath = tmp_path / "test_model.pkl"
        save_model(pipeline, str(test_filepath))
        assert test_filepath.exists()

        loaded_pipeline = load_model(str(test_filepath))
        assert isinstance(loaded_pipeline, Pipeline)

        # Ensure predictions are identical
        orig_preds = pipeline.predict(X_test)
        loaded_preds = loaded_pipeline.predict(X_test)
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)
