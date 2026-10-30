"""
Unit tests for data ingestion, schema validation, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.config import SCHEMA_FILE_PATH, RAW_DATA_FILE_PATH
from src.data.data_ingestion import load_raw_data, load_schema, validate_data
from src.features.feature_engineering import extract_features_and_target, clean_venue_names
from src.exception import MyException


class TestDataIngestion:
    """Tests for data loading and schema validation."""

    def test_schema_load(self):
        """Verify schema.yaml loads with expected keys."""
        schema = load_schema()
        assert isinstance(schema, dict)
        assert "columns" in schema
        assert "categorical_columns" in schema
        assert "numerical_columns" in schema
        assert "target_column" in schema

    def test_validate_data_success(self):
        """Verify validate_data passes when all schema columns exist."""
        schema = load_schema()
        cols = list(schema["columns"].keys())
        mock_df = pd.DataFrame({col: [0] for col in cols})
        validated_df = validate_data(mock_df)
        assert len(validated_df.columns) == len(cols)

    def test_validate_data_missing_column(self):
        """Verify validate_data raises MyException on missing columns."""
        mock_df = pd.DataFrame({"id": [1], "match_id": [101]})
        with pytest.raises(MyException):
            validate_data(mock_df)

    def test_load_raw_data_invalid_path(self):
        """Verify load_raw_data raises MyException when file does not exist."""
        with pytest.raises(MyException):
            load_raw_data("non_existent_path.csv")


class TestFeatureEngineering:
    """Tests for feature engineering transformations."""

    def test_clean_venue_names(self):
        """Verify stadium names are simplified to the primary city name."""
        venues = pd.Series(
            [
                "Gaddafi Stadium Lahore",
                "National Stadium Karachi",
                "Rawalpindi Cricket Stadium",
                "Dubai International Stadium",
            ]
        )
        cleaned = clean_venue_names(venues)
        assert cleaned.tolist() == ["Gaddafi", "National", "Rawalpindi", "Dubai"]

    def test_extract_features_and_target(self):
        """Verify feature matrix and target generation on a synthetic match dataframe."""
        mock_data = {
            "id": range(1, 11),
            "match_id": [101] * 10,
            "date": ["2023-03-01"] * 10,
            "season": [2023] * 10,
            "venue": ["Rawalpindi Cricket Stadium"] * 10,
            "inning": [1] * 10,
            "batting_team": ["Lahore Qalandars"] * 10,
            "bowling_team": ["Karachi Kings"] * 10,
            "over": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "ball": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            "batter": ["Fakhar Zaman"] * 10,
            "bowler": ["Mohammad Amir"] * 10,
            "non_striker": ["Abdullah Shafique"] * 10,
            "batsman_runs": [1, 4, 0, 6, 1, 0, 2, 4, 1, 0],
            "extra_runs": [0] * 10,
            "total_runs": [1, 4, 0, 6, 1, 0, 2, 4, 1, 0],
            "extras_type": [np.nan] * 10,
            "is_wicket": [False, False, False, False, True, False, False, False, False, False],
            "player_dismissed": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "Fakhar Zaman",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "dismissal_kind": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "caught",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "fielder": [np.nan] * 10,
            "winner": ["Lahore Qalandars"] * 10,
            "win_by": ["runs 15"] * 10,
            "match_type": ["regular"] * 10,
            "player_of_match": ["Fakhar Zaman"] * 10,
            "umpire_1": ["Aleem Dar"] * 10,
            "umpire_2": ["Richard Illingworth"] * 10,
        }
        df = pd.DataFrame(mock_data)
        X, y = extract_features_and_target(df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "current_score" in X.columns
        assert "balls_remaining" in X.columns
        assert "wickets_remaining" in X.columns
        assert "crr" in X.columns
        assert len(X) == 10
        assert len(y) == 10
        # Total runs = 19, so max score = 19
        assert y.iloc[0] == 19
