"""
Module for feature engineering and transformation on PSL match data.
"""

import sys
from typing import Tuple
import pandas as pd
import numpy as np

from src.exception import MyException
from src.utils.logger import get_logger

logger = get_logger(__name__)

ELIGIBLE_TEAMS = [
    "Peshawar Zalmi",
    "Islamabad United",
    "Lahore Qalandars",
    "Karachi Kings",
    "Quetta Gladiators",
    "Multan Sultans",
]

CATEGORICAL_FEATURES = ["venue", "batting_team", "bowling_team"]
NUMERICAL_FEATURES = ["balls_remaining", "current_score", "wickets_remaining", "crr"]


def clean_venue_names(series: pd.Series) -> pd.Series:
    """
    Standardize venue names by extracting the city (first word).
    """
    return series.astype(str).str.split(" ").str[0]


def extract_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw ball-by-ball DataFrame into model features X and target y.

    Args:
        df: Input raw PSL DataFrame

    Returns:
        Tuple of (X, y)
    """
    logger.info("Starting feature engineering pipeline...")
    try:
        data = df.copy()

        # Step 1: Filter to 1st innings only
        data = data[data["inning"] == 1].copy()
        logger.info(f"Filtered 1st innings records: {len(data)} rows")

        # Step 2: Keep only matches between the 6 eligible PSL franchises
        data = data[
            data["batting_team"].isin(ELIGIBLE_TEAMS) & data["bowling_team"].isin(ELIGIBLE_TEAMS)
        ].copy()

        # Step 3: Compute cumulative match runs and wickets
        data["current_score"] = data.groupby(["match_id", "inning"])["total_runs"].cumsum()
        data["total_out"] = data.groupby(["match_id", "inning"])["is_wicket"].cumsum()

        # Step 4: Compute balls remaining (T20 = 120 balls total)
        data["balls_remaining"] = 120 - ((data["over"] - 1) * 6 + data["ball"])
        data["balls_remaining"] = data["balls_remaining"].clip(lower=0, upper=120)

        # Step 5: Compute wickets remaining (max 10 wickets)
        data["wickets_remaining"] = (10 - data["total_out"]).clip(lower=0, upper=10)

        # Step 6: Compute Current Run Rate (CRR)
        balls_bowled = 120 - data["balls_remaining"]
        data["crr"] = np.where(balls_bowled > 0, (data["current_score"] / balls_bowled) * 6, 0.0)

        # Step 7: Compute target (maximum runs achieved in the 1st innings)
        target_map = data.groupby("match_id")["current_score"].max()
        data["target"] = data["match_id"].map(target_map)

        # Step 8: Standardize venue names
        data["venue"] = clean_venue_names(data["venue"])

        # Filter out corrupted rows where balls_bowled == 0 or target is missing
        data = data.dropna(subset=["target"])
        data = data[data["balls_remaining"] < 120]

        X = data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].copy()
        y = data["target"].astype(float).copy()

        logger.info(f"Feature engineering completed. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    except Exception as e:
        logger.error("Error occurred during feature engineering")
        raise MyException(e, sys) from e
