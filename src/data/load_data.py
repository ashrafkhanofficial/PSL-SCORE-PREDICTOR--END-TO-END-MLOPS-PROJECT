"""
Module for loading data from various sources.
"""
import pandas as pd
from pathlib import Path


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    return pd.read_csv(file_path)


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to a CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the file
    """
    df.to_csv(output_path, index=False)
