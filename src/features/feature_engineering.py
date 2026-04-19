"""
Module for feature engineering and transformation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new features
    """
    df_features = df.copy()
    
    # Add your feature engineering logic here
    # Example: df_features['new_feature'] = df_features['col1'] * df_features['col2']
    
    return df_features


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    df_filled = df.copy()
    
    if strategy == 'drop':
        df_filled = df_filled.dropna()
    elif strategy in ['mean', 'median']:
        df_filled = df_filled.fillna(df_filled.agg(strategy))
    
    return df_filled


def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded


def scale_features(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Scale numeric features.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        
    Returns:
        DataFrame with scaled features
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    
    return df_scaled
