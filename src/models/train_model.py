"""
Module for training machine learning models.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Train a machine learning model.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        
    Returns:
        Dictionary with evaluation metrics
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
    }
    
    return metrics


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str):
    """
    Load model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
