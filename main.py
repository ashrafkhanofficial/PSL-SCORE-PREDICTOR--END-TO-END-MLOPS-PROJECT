"""
Main entry point for the PSL Score Predictor pipeline.
"""
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.load_data import load_raw_data, save_processed_data
from features.feature_engineering import (
    create_features,
    handle_missing_values,
    encode_categorical_features,
    scale_features
)
from models.train_model import train_model, evaluate_model, save_model
from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TEST_SIZE,
    RANDOM_STATE
)


def main():
    """Main pipeline execution."""
    print("PSL Score Predictor Pipeline")
    print("=" * 50)
    
    # Example: Load data
    # data_path = RAW_DATA_DIR / 'psl_data.csv'
    # df = load_raw_data(str(data_path))
    # print(f"Loaded data shape: {df.shape}")
    
    # Example: Preprocessing
    # df = handle_missing_values(df)
    # df = create_features(df)
    
    # Example: Training
    # model, X_train, X_test, y_train, y_test = train_model(X, y)
    # metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    # print(f"Model metrics: {metrics}")
    
    # Example: Save model
    # save_model(model, MODELS_DIR / 'model.pkl')
    
    print("\nPipeline template ready. Replace with your data and logic.")


if __name__ == '__main__':
    main()
