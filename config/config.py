"""
Configuration settings for the project.
"""

from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PROJECT_DIR  # Alias for compatibility across modules

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Dataset & Configuration file paths
RAW_DATA_FILE_NAME = "Psl_Complete_Dataset(2016-2024).csv"
RAW_DATA_FILE_PATH = RAW_DATA_DIR / RAW_DATA_FILE_NAME
PROCESSED_DATA_FILE_PATH = PROCESSED_DATA_DIR / "psl_processed.csv"
SCHEMA_FILE_PATH = PROJECT_DIR / "config" / "schema.yaml"
PARAMS_FILE_PATH = PROJECT_DIR / "params.yaml"

MODELS_DIR = PROJECT_DIR / "models"
MODEL_FILE_PATH = MODELS_DIR / "psl_pipeline.pkl"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Logging settings
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
