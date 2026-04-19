"""
Configuration settings for the project.
"""
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

MODELS_DIR = PROJECT_DIR / 'models'
NOTEBOOKS_DIR = PROJECT_DIR / 'notebooks'
REPORTS_DIR = PROJECT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Data settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model settings
N_ESTIMATORS = 100
MAX_DEPTH = None

# Logging settings
LOG_FILE = PROJECT_DIR / 'logs' / 'app.log'
LOG_LEVEL = 'INFO'
