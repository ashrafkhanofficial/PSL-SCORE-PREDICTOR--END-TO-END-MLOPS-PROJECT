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


# Logging settings
LOG_DIR = PROJECT_DIR / 'logs'
LOG_FILE = PROJECT_DIR / LOG_DIR / 'app.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
