import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve().parent))
from config.config import LOG_FILE, LOG_DIR, MAX_LOG_SIZE, BACKUP_COUNT


# Config
# LOG_DIR = "logs"
# LOG_FILE = "app.log"
# MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
# BACKUP_COUNT = 3


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance (named logger).
    """

    # Ensure log directory exists
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    log_file_path = log_dir / LOG_FILE

    # Create/get logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers (important)
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger