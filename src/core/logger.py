import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def mask_rtsp_url(url: str) -> str:
    """
    Mask username and password in RTSP / HTTP URL for security.
    Example: rtsp://admin:secret123@192.168.1.100:554/stream1
    Output:  rtsp://***:***@192.168.1.100:554/stream1
    """
    if not url:
        return ""
    return re.sub(r"://([^:@\s]+):([^@\s]+)@", r"://***:***@", url)


class MaskingFormatter(logging.Formatter):
    """Custom logging formatter that automatically masks sensitive URLs in log messages."""
    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return mask_rtsp_url(original)


def setup_logger(
    name: str = "traffic_ai",
    logs_dir: str = "logs",
    level: str = "INFO",
    max_bytes: int = 52428800,
    backup_count: int = 5,
) -> logging.Logger:
    """Setup and return logger with console and rotating file handlers."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = MaskingFormatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # App Log (All levels >= log_level)
    app_log_path = os.path.join(logs_dir, "app.log")
    app_handler = RotatingFileHandler(
        app_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    app_handler.setLevel(log_level)
    app_handler.setFormatter(formatter)
    logger.addHandler(app_handler)

    # Error Log (Level ERROR and CRITICAL only)
    error_log_path = os.path.join(logs_dir, "error.log")
    error_handler = RotatingFileHandler(
        error_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


logger = setup_logger()
