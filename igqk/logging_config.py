"""
IGQK Logging Configuration.

Provides structured logging for production monitoring.
"""

import logging
import os
import sys
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    json_format: bool = False,
):
    """
    Configure IGQK logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
        json_format: Use JSON format (for log aggregation systems).
    """
    level = os.environ.get("IGQK_LOG_LEVEL", level).upper()

    if json_format:
        fmt = (
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"module":"%(module)s","message":"%(message)s"}'
        )
    else:
        fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger = logging.getLogger("igqk")
    logger.info(f"IGQK logging initialized (level={level})")
    return logger


def get_logger(name: str = "igqk") -> logging.Logger:
    """Get a named IGQK logger."""
    return logging.getLogger(name)
