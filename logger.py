"""
Logging setup for the trading bot.
"""

import logging
import os
import sys
from config import LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
