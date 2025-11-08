"""Logging configuration utilities."""
from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging() -> None:
    """Configure structured logging for the backend."""

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": "INFO",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO",
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
