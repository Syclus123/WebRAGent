"""
webragent.utils.logging_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logging configuration for WebRAGent.

Replaces the global-side-effect ``logs.py`` with an explicit factory function.
Import this module at any time without triggering file creation or handler
registration — call :func:`setup_logging` once at application startup.

Backward-compatible shim::

    from webragent.utils.logging_config import logger  # same as logs.logger
"""

import logging
import os
import re
import sys
import time
from typing import Optional

import colorlog

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STREAM_FMT = colorlog.ColoredFormatter(
    "%(asctime)s**[%(log_color)s%(levelname)s%(reset)s]**|| %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "green",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)


class _AnsiStrippingFormatter(colorlog.ColoredFormatter):
    """File formatter that strips ANSI escape codes before writing."""

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        formatted = super().format(record)
        return self._ANSI_RE.sub("", formatted)


_FILE_FMT = _AnsiStrippingFormatter(
    "%(asctime)s**[%(levelname)s]**|| %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "INFO": "green",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with both stream and file handlers.

    This function is idempotent — calling it multiple times with the same
    *logger_name* will not add duplicate handlers.

    Parameters
    ----------
    log_dir:
        Directory where the timestamped log file will be created.
        Defaults to ``"logs"`` (relative to the current working directory).
    level:
        Root logging level, e.g. ``logging.DEBUG``.  Defaults to
        ``logging.INFO``.
    logger_name:
        Name passed to :func:`logging.getLogger`.  ``None`` (default)
        returns the root logger, matching the original ``logs.py`` behaviour.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        time.strftime("%Y-%m-%d_%H-%M-%S") + ".log",
    )

    _logger = logging.getLogger(logger_name)
    _logger.setLevel(level)

    # Avoid adding duplicate handlers if called more than once.
    if not _logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_FILE_FMT)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(_STREAM_FMT)

        _logger.addHandler(file_handler)
        _logger.addHandler(stream_handler)

    return _logger


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.

    Uses :func:`logging.getLogger` directly — no handlers are added.
    Useful for per-module loggers that inherit the root configuration set
    up by :func:`setup_logging`.

    Parameters
    ----------
    name:
        Dotted logger name, e.g. ``"webragent.agent"``.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Backward-compatible shim
# ---------------------------------------------------------------------------
# Provides ``from webragent.utils.logging_config import logger`` without
# triggering any I/O.  Equivalent to the ``logger`` created in logs.py.
logger = logging.getLogger("webragent")
