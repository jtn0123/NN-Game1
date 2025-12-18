"""
Centralized logging infrastructure for the DQN game project.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.debug("Epsilon: 0.95")
    logger.warning("Low memory")
    logger.error("Failed to save model")

Configuration:
    Set LOG_LEVEL in config.py to control verbosity:
    - DEBUG: All messages including debug info
    - INFO: Normal operation messages (default)
    - WARNING: Warnings and errors only
    - ERROR: Errors only
"""

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class LogLevel(Enum):
    """Log levels for configuration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


# Module-level state
_initialized = False
_log_dir: Optional[Path] = None
_file_handler: Optional[logging.FileHandler] = None


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }

    def __init__(self, fmt: str, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_dir: str = 'logs',
    level: LogLevel = LogLevel.INFO,
    console_output: bool = True,
    file_output: bool = True,
    log_filename: Optional[str] = None,
) -> None:
    """
    Initialize the logging system.

    Args:
        log_dir: Directory for log files
        level: Minimum log level to capture
        console_output: Whether to output to console
        file_output: Whether to output to file
        log_filename: Custom log filename (default: training_YYYYMMDD_HHMMSS.log)
    """
    global _initialized, _log_dir, _file_handler

    if _initialized:
        return

    _log_dir = Path(log_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger for the project
    root_logger = logging.getLogger('dqn')
    root_logger.setLevel(level.value)
    root_logger.handlers.clear()

    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_fmt = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            use_colors=True
        )
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)

    # File handler without colors
    if file_output:
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'training_{timestamp}.log'

        log_path = _log_dir / log_filename
        _file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        _file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        _file_handler.setFormatter(file_fmt)
        root_logger.addHandler(_file_handler)

    _initialized = True
    root_logger.info(f"Logging initialized (level={level.name}, file={file_output})")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured with project settings

    Example:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    # Auto-initialize with defaults if not already done
    if not _initialized:
        setup_logging()

    # Create child logger under 'dqn' namespace
    # Strip 'src.' prefix for cleaner names
    if name.startswith('src.'):
        name = name[4:]

    return logging.getLogger(f'dqn.{name}')


def get_log_path() -> Optional[Path]:
    """Get the current log file path."""
    if _file_handler is not None:
        return Path(_file_handler.baseFilename)
    return None


def log_training_metrics(
    episode: int,
    score: float,
    epsilon: float,
    loss: Optional[float] = None,
    avg_q: Optional[float] = None,
    steps: Optional[int] = None,
) -> None:
    """
    Log training metrics in a consistent format.

    Args:
        episode: Current episode number
        score: Episode score/reward
        epsilon: Current exploration rate
        loss: Training loss (if available)
        avg_q: Average Q-value (if available)
        steps: Steps in episode (if available)
    """
    logger = get_logger('training')

    metrics = [
        f"ep={episode}",
        f"score={score:.1f}",
        f"eps={epsilon:.4f}",
    ]

    if loss is not None:
        metrics.append(f"loss={loss:.6f}")
    if avg_q is not None:
        metrics.append(f"avg_q={avg_q:.3f}")
    if steps is not None:
        metrics.append(f"steps={steps}")

    logger.info(" | ".join(metrics))


def log_model_event(event: str, path: str, **kwargs) -> None:
    """
    Log model-related events (save/load).

    Args:
        event: Event type ('save', 'load', 'checkpoint')
        path: Model file path
        **kwargs: Additional context (e.g., episode, score)
    """
    logger = get_logger('model')

    extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    if extra:
        logger.info(f"{event.upper()} | {path} | {extra}")
    else:
        logger.info(f"{event.upper()} | {path}")
