from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any

_LOGGER_NAME = "data_preprocessor"
_configured_log_path: Path | None = None


def _close_logger_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def _configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    current_path = next(
        (Path(handler.baseFilename) for handler in logger.handlers if hasattr(handler, "baseFilename")),
        None,
    )
    if current_path != path:
        _close_logger_handlers(logger)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        handler.formatter.default_msec_format = "%s,%03d"
        logger.addHandler(handler)
    return logger


def configure_data_preprocessor_logging(*, log_path: str | Path) -> logging.Logger:
    global _configured_log_path
    _configured_log_path = Path(log_path)
    return _configure_logger(_configured_log_path)


def close_data_preprocessor_logging() -> None:
    global _configured_log_path
    _configured_log_path = None
    _close_logger_handlers(logging.getLogger(_LOGGER_NAME))


def log_calls(log_path: Callable[[], Path]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = _configure_logger(_configured_log_path or log_path())
            logger.info("Start %s", func.__name__)
            started = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                logger.info("Finished %s in %.3fs", func.__name__, perf_counter() - started)

        return wrapper

    return decorate
