"""Per-worker logger writing to a timestamped file and to stdout."""
import logging
import os
import sys
import datetime

DEFAULT_LOGGER_NAME = "agri_sigra"


def get_logger(logger_name: str = DEFAULT_LOGGER_NAME, run_tag: str = None,
                log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """Create or fetch a logger that writes to a timestamped file in log_dir and to stdout."""
    os.makedirs(log_dir, exist_ok=True)
    if run_tag is None:
        run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(log_dir, f"{logger_name}_{run_tag}.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # file handler first: run.py reads logger.handlers[0].baseFilename
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.info("=" * 78)
    logger.info(f"Agri-SCHOLAR logger '{logger_name}' started. Full log: {log_path}")
    logger.info("=" * 78)
    return logger


def section(logger, title: str):
    """Pretty section header, used to visually separate pipeline stages in the log."""
    bar = "-" * 78
    logger.info(bar)
    logger.info(title)
    logger.info(bar)
