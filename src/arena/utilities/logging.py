import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    level = level.upper()

    # Remove default handler
    logger.remove()

    # Add custom handler with color formatting
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def get_logger(name: str):
    return logger.bind(module=name)
