"""Logging setup for PANORAMA."""
from __future__ import annotations

import logging          # absolute import -> the STDLIB module, not this file
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, stream=sys.stdout,
                        format=_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)