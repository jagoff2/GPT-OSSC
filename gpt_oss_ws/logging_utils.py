from __future__ import annotations

import logging
from typing import Optional


def init_logger(name: str, level: str = "INFO", handler: Optional[logging.Handler] = None) -> logging.Logger:
  logger = logging.getLogger(name)
  if logger.handlers:
    return logger
  logger.setLevel(getattr(logging, level.upper(), logging.INFO))
  stream = handler or logging.StreamHandler()
  fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  stream.setFormatter(logging.Formatter(fmt))
  logger.addHandler(stream)
  logger.propagate = False
  return logger
