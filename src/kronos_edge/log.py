"""Thin logging wrapper for kronos_edge, stdlib only.

Call ``configure()`` once at startup, then ``get_logger(__name__)`` in each
module. Loggers gain a ``trace()`` level below DEBUG. Pass
``extra={"stage": "..."}`` to tag a record; attach an ``ErrorCollector`` to
the ``"kronos_edge"`` logger to collect WARNING-and-above records for a run.
"""

import logging
import sys

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

PACKAGE_LOGGER = "kronos_edge"
_FORMAT = "%(levelname)s %(name)s [%(stage)s] %(message)s"


class _Logger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


logging.setLoggerClass(_Logger)


class _StageFilter(logging.Filter):
    """Fill in ``record.stage`` with "-" when the caller did not pass one."""

    def filter(self, record):
        if not hasattr(record, "stage"):
            record.stage = "-"
        return True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure(level: int | str = "INFO") -> None:
    """Attach a single stderr handler to the package logger. Idempotent."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    if any(getattr(h, "_kronos_edge", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_StageFilter())
    handler._kronos_edge = True
    logger.addHandler(handler)


class ErrorCollector(logging.Handler):
    """Keep every WARNING-or-higher record in ``self.records``."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)
