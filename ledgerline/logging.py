"""Structured logging setup. Call configure_logging(settings()) once at startup."""

import json
import logging
import sys
from datetime import UTC, datetime

# Standard LogRecord instance attributes that are NOT extra fields.
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via logger.info("msg", extra={...})
        for k, v in record.__dict__.items():
            if k not in _STANDARD_ATTRS:
                base[k] = v
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


def configure_logging(cfg) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        # Already configured (e.g. pytest log-capture or a prior call): don't touch handlers.
        return
    fmt: logging.Formatter
    if cfg.log_format == "json":
        fmt = _JsonFormatter()
    else:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root.addHandler(handler)
