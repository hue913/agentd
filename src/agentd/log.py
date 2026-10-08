"""Unified logging for agentd (stdlib logging only).

Why this exists
---------------
P0 left a handful of `print(..., file=sys.stderr)` calls and several silently
swallowed exceptions. This module gives them one channel: every logger shares
a stderr handler, a level driven by AGENTD_LOG_LEVEL (default WARNING — the
service is a daemon, so INFO chatter would drown journald), and a format that
names the emitting module so an operator can grep a single subsystem.

Two helpers matter beyond get_logger():

* TokenRedactingFilter — uvicorn.access logs the raw request line, and the
  SSE/WebSocket transports legitimately carry `?token=...` in the query
  string. Without redaction the API token lands in journald verbatim.
* install_uvicorn_filters — attaches that filter to the uvicorn loggers before
  cli.serve starts the server.
"""

from __future__ import annotations

import logging
import os
import re
import sys

LOG_LEVEL_ENV = "AGENTD_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "warning"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# `token=<value>` in a query string, and a Bearer value pasted into a URL.
# The value stops at whitespace, `&`, `"` or `>` (the last two show up when a
# request line is quoted inside a log message).
_TOKEN_RE = re.compile(r"(token=)[^\s&\">]+", re.IGNORECASE)
_BEARER_RE = re.compile(r"(Bearer\s+)[^\s&\">]+", re.IGNORECASE)


class TokenRedactingFilter(logging.Filter):
    """Redact API tokens from log records before they reach any handler.

    record.args is collapsed into record.getMessage() first: if the token
    arrived as a lazy %s argument, replacing only record.msg would miss it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            redacted = _TOKEN_RE.sub(r"\1***", message)
            redacted = _BEARER_RE.sub(r"\1***", redacted)
            if redacted != message:
                record.msg, record.args = redacted, None
        except Exception:  # a broken record must never break logging itself
            pass
        return True


def _level_from_env() -> int:
    name = os.environ.get(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL).strip().lower()
    level = logging.getLevelName(name.upper()) if name else None
    # getLevelName returns the input string back for unknown names — guard it.
    return level if isinstance(level, int) else logging.WARNING


def get_logger(name: str) -> logging.Logger:
    """Return a logger wired to stderr with the shared format and level.

    Idempotent by marker attribute on the handler, so repeated get_logger()
    calls (per module import) never duplicate output.
    """
    logger = logging.getLogger(name)
    if not any(getattr(handler, "_agentd_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._agentd_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.setLevel(_level_from_env())
        logger.propagate = False  # one stderr line per record, not two
    return logger


def install_uvicorn_filters() -> None:
    """Attach token redaction to the uvicorn loggers.

    Called from `agentd serve` before uvicorn.run(): uvicorn installs its own
    handlers at run() time, but logger-level filters apply regardless of which
    handler renders the record, so attaching early is sufficient.
    """
    redactor = TokenRedactingFilter()
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(redactor)
