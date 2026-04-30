"""Application-wide logging configuration.

Call setup_logging() from entry points (sync.py, app/gradio_app.py)
to install a RotatingFileHandler whose format records the source
filename and line number of every log call. A second 'retrieval'
logger emits one JSON object per line to a separate rotating file
for agent observability.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "%(filename)s:%(lineno)d — %(message)s"
)
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_FILE = Path("logs") / "hkia.log"
_DEFAULT_RETRIEVAL_LOG_FILE = Path("logs") / "retrieval.jsonl"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5

_configured = False
_retrieval_configured = False


def setup_logging(
    log_file: Path | str = _DEFAULT_LOG_FILE,
    level: int = logging.INFO,
    also_stdout: bool = True,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure the root logger to write to a rotating file.

    Idempotent — subsequent calls are no-ops so re-imports by different
    entry points don't stack duplicate handlers. Also initialises the
    dedicated 'retrieval' logger (see setup_retrieval_logger).

    Args:
        log_file: Destination log file. Parent directory is created if
            it doesn't exist.
        level: Minimum record level the root logger will emit.
        also_stdout: If True, also attach a StreamHandler to stdout
            using the same formatter.
        max_bytes: Size threshold in bytes that triggers rotation.
        backup_count: Number of rotated files to retain.
    """
    global _configured
    if _configured:
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    file_handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)

    if also_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    _configured = True

    # Retrieval logger is configured alongside the root logger so every
    # entry point gets JSONL observability for free.
    setup_retrieval_logger()


def setup_retrieval_logger(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Configure the dedicated 'retrieval' logger for JSONL events.

    The handler uses a bare '%(message)s' formatter — callers are
    expected to pass an already-serialised JSON string. Does not
    propagate to the root logger so retrieval events stay out of
    hkia.log.

    Args:
        log_file: Destination JSONL file. Defaults to
            settings.retrieval_log_file when None.
        level: Minimum record level for the retrieval logger.
        max_bytes: Size threshold in bytes that triggers rotation.
        backup_count: Number of rotated files to retain.

    Returns:
        The configured 'retrieval' logger.
    """
    global _retrieval_configured
    retrieval_logger = logging.getLogger("retrieval")
    if _retrieval_configured:
        return retrieval_logger

    if log_file is None:
        # Imported lazily to avoid a circular import at module load.
        from config.settings import settings

        log_file = settings.retrieval_log_file

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    retrieval_logger.setLevel(level)
    retrieval_logger.addHandler(handler)
    retrieval_logger.propagate = False

    _retrieval_configured = True
    return retrieval_logger
