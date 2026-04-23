"""Application-wide logging configuration.

Call setup_logging() from entry points (sync.py, app/gradio_app.py)
to install a RotatingFileHandler whose format records the source
filename and line number of every log call.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "%(filename)s:%(lineno)d — %(message)s"
)
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_FILE = Path("logs") / "hkia.log"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5

_configured = False


def setup_logging(
    log_file: Path | str = _DEFAULT_LOG_FILE,
    level: int = logging.INFO,
    also_stderr: bool = True,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure the root logger to write to a rotating file.

    Idempotent — subsequent calls are no-ops so re-imports by different
    entry points don't stack duplicate handlers.

    Args:
        log_file: Destination log file. Parent directory is created if
            it doesn't exist.
        level: Minimum record level the root logger will emit.
        also_stderr: If True, also attach a StreamHandler to stderr
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

    if also_stderr:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    _configured = True
