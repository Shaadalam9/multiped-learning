"""Configure shared console and timestamped file logging for the analysis."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def logs(show_level: str = "INFO", save_level: str = "INFO",
         program_name: str = "ordering_comparison", path: Path | None = None) -> Path:
    """Configure logging once per run and return the UTF-8 log file path.

    Repeated calls replace only handlers created here, avoiding duplicate messages
    without removing handlers belonging to a notebook or another application.
    """
    console_level = _logging_level(show_level)
    file_level = _logging_level(save_level)
    directory = Path(path) if path is not None else Path(__file__).resolve().parent / "_logs"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%fZ")
    log_path = directory / f"{program_name}_{timestamp}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_multiped_handler", False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(min(console_level, file_level))
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        handler._multiped_handler = True
        root.addHandler(handler)
    for name in ("matplotlib", "numexpr", "urllib3", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


def _logging_level(value: str) -> int:
    """Reject misspelled level names before creating handlers or output files."""
    levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
              "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    try:
        return levels[value.upper()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unknown logging level: {value}") from exc
