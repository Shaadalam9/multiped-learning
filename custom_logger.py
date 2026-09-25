import logging


class CustomLogger:
    """Application logger supporting standard %-style and legacy {} messages."""

    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Log an error with the active exception traceback."""
        kwargs.setdefault("exc_info", True)
        self.error(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        if self.logger.isEnabledFor(level):
            if args and "{" in msg:
                msg = msg.format(*args)
                args = ()
            self.logger.log(level, msg, *args, **kwargs)


logger = CustomLogger("ordering_comparison")


def configure_logging(level: str = "INFO"):
    """Set up shared console/file handlers and route warnings through logging."""
    from logmod import logs
    log_path = logs(show_level=level, save_level=level)
    logging.captureWarnings(True)
    return log_path
