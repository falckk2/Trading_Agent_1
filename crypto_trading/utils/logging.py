"""
Logging adapters implementing ILogger interface.
Follows Dependency Inversion Principle by depending on abstractions.
"""

from typing import Optional
from ..core.interfaces import ILogger


class LoguruAdapter(ILogger):
    """Adapter for loguru logger implementing ILogger interface."""

    def __init__(self, logger_name: Optional[str] = None):
        """Initialize with optional logger name for context."""
        from loguru import logger
        self._logger = logger
        self._context = logger_name

    def _format_message(self, message: str) -> str:
        """Add context to message if available."""
        if self._context:
            return f"[{self._context}] {message}"
        return message

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(self._format_message(message))

    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(self._format_message(message))

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(self._format_message(message))

    def error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(self._format_message(message))

    def critical(self, message: str) -> None:
        """Log critical message."""
        self._logger.critical(self._format_message(message))


class StandardLibraryAdapter(ILogger):
    """Adapter for Python's standard logging library implementing ILogger interface."""

    def __init__(self, logger_name: str = __name__):
        """Initialize with logger name."""
        import logging
        self._logger = logging.getLogger(logger_name)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self._logger.critical(message)


class NullLogger(ILogger):
    """Null Object Pattern implementation for testing or when logging is disabled."""

    def debug(self, message: str) -> None:
        """No-op debug."""
        pass

    def info(self, message: str) -> None:
        """No-op info."""
        pass

    def warning(self, message: str) -> None:
        """No-op warning."""
        pass

    def error(self, message: str) -> None:
        """No-op error."""
        pass

    def critical(self, message: str) -> None:
        """No-op critical."""
        pass


# Default logger factory
def create_logger(logger_name: Optional[str] = None, use_loguru: bool = True) -> ILogger:
    """
    Factory function to create logger instances.

    Args:
        logger_name: Optional name for the logger (used for context)
        use_loguru: If True, use LoguruAdapter; otherwise use StandardLibraryAdapter

    Returns:
        ILogger implementation
    """
    if use_loguru:
        return LoguruAdapter(logger_name)
    else:
        return StandardLibraryAdapter(logger_name or __name__)
