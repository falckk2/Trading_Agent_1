"""
Custom exceptions for the trading system.
"""


class TradingSystemError(Exception):
    """Base exception for trading system errors."""
    pass


class AgentError(TradingSystemError):
    """Base exception for agent-related errors."""
    pass


class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not found."""
    pass


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails."""
    pass


class ExchangeError(TradingSystemError):
    """Base exception for exchange-related errors."""
    pass


class ConnectionError(ExchangeError):
    """Raised when exchange connection fails."""
    pass


class OrderError(ExchangeError):
    """Raised when order operations fail."""
    pass


class DataError(TradingSystemError):
    """Base exception for data-related errors."""
    pass


class DataCollectionError(DataError):
    """Raised when data collection fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""
    pass


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid."""
    pass


class RiskManagementError(TradingSystemError):
    """Raised when risk management rules are violated."""
    pass


class SecurityError(TradingSystemError):
    """Raised when security-related operations fail."""
    pass