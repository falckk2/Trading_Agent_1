"""
Custom exceptions for the trading system.
Provides clear error hierarchy for different types of failures.
"""


class TradingException(Exception):
    """Base exception for all trading-related errors."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ExchangeException(TradingException):
    """Exception for exchange-related errors."""
    pass


class ExchangeConnectionError(ExchangeException):
    """Exception for exchange connection failures."""
    pass


class ExchangeAPIError(ExchangeException):
    """Exception for exchange API errors."""
    pass


class OrderException(TradingException):
    """Exception for order-related errors."""
    pass


class InsufficientFundsError(OrderException):
    """Exception for insufficient funds errors."""
    pass


class InvalidOrderError(OrderException):
    """Exception for invalid order parameters."""
    pass


class DataException(TradingException):
    """Exception for data-related errors."""
    pass


class DataProviderError(DataException):
    """Exception for data provider failures."""
    pass


class DataValidationError(DataException):
    """Exception for data validation failures."""
    pass


class StrategyException(TradingException):
    """Exception for strategy-related errors."""
    pass


class StrategyInitializationError(StrategyException):
    """Exception for strategy initialization failures."""
    pass


class StrategyExecutionError(StrategyException):
    """Exception for strategy execution errors."""
    pass


class RiskManagementException(TradingException):
    """Exception for risk management violations."""
    pass


class RiskLimitExceeded(RiskManagementException):
    """Exception for risk limit violations."""
    pass


class PositionSizeError(RiskManagementException):
    """Exception for position size calculation errors."""
    pass


class ConfigurationException(TradingException):
    """Exception for configuration-related errors."""
    pass


class ValidationException(TradingException):
    """Exception for validation errors."""
    pass


class BacktestException(TradingException):
    """Exception for backtesting errors."""
    pass


class AgentException(TradingException):
    """Exception for trading agent errors."""
    pass


class AgentInitializationError(AgentException):
    """Exception for agent initialization failures."""
    pass


class AgentExecutionError(AgentException):
    """Exception for agent execution errors."""
    pass


class AgentNotFoundError(AgentException):
    """Exception raised when a requested agent is not found."""
    pass


# Aliases for backward compatibility with utils.exceptions
TradingSystemError = TradingException
ExchangeError = ExchangeException
ConnectionError = ExchangeConnectionError
OrderError = OrderException
DataError = DataException
DataCollectionError = DataException
DataProcessingError = DataException
ConfigurationError = ConfigurationException
RiskManagementError = RiskManagementException
SecurityError = TradingException
AgentError = AgentException