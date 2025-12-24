"""
Validation utilities for reducing code duplication.
Provides reusable validators for orders, signals, and other trading entities.
"""

from typing import List, Optional
from decimal import Decimal
from loguru import logger

from ..core.interfaces import Order, TradingSignal, Position, OrderType, OrderSide, OrderStatus
from ..core.models import MarketData


class OrderValidator:
    """Validator for trading orders."""

    @staticmethod
    def validate_order_basic(order: Order) -> List[str]:
        """
        Perform basic order validation.

        Args:
            order: Order to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not order:
            errors.append("Order cannot be None")
            return errors

        if not order.symbol or not order.symbol.strip():
            errors.append("Order must have a valid symbol")

        if order.amount is None or order.amount <= 0:
            errors.append("Order amount must be positive")

        if order.price is not None and order.price <= 0:
            errors.append("Order price must be positive")

        if not isinstance(order.side, OrderSide):
            errors.append("Order must have a valid side (BUY/SELL)")

        if not isinstance(order.type, OrderType):
            errors.append("Order must have a valid type")

        return errors

    @staticmethod
    def is_valid_order(order: Order) -> bool:
        """
        Check if order is valid.

        Args:
            order: Order to validate

        Returns:
            True if valid, False otherwise
        """
        errors = OrderValidator.validate_order_basic(order)
        return len(errors) == 0

    @staticmethod
    def validate_or_raise(order: Order) -> None:
        """
        Validate order and raise ValueError if invalid.

        Args:
            order: Order to validate

        Raises:
            ValueError: If order is invalid
        """
        errors = OrderValidator.validate_order_basic(order)
        if errors:
            raise ValueError(f"Order validation failed: {'; '.join(errors)}")

    @staticmethod
    def validate_order_value(order: Order, min_value: Decimal) -> bool:
        """
        Validate order meets minimum value requirement.

        Args:
            order: Order to validate
            min_value: Minimum order value

        Returns:
            True if order value >= min_value
        """
        if order.price is None:
            return False

        order_value = order.amount * order.price
        return order_value >= min_value

    @staticmethod
    def validate_order_quantity(order: Order, min_qty: Decimal) -> bool:
        """
        Validate order meets minimum quantity requirement.

        Args:
            order: Order to validate
            min_qty: Minimum quantity

        Returns:
            True if order quantity >= min_qty
        """
        return order.amount >= min_qty


class SignalValidator:
    """Validator for trading signals."""

    @staticmethod
    def validate_signal(signal: TradingSignal) -> List[str]:
        """
        Validate trading signal.

        Args:
            signal: Signal to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not signal:
            errors.append("Signal cannot be None")
            return errors

        if not signal.symbol or not signal.symbol.strip():
            errors.append("Signal must have a valid symbol")

        if signal.confidence < 0 or signal.confidence > 1:
            errors.append("Signal confidence must be between 0 and 1")

        if signal.price is not None and signal.price <= 0:
            errors.append("Signal price must be positive")

        if not isinstance(signal.action, OrderSide):
            errors.append("Signal must have a valid action (BUY/SELL)")

        return errors

    @staticmethod
    def is_valid_signal(signal: TradingSignal) -> bool:
        """
        Check if signal is valid.

        Args:
            signal: Signal to validate

        Returns:
            True if valid, False otherwise
        """
        errors = SignalValidator.validate_signal(signal)
        return len(errors) == 0

    @staticmethod
    def meets_confidence_threshold(signal: TradingSignal, threshold: float) -> bool:
        """
        Check if signal meets confidence threshold.

        Args:
            signal: Signal to check
            threshold: Minimum confidence (0-1)

        Returns:
            True if signal confidence >= threshold
        """
        return signal.confidence >= threshold


class PositionValidator:
    """Validator for positions."""

    @staticmethod
    def validate_position(position: Position) -> List[str]:
        """
        Validate position.

        Args:
            position: Position to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not position:
            errors.append("Position cannot be None")
            return errors

        if not position.symbol or not position.symbol.strip():
            errors.append("Position must have a valid symbol")

        if position.amount <= 0:
            errors.append("Position amount must be positive")

        if position.entry_price <= 0:
            errors.append("Position entry price must be positive")

        if position.current_price is not None and position.current_price <= 0:
            errors.append("Position current price must be positive")

        return errors

    @staticmethod
    def is_profitable(position: Position) -> bool:
        """
        Check if position is currently profitable.

        Args:
            position: Position to check

        Returns:
            True if position has positive PnL
        """
        return position.pnl > 0 if position.pnl is not None else False


class MarketDataValidatorUtil:
    """Validator for market data (extends MarketDataValidator from data_utils)."""

    @staticmethod
    def validate_price_range(
        data: List[MarketData],
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None
    ) -> List[str]:
        """
        Validate prices are within expected range.

        Args:
            data: Market data to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price

        Returns:
            List of validation warnings
        """
        warnings = []

        for i, md in enumerate(data):
            if min_price and md.close < min_price:
                warnings.append(f"Price below minimum at index {i}: {md.close} < {min_price}")

            if max_price and md.close > max_price:
                warnings.append(f"Price above maximum at index {i}: {md.close} > {max_price}")

        return warnings

    @staticmethod
    def check_data_continuity(data: List[MarketData], max_gap_seconds: int = 3600) -> List[str]:
        """
        Check for gaps in time series data.

        Args:
            data: Market data to check
            max_gap_seconds: Maximum allowed gap in seconds

        Returns:
            List of gap warnings
        """
        if len(data) < 2:
            return []

        warnings = []
        for i in range(1, len(data)):
            gap = (data[i].timestamp - data[i - 1].timestamp).total_seconds()
            if gap > max_gap_seconds:
                warnings.append(
                    f"Large gap detected between index {i - 1} and {i}: "
                    f"{gap} seconds"
                )

        return warnings


class ConfigValidator:
    """Validator for configuration dictionaries."""

    @staticmethod
    def validate_required_keys(config: dict, required_keys: List[str]) -> List[str]:
        """
        Validate that all required keys are present in config.

        Args:
            config: Configuration dictionary
            required_keys: List of required keys

        Returns:
            List of missing keys
        """
        missing = []
        for key in required_keys:
            if key not in config:
                missing.append(key)

        return missing

    @staticmethod
    def validate_numeric_range(
        config: dict,
        key: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Optional[str]:
        """
        Validate numeric config value is within range.

        Args:
            config: Configuration dictionary
            key: Key to validate
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value

        Returns:
            Error message if invalid, None if valid
        """
        if key not in config:
            return f"Missing config key: {key}"

        value = config[key]

        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return f"Config key '{key}' must be numeric, got {type(value)}"

        if min_value is not None and num_value < min_value:
            return f"Config key '{key}' must be >= {min_value}, got {num_value}"

        if max_value is not None and num_value > max_value:
            return f"Config key '{key}' must be <= {max_value}, got {num_value}"

        return None


# ==================== Parameter Validators for SOLID Principles ====================
# Following Single Responsibility Principle - each validator has one validation purpose

from abc import ABC, abstractmethod
from typing import Any, Callable


class ValidationError(Exception):
    """Exception raised when parameter validation fails."""
    pass


class ParameterValidator(ABC):
    """Abstract base class for parameter validators."""

    def __init__(self, param_name: str, error_message: Optional[str] = None):
        """
        Initialize validator.

        Args:
            param_name: Name of the parameter being validated
            error_message: Custom error message (optional)
        """
        self.param_name = param_name
        self.error_message = error_message

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """
        Validate a value.

        Args:
            value: Value to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        pass

    def _format_error(self, detail: str) -> str:
        """Format error message."""
        if self.error_message:
            return f"{self.error_message}: {detail}"
        return f"Validation failed for '{self.param_name}': {detail}"


class RangeValidator(ParameterValidator):
    """Validates that a numeric value is within a specified range."""

    def __init__(
        self,
        param_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Initialize range validator.

        Args:
            param_name: Parameter name
            min_value: Minimum allowed value (None = no minimum)
            max_value: Maximum allowed value (None = no maximum)
            inclusive: Whether range endpoints are inclusive
            error_message: Custom error message
        """
        super().__init__(param_name, error_message)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, value: Any) -> bool:
        """Validate value is in range."""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                self._format_error(f"Must be numeric, got {type(value).__name__}")
            )

        if self.min_value is not None:
            if self.inclusive and num_value < self.min_value:
                raise ValidationError(
                    self._format_error(f"Must be >= {self.min_value}, got {num_value}")
                )
            elif not self.inclusive and num_value <= self.min_value:
                raise ValidationError(
                    self._format_error(f"Must be > {self.min_value}, got {num_value}")
                )

        if self.max_value is not None:
            if self.inclusive and num_value > self.max_value:
                raise ValidationError(
                    self._format_error(f"Must be <= {self.max_value}, got {num_value}")
                )
            elif not self.inclusive and num_value >= self.max_value:
                raise ValidationError(
                    self._format_error(f"Must be < {self.max_value}, got {num_value}")
                )

        return True


class TypeValidator(ParameterValidator):
    """Validates that a value is of the expected type."""

    def __init__(
        self,
        param_name: str,
        expected_type: type,
        error_message: Optional[str] = None
    ):
        """
        Initialize type validator.

        Args:
            param_name: Parameter name
            expected_type: Expected type
            error_message: Custom error message
        """
        super().__init__(param_name, error_message)
        self.expected_type = expected_type

    def validate(self, value: Any) -> bool:
        """Validate value type."""
        if not isinstance(value, self.expected_type):
            raise ValidationError(
                self._format_error(
                    f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
                )
            )
        return True


class ChoiceValidator(ParameterValidator):
    """Validates that a value is one of allowed choices."""

    def __init__(
        self,
        param_name: str,
        choices: List[Any],
        case_sensitive: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Initialize choice validator.

        Args:
            param_name: Parameter name
            choices: List of allowed choices
            case_sensitive: Whether string comparison is case-sensitive
            error_message: Custom error message
        """
        super().__init__(param_name, error_message)
        self.choices = choices
        self.case_sensitive = case_sensitive

    def validate(self, value: Any) -> bool:
        """Validate value is in choices."""
        if self.case_sensitive:
            if value not in self.choices:
                raise ValidationError(
                    self._format_error(
                        f"Must be one of {self.choices}, got '{value}'"
                    )
                )
        else:
            # Case-insensitive comparison for strings
            if isinstance(value, str):
                if value.lower() not in [str(c).lower() for c in self.choices]:
                    raise ValidationError(
                        self._format_error(
                            f"Must be one of {self.choices}, got '{value}'"
                        )
                    )
            elif value not in self.choices:
                raise ValidationError(
                    self._format_error(
                        f"Must be one of {self.choices}, got '{value}'"
                    )
                )
        return True


class RequiredValidator(ParameterValidator):
    """Validates that a value is not None."""

    def validate(self, value: Any) -> bool:
        """Validate value is not None."""
        if value is None:
            raise ValidationError(
                self._format_error("Parameter is required")
            )
        return True


class PositiveValidator(ParameterValidator):
    """Validates that a numeric value is positive."""

    def __init__(
        self,
        param_name: str,
        allow_zero: bool = False,
        error_message: Optional[str] = None
    ):
        """
        Initialize positive validator.

        Args:
            param_name: Parameter name
            allow_zero: Whether zero is allowed
            error_message: Custom error message
        """
        super().__init__(param_name, error_message)
        self.allow_zero = allow_zero

    def validate(self, value: Any) -> bool:
        """Validate value is positive."""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                self._format_error(f"Must be numeric, got {type(value).__name__}")
            )

        if self.allow_zero:
            if num_value < 0:
                raise ValidationError(
                    self._format_error(f"Must be >= 0, got {num_value}")
                )
        else:
            if num_value <= 0:
                raise ValidationError(
                    self._format_error(f"Must be > 0, got {num_value}")
                )

        return True


class CustomValidator(ParameterValidator):
    """Validator using a custom validation function."""

    def __init__(
        self,
        param_name: str,
        validation_func: Callable[[Any], bool],
        error_message: Optional[str] = None
    ):
        """
        Initialize custom validator.

        Args:
            param_name: Parameter name
            validation_func: Function that returns True if valid, False otherwise
            error_message: Custom error message
        """
        super().__init__(param_name, error_message)
        self.validation_func = validation_func

    def validate(self, value: Any) -> bool:
        """Validate using custom function."""
        if not self.validation_func(value):
            raise ValidationError(
                self._format_error(f"Custom validation failed for value: {value}")
            )
        return True


# Convenience functions for common validation scenarios
def validate_ma_parameters(params: dict) -> bool:
    """
    Validate moving average parameters.

    Args:
        params: Parameters to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    # Validate fast_period
    if 'fast_period' in params:
        PositiveValidator('fast_period').validate(params['fast_period'])
        RangeValidator('fast_period', min_value=1, max_value=200).validate(params['fast_period'])

    # Validate slow_period
    if 'slow_period' in params:
        PositiveValidator('slow_period').validate(params['slow_period'])
        RangeValidator('slow_period', min_value=1, max_value=200).validate(params['slow_period'])

    # Validate ma_type
    if 'ma_type' in params:
        ChoiceValidator(
            'ma_type',
            ['sma', 'ema', 'wma', 'hma'],
            case_sensitive=False
        ).validate(params['ma_type'])

    # Validate that fast < slow
    if 'fast_period' in params and 'slow_period' in params:
        if params['fast_period'] >= params['slow_period']:
            raise ValidationError(
                "fast_period must be less than slow_period"
            )

    return True


def validate_rsi_parameters(params: dict) -> bool:
    """
    Validate RSI parameters.

    Args:
        params: Parameters to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if 'period' in params:
        PositiveValidator('period').validate(params['period'])
        RangeValidator('period', min_value=2, max_value=100).validate(params['period'])

    if 'overbought' in params:
        RangeValidator('overbought', min_value=50, max_value=100).validate(params['overbought'])

    if 'oversold' in params:
        RangeValidator('oversold', min_value=0, max_value=50).validate(params['oversold'])

    return True
