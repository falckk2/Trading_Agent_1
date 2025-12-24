"""
Risk validators using Chain of Responsibility pattern.
Follows Open/Closed Principle - new validators can be added without modifying existing code.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from ..core.interfaces import ILogger, Order, Position
from ..utils.logging import create_logger


class RiskValidator(ABC):
    """
    Abstract base class for risk validators using Chain of Responsibility pattern.
    Each validator handles one specific risk check.
    """

    def __init__(self, logger: Optional[ILogger] = None):
        """Initialize validator with optional logger."""
        self._next_validator: Optional[RiskValidator] = None
        self.logger = logger or create_logger(self.__class__.__name__)

    def set_next(self, validator: 'RiskValidator') -> 'RiskValidator':
        """
        Set the next validator in the chain.

        Args:
            validator: Next validator to call

        Returns:
            The validator that was set (for chaining)
        """
        self._next_validator = validator
        return validator

    def validate(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """
        Validate order and pass to next validator if successful.

        Args:
            order: Order to validate
            positions: Current positions
            context: Additional context (config, daily losses, etc.)

        Returns:
            True if validation passes, False otherwise
        """
        # Perform this validator's check
        if not self._check(order, positions, context):
            return False

        # Pass to next validator if exists
        if self._next_validator:
            return self._next_validator.validate(order, positions, context)

        # All validators passed
        return True

    @abstractmethod
    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """
        Perform the specific validation check.

        Args:
            order: Order to validate
            positions: Current positions
            context: Additional context

        Returns:
            True if check passes, False otherwise
        """
        pass


class OrderBasicsValidator(RiskValidator):
    """Validates basic order properties (amount, price, symbol)."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check basic order validity."""
        if order.amount <= 0:
            self.logger.warning("Order rejected: Invalid amount")
            return False

        if order.price is not None and order.price <= 0:
            self.logger.warning("Order rejected: Invalid price")
            return False

        if not order.symbol:
            self.logger.warning("Order rejected: Missing symbol")
            return False

        return True


class DailyLossLimitValidator(RiskValidator):
    """Validates that daily loss limit is not exceeded."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check daily loss limit."""
        daily_losses = context.get('daily_losses', {})
        config = context.get('config', {})

        today = datetime.now().date().isoformat()
        daily_loss = daily_losses.get(today, Decimal('0'))

        max_daily_loss_pct = Decimal(str(config.get("max_daily_loss_pct", 0.05)))
        portfolio_value = Decimal(str(config.get("portfolio_value", 10000)))

        max_daily_loss = portfolio_value * max_daily_loss_pct

        if daily_loss >= max_daily_loss:
            self.logger.warning("Order rejected: Daily loss limit exceeded")
            return False

        return True


class PositionLimitValidator(RiskValidator):
    """Validates position count limits."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check position limits."""
        config = context.get('config', {})

        max_positions_per_symbol = config.get("max_positions_per_symbol", 2)
        max_total_positions = config.get("max_total_positions", 10)

        # Count existing positions for this symbol with the same side
        symbol_positions = [p for p in positions if p.symbol == order.symbol and p.side == order.side]

        # If we already have a position in the same direction, this order extends it
        opposing_positions = [p for p in positions if p.symbol == order.symbol and p.side != order.side]

        # Check if adding this order would exceed symbol limits
        total_symbol_positions = len(symbol_positions) + len(opposing_positions)
        if len(symbol_positions) == 0 and total_symbol_positions >= max_positions_per_symbol:
            self.logger.warning("Order rejected: Position limits per symbol exceeded")
            return False

        # Check total position count - only count if this creates a new position
        if len(symbol_positions) == 0 and len(positions) >= max_total_positions:
            self.logger.warning("Order rejected: Total position limit exceeded")
            return False

        return True


class ExposureLimitValidator(RiskValidator):
    """Validates total exposure limits."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check exposure limits."""
        config = context.get('config', {})

        max_exposure_pct = Decimal(str(config.get("max_total_exposure_pct", 0.8)))
        portfolio_value = Decimal(str(config.get("portfolio_value", 10000)))
        default_price = Decimal(str(config.get("default_price", 50000)))

        # Calculate current exposure
        current_exposure = sum(abs(p.amount * p.current_price) for p in positions)

        # Calculate new exposure
        order_price = order.price if order.price else default_price
        order_value = order.amount * order_price
        new_exposure = current_exposure + order_value

        max_exposure = portfolio_value * max_exposure_pct

        if new_exposure > max_exposure:
            self.logger.warning("Order rejected: Exposure limits exceeded")
            return False

        return True


class OrderSizeLimitValidator(RiskValidator):
    """Validates order size limits."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check order size limits."""
        config = context.get('config', {})

        min_order_size = Decimal(str(config.get("min_order_size", 10)))
        min_order_amount = Decimal(str(config.get("min_order_amount", 0.001)))
        default_price = Decimal(str(config.get("default_price", 50000)))

        # Calculate order value
        order_price = order.price or default_price
        order_value = order.amount * order_price

        # Check both minimum value and minimum amount
        if order_value < min_order_size:
            self.logger.warning(f"Order rejected: Order value {order_value} below minimum {min_order_size}")
            return False

        if order.amount < min_order_amount:
            self.logger.warning(f"Order rejected: Order amount {order.amount} below minimum {min_order_amount}")
            return False

        return True


class MaxPositionSizeValidator(RiskValidator):
    """Validates that a single position doesn't exceed max size."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check maximum position size."""
        config = context.get('config', {})

        max_position_size_pct = Decimal(str(config.get("max_position_size_pct", 0.1)))
        portfolio_value = Decimal(str(config.get("portfolio_value", 10000)))
        default_price = Decimal(str(config.get("default_price", 50000)))

        max_position_value = portfolio_value * max_position_size_pct

        # Calculate order value
        order_price = order.price or default_price
        order_value = order.amount * order_price

        if order_value > max_position_value:
            self.logger.warning(
                f"Order rejected: Position size {order_value} exceeds maximum {max_position_value}"
            )
            return False

        return True


class LeverageValidator(RiskValidator):
    """Validates leverage limits (for futures/margin trading)."""

    def _check(self, order: Order, positions: List[Position], context: Dict[str, Any]) -> bool:
        """Check leverage limits."""
        config = context.get('config', {})

        max_leverage = Decimal(str(config.get("max_leverage", 1.0)))

        # If order has leverage information, validate it
        order_leverage = getattr(order, 'leverage', None)
        if order_leverage is not None:
            if Decimal(str(order_leverage)) > max_leverage:
                self.logger.warning(
                    f"Order rejected: Leverage {order_leverage} exceeds maximum {max_leverage}"
                )
                return False

        return True


class RiskValidatorChain:
    """
    Builder class for creating risk validator chains.
    Simplifies creating and configuring validator chains.
    """

    def __init__(self, logger: Optional[ILogger] = None):
        """Initialize chain builder."""
        self.logger = logger
        self._validators: List[RiskValidator] = []

    def add_validator(self, validator: RiskValidator) -> 'RiskValidatorChain':
        """
        Add a validator to the chain.

        Args:
            validator: Validator to add

        Returns:
            Self for chaining
        """
        self._validators.append(validator)
        return self

    def add_standard_validators(self) -> 'RiskValidatorChain':
        """
        Add all standard validators in recommended order.

        Returns:
            Self for chaining
        """
        self.add_validator(OrderBasicsValidator(self.logger))
        self.add_validator(DailyLossLimitValidator(self.logger))
        self.add_validator(OrderSizeLimitValidator(self.logger))
        self.add_validator(MaxPositionSizeValidator(self.logger))
        self.add_validator(PositionLimitValidator(self.logger))
        self.add_validator(ExposureLimitValidator(self.logger))
        self.add_validator(LeverageValidator(self.logger))
        return self

    def build(self) -> Optional[RiskValidator]:
        """
        Build the validator chain.

        Returns:
            First validator in the chain, or None if no validators
        """
        if not self._validators:
            return None

        # Link validators together
        for i in range(len(self._validators) - 1):
            self._validators[i].set_next(self._validators[i + 1])

        # Return first validator
        return self._validators[0]

    @staticmethod
    def create_default_chain(logger: Optional[ILogger] = None) -> Optional[RiskValidator]:
        """
        Create default validator chain with all standard validators.

        Args:
            logger: Optional logger to use

        Returns:
            First validator in the chain
        """
        return (RiskValidatorChain(logger)
                .add_standard_validators()
                .build())
