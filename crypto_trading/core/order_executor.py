"""
Order execution logic extracted from TradingEngine.
Implements Single Responsibility Principle for order execution.
"""

from typing import Dict, List
from datetime import datetime
from decimal import Decimal
from loguru import logger

from .interfaces import (
    IExchangeClient, IRiskManager, TradingSignal, Order, Position,
    OrderType, OrderStatus, OrderSide
)
from .exceptions import TradingSystemError, OrderError


class OrderExecutor:
    """
    Handles order execution logic.
    Follows Single Responsibility Principle - only responsible for executing orders.
    """

    def __init__(
        self,
        exchange_client: IExchangeClient,
        risk_manager: IRiskManager
    ):
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager

    async def execute_signal(
        self,
        signal: TradingSignal,
        balance: Dict[str, Decimal],
        positions: List[Position]
    ) -> Order:
        """
        Execute a trading signal.

        Args:
            signal: The trading signal to execute
            balance: Current account balance
            positions: Current positions

        Returns:
            The placed order

        Raises:
            TradingSystemError: If order execution fails
        """
        try:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(signal, balance)
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated: {position_size}")
                raise TradingSystemError("Invalid position size calculated")

            # Create order
            order = self._create_order_from_signal(signal, position_size)

            # Validate order with risk manager
            if not self.risk_manager.validate_order(order, positions):
                logger.warning(f"Order rejected by risk manager: {order}")
                raise TradingSystemError("Order rejected by risk manager")

            # Place order
            placed_order = await self.exchange_client.place_order(order)
            logger.info(f"Order placed: {placed_order.id} - {placed_order.symbol} {placed_order.side.value} {placed_order.amount}")

            return placed_order

        except TradingSystemError:
            raise
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            raise TradingSystemError(f"Signal execution failed: {e}")

    def _create_order_from_signal(self, signal: TradingSignal, position_size: Decimal) -> Order:
        """
        Create an order from a trading signal.

        Args:
            signal: The trading signal
            position_size: Calculated position size

        Returns:
            The created order
        """
        return Order(
            id="",  # Will be set by exchange
            symbol=signal.symbol,
            side=signal.action,
            type=OrderType.MARKET if signal.price is None else OrderType.LIMIT,
            amount=position_size,
            price=signal.price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )


class SignalExecutionStrategy:
    """
    Strategy pattern for different signal execution approaches.
    Allows for flexible execution strategies (e.g., aggressive, conservative).
    """

    async def execute(
        self,
        signal: TradingSignal,
        executor: OrderExecutor,
        balance: Dict[str, Decimal],
        positions: List[Position]
    ) -> Order:
        """Execute signal using specific strategy."""
        raise NotImplementedError


class ImmediateExecutionStrategy(SignalExecutionStrategy):
    """Execute signals immediately at market price."""

    async def execute(
        self,
        signal: TradingSignal,
        executor: OrderExecutor,
        balance: Dict[str, Decimal],
        positions: List[Position]
    ) -> Order:
        """Execute signal immediately."""
        return await executor.execute_signal(signal, balance, positions)


class ConservativeExecutionStrategy(SignalExecutionStrategy):
    """Execute signals conservatively with additional checks."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    async def execute(
        self,
        signal: TradingSignal,
        executor: OrderExecutor,
        balance: Dict[str, Decimal],
        positions: List[Position]
    ) -> Order:
        """Execute signal only if confidence threshold is met."""
        if signal.confidence < self.min_confidence:
            logger.warning(f"Signal confidence {signal.confidence} below threshold {self.min_confidence}")
            raise TradingSystemError("Signal confidence below threshold")

        return await executor.execute_signal(signal, balance, positions)
