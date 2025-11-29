"""
Unit tests for order execution logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime
from decimal import Decimal

from crypto_trading.core.order_executor import (
    OrderExecutor,
    ImmediateExecutionStrategy,
    ConservativeExecutionStrategy
)
from crypto_trading.core.interfaces import (
    TradingSignal, Order, Position,
    OrderType, OrderStatus, OrderSide
)
from crypto_trading.core.exceptions import TradingSystemError


class TestOrderExecutor:
    """Tests for OrderExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.exchange_client = Mock()
        self.exchange_client.place_order = AsyncMock()
        self.risk_manager = Mock()
        self.executor = OrderExecutor(self.exchange_client, self.risk_manager)

    @pytest.mark.asyncio
    async def test_execute_signal_success(self):
        """Test successful signal execution."""
        # Setup
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock risk manager responses
        self.risk_manager.calculate_position_size.return_value = Decimal("0.1")
        self.risk_manager.validate_order.return_value = True

        # Mock exchange response
        placed_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        self.exchange_client.place_order.return_value = placed_order

        # Execute
        result = await self.executor.execute_signal(signal, balance, positions)

        # Verify
        assert result == placed_order
        self.risk_manager.calculate_position_size.assert_called_once_with(signal, balance)
        self.risk_manager.validate_order.assert_called_once()
        self.exchange_client.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_invalid_position_size(self):
        """Test signal execution with invalid position size."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10")}
        positions = []

        # Mock invalid position size
        self.risk_manager.calculate_position_size.return_value = Decimal("0")

        # Execute and verify exception
        with pytest.raises(TradingSystemError, match="Invalid position size"):
            await self.executor.execute_signal(signal, balance, positions)

    @pytest.mark.asyncio
    async def test_execute_signal_risk_manager_rejection(self):
        """Test signal execution rejected by risk manager."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock risk manager responses
        self.risk_manager.calculate_position_size.return_value = Decimal("0.1")
        self.risk_manager.validate_order.return_value = False  # Reject order

        # Execute and verify exception
        with pytest.raises(TradingSystemError, match="Order rejected by risk manager"):
            await self.executor.execute_signal(signal, balance, positions)

    @pytest.mark.asyncio
    async def test_execute_signal_exchange_error(self):
        """Test signal execution with exchange error."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock risk manager responses
        self.risk_manager.calculate_position_size.return_value = Decimal("0.1")
        self.risk_manager.validate_order.return_value = True

        # Mock exchange error
        self.exchange_client.place_order.side_effect = Exception("Exchange connection failed")

        # Execute and verify exception
        with pytest.raises(TradingSystemError, match="Signal execution failed"):
            await self.executor.execute_signal(signal, balance, positions)

    def test_create_order_from_signal_with_price(self):
        """Test order creation from signal with limit price."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        order = self.executor._create_order_from_signal(signal, Decimal("0.1"))

        assert order.symbol == "BTC-USD"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.amount == Decimal("0.1")
        assert order.price == Decimal("50000")
        assert order.status == OrderStatus.PENDING

    def test_create_order_from_signal_without_price(self):
        """Test order creation from signal without price (market order)."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.SELL,
            confidence=0.9,
            price=None,
            amount=Decimal("0.2"),
            timestamp=datetime.now(),
            metadata={}
        )

        order = self.executor._create_order_from_signal(signal, Decimal("0.2"))

        assert order.symbol == "BTC-USD"
        assert order.side == OrderSide.SELL
        assert order.type == OrderType.MARKET
        assert order.amount == Decimal("0.2")
        assert order.price is None
        assert order.status == OrderStatus.PENDING


class TestImmediateExecutionStrategy:
    """Tests for ImmediateExecutionStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ImmediateExecutionStrategy()
        self.executor = Mock()
        self.executor.execute_signal = AsyncMock()

    @pytest.mark.asyncio
    async def test_execute_immediate_strategy(self):
        """Test immediate execution strategy."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock executor response
        mock_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        self.executor.execute_signal.return_value = mock_order

        # Execute
        result = await self.strategy.execute(signal, self.executor, balance, positions)

        # Verify
        assert result == mock_order
        self.executor.execute_signal.assert_called_once_with(signal, balance, positions)

    @pytest.mark.asyncio
    async def test_execute_immediate_strategy_passes_through_exceptions(self):
        """Test that immediate strategy passes through executor exceptions."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock executor to raise exception
        self.executor.execute_signal.side_effect = TradingSystemError("Execution failed")

        # Execute and verify exception passes through
        with pytest.raises(TradingSystemError, match="Execution failed"):
            await self.strategy.execute(signal, self.executor, balance, positions)


class TestConservativeExecutionStrategy:
    """Tests for ConservativeExecutionStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ConservativeExecutionStrategy(min_confidence=0.7)
        self.executor = Mock()
        self.executor.execute_signal = AsyncMock()

    @pytest.mark.asyncio
    async def test_execute_conservative_strategy_above_threshold(self):
        """Test conservative execution with confidence above threshold."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.8,  # Above 0.7 threshold
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock executor response
        mock_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        self.executor.execute_signal.return_value = mock_order

        # Execute
        result = await self.strategy.execute(signal, self.executor, balance, positions)

        # Verify
        assert result == mock_order
        self.executor.execute_signal.assert_called_once_with(signal, balance, positions)

    @pytest.mark.asyncio
    async def test_execute_conservative_strategy_below_threshold(self):
        """Test conservative execution with confidence below threshold."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.5,  # Below 0.7 threshold
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Execute and verify rejection
        with pytest.raises(TradingSystemError, match="confidence below threshold"):
            await self.strategy.execute(signal, self.executor, balance, positions)

        # Verify executor was never called
        self.executor.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_conservative_strategy_at_threshold(self):
        """Test conservative execution with confidence exactly at threshold."""
        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.7,  # Exactly at 0.7 threshold
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Mock executor response
        mock_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        self.executor.execute_signal.return_value = mock_order

        # Execute (should succeed at threshold)
        result = await self.strategy.execute(signal, self.executor, balance, positions)

        # Verify
        assert result == mock_order
        self.executor.execute_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_conservative_strategy_custom_threshold(self):
        """Test conservative execution with custom confidence threshold."""
        # Create strategy with custom threshold
        strategy = ConservativeExecutionStrategy(min_confidence=0.9)

        signal = TradingSignal(
            symbol="BTC-USD",
            action=OrderSide.BUY,
            confidence=0.85,  # Below 0.9 threshold
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            timestamp=datetime.now(),
            metadata={}
        )

        balance = {"USD": Decimal("10000")}
        positions = []

        # Execute and verify rejection
        with pytest.raises(TradingSystemError, match="confidence below threshold"):
            await strategy.execute(signal, self.executor, balance, positions)
