"""
Test suite for core interfaces and data models.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from crypto_trading.core.interfaces import (
    MarketData, Order, Position, TradingSignal,
    OrderType, OrderSide, OrderStatus, EventType, Event
)


class TestDataModels:
    """Test data model classes."""

    def test_market_data_creation(self):
        """Test MarketData creation and validation."""
        market_data = MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49000.00"),
            close=Decimal("50500.00"),
            volume=Decimal("100.5"),
            bid=Decimal("50450.00"),
            ask=Decimal("50550.00")
        )

        assert market_data.symbol == "BTC/USDT"
        assert isinstance(market_data.open, Decimal)
        assert market_data.high >= market_data.low
        assert market_data.bid < market_data.ask

    def test_order_creation(self):
        """Test Order creation and validation."""
        order = Order(
            id="test_order_123",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("10.0"),
            price=Decimal("3000.00"),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

        assert order.id == "test_order_123"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.amount > 0
        assert order.price > 0

    def test_position_creation(self):
        """Test Position creation and validation."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            pnl=Decimal("1500.00"),
            timestamp=datetime.now()
        )

        assert position.symbol == "BTC/USDT"
        assert position.amount > 0
        assert position.pnl == Decimal("1500.00")

    def test_trading_signal_creation(self):
        """Test TradingSignal creation and validation."""
        signal = TradingSignal(
            symbol="ETH/USDT",
            action=OrderSide.SELL,
            confidence=0.85,
            price=Decimal("3100.00"),
            amount=Decimal("5.0"),
            timestamp=datetime.now(),
            metadata={"indicator": "RSI", "value": 75}
        )

        assert signal.symbol == "ETH/USDT"
        assert signal.action == OrderSide.SELL
        assert 0 <= signal.confidence <= 1
        assert signal.metadata["indicator"] == "RSI"

    def test_event_creation(self):
        """Test Event creation and validation."""
        event = Event(
            type=EventType.ORDER_FILLED,
            data={"order_id": "123", "amount": "10.0"},
            timestamp=datetime.now()
        )

        assert event.type == EventType.ORDER_FILLED
        assert "order_id" in event.data
        assert isinstance(event.timestamp, datetime)


class TestEnums:
    """Test enum classes."""

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"

    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.ORDER_FILLED.value == "order_filled"
        assert EventType.ORDER_CANCELLED.value == "order_cancelled"
        assert EventType.SIGNAL_GENERATED.value == "signal_generated"
        assert EventType.ERROR_OCCURRED.value == "error_occurred"


if __name__ == "__main__":
    pytest.main([__file__])