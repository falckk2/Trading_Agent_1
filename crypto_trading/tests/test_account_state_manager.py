"""
Unit tests for account state management.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from decimal import Decimal

from crypto_trading.core.account_state_manager import AccountStateManager
from crypto_trading.core.interfaces import (
    Order, Position, OrderStatus, OrderSide, OrderType,
    Event, EventType
)
from crypto_trading.core.exceptions import TradingSystemError


class TestAccountStateManager:
    """Tests for AccountStateManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.account_data_provider = Mock()
        self.account_data_provider.get_balance = AsyncMock()
        self.account_data_provider.get_positions = AsyncMock()
        self.account_data_provider.get_order_status = AsyncMock()

        self.event_bus = Mock()
        self.event_bus.publish = AsyncMock()

        self.manager = AccountStateManager(
            self.account_data_provider,
            self.event_bus
        )

    @pytest.mark.asyncio
    async def test_update_account_info_success(self):
        """Test successful account info update."""
        # Setup mocks
        mock_balance = {"USD": Decimal("10000"), "BTC": Decimal("1.5")}
        mock_positions = [
            Position(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                amount=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                pnl=Decimal("1000"),
                timestamp=datetime.now()
            )
        ]

        self.account_data_provider.get_balance.return_value = mock_balance
        self.account_data_provider.get_positions.return_value = mock_positions

        # Execute
        await self.manager.update_account_info()

        # Verify
        assert self.manager.get_balance() == mock_balance
        assert len(self.manager.get_positions()) == 1
        self.account_data_provider.get_balance.assert_called_once()
        self.account_data_provider.get_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_account_info_failure(self):
        """Test account info update failure."""
        # Setup mock to raise exception
        self.account_data_provider.get_balance.side_effect = Exception("Connection failed")

        # Execute and verify exception
        with pytest.raises(TradingSystemError, match="Account info update failed"):
            await self.manager.update_account_info()

    @pytest.mark.asyncio
    async def test_update_order_statuses_filled_order(self):
        """Test updating order statuses with filled order."""
        # Add an active order
        order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        self.manager.add_active_order(order)

        # Mock updated order as filled
        filled_order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now()
        )
        self.account_data_provider.get_order_status.return_value = filled_order

        # Update
        await self.manager._update_order_statuses()

        # Verify order was removed and event was published
        assert self.manager.get_active_order_count() == 0
        self.event_bus.publish.assert_called_once()

        # Verify the event type
        call_args = self.event_bus.publish.call_args[0][0]
        assert call_args.type == EventType.ORDER_FILLED

    @pytest.mark.asyncio
    async def test_update_order_statuses_cancelled_order(self):
        """Test updating order statuses with cancelled order."""
        # Add an active order
        order = Order(
            id="order_456",
            symbol="ETH-USD",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            amount=Decimal("5.0"),
            price=Decimal("3000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        self.manager.add_active_order(order)

        # Mock updated order as cancelled
        cancelled_order = Order(
            id="order_456",
            symbol="ETH-USD",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            amount=Decimal("5.0"),
            price=Decimal("3000"),
            status=OrderStatus.CANCELLED,
            timestamp=datetime.now()
        )
        self.account_data_provider.get_order_status.return_value = cancelled_order

        # Update
        await self.manager._update_order_statuses()

        # Verify order was removed and event was published
        assert self.manager.get_active_order_count() == 0
        self.event_bus.publish.assert_called_once()

        # Verify the event type
        call_args = self.event_bus.publish.call_args[0][0]
        assert call_args.type == EventType.ORDER_CANCELLED

    @pytest.mark.asyncio
    async def test_update_order_statuses_rejected_order(self):
        """Test updating order statuses with rejected order."""
        # Add an active order
        order = Order(
            id="order_789",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        self.manager.add_active_order(order)

        # Mock updated order as rejected
        rejected_order = Order(
            id="order_789",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.REJECTED,
            timestamp=datetime.now()
        )
        self.account_data_provider.get_order_status.return_value = rejected_order

        # Update
        await self.manager._update_order_statuses()

        # Verify order was removed (no event for rejected orders)
        assert self.manager.get_active_order_count() == 0

    def test_add_active_order(self):
        """Test adding an active order."""
        order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )

        self.manager.add_active_order(order)

        assert self.manager.get_active_order_count() == 1
        assert order in self.manager.get_active_orders()

    def test_add_active_order_without_id(self):
        """Test adding an order without ID (should not be added)."""
        order = Order(
            id="",  # Empty ID
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=Decimal("1.0"),
            price=None,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

        self.manager.add_active_order(order)

        # Should not be added
        assert self.manager.get_active_order_count() == 0

    def test_remove_active_order(self):
        """Test removing an active order."""
        order = Order(
            id="order_123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )

        self.manager.add_active_order(order)
        removed = self.manager.remove_active_order("order_123")

        assert removed == order
        assert self.manager.get_active_order_count() == 0

    def test_remove_nonexistent_order(self):
        """Test removing a non-existent order."""
        removed = self.manager.remove_active_order("nonexistent")
        assert removed is None

    def test_get_positions(self):
        """Test getting positions returns a copy."""
        # Set up positions via update
        positions = [
            Position(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                amount=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                pnl=Decimal("1000"),
                timestamp=datetime.now()
            )
        ]
        self.manager._positions = positions

        result = self.manager.get_positions()

        # Verify it's a copy, not the same list
        assert result == positions
        assert result is not positions

    def test_get_balance(self):
        """Test getting balance returns a copy."""
        balance = {"USD": Decimal("10000"), "BTC": Decimal("1.5")}
        self.manager._balance = balance

        result = self.manager.get_balance()

        # Verify it's a copy, not the same dict
        assert result == balance
        assert result is not balance

    def test_get_active_orders(self):
        """Test getting active orders list."""
        order1 = Order(
            id="order_1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        order2 = Order(
            id="order_2",
            symbol="ETH-USD",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            amount=Decimal("5.0"),
            price=Decimal("3000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )

        self.manager.add_active_order(order1)
        self.manager.add_active_order(order2)

        orders = self.manager.get_active_orders()

        assert len(orders) == 2
        assert order1 in orders
        assert order2 in orders

    def test_has_position_for_symbol_true(self):
        """Test checking for position when it exists."""
        position = Position(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            amount=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            pnl=Decimal("1000"),
            timestamp=datetime.now()
        )
        self.manager._positions = [position]

        assert self.manager.has_position_for_symbol("BTC-USD") is True

    def test_has_position_for_symbol_false(self):
        """Test checking for position when it doesn't exist."""
        assert self.manager.has_position_for_symbol("BTC-USD") is False

    def test_get_position_by_symbol_exists(self):
        """Test getting position by symbol when it exists."""
        position = Position(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            amount=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            pnl=Decimal("1000"),
            timestamp=datetime.now()
        )
        self.manager._positions = [position]

        result = self.manager.get_position_by_symbol("BTC-USD")

        assert result == position

    def test_get_position_by_symbol_not_exists(self):
        """Test getting position by symbol when it doesn't exist."""
        result = self.manager.get_position_by_symbol("BTC-USD")
        assert result is None

    def test_get_available_balance_exists(self):
        """Test getting available balance for existing asset."""
        self.manager._balance = {"USD": Decimal("10000")}

        balance = self.manager.get_available_balance("USD")

        assert balance == Decimal("10000")

    def test_get_available_balance_not_exists(self):
        """Test getting available balance for non-existent asset."""
        balance = self.manager.get_available_balance("USD")
        assert balance == Decimal("0")

    def test_clear_active_orders(self):
        """Test clearing all active orders."""
        order1 = Order(
            id="order_1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )
        order2 = Order(
            id="order_2",
            symbol="ETH-USD",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            amount=Decimal("5.0"),
            price=Decimal("3000"),
            status=OrderStatus.OPEN,
            timestamp=datetime.now()
        )

        self.manager.add_active_order(order1)
        self.manager.add_active_order(order2)

        self.manager.clear_active_orders()

        assert self.manager.get_active_order_count() == 0
