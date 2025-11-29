"""
Account state management extracted from TradingEngine.
Implements Single Responsibility Principle for account data management.
"""

from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from loguru import logger

from .interfaces import (
    IAccountDataProvider, IEventBus, Order, Position, OrderStatus,
    Event, EventType
)
from .exceptions import TradingSystemError


class AccountStateManager:
    """
    Manages account state including positions, balance, and active orders.
    Follows Single Responsibility Principle - only responsible for account state.
    """

    def __init__(
        self,
        account_data_provider: IAccountDataProvider,
        event_bus: IEventBus
    ):
        self.account_data_provider = account_data_provider
        self.event_bus = event_bus

        self._positions: List[Position] = []
        self._active_orders: Dict[str, Order] = {}
        self._balance: Dict[str, Decimal] = {}

    async def update_account_info(self) -> None:
        """
        Update all account information.
        Fetches balance, positions, and order statuses.
        """
        try:
            # Update balance
            self._balance = await self.account_data_provider.get_balance()

            # Update positions
            self._positions = await self.account_data_provider.get_positions()

            # Update active orders status
            await self._update_order_statuses()

        except Exception as e:
            logger.error(f"Failed to update account info: {e}")
            raise TradingSystemError(f"Account info update failed: {e}")

    async def _update_order_statuses(self) -> None:
        """Update status of all active orders."""
        orders_to_remove = []

        for order_id, order in self._active_orders.items():
            try:
                updated_order = await self.account_data_provider.get_order_status(order_id)
                self._active_orders[order_id] = updated_order

                # Remove completed orders
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    orders_to_remove.append(order_id)

                    if updated_order.status == OrderStatus.FILLED:
                        await self.event_bus.publish(
                            Event(
                                type=EventType.ORDER_FILLED,
                                data={"order": updated_order},
                                timestamp=datetime.now()
                            )
                        )
                    elif updated_order.status == OrderStatus.CANCELLED:
                        await self.event_bus.publish(
                            Event(
                                type=EventType.ORDER_CANCELLED,
                                data={"order": updated_order},
                                timestamp=datetime.now()
                            )
                        )

            except Exception as e:
                logger.error(f"Failed to update order status for {order_id}: {e}")

        # Remove completed orders
        for order_id in orders_to_remove:
            del self._active_orders[order_id]

    def add_active_order(self, order: Order) -> None:
        """Add an order to the active orders list."""
        if order.id:
            self._active_orders[order.id] = order
            logger.debug(f"Added active order: {order.id}")

    def remove_active_order(self, order_id: str) -> Optional[Order]:
        """Remove an order from the active orders list."""
        return self._active_orders.pop(order_id, None)

    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return self._positions.copy()

    def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        return self._balance.copy()

    def get_active_orders(self) -> List[Order]:
        """Get list of active orders."""
        return list(self._active_orders.values())

    def get_active_order_count(self) -> int:
        """Get count of active orders."""
        return len(self._active_orders)

    def has_position_for_symbol(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return any(pos.symbol == symbol for pos in self._positions)

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        for position in self._positions:
            if position.symbol == symbol:
                return position
        return None

    def get_available_balance(self, asset: str) -> Decimal:
        """Get available balance for a specific asset."""
        return self._balance.get(asset, Decimal(0))

    def clear_active_orders(self) -> None:
        """Clear all active orders."""
        self._active_orders.clear()
        logger.debug("Cleared all active orders")
