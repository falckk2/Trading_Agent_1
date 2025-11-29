"""
Order management system for tracking order lifecycle and state.
Provides comprehensive order tracking, partial fills, and timeout management.
"""

import asyncio
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from loguru import logger

from .interfaces import (
    Order, OrderStatus, Position, IExchangeClient, IEventBus,
    Event, EventType
)
from .exceptions import OrderError, TradingSystemError
from .constants import DEFAULT_ORDER_TIMEOUT_SECONDS, MAX_ORDER_RETRIES


class OrderState(Enum):
    """Extended order states for internal tracking."""
    CREATED = "created"
    SUBMITTED = "submitted"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class ManagedOrder:
    """Enhanced order with lifecycle tracking."""

    def __init__(self, order: Order, timeout_seconds: int = DEFAULT_ORDER_TIMEOUT_SECONDS):
        self.order = order
        self.state = OrderState.CREATED
        self.created_at = datetime.now()
        self.submitted_at: Optional[datetime] = None
        self.filled_at: Optional[datetime] = None
        self.timeout_seconds = timeout_seconds
        self.fill_history: List[Dict] = []
        self.state_history: List[Dict] = []
        self.retry_count = 0
        self.max_retries = MAX_ORDER_RETRIES

        self._add_state_change(OrderState.CREATED)

    def _add_state_change(self, new_state: OrderState) -> None:
        """Record state change with timestamp."""
        self.state_history.append({
            'state': new_state,
            'timestamp': datetime.now(),
            'previous_state': self.state
        })
        self.state = new_state

    def update_state(self, new_state: OrderState) -> None:
        """Update order state with validation."""
        if self._is_valid_state_transition(self.state, new_state):
            self._add_state_change(new_state)
            logger.debug(f"Order {self.order.id} state changed: {self.state.value}")
        else:
            logger.warning(f"Invalid state transition for order {self.order.id}: {self.state.value} -> {new_state.value}")

    def _is_valid_state_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """Validate state transitions."""
        valid_transitions = {
            OrderState.CREATED: [OrderState.SUBMITTED, OrderState.FAILED],
            OrderState.SUBMITTED: [OrderState.PENDING, OrderState.REJECTED, OrderState.FAILED],
            OrderState.PENDING: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED],
            OrderState.PARTIALLY_FILLED: [OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED],
            OrderState.FILLED: [],  # Terminal state
            OrderState.CANCELLED: [],  # Terminal state
            OrderState.REJECTED: [],  # Terminal state
            OrderState.EXPIRED: [],  # Terminal state
            OrderState.FAILED: [OrderState.SUBMITTED]  # Can retry
        }

        return to_state in valid_transitions.get(from_state, [])

    def add_fill(self, fill_amount: Decimal, fill_price: Decimal) -> None:
        """Record a partial or full fill."""
        fill_data = {
            'amount': fill_amount,
            'price': fill_price,
            'timestamp': datetime.now(),
            'cumulative_filled': self.order.filled_amount + fill_amount
        }
        self.fill_history.append(fill_data)

        # Update order
        self.order.filled_amount += fill_amount
        if not self.order.average_price:
            self.order.average_price = fill_price
        else:
            # Calculate weighted average price
            total_value = (self.order.average_price * (self.order.filled_amount - fill_amount) +
                          fill_price * fill_amount)
            self.order.average_price = total_value / self.order.filled_amount

        # Update state
        if self.order.filled_amount >= self.order.amount:
            self.update_state(OrderState.FILLED)
            self.filled_at = datetime.now()
        else:
            self.update_state(OrderState.PARTIALLY_FILLED)

    def is_expired(self) -> bool:
        """Check if order has expired."""
        if self.submitted_at:
            return datetime.now() - self.submitted_at > timedelta(seconds=self.timeout_seconds)
        return datetime.now() - self.created_at > timedelta(seconds=self.timeout_seconds * 2)

    def is_terminal_state(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED]

    def can_retry(self) -> bool:
        """Check if order can be retried."""
        return self.state == OrderState.FAILED and self.retry_count < self.max_retries


class OrderManager:
    """Comprehensive order management system."""

    def __init__(self, exchange_client: IExchangeClient, event_bus: IEventBus):
        self.exchange_client = exchange_client
        self.event_bus = event_bus

        self._active_orders: Dict[str, ManagedOrder] = {}
        self._completed_orders: Dict[str, ManagedOrder] = {}
        self._order_callbacks: Dict[str, List[Callable]] = {}

        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Statistics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'orders_expired': 0,
            'total_fill_value': Decimal('0'),
            'average_fill_time': 0.0
        }

    async def start(self) -> None:
        """Start the order management system."""
        if self._is_running:
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitor_orders())
        logger.info("OrderManager started")

    async def stop(self) -> None:
        """Stop the order management system."""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Cancel any remaining active orders
        await self._cancel_all_active_orders()
        logger.info("OrderManager stopped")

    async def submit_order(self, order: Order, timeout_seconds: int = DEFAULT_ORDER_TIMEOUT_SECONDS) -> str:
        """Submit an order with lifecycle tracking."""
        managed_order = ManagedOrder(order, timeout_seconds)

        try:
            # Submit to exchange
            managed_order.update_state(OrderState.SUBMITTED)
            managed_order.submitted_at = datetime.now()

            submitted_order = await self.exchange_client.place_order(order)

            # Update order with exchange response
            managed_order.order = submitted_order
            managed_order.update_state(OrderState.PENDING)

            # Track the order
            self._active_orders[order.id] = managed_order
            self.stats['orders_submitted'] += 1

            # Publish event
            await self.event_bus.publish(Event(
                type=EventType.ORDER_FILLED,
                data={'order_id': order.id, 'state': 'submitted'},
                timestamp=datetime.now()
            ))

            logger.info(f"Order {order.id} submitted successfully")
            return order.id

        except Exception as e:
            managed_order.update_state(OrderState.FAILED)
            logger.error(f"Failed to submit order {order.id}: {e}")

            # Store failed order for potential retry
            self._active_orders[order.id] = managed_order
            raise OrderError(f"Order submission failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self._active_orders:
            logger.warning(f"Cannot cancel order {order_id}: not found in active orders")
            return False

        managed_order = self._active_orders[order_id]

        if managed_order.is_terminal_state():
            logger.warning(f"Cannot cancel order {order_id}: already in terminal state {managed_order.state.value}")
            return False

        try:
            success = await self.exchange_client.cancel_order(order_id)

            if success:
                managed_order.update_state(OrderState.CANCELLED)
                self._move_to_completed(order_id)
                self.stats['orders_cancelled'] += 1

                await self.event_bus.publish(Event(
                    type=EventType.ORDER_CANCELLED,
                    data={'order_id': order_id},
                    timestamp=datetime.now()
                ))

                logger.info(f"Order {order_id} cancelled successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def _monitor_orders(self) -> None:
        """Background task to monitor order status."""
        while self._is_running:
            try:
                await self._check_order_status()
                await self._handle_expired_orders()
                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _check_order_status(self) -> None:
        """Check status of all active orders."""
        order_ids = list(self._active_orders.keys())

        for order_id in order_ids:
            if order_id not in self._active_orders:
                continue

            managed_order = self._active_orders[order_id]

            if managed_order.is_terminal_state():
                continue

            try:
                # Get current order status from exchange
                current_order = await self.exchange_client.get_order_status(order_id)

                # Update our tracked order
                old_filled_amount = managed_order.order.filled_amount
                managed_order.order.status = current_order.status
                managed_order.order.filled_amount = current_order.filled_amount
                managed_order.order.average_price = current_order.average_price

                # Handle fills
                if current_order.filled_amount > old_filled_amount:
                    fill_amount = current_order.filled_amount - old_filled_amount
                    fill_price = current_order.average_price or current_order.price
                    managed_order.add_fill(fill_amount, fill_price)

                    self.stats['total_fill_value'] += fill_amount * fill_price

                # Update state based on order status
                if current_order.status == OrderStatus.FILLED:
                    managed_order.update_state(OrderState.FILLED)
                    self._move_to_completed(order_id)
                    self.stats['orders_filled'] += 1

                elif current_order.status == OrderStatus.CANCELLED:
                    managed_order.update_state(OrderState.CANCELLED)
                    self._move_to_completed(order_id)

                elif current_order.status == OrderStatus.REJECTED:
                    managed_order.update_state(OrderState.REJECTED)
                    self._move_to_completed(order_id)
                    self.stats['orders_rejected'] += 1

            except Exception as e:
                logger.error(f"Error checking status for order {order_id}: {e}")

    async def _handle_expired_orders(self) -> None:
        """Handle orders that have expired."""
        expired_orders = []

        for order_id, managed_order in self._active_orders.items():
            if managed_order.is_expired() and not managed_order.is_terminal_state():
                expired_orders.append(order_id)

        for order_id in expired_orders:
            logger.warning(f"Order {order_id} has expired, attempting to cancel")

            managed_order = self._active_orders[order_id]
            managed_order.update_state(OrderState.EXPIRED)

            # Try to cancel the expired order
            try:
                await self.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel expired order {order_id}: {e}")
                self._move_to_completed(order_id)

            self.stats['orders_expired'] += 1

    async def _cancel_all_active_orders(self) -> None:
        """Cancel all active orders during shutdown."""
        order_ids = list(self._active_orders.keys())

        for order_id in order_ids:
            try:
                await self.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id} during shutdown: {e}")

    def _move_to_completed(self, order_id: str) -> None:
        """Move order from active to completed."""
        if order_id in self._active_orders:
            self._completed_orders[order_id] = self._active_orders.pop(order_id)

    def get_order_status(self, order_id: str) -> Optional[ManagedOrder]:
        """Get detailed order status."""
        if order_id in self._active_orders:
            return self._active_orders[order_id]
        elif order_id in self._completed_orders:
            return self._completed_orders[order_id]
        return None

    def get_active_orders(self) -> Dict[str, ManagedOrder]:
        """Get all active orders."""
        return self._active_orders.copy()

    def get_completed_orders(self) -> Dict[str, ManagedOrder]:
        """Get all completed orders."""
        return self._completed_orders.copy()

    def get_statistics(self) -> Dict:
        """Get order management statistics."""
        total_orders = self.stats['orders_submitted']
        fill_rate = (self.stats['orders_filled'] / total_orders * 100) if total_orders > 0 else 0

        return {
            **self.stats,
            'active_orders_count': len(self._active_orders),
            'completed_orders_count': len(self._completed_orders),
            'fill_rate_percent': round(fill_rate, 2)
        }

    async def retry_failed_orders(self) -> List[str]:
        """Retry all eligible failed orders."""
        retried_orders = []

        failed_orders = [
            (order_id, managed_order)
            for order_id, managed_order in self._active_orders.items()
            if managed_order.can_retry()
        ]

        for order_id, managed_order in failed_orders:
            try:
                managed_order.retry_count += 1
                await self.submit_order(managed_order.order)
                retried_orders.append(order_id)
                logger.info(f"Retried failed order {order_id} (attempt {managed_order.retry_count})")

            except Exception as e:
                logger.error(f"Failed to retry order {order_id}: {e}")

        return retried_orders