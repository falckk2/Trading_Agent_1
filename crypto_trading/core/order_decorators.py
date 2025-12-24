"""
Order execution decorators using Decorator pattern.
Allows adding cross-cutting concerns (logging, retry, metrics) to order execution.
Follows Open/Closed Principle - can add new decorators without modifying existing code.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from decimal import Decimal
import time
from datetime import datetime
import asyncio

from .interfaces import IOrderExecutor, ILogger, Order, Position
from ..utils.logging import create_logger


class OrderExecutorDecorator(IOrderExecutor, ABC):
    """
    Abstract decorator for order executors.
    Implements Decorator pattern for adding behavior to order execution.
    """

    def __init__(self, wrapped: IOrderExecutor):
        """
        Initialize decorator.

        Args:
            wrapped: The order executor to wrap/decorate
        """
        self._wrapped = wrapped

    async def place_order(self, order: Order) -> Order:
        """
        Place order - delegates to wrapped executor.
        Subclasses can override to add behavior before/after.
        """
        return await self._wrapped.place_order(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order - delegates to wrapped executor."""
        return await self._wrapped.cancel_order(order_id)

    async def get_order_status(self, order_id: str) -> Order:
        """Get order status - delegates to wrapped executor."""
        return await self._wrapped.get_order_status(order_id)


class LoggingOrderExecutorDecorator(OrderExecutorDecorator):
    """
    Decorator that adds logging to order execution.
    Logs every order placed, cancelled, or queried.
    """

    def __init__(self, wrapped: IOrderExecutor, logger: Optional[ILogger] = None):
        """
        Initialize logging decorator.

        Args:
            wrapped: Order executor to wrap
            logger: Logger to use (creates default if None)
        """
        super().__init__(wrapped)
        self.logger = logger or create_logger("OrderExecutor")

    async def place_order(self, order: Order) -> Order:
        """Place order with logging."""
        self.logger.info(
            f"Placing order: {order.symbol} {order.side.value} "
            f"{order.amount} @ {order.price or 'MARKET'}"
        )

        try:
            result = await self._wrapped.place_order(order)
            self.logger.info(
                f"Order placed successfully: ID={result.id} Status={result.status.value}"
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with logging."""
        self.logger.info(f"Cancelling order: {order_id}")

        try:
            result = await self._wrapped.cancel_order(order_id)
            if result:
                self.logger.info(f"Order cancelled successfully: {order_id}")
            else:
                self.logger.warning(f"Failed to cancel order: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            raise

    async def get_order_status(self, order_id: str) -> Order:
        """Get order status with logging."""
        self.logger.debug(f"Getting order status: {order_id}")

        try:
            result = await self._wrapped.get_order_status(order_id)
            self.logger.debug(f"Order {order_id} status: {result.status.value}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            raise


class RetryOrderExecutorDecorator(OrderExecutorDecorator):
    """
    Decorator that adds retry logic to order execution.
    Retries failed orders with exponential backoff.
    """

    def __init__(
        self,
        wrapped: IOrderExecutor,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        logger: Optional[ILogger] = None
    ):
        """
        Initialize retry decorator.

        Args:
            wrapped: Order executor to wrap
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            backoff_factor: Factor to multiply delay by for each retry
            logger: Logger to use
        """
        super().__init__(wrapped)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.logger = logger or create_logger("RetryOrderExecutor")

    async def place_order(self, order: Order) -> Order:
        """Place order with retry logic."""
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                return await self._wrapped.place_order(order)

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Order placement failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    self.logger.error(
                        f"Order placement failed after {self.max_retries + 1} attempts: {e}"
                    )

        # All retries exhausted
        raise last_exception

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with retry logic."""
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                return await self._wrapped.cancel_order(order_id)

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Order cancellation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor

        # All retries exhausted - log but don't raise (cancellation is best-effort)
        self.logger.error(f"Order cancellation failed after {self.max_retries + 1} attempts: {last_exception}")
        return False


class MetricsOrderExecutorDecorator(OrderExecutorDecorator):
    """
    Decorator that collects metrics about order execution.
    Tracks latency, success rate, volume, etc.
    """

    def __init__(self, wrapped: IOrderExecutor, logger: Optional[ILogger] = None):
        """
        Initialize metrics decorator.

        Args:
            wrapped: Order executor to wrap
            logger: Logger to use
        """
        super().__init__(wrapped)
        self.logger = logger or create_logger("OrderMetrics")

        # Metrics
        self.orders_placed = 0
        self.orders_failed = 0
        self.orders_cancelled = 0
        self.total_volume = Decimal('0')
        self.total_latency = 0.0
        self.start_time = datetime.now()

    async def place_order(self, order: Order) -> Order:
        """Place order with metrics collection."""
        start = time.time()

        try:
            result = await self._wrapped.place_order(order)

            # Record success metrics
            latency = time.time() - start
            self.orders_placed += 1
            self.total_latency += latency

            if order.price:
                volume = order.amount * order.price
                self.total_volume += volume

            self.logger.debug(
                f"Order metrics - Latency: {latency:.3f}s, "
                f"Total placed: {self.orders_placed}, "
                f"Success rate: {self.get_success_rate():.2%}"
            )

            return result

        except Exception as e:
            self.orders_failed += 1
            self.logger.debug(f"Order failed - Total failures: {self.orders_failed}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with metrics collection."""
        try:
            result = await self._wrapped.cancel_order(order_id)
            if result:
                self.orders_cancelled += 1
            return result
        except Exception as e:
            self.logger.debug(f"Order cancellation error: {e}")
            raise

    def get_metrics(self) -> dict:
        """
        Get collected metrics.

        Returns:
            Dictionary of metrics
        """
        total_orders = self.orders_placed + self.orders_failed
        avg_latency = self.total_latency / self.orders_placed if self.orders_placed > 0 else 0

        return {
            'orders_placed': self.orders_placed,
            'orders_failed': self.orders_failed,
            'orders_cancelled': self.orders_cancelled,
            'total_volume': float(self.total_volume),
            'average_latency': avg_latency,
            'success_rate': self.get_success_rate(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

    def get_success_rate(self) -> float:
        """Calculate order success rate."""
        total = self.orders_placed + self.orders_failed
        return self.orders_placed / total if total > 0 else 0.0

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.orders_placed = 0
        self.orders_failed = 0
        self.orders_cancelled = 0
        self.total_volume = Decimal('0')
        self.total_latency = 0.0
        self.start_time = datetime.now()
        self.logger.info("Metrics reset")


class RateLimitingOrderExecutorDecorator(OrderExecutorDecorator):
    """
    Decorator that adds rate limiting to order execution.
    Prevents exceeding exchange rate limits.
    """

    def __init__(
        self,
        wrapped: IOrderExecutor,
        max_orders_per_second: float = 10.0,
        logger: Optional[ILogger] = None
    ):
        """
        Initialize rate limiting decorator.

        Args:
            wrapped: Order executor to wrap
            max_orders_per_second: Maximum orders per second
            logger: Logger to use
        """
        super().__init__(wrapped)
        self.max_orders_per_second = max_orders_per_second
        self.min_interval = 1.0 / max_orders_per_second
        self.last_order_time = 0.0
        self.logger = logger or create_logger("RateLimiter")

    async def place_order(self, order: Order) -> Order:
        """Place order with rate limiting."""
        # Calculate time since last order
        now = time.time()
        time_since_last = now - self.last_order_time

        # Wait if necessary to respect rate limit
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            self.logger.debug(f"Rate limit: waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)

        self.last_order_time = time.time()
        return await self._wrapped.place_order(order)


class ValidationOrderExecutorDecorator(OrderExecutorDecorator):
    """
    Decorator that adds validation before order execution.
    Performs sanity checks before sending orders to exchange.
    """

    def __init__(self, wrapped: IOrderExecutor, logger: Optional[ILogger] = None):
        """
        Initialize validation decorator.

        Args:
            wrapped: Order executor to wrap
            logger: Logger to use
        """
        super().__init__(wrapped)
        self.logger = logger or create_logger("OrderValidator")

    async def place_order(self, order: Order) -> Order:
        """Place order with validation."""
        # Basic validation
        if not order.symbol:
            raise ValueError("Order must have a symbol")

        if order.amount <= 0:
            raise ValueError(f"Order amount must be positive, got {order.amount}")

        if order.price is not None and order.price <= 0:
            raise ValueError(f"Order price must be positive, got {order.price}")

        # Sanity checks
        if order.amount > 1000000:  # Arbitrary large number
            self.logger.warning(f"Large order amount detected: {order.amount}")

        if order.price is not None and order.price > 1000000:
            self.logger.warning(f"Extremely high price detected: {order.price}")

        return await self._wrapped.place_order(order)


# Builder for creating decorated order executors
class OrderExecutorBuilder:
    """
    Builder for creating decorated order executors.
    Simplifies adding multiple decorators in a fluent way.
    """

    def __init__(self, base_executor: IOrderExecutor):
        """
        Initialize builder.

        Args:
            base_executor: Base order executor to decorate
        """
        self._executor = base_executor

    def with_logging(self, logger: Optional[ILogger] = None) -> 'OrderExecutorBuilder':
        """Add logging decorator."""
        self._executor = LoggingOrderExecutorDecorator(self._executor, logger)
        return self

    def with_retry(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        logger: Optional[ILogger] = None
    ) -> 'OrderExecutorBuilder':
        """Add retry decorator."""
        self._executor = RetryOrderExecutorDecorator(
            self._executor, max_retries, initial_delay, backoff_factor, logger
        )
        return self

    def with_metrics(self, logger: Optional[ILogger] = None) -> 'OrderExecutorBuilder':
        """Add metrics decorator."""
        self._executor = MetricsOrderExecutorDecorator(self._executor, logger)
        return self

    def with_rate_limiting(
        self,
        max_orders_per_second: float = 10.0,
        logger: Optional[ILogger] = None
    ) -> 'OrderExecutorBuilder':
        """Add rate limiting decorator."""
        self._executor = RateLimitingOrderExecutorDecorator(
            self._executor, max_orders_per_second, logger
        )
        return self

    def with_validation(self, logger: Optional[ILogger] = None) -> 'OrderExecutorBuilder':
        """Add validation decorator."""
        self._executor = ValidationOrderExecutorDecorator(self._executor, logger)
        return self

    def build(self) -> IOrderExecutor:
        """
        Build the decorated order executor.

        Returns:
            Fully decorated order executor
        """
        return self._executor
