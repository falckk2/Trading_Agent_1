"""
Connection management with automatic recovery and exponential backoff.
Provides resilient connection handling for exchange clients.
"""

import asyncio
import random
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from .interfaces import IExchangeClient
from ..utils.exceptions import ConnectionError, TradingSystemError


class ConnectionState(Enum):
    """Connection states for tracking."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ConnectionManager:
    """Manages connection lifecycle with automatic recovery."""

    def __init__(
        self,
        exchange_client: IExchangeClient,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.exchange_client = exchange_client
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

        self.state = ConnectionState.DISCONNECTED
        self.retry_count = 0
        self.last_connection_attempt: Optional[datetime] = None
        self.last_successful_connection: Optional[datetime] = None
        self.connection_failures: int = 0

        self._recovery_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Connection callbacks
        self._on_connected_callbacks: list[Callable] = []
        self._on_disconnected_callbacks: list[Callable] = []
        self._on_connection_failed_callbacks: list[Callable] = []

        # Statistics
        self.stats = {
            'total_connection_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'total_downtime_seconds': 0.0,
            'longest_downtime_seconds': 0.0,
            'average_connection_time': 0.0
        }

    async def start(self) -> None:
        """Start the connection manager."""
        if self._is_running:
            return

        self._is_running = True

        # Initial connection attempt
        await self.connect()

        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("ConnectionManager started")

    async def stop(self) -> None:
        """Stop the connection manager."""
        self._is_running = False

        # Cancel background tasks
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Disconnect
        await self.disconnect()
        logger.info("ConnectionManager stopped")

    async def connect(self) -> bool:
        """Attempt to establish connection with retry logic."""
        if self.state == ConnectionState.CONNECTING:
            logger.debug("Connection attempt already in progress")
            return False

        self.state = ConnectionState.CONNECTING
        self.last_connection_attempt = datetime.now()
        self.stats['total_connection_attempts'] += 1

        start_time = datetime.now()

        try:
            success = await self.exchange_client.connect()

            if success:
                connection_time = (datetime.now() - start_time).total_seconds()

                self.state = ConnectionState.CONNECTED
                self.last_successful_connection = datetime.now()
                self.retry_count = 0
                self.connection_failures = 0

                # Update statistics
                self.stats['successful_connections'] += 1
                if self.stats['average_connection_time'] == 0:
                    self.stats['average_connection_time'] = connection_time
                else:
                    # Rolling average
                    self.stats['average_connection_time'] = (
                        self.stats['average_connection_time'] * 0.8 + connection_time * 0.2
                    )

                # Notify callbacks
                for callback in self._on_connected_callbacks:
                    try:
                        await callback()
                    except Exception as e:
                        logger.error(f"Error in connection callback: {e}")

                logger.info(f"Successfully connected to exchange (took {connection_time:.2f}s)")
                return True
            else:
                self.state = ConnectionState.FAILED
                self.connection_failures += 1
                self.stats['failed_connections'] += 1

                logger.warning("Failed to connect to exchange")

                # Start recovery if not already running
                if not self._recovery_task or self._recovery_task.done():
                    self._recovery_task = asyncio.create_task(self._recovery_loop())

                return False

        except Exception as e:
            self.state = ConnectionState.FAILED
            self.connection_failures += 1
            self.stats['failed_connections'] += 1

            logger.error(f"Exception during connection attempt: {e}")

            # Start recovery if not already running
            if not self._recovery_task or self._recovery_task.done():
                self._recovery_task = asyncio.create_task(self._recovery_loop())

            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self.state == ConnectionState.DISCONNECTED:
            return

        try:
            await self.exchange_client.disconnect()

            old_state = self.state
            self.state = ConnectionState.DISCONNECTED

            # Notify callbacks if we were previously connected
            if old_state == ConnectionState.CONNECTED:
                for callback in self._on_disconnected_callbacks:
                    try:
                        await callback()
                    except Exception as e:
                        logger.error(f"Error in disconnection callback: {e}")

            logger.info("Disconnected from exchange")

        except Exception as e:
            logger.error(f"Error during disconnection: {e}")
            self.state = ConnectionState.DISCONNECTED

    async def ensure_connected(self) -> bool:
        """Ensure connection is established, reconnect if necessary."""
        if self.state == ConnectionState.CONNECTED:
            return True

        if self.state in [ConnectionState.CONNECTING, ConnectionState.RECONNECTING]:
            # Wait for ongoing connection attempt
            timeout = 30  # 30 second timeout
            start_time = datetime.now()

            while (self.state in [ConnectionState.CONNECTING, ConnectionState.RECONNECTING] and
                   (datetime.now() - start_time).total_seconds() < timeout):
                await asyncio.sleep(0.1)

            return self.state == ConnectionState.CONNECTED

        # Attempt to connect
        return await self.connect()

    async def _recovery_loop(self) -> None:
        """Background task for connection recovery with exponential backoff."""
        while (self._is_running and
               self.state != ConnectionState.CONNECTED and
               self.retry_count < self.max_retries):

            # Calculate delay with exponential backoff
            delay = min(
                self.base_delay * (self.backoff_factor ** self.retry_count),
                self.max_delay
            )

            # Add jitter to prevent thundering herd
            if self.jitter:
                delay += random.uniform(0, delay * 0.1)

            self.retry_count += 1

            logger.info(f"Attempting reconnection {self.retry_count}/{self.max_retries} in {delay:.1f}s")

            await asyncio.sleep(delay)

            if not self._is_running:
                break

            self.state = ConnectionState.RECONNECTING

            # Attempt reconnection
            success = await self.connect()

            if success:
                logger.info(f"Reconnection successful after {self.retry_count} attempts")
                break

        # If we've exhausted retries, notify failure callbacks
        if (self.retry_count >= self.max_retries and
            self.state != ConnectionState.CONNECTED):

            logger.error(f"Failed to reconnect after {self.max_retries} attempts")

            for callback in self._on_connection_failed_callbacks:
                try:
                    await callback()
                except Exception as e:
                    logger.error(f"Error in connection failed callback: {e}")

    async def _health_check_loop(self) -> None:
        """Background task to monitor connection health."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if self.state == ConnectionState.CONNECTED:
                    # Perform a lightweight health check
                    if not await self._perform_health_check():
                        logger.warning("Health check failed, initiating reconnection")
                        self.state = ConnectionState.FAILED

                        # Start recovery
                        if not self._recovery_task or self._recovery_task.done():
                            self._recovery_task = asyncio.create_task(self._recovery_loop())

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _perform_health_check(self) -> bool:
        """Perform a lightweight health check."""
        try:
            # Try to get account balance as a health check
            await self.exchange_client.get_balance()
            return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def add_connection_callback(self, callback: Callable) -> None:
        """Add callback for successful connections."""
        self._on_connected_callbacks.append(callback)

    def add_disconnection_callback(self, callback: Callable) -> None:
        """Add callback for disconnections."""
        self._on_disconnected_callbacks.append(callback)

    def add_connection_failed_callback(self, callback: Callable) -> None:
        """Add callback for connection failures."""
        self._on_connection_failed_callbacks.append(callback)

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        now = datetime.now()

        uptime_seconds = 0.0
        if (self.last_successful_connection and
            self.state == ConnectionState.CONNECTED):
            uptime_seconds = (now - self.last_successful_connection).total_seconds()

        return {
            'state': self.state.value,
            'is_connected': self.state == ConnectionState.CONNECTED,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'connection_failures': self.connection_failures,
            'last_connection_attempt': self.last_connection_attempt,
            'last_successful_connection': self.last_successful_connection,
            'uptime_seconds': uptime_seconds,
            'stats': self.stats.copy()
        }

    def reset_retry_count(self) -> None:
        """Reset retry count (useful for manual reconnection attempts)."""
        self.retry_count = 0
        logger.info("Retry count reset")

    async def force_reconnect(self) -> bool:
        """Force a reconnection attempt."""
        logger.info("Forcing reconnection")

        # Disconnect first
        await self.disconnect()

        # Reset retry count for fresh attempt
        self.reset_retry_count()

        # Attempt to connect
        return await self.connect()


class CircuitBreaker:
    """Circuit breaker pattern for connection failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise ConnectionError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }