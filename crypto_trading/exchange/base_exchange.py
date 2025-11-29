"""
Base exchange class implementing common functionality.
Following Open/Closed Principle (OCP) - open for extension, closed for modification.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from decimal import Decimal
import logging

from ..core.interfaces import IExchangeClient
from ..core.interfaces import Order, MarketData, OrderStatus, OrderType, Position
from ..core.exceptions import (
    ExchangeError as ExchangeException,
    ConnectionError as ExchangeConnectionError,
    OrderError as ExchangeAPIError,
    OrderError as OrderException
)


class BaseExchange(IExchangeClient):
    """
    Abstract base class for exchange implementations.
    Provides common functionality while allowing for exchange-specific customization.
    """

    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self._connected = False
        self.rate_limit_delay = 0.1  # Default rate limiting
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Connection settings
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1

        # Market data cache
        self._market_data_cache: Dict[str, MarketData] = {}
        self._cache_expiry = 5  # Cache expiry in seconds
        self._last_cache_update: Dict[str, float] = {}

    async def connect(self) -> bool:
        """Establish connection to the exchange."""
        try:
            await self._authenticate()
            await self._initialize_connection()
            self._connected = True
            self.logger.info(f"Connected to {self.__class__.__name__}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise ExchangeConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to the exchange."""
        try:
            await self._cleanup_connection()
            self._connected = False
            self.logger.info(f"Disconnected from {self.__class__.__name__}")
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")

    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance for all assets."""
        self._ensure_connected()
        try:
            balances = await self._get_balance_impl()
            # Convert to Decimal if not already
            return {
                asset: Decimal(str(balance)) if not isinstance(balance, Decimal) else balance
                for asset, balance in balances.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            raise ExchangeAPIError(f"Balance retrieval failed: {e}")

    async def place_order(self, order: Order) -> Order:
        """Place an order and return the updated order with exchange ID."""
        self._ensure_connected()

        # Validate order
        self._validate_order(order)

        try:
            order_id = await self._place_order_impl(order)
            order.order_id = order_id
            order.status = OrderStatus.OPEN
            order.timestamp = datetime.now()
            self.logger.info(f"Order placed: {order_id}")
            return order
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise OrderException(f"Order placement failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        self._ensure_connected()
        try:
            result = await self._cancel_order_impl(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderException(f"Order cancellation failed: {e}")

    async def get_order_status(self, order_id: str) -> Order:
        """Get status of an order."""
        self._ensure_connected()
        try:
            return await self._get_order_status_impl(order_id)
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            raise ExchangeAPIError(f"Order status retrieval failed: {e}")

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol with caching."""
        self._ensure_connected()

        # Check cache first
        if self._is_cache_valid(symbol):
            return self._market_data_cache[symbol]

        try:
            market_data = await self._get_market_data_impl(symbol)
            self._update_cache(symbol, market_data)
            return market_data
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise ExchangeAPIError(f"Market data retrieval failed: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[MarketData]:
        """Get historical market data."""
        self._ensure_connected()
        try:
            return await self._get_historical_data_impl(
                symbol, timeframe, start_date, end_date, limit
            )
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise ExchangeAPIError(f"Historical data retrieval failed: {e}")

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        self._ensure_connected()
        try:
            return await self._get_positions_impl()
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise ExchangeAPIError(f"Position retrieval failed: {e}")

    def is_connected(self) -> bool:
        """Check if connected to the exchange."""
        return self._connected

    # Protected methods for subclasses to implement

    @abstractmethod
    async def _authenticate(self) -> None:
        """Authenticate with the exchange."""
        pass

    @abstractmethod
    async def _initialize_connection(self) -> None:
        """Initialize the connection."""
        pass

    @abstractmethod
    async def _cleanup_connection(self) -> None:
        """Cleanup connection resources."""
        pass

    @abstractmethod
    async def _get_balance_impl(self) -> Dict[str, Decimal]:
        """Implementation-specific balance retrieval for all assets."""
        pass

    @abstractmethod
    async def _place_order_impl(self, order: Order) -> str:
        """Implementation-specific order placement."""
        pass

    @abstractmethod
    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Implementation-specific order cancellation."""
        pass

    @abstractmethod
    async def _get_order_status_impl(self, order_id: str) -> Order:
        """Implementation-specific order status retrieval."""
        pass

    @abstractmethod
    async def _get_market_data_impl(self, symbol: str) -> MarketData:
        """Implementation-specific market data retrieval."""
        pass

    @abstractmethod
    async def _get_historical_data_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: int
    ) -> List[MarketData]:
        """Implementation-specific historical data retrieval."""
        pass

    @abstractmethod
    async def _get_positions_impl(self) -> List[Position]:
        """Implementation-specific positions retrieval."""
        pass

    # Helper methods

    def _ensure_connected(self) -> None:
        """Ensure the exchange is connected."""
        if not self._connected:
            raise ExchangeConnectionError("Not connected to exchange")

    def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        if order.quantity <= 0:
            raise OrderException("Order quantity must be positive")

        if order.order_type == OrderType.LIMIT and (not order.price or order.price <= 0):
            raise OrderException("Limit orders require a positive price")

        if not order.symbol:
            raise OrderException("Order symbol is required")

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self._market_data_cache:
            return False

        last_update = self._last_cache_update.get(symbol, 0)
        return (time.time() - last_update) < self._cache_expiry

    def _update_cache(self, symbol: str, data: MarketData) -> None:
        """Update the market data cache."""
        self._market_data_cache[symbol] = data
        self._last_cache_update[symbol] = time.time()

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        if self.rate_limit_delay > 0:
            await asyncio.sleep(self.rate_limit_delay)

    async def _retry_request(self, func: Callable, *args, **kwargs) -> Any:
        """Retry a request with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e

                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)