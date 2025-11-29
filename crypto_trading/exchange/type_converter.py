"""
Type conversion strategies for exchange implementations.
Implements Strategy pattern for flexible type conversions.
"""

from abc import ABC, abstractmethod
from typing import Dict
from ..core.interfaces import OrderType, OrderStatus


class ITypeConverter(ABC):
    """Interface for type conversion strategies."""

    @abstractmethod
    def convert_order_type_to_exchange(self, order_type: OrderType) -> str:
        """Convert internal order type to exchange format."""
        pass

    @abstractmethod
    def convert_order_type_from_exchange(self, exchange_type: str) -> OrderType:
        """Convert exchange order type to internal format."""
        pass

    @abstractmethod
    def convert_order_status_from_exchange(self, exchange_status: str) -> OrderStatus:
        """Convert exchange order status to internal format."""
        pass


class BlofinTypeConverter(ITypeConverter):
    """Type converter implementation for Blofin exchange."""

    def __init__(self):
        self._order_type_to_exchange: Dict[OrderType, str] = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "market",  # Blofin doesn't have separate stop orders
            OrderType.STOP_LIMIT: "limit"
        }

        self._order_type_from_exchange: Dict[str, OrderType] = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "post_only": OrderType.LIMIT,
            "fok": OrderType.MARKET,
            "ioc": OrderType.MARKET
        }

        self._order_status_from_exchange: Dict[str, OrderStatus] = {
            "live": OrderStatus.OPEN,
            "partially_filled": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED
        }

    def convert_order_type_to_exchange(self, order_type: OrderType) -> str:
        """Convert internal order type to Blofin format."""
        return self._order_type_to_exchange.get(order_type, "market")

    def convert_order_type_from_exchange(self, exchange_type: str) -> OrderType:
        """Convert Blofin order type to internal format."""
        return self._order_type_from_exchange.get(exchange_type, OrderType.MARKET)

    def convert_order_status_from_exchange(self, exchange_status: str) -> OrderStatus:
        """Convert Blofin order status to internal format."""
        return self._order_status_from_exchange.get(exchange_status, OrderStatus.PENDING)
