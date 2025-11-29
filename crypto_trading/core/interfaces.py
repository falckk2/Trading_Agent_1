"""
Core interfaces and abstract base classes for the trading system.
Following SOLID principles for modular and extensible design.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol, Callable, Awaitable
from datetime import datetime
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass

# Import data models from models.py to avoid duplication
from .models import (
    OrderType, OrderSide, OrderStatus, SignalType,
    MarketData, Order, Position, TradingSignal
)


# Split interfaces following Interface Segregation Principle

class IExchangeConnection(ABC):
    """Interface for exchange connection management."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the exchange."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the exchange."""
        pass


class IMarketDataProvider(ABC):
    """Interface for market data retrieval."""

    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        pass


class IOrderExecutor(ABC):
    """Interface for order execution."""

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place a trading order. Returns the placed order with exchange ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get the status of an order."""
        pass


class IAccountDataProvider(ABC):
    """Interface for account data retrieval."""

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        pass


class IExchangeClient(IExchangeConnection, IMarketDataProvider, IOrderExecutor, IAccountDataProvider):
    """
    Complete exchange client interface.
    Combines all exchange functionality for backward compatibility.
    New code should use the specific interfaces above.
    """
    pass


class IStrategy(ABC):
    """Interface for trading strategy implementations."""

    @abstractmethod
    def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data and generate trading signals."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        pass

    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal."""
        pass


class ISignalGenerator(ABC):
    """Interface for signal generation (core responsibility)."""

    @abstractmethod
    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data and generate trading signals."""
        pass


class IConfigurableAgent(ABC):
    """Interface for agent configuration."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the agent with configuration."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get configurable parameters for the agent."""
        pass


class ITradingAgent(ISignalGenerator, IConfigurableAgent):
    """
    Complete trading agent interface.
    Combines signal generation and configuration.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the trading agent."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the trading agent."""
        pass


class IDataCollector(ABC):
    """Interface for data collection implementations."""

    @abstractmethod
    async def collect_data(self, symbol: str, timeframe: str) -> List[MarketData]:
        """Collect market data for a symbol."""
        pass

    @abstractmethod
    async def store_data(self, data: List[MarketData]) -> None:
        """Store collected data."""
        pass


class IDataProvider(ABC):
    """Interface for data provider implementations."""

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        pass

    @abstractmethod
    async def get_realtime_data(self, symbol: str) -> MarketData:
        """Get current/realtime market data."""
        pass

    @abstractmethod
    async def get_data_range(self, symbol: str) -> Dict[str, datetime]:
        """Get available data range for a symbol."""
        pass


class IDataProcessor(ABC):
    """Interface for data processing implementations."""

    @abstractmethod
    def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """Preprocess raw market data."""
        pass

    @abstractmethod
    def calculate_features(self, data: List[MarketData]) -> Dict[str, List[float]]:
        """Calculate features from market data."""
        pass


class IRiskManager(ABC):
    """Interface for risk management implementations."""

    @abstractmethod
    def validate_order(self, order: Order, positions: List[Position]) -> bool:
        """Validate if an order meets risk criteria."""
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        signal: TradingSignal,
        balance: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate appropriate position size based on risk parameters."""
        pass


class INotificationService(ABC):
    """Interface for notification service implementations."""

    @abstractmethod
    async def send_notification(self, message: str, level: str = "info") -> None:
        """Send a notification message."""
        pass


class EventType(Enum):
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    SIGNAL_GENERATED = "signal_generated"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime


class IEventBus(ABC):
    """Interface for event bus implementations."""

    @abstractmethod
    def subscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]) -> None:
        """Subscribe to an event type."""
        pass

    @abstractmethod
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]) -> None:
        """Unsubscribe from an event type."""
        pass

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event."""
        pass


class IConfigManager(ABC):
    """Interface for configuration management."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save configuration to storage."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load configuration from storage."""
        pass


# Aliases for backward compatibility
IExchange = IExchangeClient


class IAgentManager(ABC):
    """Interface for agent management."""

    @abstractmethod
    def register_agent(self, agent: ITradingAgent) -> None:
        """Register a trading agent."""
        pass

    @abstractmethod
    def get_agent(self, name: str) -> Optional[ITradingAgent]:
        """Get agent by name."""
        pass

    @abstractmethod
    def get_active_agent(self) -> Optional[ITradingAgent]:
        """Get currently active agent."""
        pass

    @abstractmethod
    def set_active_agent(self, name: str) -> bool:
        """Set active agent by name."""
        pass

    @abstractmethod
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        pass


class IOrderManager(ABC):
    """Interface for order management."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit an order for execution."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        pass

    @abstractmethod
    def get_all_orders(self) -> List[Order]:
        """Get all orders."""
        pass


class IDataStorage(ABC):
    """Interface for data storage operations."""

    @abstractmethod
    async def save_market_data(self, data: List[MarketData]) -> None:
        """Save market data."""
        pass

    @abstractmethod
    async def save_order(self, order: Order) -> None:
        """Save order data."""
        pass

    @abstractmethod
    async def save_position(self, position: Position) -> None:
        """Save position data."""
        pass


class IPortfolioManager(ABC):
    """Interface for portfolio management implementations."""

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        pass

    @abstractmethod
    def update_position(self, position: Position) -> None:
        """Update a position."""
        pass


class ILogger(ABC):
    """Interface for logging abstraction."""

    @abstractmethod
    def debug(self, message: str) -> None:
        """Log debug message."""
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message."""
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        """Log warning message."""
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def critical(self, message: str) -> None:
        """Log critical message."""
        pass