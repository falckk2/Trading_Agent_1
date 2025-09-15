"""
Core interfaces and abstract base classes for the trading system.
Following SOLID principles for modular and extensible design.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    price: Optional[Decimal]
    status: OrderStatus
    timestamp: datetime
    filled_amount: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None


@dataclass
class Position:
    symbol: str
    side: OrderSide
    amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    pnl: Decimal
    timestamp: datetime


@dataclass
class TradingSignal:
    symbol: str
    action: OrderSide
    confidence: float
    price: Optional[Decimal]
    amount: Optional[Decimal]
    timestamp: datetime
    metadata: Dict[str, Any]


class IExchangeClient(ABC):
    """Interface for exchange client implementations."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the exchange."""
        pass

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

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place a trading order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get the status of an order."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        pass


class ITradingAgent(ABC):
    """Interface for trading agent implementations."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the trading agent with configuration."""
        pass

    @abstractmethod
    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data and generate trading signals."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the trading agent."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the trading agent."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get configurable parameters for the agent."""
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
    def subscribe(self, event_type: EventType, callback) -> None:
        """Subscribe to an event type."""
        pass

    @abstractmethod
    def unsubscribe(self, event_type: EventType, callback) -> None:
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