"""
Core data models for the trading system.
Using dataclasses and Pydantic for data validation and serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


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
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Asset:
    """Represents a cryptocurrency asset."""
    symbol: str
    name: str
    precision: int = 8
    min_order_size: float = 0.0
    tick_size: float = 0.01

    def __post_init__(self):
        self.symbol = self.symbol.upper()


@dataclass
class MarketData:
    """Market data for a specific symbol and timestamp."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

    def __post_init__(self):
        if self.bid and self.ask:
            self.spread = self.ask - self.bid

    @property
    def ohlc(self) -> tuple:
        """Return OHLC as tuple."""
        return (self.open, self.high, self.low, self.close)


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop orders require a stop price")

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is active (pending or open)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN]

    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to be filled."""
        return self.quantity - self.filled_quantity


@dataclass
class Trade:
    """Represents an executed trade."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    trade_id: Optional[str] = None
    fees: float = 0.0
    commission: float = 0.0

    @property
    def value(self) -> float:
        """Calculate trade value."""
        return self.quantity * self.price

    @property
    def net_value(self) -> float:
        """Calculate net trade value after fees."""
        return self.value - self.fees - self.commission


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    average_price: float
    current_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return 0.0

        if self.quantity > 0:  # Long position
            return (self.current_price - self.average_price) * self.quantity
        else:  # Short position
            return (self.average_price - self.current_price) * abs(self.quantity)

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    def update_price(self, new_price: float) -> None:
        """Update current price for position."""
        self.current_price = new_price


@dataclass
class Portfolio:
    """Represents a trading portfolio."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_fees: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        """Calculate total portfolio P&L."""
        return sum(pos.total_pnl for pos in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self.positions[position.symbol] = position
        self.updated_at = datetime.now()

    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        if symbol in self.positions:
            del self.positions[symbol]
            self.updated_at = datetime.now()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)


@dataclass
class TradingSignal:
    """Represents a trading signal generated by a strategy."""
    symbol: str
    signal_type: SignalType
    strength: float  # Signal strength between 0 and 1
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # Confidence level between 0 and 1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.strength <= 1:
            raise ValueError("Signal strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act upon."""
        return self.strength >= 0.5 and self.confidence >= 0.3


class PerformanceMetrics(BaseModel):
    """Performance metrics for trading strategies."""
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    total_trades: int = Field(..., description="Total number of trades")
    avg_trade_return: float = Field(..., description="Average trade return")
    volatility: float = Field(..., description="Strategy volatility")

    class Config:
        schema_extra = {
            "example": {
                "total_return": 15.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": -8.3,
                "win_rate": 65.0,
                "profit_factor": 1.8,
                "total_trades": 150,
                "avg_trade_return": 0.8,
                "volatility": 12.4
            }
        }