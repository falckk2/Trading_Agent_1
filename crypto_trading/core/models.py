"""
Core data models for the trading system.
Using dataclasses and Pydantic for data validation and serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

# Pydantic is optional - only used for PerformanceMetrics
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda *args, **kwargs: None


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
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None

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
    stop_price: Optional[Decimal] = None
    fees: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
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
    def remaining_amount(self) -> Decimal:
        """Calculate remaining amount to be filled."""
        return self.amount - self.filled_amount


@dataclass
class Trade:
    """Represents an executed trade."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    order_id: str
    trade_id: Optional[str] = None
    fees: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')

    @property
    def value(self) -> Decimal:
        """Calculate trade value."""
        return self.quantity * self.price

    @property
    def net_value(self) -> Decimal:
        """Calculate net trade value after fees."""
        return self.value - self.fees - self.commission


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: OrderSide
    amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    pnl: Decimal
    timestamp: datetime
    realized_pnl: Decimal = Decimal('0')
    fees_paid: Decimal = Decimal('0')

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        return abs(self.amount) * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.amount == 0:
            return Decimal('0')

        if self.side == OrderSide.BUY:  # Long position
            return (self.current_price - self.entry_price) * self.amount
        else:  # Short position
            return (self.entry_price - self.current_price) * self.amount

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == OrderSide.BUY

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == OrderSide.SELL

    def update_price(self, new_price: Decimal) -> None:
        """Update current price for position."""
        self.current_price = new_price
        # Recalculate pnl
        self.pnl = self.unrealized_pnl


@dataclass
class Portfolio:
    """Represents a trading portfolio."""
    cash: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    total_fees: Decimal = Decimal('0')
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total portfolio P&L."""
        return sum(pos.total_pnl for pos in self.positions.values())

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> Decimal:
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
    action: OrderSide
    confidence: float
    price: Optional[Decimal]
    amount: Optional[Decimal]
    timestamp: datetime
    metadata: Dict[str, Any]
    # Legacy fields for compatibility
    signal_type: Optional[SignalType] = None
    strength: float = 0.0
    strategy_name: str = "unknown"
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.strength and not 0 <= self.strength <= 1:
            raise ValueError("Signal strength must be between 0 and 1")
        # Map action to signal_type if not set
        if self.signal_type is None:
            if self.action == OrderSide.BUY:
                self.signal_type = SignalType.BUY
            elif self.action == OrderSide.SELL:
                self.signal_type = SignalType.SELL
            else:
                self.signal_type = SignalType.HOLD

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act upon."""
        return self.confidence >= 0.5


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