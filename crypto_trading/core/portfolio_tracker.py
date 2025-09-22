"""
Portfolio tracking system for real-time P&L calculation and performance monitoring.
Provides comprehensive portfolio analytics and risk metrics.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .interfaces import Position, Order, MarketData, OrderSide, IExchangeClient
from ..utils.exceptions import TradingSystemError


class PerformancePeriod(Enum):
    """Time periods for performance calculation."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    total_value: Decimal
    cash_balance: Dict[str, Decimal]
    positions: List[Position]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    daily_pnl: Decimal
    exposure: Decimal
    leverage: Decimal
    margin_used: Decimal
    margin_available: Decimal
    risk_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Individual trade record for P&L calculation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    value: Decimal
    fees: Decimal
    timestamp: datetime
    pnl: Optional[Decimal] = None  # Set when position is closed


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    period: PerformancePeriod
    start_date: datetime
    end_date: datetime
    start_value: Decimal
    end_value: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    volatility_pct: Decimal
    sharpe_ratio: Decimal
    max_drawdown_pct: Decimal
    win_rate_pct: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal


class PortfolioTracker:
    """Real-time portfolio tracking and performance analytics."""

    def __init__(
        self,
        exchange_client: IExchangeClient,
        initial_cash: Dict[str, Decimal] = None,
        risk_free_rate: float = 0.02
    ):
        self.exchange_client = exchange_client
        self.risk_free_rate = risk_free_rate

        # Portfolio state
        self.cash_balance: Dict[str, Decimal] = initial_cash or {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []

        # Current values
        self.current_prices: Dict[str, Decimal] = {}
        self.last_update: Optional[datetime] = None

        # Historical snapshots
        self.snapshots: List[PortfolioSnapshot] = []
        self.max_snapshots = 10000  # Keep last 10K snapshots

        # Performance tracking
        self._daily_start_value: Optional[Decimal] = None
        self._session_start_value: Optional[Decimal] = None

        # Background tasks
        self._update_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Statistics
        self.stats = {
            'total_trades': 0,
            'total_volume': Decimal('0'),
            'total_fees': Decimal('0'),
            'max_portfolio_value': Decimal('0'),
            'min_portfolio_value': Decimal('0'),
            'largest_gain': Decimal('0'),
            'largest_loss': Decimal('0')
        }

    async def start(self, update_interval: float = 5.0) -> None:
        """Start the portfolio tracker."""
        if self._is_running:
            return

        self._is_running = True

        # Initialize portfolio state
        await self._initialize_portfolio()

        # Start background update task
        self._update_task = asyncio.create_task(
            self._update_loop(update_interval)
        )

        logger.info("PortfolioTracker started")

    async def stop(self) -> None:
        """Stop the portfolio tracker."""
        self._is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("PortfolioTracker stopped")

    async def _initialize_portfolio(self) -> None:
        """Initialize portfolio state from exchange."""
        try:
            # Get current balance
            balance = await self.exchange_client.get_balance()
            self.cash_balance.update(balance)

            # Get current positions
            positions = await self.exchange_client.get_positions()
            for position in positions:
                self.positions[position.symbol] = position

            # Take initial snapshot
            await self._take_snapshot()

            # Set daily start value
            current_value = await self.get_total_portfolio_value()
            if self._daily_start_value is None:
                self._daily_start_value = current_value
            if self._session_start_value is None:
                self._session_start_value = current_value

            logger.info(f"Portfolio initialized with value: {current_value}")

        except Exception as e:
            logger.error(f"Failed to initialize portfolio: {e}")
            raise TradingSystemError(f"Portfolio initialization failed: {e}")

    async def _update_loop(self, interval: float) -> None:
        """Background task to update portfolio state."""
        while self._is_running:
            try:
                await self.update_portfolio()
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(interval * 2)  # Wait longer before retrying

    async def update_portfolio(self) -> None:
        """Update portfolio with latest data."""
        try:
            # Update positions and balance
            await self._update_positions()
            await self._update_balance()

            # Update current prices for all assets
            await self._update_current_prices()

            # Take snapshot
            await self._take_snapshot()

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")

    async def _update_positions(self) -> None:
        """Update positions from exchange."""
        try:
            positions = await self.exchange_client.get_positions()

            # Update existing positions and add new ones
            new_positions = {}
            for position in positions:
                new_positions[position.symbol] = position

            self.positions = new_positions

        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    async def _update_balance(self) -> None:
        """Update cash balance from exchange."""
        try:
            balance = await self.exchange_client.get_balance()
            self.cash_balance.update(balance)

        except Exception as e:
            logger.error(f"Failed to update balance: {e}")

    async def _update_current_prices(self) -> None:
        """Update current prices for all assets."""
        symbols = set(self.positions.keys())

        # Also add base currencies from balance
        symbols.update(
            f"{currency}/USDT" for currency in self.cash_balance.keys()
            if currency != 'USDT'
        )

        for symbol in symbols:
            try:
                market_data = await self.exchange_client.get_market_data(symbol)
                self.current_prices[symbol] = market_data.close

            except Exception as e:
                logger.debug(f"Failed to get price for {symbol}: {e}")

    async def _take_snapshot(self) -> None:
        """Take a portfolio snapshot."""
        try:
            # Calculate metrics
            total_value = await self.get_total_portfolio_value()
            unrealized_pnl = self.calculate_unrealized_pnl()
            realized_pnl = self.calculate_realized_pnl()
            total_pnl = unrealized_pnl + realized_pnl
            daily_pnl = self.calculate_daily_pnl()
            exposure = self.calculate_total_exposure()
            leverage = self.calculate_leverage()

            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=total_value,
                cash_balance=self.cash_balance.copy(),
                positions=list(self.positions.values()),
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                exposure=exposure,
                leverage=leverage,
                margin_used=Decimal('0'),  # TODO: Calculate from exchange
                margin_available=Decimal('0'),  # TODO: Calculate from exchange
                risk_metrics=self._calculate_risk_metrics()
            )

            # Store snapshot
            self.snapshots.append(snapshot)

            # Limit snapshot history
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]

            # Update statistics
            self._update_statistics(snapshot)

        except Exception as e:
            logger.error(f"Failed to take portfolio snapshot: {e}")

    def record_trade(self, order: Order, fees: Decimal = Decimal('0')) -> None:
        """Record a completed trade."""
        trade = TradeRecord(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_amount,
            price=order.average_price or order.price,
            value=order.filled_amount * (order.average_price or order.price),
            fees=fees,
            timestamp=datetime.now()
        )

        self.trade_history.append(trade)
        self.stats['total_trades'] += 1
        self.stats['total_volume'] += trade.value
        self.stats['total_fees'] += fees

        # Update position
        self._update_position_from_trade(trade)

        logger.info(f"Recorded trade: {trade.symbol} {trade.side.value} {trade.quantity} @ {trade.price}")

    def _update_position_from_trade(self, trade: TradeRecord) -> None:
        """Update position based on trade."""
        symbol = trade.symbol

        if symbol not in self.positions:
            # New position
            if trade.side == OrderSide.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    amount=trade.quantity,
                    entry_price=trade.price,
                    current_price=trade.price,
                    pnl=Decimal('0'),
                    timestamp=trade.timestamp
                )
        else:
            # Update existing position
            position = self.positions[symbol]

            if position.side == trade.side:
                # Adding to position
                total_value = position.amount * position.entry_price + trade.value
                total_quantity = position.amount + trade.quantity
                position.entry_price = total_value / total_quantity
                position.amount = total_quantity
            else:
                # Reducing or closing position
                if trade.quantity >= position.amount:
                    # Position closed or reversed
                    remaining = trade.quantity - position.amount

                    # Calculate P&L for closed portion
                    if trade.side == OrderSide.SELL:
                        pnl = (trade.price - position.entry_price) * position.amount
                    else:
                        pnl = (position.entry_price - trade.price) * position.amount

                    # Update trade record with P&L
                    trade.pnl = pnl

                    if remaining > 0:
                        # Position reversed
                        position.side = trade.side
                        position.amount = remaining
                        position.entry_price = trade.price
                    else:
                        # Position closed
                        del self.positions[symbol]
                else:
                    # Partial reduction
                    position.amount -= trade.quantity

                    # Calculate P&L for reduced portion
                    if trade.side == OrderSide.SELL:
                        pnl = (trade.price - position.entry_price) * trade.quantity
                    else:
                        pnl = (position.entry_price - trade.price) * trade.quantity

                    trade.pnl = pnl

    async def get_total_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        total_value = Decimal('0')

        # Add cash balances (convert to base currency)
        for currency, amount in self.cash_balance.items():
            if currency == 'USDT':
                total_value += amount
            else:
                # Convert to USDT
                symbol = f"{currency}/USDT"
                price = self.current_prices.get(symbol, Decimal('1'))
                total_value += amount * price

        # Add position values
        for position in self.positions.values():
            current_price = self.current_prices.get(position.symbol, position.entry_price)
            position_value = position.amount * current_price
            total_value += position_value

        return total_value

    def calculate_unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L for open positions."""
        unrealized_pnl = Decimal('0')

        for position in self.positions.values():
            current_price = self.current_prices.get(position.symbol, position.entry_price)

            if position.side == OrderSide.BUY:
                pnl = (current_price - position.entry_price) * position.amount
            else:
                pnl = (position.entry_price - current_price) * position.amount

            unrealized_pnl += pnl

        return unrealized_pnl

    def calculate_realized_pnl(self) -> Decimal:
        """Calculate realized P&L from completed trades."""
        return sum(
            trade.pnl for trade in self.trade_history
            if trade.pnl is not None
        )

    def calculate_daily_pnl(self) -> Decimal:
        """Calculate P&L since start of day."""
        if not self._daily_start_value:
            return Decimal('0')

        # Use the last snapshot for current value
        if self.snapshots:
            current_value = self.snapshots[-1].total_value
            return current_value - self._daily_start_value

        return Decimal('0')

    def calculate_total_exposure(self) -> Decimal:
        """Calculate total position exposure."""
        exposure = Decimal('0')

        for position in self.positions.values():
            current_price = self.current_prices.get(position.symbol, position.entry_price)
            position_value = abs(position.amount * current_price)
            exposure += position_value

        return exposure

    def calculate_leverage(self) -> Decimal:
        """Calculate portfolio leverage."""
        if not self.snapshots:
            return Decimal('0')

        current_value = self.snapshots[-1].total_value
        if current_value <= 0:
            return Decimal('0')

        exposure = self.calculate_total_exposure()
        return exposure / current_value

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate various risk metrics."""
        if len(self.snapshots) < 30:  # Need at least 30 data points
            return {}

        # Get recent values for volatility calculation
        recent_values = [s.total_value for s in self.snapshots[-30:]]
        returns = []

        for i in range(1, len(recent_values)):
            if recent_values[i-1] > 0:
                returns.append(float(recent_values[i] / recent_values[i-1] - 1))

        if not returns:
            return {}

        import statistics

        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        var_95 = sorted(returns)[int(len(returns) * 0.05)] if len(returns) >= 20 else 0

        return {
            'volatility_30d': volatility,
            'value_at_risk_95': var_95,
            'beta': 0.0,  # TODO: Calculate vs market benchmark
            'correlation_btc': 0.0,  # TODO: Calculate vs BTC
        }

    def _update_statistics(self, snapshot: PortfolioSnapshot) -> None:
        """Update portfolio statistics."""
        total_value = snapshot.total_value

        if total_value > self.stats['max_portfolio_value']:
            self.stats['max_portfolio_value'] = total_value

        if (self.stats['min_portfolio_value'] == 0 or
            total_value < self.stats['min_portfolio_value']):
            self.stats['min_portfolio_value'] = total_value

        # Track largest gain/loss
        if snapshot.daily_pnl > self.stats['largest_gain']:
            self.stats['largest_gain'] = snapshot.daily_pnl

        if snapshot.daily_pnl < self.stats['largest_loss']:
            self.stats['largest_loss'] = snapshot.daily_pnl

    def get_performance_metrics(self, period: PerformancePeriod) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics for specified period."""
        if not self.snapshots:
            return None

        # Determine time range
        end_time = datetime.now()

        if period == PerformancePeriod.DAILY:
            start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.WEEKLY:
            start_time = end_time - timedelta(days=7)
        elif period == PerformancePeriod.MONTHLY:
            start_time = end_time - timedelta(days=30)
        elif period == PerformancePeriod.YEARLY:
            start_time = end_time - timedelta(days=365)
        else:  # ALL_TIME
            start_time = self.snapshots[0].timestamp

        # Find snapshots in range
        period_snapshots = [
            s for s in self.snapshots
            if start_time <= s.timestamp <= end_time
        ]

        if len(period_snapshots) < 2:
            return None

        # Calculate metrics
        start_value = period_snapshots[0].total_value
        end_value = period_snapshots[-1].total_value

        if start_value <= 0:
            return None

        total_return = end_value - start_value
        total_return_pct = float(total_return / start_value * 100)

        # Annualized return
        days = (end_time - start_time).days
        if days > 0:
            annualized_return_pct = ((float(end_value / start_value) ** (365.25 / days)) - 1) * 100
        else:
            annualized_return_pct = 0

        # Calculate volatility and Sharpe ratio
        returns = []
        values = [float(s.total_value) for s in period_snapshots]

        for i in range(1, len(values)):
            if values[i-1] > 0:
                returns.append(values[i] / values[i-1] - 1)

        if len(returns) > 1:
            import statistics
            volatility_pct = statistics.stdev(returns) * 100
            avg_return = statistics.mean(returns)
            sharpe_ratio = (avg_return - self.risk_free_rate / 365) / (volatility_pct / 100) if volatility_pct > 0 else 0
        else:
            volatility_pct = 0
            sharpe_ratio = 0

        # Max drawdown
        max_drawdown_pct = self._calculate_max_drawdown(values)

        # Trade statistics
        period_trades = [
            t for t in self.trade_history
            if start_time <= t.timestamp <= end_time and t.pnl is not None
        ]

        if period_trades:
            winning_trades = [t for t in period_trades if t.pnl > 0]
            losing_trades = [t for t in period_trades if t.pnl < 0]

            win_rate_pct = len(winning_trades) / len(period_trades) * 100

            total_wins = sum(float(t.pnl) for t in winning_trades)
            total_losses = abs(sum(float(t.pnl) for t in losing_trades))

            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            avg_win = Decimal(str(total_wins / len(winning_trades))) if winning_trades else Decimal('0')
            avg_loss = Decimal(str(total_losses / len(losing_trades))) if losing_trades else Decimal('0')
            largest_win = max(winning_trades, key=lambda t: t.pnl).pnl if winning_trades else Decimal('0')
            largest_loss = min(losing_trades, key=lambda t: t.pnl).pnl if losing_trades else Decimal('0')
        else:
            win_rate_pct = 0
            profit_factor = 0
            avg_win = avg_loss = largest_win = largest_loss = Decimal('0')
            winning_trades = losing_trades = []

        return PerformanceMetrics(
            period=period,
            start_date=start_time,
            end_date=end_time,
            start_value=start_value,
            end_value=end_value,
            total_return=total_return,
            total_return_pct=Decimal(str(total_return_pct)),
            annualized_return_pct=Decimal(str(annualized_return_pct)),
            volatility_pct=Decimal(str(volatility_pct)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown_pct=Decimal(str(max_drawdown_pct)),
            win_rate_pct=Decimal(str(win_rate_pct)),
            profit_factor=Decimal(str(profit_factor)),
            total_trades=len(period_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss
        )

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not values:
            return 0

        max_value = values[0]
        max_drawdown = 0

        for value in values:
            if value > max_value:
                max_value = value
            else:
                drawdown = (max_value - value) / max_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return max_drawdown * 100

    def get_current_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Get the most recent portfolio snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        snapshot = self.get_current_snapshot()

        if not snapshot:
            return {'error': 'No portfolio data available'}

        return {
            'timestamp': snapshot.timestamp,
            'total_value': float(snapshot.total_value),
            'cash_balance': {k: float(v) for k, v in snapshot.cash_balance.items()},
            'positions_count': len(snapshot.positions),
            'unrealized_pnl': float(snapshot.unrealized_pnl),
            'realized_pnl': float(snapshot.realized_pnl),
            'total_pnl': float(snapshot.total_pnl),
            'daily_pnl': float(snapshot.daily_pnl),
            'exposure': float(snapshot.exposure),
            'leverage': float(snapshot.leverage),
            'statistics': {k: float(v) if isinstance(v, Decimal) else v for k, v in self.stats.items()},
            'risk_metrics': snapshot.risk_metrics
        }

    def reset_daily_tracking(self) -> None:
        """Reset daily P&L tracking (call at start of new trading day)."""
        if self.snapshots:
            self._daily_start_value = self.snapshots[-1].total_value
            logger.info(f"Daily P&L tracking reset. Starting value: {self._daily_start_value}")