"""
Data persistence layer for trading history, performance metrics, and system state.
Provides SQLite-based storage with async support and data validation.
"""

import asyncio
import aiosqlite
import json
import pickle
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from dataclasses import asdict
from loguru import logger

from ..core.interfaces import Order, Position, MarketData, TradingSignal, OrderSide, OrderType, OrderStatus
from ..core.portfolio_tracker import TradeRecord, PortfolioSnapshot, PerformanceMetrics
from ..core.exceptions import DataError as DataException
from ..utils.decorators import requires_connection, retry_on_failure, handle_trading_errors


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class TradingDataPersistence:
    """Comprehensive data persistence for trading system."""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection: Optional[aiosqlite.Connection] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize database and create tables."""
        if self._is_initialized:
            return

        try:
            self._connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            self._is_initialized = True
            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataException(f"Database initialization failed: {e}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._is_initialized = False

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Orders table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL,
                status TEXT NOT NULL,
                filled_amount REAL DEFAULT 0,
                average_price REAL,
                timestamp TEXT NOT NULL,
                data TEXT,  -- JSON serialized order data
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trades table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                value REAL NOT NULL,
                fees REAL DEFAULT 0,
                pnl REAL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (order_id) REFERENCES orders (id)
            )
        """)

        # Positions table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                pnl REAL NOT NULL,
                timestamp TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Market data table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL NOT NULL,
                bid REAL,
                ask REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Trading signals table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL,
                amount REAL,
                timestamp TEXT NOT NULL,
                metadata TEXT,  -- JSON serialized metadata
                agent_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Portfolio snapshots table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_balance TEXT,  -- JSON serialized balance
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                exposure REAL NOT NULL,
                leverage REAL NOT NULL,
                risk_metrics TEXT,  -- JSON serialized risk metrics
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance metrics table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                start_value REAL NOT NULL,
                end_value REAL NOT NULL,
                total_return REAL NOT NULL,
                total_return_pct REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                win_rate_pct REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                data TEXT,  -- JSON serialized full metrics
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System events table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT,  -- JSON serialized event data
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Configuration history table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT NOT NULL,
                changed_by TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)")

        await self._connection.commit()

    @requires_connection
    @retry_on_failure(max_retries=3, delay=0.5)
    async def save_order(self, order: Order) -> None:
        """Save order to database."""
        await self._connection.execute("""
            INSERT OR REPLACE INTO orders
            (id, symbol, side, type, amount, price, status, filled_amount, average_price, timestamp, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.id,
            order.symbol,
            order.side.value,
            order.type.value,
            float(order.amount),
            float(order.price) if order.price else None,
            order.status.value,
            float(order.filled_amount),
            float(order.average_price) if order.average_price else None,
            order.timestamp.isoformat(),
            json.dumps(asdict(order), cls=DecimalEncoder)
        ))

        await self._connection.commit()
        logger.debug(f"Saved order {order.id} to database")

    @requires_connection
    @retry_on_failure(max_retries=3, delay=0.5)
    async def save_trade(self, trade: TradeRecord) -> None:
        """Save trade record to database."""
        await self._connection.execute("""
            INSERT INTO trades
            (order_id, symbol, side, quantity, price, value, fees, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.order_id,
            trade.symbol,
            trade.side.value,
            float(trade.quantity),
            float(trade.price),
            float(trade.value),
                float(trade.fees),
            float(trade.pnl) if trade.pnl else None,
            trade.timestamp.isoformat()
        ))

        await self._connection.commit()
        logger.debug(f"Saved trade {trade.order_id} to database")

    async def save_position(self, position: Position, is_active: bool = True) -> None:
        """Save position to database."""
        await self._ensure_connection()

        try:
            await self._connection.execute("""
                INSERT INTO positions
                (symbol, side, amount, entry_price, current_price, pnl, timestamp, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol,
                position.side.value,
                float(position.amount),
                float(position.entry_price),
                float(position.current_price),
                float(position.pnl),
                position.timestamp.isoformat(),
                is_active
            ))

            await self._connection.commit()
            logger.debug(f"Saved position {position.symbol} to database")

        except Exception as e:
            logger.error(f"Failed to save position {position.symbol}: {e}")
            raise DataException(f"Failed to save position: {e}")

    async def save_market_data(self, market_data: MarketData) -> None:
        """Save market data to database."""
        await self._ensure_connection()

        try:
            await self._connection.execute("""
                INSERT OR REPLACE INTO market_data
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume, bid, ask)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market_data.symbol,
                market_data.timestamp.isoformat(),
                float(market_data.open),
                float(market_data.high),
                float(market_data.low),
                float(market_data.close),
                float(market_data.volume),
                float(market_data.bid) if market_data.bid else None,
                float(market_data.ask) if market_data.ask else None
            ))

            await self._connection.commit()

        except Exception as e:
            logger.debug(f"Failed to save market data for {market_data.symbol}: {e}")

    async def save_trading_signal(self, signal: TradingSignal, agent_name: str = None) -> None:
        """Save trading signal to database."""
        await self._ensure_connection()

        try:
            await self._connection.execute("""
                INSERT INTO trading_signals
                (symbol, action, confidence, price, amount, timestamp, metadata, agent_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol,
                signal.action.value,
                signal.confidence,
                float(signal.price) if signal.price else None,
                float(signal.amount) if signal.amount else None,
                signal.timestamp.isoformat(),
                json.dumps(signal.metadata, cls=DecimalEncoder),
                agent_name
            ))

            await self._connection.commit()
            logger.debug(f"Saved trading signal for {signal.symbol}")

        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")

    async def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Save portfolio snapshot to database."""
        await self._ensure_connection()

        try:
            await self._connection.execute("""
                INSERT INTO portfolio_snapshots
                (timestamp, total_value, cash_balance, unrealized_pnl, realized_pnl,
                 total_pnl, daily_pnl, exposure, leverage, risk_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                float(snapshot.total_value),
                json.dumps({k: float(v) for k, v in snapshot.cash_balance.items()}),
                float(snapshot.unrealized_pnl),
                float(snapshot.realized_pnl),
                float(snapshot.total_pnl),
                float(snapshot.daily_pnl),
                float(snapshot.exposure),
                float(snapshot.leverage),
                json.dumps(snapshot.risk_metrics, cls=DecimalEncoder)
            ))

            await self._connection.commit()

        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")

    async def save_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save performance metrics to database."""
        await self._ensure_connection()

        try:
            await self._connection.execute("""
                INSERT INTO performance_metrics
                (period, start_date, end_date, start_value, end_value, total_return,
                 total_return_pct, sharpe_ratio, max_drawdown_pct, win_rate_pct, total_trades, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.period.value,
                metrics.start_date.isoformat(),
                metrics.end_date.isoformat(),
                float(metrics.start_value),
                float(metrics.end_value),
                float(metrics.total_return),
                float(metrics.total_return_pct),
                float(metrics.sharpe_ratio),
                float(metrics.max_drawdown_pct),
                float(metrics.win_rate_pct),
                metrics.total_trades,
                json.dumps(asdict(metrics), cls=DecimalEncoder)
            ))

            await self._connection.commit()
            logger.debug(f"Saved performance metrics for {metrics.period.value}")

        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")

    async def log_system_event(
        self,
        event_type: str,
        level: str,
        message: str,
        data: Dict[str, Any] = None
    ) -> None:
        """Log system event to database."""
        await self._ensure_connection()

        try:
            await self._connection.execute("""
                INSERT INTO system_events (event_type, level, message, data, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_type,
                level,
                message,
                json.dumps(data, cls=DecimalEncoder) if data else None,
                datetime.now().isoformat()
            ))

            await self._connection.commit()

        except Exception as e:
            logger.error(f"Failed to log system event: {e}")

    async def get_orders(
        self,
        symbol: str = None,
        status: OrderStatus = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve orders from database."""
        await self._ensure_connection()

        query = "SELECT * FROM orders WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Convert to dictionaries
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    async def get_trades(
        self,
        symbol: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve trades from database."""
        await self._ensure_connection()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    async def get_portfolio_snapshots(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve portfolio snapshots from database."""
        await self._ensure_connection()

        query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get portfolio snapshots: {e}")
            return []

    async def get_market_data(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve market data from database."""
        await self._ensure_connection()

        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []

    async def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        await self._ensure_connection()

        try:
            stats = {}

            # Order statistics
            async with self._connection.execute("""
                SELECT status, COUNT(*) as count FROM orders GROUP BY status
            """) as cursor:
                order_stats = await cursor.fetchall()
                stats['orders_by_status'] = {row[0]: row[1] for row in order_stats}

            # Trade statistics
            async with self._connection.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(value) as total_volume,
                    SUM(fees) as total_fees,
                    SUM(pnl) as total_pnl
                FROM trades WHERE pnl IS NOT NULL
            """) as cursor:
                trade_stats = await cursor.fetchone()
                if trade_stats:
                    stats['trade_summary'] = {
                        'total_trades': trade_stats[0],
                        'winning_trades': trade_stats[1],
                        'losing_trades': trade_stats[2],
                        'win_rate': (trade_stats[1] / trade_stats[0] * 100) if trade_stats[0] > 0 else 0,
                        'total_volume': trade_stats[3],
                        'total_fees': trade_stats[4],
                        'total_pnl': trade_stats[5]
                    }

            # Recent performance
            async with self._connection.execute("""
                SELECT total_value, timestamp FROM portfolio_snapshots
                ORDER BY timestamp DESC LIMIT 2
            """) as cursor:
                recent_values = await cursor.fetchall()
                if len(recent_values) >= 2:
                    current_value = recent_values[0][0]
                    previous_value = recent_values[1][0]
                    change = current_value - previous_value
                    change_pct = (change / previous_value * 100) if previous_value > 0 else 0

                    stats['recent_performance'] = {
                        'current_value': current_value,
                        'previous_value': previous_value,
                        'change': change,
                        'change_percent': change_pct
                    }

            return stats

        except Exception as e:
            logger.error(f"Failed to get trading statistics: {e}")
            return {}

    async def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old data to manage database size."""
        await self._ensure_connection()

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.isoformat()

        try:
            # Clean up old market data (keep less history)
            await self._connection.execute("""
                DELETE FROM market_data WHERE timestamp < ?
            """, (cutoff_str,))

            # Clean up old portfolio snapshots (keep less frequent)
            await self._connection.execute("""
                DELETE FROM portfolio_snapshots
                WHERE timestamp < ?
                AND id NOT IN (
                    SELECT id FROM portfolio_snapshots
                    WHERE timestamp < ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                )
            """, (cutoff_str, cutoff_str))

            # Clean up old system events
            await self._connection.execute("""
                DELETE FROM system_events WHERE timestamp < ?
            """, (cutoff_str,))

            await self._connection.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def _ensure_connection(self) -> None:
        """Ensure database connection is available."""
        if not self._is_initialized:
            await self.initialize()

    async def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database."""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backup_trading_{timestamp}.db"

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Simple file copy for SQLite
            import shutil
            shutil.copy2(self.db_path, backup_path)

            logger.info(f"Database backed up to {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise DataException(f"Database backup failed: {e}")


# Factory function for easy instantiation
async def create_persistence_layer(db_path: str = "data/trading.db") -> TradingDataPersistence:
    """Create and initialize a persistence layer."""
    persistence = TradingDataPersistence(db_path)
    await persistence.initialize()
    return persistence