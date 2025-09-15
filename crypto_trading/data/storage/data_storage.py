"""
Data storage implementation using SQLite for local storage.
Provides efficient storage and retrieval of market data.
"""

import sqlite3
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiosqlite
import pandas as pd
import logging

from ...core.models import MarketData
from ...core.exceptions import DataException


class DataStorage:
    """
    SQLite-based data storage for market data.
    Provides efficient storage with indexing and querying capabilities.
    """

    def __init__(self, db_path: str = "crypto_trading.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Table schemas
        self.schemas = {
            "market_data": """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    bid REAL,
                    ask REAL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """,
            "data_statistics": """
                CREATE TABLE IF NOT EXISTS data_statistics (
                    symbol TEXT PRIMARY KEY,
                    first_timestamp DATETIME,
                    last_timestamp DATETIME,
                    total_records INTEGER,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
        }

        # Indexes for performance
        self.indexes = [
            "CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_symbol ON market_data(symbol)"
        ]

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create tables
                for table_name, schema in self.schemas.items():
                    await db.execute(schema)

                # Create indexes
                for index_sql in self.indexes:
                    await db.execute(index_sql)

                await db.commit()

            self.logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise DataException(f"Database initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the storage system."""
        self.logger.info("Data storage shutdown completed")

    async def store_historical_data(self, data: List[MarketData]) -> int:
        """Store historical market data."""
        if not data:
            return 0

        try:
            async with aiosqlite.connect(self.db_path) as db:
                stored_count = 0

                for market_data in data:
                    try:
                        await db.execute(
                            """
                            INSERT OR REPLACE INTO market_data
                            (symbol, timestamp, open, high, low, close, volume, bid, ask, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                market_data.symbol,
                                market_data.timestamp,
                                market_data.open,
                                market_data.high,
                                market_data.low,
                                market_data.close,
                                market_data.volume,
                                market_data.bid,
                                market_data.ask,
                                json.dumps({})  # Placeholder for metadata
                            )
                        )
                        stored_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to store data point: {e}")

                await db.commit()

                # Update statistics
                if stored_count > 0:
                    await self._update_statistics(db, data[0].symbol)

                self.logger.info(f"Stored {stored_count} historical data points")
                return stored_count

        except Exception as e:
            self.logger.error(f"Failed to store historical data: {e}")
            raise DataException(f"Historical data storage failed: {e}")

    async def store_realtime_data(self, data: MarketData) -> bool:
        """Store real-time market data."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO market_data
                    (symbol, timestamp, open, high, low, close, volume, bid, ask, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data.symbol,
                        data.timestamp,
                        data.open,
                        data.high,
                        data.low,
                        data.close,
                        data.volume,
                        data.bid,
                        data.ask,
                        json.dumps({})
                    )
                )

                await db.commit()
                await self._update_statistics(db, data.symbol)

                return True

        except Exception as e:
            self.logger.error(f"Failed to store real-time data: {e}")
            return False

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """Retrieve historical market data."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                cursor = await db.execute(
                    """
                    SELECT symbol, timestamp, open, high, low, close, volume, bid, ask
                    FROM market_data
                    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                    """,
                    (symbol, start_date, end_date)
                )

                rows = await cursor.fetchall()
                data_list = []

                for row in rows:
                    data_list.append(MarketData(
                        symbol=row['symbol'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        bid=row['bid'],
                        ask=row['ask']
                    ))

                self.logger.debug(
                    f"Retrieved {len(data_list)} records for {symbol} "
                    f"from {start_date} to {end_date}"
                )

                return data_list

        except Exception as e:
            self.logger.error(f"Failed to retrieve historical data: {e}")
            raise DataException(f"Historical data retrieval failed: {e}")

    async def get_latest_data(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get the latest data points for a symbol."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                cursor = await db.execute(
                    """
                    SELECT symbol, timestamp, open, high, low, close, volume, bid, ask
                    FROM market_data
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (symbol, limit)
                )

                rows = await cursor.fetchall()
                data_list = []

                for row in rows:
                    data_list.append(MarketData(
                        symbol=row['symbol'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        bid=row['bid'],
                        ask=row['ask']
                    ))

                # Reverse to get chronological order
                return list(reversed(data_list))

        except Exception as e:
            self.logger.error(f"Failed to get latest data for {symbol}: {e}")
            return []

    async def get_data_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics about stored data for a symbol."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Get basic statistics
                cursor = await db.execute(
                    """
                    SELECT
                        COUNT(*) as total_records,
                        MIN(timestamp) as first_timestamp,
                        MAX(timestamp) as last_timestamp,
                        AVG(volume) as avg_volume,
                        MIN(close) as min_price,
                        MAX(close) as max_price
                    FROM market_data
                    WHERE symbol = ?
                    """,
                    (symbol,)
                )

                row = await cursor.fetchone()

                if row and row['total_records'] > 0:
                    return {
                        "symbol": symbol,
                        "total_records": row['total_records'],
                        "first_timestamp": datetime.fromisoformat(row['first_timestamp'])
                                         if row['first_timestamp'] else None,
                        "last_timestamp": datetime.fromisoformat(row['last_timestamp'])
                                        if row['last_timestamp'] else None,
                        "average_volume": row['avg_volume'],
                        "min_price": row['min_price'],
                        "max_price": row['max_price']
                    }
                else:
                    return {
                        "symbol": symbol,
                        "total_records": 0,
                        "first_timestamp": None,
                        "last_timestamp": None,
                        "average_volume": 0,
                        "min_price": 0,
                        "max_price": 0
                    }

        except Exception as e:
            self.logger.error(f"Failed to get statistics for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    async def get_all_symbols(self) -> List[str]:
        """Get all symbols with stored data."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT DISTINCT symbol FROM market_data ORDER BY symbol"
                )

                rows = await cursor.fetchall()
                return [row[0] for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            return []

    async def cleanup_old_data(self, cutoff_date: datetime) -> int:
        """Remove data older than cutoff date."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM market_data WHERE timestamp < ?",
                    (cutoff_date,)
                )

                deleted_count = cursor.rowcount
                await db.commit()

                self.logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return 0

    async def export_data_to_csv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        file_path: str
    ) -> bool:
        """Export data to CSV file."""
        try:
            data = await self.get_historical_data(symbol, "1d", start_date, end_date)

            if not data:
                self.logger.warning(f"No data to export for {symbol}")
                return False

            # Convert to DataFrame
            records = []
            for item in data:
                records.append({
                    'timestamp': item.timestamp,
                    'open': item.open,
                    'high': item.high,
                    'low': item.low,
                    'close': item.close,
                    'volume': item.volume
                })

            df = pd.DataFrame(records)
            df.to_csv(file_path, index=False)

            self.logger.info(f"Exported {len(data)} records to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False

    async def get_data_gaps(self, symbol: str, expected_interval_minutes: int = 1) -> List[Dict]:
        """Identify gaps in the data."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT timestamp FROM market_data
                    WHERE symbol = ?
                    ORDER BY timestamp ASC
                    """,
                    (symbol,)
                )

                rows = await cursor.fetchall()
                timestamps = [datetime.fromisoformat(row[0]) for row in rows]

                gaps = []
                expected_interval = timedelta(minutes=expected_interval_minutes)

                for i in range(1, len(timestamps)):
                    time_diff = timestamps[i] - timestamps[i-1]
                    if time_diff > expected_interval * 1.5:  # Allow some tolerance
                        gaps.append({
                            "start": timestamps[i-1],
                            "end": timestamps[i],
                            "duration_minutes": time_diff.total_seconds() / 60
                        })

                return gaps

        except Exception as e:
            self.logger.error(f"Failed to identify data gaps: {e}")
            return []

    # Private methods

    async def _update_statistics(self, db: aiosqlite.Connection, symbol: str) -> None:
        """Update data statistics for a symbol."""
        try:
            cursor = await db.execute(
                """
                SELECT
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(*) as total_records
                FROM market_data
                WHERE symbol = ?
                """,
                (symbol,)
            )

            row = await cursor.fetchone()

            if row:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO data_statistics
                    (symbol, first_timestamp, last_timestamp, total_records, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        row[0],
                        row[1],
                        row[2],
                        datetime.now()
                    )
                )

        except Exception as e:
            self.logger.warning(f"Failed to update statistics for {symbol}: {e}")

    async def get_database_size(self) -> Dict[str, Any]:
        """Get database size information."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get table sizes
                cursor = await db.execute(
                    """
                    SELECT
                        name,
                        COUNT(*) as row_count
                    FROM sqlite_master sm
                    JOIN (
                        SELECT 'market_data' as name, COUNT(*) as cnt FROM market_data
                        UNION ALL
                        SELECT 'data_statistics' as name, COUNT(*) as cnt FROM data_statistics
                    ) counts ON sm.name = counts.name
                    WHERE sm.type = 'table'
                    GROUP BY name
                    """
                )

                tables = await cursor.fetchall()
                table_info = {row[0]: row[1] for row in tables}

                # Get file size
                import os
                file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                return {
                    "file_size_bytes": file_size,
                    "file_size_mb": file_size / (1024 * 1024),
                    "tables": table_info
                }

        except Exception as e:
            self.logger.error(f"Failed to get database size: {e}")
            return {"error": str(e)}