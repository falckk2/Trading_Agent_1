"""
Data storage manager for handling training data persistence.
Supports SQLite and CSV storage formats.
"""

import sqlite3
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

from ...core.interfaces import MarketData
from ...core.exceptions import DataError


class DataStorageManager:
    """Manages storage and retrieval of market data for training."""

    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        open_price DECIMAL(20, 8) NOT NULL,
                        high_price DECIMAL(20, 8) NOT NULL,
                        low_price DECIMAL(20, 8) NOT NULL,
                        close_price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        bid_price DECIMAL(20, 8),
                        ask_price DECIMAL(20, 8),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
                    ON market_data(symbol, timestamp)
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        symbols TEXT NOT NULL,
                        start_date DATETIME NOT NULL,
                        end_date DATETIME NOT NULL,
                        features TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise DataError(f"Database initialization failed: {e}")

    async def save_market_data(self, data: List[MarketData]) -> None:
        """Save market data to database."""
        if not data:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                for item in data:
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data
                        (symbol, timestamp, open_price, high_price, low_price,
                         close_price, volume, bid_price, ask_price)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item.symbol,
                        item.timestamp,
                        float(item.open),
                        float(item.high),
                        float(item.low),
                        float(item.close),
                        float(item.volume),
                        float(item.bid) if item.bid else None,
                        float(item.ask) if item.ask else None
                    ))
                conn.commit()

            logger.debug(f"Saved {len(data)} market data records")

        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            raise DataError(f"Failed to save market data: {e}")

    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[MarketData]:
        """Retrieve market data from database."""
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp ASC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            data = []
            for row in rows:
                data.append(MarketData(
                    symbol=row['symbol'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    open=row['open_price'],
                    high=row['high_price'],
                    low=row['low_price'],
                    close=row['close_price'],
                    volume=row['volume'],
                    bid=row['bid_price'],
                    ask=row['ask_price']
                ))

            return data

        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            raise DataError(f"Failed to retrieve market data: {e}")

    def export_to_csv(
        self,
        symbol: str,
        output_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> None:
        """Export market data to CSV file."""
        try:
            data = self.get_market_data(symbol, start_date, end_date)

            if not data:
                logger.warning(f"No data found for {symbol}")
                return

            # Convert to DataFrame
            df_data = []
            for item in data:
                df_data.append({
                    'timestamp': item.timestamp,
                    'symbol': item.symbol,
                    'open': float(item.open),
                    'high': float(item.high),
                    'low': float(item.low),
                    'close': float(item.close),
                    'volume': float(item.volume),
                    'bid': float(item.bid) if item.bid else None,
                    'ask': float(item.ask) if item.ask else None
                })

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(data)} records to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise DataError(f"Failed to export to CSV: {e}")

    def create_training_dataset(
        self,
        name: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        description: str = "",
        features: Optional[List[str]] = None
    ) -> str:
        """Create a training dataset specification."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO training_datasets
                    (name, description, symbols, start_date, end_date, features)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    name,
                    description,
                    ",".join(symbols),
                    start_date,
                    end_date,
                    ",".join(features) if features else ""
                ))
                conn.commit()

            logger.info(f"Created training dataset: {name}")
            return name

        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            raise DataError(f"Failed to create training dataset: {e}")

    def get_training_dataset(self, name: str) -> Dict[str, Any]:
        """Get training dataset specification."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM training_datasets WHERE name = ?", (name,)
                )
                row = cursor.fetchone()

            if not row:
                raise DataError(f"Training dataset '{name}' not found")

            return {
                "name": row["name"],
                "description": row["description"],
                "symbols": row["symbols"].split(","),
                "start_date": datetime.fromisoformat(row["start_date"]),
                "end_date": datetime.fromisoformat(row["end_date"]),
                "features": row["features"].split(",") if row["features"] else [],
                "created_at": datetime.fromisoformat(row["created_at"])
            }

        except Exception as e:
            logger.error(f"Error getting training dataset: {e}")
            raise DataError(f"Failed to get training dataset: {e}")

    def prepare_ml_dataset(
        self,
        dataset_name: str,
        output_format: str = "pandas"
    ) -> Any:
        """Prepare dataset for machine learning training."""
        try:
            dataset_spec = self.get_training_dataset(dataset_name)

            # Collect data for all symbols in the dataset
            all_data = []
            for symbol in dataset_spec["symbols"]:
                data = self.get_market_data(
                    symbol,
                    dataset_spec["start_date"],
                    dataset_spec["end_date"]
                )
                all_data.extend(data)

            if output_format == "pandas":
                # Convert to pandas DataFrame
                df_data = []
                for item in all_data:
                    df_data.append({
                        'timestamp': item.timestamp,
                        'symbol': item.symbol,
                        'open': float(item.open),
                        'high': float(item.high),
                        'low': float(item.low),
                        'close': float(item.close),
                        'volume': float(item.volume),
                        'bid': float(item.bid) if item.bid else None,
                        'ask': float(item.ask) if item.ask else None
                    })

                df = pd.DataFrame(df_data)
                df = df.sort_values(['symbol', 'timestamp'])

                logger.info(f"Prepared ML dataset with {len(df)} records")
                return df

            else:
                return all_data

        except Exception as e:
            logger.error(f"Error preparing ML dataset: {e}")
            raise DataError(f"Failed to prepare ML dataset: {e}")

    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get record count by symbol
                cursor = conn.execute("""
                    SELECT symbol, COUNT(*) as count,
                           MIN(timestamp) as earliest,
                           MAX(timestamp) as latest
                    FROM market_data
                    GROUP BY symbol
                """)
                symbol_stats = cursor.fetchall()

                # Get total record count
                cursor = conn.execute("SELECT COUNT(*) FROM market_data")
                total_records = cursor.fetchone()[0]

                # Get dataset count
                cursor = conn.execute("SELECT COUNT(*) FROM training_datasets")
                dataset_count = cursor.fetchone()[0]

            stats = {
                "total_records": total_records,
                "dataset_count": dataset_count,
                "symbols": {}
            }

            for symbol, count, earliest, latest in symbol_stats:
                stats["symbols"][symbol] = {
                    "record_count": count,
                    "earliest_data": earliest,
                    "latest_data": latest
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
            raise DataError(f"Failed to get data stats: {e}")