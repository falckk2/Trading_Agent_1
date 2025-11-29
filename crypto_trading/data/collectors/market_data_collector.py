"""
Market data collector for gathering training data.
Implements data collection interface for various sources.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from ...core.interfaces import IDataCollector, IExchangeClient, MarketData
from ...core.exceptions import DataCollectionError


class MarketDataCollector(IDataCollector):
    """Collects market data from exchange for training and analysis."""

    def __init__(self, exchange_client: IExchangeClient, storage_manager: 'DataStorageManager'):
        self.exchange_client = exchange_client
        self.storage_manager = storage_manager
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._is_collecting = False

    async def collect_data(self, symbol: str, timeframe: str) -> List[MarketData]:
        """Collect market data for a symbol and timeframe."""
        try:
            # Get recent historical data (last 24 hours)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)

            data = await self.exchange_client.get_historical_data(
                symbol, timeframe, start_time, end_time
            )

            logger.info(f"Collected {len(data)} data points for {symbol} ({timeframe})")
            return data

        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            raise DataCollectionError(f"Failed to collect data: {e}")

    async def store_data(self, data: List[MarketData]) -> None:
        """Store collected data."""
        try:
            await self.storage_manager.save_market_data(data)
            logger.info(f"Stored {len(data)} market data points")

        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise DataCollectionError(f"Failed to store data: {e}")

    async def start_continuous_collection(self, symbols: List[str], timeframes: List[str], interval: int = 60) -> None:
        """Start continuous data collection for specified symbols and timeframes."""
        if self._is_collecting:
            logger.warning("Data collection is already running")
            return

        self._is_collecting = True
        logger.info(f"Starting continuous data collection for {symbols}")

        for symbol in symbols:
            for timeframe in timeframes:
                task_key = f"{symbol}_{timeframe}"
                task = asyncio.create_task(
                    self._collect_continuously(symbol, timeframe, interval)
                )
                self._collection_tasks[task_key] = task

    async def stop_continuous_collection(self) -> None:
        """Stop continuous data collection."""
        if not self._is_collecting:
            return

        self._is_collecting = False

        # Cancel all collection tasks
        for task in self._collection_tasks.values():
            task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self._collection_tasks.values(), return_exceptions=True)
        self._collection_tasks.clear()

        logger.info("Stopped continuous data collection")

    async def _collect_continuously(self, symbol: str, timeframe: str, interval: int) -> None:
        """Continuously collect data for a symbol and timeframe."""
        while self._is_collecting:
            try:
                # Collect current market data
                current_data = await self.exchange_client.get_market_data(symbol)
                await self.store_data([current_data])

                # Wait for next collection cycle
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous collection for {symbol}: {e}")
                await asyncio.sleep(interval)  # Continue on error

    async def collect_historical_bulk(
        self,
        symbols: List[str],
        timeframes: List[str],
        days_back: int = 30
    ) -> None:
        """Collect bulk historical data for multiple symbols."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        logger.info(f"Starting bulk historical data collection for {len(symbols)} symbols, {days_back} days back")

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    data = await self.exchange_client.get_historical_data(
                        symbol, timeframe, start_time, end_time
                    )
                    await self.store_data(data)

                    logger.info(f"Collected and stored {len(data)} points for {symbol} ({timeframe})")

                except Exception as e:
                    logger.error(f"Error collecting bulk data for {symbol} ({timeframe}): {e}")

        logger.info("Bulk historical data collection completed")