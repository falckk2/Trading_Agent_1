"""
Central data management class coordinating all data operations.
Implements Dependency Injection and Facade patterns.
"""

from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import asyncio
import logging

from ..core.interfaces import IDataProvider
from ..core.models import MarketData
from ..core.exceptions import DataException
from .historical.data_provider import HistoricalDataProvider
from .realtime.realtime_feed import RealtimeFeed
from .preprocessing.data_preprocessor import DataPreprocessor
from .storage.data_storage import DataStorage


class DataManager(IDataProvider):
    """
    Central data manager providing unified access to historical and real-time data.
    Coordinates data collection, storage, preprocessing, and distribution.
    """

    def __init__(
        self,
        historical_provider: HistoricalDataProvider,
        realtime_feed: RealtimeFeed,
        preprocessor: DataPreprocessor,
        storage: DataStorage
    ):
        self.historical_provider = historical_provider
        self.realtime_feed = realtime_feed
        self.preprocessor = preprocessor
        self.storage = storage
        self.logger = logging.getLogger(__name__)

        # Subscription management
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._active_symbols: set = set()

        # Data cache
        self._data_cache: Dict[str, List[MarketData]] = {}
        self._cache_size = 1000  # Maximum number of data points per symbol

    async def initialize(self) -> None:
        """Initialize all data components."""
        try:
            await self.historical_provider.initialize()
            await self.realtime_feed.initialize()
            await self.storage.initialize()
            self.logger.info("Data manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize data manager: {e}")
            raise DataException(f"Initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown all data components."""
        try:
            await self.realtime_feed.shutdown()
            await self.historical_provider.shutdown()
            await self.storage.shutdown()
            self.logger.info("Data manager shutdown completed")
        except Exception as e:
            self.logger.warning(f"Error during shutdown: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> List[MarketData]:
        """
        Get historical market data for a symbol.
        Checks storage first, then fetches from provider if needed.
        """
        try:
            # Check local storage first if use_cache is True
            if use_cache:
                stored_data = await self.storage.get_historical_data(
                    symbol, timeframe, start_date, end_date
                )
                if stored_data:
                    self.logger.debug(f"Retrieved {len(stored_data)} records from storage")
                    return stored_data

            # Fetch from external provider
            data = await self.historical_provider.get_historical_data(
                symbol, timeframe, start_date, end_date
            )

            # Preprocess data
            processed_data = await self.preprocessor.process_historical_data(data)

            # Store for future use
            if processed_data:
                await self.storage.store_historical_data(processed_data)

            self.logger.info(
                f"Retrieved {len(processed_data)} historical records for {symbol}"
            )
            return processed_data

        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise DataException(f"Historical data retrieval failed: {e}")

    async def get_realtime_data(self, symbol: str) -> MarketData:
        """Get current real-time market data."""
        try:
            data = await self.realtime_feed.get_current_data(symbol)

            # Process real-time data
            processed_data = await self.preprocessor.process_realtime_data(data)

            # Update cache
            self._update_cache(symbol, processed_data)

            return processed_data

        except Exception as e:
            self.logger.error(f"Failed to get real-time data for {symbol}: {e}")
            raise DataException(f"Real-time data retrieval failed: {e}")

    async def subscribe_to_data(self, symbols: List[str], callback: Callable) -> None:
        """Subscribe to real-time data updates for multiple symbols."""
        try:
            for symbol in symbols:
                if symbol not in self._subscriptions:
                    self._subscriptions[symbol] = []
                self._subscriptions[symbol].append(callback)

                # Start data feed if not already active
                if symbol not in self._active_symbols:
                    await self.realtime_feed.subscribe(symbol, self._data_callback)
                    self._active_symbols.add(symbol)

            self.logger.info(f"Subscribed to data for symbols: {symbols}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to data: {e}")
            raise DataException(f"Subscription failed: {e}")

    async def unsubscribe_from_data(self, symbols: List[str]) -> None:
        """Unsubscribe from data updates."""
        try:
            for symbol in symbols:
                if symbol in self._subscriptions:
                    del self._subscriptions[symbol]

                if symbol in self._active_symbols:
                    await self.realtime_feed.unsubscribe(symbol)
                    self._active_symbols.discard(symbol)

            self.logger.info(f"Unsubscribed from data for symbols: {symbols}")

        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from data: {e}")

    async def get_cached_data(
        self,
        symbol: str,
        limit: Optional[int] = None
    ) -> List[MarketData]:
        """Get cached data for a symbol."""
        if symbol not in self._data_cache:
            return []

        data = self._data_cache[symbol]
        return data[-limit:] if limit else data

    async def preload_data(
        self,
        symbols: List[str],
        timeframe: str,
        days_back: int = 30
    ) -> None:
        """Preload historical data for multiple symbols."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        tasks = []
        for symbol in symbols:
            task = self.get_historical_data(symbol, timeframe, start_date, end_date)
            tasks.append(task)

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info(f"Preloaded data for {len(symbols)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to preload data: {e}")

    async def get_data_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics about available data for a symbol."""
        try:
            stats = await self.storage.get_data_statistics(symbol)

            # Add cache statistics
            cache_data = self._data_cache.get(symbol, [])
            stats.update({
                "cached_records": len(cache_data),
                "cache_start": cache_data[0].timestamp if cache_data else None,
                "cache_end": cache_data[-1].timestamp if cache_data else None,
                "is_subscribed": symbol in self._active_symbols
            })

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get data statistics for {symbol}: {e}")
            return {}

    async def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old data from storage."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            await self.storage.cleanup_old_data(cutoff_date)
            self.logger.info(f"Cleaned up data older than {cutoff_date}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")

    # Private methods

    async def _data_callback(self, symbol: str, data: MarketData) -> None:
        """Internal callback for processing incoming data."""
        try:
            # Process the data
            processed_data = await self.preprocessor.process_realtime_data(data)

            # Update cache
            self._update_cache(symbol, processed_data)

            # Store data
            await self.storage.store_realtime_data(processed_data)

            # Notify subscribers
            if symbol in self._subscriptions:
                for callback in self._subscriptions[symbol]:
                    try:
                        await callback(symbol, processed_data)
                    except Exception as e:
                        self.logger.error(f"Error in callback for {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error processing data callback for {symbol}: {e}")

    def _update_cache(self, symbol: str, data: MarketData) -> None:
        """Update the internal data cache."""
        if symbol not in self._data_cache:
            self._data_cache[symbol] = []

        self._data_cache[symbol].append(data)

        # Maintain cache size limit
        if len(self._data_cache[symbol]) > self._cache_size:
            self._data_cache[symbol] = self._data_cache[symbol][-self._cache_size:]

    def get_subscription_count(self) -> int:
        """Get the number of active subscriptions."""
        return len(self._active_symbols)

    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self._active_symbols)