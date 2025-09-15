"""
Historical data provider for fetching market data from various sources.
Supports multiple data sources with fallback mechanisms.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
import yfinance as yf
import pandas as pd

from ...core.interfaces import IDataProvider
from ...core.models import MarketData
from ...core.exceptions import DataProviderError


class HistoricalDataProvider(IDataProvider):
    """
    Historical data provider supporting multiple data sources.
    Implements fallback mechanism for reliability.
    """

    def __init__(self, primary_source: str = "yfinance", fallback_sources: List[str] = None):
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or ["yfinance"]
        self.logger = logging.getLogger(__name__)

        # Rate limiting
        self.rate_limit_delay = 0.1
        self.max_requests_per_minute = 100
        self._request_timestamps: List[float] = []

        # Data source configurations
        self.source_configs = {
            "yfinance": {
                "symbol_mapping": self._get_yahoo_symbol,
                "fetch_method": self._fetch_from_yfinance
            }
        }

    async def initialize(self) -> None:
        """Initialize the data provider."""
        self.logger.info("Historical data provider initialized")

    async def shutdown(self) -> None:
        """Shutdown the data provider."""
        self.logger.info("Historical data provider shutdown")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """
        Get historical market data with fallback mechanism.
        """
        # Try primary source first
        try:
            data = await self._fetch_data(
                self.primary_source, symbol, timeframe, start_date, end_date
            )
            if data:
                self.logger.info(
                    f"Retrieved {len(data)} records from {self.primary_source}"
                )
                return data
        except Exception as e:
            self.logger.warning(f"Primary source {self.primary_source} failed: {e}")

        # Try fallback sources
        for source in self.fallback_sources:
            if source == self.primary_source:
                continue

            try:
                data = await self._fetch_data(
                    source, symbol, timeframe, start_date, end_date
                )
                if data:
                    self.logger.info(
                        f"Retrieved {len(data)} records from fallback source {source}"
                    )
                    return data
            except Exception as e:
                self.logger.warning(f"Fallback source {source} failed: {e}")

        raise DataProviderError(f"All data sources failed for {symbol}")

    async def get_realtime_data(self, symbol: str) -> MarketData:
        """Get current market data."""
        # For historical provider, return latest available data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        data = await self.get_historical_data(symbol, "1m", start_date, end_date)
        if not data:
            raise DataProviderError(f"No recent data available for {symbol}")

        return data[-1]  # Return most recent data point

    async def subscribe_to_data(self, symbols: List[str], callback) -> None:
        """Not implemented for historical provider."""
        raise NotImplementedError("Historical provider doesn't support real-time subscriptions")

    async def unsubscribe_from_data(self, symbols: List[str]) -> None:
        """Not implemented for historical provider."""
        raise NotImplementedError("Historical provider doesn't support real-time subscriptions")

    async def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        # This would typically come from the data provider's API
        # For now, return common crypto symbols
        return [
            "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD",
            "MATIC-USD", "AVAX-USD", "LUNA-USD", "ATOM-USD", "NEAR-USD"
        ]

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a symbol."""
        try:
            ticker = yf.Ticker(self._get_yahoo_symbol(symbol))
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "last_updated": datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    # Private methods

    async def _fetch_data(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """Fetch data from specified source."""
        if source not in self.source_configs:
            raise DataProviderError(f"Unsupported data source: {source}")

        await self._apply_rate_limit()

        config = self.source_configs[source]
        fetch_method = config["fetch_method"]

        return await fetch_method(symbol, timeframe, start_date, end_date)

    async def _fetch_from_yfinance(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """Fetch data from Yahoo Finance."""
        try:
            yahoo_symbol = self._get_yahoo_symbol(symbol)
            interval = self._convert_timeframe_to_yahoo(timeframe)

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(yahoo_symbol)

            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True,
                    threads=True
                )
            )

            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return []

            market_data_list = []
            for timestamp, row in df.iterrows():
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"])
                )
                market_data_list.append(market_data)

            return market_data_list

        except Exception as e:
            raise DataProviderError(f"Yahoo Finance fetch failed: {e}")

    def _get_yahoo_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format."""
        # Handle common crypto symbol conversions
        symbol_mappings = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "ADA": "ADA-USD",
            "DOT": "DOT-USD",
            "SOL": "SOL-USD",
            "MATIC": "MATIC-USD",
            "AVAX": "AVAX-USD",
            "LUNA": "LUNA1-USD",
            "ATOM": "ATOM-USD",
            "NEAR": "NEAR-USD"
        }

        # If symbol already has suffix, return as is
        if "-" in symbol:
            return symbol

        # Map to Yahoo format
        return symbol_mappings.get(symbol.upper(), f"{symbol.upper()}-USD")

    def _convert_timeframe_to_yahoo(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance interval."""
        mapping = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "60m": "1h",
            "1h": "1h",
            "90m": "90m",
            "1d": "1d",
            "5d": "5d",
            "1w": "1wk",
            "1mo": "1mo",
            "3mo": "3mo"
        }
        return mapping.get(timeframe, "1d")

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to prevent API abuse."""
        current_time = asyncio.get_event_loop().time()

        # Clean old timestamps (older than 1 minute)
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if current_time - ts < 60
        ]

        # Check if we're hitting rate limits
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self._request_timestamps[0])
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        # Add current request timestamp
        self._request_timestamps.append(current_time)

        # Apply basic delay
        if self.rate_limit_delay > 0:
            await asyncio.sleep(self.rate_limit_delay)

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available."""
        try:
            yahoo_symbol = self._get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: ticker.info)

            return bool(info and info.get("symbol"))
        except Exception:
            return False

    async def get_data_range(self, symbol: str) -> Dict[str, datetime]:
        """Get available data range for a symbol."""
        try:
            yahoo_symbol = self._get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)

            loop = asyncio.get_event_loop()
            history = await loop.run_in_executor(
                None,
                lambda: ticker.history(period="max", interval="1d")
            )

            if not history.empty:
                return {
                    "start_date": history.index[0].to_pydatetime(),
                    "end_date": history.index[-1].to_pydatetime()
                }
        except Exception as e:
            self.logger.error(f"Failed to get data range for {symbol}: {e}")

        return {"start_date": None, "end_date": None}