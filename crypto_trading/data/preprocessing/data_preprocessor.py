"""
Data preprocessing module for cleaning and transforming market data.
Implements data validation, cleaning, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ...core.models import MarketData
from ...core.exceptions import DataValidationError


class DataPreprocessor:
    """
    Data preprocessor for market data cleaning and transformation.
    Provides various preprocessing techniques for improving data quality.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Validation thresholds
        self.max_price_deviation = self.config.get("max_price_deviation", 0.5)  # 50%
        self.min_volume = self.config.get("min_volume", 0)
        self.max_gap_minutes = self.config.get("max_gap_minutes", 60)

        # Cleaning parameters
        self.outlier_method = self.config.get("outlier_method", "iqr")
        self.outlier_threshold = self.config.get("outlier_threshold", 1.5)
        self.fill_method = self.config.get("fill_method", "forward")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            "max_price_deviation": 0.5,
            "min_volume": 0,
            "max_gap_minutes": 60,
            "outlier_method": "iqr",
            "outlier_threshold": 1.5,
            "fill_method": "forward",
            "validate_ohlc": True,
            "remove_duplicates": True,
            "normalize_symbols": True
        }

    async def process_historical_data(self, data: List[MarketData]) -> List[MarketData]:
        """
        Process historical market data with comprehensive cleaning.
        """
        if not data:
            return data

        try:
            self.logger.info(f"Processing {len(data)} historical data points")

            # Convert to DataFrame for easier processing
            df = self._to_dataframe(data)

            # Basic validation
            df = self._validate_basic_data(df)

            # Remove duplicates
            if self.config.get("remove_duplicates", True):
                df = self._remove_duplicates(df)

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Validate OHLC relationships
            if self.config.get("validate_ohlc", True):
                df = self._validate_ohlc(df)

            # Handle outliers
            df = self._handle_outliers(df)

            # Fill missing values
            df = self._fill_missing_values(df)

            # Normalize symbols
            if self.config.get("normalize_symbols", True):
                df = self._normalize_symbols(df)

            # Convert back to MarketData objects
            processed_data = self._from_dataframe(df)

            self.logger.info(
                f"Processed {len(processed_data)} data points "
                f"(removed {len(data) - len(processed_data)} invalid records)"
            )

            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing historical data: {e}")
            raise DataValidationError(f"Data processing failed: {e}")

    async def process_realtime_data(self, data: MarketData) -> MarketData:
        """
        Process real-time market data with basic validation.
        """
        try:
            # Basic validation
            self._validate_single_data_point(data)

            # Normalize symbol
            if self.config.get("normalize_symbols", True):
                data.symbol = self._normalize_symbol(data.symbol)

            # Validate OHLC for single data point
            if self.config.get("validate_ohlc", True):
                self._validate_single_ohlc(data)

            return data

        except Exception as e:
            self.logger.error(f"Error processing real-time data: {e}")
            raise DataValidationError(f"Real-time data processing failed: {e}")

    def calculate_technical_indicators(self, data: List[MarketData]) -> pd.DataFrame:
        """
        Calculate common technical indicators for the data.
        """
        if len(data) < 20:  # Minimum data points for indicators
            return pd.DataFrame()

        df = self._to_dataframe(data)

        try:
            # Simple Moving Averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # RSI
            df['rsi'] = self._calculate_rsi(df['close'])

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_calc = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_calc * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_calc * bb_std)

            # Price change and returns
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['close'].diff()

            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df

    def detect_anomalies(self, data: List[MarketData]) -> List[int]:
        """
        Detect anomalies in market data.
        Returns indices of anomalous data points.
        """
        if len(data) < 10:
            return []

        df = self._to_dataframe(data)
        anomaly_indices = []

        try:
            # Price jump detection
            price_changes = df['close'].pct_change().abs()
            price_threshold = price_changes.quantile(0.95)
            price_anomalies = price_changes > price_threshold

            # Volume spike detection
            volume_changes = df['volume'].pct_change().abs()
            volume_threshold = volume_changes.quantile(0.95)
            volume_anomalies = volume_changes > volume_threshold

            # Combine anomalies
            combined_anomalies = price_anomalies | volume_anomalies
            anomaly_indices = df[combined_anomalies].index.tolist()

            self.logger.info(f"Detected {len(anomaly_indices)} anomalies")

            return anomaly_indices

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []

    def create_features_for_ml(self, data: List[MarketData]) -> pd.DataFrame:
        """
        Create features suitable for machine learning models.
        """
        if len(data) < 50:
            return pd.DataFrame()

        # Get technical indicators
        df = self.calculate_technical_indicators(data)

        try:
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()

            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month

            # Interaction features
            df['price_volume_interaction'] = df['close'] * df['volume']
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']

            # Target variable (next period return)
            df['target'] = df['close'].shift(-1) / df['close'] - 1

            # Drop rows with NaN values
            df = df.dropna()

            self.logger.info(f"Created {df.shape[1]} features for {df.shape[0]} samples")

            return df

        except Exception as e:
            self.logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()

    # Private methods

    def _to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData list to DataFrame."""
        records = []
        for item in data:
            records.append({
                'symbol': item.symbol,
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume,
                'bid': item.bid,
                'ask': item.ask
            })

        return pd.DataFrame(records)

    def _from_dataframe(self, df: pd.DataFrame) -> List[MarketData]:
        """Convert DataFrame back to MarketData list."""
        data_list = []
        for _, row in df.iterrows():
            data_list.append(MarketData(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                bid=row.get('bid'),
                ask=row.get('ask')
            ))

        return data_list

    def _validate_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data validation."""
        initial_count = len(df)

        # Remove rows with null values in essential columns
        essential_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=essential_columns)

        # Remove rows with non-positive prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df = df[df[col] > 0]

        # Remove rows with negative volume
        df = df[df['volume'] >= 0]

        # Remove rows with extreme price deviations
        for col in price_columns:
            median_price = df[col].median()
            max_deviation = median_price * self.max_price_deviation
            df = df[
                (df[col] >= median_price - max_deviation) &
                (df[col] <= median_price + max_deviation)
            ]

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid data points")

        return df

    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships."""
        initial_count = len(df)

        # High should be >= Open, Close, Low
        df = df[df['high'] >= df['open']]
        df = df[df['high'] >= df['close']]
        df = df[df['high'] >= df['low']]

        # Low should be <= Open, Close, High
        df = df[df['low'] <= df['open']]
        df = df[df['low'] <= df['close']]
        df = df[df['low'] <= df['high']]

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} OHLC inconsistent records")

        return df

    def _validate_single_data_point(self, data: MarketData) -> None:
        """Validate a single data point."""
        if data.open <= 0 or data.high <= 0 or data.low <= 0 or data.close <= 0:
            raise DataValidationError("All prices must be positive")

        if data.volume < 0:
            raise DataValidationError("Volume cannot be negative")

        if not data.symbol:
            raise DataValidationError("Symbol is required")

    def _validate_single_ohlc(self, data: MarketData) -> None:
        """Validate OHLC relationships for a single data point."""
        if data.high < max(data.open, data.close, data.low):
            raise DataValidationError("High price must be >= Open, Close, Low")

        if data.low > min(data.open, data.close, data.high):
            raise DataValidationError("Low price must be <= Open, Close, High")

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['symbol', 'timestamp'])
        removed_count = initial_count - len(df)

        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate records")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        if self.outlier_method == "iqr":
            return self._remove_outliers_iqr(df)
        elif self.outlier_method == "zscore":
            return self._remove_outliers_zscore(df)
        else:
            return df

    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        initial_count = len(df)
        price_columns = ['open', 'high', 'low', 'close']

        for col in price_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outliers using IQR method")

        return df

    def _remove_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        from scipy import stats

        initial_count = len(df)
        price_columns = ['open', 'high', 'low', 'close']

        for col in price_columns:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < 3]  # Remove data points with z-score > 3

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outliers using Z-score method")

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the data."""
        if self.fill_method == "forward":
            df = df.fillna(method='ffill')
        elif self.fill_method == "backward":
            df = df.fillna(method='bfill')
        elif self.fill_method == "interpolate":
            df = df.interpolate()

        return df.dropna()  # Remove any remaining NaN values

    def _normalize_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize symbol names."""
        df['symbol'] = df['symbol'].apply(self._normalize_symbol)
        return df

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize a single symbol."""
        if not symbol:
            return symbol

        # Convert to uppercase and remove extra spaces
        symbol = symbol.upper().strip()

        # Common normalizations
        symbol_mappings = {
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "ADAUSD": "ADA-USD",
        }

        return symbol_mappings.get(symbol, symbol)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi