"""
Feature engineering for market data.
Implements technical indicators and feature extraction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from loguru import logger
import pandas_ta as ta

from ...core.interfaces import IDataProcessor, MarketData
from ...core.exceptions import DataProcessingError


class FeatureEngineer(IDataProcessor):
    """Feature engineering and technical indicator calculation."""

    def __init__(self):
        self.features_config = {
            'price_features': True,
            'volume_features': True,
            'technical_indicators': True,
            'momentum_indicators': True,
            'volatility_indicators': True,
            'trend_indicators': True
        }

    def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """Preprocess raw market data."""
        if not data:
            return data

        try:
            # Sort by timestamp
            sorted_data = sorted(data, key=lambda x: x.timestamp)

            # Remove duplicates
            seen_timestamps = set()
            unique_data = []
            for item in sorted_data:
                if item.timestamp not in seen_timestamps:
                    unique_data.append(item)
                    seen_timestamps.add(item.timestamp)

            # Fill missing values (if any gaps in data)
            filled_data = self._fill_missing_data(unique_data)

            logger.info(f"Preprocessed {len(data)} -> {len(filled_data)} data points")
            return filled_data

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise DataProcessingError(f"Preprocessing failed: {e}")

    def calculate_features(self, data: List[MarketData]) -> Dict[str, List[float]]:
        """Calculate features from market data."""
        if not data:
            return {}

        try:
            # Convert to DataFrame for easier processing
            df = self._to_dataframe(data)

            features = {}

            if self.features_config['price_features']:
                features.update(self._calculate_price_features(df))

            if self.features_config['volume_features']:
                features.update(self._calculate_volume_features(df))

            if self.features_config['technical_indicators']:
                features.update(self._calculate_technical_indicators(df))

            if self.features_config['momentum_indicators']:
                features.update(self._calculate_momentum_indicators(df))

            if self.features_config['volatility_indicators']:
                features.update(self._calculate_volatility_indicators(df))

            if self.features_config['trend_indicators']:
                features.update(self._calculate_trend_indicators(df))

            logger.info(f"Calculated {len(features)} feature sets")
            return features

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            raise DataProcessingError(f"Feature calculation failed: {e}")

    def _to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData list to pandas DataFrame."""
        df_data = []
        for item in data:
            df_data.append({
                'timestamp': item.timestamp,
                'open': float(item.open),
                'high': float(item.high),
                'low': float(item.low),
                'close': float(item.close),
                'volume': float(item.volume)
            })

        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _calculate_price_features(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate price-based features."""
        features = {}

        # Price changes
        features['price_change'] = df['close'].pct_change().fillna(0).tolist()
        features['price_change_log'] = np.log(df['close'] / df['close'].shift(1)).fillna(0).tolist()

        # OHLC ratios
        features['high_low_ratio'] = (df['high'] / df['low']).tolist()
        features['close_open_ratio'] = (df['close'] / df['open']).tolist()

        # Price position within range
        features['price_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5).tolist()

        # Moving averages
        for period in [5, 10, 20, 50]:
            ma_col = f'ma_{period}'
            features[ma_col] = df['close'].rolling(window=period).mean().fillna(df['close']).tolist()
            features[f'price_ma_{period}_ratio'] = (df['close'] / features[ma_col]).tolist()

        return features

    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate volume-based features."""
        features = {}

        # Volume changes
        features['volume_change'] = df['volume'].pct_change().fillna(0).tolist()

        # Volume moving averages
        for period in [5, 10, 20]:
            ma_vol = df['volume'].rolling(window=period).mean()
            features[f'volume_ma_{period}'] = ma_vol.fillna(df['volume']).tolist()
            features[f'volume_ratio_{period}'] = (df['volume'] / ma_vol).fillna(1).tolist()

        # Price-volume relationship
        features['price_volume_trend'] = (df['close'].pct_change() * df['volume']).fillna(0).tolist()

        return features

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate technical indicators using pandas-ta."""
        features = {}

        # RSI
        rsi = ta.rsi(df['close'], length=14)
        features['rsi'] = rsi.fillna(50).tolist()

        # MACD
        macd_data = ta.macd(df['close'])
        if macd_data is not None:
            features['macd'] = macd_data['MACD_12_26_9'].fillna(0).tolist()
            features['macd_signal'] = macd_data['MACDs_12_26_9'].fillna(0).tolist()
            features['macd_histogram'] = macd_data['MACDh_12_26_9'].fillna(0).tolist()

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            features['stoch_k'] = stoch['STOCHk_14_3_3'].fillna(50).tolist()
            features['stoch_d'] = stoch['STOCHd_14_3_3'].fillna(50).tolist()

        # Williams %R
        willr = ta.willr(df['high'], df['low'], df['close'])
        features['williams_r'] = willr.fillna(-50).tolist()

        return features

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate momentum indicators."""
        features = {}

        # Rate of Change
        for period in [5, 10, 20]:
            roc = ta.roc(df['close'], length=period)
            features[f'roc_{period}'] = roc.fillna(0).tolist()

        # Momentum
        for period in [5, 10]:
            mom = ta.mom(df['close'], length=period)
            features[f'momentum_{period}'] = mom.fillna(0).tolist()

        # Commodity Channel Index
        cci = ta.cci(df['high'], df['low'], df['close'])
        features['cci'] = cci.fillna(0).tolist()

        return features

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate volatility indicators."""
        features = {}

        # Bollinger Bands
        bb = ta.bbands(df['close'])
        if bb is not None:
            features['bb_upper'] = bb['BBU_5_2.0'].fillna(df['close']).tolist()
            features['bb_middle'] = bb['BBM_5_2.0'].fillna(df['close']).tolist()
            features['bb_lower'] = bb['BBL_5_2.0'].fillna(df['close']).tolist()
            features['bb_width'] = ((bb['BBU_5_2.0'] - bb['BBL_5_2.0']) / bb['BBM_5_2.0']).fillna(0).tolist()
            features['bb_position'] = ((df['close'] - bb['BBL_5_2.0']) / (bb['BBU_5_2.0'] - bb['BBL_5_2.0'])).fillna(0.5).tolist()

        # Average True Range
        atr = ta.atr(df['high'], df['low'], df['close'])
        features['atr'] = atr.fillna(0).tolist()

        # True Range
        tr = ta.true_range(df['high'], df['low'], df['close'])
        features['true_range'] = tr.fillna(0).tolist()

        return features

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate trend indicators."""
        features = {}

        # ADX (Average Directional Index)
        adx_data = ta.adx(df['high'], df['low'], df['close'])
        if adx_data is not None:
            features['adx'] = adx_data['ADX_14'].fillna(25).tolist()
            features['di_plus'] = adx_data['DMP_14'].fillna(25).tolist()
            features['di_minus'] = adx_data['DMN_14'].fillna(25).tolist()

        # Parabolic SAR
        psar = ta.psar(df['high'], df['low'], df['close'])
        if psar is not None:
            features['psar'] = psar['PSARl_0.02_0.2'].fillna(df['close']).tolist()

        # Linear regression slope
        for period in [10, 20]:
            lr_slope = ta.linreg(df['close'], length=period, slope=True)
            features[f'lr_slope_{period}'] = lr_slope.fillna(0).tolist()

        return features

    def _fill_missing_data(self, data: List[MarketData]) -> List[MarketData]:
        """Fill missing data points using forward fill method."""
        if len(data) <= 1:
            return data

        filled_data = [data[0]]

        for i in range(1, len(data)):
            current = data[i]
            previous = filled_data[-1]

            # Check for significant time gaps (more than expected interval)
            time_diff = (current.timestamp - previous.timestamp).total_seconds()

            # If gap is reasonable, just add the data point
            if time_diff <= 3600:  # 1 hour or less
                filled_data.append(current)
            else:
                # For larger gaps, we might want to interpolate or just use the data as-is
                # For now, just add the current data point
                filled_data.append(current)

        return filled_data

    def create_feature_matrix(self, data: List[MarketData], lookback_periods: int = 20) -> np.ndarray:
        """Create feature matrix for machine learning models."""
        try:
            features = self.calculate_features(data)

            if not features:
                return np.array([])

            # Convert to matrix format
            feature_names = list(features.keys())
            feature_matrix = np.column_stack([features[name] for name in feature_names])

            # Create lookback sequences for time series modeling
            if lookback_periods > 1 and len(feature_matrix) > lookback_periods:
                sequences = []
                for i in range(lookback_periods, len(feature_matrix)):
                    sequence = feature_matrix[i-lookback_periods:i].flatten()
                    sequences.append(sequence)

                return np.array(sequences)
            else:
                return feature_matrix

        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            raise DataProcessingError(f"Feature matrix creation failed: {e}")

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # This is a static list of all features that can be calculated
        feature_names = []

        # Price features
        feature_names.extend([
            'price_change', 'price_change_log', 'high_low_ratio', 'close_open_ratio', 'price_position'
        ])

        # Moving averages
        for period in [5, 10, 20, 50]:
            feature_names.extend([f'ma_{period}', f'price_ma_{period}_ratio'])

        # Volume features
        feature_names.extend(['volume_change', 'price_volume_trend'])
        for period in [5, 10, 20]:
            feature_names.extend([f'volume_ma_{period}', f'volume_ratio_{period}'])

        # Technical indicators
        feature_names.extend([
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r'
        ])

        # Momentum indicators
        for period in [5, 10, 20]:
            feature_names.append(f'roc_{period}')
        for period in [5, 10]:
            feature_names.append(f'momentum_{period}')
        feature_names.append('cci')

        # Volatility indicators
        feature_names.extend([
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'true_range'
        ])

        # Trend indicators
        feature_names.extend([
            'adx', 'di_plus', 'di_minus', 'psar'
        ])
        for period in [10, 20]:
            feature_names.append(f'lr_slope_{period}')

        return feature_names