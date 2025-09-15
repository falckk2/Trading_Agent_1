"""
Base technical analysis strategy implementation.
Provides common functionality for technical indicators.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ...core.interfaces import IStrategy
from ...core.models import MarketData, TradingSignal, SignalType
from ...core.exceptions import StrategyException


class TechnicalStrategy(IStrategy, ABC):
    """
    Base class for technical analysis strategies.
    Provides common technical indicator calculations and utilities.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.name = self.__class__.__name__

        # Common parameters
        self.min_data_points = self.parameters.get("min_data_points", 50)
        self.signal_threshold = self.parameters.get("signal_threshold", 0.7)

        # Risk management parameters
        self.stop_loss_pct = self.parameters.get("stop_loss_pct", 0.05)  # 5%
        self.take_profit_pct = self.parameters.get("take_profit_pct", 0.10)  # 10%

    def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """
        Analyze market data and generate trading signals.
        Template method that uses strategy-specific analysis.
        """
        if len(market_data) < self.min_data_points:
            return self._create_hold_signal(market_data[-1] if market_data else None)

        try:
            # Convert to DataFrame for analysis
            df = self._market_data_to_dataframe(market_data)

            # Calculate technical indicators
            df = self._calculate_indicators(df)

            # Generate signal
            signal_info = self._generate_signal(df)

            # Create trading signal
            return self._create_signal(market_data[-1], signal_info)

        except Exception as e:
            raise StrategyException(f"Analysis failed: {e}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        self.parameters.update(parameters)

    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal."""
        if not signal:
            return False

        # Basic validation
        if signal.strength < 0 or signal.strength > 1:
            return False

        if signal.confidence < 0 or signal.confidence > 1:
            return False

        # Strategy-specific validation
        return self._validate_signal_specific(signal)

    # Abstract methods for subclasses

    @abstractmethod
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific technical indicators."""
        pass

    @abstractmethod
    def _generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal based on indicators."""
        pass

    # Protected methods for subclasses

    def _validate_signal_specific(self, signal: TradingSignal) -> bool:
        """Strategy-specific signal validation."""
        return True

    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def _create_signal(
        self,
        current_data: MarketData,
        signal_info: Dict[str, Any]
    ) -> TradingSignal:
        """Create a trading signal from analysis results."""
        signal_type = signal_info.get('type', SignalType.HOLD)
        strength = signal_info.get('strength', 0.0)
        confidence = signal_info.get('confidence', 0.0)

        # Calculate stop loss and take profit levels
        current_price = current_data.close
        stop_loss = None
        take_profit = None

        if signal_type == SignalType.BUY:
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif signal_type == SignalType.SELL:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        return TradingSignal(
            symbol=current_data.symbol,
            signal_type=signal_type,
            strength=strength,
            price=current_price,
            strategy_name=self.name,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=signal_info.get('metadata', {})
        )

    def _create_hold_signal(self, current_data: Optional[MarketData]) -> TradingSignal:
        """Create a HOLD signal."""
        if not current_data:
            # Create a dummy signal when no data is available
            return TradingSignal(
                symbol="UNKNOWN",
                signal_type=SignalType.HOLD,
                strength=0.0,
                price=0.0,
                strategy_name=self.name,
                confidence=0.0
            )

        return TradingSignal(
            symbol=current_data.symbol,
            signal_type=SignalType.HOLD,
            strength=0.0,
            price=current_data.close,
            strategy_name=self.name,
            confidence=1.0,
            metadata={"reason": "insufficient_data"}
        )

    # Technical indicator utility methods

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)

        macd = ema_fast - ema_slow
        signal = self._calculate_ema(macd, signal_period)
        histogram = macd - signal

        return macd, signal, histogram

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, sma, lower_band

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def _detect_support_resistance(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels."""
        support_levels = []
        resistance_levels = []

        for i in range(window, len(prices) - window):
            # Check for local minima (support)
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                support_levels.append(prices.iloc[i])

            # Check for local maxima (resistance)
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                resistance_levels.append(prices.iloc[i])

        return support_levels, resistance_levels

    def _calculate_volume_profile(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        bins: int = 20
    ) -> Dict[str, Any]:
        """Calculate volume profile."""
        price_min, price_max = prices.min(), prices.max()
        price_bins = np.linspace(price_min, price_max, bins)

        volume_at_price = []
        for i in range(len(price_bins) - 1):
            mask = (prices >= price_bins[i]) & (prices < price_bins[i + 1])
            volume_at_price.append(volumes[mask].sum())

        # Find Point of Control (POC) - price level with highest volume
        max_volume_idx = np.argmax(volume_at_price)
        poc = (price_bins[max_volume_idx] + price_bins[max_volume_idx + 1]) / 2

        return {
            'price_bins': price_bins,
            'volume_at_price': volume_at_price,
            'poc': poc,
            'total_volume': volumes.sum()
        }

    def _is_trend_bullish(self, prices: pd.Series, period: int = 20) -> bool:
        """Determine if trend is bullish based on moving average slope."""
        ma = self._calculate_sma(prices, period)
        if len(ma) < 2:
            return False

        # Calculate slope of moving average
        recent_ma = ma.iloc[-5:].values  # Last 5 values
        if len(recent_ma) < 2:
            return False

        slope = np.polyfit(range(len(recent_ma)), recent_ma, 1)[0]
        return slope > 0

    def _is_trend_bearish(self, prices: pd.Series, period: int = 20) -> bool:
        """Determine if trend is bearish based on moving average slope."""
        return not self._is_trend_bullish(prices, period)

    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate price volatility."""
        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0

        return returns.rolling(window=period).std().iloc[-1] * np.sqrt(252)  # Annualized

    def _normalize_strength(self, raw_value: float, min_val: float, max_val: float) -> float:
        """Normalize a raw indicator value to strength between 0 and 1."""
        if max_val == min_val:
            return 0.5

        normalized = (raw_value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))