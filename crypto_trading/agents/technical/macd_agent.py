"""
MACD-based trading agent.
Uses Moving Average Convergence Divergence for trend following signals.
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Any
from loguru import logger

from ...core.interfaces import MarketData, TradingSignal, OrderSide
from ..base_agent import BaseAgent


class MACDAgent(BaseAgent):
    """Trading agent using MACD (Moving Average Convergence Divergence) strategy."""

    def __init__(self):
        super().__init__(
            name="MACD Agent",
            description="Uses MACD indicator for trend following and momentum analysis"
        )

    def get_required_parameters(self) -> List[str]:
        return ["fast_period", "slow_period", "signal_period"]

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "minimum_confidence": 0.5,
            "use_histogram": True,
            "divergence_detection": False
        }

    def _get_minimum_data_points(self) -> int:
        slow_period = self._get_config_value("slow_period", 26)
        signal_period = self._get_config_value("signal_period", 9)
        return slow_period + signal_period + 10

    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data and generate MACD-based trading signals."""
        self._ensure_initialized()
        self._validate_market_data(market_data)

        try:
            # Store current symbol for signal generation
            self._current_symbol = self._get_symbol_from_data(market_data)

            # Convert to DataFrame
            df = self._to_dataframe(market_data)

            # Calculate MACD
            fast_period = self._get_config_value("fast_period", 12)
            slow_period = self._get_config_value("slow_period", 26)
            signal_period = self._get_config_value("signal_period", 9)

            macd_data = ta.macd(df['close'], fast=fast_period, slow=slow_period, signal=signal_period)

            if macd_data is None or macd_data.empty:
                return self._create_neutral_signal(market_data)

            macd_line = macd_data[f'MACD_{fast_period}_{slow_period}_{signal_period}']
            signal_line = macd_data[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
            histogram = macd_data[f'MACDh_{fast_period}_{slow_period}_{signal_period}']

            if macd_line.isna().all() or signal_line.isna().all():
                return self._create_neutral_signal(market_data)

            # Get current and previous values
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]

            previous_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
            previous_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
            previous_histogram = histogram.iloc[-2] if len(histogram) > 1 else current_histogram

            # Generate signal
            signal = self._generate_macd_signal(
                current_macd, current_signal, current_histogram,
                previous_macd, previous_signal, previous_histogram, df
            )

            # Add metadata
            signal.metadata.update({
                "macd_current": float(current_macd),
                "signal_current": float(current_signal),
                "histogram_current": float(current_histogram),
                "macd_previous": float(previous_macd),
                "signal_previous": float(previous_signal),
                "histogram_previous": float(previous_histogram),
                "signal_type": "macd_crossover"
            })

            logger.debug(f"MACD Analysis: MACD={current_macd:.4f}, Signal={current_signal:.4f}, "
                        f"Histogram={current_histogram:.4f}, Action={signal.action}, Confidence={signal.confidence:.2f}")
            return signal

        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}")
            return self._create_neutral_signal(market_data)

    def _generate_macd_signal(
        self,
        current_macd: float, current_signal: float, current_histogram: float,
        previous_macd: float, previous_signal: float, previous_histogram: float,
        df: pd.DataFrame
    ) -> TradingSignal:
        """Generate trading signal based on MACD values."""
        # Extract symbol from stored symbol or default
        symbol = "UNKNOWN"
        if hasattr(self, '_current_symbol'):
            symbol = self._current_symbol

        current_price = float(df['close'].iloc[-1])

        # MACD Line crosses above Signal Line (Bullish)
        if current_macd > current_signal and previous_macd <= previous_signal:
            confidence = self._calculate_crossover_confidence(
                current_macd, current_signal, current_histogram, True
            )
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.BUY,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "macd_bullish_crossover"}
            )

        # MACD Line crosses below Signal Line (Bearish)
        elif current_macd < current_signal and previous_macd >= previous_signal:
            confidence = self._calculate_crossover_confidence(
                current_macd, current_signal, current_histogram, False
            )
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.SELL,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "macd_bearish_crossover"}
            )

        # Histogram analysis (if enabled)
        if self._get_config_value("use_histogram", True):
            histogram_signal = self._analyze_histogram(
                current_histogram, previous_histogram, current_macd, current_signal
            )
            if histogram_signal:
                confidence = self._calculate_histogram_confidence(current_histogram, previous_histogram)
                return self._create_signal(
                    symbol=symbol,
                    action=histogram_signal,
                    confidence=confidence,
                    price=current_price,
                    metadata={"signal_reason": "macd_histogram_divergence"}
                )

        # Zero line crossovers
        zero_line_signal = self._analyze_zero_line_crossover(
            current_macd, previous_macd
        )
        if zero_line_signal:
            confidence = self._calculate_zero_line_confidence(current_macd)
            return self._create_signal(
                symbol=symbol,
                action=zero_line_signal,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "macd_zero_line_crossover"}
            )

        # No clear signal
        return self._create_signal(
            symbol=symbol,
            action=OrderSide.BUY,  # Default action
            confidence=0.0,
            price=current_price,
            metadata={"signal_reason": "no_signal"}
        )

    def _calculate_crossover_confidence(
        self, macd: float, signal: float, histogram: float, is_bullish: bool
    ) -> float:
        """Calculate confidence for MACD crossover signals."""
        # Base confidence from the magnitude of the crossover
        crossover_magnitude = abs(macd - signal)

        # Normalize to 0-1 range (assuming typical MACD values)
        base_confidence = min(crossover_magnitude / 0.01, 1.0)

        # Adjust based on histogram direction
        if is_bullish:
            if histogram > 0:
                base_confidence *= 1.2  # Histogram confirms bullish signal
            else:
                base_confidence *= 0.8  # Histogram diverges from signal
        else:
            if histogram < 0:
                base_confidence *= 1.2  # Histogram confirms bearish signal
            else:
                base_confidence *= 0.8  # Histogram diverges from signal

        # Adjust based on MACD line position relative to zero
        if is_bullish and macd > 0:
            base_confidence *= 1.1  # Bullish signal above zero line
        elif not is_bullish and macd < 0:
            base_confidence *= 1.1  # Bearish signal below zero line

        return max(0.3, min(base_confidence, 0.9))

    def _analyze_histogram(
        self, current_histogram: float, previous_histogram: float,
        current_macd: float, current_signal: float
    ) -> OrderSide:
        """Analyze MACD histogram for early signals."""
        # Histogram turning positive (early bullish signal)
        if current_histogram > 0 and previous_histogram <= 0:
            return OrderSide.BUY

        # Histogram turning negative (early bearish signal)
        elif current_histogram < 0 and previous_histogram >= 0:
            return OrderSide.SELL

        # Histogram increasing while MACD is below signal (bullish divergence)
        elif (current_histogram > previous_histogram and
              current_macd < current_signal and
              current_histogram > -0.001):  # Close to turning positive
            return OrderSide.BUY

        # Histogram decreasing while MACD is above signal (bearish divergence)
        elif (current_histogram < previous_histogram and
              current_macd > current_signal and
              current_histogram < 0.001):  # Close to turning negative
            return OrderSide.SELL

        return None

    def _calculate_histogram_confidence(
        self, current_histogram: float, previous_histogram: float
    ) -> float:
        """Calculate confidence for histogram-based signals."""
        histogram_change = abs(current_histogram - previous_histogram)

        # Normalize the change (typical histogram changes are small)
        normalized_change = min(histogram_change / 0.005, 1.0)

        # Base confidence is lower for histogram signals as they're earlier
        base_confidence = normalized_change * 0.6

        return max(0.2, min(base_confidence, 0.7))

    def _analyze_zero_line_crossover(self, current_macd: float, previous_macd: float) -> OrderSide:
        """Analyze MACD zero line crossovers."""
        # MACD crosses above zero line (bullish)
        if current_macd > 0 and previous_macd <= 0:
            return OrderSide.BUY

        # MACD crosses below zero line (bearish)
        elif current_macd < 0 and previous_macd >= 0:
            return OrderSide.SELL

        return None

    def _calculate_zero_line_confidence(self, current_macd: float) -> float:
        """Calculate confidence for zero line crossover signals."""
        # Confidence based on how far MACD has moved from zero
        distance_from_zero = abs(current_macd)

        # Normalize (typical MACD values range from -0.1 to 0.1)
        normalized_distance = min(distance_from_zero / 0.01, 1.0)

        # Zero line crossovers are strong signals
        base_confidence = 0.5 + (normalized_distance * 0.3)

        return max(0.4, min(base_confidence, 0.8))

    def _create_neutral_signal(self, market_data: List[MarketData]) -> TradingSignal:
        """Create a neutral signal when analysis fails."""
        symbol = self._get_symbol_from_data(market_data)
        current_price = float(market_data[-1].close) if market_data else None

        return self._create_signal(
            symbol=symbol,
            action=OrderSide.BUY,
            confidence=0.0,
            price=current_price,
            metadata={"signal_reason": "analysis_failed"}
        )

    def _to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        data = []
        for item in market_data:
            data.append({
                'timestamp': item.timestamp,
                'open': float(item.open),
                'high': float(item.high),
                'low': float(item.low),
                'close': float(item.close),
                'volume': float(item.volume)
            })

        df = pd.DataFrame(data)
        return df.sort_values('timestamp').reset_index(drop=True)