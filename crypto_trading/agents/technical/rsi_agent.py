"""
RSI-based trading agent.
Uses Relative Strength Index for buy/sell signal generation.
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Any, Optional
from loguru import logger

from ...core.interfaces import MarketData, TradingSignal, OrderSide
from ..base_agent import BaseAgent


class RSIAgent(BaseAgent):
    """Trading agent using RSI (Relative Strength Index) strategy."""

    def __init__(self):
        super().__init__(
            name="RSI Agent",
            description="Uses RSI indicator to generate buy/sell signals when price is oversold/overbought"
        )

    def get_required_parameters(self) -> List[str]:
        return ["rsi_period"]

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "minimum_confidence": 0.6,
            "use_divergence": False
        }

    def _get_minimum_data_points(self) -> int:
        return max(self._get_config_value("rsi_period", 14) + 5, 20)

    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data and generate RSI-based trading signals."""
        self._ensure_initialized()
        self._validate_market_data(market_data)

        try:
            # Store current symbol for signal generation
            self._current_symbol = self._get_symbol_from_data(market_data)

            # Convert to DataFrame
            df = self._to_dataframe(market_data)

            # Calculate RSI
            rsi_period = self._get_config_value("rsi_period", 14)
            rsi_values = ta.rsi(df['close'], length=rsi_period)

            if rsi_values is None or rsi_values.isna().all():
                return self._create_neutral_signal(market_data)

            current_rsi = rsi_values.iloc[-1]
            previous_rsi = rsi_values.iloc[-2] if len(rsi_values) > 1 else current_rsi

            # Get thresholds
            oversold_threshold = self._get_config_value("oversold_threshold", 30)
            overbought_threshold = self._get_config_value("overbought_threshold", 70)

            # Generate signal
            signal = self._generate_rsi_signal(
                current_rsi, previous_rsi, oversold_threshold, overbought_threshold, df
            )

            # Add metadata
            signal.metadata.update({
                "rsi_current": float(current_rsi),
                "rsi_previous": float(previous_rsi),
                "oversold_threshold": oversold_threshold,
                "overbought_threshold": overbought_threshold,
                "signal_type": "rsi_crossover"
            })

            logger.debug(f"RSI Analysis: RSI={current_rsi:.2f}, Signal={signal.action}, Confidence={signal.confidence:.2f}")
            return signal

        except Exception as e:
            logger.error(f"Error in RSI analysis: {e}")
            return self._create_neutral_signal(market_data)

    def _generate_rsi_signal(
        self,
        current_rsi: float,
        previous_rsi: float,
        oversold_threshold: float,
        overbought_threshold: float,
        df: pd.DataFrame
    ) -> TradingSignal:
        """Generate trading signal based on RSI values."""
        # Extract symbol from the first market data point if available
        symbol = "UNKNOWN"
        if hasattr(self, '_current_symbol'):
            symbol = self._current_symbol

        current_price = float(df['close'].iloc[-1])

        # Check for oversold condition (buy signal)
        if current_rsi <= oversold_threshold and previous_rsi > current_rsi:
            # RSI is oversold and declining - potential reversal
            confidence = self._calculate_rsi_confidence(current_rsi, oversold_threshold, True)
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.BUY,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "oversold_reversal"}
            )

        elif current_rsi <= oversold_threshold and current_rsi > previous_rsi:
            # RSI is oversold but starting to rise - strong buy signal
            confidence = self._calculate_rsi_confidence(current_rsi, oversold_threshold, True)
            confidence += 0.1  # Bonus for rising from oversold
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.BUY,
                confidence=min(confidence, 1.0),
                price=current_price,
                metadata={"signal_reason": "oversold_rising"}
            )

        # Check for overbought condition (sell signal)
        elif current_rsi >= overbought_threshold and previous_rsi < current_rsi:
            # RSI is overbought and rising - potential reversal
            confidence = self._calculate_rsi_confidence(current_rsi, overbought_threshold, False)
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.SELL,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "overbought_reversal"}
            )

        elif current_rsi >= overbought_threshold and current_rsi < previous_rsi:
            # RSI is overbought but starting to fall - strong sell signal
            confidence = self._calculate_rsi_confidence(current_rsi, overbought_threshold, False)
            confidence += 0.1  # Bonus for falling from overbought
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.SELL,
                confidence=min(confidence, 1.0),
                price=current_price,
                metadata={"signal_reason": "overbought_falling"}
            )

        # Check for neutral zone crossovers
        elif previous_rsi < 50 and current_rsi >= 50:
            # Bullish momentum shift
            confidence = 0.4
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.BUY,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "bullish_momentum"}
            )

        elif previous_rsi > 50 and current_rsi <= 50:
            # Bearish momentum shift
            confidence = 0.4
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.SELL,
                confidence=confidence,
                price=current_price,
                metadata={"signal_reason": "bearish_momentum"}
            )

        # No clear signal
        return self._create_signal(
            symbol=symbol,
            action=OrderSide.BUY,  # Default action
            confidence=0.0,
            price=current_price,
            metadata={"signal_reason": "no_signal"}
        )

    def _calculate_rsi_confidence(self, rsi_value: float, threshold: float, is_buy_signal: bool) -> float:
        """Calculate confidence based on how extreme the RSI value is."""
        if is_buy_signal:
            # For buy signals, confidence increases as RSI gets lower (more oversold)
            if rsi_value <= threshold:
                # Max confidence when RSI is very low
                confidence = (threshold - rsi_value) / threshold
                return min(max(confidence * 0.8 + 0.2, 0.2), 0.9)
        else:
            # For sell signals, confidence increases as RSI gets higher (more overbought)
            if rsi_value >= threshold:
                # Max confidence when RSI is very high
                confidence = (rsi_value - threshold) / (100 - threshold)
                return min(max(confidence * 0.8 + 0.2, 0.2), 0.9)

        return 0.3  # Default confidence for neutral zones

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