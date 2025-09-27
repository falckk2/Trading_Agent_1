"""
Moving Average trading agent implementation.
Uses multiple moving averages for trend following and crossover signals.
"""

import pandas as pd
from typing import Dict, List, Optional, Any

from ...core.models import MarketData, TradingSignal, SignalType
from ..base_agent import BaseAgent as BaseTradingAgent
from .technical_strategy import TechnicalStrategy


class MovingAverageStrategy(TechnicalStrategy):
    """
    Moving Average-based trading strategy.
    Supports various MA types and crossover strategies.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            "fast_period": 10,
            "slow_period": 20,
            "ma_type": "sma",  # sma, ema, wma
            "crossover_signals": True,
            "trend_filter": True,
            "price_above_ma": True,  # Price position relative to MA
            "multiple_timeframes": False,
            "long_period": 50  # For triple MA system
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(default_params)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages and supporting indicators."""
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters.get('ma_type', 'sma')

        # Calculate fast and slow moving averages
        if ma_type == 'sma':
            df['ma_fast'] = self._calculate_sma(df['close'], fast_period)
            df['ma_slow'] = self._calculate_sma(df['close'], slow_period)
        elif ma_type == 'ema':
            df['ma_fast'] = self._calculate_ema(df['close'], fast_period)
            df['ma_slow'] = self._calculate_ema(df['close'], slow_period)
        else:  # Default to SMA
            df['ma_fast'] = self._calculate_sma(df['close'], fast_period)
            df['ma_slow'] = self._calculate_sma(df['close'], slow_period)

        # Calculate long-term MA for triple MA system
        if self.parameters.get('multiple_timeframes', False):
            long_period = self.parameters.get('long_period', 50)
            if ma_type == 'sma':
                df['ma_long'] = self._calculate_sma(df['close'], long_period)
            else:
                df['ma_long'] = self._calculate_ema(df['close'], long_period)

        # Calculate previous values for crossover detection
        df['ma_fast_prev'] = df['ma_fast'].shift(1)
        df['ma_slow_prev'] = df['ma_slow'].shift(1)

        # Calculate MA spread and slope
        df['ma_spread'] = df['ma_fast'] - df['ma_slow']
        df['ma_spread_prev'] = df['ma_spread'].shift(1)

        # Calculate MA slope for trend strength
        df['ma_fast_slope'] = df['ma_fast'].diff(periods=3)
        df['ma_slow_slope'] = df['ma_slow'].diff(periods=3)

        return df

    def _generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal based on moving average analysis."""
        if len(df) < 2:
            return {'type': SignalType.HOLD, 'strength': 0.0, 'confidence': 0.0}

        latest = df.iloc[-1]

        # Check if we have valid MA data
        if pd.isna(latest['ma_fast']) or pd.isna(latest['ma_slow']):
            return {'type': SignalType.HOLD, 'strength': 0.0, 'confidence': 0.0}

        signal_type = SignalType.HOLD
        strength = 0.0
        confidence = 0.0

        metadata = {
            'ma_fast': latest['ma_fast'],
            'ma_slow': latest['ma_slow'],
            'ma_spread': latest.get('ma_spread', 0),
            'price': latest['close']
        }

        # Check for crossover signals
        if self.parameters.get('crossover_signals', True):
            crossover_signal = self._check_ma_crossover(latest)
            if crossover_signal['found']:
                signal_type = crossover_signal['type']
                strength = crossover_signal['strength']
                confidence = crossover_signal['confidence']
                metadata.update(crossover_signal['metadata'])

        # Check price position relative to MAs
        if self.parameters.get('price_above_ma', True):
            price_signal = self._check_price_ma_relationship(latest)
            if price_signal['found']:
                if signal_type == SignalType.HOLD:
                    signal_type = price_signal['type']
                    strength = price_signal['strength']
                    confidence = price_signal['confidence']
                elif signal_type == price_signal['type']:
                    # Strengthen signal if both agree
                    strength = min(1.0, strength * 1.2)
                    confidence = min(1.0, confidence * 1.1)

                metadata.update(price_signal['metadata'])

        # Apply trend filter
        if self.parameters.get('trend_filter', True):
            trend_strength = self._calculate_trend_strength(df.tail(10))
            metadata['trend_strength'] = trend_strength

            # Adjust confidence based on trend strength
            if signal_type != SignalType.HOLD:
                if ((signal_type == SignalType.BUY and trend_strength > 0) or
                    (signal_type == SignalType.SELL and trend_strength < 0)):
                    confidence = min(1.0, confidence * (1 + abs(trend_strength)))
                else:
                    confidence *= 0.7  # Reduce confidence if against trend

        # Check triple MA alignment for additional confirmation
        if self.parameters.get('multiple_timeframes', False) and 'ma_long' in latest:
            triple_ma_signal = self._check_triple_ma_alignment(latest)
            if triple_ma_signal['aligned']:
                metadata.update(triple_ma_signal['metadata'])
                if signal_type == triple_ma_signal['type']:
                    confidence = min(1.0, confidence * 1.3)

        return {
            'type': signal_type,
            'strength': strength,
            'confidence': confidence,
            'metadata': metadata
        }

    def _check_ma_crossover(self, latest: pd.Series) -> Dict[str, Any]:
        """Check for moving average crossover signals."""
        if (pd.isna(latest['ma_fast_prev']) or pd.isna(latest['ma_slow_prev'])):
            return {'found': False}

        ma_fast = latest['ma_fast']
        ma_slow = latest['ma_slow']
        ma_fast_prev = latest['ma_fast_prev']
        ma_slow_prev = latest['ma_slow_prev']

        # Golden cross: Fast MA crosses above slow MA
        if ma_fast_prev <= ma_slow_prev and ma_fast > ma_slow:
            # Calculate crossover strength
            crossover_magnitude = abs(ma_fast - ma_slow) / ma_slow
            strength = min(1.0, max(0.5, crossover_magnitude * 100))

            return {
                'found': True,
                'type': SignalType.BUY,
                'strength': strength,
                'confidence': 0.8,
                'metadata': {
                    'crossover_type': 'golden_cross',
                    'crossover_magnitude': crossover_magnitude
                }
            }

        # Death cross: Fast MA crosses below slow MA
        elif ma_fast_prev >= ma_slow_prev and ma_fast < ma_slow:
            crossover_magnitude = abs(ma_fast - ma_slow) / ma_slow
            strength = min(1.0, max(0.5, crossover_magnitude * 100))

            return {
                'found': True,
                'type': SignalType.SELL,
                'strength': strength,
                'confidence': 0.8,
                'metadata': {
                    'crossover_type': 'death_cross',
                    'crossover_magnitude': crossover_magnitude
                }
            }

        return {'found': False}

    def _check_price_ma_relationship(self, latest: pd.Series) -> Dict[str, Any]:
        """Check price position relative to moving averages."""
        price = latest['close']
        ma_fast = latest['ma_fast']
        ma_slow = latest['ma_slow']

        # Price above both MAs (bullish)
        if price > ma_fast and price > ma_slow and ma_fast > ma_slow:
            distance_from_ma = ((price - ma_slow) / ma_slow)
            strength = min(1.0, max(0.3, distance_from_ma * 20))

            return {
                'found': True,
                'type': SignalType.BUY,
                'strength': strength,
                'confidence': 0.6,
                'metadata': {
                    'signal_reason': 'price_above_mas',
                    'distance_from_slow_ma': distance_from_ma
                }
            }

        # Price below both MAs (bearish)
        elif price < ma_fast and price < ma_slow and ma_fast < ma_slow:
            distance_from_ma = ((ma_slow - price) / ma_slow)
            strength = min(1.0, max(0.3, distance_from_ma * 20))

            return {
                'found': True,
                'type': SignalType.SELL,
                'strength': strength,
                'confidence': 0.6,
                'metadata': {
                    'signal_reason': 'price_below_mas',
                    'distance_from_slow_ma': distance_from_ma
                }
            }

        return {'found': False}

    def _calculate_trend_strength(self, recent_df: pd.DataFrame) -> float:
        """Calculate trend strength based on MA slopes."""
        if len(recent_df) < 5:
            return 0.0

        # Get recent slope values
        fast_slopes = recent_df['ma_fast_slope'].dropna()
        slow_slopes = recent_df['ma_slow_slope'].dropna()

        if len(fast_slopes) == 0 or len(slow_slopes) == 0:
            return 0.0

        # Average slopes
        avg_fast_slope = fast_slopes.mean()
        avg_slow_slope = slow_slopes.mean()

        # Combine slopes for overall trend strength
        # Positive = bullish, negative = bearish
        trend_strength = (avg_fast_slope + avg_slow_slope) / 2

        # Normalize to reasonable range
        return max(-1.0, min(1.0, trend_strength * 1000))

    def _check_triple_ma_alignment(self, latest: pd.Series) -> Dict[str, Any]:
        """Check alignment of three moving averages."""
        if 'ma_long' not in latest or pd.isna(latest['ma_long']):
            return {'aligned': False}

        ma_fast = latest['ma_fast']
        ma_slow = latest['ma_slow']
        ma_long = latest['ma_long']

        # Bullish alignment: fast > slow > long
        if ma_fast > ma_slow > ma_long:
            return {
                'aligned': True,
                'type': SignalType.BUY,
                'metadata': {
                    'ma_alignment': 'bullish',
                    'ma_long': ma_long
                }
            }

        # Bearish alignment: fast < slow < long
        elif ma_fast < ma_slow < ma_long:
            return {
                'aligned': True,
                'type': SignalType.SELL,
                'metadata': {
                    'ma_alignment': 'bearish',
                    'ma_long': ma_long
                }
            }

        return {'aligned': False}


class MovingAverageAgent(BaseTradingAgent):
    """
    Moving Average-based trading agent.
    Uses moving average strategy for trend following signals.
    """

    def __init__(self, strategy=None, exchange=None, risk_manager=None,
                 portfolio_manager=None, agent_config=None):
        # Create Moving Average strategy if not provided
        if strategy is None:
            strategy_params = agent_config.get('strategy_params', {}) if agent_config else {}
            strategy = MovingAverageStrategy(strategy_params)

        super().__init__(strategy, exchange, risk_manager, portfolio_manager, agent_config)

    async def _initialize_agent(self) -> None:
        """Moving Average agent specific initialization."""
        self.logger.info("Initializing Moving Average trading agent")

        # Validate configuration
        strategy_params = self.strategy.get_parameters()

        fast_period = strategy_params.get('fast_period', 10)
        slow_period = strategy_params.get('slow_period', 20)
        ma_type = strategy_params.get('ma_type', 'sma')

        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")

        if fast_period < 1 or slow_period < 1:
            raise ValueError("MA periods must be positive")

        if ma_type not in ['sma', 'ema']:
            self.logger.warning(f"Unknown MA type '{ma_type}', defaulting to SMA")

        self.logger.info(f"Moving Average agent initialized: {ma_type.upper()} "
                        f"fast={fast_period}, slow={slow_period}")

    async def _process_market_data(self, market_data: List[MarketData]) -> Optional[TradingSignal]:
        """Process market data using Moving Average strategy."""
        try:
            signal = self.strategy.analyze(market_data)

            # Log signal details
            if signal.signal_type != SignalType.HOLD:
                metadata = signal.metadata
                self.logger.info(
                    f"MA Signal: {signal.symbol} {signal.signal_type.value} "
                    f"strength={signal.strength:.2f} confidence={signal.confidence:.2f} "
                    f"fast_MA={metadata.get('ma_fast', 'N/A'):.2f} "
                    f"slow_MA={metadata.get('ma_slow', 'N/A'):.2f} "
                    f"price={metadata.get('price', 'N/A'):.2f}"
                )

                if 'crossover_type' in metadata:
                    self.logger.info(f"Crossover: {metadata['crossover_type']}")

            return signal

        except Exception as e:
            self.logger.error(f"Error processing market data with MA strategy: {e}")
            return None

    def _get_strategy_metrics(self) -> Dict[str, float]:
        """Get Moving Average-specific metrics."""
        strategy_params = self.strategy.get_parameters()

        metrics = {
            "fast_period": strategy_params.get('fast_period', 10),
            "slow_period": strategy_params.get('slow_period', 20),
            "crossover_signals_enabled": 1.0 if strategy_params.get('crossover_signals', True) else 0.0,
            "trend_filter_enabled": 1.0 if strategy_params.get('trend_filter', True) else 0.0,
            "price_above_ma_enabled": 1.0 if strategy_params.get('price_above_ma', True) else 0.0,
            "multiple_timeframes_enabled": 1.0 if strategy_params.get('multiple_timeframes', False) else 0.0
        }

        # Add MA type as numeric value
        ma_type = strategy_params.get('ma_type', 'sma')
        metrics["ma_type_sma"] = 1.0 if ma_type == 'sma' else 0.0
        metrics["ma_type_ema"] = 1.0 if ma_type == 'ema' else 0.0

        return metrics