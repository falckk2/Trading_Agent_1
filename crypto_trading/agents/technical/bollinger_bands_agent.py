"""
Bollinger Bands trading agent implementation.
Uses Bollinger Bands for mean reversion and volatility-based signals.
"""

import pandas as pd
from typing import Dict, List, Optional, Any

from ...core.models import MarketData, TradingSignal, SignalType
from ..base_agent import BaseAgent as BaseTradingAgent
from .technical_strategy import TechnicalStrategy


class BollingerBandsStrategy(TechnicalStrategy):
    """
    Bollinger Bands-based trading strategy.
    Generates signals based on price position relative to bands and band squeeze/expansion.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            "period": 20,
            "std_dev": 2.0,
            "band_touch_threshold": 0.02,  # Within 2% of band
            "squeeze_threshold": 0.05,  # Band width ratio for squeeze detection
            "expansion_threshold": 0.15,  # Band width ratio for expansion
            "mean_reversion": True,  # Use mean reversion signals
            "breakout_signals": True  # Use breakout signals
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(default_params)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and supporting indicators."""
        period = self.parameters['period']
        std_dev = self.parameters['std_dev']

        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(
            df['close'], period, std_dev
        )

        df['bb_upper'] = upper_band
        df['bb_middle'] = middle_band
        df['bb_lower'] = lower_band

        # Calculate band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Calculate %B (position within bands)
        df['percent_b'] = df['bb_position']

        # Calculate band width percentile for squeeze/expansion detection
        df['bb_width_percentile'] = df['bb_width'].rolling(window=50).rank(pct=True)

        # Previous values for change detection
        df['bb_width_prev'] = df['bb_width'].shift(1)
        df['close_prev'] = df['close'].shift(1)

        return df

    def _generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal based on Bollinger Bands analysis."""
        if len(df) < 2:
            return {'type': SignalType.HOLD, 'strength': 0.0, 'confidence': 0.0}

        latest = df.iloc[-1]

        # Check if we have valid data
        required_fields = ['bb_upper', 'bb_lower', 'bb_middle', 'bb_position', 'bb_width']
        if any(pd.isna(latest[field]) for field in required_fields):
            return {'type': SignalType.HOLD, 'strength': 0.0, 'confidence': 0.0}

        signal_type = SignalType.HOLD
        strength = 0.0
        confidence = 0.0

        metadata = {
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'bb_middle': latest['bb_middle'],
            'bb_position': latest['bb_position'],
            'bb_width': latest['bb_width'],
            'percent_b': latest.get('percent_b', 0)
        }

        # Check for mean reversion signals
        if self.parameters.get('mean_reversion', True):
            reversion_signal = self._check_mean_reversion_signal(latest)
            if reversion_signal['found']:
                signal_type = reversion_signal['type']
                strength = reversion_signal['strength']
                confidence = reversion_signal['confidence']
                metadata.update(reversion_signal['metadata'])

        # Check for breakout signals
        if self.parameters.get('breakout_signals', True):
            breakout_signal = self._check_breakout_signal(df.tail(5))
            if breakout_signal['found']:
                if signal_type == SignalType.HOLD:
                    signal_type = breakout_signal['type']
                    strength = breakout_signal['strength']
                    confidence = breakout_signal['confidence']
                elif signal_type == breakout_signal['type']:
                    # Strengthen signal if both agree
                    strength = min(1.0, strength * 1.2)
                    confidence = min(1.0, confidence * 1.1)

                metadata.update(breakout_signal['metadata'])

        # Check for squeeze/expansion patterns
        squeeze_signal = self._check_squeeze_expansion(df.tail(10))
        if squeeze_signal['found']:
            metadata.update(squeeze_signal['metadata'])
            # Adjust confidence based on volatility conditions
            if squeeze_signal['type'] == 'post_squeeze':
                confidence = min(1.0, confidence * 1.3)

        return {
            'type': signal_type,
            'strength': strength,
            'confidence': confidence,
            'metadata': metadata
        }

    def _check_mean_reversion_signal(self, latest: pd.Series) -> Dict[str, Any]:
        """Check for mean reversion signals near bands."""
        threshold = self.parameters.get('band_touch_threshold', 0.02)
        bb_position = latest['bb_position']

        # Oversold condition (near lower band)
        if bb_position <= threshold:
            # Strength increases as price gets closer to lower band
            strength = max(0.5, 1.0 - (bb_position / threshold))

            return {
                'found': True,
                'type': SignalType.BUY,
                'strength': strength,
                'confidence': 0.7,
                'metadata': {
                    'signal_reason': 'mean_reversion_oversold',
                    'distance_from_lower_band': bb_position
                }
            }

        # Overbought condition (near upper band)
        elif bb_position >= (1.0 - threshold):
            # Strength increases as price gets closer to upper band
            strength = max(0.5, (bb_position - (1.0 - threshold)) / threshold)

            return {
                'found': True,
                'type': SignalType.SELL,
                'strength': strength,
                'confidence': 0.7,
                'metadata': {
                    'signal_reason': 'mean_reversion_overbought',
                    'distance_from_upper_band': 1.0 - bb_position
                }
            }

        return {'found': False}

    def _check_breakout_signal(self, recent_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for breakout signals through bands."""
        if len(recent_df) < 2:
            return {'found': False}

        latest = recent_df.iloc[-1]
        previous = recent_df.iloc[-2]

        current_close = latest['close']
        prev_close = previous['close']

        # Bullish breakout above upper band
        if (prev_close <= previous['bb_upper'] and
            current_close > latest['bb_upper']):

            # Calculate breakout strength
            breakout_magnitude = (current_close - latest['bb_upper']) / latest['bb_middle']
            strength = min(1.0, max(0.6, breakout_magnitude * 10))

            return {
                'found': True,
                'type': SignalType.BUY,
                'strength': strength,
                'confidence': 0.8,
                'metadata': {
                    'signal_reason': 'bullish_breakout',
                    'breakout_magnitude': breakout_magnitude,
                    'breakout_price': current_close,
                    'upper_band': latest['bb_upper']
                }
            }

        # Bearish breakout below lower band
        elif (prev_close >= previous['bb_lower'] and
              current_close < latest['bb_lower']):

            breakout_magnitude = (latest['bb_lower'] - current_close) / latest['bb_middle']
            strength = min(1.0, max(0.6, breakout_magnitude * 10))

            return {
                'found': True,
                'type': SignalType.SELL,
                'strength': strength,
                'confidence': 0.8,
                'metadata': {
                    'signal_reason': 'bearish_breakout',
                    'breakout_magnitude': breakout_magnitude,
                    'breakout_price': current_close,
                    'lower_band': latest['bb_lower']
                }
            }

        return {'found': False}

    def _check_squeeze_expansion(self, recent_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for Bollinger Band squeeze and expansion patterns."""
        if len(recent_df) < 5:
            return {'found': False}

        latest = recent_df.iloc[-1]
        squeeze_threshold = self.parameters.get('squeeze_threshold', 0.05)
        expansion_threshold = self.parameters.get('expansion_threshold', 0.15)

        current_width = latest['bb_width']
        avg_width = recent_df['bb_width'].mean()

        # Check for squeeze (low volatility)
        if current_width < squeeze_threshold:
            return {
                'found': True,
                'type': 'squeeze',
                'metadata': {
                    'volatility_condition': 'squeeze',
                    'current_width': current_width,
                    'squeeze_threshold': squeeze_threshold
                }
            }

        # Check for expansion (high volatility)
        elif current_width > expansion_threshold:
            return {
                'found': True,
                'type': 'expansion',
                'metadata': {
                    'volatility_condition': 'expansion',
                    'current_width': current_width,
                    'expansion_threshold': expansion_threshold
                }
            }

        # Check for post-squeeze breakout
        elif avg_width < squeeze_threshold and current_width > avg_width * 1.5:
            return {
                'found': True,
                'type': 'post_squeeze',
                'metadata': {
                    'volatility_condition': 'post_squeeze_breakout',
                    'width_increase': current_width / avg_width
                }
            }

        return {'found': False}


class BollingerBandsAgent(BaseTradingAgent):
    """
    Bollinger Bands-based trading agent.
    Uses Bollinger Bands strategy for volatility-based signal generation.
    """

    def __init__(self, strategy=None, exchange=None, risk_manager=None,
                 portfolio_manager=None, agent_config=None):
        # Create Bollinger Bands strategy if not provided
        if strategy is None:
            strategy_params = agent_config.get('strategy_params', {}) if agent_config else {}
            strategy = BollingerBandsStrategy(strategy_params)

        super().__init__(strategy, exchange, risk_manager, portfolio_manager, agent_config)

    async def _initialize_agent(self) -> None:
        """Bollinger Bands agent specific initialization."""
        self.logger.info("Initializing Bollinger Bands trading agent")

        # Validate configuration
        strategy_params = self.strategy.get_parameters()

        period = strategy_params.get('period', 20)
        std_dev = strategy_params.get('std_dev', 2.0)

        if period < 2:
            raise ValueError("Bollinger Bands period must be at least 2")

        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")

        self.logger.info(f"Bollinger Bands agent initialized with period={period}, std_dev={std_dev}")

    async def _process_market_data(self, market_data: List[MarketData]) -> Optional[TradingSignal]:
        """Process market data using Bollinger Bands strategy."""
        try:
            signal = self.strategy.analyze(market_data)

            # Log signal details
            if signal.signal_type != SignalType.HOLD:
                metadata = signal.metadata
                self.logger.info(
                    f"BB Signal: {signal.symbol} {signal.signal_type.value} "
                    f"strength={signal.strength:.2f} confidence={signal.confidence:.2f} "
                    f"BB_pos={metadata.get('bb_position', 'N/A'):.3f} "
                    f"width={metadata.get('bb_width', 'N/A'):.4f}"
                )

                if 'signal_reason' in metadata:
                    self.logger.info(f"Signal reason: {metadata['signal_reason']}")

            return signal

        except Exception as e:
            self.logger.error(f"Error processing market data with Bollinger Bands strategy: {e}")
            return None

    def _get_strategy_metrics(self) -> Dict[str, float]:
        """Get Bollinger Bands-specific metrics."""
        strategy_params = self.strategy.get_parameters()

        return {
            "bb_period": strategy_params.get('period', 20),
            "bb_std_dev": strategy_params.get('std_dev', 2.0),
            "band_touch_threshold": strategy_params.get('band_touch_threshold', 0.02),
            "squeeze_threshold": strategy_params.get('squeeze_threshold', 0.05),
            "expansion_threshold": strategy_params.get('expansion_threshold', 0.15),
            "mean_reversion_enabled": 1.0 if strategy_params.get('mean_reversion', True) else 0.0,
            "breakout_signals_enabled": 1.0 if strategy_params.get('breakout_signals', True) else 0.0
        }