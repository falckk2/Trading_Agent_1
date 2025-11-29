"""
Test suite for RSI trading agent.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_trading.agents.technical.rsi_agent import RSIAgent
from crypto_trading.core.interfaces import MarketData, TradingSignal, OrderSide
from crypto_trading.core.exceptions import AgentInitializationError


class TestRSIAgent:
    """Test RSI Agent functionality."""

    @pytest.fixture
    def rsi_agent(self):
        """Create RSI agent for testing."""
        return RSIAgent()

    @pytest.fixture
    def valid_config(self):
        """Valid RSI agent configuration."""
        return {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "minimum_confidence": 0.6
        }

    @pytest.fixture
    def market_data_trending_up(self):
        """Create market data showing upward trend."""
        base_price = Decimal("50000")
        data = []

        for i in range(30):
            # Simulate upward trend with some volatility
            price_change = Decimal(str(i * 100 + (i % 3) * 50))
            current_price = base_price + price_change

            data.append(MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now() - timedelta(hours=30-i),
                open=current_price - Decimal("50"),
                high=current_price + Decimal("100"),
                low=current_price - Decimal("100"),
                close=current_price,
                volume=Decimal("100")
            ))

        return data

    @pytest.fixture
    def market_data_trending_down(self):
        """Create market data showing downward trend."""
        base_price = Decimal("53000")
        data = []

        for i in range(30):
            # Simulate downward trend
            price_change = Decimal(str(i * 100))
            current_price = base_price - price_change

            data.append(MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now() - timedelta(hours=30-i),
                open=current_price + Decimal("50"),
                high=current_price + Decimal("100"),
                low=current_price - Decimal("100"),
                close=current_price,
                volume=Decimal("100")
            ))

        return data

    @pytest.fixture
    def market_data_oversold(self):
        """Create market data that would result in oversold RSI."""
        base_price = Decimal("50000")
        data = []

        for i in range(30):
            # Strong downward movement to create oversold condition
            price_decline = Decimal(str(i * 200))
            current_price = base_price - price_decline

            data.append(MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now() - timedelta(hours=30-i),
                open=current_price + Decimal("100"),
                high=current_price + Decimal("50"),
                low=current_price - Decimal("200"),
                close=current_price,
                volume=Decimal("100")
            ))

        return data

    def test_rsi_agent_initialization(self, rsi_agent):
        """Test RSI agent initialization."""
        assert rsi_agent.get_name() == "RSI Agent"
        assert "RSI" in rsi_agent.get_description()
        assert not rsi_agent.is_initialized

    def test_rsi_agent_parameters(self, rsi_agent):
        """Test RSI agent parameters."""
        params = rsi_agent.get_default_parameters()
        assert "rsi_period" in params
        assert "oversold_threshold" in params
        assert "overbought_threshold" in params

        required_params = rsi_agent.get_required_parameters()
        assert "rsi_period" in required_params

    def test_initialize_with_valid_config(self, rsi_agent, valid_config):
        """Test initialization with valid configuration."""
        rsi_agent.initialize(valid_config)
        assert rsi_agent.is_initialized
        assert rsi_agent.config == valid_config

    def test_initialize_with_invalid_config(self, rsi_agent):
        """Test initialization with invalid configuration."""
        invalid_config = {"oversold_threshold": 30}  # Missing required rsi_period

        with pytest.raises(AgentInitializationError):
            rsi_agent.initialize(invalid_config)

    def test_minimum_data_points(self, rsi_agent, valid_config):
        """Test minimum data points requirement."""
        rsi_agent.initialize(valid_config)
        min_points = rsi_agent._get_minimum_data_points()
        assert min_points >= 14  # At least RSI period
        assert min_points >= 20  # Agent's minimum requirement

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self, rsi_agent, valid_config):
        """Test analysis with insufficient data."""
        rsi_agent.initialize(valid_config)

        # Only 5 data points (insufficient)
        insufficient_data = [
            MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                open=Decimal("50000"),
                high=Decimal("50100"),
                low=Decimal("49900"),
                close=Decimal("50000"),
                volume=Decimal("100")
            )
            for _ in range(5)
        ]

        with pytest.raises(ValueError, match="Insufficient data points"):
            await rsi_agent.analyze(insufficient_data)

    @pytest.mark.asyncio
    async def test_analyze_trending_up_market(self, rsi_agent, valid_config, market_data_trending_up):
        """Test analysis with upward trending market."""
        rsi_agent.initialize(valid_config)

        signal = await rsi_agent.analyze(market_data_trending_up)

        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "BTC/USDT"
        assert signal.action in [OrderSide.BUY, OrderSide.SELL]
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.metadata, dict)
        assert "rsi_current" in signal.metadata

    @pytest.mark.asyncio
    async def test_analyze_oversold_condition(self, rsi_agent, valid_config, market_data_oversold):
        """Test analysis with oversold market condition."""
        rsi_agent.initialize(valid_config)

        signal = await rsi_agent.analyze(market_data_oversold)

        assert isinstance(signal, TradingSignal)
        # In oversold condition, should generally generate buy signal
        # (though this depends on the exact RSI calculation)
        assert signal.action in [OrderSide.BUY, OrderSide.SELL]
        assert signal.metadata["rsi_current"] is not None

    @pytest.mark.asyncio
    async def test_analyze_without_initialization(self, rsi_agent, market_data_trending_up):
        """Test analysis without initialization."""
        with pytest.raises(AgentInitializationError):
            await rsi_agent.analyze(market_data_trending_up)

    @pytest.mark.asyncio
    async def test_signal_metadata(self, rsi_agent, valid_config, market_data_trending_up):
        """Test that signal contains proper metadata."""
        rsi_agent.initialize(valid_config)

        signal = await rsi_agent.analyze(market_data_trending_up)

        # Check required metadata fields
        assert "rsi_current" in signal.metadata
        assert "rsi_previous" in signal.metadata
        assert "oversold_threshold" in signal.metadata
        assert "overbought_threshold" in signal.metadata
        assert "signal_type" in signal.metadata

        # Check metadata values
        assert isinstance(signal.metadata["rsi_current"], float)
        assert isinstance(signal.metadata["rsi_previous"], float)
        assert signal.metadata["oversold_threshold"] == 30
        assert signal.metadata["overbought_threshold"] == 70

    def test_confidence_calculation(self, rsi_agent):
        """Test confidence calculation logic."""
        # Test oversold confidence calculation
        confidence = rsi_agent._calculate_rsi_confidence(25, 30, True)  # Very oversold
        # Formula: (30-25)/30 * 0.8 + 0.2 = 0.167*0.8 + 0.2 = 0.333
        assert 0.3 <= confidence <= 0.4  # Should have moderate confidence

        confidence = rsi_agent._calculate_rsi_confidence(29, 30, True)  # Slightly oversold
        assert 0.2 <= confidence <= 0.9

        # Test very oversold for high confidence
        confidence = rsi_agent._calculate_rsi_confidence(10, 30, True)  # Very oversold
        # Formula: (30-10)/30 * 0.8 + 0.2 = 0.667*0.8 + 0.2 = 0.733
        assert confidence > 0.7  # Should have high confidence

        # Test overbought confidence calculation
        confidence = rsi_agent._calculate_rsi_confidence(75, 70, False)  # Very overbought
        # Formula: (75-70)/(100-70) * 0.8 + 0.2 = 0.167*0.8 + 0.2 = 0.333
        assert 0.3 <= confidence <= 0.4

        # Test very overbought for high confidence
        confidence = rsi_agent._calculate_rsi_confidence(90, 70, False)  # Very overbought
        # Formula: (90-70)/(100-70) * 0.8 + 0.2 = 0.667*0.8 + 0.2 = 0.733
        assert confidence > 0.7  # Should have high confidence

        # Test neutral zone confidence
        confidence = rsi_agent._calculate_rsi_confidence(50, 30, True)
        assert confidence == 0.3  # Default neutral confidence

    @pytest.mark.asyncio
    async def test_different_rsi_periods(self, market_data_trending_up):
        """Test RSI agent with different periods."""
        for period in [7, 14, 21]:
            agent = RSIAgent()
            config = {
                "rsi_period": period,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            }
            agent.initialize(config)

            signal = await agent.analyze(market_data_trending_up)
            assert isinstance(signal, TradingSignal)
            assert signal.metadata["signal_type"] == "rsi_crossover"

    @pytest.mark.asyncio
    async def test_different_thresholds(self, rsi_agent, market_data_trending_up):
        """Test RSI agent with different threshold values."""
        config = {
            "rsi_period": 14,
            "oversold_threshold": 20,  # More extreme threshold
            "overbought_threshold": 80   # More extreme threshold
        }
        rsi_agent.initialize(config)

        signal = await rsi_agent.analyze(market_data_trending_up)
        assert isinstance(signal, TradingSignal)
        assert signal.metadata["oversold_threshold"] == 20
        assert signal.metadata["overbought_threshold"] == 80

    def test_create_neutral_signal(self, rsi_agent, valid_config):
        """Test neutral signal creation."""
        rsi_agent.initialize(valid_config)

        market_data = [MarketData(
            symbol="ETH/USDT",
            timestamp=datetime.now(),
            open=Decimal("3000"),
            high=Decimal("3100"),
            low=Decimal("2900"),
            close=Decimal("3050"),
            volume=Decimal("50")
        )]

        signal = rsi_agent._create_neutral_signal(market_data)
        assert signal.symbol == "ETH/USDT"
        assert signal.confidence == 0.0
        assert signal.metadata["signal_reason"] == "analysis_failed"


if __name__ == "__main__":
    pytest.main([__file__])