"""
Test suite for AgentManager functionality.
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal

from crypto_trading.core.agent_manager import AgentManager
from crypto_trading.core.interfaces import MarketData, TradingSignal, OrderSide
from crypto_trading.agents.technical.rsi_agent import RSIAgent
from crypto_trading.agents.technical.macd_agent import MACDAgent
from crypto_trading.utils.exceptions import AgentNotFoundError, AgentInitializationError


class TestAgentManager:
    """Test AgentManager functionality."""

    @pytest.fixture
    def agent_manager(self):
        """Create AgentManager instance for testing."""
        return AgentManager()

    @pytest.fixture
    def rsi_agent(self):
        """Create RSI agent for testing."""
        return RSIAgent()

    @pytest.fixture
    def macd_agent(self):
        """Create MACD agent for testing."""
        return MACDAgent()

    @pytest.fixture
    def sample_config(self):
        """Sample agent configuration."""
        return {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70
        }

    def test_agent_manager_initialization(self, agent_manager):
        """Test AgentManager initialization."""
        assert agent_manager._agents == {}
        assert agent_manager._active_agent is None
        assert agent_manager._agent_configs == {}

    def test_register_agent(self, agent_manager, rsi_agent, sample_config):
        """Test agent registration."""
        # Register agent without config
        agent_manager.register_agent(rsi_agent)
        assert "RSI Agent" in agent_manager._agents
        assert agent_manager._agents["RSI Agent"] == rsi_agent

        # Register agent with config
        macd_agent = MACDAgent()
        macd_config = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        agent_manager.register_agent(macd_agent, macd_config)
        assert "MACD Agent" in agent_manager._agents
        assert "MACD Agent" in agent_manager._agent_configs

    def test_register_agent_with_invalid_config(self, agent_manager, rsi_agent):
        """Test agent registration with invalid config."""
        invalid_config = {}  # Missing required parameters

        with pytest.raises(AgentInitializationError):
            agent_manager.register_agent(rsi_agent, invalid_config)

    def test_unregister_agent(self, agent_manager, rsi_agent):
        """Test agent unregistration."""
        # Register agent first
        agent_manager.register_agent(rsi_agent)
        assert "RSI Agent" in agent_manager._agents

        # Unregister agent
        agent_manager.unregister_agent("RSI Agent")
        assert "RSI Agent" not in agent_manager._agents

    def test_unregister_nonexistent_agent(self, agent_manager):
        """Test unregistering non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            agent_manager.unregister_agent("NonExistent Agent")

    def test_set_active_agent(self, agent_manager, rsi_agent):
        """Test setting active agent."""
        # Register agent first
        agent_manager.register_agent(rsi_agent)

        # Set as active
        agent_manager.set_active_agent("RSI Agent")
        assert agent_manager._active_agent == "RSI Agent"

    def test_set_active_nonexistent_agent(self, agent_manager):
        """Test setting non-existent agent as active."""
        with pytest.raises(AgentNotFoundError):
            agent_manager.set_active_agent("NonExistent Agent")

    def test_get_active_agent(self, agent_manager, rsi_agent):
        """Test getting active agent."""
        # No active agent
        assert agent_manager.get_active_agent() is None

        # Register and set active agent
        agent_manager.register_agent(rsi_agent)
        agent_manager.set_active_agent("RSI Agent")

        active_agent = agent_manager.get_active_agent()
        assert active_agent == rsi_agent

    def test_get_agent(self, agent_manager, rsi_agent):
        """Test getting specific agent."""
        agent_manager.register_agent(rsi_agent)

        retrieved_agent = agent_manager.get_agent("RSI Agent")
        assert retrieved_agent == rsi_agent

    def test_get_nonexistent_agent(self, agent_manager):
        """Test getting non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            agent_manager.get_agent("NonExistent Agent")

    def test_list_agents(self, agent_manager, rsi_agent, macd_agent):
        """Test listing all agents."""
        # No agents initially
        assert agent_manager.list_agents() == []

        # Register agents
        agent_manager.register_agent(rsi_agent)
        agent_manager.register_agent(macd_agent)

        agents = agent_manager.list_agents()
        assert len(agents) == 2
        assert "RSI Agent" in agents
        assert "MACD Agent" in agents

    def test_get_agent_info(self, agent_manager, rsi_agent, sample_config):
        """Test getting agent information."""
        agent_manager.register_agent(rsi_agent, sample_config)
        agent_manager.set_active_agent("RSI Agent")

        info = agent_manager.get_agent_info("RSI Agent")
        assert info["name"] == "RSI Agent"
        assert info["description"] is not None
        assert info["is_active"] is True
        assert info["config"] == sample_config

    def test_get_agent_status(self, agent_manager, rsi_agent, macd_agent):
        """Test getting agent status."""
        agent_manager.register_agent(rsi_agent)
        agent_manager.register_agent(macd_agent)
        agent_manager.set_active_agent("RSI Agent")

        status = agent_manager.get_agent_status()
        assert status["total_agents"] == 2
        assert status["active_agent"] == "RSI Agent"
        assert len(status["agents"]) == 2

    def test_update_agent_config(self, agent_manager, rsi_agent, sample_config):
        """Test updating agent configuration."""
        agent_manager.register_agent(rsi_agent, sample_config)

        new_config = {
            "rsi_period": 21,
            "oversold_threshold": 25,
            "overbought_threshold": 75
        }

        agent_manager.update_agent_config("RSI Agent", new_config)

        info = agent_manager.get_agent_info("RSI Agent")
        assert info["config"] == new_config

    @pytest.mark.asyncio
    async def test_analyze_with_active_agent(self, agent_manager, rsi_agent, sample_config):
        """Test analyzing with active agent."""
        # Create sample market data
        market_data = [
            MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("100")
            )
            for _ in range(20)  # Need enough data for RSI calculation
        ]

        agent_manager.register_agent(rsi_agent, sample_config)
        agent_manager.set_active_agent("RSI Agent")

        signal = await agent_manager.analyze_with_active_agent(market_data)
        assert signal is not None
        assert isinstance(signal, TradingSignal)

    @pytest.mark.asyncio
    async def test_analyze_with_no_active_agent(self, agent_manager):
        """Test analyzing with no active agent."""
        market_data = []
        signal = await agent_manager.analyze_with_active_agent(market_data)
        assert signal is None

    @pytest.mark.asyncio
    async def test_analyze_with_specific_agent(self, agent_manager, rsi_agent, sample_config):
        """Test analyzing with specific agent."""
        # Create sample market data
        market_data = [
            MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("100")
            )
            for _ in range(20)
        ]

        agent_manager.register_agent(rsi_agent, sample_config)

        signal = await agent_manager.analyze_with_agent("RSI Agent", market_data)
        assert signal is not None
        assert isinstance(signal, TradingSignal)


if __name__ == "__main__":
    pytest.main([__file__])