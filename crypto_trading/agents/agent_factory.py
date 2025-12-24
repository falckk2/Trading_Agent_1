"""
Agent Factory for creating trading agents.
Follows Factory Pattern and Open/Closed Principle - new agents can be registered without modifying code.
"""

from typing import Dict, Type, Optional, Any, Callable
from ..core.interfaces import ITradingAgent, IStrategy, IExchangeClient, IRiskManager, ILogger
from ..core.exceptions import AgentInitializationError
from ..utils.logging import create_logger


class AgentFactory:
    """
    Factory for creating trading agent instances.
    Follows Open/Closed Principle - new agent types can be registered at runtime.
    """

    _agent_types: Dict[str, Callable] = {}
    _logger: ILogger = create_logger("AgentFactory")

    @classmethod
    def register_agent_type(
        cls,
        agent_type: str,
        creator: Callable[[Dict[str, Any]], ITradingAgent]
    ) -> None:
        """
        Register a new agent type with its creator function.

        Args:
            agent_type: Unique identifier for the agent type (e.g., 'rsi', 'macd', 'lstm')
            creator: Function that creates an agent instance from config

        Example:
            def create_rsi_agent(config: Dict) -> ITradingAgent:
                return RSIAgent(**config)

            AgentFactory.register_agent_type('rsi', create_rsi_agent)
        """
        if agent_type in cls._agent_types:
            cls._logger.warning(f"Agent type '{agent_type}' already registered, overwriting")

        cls._agent_types[agent_type] = creator
        cls._logger.info(f"Registered agent type: {agent_type}")

    @classmethod
    def create_agent(cls, agent_type: str, config: Optional[Dict[str, Any]] = None) -> ITradingAgent:
        """
        Create an agent instance.

        Args:
            agent_type: Type of agent to create
            config: Configuration for the agent

        Returns:
            ITradingAgent instance

        Raises:
            AgentInitializationError: If agent type not found or creation fails
        """
        if agent_type not in cls._agent_types:
            available_types = ', '.join(cls._agent_types.keys())
            raise AgentInitializationError(
                f"Unknown agent type: '{agent_type}'. "
                f"Available types: {available_types}"
            )

        try:
            config = config or {}
            agent = cls._agent_types[agent_type](config)
            cls._logger.info(f"Created agent of type '{agent_type}'")
            return agent

        except Exception as e:
            cls._logger.error(f"Failed to create agent of type '{agent_type}': {e}")
            raise AgentInitializationError(f"Agent creation failed: {e}")

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of registered agent types."""
        return list(cls._agent_types.keys())

    @classmethod
    def is_type_registered(cls, agent_type: str) -> bool:
        """Check if an agent type is registered."""
        return agent_type in cls._agent_types


# Builder class for easier agent construction
class AgentBuilder:
    """
    Builder for constructing agents with fluent interface.
    Helps create agents with all dependencies properly configured.
    """

    def __init__(self, agent_type: str):
        """
        Initialize builder for a specific agent type.

        Args:
            agent_type: Type of agent to build
        """
        self.agent_type = agent_type
        self._config: Dict[str, Any] = {}
        self._strategy: Optional[IStrategy] = None
        self._exchange: Optional[IExchangeClient] = None
        self._risk_manager: Optional[IRiskManager] = None
        self._portfolio_manager = None
        self._logger: Optional[ILogger] = None

    def with_config(self, config: Dict[str, Any]) -> 'AgentBuilder':
        """
        Set agent configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Self for chaining
        """
        self._config.update(config)
        return self

    def with_strategy(self, strategy: IStrategy) -> 'AgentBuilder':
        """
        Set agent strategy.

        Args:
            strategy: Strategy instance

        Returns:
            Self for chaining
        """
        self._strategy = strategy
        self._config['strategy'] = strategy
        return self

    def with_exchange(self, exchange: IExchangeClient) -> 'AgentBuilder':
        """
        Set exchange client.

        Args:
            exchange: Exchange client instance

        Returns:
            Self for chaining
        """
        self._exchange = exchange
        self._config['exchange'] = exchange
        return self

    def with_risk_manager(self, risk_manager: IRiskManager) -> 'AgentBuilder':
        """
        Set risk manager.

        Args:
            risk_manager: Risk manager instance

        Returns:
            Self for chaining
        """
        self._risk_manager = risk_manager
        self._config['risk_manager'] = risk_manager
        return self

    def with_portfolio_manager(self, portfolio_manager) -> 'AgentBuilder':
        """
        Set portfolio manager.

        Args:
            portfolio_manager: Portfolio manager instance

        Returns:
            Self for chaining
        """
        self._portfolio_manager = portfolio_manager
        self._config['portfolio_manager'] = portfolio_manager
        return self

    def with_logger(self, logger: ILogger) -> 'AgentBuilder':
        """
        Set logger.

        Args:
            logger: Logger instance

        Returns:
            Self for chaining
        """
        self._logger = logger
        self._config['logger'] = logger
        return self

    def with_strategy_params(self, params: Dict[str, Any]) -> 'AgentBuilder':
        """
        Set strategy parameters.

        Args:
            params: Strategy parameters

        Returns:
            Self for chaining
        """
        if 'strategy_params' not in self._config:
            self._config['strategy_params'] = {}
        self._config['strategy_params'].update(params)
        return self

    def with_agent_config(self, agent_config: Dict[str, Any]) -> 'AgentBuilder':
        """
        Set agent-specific configuration.

        Args:
            agent_config: Agent configuration

        Returns:
            Self for chaining
        """
        self._config['agent_config'] = agent_config
        return self

    def build(self) -> ITradingAgent:
        """
        Build the agent instance.

        Returns:
            Configured ITradingAgent instance

        Raises:
            AgentInitializationError: If agent creation fails
        """
        return AgentFactory.create_agent(self.agent_type, self._config)


# Convenience function for creating agents
def create_agent(agent_type: str, **kwargs) -> ITradingAgent:
    """
    Convenience function to create an agent with keyword arguments.

    Args:
        agent_type: Type of agent to create
        **kwargs: Configuration parameters

    Returns:
        ITradingAgent instance

    Example:
        agent = create_agent('rsi', period=14, overbought=70, oversold=30)
    """
    return AgentFactory.create_agent(agent_type, kwargs)
