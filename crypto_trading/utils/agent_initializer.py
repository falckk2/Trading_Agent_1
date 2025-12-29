"""
Agent initialization utility.
Automatically registers and configures trading agents based on configuration.
"""

from typing import Dict, Any
from loguru import logger

from ..core.interfaces import IAgentManager, IConfigManager
from ..agents.technical.rsi_agent import RSIAgent
from ..agents.technical.macd_agent import MACDAgent
from ..agents.technical.bollinger_bands_agent import BollingerBandsAgent
from ..agents.ml.random_forest_agent import RandomForestAgent
from ..agents.meta.consensus_agent import ConsensusAgent
from ..agents.meta.validator_agent import ValidatorAgent


def initialize_agents(agent_manager: IAgentManager, config_manager: IConfigManager) -> None:
    """
    Initialize and register all available trading agents based on configuration.

    Args:
        agent_manager: Agent manager instance
        config_manager: Configuration manager instance
    """
    logger.info("Initializing trading agents...")

    # Get agent configurations
    agents_config = config_manager.get_section('agents')

    # Initialize RSI Agent
    try:
        rsi_agent = RSIAgent()
        rsi_config = agents_config.get('rsi', {})
        agent_manager.register_agent(rsi_agent, rsi_config)
        logger.info("RSI Agent registered")
    except Exception as e:
        logger.error(f"Failed to register RSI Agent: {e}")

    # Initialize MACD Agent
    try:
        macd_agent = MACDAgent()
        macd_config = agents_config.get('macd', {})
        agent_manager.register_agent(macd_agent, macd_config)
        logger.info("MACD Agent registered")
    except Exception as e:
        logger.error(f"Failed to register MACD Agent: {e}")

    # Initialize Bollinger Bands Agent
    try:
        bb_agent = BollingerBandsAgent()
        bb_config = agents_config.get('bollinger_bands', {})
        agent_manager.register_agent(bb_agent, bb_config)
        logger.info("Bollinger Bands Agent registered")
    except Exception as e:
        logger.error(f"Failed to register Bollinger Bands Agent: {e}")

    # Initialize Random Forest Agent
    try:
        rf_agent = RandomForestAgent()
        rf_config = agents_config.get('random_forest', {})
        agent_manager.register_agent(rf_agent, rf_config)
        logger.info("Random Forest Agent registered")
    except Exception as e:
        logger.error(f"Failed to register Random Forest Agent: {e}")

    # Initialize Meta-Agents (Consensus and Validator)
    # These agents use the base agents registered above

    # Get registered base agents for meta-agent composition
    base_agents = []
    for agent_name in ['RSI Agent', 'MACD Agent', 'Random Forest Agent']:
        try:
            agent = agent_manager.get_agent(agent_name)
            base_agents.append(agent)
        except Exception:
            pass  # Agent not registered, skip it

    # Initialize Consensus Agent
    if len(base_agents) >= 2:
        try:
            consensus_agent = ConsensusAgent()
            consensus_agent.set_sub_agents(base_agents)
            # Ensure config dict exists for meta-agents with testing-friendly defaults
            consensus_config = agents_config.get('consensus', {})
            if not consensus_config:
                consensus_config = {
                    'minimum_agreement': 0.5,
                    'minimum_confidence': 0.3,  # Lowered for testing
                    'equal_weights': True
                }
            agent_manager.register_agent(consensus_agent, consensus_config)
            logger.info(f"Consensus Agent registered with {len(base_agents)} sub-agents")
        except Exception as e:
            logger.error(f"Failed to register Consensus Agent: {e}")

    # Initialize Validator Agent (RSI as primary, MACD + RF as validators)
    if len(base_agents) >= 2:
        try:
            validator_agent = ValidatorAgent()

            # Set RSI as primary agent if available
            rsi_agent = None
            validators = []
            for agent in base_agents:
                if agent.get_name() == 'RSI Agent':
                    rsi_agent = agent
                else:
                    validators.append(agent)

            if rsi_agent and validators:
                validator_agent.set_primary_agent(rsi_agent)
                validator_agent.set_validator_agents(validators)
                # Ensure config dict exists for meta-agents with testing-friendly defaults
                validator_config = agents_config.get('validator', {})
                if not validator_config:
                    validator_config = {
                        'validation_mode': 'majority',
                        'min_validator_confidence': 0.3,  # Lowered for testing
                        'require_same_action': True,
                        'confidence_boost': 0.1
                    }
                agent_manager.register_agent(validator_agent, validator_config)
                logger.info(
                    f"Validator Agent registered with primary '{rsi_agent.get_name()}' "
                    f"and {len(validators)} validators"
                )
        except Exception as e:
            logger.error(f"Failed to register Validator Agent: {e}")

    # Set default active agent
    default_agent = config_manager.get('trading.default_agent', 'RSI Agent')
    available_agents = agent_manager.list_agents()

    if available_agents:
        # Try to set the configured default agent
        if default_agent in available_agents:
            agent_manager.set_active_agent(default_agent)
            logger.info(f"Active agent set to: {default_agent}")
        else:
            # Fall back to first available agent
            first_agent = available_agents[0]
            agent_manager.set_active_agent(first_agent)
            logger.warning(f"Default agent '{default_agent}' not found, using '{first_agent}'")
    else:
        logger.error("No agents were successfully registered!")

    logger.info(f"Agent initialization complete. {len(available_agents)} agents registered.")


def get_agent_status(agent_manager: IAgentManager) -> Dict[str, Any]:
    """
    Get current agent status information.

    Args:
        agent_manager: Agent manager instance

    Returns:
        Dictionary with agent status information
    """
    available_agents = agent_manager.list_agents()
    active_agent = agent_manager.get_active_agent()
    active_agent_name = active_agent.get_name() if active_agent else None

    return {
        'available_agents': available_agents,
        'active_agent': active_agent_name,
        'total_agents': len(available_agents)
    }
