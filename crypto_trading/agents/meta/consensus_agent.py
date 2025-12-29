"""
Consensus Agent - Combines signals from multiple agents using weighted voting.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from collections import Counter

from ...core.interfaces import MarketData, TradingSignal, OrderSide
from ..base_agent import BaseAgent


class ConsensusAgent(BaseAgent):
    """
    Meta-agent that combines signals from multiple sub-agents using consensus voting.

    The consensus agent analyzes signals from all sub-agents and:
    1. Counts votes for each action (BUY/SELL/HOLD)
    2. Applies optional weights to each agent
    3. Combines confidence scores
    4. Returns the majority decision
    """

    def __init__(self, sub_agents: Optional[List[BaseAgent]] = None):
        """
        Initialize the Consensus Agent.

        Args:
            sub_agents: List of agents to use for consensus (can be set later)
        """
        super().__init__(
            name="Consensus Agent",
            description="Combines signals from multiple agents using weighted voting"
        )
        self.sub_agents: List[BaseAgent] = sub_agents or []
        self.agent_weights: Dict[str, float] = {}  # Optional weights per agent

    def set_sub_agents(self, agents: List[BaseAgent]) -> None:
        """Set the sub-agents to use for consensus."""
        self.sub_agents = agents
        logger.info(f"Consensus Agent configured with {len(agents)} sub-agents: {[a.get_name() for a in agents]}")

    def set_agent_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for each agent.

        Args:
            weights: Dictionary mapping agent names to their weights
        """
        self.agent_weights = weights
        logger.info(f"Agent weights set: {weights}")

    def get_required_parameters(self) -> List[str]:
        return []  # No required parameters

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "minimum_agreement": 0.5,  # Minimum fraction of agents that must agree
            "minimum_confidence": 0.3,  # Minimum combined confidence threshold (lowered for testing)
            "equal_weights": True  # If True, all agents have equal weight
        }

    def _get_minimum_data_points(self) -> int:
        # Use the maximum requirement from all sub-agents
        if not self.sub_agents:
            return 20
        return max(agent._get_minimum_data_points() for agent in self.sub_agents)

    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """
        Analyze market data using consensus from multiple agents.

        Args:
            market_data: Market data to analyze

        Returns:
            Trading signal based on consensus
        """
        self._ensure_initialized()

        if not self.sub_agents:
            logger.warning("Consensus Agent has no sub-agents configured")
            return self._create_neutral_signal(market_data)

        self._validate_market_data(market_data)

        try:
            # Store current symbol
            self._current_symbol = self._get_symbol_from_data(market_data)

            # Get signals from all sub-agents
            signals: Dict[str, TradingSignal] = {}
            for agent in self.sub_agents:
                try:
                    signal = await agent.analyze(market_data)
                    signals[agent.get_name()] = signal
                except Exception as e:
                    logger.error(f"Error getting signal from {agent.get_name()}: {e}")

            if not signals:
                logger.warning("No signals received from sub-agents")
                return self._create_neutral_signal(market_data)

            # Apply consensus logic
            consensus_signal = self._calculate_consensus(signals, market_data)

            logger.info(
                f"🟣 Consensus Agent → {consensus_signal.action.value.upper()} {consensus_signal.symbol} "
                f"(confidence: {consensus_signal.confidence * 100:.1f}%) "
                f"from {len(signals)} agents"
            )

            return consensus_signal

        except Exception as e:
            logger.error(f"Error in Consensus Agent analysis: {e}")
            return self._create_neutral_signal(market_data)

    def _calculate_consensus(
        self,
        signals: Dict[str, TradingSignal],
        market_data: List[MarketData]
    ) -> TradingSignal:
        """
        Calculate consensus from multiple signals.

        Args:
            signals: Dictionary of agent names to their signals
            market_data: Market data for creating signal

        Returns:
            Consensus trading signal
        """
        # Get configuration
        minimum_agreement = self._get_config_value("minimum_agreement", 0.5)
        minimum_confidence = self._get_config_value("minimum_confidence", 0.6)
        equal_weights = self._get_config_value("equal_weights", True)

        # Count votes for each action
        votes: Dict[OrderSide, float] = {
            OrderSide.BUY: 0.0,
            OrderSide.SELL: 0.0
        }

        total_weight = 0.0
        confidence_sum = 0.0

        for agent_name, signal in signals.items():
            # Get weight for this agent
            if equal_weights:
                weight = 1.0
            else:
                weight = self.agent_weights.get(agent_name, 1.0)

            # Count the vote
            if signal.action == OrderSide.BUY:
                votes[OrderSide.BUY] += weight
            elif signal.action == OrderSide.SELL:
                votes[OrderSide.SELL] += weight
            # HOLD votes don't count toward action

            # Accumulate confidence
            confidence_sum += signal.confidence * weight
            total_weight += weight

        # Calculate winning action
        winning_action = OrderSide.BUY if votes[OrderSide.BUY] > votes[OrderSide.SELL] else OrderSide.SELL

        # Calculate agreement level
        winning_votes = votes[winning_action]
        agreement_level = winning_votes / total_weight if total_weight > 0 else 0

        # Calculate average confidence
        avg_confidence = confidence_sum / total_weight if total_weight > 0 else 0

        # Check if consensus meets minimum requirements
        if agreement_level < minimum_agreement or avg_confidence < minimum_confidence:
            # Not enough agreement or confidence - return HOLD
            logger.debug(
                f"Consensus insufficient: agreement={agreement_level:.2f}, "
                f"confidence={avg_confidence:.2f}"
            )
            return self._create_neutral_signal(market_data)

        # Create consensus signal
        signal = TradingSignal(
            symbol=self._current_symbol,
            action=winning_action,
            confidence=avg_confidence,
            price=market_data[-1].close,
            metadata={
                "consensus_type": "weighted_voting",
                "agreement_level": agreement_level,
                "votes": {
                    "buy": votes[OrderSide.BUY],
                    "sell": votes[OrderSide.SELL]
                },
                "agent_signals": {
                    name: {
                        "action": sig.action.value,
                        "confidence": sig.confidence
                    }
                    for name, sig in signals.items()
                },
                "sub_agent_count": len(signals)
            }
        )

        return signal
