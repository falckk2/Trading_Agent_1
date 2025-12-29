"""
Validator Agent - Validates signals from a primary agent using validator agents.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ...core.interfaces import MarketData, TradingSignal, OrderSide
from ..base_agent import BaseAgent


class ValidatorAgent(BaseAgent):
    """
    Meta-agent that validates signals from a primary agent using validator agents.

    The validator agent:
    1. Gets signal from primary agent
    2. Checks if validator agents agree
    3. Only executes if validators confirm the signal
    4. Can require unanimous or majority validation
    """

    def __init__(
        self,
        primary_agent: Optional[BaseAgent] = None,
        validator_agents: Optional[List[BaseAgent]] = None
    ):
        """
        Initialize the Validator Agent.

        Args:
            primary_agent: The main agent that generates signals
            validator_agents: List of agents that validate the primary signal
        """
        super().__init__(
            name="Validator Agent",
            description="Validates primary agent signals using validator agents"
        )
        self.primary_agent: Optional[BaseAgent] = primary_agent
        self.validator_agents: List[BaseAgent] = validator_agents or []

    def set_primary_agent(self, agent: BaseAgent) -> None:
        """Set the primary signal-generating agent."""
        self.primary_agent = agent
        logger.info(f"Validator Agent primary set to: {agent.get_name()}")

    def set_validator_agents(self, agents: List[BaseAgent]) -> None:
        """Set the validator agents."""
        self.validator_agents = agents
        logger.info(f"Validator Agent configured with {len(agents)} validators: {[a.get_name() for a in agents]}")

    def get_required_parameters(self) -> List[str]:
        return []  # No required parameters

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "validation_mode": "majority",  # "majority" or "unanimous"
            "min_validator_confidence": 0.3,  # Minimum confidence required from validators (lowered for testing)
            "require_same_action": True,  # Validators must suggest same action as primary
            "confidence_boost": 0.1,  # Confidence boost if all validators agree
        }

    def _get_minimum_data_points(self) -> int:
        # Use the maximum requirement from primary and validator agents
        agents = [self.primary_agent] + self.validator_agents if self.primary_agent else self.validator_agents
        if not agents:
            return 20
        return max(agent._get_minimum_data_points() for agent in agents if agent)

    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """
        Analyze market data using primary agent with validation.

        Args:
            market_data: Market data to analyze

        Returns:
            Trading signal (validated or rejected)
        """
        self._ensure_initialized()

        if not self.primary_agent:
            logger.warning("Validator Agent has no primary agent configured")
            return self._create_neutral_signal(market_data)

        if not self.validator_agents:
            logger.warning("Validator Agent has no validators configured, using primary agent only")
            return await self.primary_agent.analyze(market_data)

        self._validate_market_data(market_data)

        try:
            # Store current symbol
            self._current_symbol = self._get_symbol_from_data(market_data)

            # Get signal from primary agent
            primary_signal = await self.primary_agent.analyze(market_data)

            logger.debug(
                f"Primary signal from {self.primary_agent.get_name()}: "
                f"{primary_signal.action.value.upper()} (confidence: {primary_signal.confidence * 100:.1f}%)"
            )

            # If primary signal is HOLD, no need to validate
            if primary_signal.action not in [OrderSide.BUY, OrderSide.SELL]:
                return primary_signal

            # Get validator signals
            validator_signals: Dict[str, TradingSignal] = {}
            for agent in self.validator_agents:
                try:
                    signal = await agent.analyze(market_data)
                    validator_signals[agent.get_name()] = signal
                except Exception as e:
                    logger.error(f"Error getting validation from {agent.get_name()}: {e}")

            if not validator_signals:
                logger.warning("No validator signals received, rejecting primary signal")
                return self._create_neutral_signal(market_data)

            # Validate the primary signal
            validated_signal = self._validate_signal(
                primary_signal,
                validator_signals,
                market_data
            )

            return validated_signal

        except Exception as e:
            logger.error(f"Error in Validator Agent analysis: {e}")
            return self._create_neutral_signal(market_data)

    def _validate_signal(
        self,
        primary_signal: TradingSignal,
        validator_signals: Dict[str, TradingSignal],
        market_data: List[MarketData]
    ) -> TradingSignal:
        """
        Validate primary signal using validator signals.

        Args:
            primary_signal: Signal from primary agent
            validator_signals: Signals from validator agents
            market_data: Market data

        Returns:
            Validated signal or HOLD if validation fails
        """
        # Get configuration
        validation_mode = self._get_config_value("validation_mode", "majority")
        min_validator_confidence = self._get_config_value("min_validator_confidence", 0.6)
        require_same_action = self._get_config_value("require_same_action", True)
        confidence_boost = self._get_config_value("confidence_boost", 0.1)

        # Count validators that agree
        validators_agree = 0
        validators_disagree = 0
        validator_details = []

        for agent_name, signal in validator_signals.items():
            # Check confidence threshold
            if signal.confidence < min_validator_confidence:
                logger.debug(f"Validator {agent_name} confidence too low: {signal.confidence:.2f}")
                continue

            # Check if validator agrees with primary action
            if require_same_action:
                if signal.action == primary_signal.action:
                    validators_agree += 1
                    validator_details.append({
                        "agent": agent_name,
                        "agreed": True,
                        "action": signal.action.value,
                        "confidence": signal.confidence
                    })
                else:
                    validators_disagree += 1
                    validator_details.append({
                        "agent": agent_name,
                        "agreed": False,
                        "action": signal.action.value,
                        "confidence": signal.confidence
                    })
            else:
                # Just check that validator is not suggesting opposite action
                opposite_action = OrderSide.SELL if primary_signal.action == OrderSide.BUY else OrderSide.BUY
                if signal.action != opposite_action:
                    validators_agree += 1
                    validator_details.append({
                        "agent": agent_name,
                        "agreed": True,
                        "action": signal.action.value,
                        "confidence": signal.confidence
                    })
                else:
                    validators_disagree += 1
                    validator_details.append({
                        "agent": agent_name,
                        "agreed": False,
                        "action": signal.action.value,
                        "confidence": signal.confidence
                    })

        total_validators = validators_agree + validators_disagree

        # Determine if validation passed
        validation_passed = False

        if validation_mode == "unanimous":
            validation_passed = (validators_agree == len(validator_signals) and validators_disagree == 0)
        elif validation_mode == "majority":
            validation_passed = (validators_agree > validators_disagree)

        # Log validation result
        logger.info(
            f"Validation result: {validators_agree}/{total_validators} validators agree "
            f"(mode: {validation_mode}, passed: {validation_passed})"
        )

        # If validation failed, return HOLD
        if not validation_passed:
            logger.info("Signal validation FAILED - returning HOLD")
            return self._create_neutral_signal(market_data)

        # Validation passed - adjust confidence
        final_confidence = primary_signal.confidence

        # Boost confidence if all validators agree
        if validators_agree == len(validator_signals):
            final_confidence = min(1.0, final_confidence + confidence_boost)
            logger.debug(f"All validators agree - confidence boosted to {final_confidence:.2f}")

        # Create validated signal
        validated_signal = TradingSignal(
            symbol=primary_signal.symbol,
            action=primary_signal.action,
            confidence=final_confidence,
            price=primary_signal.price,
            metadata={
                "validation_type": "validator_agent",
                "primary_agent": self.primary_agent.get_name(),
                "primary_confidence": primary_signal.confidence,
                "validation_mode": validation_mode,
                "validators_agree": validators_agree,
                "validators_disagree": validators_disagree,
                "validator_details": validator_details,
                "validation_passed": True
            }
        )

        logger.info(
            f"🟠 Validator Agent → {validated_signal.action.value.upper()} {validated_signal.symbol} "
            f"(confidence: {final_confidence * 100:.1f}%, {validators_agree}/{len(validator_signals)} validators agree)"
        )

        return validated_signal
