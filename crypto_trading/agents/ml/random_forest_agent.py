"""
Random Forest-based trading agent.
Uses Random Forest classifier for price direction prediction.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from ...core.interfaces import MarketData, TradingSignal
from ..base_agent import BaseAgent
from .random_forest_strategy import RandomForestStrategy


class RandomForestAgent(BaseAgent):
    """Trading agent using Random Forest for price direction prediction."""

    def __init__(self, strategy_parameters: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="Random Forest Agent",
            description="Uses Random Forest classifier to predict price direction and generate trading signals"
        )

        # Initialize strategy with parameters
        self.strategy = RandomForestStrategy(strategy_parameters)

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "strategy_parameters": self.strategy.get_parameters()
        }

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Random Forest agent with configuration."""
        super().initialize(config)

        # Update strategy parameters if provided
        if "strategy_parameters" in config:
            self.strategy.set_parameters(config["strategy_parameters"])

    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data using Random Forest strategy."""
        self._ensure_initialized()
        self._validate_market_data(market_data)

        try:
            # Delegate to strategy
            signal = self.strategy.analyze(market_data)

            # Add agent-specific metadata
            signal.metadata.update({
                "agent_name": self.name,
                "agent_type": "RandomForestAgent"
            })

            logger.debug(f"RandomForest Agent ({self.name}): Generated signal {signal.signal_type} "
                        f"with confidence {signal.confidence:.3f}")

            return signal

        except Exception as e:
            logger.error(f"Error in RandomForest analysis: {e}")
            return self._create_neutral_signal(market_data)

    async def train_model(self, training_data: List[MarketData], retrain: bool = False) -> Dict[str, Any]:
        """Train the Random Forest model using the strategy."""
        try:
            if len(training_data) < self._get_minimum_data_points():
                raise ValueError("Insufficient data for training")

            # Delegate training to strategy
            training_metrics = await self.strategy.train_model(training_data, retrain)

            logger.info(f"RandomForest Agent training completed: {training_metrics}")
            return training_metrics

        except Exception as e:
            logger.error(f"Error training RandomForest Agent: {e}")
            raise

    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self.strategy.is_trained

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        self.strategy.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.strategy.load_model(filepath)

    def needs_retraining(self) -> bool:
        """Check if model needs retraining."""
        return self.strategy.needs_retraining()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained Random Forest model."""
        return self.strategy.get_feature_importance()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        model_info = self.strategy.get_model_info()
        model_info.update({
            "agent_name": self.name,
            "agent_type": "RandomForestAgent"
        })
        return model_info

    def _get_minimum_data_points(self) -> int:
        """Get minimum data points required for training."""
        return max(self.strategy.lookback_window + 50, 100)