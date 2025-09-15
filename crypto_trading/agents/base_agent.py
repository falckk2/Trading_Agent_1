"""
Base trading agent implementation.
Provides common functionality for all trading agents.
"""

from abc import ABC
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from ..core.interfaces import ITradingAgent, MarketData, TradingSignal, OrderSide
from ..utils.exceptions import AgentInitializationError


class BaseAgent(ITradingAgent, ABC):
    """Base class for all trading agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.config: Dict[str, Any] = {}
        self.is_initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the trading agent with configuration."""
        try:
            self.config = config.copy()
            self._validate_config()
            self.is_initialized = True
            logger.info(f"Agent '{self.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent '{self.name}': {e}")
            raise AgentInitializationError(f"Agent initialization failed: {e}")

    def _validate_config(self) -> None:
        """Validate agent configuration. Override in subclasses."""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in config")

    def get_name(self) -> str:
        """Get the name of the trading agent."""
        return self.name

    def get_description(self) -> str:
        """Get a description of the trading agent."""
        return self.description

    def get_parameters(self) -> Dict[str, Any]:
        """Get configurable parameters for the agent."""
        return self.get_default_parameters()

    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters. Override in subclasses."""
        return []

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters. Override in subclasses."""
        return {}

    def _create_signal(
        self,
        symbol: str,
        action: OrderSide,
        confidence: float,
        price: Optional[float] = None,
        amount: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """Helper method to create trading signals."""
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=max(0.0, min(1.0, confidence)),  # Clamp between 0 and 1
            price=price,
            amount=amount,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config.get(key, default)

    def _ensure_initialized(self) -> None:
        """Ensure agent is initialized before use."""
        if not self.is_initialized:
            raise AgentInitializationError(f"Agent '{self.name}' is not initialized")

    def _validate_market_data(self, market_data: List[MarketData]) -> None:
        """Validate market data before analysis."""
        if not market_data:
            raise ValueError("Market data is empty")

        if len(market_data) < self._get_minimum_data_points():
            raise ValueError(f"Insufficient data points. Need at least {self._get_minimum_data_points()}")

    def _get_minimum_data_points(self) -> int:
        """Get minimum required data points. Override in subclasses."""
        return 1

    def _calculate_confidence(self, signal_strength: float, volatility: float = 0.0) -> float:
        """Calculate confidence based on signal strength and market conditions."""
        # Base confidence from signal strength
        confidence = abs(signal_strength)

        # Adjust for volatility (higher volatility = lower confidence)
        if volatility > 0:
            volatility_adjustment = 1 / (1 + volatility)
            confidence *= volatility_adjustment

        # Ensure confidence is within valid range
        return max(0.0, min(1.0, confidence))

    def _get_symbol_from_data(self, market_data: List[MarketData]) -> str:
        """Extract symbol from market data."""
        if not market_data:
            raise ValueError("No market data provided")
        return market_data[0].symbol