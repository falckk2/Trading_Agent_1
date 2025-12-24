"""
Moving Average calculation strategies.
Follows Open/Closed Principle - new MA types can be added without modifying existing code.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Type


class MACalculationStrategy(ABC):
    """Abstract base class for moving average calculation strategies."""

    @abstractmethod
    def calculate(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate moving average.

        Args:
            data: Price series to calculate MA on
            period: Period for the moving average

        Returns:
            Series with moving average values
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the MA type."""
        pass


class SMACalculator(MACalculationStrategy):
    """Simple Moving Average calculator."""

    def calculate(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period, min_periods=period).mean()

    def get_name(self) -> str:
        """Get the name."""
        return "sma"


class EMACalculator(MACalculationStrategy):
    """Exponential Moving Average calculator."""

    def calculate(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False, min_periods=period).mean()

    def get_name(self) -> str:
        """Get the name."""
        return "ema"


class WMACalculator(MACalculationStrategy):
    """Weighted Moving Average calculator."""

    def calculate(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Weighted Moving Average.
        More recent prices have higher weight.
        """
        weights = pd.Series(range(1, period + 1))

        def wma(x):
            if len(x) < period:
                return pd.NA
            return (x * weights).sum() / weights.sum()

        return data.rolling(window=period, min_periods=period).apply(wma, raw=False)

    def get_name(self) -> str:
        """Get the name."""
        return "wma"


class HMACalculator(MACalculationStrategy):
    """Hull Moving Average calculator - smoother and more responsive."""

    def calculate(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Hull Moving Average.
        Formula: HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
        """
        half_period = period // 2
        sqrt_period = int(period ** 0.5)

        wma_calculator = WMACalculator()

        # Calculate WMA for half period
        wma_half = wma_calculator.calculate(data, half_period)

        # Calculate WMA for full period
        wma_full = wma_calculator.calculate(data, period)

        # Calculate the difference and then WMA of the result
        raw_hma = 2 * wma_half - wma_full
        hma = wma_calculator.calculate(raw_hma, sqrt_period)

        return hma

    def get_name(self) -> str:
        """Get the name."""
        return "hma"


class MACalculatorFactory:
    """
    Factory for creating MA calculator instances.
    Follows Open/Closed Principle - register new types without modifying existing code.
    """

    _calculators: Dict[str, Type[MACalculationStrategy]] = {
        'sma': SMACalculator,
        'ema': EMACalculator,
        'wma': WMACalculator,
        'hma': HMACalculator,
    }

    @classmethod
    def create(cls, ma_type: str) -> MACalculationStrategy:
        """
        Create a MA calculator instance.

        Args:
            ma_type: Type of MA (sma, ema, wma, hma)

        Returns:
            MACalculationStrategy instance

        Raises:
            ValueError: If ma_type is not supported
        """
        ma_type = ma_type.lower()

        if ma_type not in cls._calculators:
            raise ValueError(
                f"Unsupported MA type: {ma_type}. "
                f"Supported types: {', '.join(cls._calculators.keys())}"
            )

        return cls._calculators[ma_type]()

    @classmethod
    def register(cls, ma_type: str, calculator_class: Type[MACalculationStrategy]) -> None:
        """
        Register a new MA calculator type.
        Allows extending the factory without modifying its code.

        Args:
            ma_type: Name/key for the MA type
            calculator_class: Calculator class to register

        Example:
            class MyCustomMA(MACalculationStrategy):
                ...

            MACalculatorFactory.register('custom', MyCustomMA)
        """
        if not issubclass(calculator_class, MACalculationStrategy):
            raise TypeError("calculator_class must be a subclass of MACalculationStrategy")

        cls._calculators[ma_type.lower()] = calculator_class

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported MA types."""
        return list(cls._calculators.keys())
