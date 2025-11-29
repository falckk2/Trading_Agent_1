"""
Data transformation utilities for reducing code duplication.
Provides common data conversion and transformation functions.
"""

from typing import List, Dict, Any
from decimal import Decimal
import pandas as pd
from loguru import logger

from ..core.interfaces import MarketData


class DataTransformUtils:
    """Utility class for common data transformations."""

    @staticmethod
    def market_data_to_dataframe(data: List[MarketData]) -> pd.DataFrame:
        """
        Convert list of MarketData to pandas DataFrame.

        Args:
            data: List of MarketData objects

        Returns:
            DataFrame with market data

        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Cannot convert empty market data to DataFrame")

        df_data = []
        for md in data:
            df_data.append({
                'timestamp': md.timestamp,
                'open': float(md.open),
                'high': float(md.high),
                'low': float(md.low),
                'close': float(md.close),
                'volume': float(md.volume),
                'symbol': md.symbol
            })

        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df

    @staticmethod
    def dataframe_to_market_data(df: pd.DataFrame, symbol: str) -> List[MarketData]:
        """
        Convert pandas DataFrame to list of MarketData.

        Args:
            df: DataFrame with OHLCV columns
            symbol: Trading symbol

        Returns:
            List of MarketData objects
        """
        market_data = []

        for timestamp, row in df.iterrows():
            market_data.append(MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=Decimal(str(row['open'])),
                high=Decimal(str(row['high'])),
                low=Decimal(str(row['low'])),
                close=Decimal(str(row['close'])),
                volume=Decimal(str(row['volume']))
            ))

        return market_data

    @staticmethod
    def extract_close_prices(data: List[MarketData]) -> List[float]:
        """
        Extract close prices from market data.

        Args:
            data: List of MarketData objects

        Returns:
            List of close prices as floats
        """
        return [float(md.close) for md in data]

    @staticmethod
    def extract_volumes(data: List[MarketData]) -> List[float]:
        """
        Extract volumes from market data.

        Args:
            data: List of MarketData objects

        Returns:
            List of volumes as floats
        """
        return [float(md.volume) for md in data]

    @staticmethod
    def calculate_price_changes(data: List[MarketData]) -> List[float]:
        """
        Calculate price changes (close - open) for each data point.

        Args:
            data: List of MarketData objects

        Returns:
            List of price changes
        """
        return [float(md.close - md.open) for md in data]

    @staticmethod
    def calculate_returns(data: List[MarketData]) -> List[float]:
        """
        Calculate percentage returns between consecutive data points.

        Args:
            data: List of MarketData objects

        Returns:
            List of percentage returns
        """
        if len(data) < 2:
            return []

        returns = []
        for i in range(1, len(data)):
            prev_close = float(data[i - 1].close)
            curr_close = float(data[i].close)

            if prev_close != 0:
                ret = ((curr_close - prev_close) / prev_close) * 100
                returns.append(ret)
            else:
                returns.append(0.0)

        return returns


class NumericConverter:
    """Utility class for numeric type conversions."""

    @staticmethod
    def to_decimal(value: Any) -> Decimal:
        """
        Convert any numeric value to Decimal safely.

        Args:
            value: Numeric value (int, float, str, Decimal)

        Returns:
            Decimal representation
        """
        if isinstance(value, Decimal):
            return value
        if value is None:
            return Decimal('0')
        return Decimal(str(value))

    @staticmethod
    def to_float(value: Any) -> float:
        """
        Convert any numeric value to float safely.

        Args:
            value: Numeric value

        Returns:
            Float representation
        """
        if isinstance(value, float):
            return value
        if isinstance(value, Decimal):
            return float(value)
        if value is None:
            return 0.0
        return float(value)

    @staticmethod
    def decimal_dict_to_float(data: Dict[str, Decimal]) -> Dict[str, float]:
        """
        Convert dictionary values from Decimal to float.

        Args:
            data: Dictionary with Decimal values

        Returns:
            Dictionary with float values
        """
        return {k: float(v) for k, v in data.items()}

    @staticmethod
    def float_dict_to_decimal(data: Dict[str, float]) -> Dict[str, Decimal]:
        """
        Convert dictionary values from float to Decimal.

        Args:
            data: Dictionary with float values

        Returns:
            Dictionary with Decimal values
        """
        return {k: Decimal(str(v)) for k, v in data.items()}

    @staticmethod
    def safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = Decimal('0')) -> Decimal:
        """
        Safely divide two Decimal values.

        Args:
            numerator: Numerator value
            denominator: Denominator value
            default: Default value if division by zero

        Returns:
            Result of division or default
        """
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def format_currency(value: Decimal, precision: int = 2) -> str:
        """
        Format Decimal value as currency string.

        Args:
            value: Decimal value
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        from decimal import ROUND_HALF_UP
        quantize_str = '0.' + '0' * precision
        formatted = value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        return f"{formatted:,.{precision}f}"


class MarketDataValidator:
    """Validator for market data."""

    @staticmethod
    def validate_sufficient_data(data: List[MarketData], min_points: int, operation: str = "analysis") -> None:
        """
        Validate that market data has sufficient points.

        Args:
            data: List of MarketData objects
            min_points: Minimum required data points
            operation: Description of operation (for error message)

        Raises:
            ValueError: If data is insufficient
        """
        if not data:
            raise ValueError(f"No market data provided for {operation}")

        if len(data) < min_points:
            raise ValueError(
                f"Insufficient data for {operation}. "
                f"Need at least {min_points} points, got {len(data)}"
            )

    @staticmethod
    def validate_data_quality(data: List[MarketData]) -> List[str]:
        """
        Validate data quality and return list of issues.

        Args:
            data: List of MarketData objects

        Returns:
            List of validation warnings/errors
        """
        issues = []

        for i, md in enumerate(data):
            # Check for zero/negative prices
            if md.close <= 0:
                issues.append(f"Invalid close price at index {i}: {md.close}")

            if md.high < md.low:
                issues.append(f"High < Low at index {i}")

            if md.close > md.high or md.close < md.low:
                issues.append(f"Close outside high/low range at index {i}")

            if md.open > md.high or md.open < md.low:
                issues.append(f"Open outside high/low range at index {i}")

            # Check for zero volume (might be valid for some instruments)
            if md.volume == 0:
                issues.append(f"Zero volume at index {i}")

        return issues

    @staticmethod
    def get_symbol_from_data(data: List[MarketData]) -> str:
        """
        Extract symbol from market data.

        Args:
            data: List of MarketData objects

        Returns:
            Trading symbol

        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Cannot extract symbol from empty market data")
        return data[0].symbol
