"""
Constants and configuration values for the trading system.
Centralizes magic numbers and commonly used values.
"""

from decimal import Decimal

# Signal thresholds
SIGNAL_THRESHOLD_POSITIVE = 0.001  # Positive return threshold for BUY signal
SIGNAL_THRESHOLD_NEGATIVE = -0.001  # Negative return threshold for SELL signal
MIN_ACTIONABLE_CONFIDENCE = 0.5  # Minimum confidence for actionable signals

# Default values
DEFAULT_BTC_PRICE = Decimal('50000')  # Default BTC price for calculations
DEFAULT_PORTFOLIO_VALUE = Decimal('100000')  # Default portfolio value
MAX_SNAPSHOT_HISTORY = 10000  # Maximum number of portfolio snapshots to keep

# Risk management defaults
DEFAULT_MAX_POSITION_SIZE_PCT = 0.1  # 10% of balance per position
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05  # 5% daily loss limit
DEFAULT_MAX_TOTAL_EXPOSURE_PCT = 0.5  # 50% total exposure
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MAX_POSITIONS_PER_SYMBOL = 1
DEFAULT_MAX_TOTAL_POSITIONS = 5
DEFAULT_STOP_LOSS_PCT = 0.02  # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.06  # 6% take profit
DEFAULT_MIN_ORDER_SIZE = 10.0  # Minimum order size in base currency
DEFAULT_MIN_ORDER_AMOUNT = 0.01  # Minimum order amount
DEFAULT_MAX_LEVERAGE = 1.0  # No leverage by default
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# ML Strategy defaults
DEFAULT_LOOKBACK_WINDOW = 60
DEFAULT_PREDICTION_HORIZON = 1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_RETRAIN_INTERVAL_HOURS = 24
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_SIGNAL_STRENGTH_MULTIPLIER = 1.0

# Order management
DEFAULT_ORDER_TIMEOUT_SECONDS = 300  # 5 minutes
MAX_ORDER_RETRIES = 3
ORDER_MONITORING_INTERVAL_SECONDS = 1  # Check orders every second

# Position validation
HIGH_CONCENTRATION_THRESHOLD = 0.4  # 40% in single asset is high risk
HIGH_UTILIZATION_THRESHOLD = 0.8  # 80% of capital at risk
HIGH_DAILY_LOSS_THRESHOLD = 0.03  # 3% daily loss
MAX_POSITIONS_WARNING = 8  # Warn if more than 8 positions

# Technical indicator defaults
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_BOLLINGER_STD = 2
DEFAULT_MA_PERIOD = 20

# Data processing
MIN_DATA_POINTS_FOR_ANALYSIS = 1
MIN_DATA_POINTS_FOR_TRAINING = 100
