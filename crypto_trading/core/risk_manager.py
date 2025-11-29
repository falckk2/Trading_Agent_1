"""
Risk management implementation for trading operations.
Provides position sizing, risk validation, and loss limits.
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta
from loguru import logger

from .interfaces import (
    IRiskManager, Order, Position, TradingSignal, OrderSide,
    IConfigManager
)
from .exceptions import RiskManagementError
from .constants import (
    DEFAULT_MAX_POSITION_SIZE_PCT, DEFAULT_MAX_DAILY_LOSS_PCT,
    DEFAULT_MAX_TOTAL_EXPOSURE_PCT, DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_POSITIONS_PER_SYMBOL, DEFAULT_MAX_TOTAL_POSITIONS,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_MIN_ORDER_SIZE, DEFAULT_MIN_ORDER_AMOUNT,
    DEFAULT_MAX_LEVERAGE, DEFAULT_PORTFOLIO_VALUE,
    DEFAULT_RISK_FREE_RATE, DEFAULT_BTC_PRICE
)


class RiskManager(IRiskManager):
    """Risk management implementation with configurable limits."""

    def __init__(self, config_manager: Optional[IConfigManager] = None):
        self.config_manager = config_manager
        self.daily_losses: Dict[str, Decimal] = {}  # Track daily losses by date
        self.position_history: List[Position] = []

        # Default risk parameters from constants
        self.default_config = {
            "max_position_size_pct": DEFAULT_MAX_POSITION_SIZE_PCT,
            "max_daily_loss_pct": DEFAULT_MAX_DAILY_LOSS_PCT,
            "max_total_exposure_pct": DEFAULT_MAX_TOTAL_EXPOSURE_PCT,
            "min_confidence_threshold": DEFAULT_MIN_CONFIDENCE_THRESHOLD,
            "max_positions_per_symbol": DEFAULT_MAX_POSITIONS_PER_SYMBOL,
            "max_total_positions": DEFAULT_MAX_TOTAL_POSITIONS,
            "stop_loss_pct": DEFAULT_STOP_LOSS_PCT,
            "take_profit_pct": DEFAULT_TAKE_PROFIT_PCT,
            "min_order_size": DEFAULT_MIN_ORDER_SIZE,
            "min_order_amount": DEFAULT_MIN_ORDER_AMOUNT,
            "max_leverage": DEFAULT_MAX_LEVERAGE,
            "portfolio_value": float(DEFAULT_PORTFOLIO_VALUE),
            "risk_free_rate": DEFAULT_RISK_FREE_RATE,
            "default_price": float(DEFAULT_BTC_PRICE)
        }

    def _get_config(self, key: str) -> Any:
        """Get configuration value with fallback to default."""
        if self.config_manager:
            return self.config_manager.get(f"risk.{key}", self.default_config.get(key))
        return self.default_config.get(key)

    def validate_order(self, order: Order, positions: List[Position]) -> bool:
        """Validate if an order meets risk criteria."""
        try:
            # Basic validation
            if not self._validate_order_basics(order):
                return False

            # Check daily loss limits
            if not self._check_daily_loss_limit():
                logger.warning("Order rejected: Daily loss limit exceeded")
                return False

            # Check position limits
            if not self._check_position_limits(order, positions):
                logger.warning("Order rejected: Position limits exceeded")
                return False

            # Check exposure limits
            if not self._check_exposure_limits(order, positions):
                logger.warning("Order rejected: Exposure limits exceeded")
                return False

            # Check order size limits
            if not self._check_order_size_limits(order):
                logger.warning("Order rejected: Order size limits exceeded")
                return False

            logger.debug(f"Order validation passed for {order.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

    def _validate_order_basics(self, order: Order) -> bool:
        """Basic order validation."""
        if order.amount <= 0:
            logger.warning("Order rejected: Invalid amount")
            return False

        if order.price is not None and order.price <= 0:
            logger.warning("Order rejected: Invalid price")
            return False

        if not order.symbol:
            logger.warning("Order rejected: Missing symbol")
            return False

        return True

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded."""
        today = datetime.now().date().isoformat()
        daily_loss = self.daily_losses.get(today, Decimal('0'))
        max_daily_loss_pct = Decimal(str(self._get_config("max_daily_loss_pct")))
        portfolio_value = Decimal(str(self._get_config("portfolio_value")))

        max_daily_loss = portfolio_value * max_daily_loss_pct

        return daily_loss < max_daily_loss

    def _check_position_limits(self, order: Order, positions: List[Position]) -> bool:
        """Check position count limits."""
        max_positions_per_symbol = self._get_config("max_positions_per_symbol")
        max_total_positions = self._get_config("max_total_positions")

        # Count existing positions for this symbol with the same side
        symbol_positions = [p for p in positions if p.symbol == order.symbol and p.side == order.side]

        # If we already have a position in the same direction, this order extends it
        # Only reject if we would create a new opposing position and exceed limits
        opposing_positions = [p for p in positions if p.symbol == order.symbol and p.side != order.side]

        # Check if adding this order would exceed symbol limits
        total_symbol_positions = len(symbol_positions) + len(opposing_positions)
        if len(symbol_positions) == 0 and total_symbol_positions >= max_positions_per_symbol:
            return False

        # Check total position count - only count if this creates a new position
        if len(symbol_positions) == 0 and len(positions) >= max_total_positions:
            return False

        return True

    def _check_exposure_limits(self, order: Order, positions: List[Position]) -> bool:
        """Check total exposure limits."""
        max_exposure_pct = Decimal(str(self._get_config("max_total_exposure_pct")))
        portfolio_value = Decimal(str(self._get_config("portfolio_value")))

        # Calculate current exposure
        current_exposure = sum(abs(p.amount * p.current_price) for p in positions)

        # Calculate new exposure - use order price if available, otherwise use a reasonable default
        order_price = order.price if order.price else DEFAULT_BTC_PRICE
        order_value = order.amount * order_price
        new_exposure = current_exposure + order_value

        max_exposure = portfolio_value * max_exposure_pct

        return new_exposure <= max_exposure

    def _check_order_size_limits(self, order: Order) -> bool:
        """Check order size limits."""
        min_order_size = Decimal(str(self._get_config("min_order_size")))
        min_order_amount = Decimal(str(self._get_config("min_order_amount") or DEFAULT_MIN_ORDER_AMOUNT))

        # Calculate order value
        order_price = order.price or DEFAULT_BTC_PRICE
        order_value = order.amount * order_price

        # Check both minimum value and minimum amount
        return order_value >= min_order_size and order.amount >= min_order_amount

    def calculate_position_size(
        self,
        signal: TradingSignal,
        balance: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate appropriate position size based on risk parameters."""
        try:
            # Get available balance
            base_currency = 'USDT'  # Assuming USDT as base
            available_balance = balance.get(base_currency, Decimal('0'))

            if available_balance <= 0:
                logger.warning("No available balance for position sizing")
                return Decimal('0')

            # Calculate base position size
            max_position_pct = Decimal(str(self._get_config("max_position_size_pct")))
            base_position_size = available_balance * max_position_pct

            # Adjust based on signal confidence
            confidence_adjustment = Decimal(str(signal.confidence))
            adjusted_size = base_position_size * confidence_adjustment

            # Apply volatility adjustment
            volatility_adj = self._calculate_volatility_adjustment(signal)
            final_size = adjusted_size * volatility_adj

            # Convert to position amount based on price
            if signal.price:
                position_amount = final_size / signal.price
            else:
                # Use default conversion if no price provided - get from config or use default
                default_price = Decimal(str(self._get_config("default_price") or DEFAULT_BTC_PRICE))
                position_amount = final_size / default_price

            # Apply minimum size check
            min_order_size = Decimal(str(self._get_config("min_order_size")))
            if final_size < min_order_size:
                return Decimal('0')

            logger.debug(f"Calculated position size: {position_amount} for {signal.symbol}")
            return position_amount

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Decimal('0')

    def _calculate_volatility_adjustment(self, signal: TradingSignal) -> Decimal:
        """Calculate position size adjustment based on volatility."""
        # This is a simplified volatility adjustment
        # In production, you'd use actual volatility calculations
        base_adjustment = Decimal('1.0')

        # Reduce size for lower confidence signals
        if signal.confidence < 0.7:
            base_adjustment *= Decimal('0.8')
        elif signal.confidence > 0.9:
            base_adjustment *= Decimal('1.2')

        return min(base_adjustment, Decimal('1.5'))  # Cap at 1.5x

    def update_daily_pnl(self, date: str, pnl: Decimal) -> None:
        """Update daily P&L tracking."""
        if date not in self.daily_losses:
            self.daily_losses[date] = Decimal('0')

        if pnl < 0:  # Only track losses
            self.daily_losses[date] += abs(pnl)

    def get_risk_metrics(self, positions: List[Position]) -> Dict[str, Any]:
        """Get current risk metrics."""
        try:
            total_exposure = sum(abs(p.amount * p.current_price) for p in positions)
            total_pnl = sum(p.pnl for p in positions)

            # Get today's losses
            today = datetime.now().date().isoformat()
            daily_loss = self.daily_losses.get(today, Decimal('0'))

            # Calculate position concentration
            symbol_exposure = {}
            for position in positions:
                exposure = abs(position.amount * position.current_price)
                symbol_exposure[position.symbol] = symbol_exposure.get(position.symbol, Decimal('0')) + exposure

            max_symbol_exposure = max(symbol_exposure.values()) if symbol_exposure else Decimal('0')
            concentration_ratio = max_symbol_exposure / total_exposure if total_exposure > 0 else Decimal('0')

            portfolio_value = Decimal(str(self._get_config("portfolio_value")))
            risk_utilization = total_exposure / portfolio_value if portfolio_value > 0 else Decimal('0')

            return {
                "total_positions": len(positions),
                "total_exposure": float(total_exposure),
                "total_pnl": float(total_pnl),
                "daily_loss": float(daily_loss),
                "concentration_ratio": float(concentration_ratio),
                "risk_utilization": float(risk_utilization),
                "largest_position": float(max_symbol_exposure),
                "symbol_distribution": {k: float(v) for k, v in symbol_exposure.items()}
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def check_stop_loss(self, position: Position) -> bool:
        """Check if position should be stopped out."""
        stop_loss_pct = Decimal(str(self._get_config("stop_loss_pct")))

        if position.side == OrderSide.BUY:
            # Long position: stop if price drops below entry - stop_loss_pct
            stop_price = position.entry_price * (Decimal('1') - stop_loss_pct)
            return position.current_price <= stop_price
        else:
            # Short position: stop if price rises above entry + stop_loss_pct
            stop_price = position.entry_price * (Decimal('1') + stop_loss_pct)
            return position.current_price >= stop_price

    def check_take_profit(self, position: Position) -> bool:
        """Check if position should take profit."""
        take_profit_pct = Decimal(str(self._get_config("take_profit_pct")))

        if position.side == OrderSide.BUY:
            # Long position: take profit if price rises above entry + take_profit_pct
            profit_price = position.entry_price * (Decimal('1') + take_profit_pct)
            return position.current_price >= profit_price
        else:
            # Short position: take profit if price drops below entry - take_profit_pct
            profit_price = position.entry_price * (Decimal('1') - take_profit_pct)
            return position.current_price <= profit_price

    def assess_portfolio_risk(self, positions: List[Position]) -> Dict[str, str]:
        """Assess overall portfolio risk level."""
        metrics = self.get_risk_metrics(positions)

        risk_level = "LOW"
        warnings = []

        # Check concentration risk
        if metrics.get("concentration_ratio", 0) > 0.4:
            risk_level = "HIGH"
            warnings.append("High concentration risk - over 40% in single asset")

        # Check total exposure
        if metrics.get("risk_utilization", 0) > 0.8:
            risk_level = "HIGH"
            warnings.append("High portfolio utilization - over 80% of capital at risk")

        # Check daily losses
        portfolio_value = float(self._get_config("portfolio_value"))
        daily_loss_pct = metrics.get("daily_loss", 0) / portfolio_value if portfolio_value > 0 else 0
        if daily_loss_pct > 0.03:
            risk_level = "HIGH"
            warnings.append("High daily losses - over 3% of portfolio")

        # Check number of positions
        if metrics.get("total_positions", 0) > 8:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            warnings.append("High number of positions - may be difficult to manage")

        return {
            "risk_level": risk_level,
            "warnings": warnings,
            "assessment_time": datetime.now().isoformat()
        }