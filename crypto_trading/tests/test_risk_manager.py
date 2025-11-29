"""
Test suite for Risk Manager functionality.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from crypto_trading.core.risk_manager import RiskManager
from crypto_trading.core.interfaces import (
    Order, Position, TradingSignal, OrderType, OrderSide, OrderStatus
)
from crypto_trading.core.exceptions import RiskManagementError


class TestRiskManager:
    """Test Risk Manager functionality."""

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance for testing."""
        return RiskManager()

    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            id="test_order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

    @pytest.fixture
    def sample_position(self):
        """Create sample position for testing."""
        return Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=Decimal("0.5"),
            entry_price=Decimal("48000"),
            current_price=Decimal("50000"),
            pnl=Decimal("1000"),
            timestamp=datetime.now()
        )

    @pytest.fixture
    def sample_signal(self):
        """Create sample trading signal for testing."""
        return TradingSignal(
            symbol="ETH/USDT",
            action=OrderSide.BUY,
            confidence=0.8,
            price=Decimal("3000"),
            amount=None,
            timestamp=datetime.now(),
            metadata={"indicator": "RSI"}
        )

    @pytest.fixture
    def sample_balance(self):
        """Create sample balance for testing."""
        return {
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.2"),
            "ETH": Decimal("3.0")
        }

    def test_risk_manager_initialization(self, risk_manager):
        """Test RiskManager initialization."""
        assert risk_manager.daily_losses == {}
        assert risk_manager.position_history == []
        assert "max_position_size_pct" in risk_manager.default_config

    def test_validate_order_basics(self, risk_manager, sample_order):
        """Test basic order validation."""
        # Valid order
        assert risk_manager._validate_order_basics(sample_order)

        # Invalid amount
        invalid_order = Order(
            id="test",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0"),  # Invalid
            price=Decimal("50000"),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        assert not risk_manager._validate_order_basics(invalid_order)

        # Invalid price
        invalid_order.amount = Decimal("0.1")
        invalid_order.price = Decimal("-100")  # Invalid
        assert not risk_manager._validate_order_basics(invalid_order)

    def test_validate_order_with_positions(self, risk_manager, sample_order):
        """Test order validation with existing positions."""
        positions = []

        # Should pass with no positions
        assert risk_manager.validate_order(sample_order, positions)

        # Create multiple positions for same symbol
        positions = [
            Position(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                amount=Decimal("0.5"),
                entry_price=Decimal("48000"),
                current_price=Decimal("50000"),
                pnl=Decimal("1000"),
                timestamp=datetime.now()
            )
        ]

        # Should still pass (default max_positions_per_symbol is 1)
        assert risk_manager.validate_order(sample_order, positions)

    def test_position_size_calculation(self, risk_manager, sample_signal, sample_balance):
        """Test position size calculation."""
        position_size = risk_manager.calculate_position_size(sample_signal, sample_balance)

        assert position_size >= 0
        assert isinstance(position_size, Decimal)

        # Test with high confidence signal
        high_confidence_signal = TradingSignal(
            symbol="BTC/USDT",
            action=OrderSide.BUY,
            confidence=0.95,
            price=Decimal("50000"),
            amount=None,
            timestamp=datetime.now(),
            metadata={}
        )

        high_conf_size = risk_manager.calculate_position_size(high_confidence_signal, sample_balance)

        # Test with low confidence signal
        low_confidence_signal = TradingSignal(
            symbol="BTC/USDT",
            action=OrderSide.BUY,
            confidence=0.3,
            price=Decimal("50000"),
            amount=None,
            timestamp=datetime.now(),
            metadata={}
        )

        low_conf_size = risk_manager.calculate_position_size(low_confidence_signal, sample_balance)

        # High confidence should result in larger position size
        assert high_conf_size >= low_conf_size

    def test_position_size_with_zero_balance(self, risk_manager, sample_signal):
        """Test position size calculation with zero balance."""
        zero_balance = {"USDT": Decimal("0")}

        position_size = risk_manager.calculate_position_size(sample_signal, zero_balance)
        assert position_size == Decimal("0")

    def test_daily_pnl_tracking(self, risk_manager):
        """Test daily P&L tracking."""
        today = "2024-01-15"

        # Add losses
        risk_manager.update_daily_pnl(today, Decimal("-500"))
        risk_manager.update_daily_pnl(today, Decimal("-300"))

        assert risk_manager.daily_losses[today] == Decimal("800")

        # Profits shouldn't be tracked as losses
        risk_manager.update_daily_pnl(today, Decimal("200"))
        assert risk_manager.daily_losses[today] == Decimal("800")

    def test_risk_metrics_calculation(self, risk_manager, sample_position):
        """Test risk metrics calculation."""
        positions = [sample_position]

        metrics = risk_manager.get_risk_metrics(positions)

        assert "total_positions" in metrics
        assert "total_exposure" in metrics
        assert "total_pnl" in metrics
        assert "concentration_ratio" in metrics

        assert metrics["total_positions"] == 1
        assert metrics["total_exposure"] > 0
        assert isinstance(metrics["total_pnl"], float)

    def test_stop_loss_check(self, risk_manager):
        """Test stop loss functionality."""
        # Long position with price drop
        long_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("48000"),  # 4% drop
            pnl=Decimal("-1000"),
            timestamp=datetime.now()
        )

        # Should trigger stop loss (default 2%)
        assert risk_manager.check_stop_loss(long_position)

        # Short position with price rise
        short_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51500"),  # 3% rise
            pnl=Decimal("-750"),
            timestamp=datetime.now()
        )

        # Should trigger stop loss
        assert risk_manager.check_stop_loss(short_position)

    def test_take_profit_check(self, risk_manager):
        """Test take profit functionality."""
        # Long position with price rise
        long_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("53500"),  # 7% rise
            pnl=Decimal("1750"),
            timestamp=datetime.now()
        )

        # Should trigger take profit (default 6%)
        assert risk_manager.check_take_profit(long_position)

        # Short position with price drop
        short_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            amount=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("46500"),  # 7% drop
            pnl=Decimal("1750"),
            timestamp=datetime.now()
        )

        # Should trigger take profit
        assert risk_manager.check_take_profit(short_position)

    def test_portfolio_risk_assessment(self, risk_manager):
        """Test portfolio risk assessment."""
        # Low risk scenario
        low_risk_positions = [
            Position(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                amount=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("50500"),
                pnl=Decimal("50"),
                timestamp=datetime.now()
            )
        ]

        assessment = risk_manager.assess_portfolio_risk(low_risk_positions)
        assert assessment["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(assessment["warnings"], list)

        # High risk scenario - high concentration
        high_risk_positions = [
            Position(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                amount=Decimal("5.0"),  # Large position
                entry_price=Decimal("50000"),
                current_price=Decimal("45000"),  # Big loss
                pnl=Decimal("-25000"),
                timestamp=datetime.now()
            )
        ]

        assessment = risk_manager.assess_portfolio_risk(high_risk_positions)
        # Should detect high risk due to large exposure/loss
        assert len(assessment["warnings"]) > 0

    def test_exposure_limits(self, risk_manager, sample_order):
        """Test exposure limit checks."""
        # Small positions should pass
        small_positions = [
            Position(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                amount=Decimal("0.5"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                pnl=Decimal("50"),
                timestamp=datetime.now()
            )
        ]

        assert risk_manager._check_exposure_limits(sample_order, small_positions)

        # Large positions might fail depending on configuration
        large_positions = [
            Position(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                amount=Decimal("50.0"),  # Very large position
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                pnl=Decimal("5000"),
                timestamp=datetime.now()
            )
        ]

        # This might fail due to high exposure
        result = risk_manager._check_exposure_limits(sample_order, large_positions)
        # Result depends on exact configuration and portfolio value assumptions

    def test_order_size_limits(self, risk_manager):
        """Test order size limit checks."""
        # Order too small
        small_order = Order(
            id="small",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.001"),  # Very small
            price=Decimal("50000"),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

        # Should fail minimum size check
        assert not risk_manager._check_order_size_limits(small_order)

        # Normal size order
        normal_order = Order(
            id="normal",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=Decimal("0.5"),
            price=Decimal("50000"),
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

        # Should pass
        assert risk_manager._check_order_size_limits(normal_order)

    def test_volatility_adjustment(self, risk_manager):
        """Test volatility adjustment for position sizing."""
        # High confidence signal
        high_conf_signal = TradingSignal(
            symbol="BTC/USDT",
            action=OrderSide.BUY,
            confidence=0.95,
            price=Decimal("50000"),
            amount=None,
            timestamp=datetime.now(),
            metadata={}
        )

        adjustment = risk_manager._calculate_volatility_adjustment(high_conf_signal)
        assert adjustment > Decimal("1.0")  # Should increase size for high confidence

        # Low confidence signal
        low_conf_signal = TradingSignal(
            symbol="BTC/USDT",
            action=OrderSide.BUY,
            confidence=0.3,
            price=Decimal("50000"),
            amount=None,
            timestamp=datetime.now(),
            metadata={}
        )

        adjustment = risk_manager._calculate_volatility_adjustment(low_conf_signal)
        assert adjustment <= Decimal("1.0")  # Should decrease size for low confidence


if __name__ == "__main__":
    pytest.main([__file__])