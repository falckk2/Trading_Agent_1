"""
Unit tests for type conversion strategies.
"""

import pytest
from crypto_trading.exchange.type_converter import BlofinTypeConverter, ITypeConverter
from crypto_trading.core.interfaces import OrderType, OrderStatus


class TestBlofinTypeConverter:
    """Tests for BlofinTypeConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = BlofinTypeConverter()

    def test_converter_implements_interface(self):
        """Test that BlofinTypeConverter implements ITypeConverter."""
        assert isinstance(self.converter, ITypeConverter)

    def test_convert_market_order_to_exchange(self):
        """Test converting MARKET order type to exchange format."""
        result = self.converter.convert_order_type_to_exchange(OrderType.MARKET)
        assert result == "market"

    def test_convert_limit_order_to_exchange(self):
        """Test converting LIMIT order type to exchange format."""
        result = self.converter.convert_order_type_to_exchange(OrderType.LIMIT)
        assert result == "limit"

    def test_convert_stop_order_to_exchange(self):
        """Test converting STOP order type to exchange format."""
        result = self.converter.convert_order_type_to_exchange(OrderType.STOP)
        assert result == "market"  # Blofin maps STOP to market

    def test_convert_stop_limit_order_to_exchange(self):
        """Test converting STOP_LIMIT order type to exchange format."""
        result = self.converter.convert_order_type_to_exchange(OrderType.STOP_LIMIT)
        assert result == "limit"

    def test_convert_unknown_order_to_exchange_defaults_to_market(self):
        """Test that unknown order types default to market."""
        # Create a mock OrderType that's not in the mapping
        # This tests the .get() default behavior
        class UnknownType:
            pass

        result = self.converter.convert_order_type_to_exchange(UnknownType())
        assert result == "market"

    def test_convert_market_from_exchange(self):
        """Test converting 'market' from exchange to internal format."""
        result = self.converter.convert_order_type_from_exchange("market")
        assert result == OrderType.MARKET

    def test_convert_limit_from_exchange(self):
        """Test converting 'limit' from exchange to internal format."""
        result = self.converter.convert_order_type_from_exchange("limit")
        assert result == OrderType.LIMIT

    def test_convert_post_only_from_exchange(self):
        """Test converting 'post_only' from exchange to internal format."""
        result = self.converter.convert_order_type_from_exchange("post_only")
        assert result == OrderType.LIMIT

    def test_convert_fok_from_exchange(self):
        """Test converting 'fok' (Fill-or-Kill) from exchange to internal format."""
        result = self.converter.convert_order_type_from_exchange("fok")
        assert result == OrderType.MARKET

    def test_convert_ioc_from_exchange(self):
        """Test converting 'ioc' (Immediate-or-Cancel) from exchange to internal format."""
        result = self.converter.convert_order_type_from_exchange("ioc")
        assert result == OrderType.MARKET

    def test_convert_unknown_order_from_exchange_defaults_to_market(self):
        """Test that unknown exchange order types default to MARKET."""
        result = self.converter.convert_order_type_from_exchange("unknown_type")
        assert result == OrderType.MARKET

    def test_convert_live_status_from_exchange(self):
        """Test converting 'live' status from exchange to internal format."""
        result = self.converter.convert_order_status_from_exchange("live")
        assert result == OrderStatus.OPEN

    def test_convert_partially_filled_status_from_exchange(self):
        """Test converting 'partially_filled' status from exchange to internal format."""
        result = self.converter.convert_order_status_from_exchange("partially_filled")
        assert result == OrderStatus.OPEN

    def test_convert_filled_status_from_exchange(self):
        """Test converting 'filled' status from exchange to internal format."""
        result = self.converter.convert_order_status_from_exchange("filled")
        assert result == OrderStatus.FILLED

    def test_convert_canceled_status_from_exchange(self):
        """Test converting 'canceled' status from exchange to internal format."""
        result = self.converter.convert_order_status_from_exchange("canceled")
        assert result == OrderStatus.CANCELLED

    def test_convert_rejected_status_from_exchange(self):
        """Test converting 'rejected' status from exchange to internal format."""
        result = self.converter.convert_order_status_from_exchange("rejected")
        assert result == OrderStatus.REJECTED

    def test_convert_unknown_status_from_exchange_defaults_to_pending(self):
        """Test that unknown exchange statuses default to PENDING."""
        result = self.converter.convert_order_status_from_exchange("unknown_status")
        assert result == OrderStatus.PENDING

    def test_all_internal_order_types_mapped(self):
        """Test that all internal OrderType enums have mappings."""
        for order_type in OrderType:
            result = self.converter.convert_order_type_to_exchange(order_type)
            assert result is not None
            assert isinstance(result, str)

    def test_converter_is_stateless(self):
        """Test that converter operations don't affect state."""
        # Perform multiple conversions
        self.converter.convert_order_type_to_exchange(OrderType.MARKET)
        self.converter.convert_order_type_from_exchange("limit")
        self.converter.convert_order_status_from_exchange("filled")

        # Verify conversions still work the same
        assert self.converter.convert_order_type_to_exchange(OrderType.MARKET) == "market"
        assert self.converter.convert_order_type_from_exchange("limit") == OrderType.LIMIT
        assert self.converter.convert_order_status_from_exchange("filled") == OrderStatus.FILLED

    def test_round_trip_conversion_market(self):
        """Test round-trip conversion for MARKET orders."""
        # Internal -> Exchange -> Internal
        exchange_format = self.converter.convert_order_type_to_exchange(OrderType.MARKET)
        back_to_internal = self.converter.convert_order_type_from_exchange(exchange_format)
        assert back_to_internal == OrderType.MARKET

    def test_round_trip_conversion_limit(self):
        """Test round-trip conversion for LIMIT orders."""
        # Internal -> Exchange -> Internal
        exchange_format = self.converter.convert_order_type_to_exchange(OrderType.LIMIT)
        back_to_internal = self.converter.convert_order_type_from_exchange(exchange_format)
        assert back_to_internal == OrderType.LIMIT

    def test_multiple_converter_instances_independent(self):
        """Test that multiple converter instances are independent."""
        converter1 = BlofinTypeConverter()
        converter2 = BlofinTypeConverter()

        result1 = converter1.convert_order_type_to_exchange(OrderType.MARKET)
        result2 = converter2.convert_order_type_to_exchange(OrderType.MARKET)

        assert result1 == result2
        assert result1 == "market"
