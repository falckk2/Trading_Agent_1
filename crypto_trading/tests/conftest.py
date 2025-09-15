"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_trading.core.interfaces import MarketData


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    base_time = datetime.now()
    data = []

    for i in range(50):
        # Create realistic price movement
        base_price = 50000 + (i * 10) + (i % 5) * 50
        volatility = 100

        data.append(MarketData(
            symbol="BTC/USDT",
            timestamp=base_time - timedelta(hours=50-i),
            open=Decimal(str(base_price - 25)),
            high=Decimal(str(base_price + volatility)),
            low=Decimal(str(base_price - volatility)),
            close=Decimal(str(base_price)),
            volume=Decimal("100.5"),
            bid=Decimal(str(base_price - 10)),
            ask=Decimal(str(base_price + 10))
        ))

    return data


@pytest.fixture
def sample_eth_data():
    """Create sample ETH market data for testing."""
    base_time = datetime.now()
    data = []

    for i in range(30):
        base_price = 3000 + (i * 5)

        data.append(MarketData(
            symbol="ETH/USDT",
            timestamp=base_time - timedelta(hours=30-i),
            open=Decimal(str(base_price - 10)),
            high=Decimal(str(base_price + 50)),
            low=Decimal(str(base_price - 50)),
            close=Decimal(str(base_price)),
            volume=Decimal("200.0"),
            bid=Decimal(str(base_price - 5)),
            ask=Decimal(str(base_price + 5))
        ))

    return data


@pytest.fixture
def mock_exchange_responses():
    """Mock responses for exchange API calls."""
    return {
        "balance": {
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("3.0")
        },
        "ticker": {
            "symbol": "BTC/USDT",
            "price": "50000.00",
            "volume": "1000.0"
        },
        "order_response": {
            "id": "order_123",
            "status": "filled",
            "filled_amount": "0.1"
        }
    }


@pytest.fixture
def test_config():
    """Standard test configuration."""
    return {
        "trading": {
            "enabled": False,  # Disabled for testing
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "loop_interval": 1  # Fast for testing
        },
        "risk": {
            "max_position_size_pct": 0.1,
            "max_daily_loss_pct": 0.05,
            "max_total_exposure_pct": 0.5
        },
        "agents": {
            "rsi": {
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            },
            "macd": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        }
    }