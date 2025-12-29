"""
Mock exchange client for testing without real exchange connection.
Generates realistic market data and simulates order execution.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from random import uniform, random
from loguru import logger

from ..core.interfaces import IExchangeClient, MarketData, Order, Position, OrderStatus, OrderSide, OrderType


class MockExchangeClient(IExchangeClient):
    """Mock exchange client for testing purposes."""

    def __init__(self):
        self._is_connected = False
        self.balance = {
            "USDT": Decimal("50000.0"),
            "BTC": Decimal("0.0"),
            "ETH": Decimal("0.0")
        }
        self.positions: List[Position] = []
        self.active_orders: List[Order] = []
        self.order_counter = 1

        # Simulated prices (starting values)
        self.current_prices = {
            "BTC/USDT": Decimal("43000.0"),
            "ETH/USDT": Decimal("2300.0")
        }

        # Price movement simulation
        self.price_direction = {
            "BTC/USDT": 1,  # 1 = up, -1 = down
            "ETH/USDT": 1
        }

    def is_connected(self) -> bool:
        """Check if connected to the exchange."""
        return self._is_connected

    async def connect(self) -> bool:
        """Simulate connection to exchange."""
        logger.info("🎭 Connecting to MOCK exchange (paper trading mode)...")
        await asyncio.sleep(0.5)  # Simulate network delay
        self._is_connected = True
        logger.info("✅ Connected to MOCK exchange - No real money at risk!")
        return True

    async def disconnect(self) -> None:
        """Disconnect from mock exchange."""
        self._is_connected = False
        logger.info("Disconnected from MOCK exchange")

    def _simulate_price_movement(self, symbol: str) -> Decimal:
        """Simulate realistic price movement."""
        current_price = self.current_prices[symbol]

        # Random walk with trend
        volatility = Decimal(str(uniform(0.001, 0.003)))  # 0.1% to 0.3% movement
        direction = self.price_direction[symbol]

        # Occasionally change direction (10% chance)
        if random() < 0.1:
            self.price_direction[symbol] = -direction
            direction = self.price_direction[symbol]

        # Add some noise
        noise = Decimal(str(uniform(-0.001, 0.001)))
        movement = (volatility * Decimal(str(direction))) + noise

        new_price = current_price * (Decimal("1.0") + movement)
        self.current_prices[symbol] = new_price

        return new_price

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""
        if not self._is_connected:
            raise ConnectionError("Not connected to exchange")

        # Simulate network delay
        await asyncio.sleep(0.1)

        # Generate realistic OHLC data
        close = self._simulate_price_movement(symbol)
        open_price = close * Decimal(str(uniform(0.998, 1.002)))
        high = max(open_price, close) * Decimal(str(uniform(1.0, 1.005)))
        low = min(open_price, close) * Decimal(str(uniform(0.995, 1.0)))
        volume = Decimal(str(uniform(100, 1000)))

        bid = close * Decimal("0.9995")
        ask = close * Decimal("1.0005")

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            bid=bid,
            ask=ask
        )

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        if not self._is_connected:
            raise ConnectionError("Not connected to exchange")

        # Simulate delay
        await asyncio.sleep(0.2)

        # Generate historical data
        data = []
        current_time = start_time
        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_minutes.get(timeframe, 60)

        base_price = self.current_prices[symbol]

        while current_time <= end_time:
            # Generate OHLC
            close = base_price * Decimal(str(uniform(0.98, 1.02)))
            open_price = close * Decimal(str(uniform(0.998, 1.002)))
            high = max(open_price, close) * Decimal(str(uniform(1.0, 1.005)))
            low = min(open_price, close) * Decimal(str(uniform(0.995, 1.0)))
            volume = Decimal(str(uniform(50, 500)))

            data.append(MarketData(
                symbol=symbol,
                timestamp=current_time,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            ))

            current_time += timedelta(minutes=interval)
            base_price = close  # Next candle starts from previous close

        logger.info(f"🎭 Mock Exchange: Generated {len(data)} historical candles for {symbol}")
        return data

    async def place_order(self, order: Order) -> Order:
        """Place a simulated order."""
        if not self._is_connected:
            raise ConnectionError("Not connected to exchange")

        # Simulate network delay
        await asyncio.sleep(0.3)

        # Extract order details
        symbol = order.symbol
        side = order.side
        order_type = order.type
        amount = order.amount
        price = order.price

        # Get current market price
        market_data = await self.get_market_data(symbol)
        execution_price = price if price else market_data.close

        # Create filled order with exchange ID
        filled_order = Order(
            id=f"MOCK-{self.order_counter}",
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=execution_price,
            status=OrderStatus.FILLED,  # Instantly fill for simulation
            timestamp=datetime.now(),
            filled_amount=amount,
            average_price=execution_price,
            metadata=order.metadata  # Preserve metadata (includes agent_name)
        )

        self.order_counter += 1

        # Update balance
        base_currency = symbol.split('/')[0]
        quote_currency = symbol.split('/')[1]

        if side == OrderSide.BUY:
            cost = amount * execution_price
            self.balance[quote_currency] -= cost
            self.balance[base_currency] = self.balance.get(base_currency, Decimal("0")) + amount
            logger.info(f"🎭 MOCK ORDER: Bought {amount} {base_currency} @ ${execution_price} = ${cost}")
        else:
            proceeds = amount * execution_price
            self.balance[base_currency] -= amount
            self.balance[quote_currency] += proceeds
            logger.info(f"🎭 MOCK ORDER: Sold {amount} {base_currency} @ ${execution_price} = ${proceeds}")

        return filled_order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order."""
        logger.info(f"🎭 MOCK: Cancelled order {order_id}")
        return True

    async def get_order_status(self, order_id: str) -> Order:
        """Get order status (always filled in mock)."""
        return Order(
            id=order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=Decimal("0.001"),
            price=Decimal("43000"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            filled_amount=Decimal("0.001"),
            average_price=Decimal("43000")
        )

    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Decimal]:
        """Get account balance."""
        if currency:
            return {currency: self.balance.get(currency, Decimal("0"))}
        return self.balance.copy()

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get current positions."""
        # Mock doesn't maintain positions separately
        return []

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders (always empty in instant-fill mode)."""
        return []
