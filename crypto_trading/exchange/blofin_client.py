"""
Blofin exchange client implementation.
Provides connectivity to Blofin exchange for trading operations.
"""

import asyncio
import json
import hmac
import hashlib
import base64
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import aiohttp
import websockets
from loguru import logger

from ..core.interfaces import (
    IExchangeClient, MarketData, Order, Position, OrderType, OrderSide, OrderStatus
)
from ..core.exceptions import ConnectionError, OrderError, ExchangeError
from .type_converter import ITypeConverter, BlofinTypeConverter
from .base_exchange import BaseExchange


class BlofinClient(BaseExchange):
    """Blofin exchange client implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        sandbox: bool = True,
        type_converter: Optional[ITypeConverter] = None
    ):
        super().__init__(api_key, api_secret, sandbox)
        self.passphrase = passphrase
        self.type_converter = type_converter or BlofinTypeConverter()

        self.base_url = "https://openapi.blofin.com" if not sandbox else "https://demo-trading-openapi.blofin.com"
        self.ws_url = "wss://openapi.blofin.com/ws/v1/stream" if not sandbox else "wss://demo-trading-openapi.blofin.com/ws/public"

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connection: Optional[websockets.WebSocketServerProtocol] = None

    # BaseExchange abstract methods implementation

    async def _authenticate(self) -> None:
        """Authenticate with Blofin exchange."""
        # Authentication is done per-request via headers
        pass

    async def _initialize_connection(self) -> None:
        """Initialize connection to Blofin."""
        # Create HTTP session
        self._session = aiohttp.ClientSession()

        # Test connection with account info request
        response = await self._make_request("GET", "/api/v1/account/balance")
        if response.get("code") != "0":
            raise ConnectionError(f"Failed to connect to Blofin: {response}")

        logger.info("Connected to Blofin exchange successfully")

    async def _cleanup_connection(self) -> None:
        """Cleanup Blofin connection resources."""
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Disconnected from Blofin exchange")

    async def _get_market_data_impl(self, symbol: str) -> MarketData:
        """Get current market data for a symbol."""
        # Get ticker data (note: endpoint is /tickers plural)
        ticker_response = await self._make_request("GET", "/api/v1/market/tickers", {"instId": symbol})
        if ticker_response.get("code") != "0":
            raise ExchangeError(f"Failed to get ticker data: {ticker_response}")

        ticker_data = ticker_response["data"][0]

        # Ticker already includes bid/ask prices, no need for separate orderbook request
        return MarketData(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(ticker_data["ts"]) / 1000),
            open=Decimal(ticker_data["open24h"]),
            high=Decimal(ticker_data["high24h"]),
            low=Decimal(ticker_data["low24h"]),
            close=Decimal(ticker_data["last"]),
            volume=Decimal(ticker_data["vol24h"]),
            bid=Decimal(ticker_data["bidPrice"]) if ticker_data.get("bidPrice") else None,
            ask=Decimal(ticker_data["askPrice"]) if ticker_data.get("askPrice") else None
        )

    async def _get_historical_data_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[MarketData]:
        """Get historical market data."""
        # Convert timeframe to Blofin format
        timeframe_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"
        }
        blofin_timeframe = timeframe_map.get(timeframe, "1H")

        params = {
            "instId": symbol,
            "bar": blofin_timeframe,
            "before": str(int(start_date.timestamp() * 1000)),
            "after": str(int(end_date.timestamp() * 1000)),
            "limit": str(min(limit, 100))
        }

        response = await self._make_request("GET", "/api/v1/market/candles", params)
        if response.get("code") != "0":
            raise ExchangeError(f"Failed to get historical data: {response}")

        candles = response["data"]
        market_data = []

        for candle in candles:
            market_data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(candle[0]) / 1000),
                open=Decimal(candle[1]),
                high=Decimal(candle[2]),
                low=Decimal(candle[3]),
                close=Decimal(candle[4]),
                volume=Decimal(candle[5])
            ))

        return sorted(market_data, key=lambda x: x.timestamp)

    async def _place_order_impl(self, order: Order) -> str:
        """Place a trading order and return order ID."""
        order_data = {
            "instId": order.symbol,
            "tdMode": "cash",  # Cash trading mode
            "side": order.side.value,
            "ordType": self.type_converter.convert_order_type_to_exchange(order.type),
            "sz": str(order.amount)
        }

        if order.price is not None:
            order_data["px"] = str(order.price)

        response = await self._make_request("POST", "/api/v1/trade/order", order_data)
        if response.get("code") != "0":
            raise OrderError(f"Failed to place order: {response}")

        order_result = response["data"][0]
        return order_result["ordId"]

    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Cancel an existing order."""
        response = await self._make_request("POST", "/api/v1/trade/cancel-order", {"ordId": order_id})
        return response.get("code") == "0"

    async def _get_order_status_impl(self, order_id: str) -> Order:
        """Get the status of an order."""
        response = await self._make_request("GET", "/api/v1/trade/order", {"ordId": order_id})
        if response.get("code") != "0":
            raise OrderError(f"Failed to get order status: {response}")

        order_data = response["data"][0]
        return Order(
            id=order_data["ordId"],
            symbol=order_data["instId"],
            side=OrderSide(order_data["side"]),
            type=self.type_converter.convert_order_type_from_exchange(order_data["ordType"]),
            amount=Decimal(order_data["sz"]),
            price=Decimal(order_data["px"]) if order_data["px"] else None,
            status=self.type_converter.convert_order_status_from_exchange(order_data["state"]),
            timestamp=datetime.fromtimestamp(int(order_data["cTime"]) / 1000),
            filled_amount=Decimal(order_data["fillSz"]),
            average_price=Decimal(order_data["avgPx"]) if order_data["avgPx"] else None
        )

    async def _get_positions_impl(self) -> List[Position]:
        """Get current positions."""
        response = await self._make_request("GET", "/api/v1/account/positions")
        if response.get("code") != "0":
            raise ExchangeError(f"Failed to get positions: {response}")

        positions = []
        for pos_data in response["data"]:
            position_size = Decimal(pos_data["positions"])
            if position_size != 0:  # Only include non-zero positions
                positions.append(Position(
                    symbol=pos_data["instId"],
                    side=OrderSide.BUY if position_size > 0 else OrderSide.SELL,
                    amount=abs(position_size),
                    entry_price=Decimal(pos_data["averagePrice"]),
                    current_price=Decimal(pos_data["markPrice"]),
                    pnl=Decimal(pos_data["unrealizedPnl"]),
                    timestamp=datetime.fromtimestamp(int(pos_data["updateTime"]) / 1000)
                ))

        return positions

    async def _get_balance_impl(self) -> Dict[str, Decimal]:
        """Get account balance for all assets."""
        response = await self._make_request("GET", "/api/v1/account/balance")
        if response.get("code") != "0":
            raise ExchangeError(f"Failed to get balance: {response}")

        balance = {}
        # Response structure: {"code": "0", "data": {"details": [...]}}
        for balance_data in response["data"]["details"]:
            currency = balance_data["currency"]
            available = Decimal(balance_data["available"])
            balance[currency] = available

        return balance

    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Blofin API."""
        if not self._session:
            raise ConnectionError("Not connected to exchange")

        url = self.base_url + endpoint
        timestamp = str(int(datetime.now().timestamp() * 1000))
        nonce = str(uuid.uuid4())

        # Prepare request data
        if method == "GET":
            query_string = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
            request_path = endpoint + ("?" + query_string if query_string else "")
            body = ""
        else:
            request_path = endpoint
            body = json.dumps(params or {})

        # Create signature - Blofin format: requestPath + method + timestamp + nonce + body
        prehash_string = request_path + method + timestamp + nonce + body

        # Generate HMAC-SHA256 signature
        signature_bytes = hmac.new(
            self.api_secret.encode(),
            prehash_string.encode(),
            hashlib.sha256
        ).digest()

        # Convert to hex then base64 encode
        signature_hex = signature_bytes.hex()
        signature = base64.b64encode(signature_hex.encode()).decode()

        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-NONCE": nonce,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }

        async def make_request():
            if method == "GET":
                async with self._session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            else:
                async with self._session.post(url, data=body, headers=headers) as response:
                    return await response.json()

        try:
            return await asyncio.wait_for(make_request(), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout} seconds")
            raise ConnectionError(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise ConnectionError(f"Request failed: {e}")