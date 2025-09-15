"""
Blofin exchange client implementation.
Provides connectivity to Blofin exchange for trading operations.
"""

import asyncio
import json
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import aiohttp
import websockets
from loguru import logger

from ..core.interfaces import (
    IExchangeClient, MarketData, Order, Position, OrderType, OrderSide, OrderStatus
)
from ..utils.exceptions import ConnectionError, OrderError, ExchangeError


class BlofinClient(IExchangeClient):
    """Blofin exchange client implementation."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox

        self.base_url = "https://openapi.blofin.com" if not sandbox else "https://sandbox-openapi.blofin.com"
        self.ws_url = "wss://openapi.blofin.com/ws/v1/stream" if not sandbox else "wss://sandbox-openapi.blofin.com/ws/v1/stream"

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self._is_connected = False

    async def connect(self) -> bool:
        """Establish connection to Blofin exchange."""
        try:
            # Create HTTP session
            self._session = aiohttp.ClientSession()

            # Test connection with account info request
            response = await self._make_request("GET", "/api/v1/account/balance")
            if response.get("code") == "0":
                self._is_connected = True
                logger.info("Connected to Blofin exchange successfully")
                return True
            else:
                logger.error(f"Failed to connect to Blofin: {response}")
                return False

        except Exception as e:
            logger.error(f"Error connecting to Blofin: {e}")
            return False

    async def disconnect(self) -> None:
        """Close connection to Blofin exchange."""
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None

        if self._session:
            await self._session.close()
            self._session = None

        self._is_connected = False
        logger.info("Disconnected from Blofin exchange")

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol."""
        try:
            # Get ticker data
            ticker_response = await self._make_request("GET", f"/api/v1/market/ticker", {"instId": symbol})
            if ticker_response.get("code") != "0":
                raise ExchangeError(f"Failed to get ticker data: {ticker_response}")

            ticker_data = ticker_response["data"][0]

            # Get order book for bid/ask
            orderbook_response = await self._make_request("GET", f"/api/v1/market/books", {"instId": symbol})
            orderbook_data = orderbook_response["data"][0] if orderbook_response.get("code") == "0" else {}

            return MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(ticker_data["ts"]) / 1000),
                open=Decimal(ticker_data["open24h"]),
                high=Decimal(ticker_data["high24h"]),
                low=Decimal(ticker_data["low24h"]),
                close=Decimal(ticker_data["last"]),
                volume=Decimal(ticker_data["vol24h"]),
                bid=Decimal(orderbook_data["bids"][0][0]) if orderbook_data.get("bids") else None,
                ask=Decimal(orderbook_data["asks"][0][0]) if orderbook_data.get("asks") else None
            )

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise ExchangeError(f"Failed to get market data: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[MarketData]:
        """Get historical market data."""
        try:
            # Convert timeframe to Blofin format
            timeframe_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"
            }
            blofin_timeframe = timeframe_map.get(timeframe, "1H")

            params = {
                "instId": symbol,
                "bar": blofin_timeframe,
                "before": str(int(start.timestamp() * 1000)),
                "after": str(int(end.timestamp() * 1000)),
                "limit": "100"
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

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise ExchangeError(f"Failed to get historical data: {e}")

    async def place_order(self, order: Order) -> Order:
        """Place a trading order."""
        try:
            order_data = {
                "instId": order.symbol,
                "tdMode": "cash",  # Cash trading mode
                "side": order.side.value,
                "ordType": self._convert_order_type(order.type),
                "sz": str(order.amount)
            }

            if order.price is not None:
                order_data["px"] = str(order.price)

            response = await self._make_request("POST", "/api/v1/trade/order", order_data)
            if response.get("code") != "0":
                raise OrderError(f"Failed to place order: {response}")

            order_result = response["data"][0]
            order.id = order_result["ordId"]
            order.status = self._convert_order_status(order_result["sCode"])

            logger.info(f"Order placed successfully: {order.id}")
            return order

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise OrderError(f"Failed to place order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            response = await self._make_request("POST", "/api/v1/trade/cancel-order", {"ordId": order_id})
            if response.get("code") == "0":
                logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {response}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Order:
        """Get the status of an order."""
        try:
            response = await self._make_request("GET", "/api/v1/trade/order", {"ordId": order_id})
            if response.get("code") != "0":
                raise OrderError(f"Failed to get order status: {response}")

            order_data = response["data"][0]
            return Order(
                id=order_data["ordId"],
                symbol=order_data["instId"],
                side=OrderSide(order_data["side"]),
                type=self._convert_order_type_from_blofin(order_data["ordType"]),
                amount=Decimal(order_data["sz"]),
                price=Decimal(order_data["px"]) if order_data["px"] else None,
                status=self._convert_order_status(order_data["state"]),
                timestamp=datetime.fromtimestamp(int(order_data["cTime"]) / 1000),
                filled_amount=Decimal(order_data["fillSz"]),
                average_price=Decimal(order_data["avgPx"]) if order_data["avgPx"] else None
            )

        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            raise OrderError(f"Failed to get order status: {e}")

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        try:
            response = await self._make_request("GET", "/api/v1/account/positions")
            if response.get("code") != "0":
                raise ExchangeError(f"Failed to get positions: {response}")

            positions = []
            for pos_data in response["data"]:
                if Decimal(pos_data["pos"]) != 0:  # Only include non-zero positions
                    positions.append(Position(
                        symbol=pos_data["instId"],
                        side=OrderSide.BUY if Decimal(pos_data["pos"]) > 0 else OrderSide.SELL,
                        amount=abs(Decimal(pos_data["pos"])),
                        entry_price=Decimal(pos_data["avgPx"]),
                        current_price=Decimal(pos_data["last"]),
                        pnl=Decimal(pos_data["upl"]),
                        timestamp=datetime.fromtimestamp(int(pos_data["uTime"]) / 1000)
                    ))

            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise ExchangeError(f"Failed to get positions: {e}")

    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        try:
            response = await self._make_request("GET", "/api/v1/account/balance")
            if response.get("code") != "0":
                raise ExchangeError(f"Failed to get balance: {response}")

            balance = {}
            for balance_data in response["data"][0]["details"]:
                currency = balance_data["ccy"]
                available = Decimal(balance_data["availBal"])
                balance[currency] = available

            return balance

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            raise ExchangeError(f"Failed to get balance: {e}")

    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Blofin API."""
        if not self._session:
            raise ConnectionError("Not connected to exchange")

        url = self.base_url + endpoint
        timestamp = str(int(datetime.now().timestamp() * 1000))

        # Prepare request data
        if method == "GET":
            query_string = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
            request_path = endpoint + ("?" + query_string if query_string else "")
            data = ""
        else:
            request_path = endpoint
            data = json.dumps(params or {})

        # Create signature
        message = timestamp + method + request_path + data
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode()

        headers = {
            "BF-ACCESS-KEY": self.api_key,
            "BF-ACCESS-SIGN": signature,
            "BF-ACCESS-TIMESTAMP": timestamp,
            "BF-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }

        try:
            if method == "GET":
                async with self._session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            else:
                async with self._session.post(url, data=data, headers=headers) as response:
                    return await response.json()

        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise ConnectionError(f"Request failed: {e}")

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Blofin format."""
        type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "market",  # Blofin doesn't have separate stop orders
            OrderType.STOP_LIMIT: "limit"
        }
        return type_map[order_type]

    def _convert_order_type_from_blofin(self, blofin_type: str) -> OrderType:
        """Convert Blofin order type to internal format."""
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "post_only": OrderType.LIMIT,
            "fok": OrderType.MARKET,
            "ioc": OrderType.MARKET
        }
        return type_map.get(blofin_type, OrderType.MARKET)

    def _convert_order_status(self, blofin_status: str) -> OrderStatus:
        """Convert Blofin order status to internal format."""
        status_map = {
            "live": OrderStatus.OPEN,
            "partially_filled": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED
        }
        return status_map.get(blofin_status, OrderStatus.PENDING)