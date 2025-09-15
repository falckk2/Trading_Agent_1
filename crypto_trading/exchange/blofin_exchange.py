"""
Blofin exchange implementation.
Provides concrete implementation of Blofin exchange API integration.
"""

import json
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
import asyncio

from .base_exchange import BaseExchange
from ..core.models import Order, MarketData, OrderType, OrderSide, OrderStatus
from ..core.exceptions import ExchangeAPIError, ExchangeConnectionError


class BlofinExchange(BaseExchange):
    """
    Blofin exchange implementation.
    Provides specific integration with Blofin cryptocurrency exchange.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        sandbox: bool = False
    ):
        super().__init__(api_key, api_secret, sandbox)
        self.passphrase = passphrase
        self.base_url = self._get_base_url()
        self.session: Optional[aiohttp.ClientSession] = None

    def _get_base_url(self) -> str:
        """Get the appropriate base URL based on environment."""
        if self.sandbox:
            return "https://openapi.blofin.com"  # Sandbox URL
        else:
            return "https://openapi.blofin.com"  # Production URL

    async def _authenticate(self) -> None:
        """Authenticate with Blofin exchange."""
        # Test authentication by making a simple API call
        try:
            await self._make_authenticated_request("GET", "/api/v1/account/balance")
            self.logger.info("Authentication successful")
        except Exception as e:
            raise ExchangeConnectionError(f"Authentication failed: {e}")

    async def _initialize_connection(self) -> None:
        """Initialize HTTP session and connection."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'Content-Type': 'application/json'}
        )

    async def _cleanup_connection(self) -> None:
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _get_balance_impl(self, asset: str) -> float:
        """Get balance for specific asset from Blofin."""
        response = await self._make_authenticated_request(
            "GET", "/api/v1/account/balance"
        )

        balances = response.get("data", [])
        for balance in balances:
            if balance.get("currency", "").upper() == asset.upper():
                return float(balance.get("available", 0))

        return 0.0

    async def _place_order_impl(self, order: Order) -> str:
        """Place order on Blofin exchange."""
        order_data = {
            "instId": order.symbol,
            "tdMode": "cash",  # Cash trading mode
            "side": order.side.value,
            "ordType": self._convert_order_type(order.order_type),
            "sz": str(order.quantity)
        }

        if order.price:
            order_data["px"] = str(order.price)

        if order.stop_price:
            order_data["slTriggerPx"] = str(order.stop_price)

        response = await self._make_authenticated_request(
            "POST", "/api/v1/trade/order", data=order_data
        )

        if response.get("code") != "0":
            raise ExchangeAPIError(f"Order placement failed: {response.get('msg')}")

        order_info = response.get("data", [{}])[0]
        return order_info.get("ordId", "")

    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Cancel order on Blofin exchange."""
        cancel_data = {
            "ordId": order_id
        }

        response = await self._make_authenticated_request(
            "POST", "/api/v1/trade/cancel-order", data=cancel_data
        )

        return response.get("code") == "0"

    async def _get_order_status_impl(self, order_id: str) -> Dict[str, Any]:
        """Get order status from Blofin exchange."""
        params = {"ordId": order_id}

        response = await self._make_authenticated_request(
            "GET", "/api/v1/trade/order", params=params
        )

        if response.get("code") != "0":
            raise ExchangeAPIError(f"Failed to get order status: {response.get('msg')}")

        order_data = response.get("data", [{}])[0]
        return self._parse_order_status(order_data)

    async def _get_market_data_impl(self, symbol: str) -> MarketData:
        """Get current market data from Blofin."""
        # Get ticker data
        ticker_response = await self._make_public_request(
            "GET", "/api/v1/market/ticker", params={"instId": symbol}
        )

        if ticker_response.get("code") != "0":
            raise ExchangeAPIError(f"Failed to get ticker: {ticker_response.get('msg')}")

        ticker_data = ticker_response.get("data", [{}])[0]

        # Get order book for bid/ask
        book_response = await self._make_public_request(
            "GET", "/api/v1/market/books", params={"instId": symbol, "sz": "1"}
        )

        book_data = book_response.get("data", [{}])[0] if book_response.get("code") == "0" else {}

        bid = None
        ask = None
        if book_data:
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])
            if bids:
                bid = float(bids[0][0])
            if asks:
                ask = float(asks[0][0])

        return MarketData(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(ticker_data.get("ts", 0)) / 1000),
            open=float(ticker_data.get("open24h", 0)),
            high=float(ticker_data.get("high24h", 0)),
            low=float(ticker_data.get("low24h", 0)),
            close=float(ticker_data.get("last", 0)),
            volume=float(ticker_data.get("volCcy24h", 0)),
            bid=bid,
            ask=ask
        )

    async def _get_historical_data_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: int
    ) -> List[MarketData]:
        """Get historical candle data from Blofin."""
        params = {
            "instId": symbol,
            "bar": self._convert_timeframe(timeframe),
            "limit": str(min(limit, 100))  # Blofin max limit
        }

        # Add time parameters if provided
        if start_date:
            params["after"] = str(int(start_date.timestamp() * 1000))
        if end_date:
            params["before"] = str(int(end_date.timestamp() * 1000))

        response = await self._make_public_request(
            "GET", "/api/v1/market/candles", params=params
        )

        if response.get("code") != "0":
            raise ExchangeAPIError(f"Failed to get historical data: {response.get('msg')}")

        candles = response.get("data", [])
        market_data_list = []

        for candle in candles:
            if len(candle) >= 6:
                market_data_list.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(candle[0]) / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                ))

        return market_data_list

    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Blofin API."""
        if not self.session:
            raise ExchangeConnectionError("No active session")

        timestamp = str(int(datetime.now().timestamp()))

        # Prepare request body
        body = ""
        if data:
            body = json.dumps(data, separators=(',', ':'))

        # Create signature
        message = timestamp + method.upper() + endpoint + body
        signature = self._create_signature(message)

        headers = {
            "BF-ACCESS-KEY": self.api_key,
            "BF-ACCESS-SIGN": signature,
            "BF-ACCESS-TIMESTAMP": timestamp,
            "BF-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }

        url = self.base_url + endpoint

        try:
            if method.upper() == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            else:
                async with self.session.request(
                    method.upper(), url, json=data, headers=headers
                ) as response:
                    return await response.json()
        except Exception as e:
            raise ExchangeAPIError(f"API request failed: {e}")

    async def _make_public_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make public request to Blofin API."""
        if not self.session:
            raise ExchangeConnectionError("No active session")

        url = self.base_url + endpoint

        try:
            async with self.session.get(url, params=params) as response:
                return await response.json()
        except Exception as e:
            raise ExchangeAPIError(f"Public API request failed: {e}")

    def _create_signature(self, message: str) -> str:
        """Create HMAC signature for Blofin API."""
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Blofin format."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "market",
            OrderType.STOP_LIMIT: "limit"
        }
        return mapping.get(order_type, "limit")

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Blofin format."""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W"
        }
        return mapping.get(timeframe, "1H")

    def _parse_order_status(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Blofin order data to standard format."""
        state = order_data.get("state", "")

        status_mapping = {
            "live": OrderStatus.OPEN,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED
        }

        return {
            "order_id": order_data.get("ordId"),
            "status": status_mapping.get(state, OrderStatus.OPEN),
            "symbol": order_data.get("instId"),
            "side": OrderSide.BUY if order_data.get("side") == "buy" else OrderSide.SELL,
            "quantity": float(order_data.get("sz", 0)),
            "filled_quantity": float(order_data.get("fillSz", 0)),
            "price": float(order_data.get("px", 0)) if order_data.get("px") else None,
            "average_price": float(order_data.get("avgPx", 0)),
            "timestamp": datetime.fromtimestamp(int(order_data.get("cTime", 0)) / 1000),
            "fees": float(order_data.get("fee", 0))
        }