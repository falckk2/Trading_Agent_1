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
            # Give aiohttp time to close connections properly
            await asyncio.sleep(0.25)
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
        order_type_value = self.type_converter.convert_order_type_to_exchange(order.type)

        # Convert BTC amount to contracts
        # For BTC-USDT: 1 contract ≈ 0.001 BTC (typical)
        # Minimum: 0.1 contracts, Increment: 0.1 contracts
        btc_per_contract = 0.001
        contracts = float(order.amount) / btc_per_contract

        # Round to nearest 0.1 (lot size increment)
        contracts_rounded = round(contracts / 0.1) * 0.1

        # Ensure minimum of 0.1 contracts
        if contracts_rounded < 0.1:
            contracts_rounded = 0.1

        # Format to 1 decimal place
        size_str = f"{contracts_rounded:.1f}"

        # Blofin API required parameters (based on official docs)
        order_data = {
            "instId": order.symbol,
            "marginMode": "isolated",  # "cross" or "isolated" - using isolated for position isolation
            "positionSide": "net",  # "net" for One-way Mode (default), "long"/"short" for Hedge Mode
            "side": order.side.value,
            "orderType": order_type_value,  # Blofin uses "orderType" in requests, "ordType" in responses
            "size": size_str  # Number of contracts (not BTC amount!)
        }

        if order.price is not None:
            # Round price to 2 decimal places for USDT pairs
            rounded_price = round(float(order.price), 2)
            order_data["price"] = str(rounded_price)  # Blofin uses "price" in requests, "px" in responses

        # Add reduceOnly parameter if set (for closing positions safely)
        if order.reduce_only:
            order_data["reduceOnly"] = "true"  # Blofin expects string "true", not boolean

        logger.debug(f"Placing order with data: {order_data}")
        response = await self._make_request("POST", "/api/v1/trade/order", order_data)

        # Log the full response for debugging
        logger.debug(f"Order placement response: {response}")

        if response.get("code") != "0":
            logger.error(f"Order placement failed. Request: {order_data}, Response: {response}")
            error_msg = response.get("msg", "Unknown error")
            raise OrderError(f"Failed to place order: {error_msg} (code: {response.get('code')})")

        # Validate response structure
        if "data" not in response:
            logger.error(f"Invalid response structure - missing 'data': {response}")
            raise OrderError(f"Invalid API response structure: {response}")

        if not response["data"] or len(response["data"]) == 0:
            logger.error(f"Empty data in response: {response}")
            raise OrderError(f"No order data returned: {response}")

        order_result = response["data"][0]

        # Blofin API returns 'orderId' (lowercase 'd') in responses, not 'ordId'
        if "orderId" not in order_result:
            logger.error(f"Missing 'orderId' in order result: {order_result}")
            raise OrderError(f"Order ID not found in response: {order_result}")

        order_id = order_result["orderId"]
        logger.info(f"Order placed successfully with ID: {order_id}")
        return order_id

    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Cancel an existing order."""
        response = await self._make_request("POST", "/api/v1/trade/cancel-order", {"orderId": order_id})
        return response.get("code") == "0"

    async def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all pending orders from the exchange.

        Args:
            symbol: Optional symbol to filter orders. If None, returns all pending orders.

        Returns:
            List of pending Order objects
        """
        params = {}
        if symbol:
            params["instId"] = symbol

        response = await self._make_request("GET", "/api/v1/trade/orders-pending", params if params else None)
        if response.get("code") != "0":
            raise ExchangeError(f"Failed to get pending orders: {response}")

        orders = []
        for order_data in response.get("data", []):
            try:
                orders.append(Order(
                    id=order_data["orderId"],
                    symbol=order_data["instId"],
                    side=OrderSide(order_data["side"]),
                    type=self.type_converter.convert_order_type_from_exchange(order_data["orderType"]),  # Fixed: orderType not ordType
                    amount=Decimal(order_data["size"]),  # Fixed: size not sz
                    price=Decimal(order_data["price"]) if order_data.get("price") else None,  # Fixed: price not px
                    status=self.type_converter.convert_order_status_from_exchange(order_data["state"]),
                    timestamp=datetime.fromtimestamp(int(order_data["createTime"]) / 1000),  # Fixed: createTime not cTime
                    filled_amount=Decimal(order_data.get("filledSize", "0")),  # Fixed: filledSize not fillSz
                    average_price=Decimal(order_data["averagePrice"]) if order_data.get("averagePrice") else None  # Fixed: averagePrice not avgPx
                ))
            except Exception as e:
                logger.error(f"Error parsing pending order {order_data.get('orderId', 'unknown')}: {e}")
                logger.debug(f"Failed order data: {order_data}")

        return orders

    async def cancel_batch_orders(self, order_ids: List[str]) -> Dict[str, bool]:
        """
        Cancel multiple orders in a single batch request.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            Dictionary mapping order_id to success status (True if cancelled, False otherwise)
        """
        if not order_ids:
            return {}

        # Blofin batch cancel expects array of objects with orderId field
        order_data_list = [{"orderId": order_id} for order_id in order_ids]

        response = await self._make_request("POST", "/api/v1/trade/cancel-batch-orders", order_data_list)

        # Parse results
        results = {}
        if response.get("code") == "0":
            # Batch request succeeded, check individual results
            for result in response.get("data", []):
                order_id = result.get("orderId")
                # sCode "0" means success for individual order
                success = result.get("sCode") == "0"
                if order_id:
                    results[order_id] = success
                    if not success:
                        logger.warning(f"Failed to cancel order {order_id}: {result.get('sMsg', 'unknown error')}")
        else:
            # Entire batch request failed
            logger.error(f"Batch cancel request failed: {response}")
            # Mark all as failed
            for order_id in order_ids:
                results[order_id] = False

        return results

    async def _get_order_status_impl(self, order_id: str) -> Order:
        """Get the status of an order."""
        response = await self._make_request("GET", "/api/v1/trade/order-detail", {"orderId": order_id})

        logger.debug(f"Order status response for {order_id}: {response}")

        if response.get("code") != "0":
            raise OrderError(f"Failed to get order status: {response}")

        # Validate response structure
        if "data" not in response:
            raise OrderError(f"Missing 'data' in order status response: {response}")

        if not response["data"] or len(response["data"]) == 0:
            raise OrderError(f"Empty data in order status response: {response}")

        order_data = response["data"][0]
        return Order(
            id=order_data["orderId"],
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
            # Position size from API is in contracts
            position_size_contracts = Decimal(pos_data["positions"])

            if position_size_contracts != 0:  # Only include non-zero positions
                # Convert contracts to BTC (1 contract = 0.001 BTC for BTC-USDT)
                btc_per_contract = Decimal("0.001")
                position_size_btc = abs(position_size_contracts) * btc_per_contract

                positions.append(Position(
                    symbol=pos_data["instId"],
                    side=OrderSide.BUY if position_size_contracts > 0 else OrderSide.SELL,
                    amount=position_size_btc,  # Store in BTC, not contracts
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

    async def get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get instrument specifications including precision requirements.

        Returns:
            Dict with keys: tick_size (price precision), lot_size (size precision),
            min_size (minimum order size), contract_val (contract value in base currency)
        """
        response = await self._make_request("GET", "/api/v1/market/instruments", {"instId": symbol})
        if response.get("code") != "0":
            raise ExchangeError(f"Failed to get instrument info for {symbol}: {response}")

        if not response.get("data") or len(response["data"]) == 0:
            raise ExchangeError(f"No instrument data found for {symbol}")

        inst_data = response["data"][0]

        return {
            "tick_size": Decimal(inst_data["tickSize"]),  # Price precision (e.g., 0.5 for BTC-USDT)
            "lot_size": Decimal(inst_data["lotSize"]),    # Size precision (e.g., 0.1 for BTC-USDT)
            "min_size": Decimal(inst_data["minSize"]),    # Minimum order size
            "contract_val": Decimal(inst_data["contractValue"]),  # Contract value (0.001 for BTC-USDT)
        }

    def round_price(self, price: Decimal, tick_size: Decimal) -> Decimal:
        """Round price to the nearest tick size."""
        return (price / tick_size).quantize(Decimal('1')) * tick_size

    def round_size(self, size: Decimal, lot_size: Decimal) -> Decimal:
        """Round size down to the nearest lot size (always round down to avoid exceeding position)."""
        return (size / lot_size).quantize(Decimal('1'), rounding='ROUND_DOWN') * lot_size

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
            logger.debug(f"POST {endpoint} - Body: {body}")

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

        try:
            # Use aiohttp's built-in timeout within the async context
            if method == "GET":
                async with self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"GET {endpoint} raw response (status {response.status}): {response_text}")
                    try:
                        result = json.loads(response_text)
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response from GET {endpoint}: {response_text}")
                        raise ConnectionError(f"Invalid JSON response: {response_text}")
            else:
                async with self._session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"POST {endpoint} raw response (status {response.status}): {response_text}")
                    try:
                        result = json.loads(response_text)
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {response_text}")
                        raise ConnectionError(f"Invalid JSON response: {response_text}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout} seconds")
            raise ConnectionError(f"Request timed out after {self.timeout} seconds")
        except aiohttp.ClientError as e:
            logger.error(f"Request failed (client error): {e}")
            raise ConnectionError(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise ConnectionError(f"Request failed: {e}")