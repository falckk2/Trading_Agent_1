"""
Real-time data feed implementation using WebSocket connections.
Provides live market data with automatic reconnection and error handling.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import websockets
import logging

from ...core.models import MarketData
from ...core.exceptions import DataException, DataProviderError


class RealtimeFeed:
    """
    Real-time data feed using WebSocket connections.
    Supports multiple exchanges and automatic reconnection.
    """

    def __init__(self, exchange_config: Dict[str, Any] = None):
        self.exchange_config = exchange_config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # WebSocket connections
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[Callable]] = {}

        # Connection management
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        self.ping_interval = 30
        self.connection_timeout = 10

        # Data management
        self._latest_data: Dict[str, MarketData] = {}
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

        # Status tracking
        self.is_running = False
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize the real-time feed."""
        self.is_running = True
        self.logger.info("Real-time feed initialized")

    async def shutdown(self) -> None:
        """Shutdown all connections and tasks."""
        self.is_running = False

        # Cancel all reconnect tasks
        for task in self._reconnect_tasks.values():
            if not task.done():
                task.cancel()

        # Cancel heartbeat tasks
        for task in self._heartbeat_tasks.values():
            if not task.done():
                task.cancel()

        # Close all connections
        for exchange, connection in self.connections.items():
            try:
                await connection.close()
                self.logger.info(f"Closed connection to {exchange}")
            except Exception as e:
                self.logger.warning(f"Error closing connection to {exchange}: {e}")

        self.connections.clear()
        self.subscriptions.clear()
        self.logger.info("Real-time feed shutdown completed")

    async def subscribe(self, symbol: str, callback: Callable) -> None:
        """Subscribe to real-time data for a symbol."""
        try:
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = []

            self.subscriptions[symbol].append(callback)

            # Connect to appropriate exchange if not already connected
            exchange = self._get_exchange_for_symbol(symbol)
            if exchange not in self.connections:
                await self._connect_to_exchange(exchange)

            # Send subscription message
            await self._send_subscription(exchange, symbol)

            self.logger.info(f"Subscribed to {symbol} on {exchange}")

        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            raise DataException(f"Subscription failed: {e}")

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from real-time data for a symbol."""
        try:
            if symbol in self.subscriptions:
                del self.subscriptions[symbol]

            # Send unsubscription message
            exchange = self._get_exchange_for_symbol(symbol)
            if exchange in self.connections:
                await self._send_unsubscription(exchange, symbol)

            self.logger.info(f"Unsubscribed from {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {symbol}: {e}")

    async def get_current_data(self, symbol: str) -> MarketData:
        """Get the latest data for a symbol."""
        if symbol not in self._latest_data:
            raise DataProviderError(f"No data available for {symbol}")

        return self._latest_data[symbol]

    async def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all exchanges."""
        status = {}
        for exchange, connection in self.connections.items():
            status[exchange] = connection.open if connection else False
        return status

    # Private methods

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default exchange configuration."""
        return {
            "binance": {
                "ws_url": "wss://stream.binance.com:9443/ws",
                "symbols_endpoint": "/stream?streams=",
                "message_format": "binance"
            },
            "coinbase": {
                "ws_url": "wss://ws-feed.pro.coinbase.com",
                "message_format": "coinbase"
            },
            "kraken": {
                "ws_url": "wss://ws.kraken.com",
                "message_format": "kraken"
            }
        }

    def _get_exchange_for_symbol(self, symbol: str) -> str:
        """Determine which exchange to use for a symbol."""
        # Simple logic - can be enhanced with symbol mapping
        return "binance"  # Default to Binance for now

    async def _connect_to_exchange(self, exchange: str) -> None:
        """Connect to a specific exchange."""
        if exchange not in self.exchange_config:
            raise DataProviderError(f"Unknown exchange: {exchange}")

        config = self.exchange_config[exchange]
        ws_url = config["ws_url"]

        try:
            self.logger.info(f"Connecting to {exchange}...")

            connection = await websockets.connect(
                ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=self.connection_timeout,
                close_timeout=self.connection_timeout
            )

            self.connections[exchange] = connection

            # Start message handler
            asyncio.create_task(self._handle_messages(exchange, connection))

            # Start heartbeat task
            self._heartbeat_tasks[exchange] = asyncio.create_task(
                self._heartbeat(exchange)
            )

            self.logger.info(f"Connected to {exchange}")

        except Exception as e:
            self.logger.error(f"Failed to connect to {exchange}: {e}")
            raise DataException(f"Connection to {exchange} failed: {e}")

    async def _handle_messages(self, exchange: str, connection) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in connection:
                try:
                    await self._process_message(exchange, message)
                except Exception as e:
                    self.logger.error(f"Error processing message from {exchange}: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"Connection to {exchange} closed")
        except Exception as e:
            self.logger.error(f"Error in message handler for {exchange}: {e}")
        finally:
            # Schedule reconnection
            if self.is_running:
                self._reconnect_tasks[exchange] = asyncio.create_task(
                    self._reconnect(exchange)
                )

    async def _process_message(self, exchange: str, message: str) -> None:
        """Process incoming message and extract market data."""
        try:
            data = json.loads(message)
            config = self.exchange_config[exchange]
            message_format = config["message_format"]

            market_data = None

            if message_format == "binance":
                market_data = self._parse_binance_message(data)
            elif message_format == "coinbase":
                market_data = self._parse_coinbase_message(data)
            elif message_format == "kraken":
                market_data = self._parse_kraken_message(data)

            if market_data:
                symbol = market_data.symbol
                self._latest_data[symbol] = market_data

                # Notify subscribers
                if symbol in self.subscriptions:
                    for callback in self.subscriptions[symbol]:
                        try:
                            await callback(symbol, market_data)
                        except Exception as e:
                            self.logger.error(f"Error in callback for {symbol}: {e}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON from {exchange}: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Error processing message from {exchange}: {e}")

    def _parse_binance_message(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Binance WebSocket message format."""
        try:
            # Handle ticker stream
            if "stream" in data and "data" in data:
                ticker_data = data["data"]
                if "s" in ticker_data and "c" in ticker_data:
                    return MarketData(
                        symbol=ticker_data["s"],
                        timestamp=datetime.fromtimestamp(ticker_data["E"] / 1000),
                        open=float(ticker_data["o"]),
                        high=float(ticker_data["h"]),
                        low=float(ticker_data["l"]),
                        close=float(ticker_data["c"]),
                        volume=float(ticker_data["v"]),
                        bid=float(ticker_data.get("b", 0)),
                        ask=float(ticker_data.get("a", 0))
                    )

            # Handle individual ticker
            elif "s" in data and "c" in data:
                return MarketData(
                    symbol=data["s"],
                    timestamp=datetime.fromtimestamp(data["E"] / 1000),
                    open=float(data["o"]),
                    high=float(data["h"]),
                    low=float(data["l"]),
                    close=float(data["c"]),
                    volume=float(data["v"]),
                    bid=float(data.get("b", 0)),
                    ask=float(data.get("a", 0))
                )

        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"Failed to parse Binance message: {e}")

        return None

    def _parse_coinbase_message(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Coinbase WebSocket message format."""
        try:
            if data.get("type") == "ticker":
                return MarketData(
                    symbol=data["product_id"],
                    timestamp=datetime.fromisoformat(data["time"].replace("Z", "+00:00")),
                    open=float(data["open_24h"]),
                    high=float(data["high_24h"]),
                    low=float(data["low_24h"]),
                    close=float(data["price"]),
                    volume=float(data["volume_24h"]),
                    bid=float(data.get("best_bid", 0)),
                    ask=float(data.get("best_ask", 0))
                )

        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"Failed to parse Coinbase message: {e}")

        return None

    def _parse_kraken_message(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Kraken WebSocket message format."""
        try:
            # Kraken has a more complex message format
            if isinstance(data, list) and len(data) >= 2:
                channel_data = data[1]
                if isinstance(channel_data, dict) and "c" in channel_data:
                    # Ticker data
                    ticker = channel_data
                    return MarketData(
                        symbol=data[-1],  # Symbol is usually the last element
                        timestamp=datetime.now(),  # Kraken doesn't always provide timestamp
                        open=float(ticker.get("o", [0, 0])[0]),
                        high=float(ticker.get("h", [0, 0])[0]),
                        low=float(ticker.get("l", [0, 0])[0]),
                        close=float(ticker.get("c", [0, 0])[0]),
                        volume=float(ticker.get("v", [0, 0])[0]),
                        bid=float(ticker.get("b", [0, 0])[0]),
                        ask=float(ticker.get("a", [0, 0])[0])
                    )

        except (KeyError, ValueError, TypeError, IndexError) as e:
            self.logger.debug(f"Failed to parse Kraken message: {e}")

        return None

    async def _send_subscription(self, exchange: str, symbol: str) -> None:
        """Send subscription message to exchange."""
        if exchange not in self.connections:
            return

        connection = self.connections[exchange]
        config = self.exchange_config[exchange]

        try:
            if exchange == "binance":
                # Subscribe to ticker stream
                sub_message = {
                    "method": "SUBSCRIBE",
                    "params": [f"{symbol.lower()}@ticker"],
                    "id": int(time.time())
                }
            elif exchange == "coinbase":
                sub_message = {
                    "type": "subscribe",
                    "product_ids": [symbol],
                    "channels": ["ticker"]
                }
            elif exchange == "kraken":
                sub_message = {
                    "event": "subscribe",
                    "pair": [symbol],
                    "subscription": {"name": "ticker"}
                }
            else:
                return

            await connection.send(json.dumps(sub_message))
            self.logger.debug(f"Sent subscription for {symbol} to {exchange}")

        except Exception as e:
            self.logger.error(f"Failed to send subscription to {exchange}: {e}")

    async def _send_unsubscription(self, exchange: str, symbol: str) -> None:
        """Send unsubscription message to exchange."""
        if exchange not in self.connections:
            return

        connection = self.connections[exchange]

        try:
            if exchange == "binance":
                unsub_message = {
                    "method": "UNSUBSCRIBE",
                    "params": [f"{symbol.lower()}@ticker"],
                    "id": int(time.time())
                }
            elif exchange == "coinbase":
                unsub_message = {
                    "type": "unsubscribe",
                    "product_ids": [symbol],
                    "channels": ["ticker"]
                }
            elif exchange == "kraken":
                unsub_message = {
                    "event": "unsubscribe",
                    "pair": [symbol],
                    "subscription": {"name": "ticker"}
                }
            else:
                return

            await connection.send(json.dumps(unsub_message))
            self.logger.debug(f"Sent unsubscription for {symbol} to {exchange}")

        except Exception as e:
            self.logger.error(f"Failed to send unsubscription to {exchange}: {e}")

    async def _heartbeat(self, exchange: str) -> None:
        """Send periodic heartbeat to maintain connection."""
        while self.is_running and exchange in self.connections:
            try:
                connection = self.connections[exchange]
                if connection.open:
                    await connection.ping()
                    await asyncio.sleep(self.ping_interval)
                else:
                    break
            except Exception as e:
                self.logger.warning(f"Heartbeat failed for {exchange}: {e}")
                break

    async def _reconnect(self, exchange: str) -> None:
        """Reconnect to exchange with exponential backoff."""
        attempt = 0
        while self.is_running and attempt < self.max_reconnect_attempts:
            try:
                await asyncio.sleep(self.reconnect_delay * (2 ** attempt))

                self.logger.info(f"Reconnecting to {exchange} (attempt {attempt + 1})")

                # Clean up old connection
                if exchange in self.connections:
                    del self.connections[exchange]

                # Clean up heartbeat task
                if exchange in self._heartbeat_tasks:
                    task = self._heartbeat_tasks[exchange]
                    if not task.done():
                        task.cancel()
                    del self._heartbeat_tasks[exchange]

                # Reconnect
                await self._connect_to_exchange(exchange)

                # Resubscribe to all symbols for this exchange
                for symbol in self.subscriptions.keys():
                    if self._get_exchange_for_symbol(symbol) == exchange:
                        await self._send_subscription(exchange, symbol)

                self.logger.info(f"Successfully reconnected to {exchange}")
                return

            except Exception as e:
                attempt += 1
                self.logger.error(
                    f"Reconnection attempt {attempt} failed for {exchange}: {e}"
                )

        self.logger.error(f"Failed to reconnect to {exchange} after {attempt} attempts")