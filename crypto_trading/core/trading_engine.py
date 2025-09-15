"""
Main trading engine that coordinates all components.
Implements the facade pattern for simplified interaction.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from decimal import Decimal

from .interfaces import (
    IExchangeClient, ITradingAgent, IRiskManager, IEventBus, IConfigManager,
    MarketData, Order, Position, TradingSignal, OrderStatus, OrderSide, OrderType,
    Event, EventType
)
from .agent_manager import AgentManager
from ..utils.exceptions import TradingSystemError, RiskManagementError


class TradingEngine:
    """Main trading engine coordinating all system components."""

    def __init__(
        self,
        exchange_client: IExchangeClient,
        risk_manager: IRiskManager,
        event_bus: IEventBus,
        config_manager: IConfigManager
    ):
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.agent_manager = AgentManager()

        self._is_running = False
        self._trading_enabled = False
        self._positions: List[Position] = []
        self._active_orders: Dict[str, Order] = {}
        self._balance: Dict[str, Decimal] = {}

        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for system events."""
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self.event_bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._handle_signal_generated)
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._handle_error)

    async def start(self) -> None:
        """Start the trading engine."""
        if self._is_running:
            logger.warning("Trading engine is already running")
            return

        try:
            # Connect to exchange
            connected = await self.exchange_client.connect()
            if not connected:
                raise TradingSystemError("Failed to connect to exchange")

            # Load initial data
            await self._update_account_info()

            self._is_running = True
            logger.info("Trading engine started successfully")

            # Start main trading loop
            await self._run_trading_loop()

        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the trading engine."""
        if not self._is_running:
            return

        self._is_running = False
        self._trading_enabled = False

        # Cancel all active orders
        await self._cancel_all_orders()

        # Disconnect from exchange
        await self.exchange_client.disconnect()

        logger.info("Trading engine stopped")

    async def enable_trading(self) -> None:
        """Enable automatic trading."""
        if not self._is_running:
            raise TradingSystemError("Trading engine is not running")

        self._trading_enabled = True
        logger.info("Trading enabled")

    async def disable_trading(self) -> None:
        """Disable automatic trading."""
        self._trading_enabled = False
        logger.info("Trading disabled")

    async def _run_trading_loop(self) -> None:
        """Main trading loop."""
        loop_interval = self.config_manager.get("trading_loop_interval", 10)

        while self._is_running:
            try:
                if self._trading_enabled:
                    await self._execute_trading_cycle()

                await asyncio.sleep(loop_interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await self.event_bus.publish(
                    Event(
                        type=EventType.ERROR_OCCURRED,
                        data={"error": str(e), "source": "trading_loop"},
                        timestamp=datetime.now()
                    )
                )

    async def _execute_trading_cycle(self) -> None:
        """Execute one trading cycle."""
        # Update account information
        await self._update_account_info()

        # Get active agent
        active_agent = self.agent_manager.get_active_agent()
        if active_agent is None:
            return

        # Get symbols to trade
        symbols = self.config_manager.get("trading_symbols", ["BTC/USDT"])

        for symbol in symbols:
            try:
                # Get market data
                market_data = await self._get_market_data_for_analysis(symbol)
                if not market_data:
                    continue

                # Generate trading signal
                signal = await active_agent.analyze(market_data)
                if signal is None:
                    continue

                # Publish signal event
                await self.event_bus.publish(
                    Event(
                        type=EventType.SIGNAL_GENERATED,
                        data={"signal": signal, "agent": active_agent.get_name()},
                        timestamp=datetime.now()
                    )
                )

                # Execute signal if trading is enabled
                if self._trading_enabled:
                    await self._execute_signal(signal)

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")

    async def _get_market_data_for_analysis(self, symbol: str) -> List[MarketData]:
        """Get market data for analysis."""
        try:
            # Get current market data
            current_data = await self.exchange_client.get_market_data(symbol)

            # Get historical data for analysis
            lookback_hours = self.config_manager.get("analysis_lookback_hours", 24)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)

            historical_data = await self.exchange_client.get_historical_data(
                symbol, "1h", start_time, end_time
            )

            # Combine current and historical data
            if historical_data:
                historical_data.append(current_data)
                return historical_data
            else:
                return [current_data]

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []

    async def _execute_signal(self, signal: TradingSignal) -> None:
        """Execute a trading signal."""
        try:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(signal, self._balance)
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated: {position_size}")
                return

            # Create order
            order = Order(
                id="",  # Will be set by exchange
                symbol=signal.symbol,
                side=signal.action,
                type=OrderType.MARKET if signal.price is None else OrderType.LIMIT,
                amount=position_size,
                price=signal.price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now()
            )

            # Validate order with risk manager
            if not self.risk_manager.validate_order(order, self._positions):
                logger.warning(f"Order rejected by risk manager: {order}")
                return

            # Place order
            placed_order = await self.exchange_client.place_order(order)
            self._active_orders[placed_order.id] = placed_order

            logger.info(f"Order placed: {placed_order}")

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            raise

    async def _update_account_info(self) -> None:
        """Update account information."""
        try:
            # Update balance
            self._balance = await self.exchange_client.get_balance()

            # Update positions
            self._positions = await self.exchange_client.get_positions()

            # Update active orders status
            await self._update_order_statuses()

        except Exception as e:
            logger.error(f"Failed to update account info: {e}")

    async def _update_order_statuses(self) -> None:
        """Update status of active orders."""
        orders_to_remove = []

        for order_id, order in self._active_orders.items():
            try:
                updated_order = await self.exchange_client.get_order_status(order_id)
                self._active_orders[order_id] = updated_order

                # Remove completed orders
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    orders_to_remove.append(order_id)

                    if updated_order.status == OrderStatus.FILLED:
                        await self.event_bus.publish(
                            Event(
                                type=EventType.ORDER_FILLED,
                                data={"order": updated_order},
                                timestamp=datetime.now()
                            )
                        )

            except Exception as e:
                logger.error(f"Failed to update order status for {order_id}: {e}")

        # Remove completed orders
        for order_id in orders_to_remove:
            del self._active_orders[order_id]

    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        for order_id in list(self._active_orders.keys()):
            try:
                await self.exchange_client.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")

        self._active_orders.clear()

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled event."""
        order = event.data["order"]
        logger.info(f"Order filled: {order}")

    async def _handle_order_cancelled(self, event: Event) -> None:
        """Handle order cancelled event."""
        order = event.data["order"]
        logger.info(f"Order cancelled: {order}")

    async def _handle_signal_generated(self, event: Event) -> None:
        """Handle signal generated event."""
        signal = event.data["signal"]
        agent = event.data["agent"]
        logger.info(f"Signal generated by {agent}: {signal}")

    async def _handle_error(self, event: Event) -> None:
        """Handle error event."""
        error = event.data["error"]
        source = event.data.get("source", "unknown")
        logger.error(f"Error from {source}: {error}")

    # Public API methods

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self._is_running,
            "trading_enabled": self._trading_enabled,
            "active_agent": self.agent_manager._active_agent,
            "active_orders": len(self._active_orders),
            "positions": len(self._positions),
            "balance": {k: float(v) for k, v in self._balance.items()},
            "timestamp": datetime.now().isoformat()
        }

    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return self._positions.copy()

    def get_active_orders(self) -> List[Order]:
        """Get active orders."""
        return list(self._active_orders.values())

    def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        return self._balance.copy()