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
    IAgentManager, MarketData, Order, Position, TradingSignal, OrderStatus, OrderSide, OrderType,
    Event, EventType
)
from .order_executor import OrderExecutor, ImmediateExecutionStrategy, SignalExecutionStrategy
from .account_state_manager import AccountStateManager
from ..core.exceptions import TradingSystemError, RiskManagementError


class TradingEngine:
    """Main trading engine coordinating all system components."""

    def __init__(
        self,
        exchange_client: IExchangeClient,
        risk_manager: IRiskManager,
        event_bus: IEventBus,
        config_manager: IConfigManager,
        agent_manager: IAgentManager,
        execution_strategy: Optional[SignalExecutionStrategy] = None
    ):
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.agent_manager = agent_manager

        # Initialize order executor
        self.order_executor = OrderExecutor(exchange_client, risk_manager)
        self.execution_strategy = execution_strategy or ImmediateExecutionStrategy()

        # Initialize account state manager
        self.account_state_manager = AccountStateManager(exchange_client, event_bus)

        self._is_running = False
        self._trading_enabled = False

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

        logger.info("=" * 70)
        logger.info("🚀 STARTING TRADING ENGINE")
        logger.info("=" * 70)

        try:
            # Connect to exchange
            logger.info("Connecting to exchange...")
            logger.info(f"Exchange client type: {type(self.exchange_client).__name__}")
            connected = await self.exchange_client.connect()
            if not connected:
                logger.error("Exchange connection returned False")
                raise TradingSystemError("Failed to connect to exchange")
            logger.info("✅ Connected to exchange")

            # Load initial data
            logger.info("Loading account information...")
            await self.account_state_manager.update_account_info()
            balance = self.account_state_manager.get_balance()
            logger.info(f"✅ Account balance: {dict(balance)}")

            self._is_running = True
            self._trading_enabled = True  # Enable trading by default when started

            logger.info("=" * 70)
            logger.info("✅ TRADING ENGINE STARTED SUCCESSFULLY")
            logger.info(f"📊 Trading loop will run every {self.config_manager.get('trading_loop_interval', 10)} seconds")
            logger.info("🤖 Agents will analyze market data and generate signals")
            logger.info("=" * 70)

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

        logger.info("Stopping trading engine...")

        # Set flags to stop the trading loop
        self._is_running = False
        self._trading_enabled = False

        # Give the trading loop time to finish current cycle and exit
        # Wait up to 5 seconds (loop now exits within 1 second)
        for i in range(5):
            await asyncio.sleep(1)

        # Cancel orders with timeout to prevent hanging
        try:
            logger.info("Cancelling pending orders...")
            await asyncio.wait_for(self._cancel_all_orders(), timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning("Order cancellation timed out after 20 seconds")
        except Exception as e:
            logger.warning(f"Error cancelling orders during shutdown: {e}")

        # ALWAYS disconnect from exchange (even if cancel failed)
        try:
            logger.info("Disconnecting from exchange...")
            await asyncio.wait_for(self.exchange_client.disconnect(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Exchange disconnect timed out after 10 seconds")
        except Exception as e:
            logger.warning(f"Error disconnecting during shutdown: {e}")

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
                # Check again before executing cycle (user might have stopped during sleep)
                if not self._is_running:
                    break

                if self._trading_enabled:
                    await self._execute_trading_cycle()

                # Sleep in small chunks to check _is_running flag more frequently
                for _ in range(loop_interval):
                    if not self._is_running:
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await self.event_bus.publish(
                    Event(
                        type=EventType.ERROR_OCCURRED,
                        data={"error": str(e), "source": "trading_loop"},
                        timestamp=datetime.now()
                    )
                )

        logger.info("Trading loop exited cleanly")

    async def _execute_trading_cycle(self) -> None:
        """Execute one trading cycle."""
        logger.debug("Starting trading cycle...")

        # Update account information
        await self.account_state_manager.update_account_info()

        # Get symbols to trade
        symbols = self.config_manager.get("trading.symbols", ["BTC-USDT"])
        logger.debug(f"Analyzing symbols: {symbols}")

        # Check if multi-agent mode is enabled
        is_multi_agent = self.agent_manager.is_multi_agent_mode()
        if is_multi_agent:
            active_agents = self.agent_manager.get_active_agents()
            logger.info(f"Multi-agent mode: {len(active_agents)} agents active")
        else:
            active_agent = self.agent_manager.get_active_agent()
            logger.info(f"Single agent mode: {active_agent.get_name() if active_agent else 'None'}")

        for symbol in symbols:
            try:
                # Get market data
                market_data = await self._get_market_data_for_analysis(symbol)
                if not market_data:
                    continue

                if is_multi_agent:
                    # Multi-agent mode: get signals from all active agents
                    logger.debug(f"Fetching signals from all agents for {symbol}...")
                    signals_dict = await self.agent_manager.analyze_with_all_agents(market_data)

                    if not signals_dict:
                        logger.debug(f"No signals from any active agents for {symbol}")
                        continue

                    # Process each agent's signal
                    signals_received = 0
                    for agent_name, signal in signals_dict.items():
                        if signal is None:
                            logger.debug(f"{agent_name}: No signal (HOLD/neutral)")
                            continue

                        signals_received += 1
                        action = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
                        logger.info(f"📊 {agent_name} → {action} {signal.symbol} (confidence: {signal.confidence:.1%})")

                        # Ensure signal has agent name
                        signal.strategy_name = agent_name

                        # Publish signal event
                        await self.event_bus.publish(
                            Event(
                                type=EventType.SIGNAL_GENERATED,
                                data={"signal": signal, "agent": agent_name},
                                timestamp=datetime.now()
                            )
                        )

                        # Execute signal if trading is enabled
                        if self._trading_enabled:
                            await self._execute_signal(signal)
                        else:
                            logger.warning(f"Trading disabled - signal from {agent_name} not executed")

                    if signals_received == 0:
                        logger.debug(f"All agents returned HOLD for {symbol}")

                else:
                    # Single agent mode: use active agent
                    active_agent = self.agent_manager.get_active_agent()
                    if active_agent is None:
                        logger.warning("No active agent set")
                        return

                    # Generate trading signal
                    logger.debug(f"Fetching signal from {active_agent.get_name()} for {symbol}...")
                    signal = await active_agent.analyze(market_data)

                    if signal is None:
                        logger.debug(f"{active_agent.get_name()}: No signal (HOLD/neutral) for {symbol}")
                        continue

                    action = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
                    logger.info(f"📊 {active_agent.get_name()} → {action} {signal.symbol} (confidence: {signal.confidence:.1%})")

                    # Ensure signal has agent name
                    signal.strategy_name = active_agent.get_name()

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
                    else:
                        logger.warning(f"Trading disabled - signal from {active_agent.get_name()} not executed")

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")

    async def _get_market_data_for_analysis(self, symbol: str) -> List[MarketData]:
        """Get market data for analysis."""
        try:
            # Get current market data
            current_data = await self.exchange_client.get_market_data(symbol)

            # Get historical data for analysis
            lookback_hours = self.config_manager.get("trading.analysis_lookback_hours", 24)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)
            logger.info(f"Fetching {lookback_hours} hours of historical data for {symbol}")

            historical_data = await self.exchange_client.get_historical_data(
                symbol, "1h", start_time, end_time
            )

            # Combine current and historical data
            if historical_data:
                historical_data.append(current_data)
                logger.info(f"Retrieved {len(historical_data)} data points for {symbol}")
                return historical_data
            else:
                logger.warning(f"No historical data returned, using only current candle")
                return [current_data]

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []

    async def _execute_signal(self, signal: TradingSignal) -> None:
        """Execute a trading signal using the order executor."""
        try:
            # Get current balance and positions
            balance = self.account_state_manager.get_balance()
            positions = self.account_state_manager.get_positions()

            # Execute signal using strategy pattern
            placed_order = await self.execution_strategy.execute(
                signal,
                self.order_executor,
                balance,
                positions
            )

            # Track the order
            self.account_state_manager.add_active_order(placed_order)

            # Publish ORDER_PLACED event so GUI can update immediately
            await self.event_bus.publish(
                Event(
                    type=EventType.ORDER_PLACED,
                    data={"order": placed_order},
                    timestamp=datetime.now()
                )
            )

            # Skip immediate status check - the API endpoint seems unreliable for newly placed orders
            # Orders will be checked in the periodic update cycle instead
            # This prevents spamming error logs for orders that haven't propagated yet

        except TradingSystemError as e:
            logger.warning(f"Signal execution skipped: {e}")
            # Publish error event so GUI can display it
            await self.event_bus.publish(
                Event(
                    type=EventType.ERROR_OCCURRED,
                    data={
                        "error": str(e),
                        "source": "order_execution",
                        "signal": signal.symbol,
                        "action": signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
                    },
                    timestamp=datetime.now()
                )
            )
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            # Publish error event
            await self.event_bus.publish(
                Event(
                    type=EventType.ERROR_OCCURRED,
                    data={
                        "error": str(e),
                        "source": "order_execution"
                    },
                    timestamp=datetime.now()
                )
            )
            raise

    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        try:
            # Get all pending orders from the exchange to ensure we cancel everything
            logger.info("Fetching pending orders from exchange...")
            pending_orders = await asyncio.wait_for(
                self.exchange_client.get_pending_orders(),
                timeout=10.0
            )

            if not pending_orders:
                logger.info("No pending orders to cancel")
                self.account_state_manager.clear_active_orders()
                return

            logger.info(f"Found {len(pending_orders)} pending orders to cancel")

            # Extract order IDs
            order_ids = [order.id for order in pending_orders]

            # Use batch cancellation for efficiency
            logger.info(f"Cancelling {len(order_ids)} orders in batch...")
            results = await asyncio.wait_for(
                self.exchange_client.cancel_batch_orders(order_ids),
                timeout=10.0
            )

            # Log results
            successful = sum(1 for success in results.values() if success)
            failed = len(results) - successful

            logger.info(f"Order cancellation complete: {successful} cancelled, {failed} failed")

            if failed > 0:
                failed_orders = [order_id for order_id, success in results.items() if not success]
                logger.warning(f"Failed to cancel orders: {failed_orders}")

        except asyncio.TimeoutError:
            logger.warning("Order cancellation API calls timed out")
            # Clear local tracking anyway
            self.account_state_manager.clear_active_orders()
        except Exception as e:
            logger.warning(f"Error during order cancellation: {e}")
            # Clear local tracking anyway
            self.account_state_manager.clear_active_orders()

    async def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions by placing market orders.

        Returns:
            Dictionary with results: {
                'success': bool,
                'closed_count': int,
                'failed_count': int,
                'details': List[str]
            }
        """
        positions = self.account_state_manager.get_positions()

        if not positions:
            logger.info("No open positions to close")
            return {
                'success': True,
                'closed_count': 0,
                'failed_count': 0,
                'details': ['No open positions']
            }

        closed_count = 0
        failed_count = 0
        details = []

        logger.info(f"Closing {len(positions)} open positions...")

        for position in positions:
            try:
                # Determine order side (opposite of position)
                order_side = OrderSide.SELL if position.is_long else OrderSide.BUY

                # Use position amount directly (in BTC)
                # The exchange client will handle BTC->contracts conversion and precision rounding
                position_amount_btc = abs(position.amount)
                logger.debug(f"Closing position: {position.symbol} {position_amount_btc} BTC")

                # Create MARKET order to close position immediately
                # IMPORTANT: Set reduce_only=True to ensure we only close positions, not open new ones
                close_order = Order(
                    id="",  # Will be set by exchange
                    symbol=position.symbol,
                    side=order_side,
                    type=OrderType.MARKET,  # Use market order for immediate fill
                    amount=position_amount_btc,  # Amount in BTC, will be converted to contracts by exchange client
                    price=None,  # No price for market orders
                    status=OrderStatus.PENDING,
                    timestamp=datetime.now(),
                    reduce_only=True  # Only reduces position, won't open new position if size exceeds
                )

                # Place the order
                order = await self.exchange_client.place_order(close_order)

                # Log order placement
                logger.info(f"Close order placed: {order.id} - {position.symbol} {order_side.value} {position_amount_btc} BTC (MARKET)")

                # Wait briefly and check if filled (best effort)
                await asyncio.sleep(0.5)
                try:
                    updated_order = await self.exchange_client.get_order_status(order.id)
                    if updated_order.status == OrderStatus.FILLED:
                        closed_count += 1
                        details.append(f"✅ Closed {position.symbol}: {order_side.value} {position_amount_btc} BTC")
                        logger.info(f"✅ Position closed (filled): {position.symbol} - {order_side.value} {position_amount_btc} BTC")
                    else:
                        # Order placed but not filled yet
                        details.append(f"⏳ Close order placed for {position.symbol}: {order_side.value} {position_amount_btc} BTC (status: {updated_order.status.value})")
                        logger.warning(f"Close order placed but not filled yet: {order.id} - status: {updated_order.status.value}")
                except Exception as status_error:
                    # Could not verify status - assume pending
                    details.append(f"⏳ Close order placed for {position.symbol} (order {order.id})")
                    logger.debug(f"Could not verify order status: {status_error}")

            except Exception as e:
                failed_count += 1
                details.append(f"❌ Failed to close {position.symbol}: {str(e)}")
                logger.error(f"Failed to close position {position.symbol}: {e}")

        # Calculate pending orders (total - closed - failed)
        pending_count = len(positions) - closed_count - failed_count

        result = {
            'success': failed_count == 0,
            'closed_count': closed_count,
            'failed_count': failed_count,
            'pending_count': pending_count,
            'details': details
        }

        # Log comprehensive summary
        if pending_count > 0:
            logger.info(f"Position closing complete: {closed_count} filled, {pending_count} pending, {failed_count} failed")
            logger.warning(f"⚠️  {pending_count} close order(s) placed but not filled yet - check Blofin exchange for order status")
            logger.warning("Pending orders may fill in the next few seconds/minutes. Check the 'details' for order IDs.")
        else:
            logger.info(f"Position closing complete: {closed_count} closed, {failed_count} failed")

        # Force account state refresh to update positions immediately
        try:
            await self.account_state_manager.update_account_info()
            logger.debug("Account state refreshed after closing positions")
        except Exception as e:
            logger.warning(f"Failed to refresh account state after closing positions: {e}")

        return result

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
        is_multi_agent = self.agent_manager.is_multi_agent_mode()

        if is_multi_agent:
            # Multi-agent mode
            active_agents = self.agent_manager.get_active_agents()
            active_agent_names = [agent.get_name() for agent in active_agents]
            active_agent_name = None  # Not applicable in multi-agent mode
        else:
            # Single agent mode
            active_agent = self.agent_manager.get_active_agent()
            active_agent_name = active_agent.get_name() if active_agent else None
            active_agent_names = [active_agent_name] if active_agent_name else []

        balance = self.account_state_manager.get_balance()

        return {
            "is_running": self._is_running,
            "trading_enabled": self._trading_enabled,
            "multi_agent_mode": is_multi_agent,
            "active_agent": active_agent_name,  # For backward compatibility (single mode only)
            "active_agents": active_agent_names,  # List of active agents
            "active_orders": self.account_state_manager.get_active_order_count(),
            "positions": len(self.account_state_manager.get_positions()),
            "balance": {k: float(v) for k, v in balance.items()},
            "timestamp": datetime.now().isoformat()
        }

    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return self.account_state_manager.get_positions()

    def get_active_orders(self) -> List[Order]:
        """Get active orders."""
        return self.account_state_manager.get_active_orders()

    def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance."""
        return self.account_state_manager.get_balance()