"""
Simplified Trading GUI for the cryptocurrency trading application.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
from typing import Optional
from loguru import logger

from ..core.trading_engine import TradingEngine
from ..core.config_manager import ConfigManager
from ..core.event_bus import EventBus
from ..core.risk_manager import RiskManager
from ..core.agent_manager import AgentManager
from ..core.interfaces import Event, EventType
from ..exchange.blofin_client import BlofinClient
from ..utils.agent_initializer import initialize_agents, get_agent_status


class TradingGUI:
    """
    Simplified trading GUI providing basic interface for monitoring and control.
    """

    def __init__(self):
        """Initialize the trading GUI."""
        self.root = None
        self.trading_engine: Optional[TradingEngine] = None
        self.config_manager: Optional[ConfigManager] = None
        self.agent_manager: Optional[AgentManager] = None
        self.event_bus: Optional[EventBus] = None

        # GUI state
        self.is_running = False
        self.update_interval = 1000  # ms
        self.positions_update_interval = 2000  # ms - update positions every 2 seconds

        # Trading activity tracking
        self.signal_count = 0
        self.order_count = 0

        # Market indicators
        self.current_rsi = None
        self.rsi_oversold_threshold = 45
        self.rsi_overbought_threshold = 55

        # Agent tracking - maps agent names to their orders and signals
        self.agent_orders = {}  # agent_name -> list of order_ids
        self.agent_signals = {}  # agent_name -> last signal info
        self.order_to_agent = {}  # order_id -> agent_name

        # Initialize components
        self._initialize_backend()
        self._initialize_gui()

    def _initialize_backend(self):
        """Initialize backend trading components."""
        try:
            logger.info("Initializing trading backend...")

            # Initialize components
            self.config_manager = ConfigManager()
            self.event_bus = EventBus()
            risk_manager = RiskManager(self.config_manager)
            self.agent_manager = AgentManager()

            # Initialize and register trading agents
            initialize_agents(self.agent_manager, self.config_manager)

            # Subscribe to trading events
            self._subscribe_to_events()

            # Get exchange configuration
            exchange_config = self.config_manager.get_section('exchange')

            # Use mock exchange for testing or real exchange
            exchange_name = exchange_config.get('name', 'blofin')
            if exchange_name == 'mock':
                from ..exchange.mock_client import MockExchangeClient
                exchange_client = MockExchangeClient()
                logger.warning("🎭 Using MOCK exchange - Paper trading mode (no real trades)")
            else:
                exchange_client = BlofinClient(
                    api_key=exchange_config.get('api_key', ''),
                    api_secret=exchange_config.get('api_secret', ''),
                    passphrase=exchange_config.get('passphrase', ''),
                    sandbox=exchange_config.get('sandbox', True)
                )

            # Initialize trading engine
            self.trading_engine = TradingEngine(
                exchange_client,
                risk_manager,
                self.event_bus,
                self.config_manager,
                self.agent_manager
            )

            logger.info("Trading backend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize trading backend: {e}")
            raise

    def _subscribe_to_events(self):
        """Subscribe to trading events from the event bus."""
        try:
            # Subscribe to signal events
            self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal_generated)

            # Subscribe to order events
            self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
            self.event_bus.subscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)

            # Subscribe to error events
            self.event_bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

            logger.info("Subscribed to trading events")
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")

    async def _on_signal_generated(self, event: Event):
        """Handle signal generated event."""
        try:
            signal = event.data.get('signal')
            agent = event.data.get('agent', 'Unknown')

            self.signal_count += 1

            # Format message
            action = signal.action.value.upper() if hasattr(signal.action, 'value') else str(signal.action).upper()
            symbol = signal.symbol
            confidence = signal.confidence * 100

            # Extract RSI value from metadata if available
            if hasattr(signal, 'metadata') and signal.metadata:
                rsi_current = signal.metadata.get('rsi_current')
                if rsi_current is not None:
                    self.current_rsi = rsi_current
                    self.rsi_oversold_threshold = signal.metadata.get('oversold_threshold', 45)
                    self.rsi_overbought_threshold = signal.metadata.get('overbought_threshold', 55)

            # Track agent signal
            self.agent_signals[agent] = {
                'action': action,
                'symbol': symbol,
                'confidence': confidence,
                'timestamp': datetime.now()
            }

            message = f"🔔 SIGNAL #{self.signal_count}: {agent} → {action} {symbol} (Confidence: {confidence:.1f}%)"

            # Update GUI on main thread
            if self.root:
                self.root.after(0, lambda: self._log_message(message, "signal"))
                self.root.after(0, lambda: self.signal_counter_label.config(text=str(self.signal_count)))
                self.root.after(0, lambda: self._update_agent_status_display())
                self.root.after(0, lambda: self._update_rsi_display())

            logger.info(f"Signal: {agent} {action} {symbol} confidence={confidence:.1f}%")
        except Exception as e:
            logger.error(f"Error handling signal event: {e}")

    async def _on_order_filled(self, event: Event):
        """Handle order filled event."""
        try:
            order = event.data.get('order')
            self.order_count += 1

            if order:
                action = order.side.value.upper() if hasattr(order.side, 'value') else str(order.side).upper()
                symbol = order.symbol
                amount = order.filled_amount if hasattr(order, 'filled_amount') else order.amount
                price = order.average_price if hasattr(order, 'average_price') and order.average_price else order.price

                # Track order to agent mapping
                agent_name = order.metadata.get('agent_name', 'Unknown') if hasattr(order, 'metadata') and order.metadata else 'Unknown'
                if agent_name != 'Unknown':
                    self.order_to_agent[order.id] = agent_name
                    if agent_name not in self.agent_orders:
                        self.agent_orders[agent_name] = []
                    self.agent_orders[agent_name].append(order.id)

                message = f"✅ ORDER #{self.order_count}: {action} {amount} {symbol} @ ${price}"
            else:
                message = f"✅ ORDER #{self.order_count}: Order filled"

            # Update GUI on main thread
            if self.root:
                self.root.after(0, lambda: self._log_message(message, "order"))
                self.root.after(0, lambda: self.order_counter_label.config(text=str(self.order_count)))
                # Update positions display since a new order was filled
                self.root.after(0, lambda: self._update_positions_display())
                # Update agent status display
                self.root.after(0, lambda: self._update_agent_status_display())

            logger.info(f"Order filled: {message}")
        except Exception as e:
            logger.error(f"Error handling order filled event: {e}")

    async def _on_order_cancelled(self, event: Event):
        """Handle order cancelled event."""
        try:
            order = event.data.get('order')

            if order:
                message = f"❌ ORDER CANCELLED: {order.symbol}"
            else:
                message = "❌ ORDER CANCELLED"

            # Update GUI on main thread
            if self.root:
                self.root.after(0, lambda: self._log_message(message, "cancelled"))

            logger.info(f"Order cancelled: {message}")
        except Exception as e:
            logger.error(f"Error handling order cancelled event: {e}")

    async def _on_error(self, event: Event):
        """Handle error event."""
        try:
            error = event.data.get('error', 'Unknown error')
            source = event.data.get('source', 'Unknown')

            message = f"⚠️  ERROR from {source}: {error}"

            # Update GUI on main thread
            if self.root:
                self.root.after(0, lambda: self._log_message(message, "error"))

            logger.error(f"Trading error: {message}")
        except Exception as e:
            logger.error(f"Error handling error event: {e}")

    def _initialize_gui(self):
        """Initialize the GUI components."""
        try:
            self.root = tk.Tk()
            self.root.title("Cryptocurrency Trading System")
            self.root.geometry("1800x1200")
            self.root.minsize(1400, 1000)

            # Configure style
            self._setup_styles()

            # Create main layout
            self._create_layout()

            # Setup event handlers
            self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

            # Display initial status
            self._update_status()

            logger.info("GUI initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GUI: {e}")
            raise

    def _setup_styles(self):
        """Setup GUI styles."""
        style = ttk.Style()

        # Configure colors
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"

        self.root.configure(bg=bg_color)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", padding=6)
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))

    def _create_layout(self):
        """Create the main application layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(
            header_frame,
            text="Cryptocurrency Trading System",
            style="Header.TLabel"
        )
        title_label.pack(side=tk.LEFT)

        # Status
        self.status_label = ttk.Label(header_frame, text="Status: Initializing...")
        self.status_label.pack(side=tk.RIGHT)

        # Connection Status Indicators
        status_indicator_frame = ttk.Frame(main_frame)
        status_indicator_frame.pack(fill=tk.X, pady=(0, 10))

        # Exchange Connection Status
        exchange_status_frame = tk.Frame(status_indicator_frame, bg="#2b2b2b")
        exchange_status_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(
            exchange_status_frame,
            text="Exchange:",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.exchange_status_indicator = tk.Label(
            exchange_status_frame,
            text="● DISCONNECTED",
            bg="#2b2b2b",
            fg="#FF4444",
            font=("Arial", 10, "bold")
        )
        self.exchange_status_indicator.pack(side=tk.LEFT)

        # Engine Status
        engine_status_frame = tk.Frame(status_indicator_frame, bg="#2b2b2b")
        engine_status_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(
            engine_status_frame,
            text="Engine:",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.engine_status_indicator = tk.Label(
            engine_status_frame,
            text="● STOPPED",
            bg="#2b2b2b",
            fg="#888888",
            font=("Arial", 10, "bold")
        )
        self.engine_status_indicator.pack(side=tk.LEFT)

        # Trading Status
        trading_status_frame = tk.Frame(status_indicator_frame, bg="#2b2b2b")
        trading_status_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(
            trading_status_frame,
            text="Trading:",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.trading_status_indicator = tk.Label(
            trading_status_frame,
            text="● DISABLED",
            bg="#2b2b2b",
            fg="#888888",
            font=("Arial", 10, "bold")
        )
        self.trading_status_indicator.pack(side=tk.LEFT)

        # RSI Indicator
        rsi_status_frame = tk.Frame(status_indicator_frame, bg="#2b2b2b")
        rsi_status_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(
            rsi_status_frame,
            text="RSI:",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.rsi_indicator = tk.Label(
            rsi_status_frame,
            text="--",
            bg="#2b2b2b",
            fg="#FFFFFF",
            font=("Arial", 10, "bold")
        )
        self.rsi_indicator.pack(side=tk.LEFT)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Trading",
            command=self._start_trading,
            state=tk.DISABLED  # Disabled until API keys configured
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(
            btn_frame,
            text="Pause Trading",
            command=self._pause_trading,
            state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="Stop Trading",
            command=self._stop_trading,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        refresh_btn = ttk.Button(
            btn_frame,
            text="Refresh Status",
            command=self._update_status
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        self.close_positions_btn = ttk.Button(
            btn_frame,
            text="Close All Positions",
            command=self._close_all_positions,
            state=tk.DISABLED
        )
        self.close_positions_btn.pack(side=tk.LEFT, padx=5)

        # Agent selection
        agent_frame = ttk.LabelFrame(control_frame, text="Agent Management", padding="10")
        agent_frame.pack(fill=tk.X, pady=(10, 0))

        # Mode selection
        mode_frame = ttk.Frame(agent_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)

        self.agent_mode_var = tk.StringVar(value="single")
        single_radio = ttk.Radiobutton(
            mode_frame,
            text="Single Agent",
            variable=self.agent_mode_var,
            value="single",
            command=self._on_mode_changed
        )
        single_radio.pack(side=tk.LEFT, padx=5)

        multi_radio = ttk.Radiobutton(
            mode_frame,
            text="Multiple Agents (Parallel)",
            variable=self.agent_mode_var,
            value="multi",
            command=self._on_mode_changed
        )
        multi_radio.pack(side=tk.LEFT, padx=5)

        # Single agent selection (shown by default)
        self.single_agent_frame = ttk.Frame(agent_frame)
        self.single_agent_frame.pack(fill=tk.X)

        ttk.Label(self.single_agent_frame, text="Active Agent:").pack(side=tk.LEFT, padx=5)

        self.active_agent_label = ttk.Label(
            self.single_agent_frame,
            text="None",
            foreground="green",
            font=("Arial", 10, "bold")
        )
        self.active_agent_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.single_agent_frame, text="Select:").pack(side=tk.LEFT, padx=(20, 5))

        self.agent_var = tk.StringVar()
        self.agent_dropdown = ttk.Combobox(
            self.single_agent_frame,
            textvariable=self.agent_var,
            state="readonly",
            width=20
        )
        self.agent_dropdown.pack(side=tk.LEFT, padx=5)
        self.agent_dropdown.bind("<<ComboboxSelected>>", self._on_agent_selected)

        # Multi-agent selection (hidden by default)
        self.multi_agent_frame = ttk.Frame(agent_frame)
        # Don't pack yet - will show when multi mode is selected

        ttk.Label(self.multi_agent_frame, text="Select agents to run in parallel:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(5, 5))

        # Scrollable frame for agent checkboxes
        self.agent_checkboxes_frame = ttk.Frame(self.multi_agent_frame)
        self.agent_checkboxes_frame.pack(fill=tk.X)

        # Store checkbox variables
        self.agent_check_vars = {}

        # Apply button for multi-agent selection
        apply_frame = ttk.Frame(self.multi_agent_frame)
        apply_frame.pack(fill=tk.X, pady=(10, 0))

        self.apply_agents_btn = ttk.Button(
            apply_frame,
            text="Apply Agent Selection",
            command=self._apply_multi_agent_selection
        )
        self.apply_agents_btn.pack(side=tk.LEFT)

        # Trading activity panel
        activity_frame = ttk.LabelFrame(main_frame, text="Trading Activity", padding="10")
        activity_frame.pack(fill=tk.X, pady=(0, 10))

        # Activity counters
        counters_frame = ttk.Frame(activity_frame)
        counters_frame.pack(fill=tk.X)

        # Signal counter
        ttk.Label(counters_frame, text="Signals Generated:").pack(side=tk.LEFT, padx=5)
        self.signal_counter_label = ttk.Label(
            counters_frame,
            text="0",
            foreground="#00BFFF",
            font=("Arial", 11, "bold")
        )
        self.signal_counter_label.pack(side=tk.LEFT, padx=5)

        # Order counter
        ttk.Label(counters_frame, text="Orders Executed:").pack(side=tk.LEFT, padx=(20, 5))
        self.order_counter_label = ttk.Label(
            counters_frame,
            text="0",
            foreground="#00FF00",
            font=("Arial", 11, "bold")
        )
        self.order_counter_label.pack(side=tk.LEFT, padx=5)

        # Configuration info
        info_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.config_text = scrolledtext.ScrolledText(
            info_frame,
            height=8,
            bg="#1e1e1e",
            fg="#ffffff",
            font=("Courier", 10)
        )
        self.config_text.pack(fill=tk.BOTH, expand=True)

        # Agent Status Panel - shows each agent's positions and orders
        agent_status_frame = ttk.LabelFrame(main_frame, text="Agent Status", padding="10")
        agent_status_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.agent_status_text = scrolledtext.ScrolledText(
            agent_status_frame,
            height=8,
            bg="#1e1e1e",
            fg="#00FF00",
            font=("Courier", 10)
        )
        self.agent_status_text.pack(fill=tk.BOTH, expand=True)

        # Open Positions panel (consolidated view)
        positions_frame = ttk.LabelFrame(main_frame, text="All Open Positions", padding="10")
        positions_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.positions_text = scrolledtext.ScrolledText(
            positions_frame,
            height=5,
            bg="#1e1e1e",
            fg="#00FF00",
            font=("Courier", 10)
        )
        self.positions_text.pack(fill=tk.BOTH, expand=True)

        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="System Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=20,
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Add initial log message
        self._log_message("System initialized")
        self._log_message("Welcome to Cryptocurrency Trading System")
        self._log_message("Agent status and positions update automatically every 2 seconds", "info")

    def _update_status(self):
        """Update status display."""
        try:
            # Get configuration
            trading_enabled = self.config_manager.get("trading.enabled", False)
            exchange_name = self.config_manager.get("exchange.name", "Unknown")
            sandbox_mode = self.config_manager.get("exchange.sandbox", True)
            symbols = self.config_manager.get("trading.symbols", [])
            api_key_set = bool(self.config_manager.get("exchange.api_key", ""))

            # Get agent status
            agent_status = get_agent_status(self.agent_manager)
            available_agents = agent_status['available_agents']
            active_agent = agent_status['active_agent']

            # Update agent dropdown and checkboxes
            self._update_agent_controls(available_agents, active_agent)

            # Update active agent label (for single mode)
            if active_agent:
                self.active_agent_label.config(text=active_agent, foreground="green")
            else:
                self.active_agent_label.config(text="None", foreground="orange")

            # Update status label
            if trading_enabled:
                self.status_label.config(text="Status: Trading Active", foreground="green")
                self.pause_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="Status: Trading Disabled", foreground="orange")
                self.pause_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.DISABLED)

            # Enable/disable start button based on API key or mock exchange
            # Mock exchange doesn't need API keys
            is_mock_exchange = exchange_name.lower() == "mock"
            can_trade = api_key_set or is_mock_exchange

            if can_trade:
                self.start_btn.config(state=tk.NORMAL if not trading_enabled else tk.DISABLED)
                # Enable close positions button if API configured or mock
                self.close_positions_btn.config(state=tk.NORMAL)
            else:
                self.start_btn.config(state=tk.DISABLED)
                self.close_positions_btn.config(state=tk.DISABLED)

            # Build agent list for display - handle both single and multi-agent modes
            is_multi_mode = self.agent_manager.is_multi_agent_mode()

            if is_multi_mode:
                # Multi-agent mode: show all active agents
                active_agents_objs = self.agent_manager.get_active_agents()
                active_agent_names = [agent.get_name() for agent in active_agents_objs]
                agent_list = '\n'.join([
                    f"  {'✓✓' if agent in active_agent_names else '•'} {agent}"
                    for agent in available_agents
                ]) if available_agents else "  None registered"
                agent_mode_text = f"MULTI-AGENT ({len(active_agent_names)} active)"
            else:
                # Single-agent mode: show one active agent
                agent_list = '\n'.join([
                    f"  {'✓' if agent == active_agent else '•'} {agent}"
                    for agent in available_agents
                ]) if available_agents else "  None registered"
                agent_mode_text = f"SINGLE-AGENT"

            # Update configuration display
            if is_mock_exchange:
                api_status = "NOT REQUIRED (Mock Exchange)"
            else:
                api_status = 'YES' if api_key_set else 'NO (Required for trading)'

            config_info = f"""
Exchange:           {exchange_name.upper()}
Mode:               {'SANDBOX (Paper Trading)' if sandbox_mode else 'LIVE TRADING'}
Trading Enabled:    {'YES' if trading_enabled else 'NO'}
API Key Configured: {api_status}
Symbols:            {', '.join(symbols)}

Risk Management:
  Max Position Size:  {self.config_manager.get('risk.max_position_size_pct', 0) * 100:.1f}%
  Max Daily Loss:     {self.config_manager.get('risk.max_daily_loss_pct', 0) * 100:.1f}%
  Max Exposure:       {self.config_manager.get('risk.max_total_exposure_pct', 0) * 100:.1f}%
  Portfolio Value:    ${self.config_manager.get('risk.portfolio_value', 0):,.2f}

Trading Agents ({len(available_agents)} registered) - {agent_mode_text}:
{agent_list}
"""

            self.config_text.delete(1.0, tk.END)
            self.config_text.insert(1.0, config_info.strip())

            # Update positions and agent status displays
            self._update_positions_display()
            self._update_agent_status_display()
            self._update_status_indicators()

            self._log_message("Status updated")

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _update_rsi_display(self):
        """Update RSI indicator display."""
        try:
            if self.current_rsi is None:
                self.rsi_indicator.config(text="--", fg="#888888")
                return

            rsi_value = self.current_rsi
            rsi_text = f"{rsi_value:.1f}"

            # Color code based on RSI levels
            if rsi_value <= self.rsi_oversold_threshold:
                # Oversold - Green (potential buy)
                color = "#00FF00"
                rsi_text += " ↓"
            elif rsi_value >= self.rsi_overbought_threshold:
                # Overbought - Red (potential sell)
                color = "#FF4444"
                rsi_text += " ↑"
            else:
                # Neutral zone - White/Gray
                color = "#FFFFFF"

            self.rsi_indicator.config(text=rsi_text, fg=color)

        except Exception as e:
            logger.error(f"Error updating RSI display: {e}")

    def _update_positions_display(self):
        """Update the positions display with current open positions."""
        try:
            # Get current positions from trading engine
            if not self.trading_engine:
                self.positions_text.delete(1.0, tk.END)
                self.positions_text.insert(1.0, "Trading engine not initialized")
                return

            positions = self.trading_engine.get_positions()

            # Clear existing display
            self.positions_text.delete(1.0, tk.END)

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.positions_text.insert(1.0, f"Last Updated: {timestamp}\n\n", "timestamp")
            self.positions_text.tag_config("timestamp", foreground="#888888", font=("Courier", 9, "italic"))

            if not positions:
                self.positions_text.insert(tk.END, "No open positions\n", "info")
                self.positions_text.tag_config("info", foreground="#888888")
                return

            # Header
            header = f"{'Symbol':<12} {'Side':<8} {'Amount':<15} {'Entry Price':<15} {'Current PnL':<15}\n"
            header += "=" * 80 + "\n"
            self.positions_text.insert(tk.END, header, "header")
            self.positions_text.tag_config("header", foreground="#00BFFF", font=("Courier", 10, "bold"))

            # Display each position
            for position in positions:
                side = "LONG (BUY)" if position.is_long else "SHORT (SELL)"
                side_tag = "long" if position.is_long else "short"

                # Format position details
                symbol = position.symbol[:12]
                amount = f"{abs(position.amount):.4f}"
                entry_price = f"${position.entry_price:.2f}" if hasattr(position, 'entry_price') and position.entry_price else "N/A"

                # Calculate unrealized PnL if available
                pnl_display = "N/A"
                pnl_tag = "neutral"
                if hasattr(position, 'unrealized_pnl') and position.unrealized_pnl is not None:
                    pnl = float(position.unrealized_pnl)
                    pnl_display = f"${pnl:+.2f}"
                    pnl_tag = "profit" if pnl > 0 else "loss" if pnl < 0 else "neutral"

                # Create line
                line = f"{symbol:<12} {side:<8} {amount:<15} {entry_price:<15} {pnl_display:<15}\n"

                # Insert with appropriate tags
                self.positions_text.insert(tk.END, f"{symbol:<12} ", "default")
                self.positions_text.insert(tk.END, f"{side:<8} ", side_tag)
                self.positions_text.insert(tk.END, f"{amount:<15} {entry_price:<15} ", "default")
                self.positions_text.insert(tk.END, f"{pnl_display:<15}\n", pnl_tag)

            # Configure tags with colors
            self.positions_text.tag_config("default", foreground="#FFFFFF")
            self.positions_text.tag_config("long", foreground="#00FF00")  # Green for long
            self.positions_text.tag_config("short", foreground="#FF6B6B")  # Red for short
            self.positions_text.tag_config("profit", foreground="#00FF00")  # Green for profit
            self.positions_text.tag_config("loss", foreground="#FF4444")  # Red for loss
            self.positions_text.tag_config("neutral", foreground="#888888")  # Gray for neutral

            # Add summary
            total_positions = len(positions)
            long_count = sum(1 for p in positions if p.is_long)
            short_count = total_positions - long_count

            summary = f"\n{'─' * 80}\n"
            summary += f"Total Positions: {total_positions}  |  Long: {long_count}  |  Short: {short_count}\n"

            self.positions_text.insert(tk.END, summary, "summary")
            self.positions_text.tag_config("summary", foreground="#FFD700")

        except Exception as e:
            logger.error(f"Error updating positions display: {e}")
            self.positions_text.delete(1.0, tk.END)
            self.positions_text.insert(1.0, f"Error loading positions: {e}", "error")
            self.positions_text.tag_config("error", foreground="#FF4444")

    def _update_agent_status_display(self):
        """Update the agent status display showing each agent's positions and orders."""
        try:
            # Clear existing display
            self.agent_status_text.delete(1.0, tk.END)

            # Get timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.agent_status_text.insert(1.0, f"Last Updated: {timestamp}\n\n", "timestamp")
            self.agent_status_text.tag_config("timestamp", foreground="#888888", font=("Courier", 9, "italic"))

            if not self.agent_manager:
                self.agent_status_text.insert(tk.END, "Agent manager not initialized\n", "error")
                return

            # Get all registered agents
            all_agents = self.agent_manager.list_agents()
            if not all_agents:
                self.agent_status_text.insert(tk.END, "No agents registered\n", "info")
                self.agent_status_text.tag_config("info", foreground="#888888")
                return

            # Get current positions and orders
            positions = self.trading_engine.get_positions() if self.trading_engine else []
            active_orders = self.trading_engine.get_active_orders() if self.trading_engine else []

            # Create agent-to-position mapping based on order tracking
            agent_positions = {}
            agent_active_orders = {}

            # Map positions to agents based on order tracking
            for position in positions:
                # Try to find which agent created this position
                # Since positions don't have agent info, we'll use order tracking
                found_agent = None
                for order_id, agent_name in self.order_to_agent.items():
                    # Match position symbol with order
                    for order in active_orders:
                        if order.id == order_id and order.symbol == position.symbol:
                            found_agent = agent_name
                            break
                    if found_agent:
                        break

                if found_agent:
                    if found_agent not in agent_positions:
                        agent_positions[found_agent] = []
                    agent_positions[found_agent].append(position)

            # Map active orders to agents
            for order in active_orders:
                agent_name = order.metadata.get('agent_name', None) if hasattr(order, 'metadata') and order.metadata else None
                if agent_name:
                    if agent_name not in agent_active_orders:
                        agent_active_orders[agent_name] = []
                    agent_active_orders[agent_name].append(order)

            # Display each agent's status
            header = f"{'Agent Name':<25} {'Status':<15} {'Position/Order':<40}\n"
            header += "=" * 90 + "\n"
            self.agent_status_text.insert(tk.END, header, "header")
            self.agent_status_text.tag_config("header", foreground="#00BFFF", font=("Courier", 10, "bold"))

            for agent_name in all_agents:
                # Agent name column
                agent_display = agent_name[:24]
                self.agent_status_text.insert(tk.END, f"{agent_display:<25} ", "agent_name")

                # Determine status and details
                has_position = agent_name in agent_positions and agent_positions[agent_name]
                has_orders = agent_name in agent_active_orders and agent_active_orders[agent_name]
                has_signal = agent_name in self.agent_signals

                if has_position:
                    # Show position info
                    pos = agent_positions[agent_name][0]  # Show first position
                    side = "LONG" if pos.is_long else "SHORT"
                    status_tag = "long" if pos.is_long else "short"
                    status_text = f"{side:<15}"
                    detail_text = f"{pos.symbol} {abs(pos.amount):.4f} @ ${pos.entry_price:.2f}"

                    if hasattr(pos, 'unrealized_pnl') and pos.unrealized_pnl:
                        pnl = float(pos.unrealized_pnl)
                        detail_text += f" (PnL: ${pnl:+.2f})"

                    self.agent_status_text.insert(tk.END, status_text, status_tag)
                    self.agent_status_text.insert(tk.END, f"{detail_text}\n", "detail")

                elif has_orders:
                    # Show order info
                    order = agent_active_orders[agent_name][0]  # Show first order
                    side = order.side.value.upper() if hasattr(order.side, 'value') else str(order.side).upper()
                    status_tag = "order"
                    status_text = f"ORDER {side:<9}"
                    detail_text = f"{order.symbol} {order.amount} @ ${order.price if order.price else 'MARKET'}"

                    self.agent_status_text.insert(tk.END, status_text, status_tag)
                    self.agent_status_text.insert(tk.END, f"{detail_text}\n", "detail")

                elif has_signal:
                    # Show last signal
                    signal_info = self.agent_signals[agent_name]
                    status_text = "SIGNALED      "
                    detail_text = f"{signal_info['action']} {signal_info['symbol']} ({signal_info['confidence']:.1f}% confidence)"

                    self.agent_status_text.insert(tk.END, status_text, "signal_status")
                    self.agent_status_text.insert(tk.END, f"{detail_text}\n", "detail")

                else:
                    # Neutral - no activity
                    status_text = "NEUTRAL       "
                    detail_text = "No position, orders, or signals"

                    self.agent_status_text.insert(tk.END, status_text, "neutral")
                    self.agent_status_text.insert(tk.END, f"{detail_text}\n", "detail_neutral")

            # Configure color tags
            self.agent_status_text.tag_config("agent_name", foreground="#FFFFFF", font=("Courier", 10, "bold"))
            self.agent_status_text.tag_config("long", foreground="#00FF00")  # Green for long
            self.agent_status_text.tag_config("short", foreground="#FF6B6B")  # Red for short
            self.agent_status_text.tag_config("order", foreground="#FFA500")  # Orange for orders
            self.agent_status_text.tag_config("signal_status", foreground="#00BFFF")  # Blue for signals
            self.agent_status_text.tag_config("neutral", foreground="#888888")  # Gray for neutral
            self.agent_status_text.tag_config("detail", foreground="#CCCCCC")
            self.agent_status_text.tag_config("detail_neutral", foreground="#666666")

            # Add summary
            active_count = len(agent_positions) + len(agent_active_orders)
            summary = f"\n{'─' * 90}\n"
            summary += f"Total Agents: {len(all_agents)}  |  Active: {active_count}  |  Neutral: {len(all_agents) - active_count}\n"

            self.agent_status_text.insert(tk.END, summary, "summary")
            self.agent_status_text.tag_config("summary", foreground="#FFD700")

        except Exception as e:
            logger.error(f"Error updating agent status display: {e}")
            self.agent_status_text.delete(1.0, tk.END)
            self.agent_status_text.insert(1.0, f"Error loading agent status: {e}\n", "error")
            self.agent_status_text.tag_config("error", foreground="#FF4444")

    def _update_status_indicators(self):
        """Update the visual status indicators."""
        try:
            # Check exchange connection status
            if self.trading_engine and hasattr(self.trading_engine.exchange_client, 'is_connected'):
                is_connected = self.trading_engine.exchange_client.is_connected
                if is_connected:
                    self.exchange_status_indicator.config(text="● CONNECTED", fg="#00FF00")
                else:
                    self.exchange_status_indicator.config(text="● DISCONNECTED", fg="#FF4444")
            else:
                self.exchange_status_indicator.config(text="● UNKNOWN", fg="#888888")

            # Check engine status
            if self.trading_engine and hasattr(self.trading_engine, '_is_running'):
                is_running = self.trading_engine._is_running
                if is_running:
                    self.engine_status_indicator.config(text="● RUNNING", fg="#00FF00")
                else:
                    self.engine_status_indicator.config(text="● STOPPED", fg="#888888")
            else:
                self.engine_status_indicator.config(text="● STOPPED", fg="#888888")

            # Check trading status
            trading_enabled = self.config_manager.get("trading.enabled", False)
            if trading_enabled:
                self.trading_status_indicator.config(text="● ENABLED", fg="#00FF00")
            else:
                self.trading_status_indicator.config(text="● DISABLED", fg="#888888")

        except Exception as e:
            logger.error(f"Error updating status indicators: {e}")

    def _start_trading(self):
        """Start trading."""
        try:
            if messagebox.askyesno("Confirm", "Start trading with current configuration?"):
                self.config_manager.set("trading.enabled", True)
                self.config_manager.save()

                # Update indicators immediately
                self._update_status_indicators()

                # Actually start the trading engine if not already running
                if self.trading_engine and not self.trading_engine._is_running:
                    self._log_message("🚀 Starting trading engine...", "signal")
                    # Start the trading engine in a background task
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Run trading engine start in background thread
                    def start_engine():
                        loop.run_until_complete(self.trading_engine.start())

                    import threading
                    engine_thread = threading.Thread(target=start_engine, daemon=True)
                    engine_thread.start()

                    self._log_message("✅ Trading engine started - monitoring market", "signal")

                    # Schedule status indicator update after engine starts (give it time to connect)
                    self.root.after(2000, self._update_status_indicators)
                else:
                    self._log_message("✅ Trading enabled", "signal")

                self._update_status()
                messagebox.showinfo("Success", "Trading started successfully!\n\nAgents are now analyzing market data every 5 seconds.")
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")

    def _pause_trading(self):
        """Pause trading (keeps positions and orders active)."""
        try:
            self.config_manager.set("trading.enabled", False)
            self.config_manager.save()
            self._log_message("⏸️  Trading paused (positions and orders still active)", "signal")
            self._update_status()
            self._update_status_indicators()
            messagebox.showinfo(
                "Trading Paused",
                "Trading has been paused.\n\n"
                "✅ No new trades will be placed\n"
                "⚠️  Open positions remain active\n"
                "⚠️  Pending orders remain active\n\n"
                "Use 'Stop Trading' for full shutdown."
            )
        except Exception as e:
            logger.error(f"Error pausing trading: {e}")
            messagebox.showerror("Error", f"Failed to pause trading: {e}")

    def _stop_trading(self):
        """Stop trading completely (cancel orders, offer to close positions)."""
        try:
            # First confirmation
            response = messagebox.askyesnocancel(
                "Stop Trading",
                "⚠️  STOP TRADING will:\n\n"
                "✅ Disable trading (no new trades)\n"
                "✅ Cancel all pending orders\n"
                "❓ Close all open positions?\n\n"
                "Would you like to close all positions?\n\n"
                "• YES: Cancel orders AND close positions (FULL STOP)\n"
                "• NO: Cancel orders, keep positions open\n"
                "• CANCEL: Don't stop trading",
                icon='warning'
            )

            if response is None:  # Cancel
                return

            # Disable trading
            self.config_manager.set("trading.enabled", False)
            self.config_manager.save()
            self._log_message("🛑 Stopping trading...", "signal")

            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Cancel all pending orders
            self._log_message("Cancelling pending orders...", "signal")
            try:
                loop.run_until_complete(self.trading_engine._cancel_all_orders())
                self._log_message("✅ All pending orders cancelled", "order")
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")
                self._log_message(f"⚠️  Error cancelling orders: {e}", "error")

            # Close positions if user selected YES
            if response:  # YES - close positions
                self._log_message("Closing all positions...", "signal")
                try:
                    result = loop.run_until_complete(self.trading_engine.close_all_positions())

                    if result['closed_count'] > 0:
                        self._log_message(f"✅ Closed {result['closed_count']} positions", "order")
                        for detail in result['details']:
                            self._log_message(f"  {detail}", "order")

                    if result['failed_count'] > 0:
                        self._log_message(f"⚠️  Failed to close {result['failed_count']} positions", "error")
                        # Show detailed error messages for each failed position
                        failed_details = [d for d in result['details'] if d.startswith("Failed")]
                        for detail in failed_details:
                            self._log_message(f"  {detail}", "error")

                except Exception as e:
                    logger.error(f"Error closing positions: {e}")
                    self._log_message(f"❌ Error closing positions: {e}", "error")

            loop.close()

            # Final message
            self._log_message("🛑 Trading stopped", "signal")
            self._update_status()

            if response:
                messagebox.showinfo(
                    "Trading Stopped",
                    "✅ Trading has been fully stopped\n"
                    "✅ Pending orders cancelled\n"
                    "✅ Positions closed"
                )
            else:
                messagebox.showinfo(
                    "Trading Stopped",
                    "✅ Trading has been stopped\n"
                    "✅ Pending orders cancelled\n"
                    "⚠️  Positions remain open\n\n"
                    "Manage positions manually or use\n"
                    "'Close All Positions' button"
                )

        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            messagebox.showerror("Error", f"Failed to stop trading: {e}")

    def _close_all_positions(self):
        """Close all open positions."""
        try:
            # Confirm action
            response = messagebox.askyesno(
                "Close All Positions?",
                "⚠️  This will close ALL open positions with market orders.\n\n"
                "Are you sure you want to continue?",
                icon='warning'
            )

            if not response:
                return

            self._log_message("Closing all positions...", "signal")

            # Run async operation
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.trading_engine.close_all_positions())
            loop.close()

            # Display results
            if result['success']:
                message = f"✅ Successfully closed {result['closed_count']} positions"
                self._log_message(message, "order")
                for detail in result['details']:
                    self._log_message(f"  {detail}", "order")

                messagebox.showinfo("Success", message)
            else:
                message = f"⚠️  Closed {result['closed_count']}, Failed {result['failed_count']}"
                self._log_message(message, "error")
                for detail in result['details']:
                    self._log_message(f"  {detail}", "info")

                messagebox.showwarning("Partial Success", message + "\n\nCheck logs for details.")

            # Update status
            self._update_status()

        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            self._log_message(f"❌ Error closing positions: {e}", "error")
            messagebox.showerror("Error", f"Failed to close positions: {e}")

    def _update_agent_controls(self, available_agents, active_agent):
        """Update agent dropdown and checkboxes with current agents."""
        # Update dropdown
        self.agent_dropdown['values'] = available_agents
        if active_agent and active_agent in available_agents:
            self.agent_var.set(active_agent)
        elif available_agents:
            self.agent_var.set(available_agents[0])

        # Update or create checkboxes for multi-agent mode
        # Clear existing checkboxes
        for widget in self.agent_checkboxes_frame.winfo_children():
            widget.destroy()
        self.agent_check_vars.clear()

        # Get current active agents from agent manager
        active_agents = []
        if self.agent_manager.is_multi_agent_mode():
            active_agents_objs = self.agent_manager.get_active_agents()
            active_agents = [agent.get_name() for agent in active_agents_objs]

        # Create checkbox for each available agent
        for agent_name in available_agents:
            var = tk.BooleanVar(value=agent_name in active_agents)
            self.agent_check_vars[agent_name] = var

            checkbox = ttk.Checkbutton(
                self.agent_checkboxes_frame,
                text=agent_name,
                variable=var
            )
            checkbox.pack(anchor=tk.W, pady=2)

    def _on_mode_changed(self):
        """Handle switching between single and multi-agent mode."""
        mode = self.agent_mode_var.get()

        if mode == "single":
            # Show single agent frame, hide multi agent frame
            self.multi_agent_frame.pack_forget()
            self.single_agent_frame.pack(fill=tk.X)

            # Disable multi-agent mode in agent manager
            if self.agent_manager.is_multi_agent_mode():
                self.agent_manager.disable_multi_agent_mode()

                # Set back to single agent
                if self.agent_var.get():
                    self.agent_manager.set_active_agent(self.agent_var.get())

                self._log_message("Switched to single-agent mode", "info")
                logger.info("GUI switched to single-agent mode")

        else:  # multi mode
            # Show multi agent frame, hide single agent frame
            self.single_agent_frame.pack_forget()
            self.multi_agent_frame.pack(fill=tk.X)

            self._log_message("Switched to multi-agent mode - select agents and click Apply", "signal")
            logger.info("GUI switched to multi-agent mode")

        # Auto-resize window to fit content
        self.root.update_idletasks()

        # Get the required size for all widgets
        self.root.geometry("")  # Reset to auto-size based on content

    def _apply_multi_agent_selection(self):
        """Apply the selected agents in multi-agent mode."""
        try:
            # Get selected agents
            selected_agents = [name for name, var in self.agent_check_vars.items() if var.get()]

            if not selected_agents:
                messagebox.showwarning(
                    "No Agents Selected",
                    "Please select at least one agent to activate."
                )
                return

            # Warn if trading is active
            if self.config_manager.get("trading.enabled", False):
                response = messagebox.askyesno(
                    "Change Active Agents?",
                    f"⚠️  You are about to change active agents while trading is running.\n\n"
                    f"Selected agents: {', '.join(selected_agents)}\n\n"
                    "This may affect open positions. Continue?",
                    icon='warning'
                )

                if not response:
                    return

            # Set active agents
            self.agent_manager.set_active_agents(selected_agents)

            # Log the change
            self._log_message(
                f"✅ Active agents set: {', '.join(selected_agents)}",
                "signal"
            )

            logger.info(f"Multi-agent mode activated with: {selected_agents}")

            # Update status display
            self._update_status()

            messagebox.showinfo(
                "Agents Activated",
                f"Successfully activated {len(selected_agents)} agents:\n\n" +
                "\n".join(f"• {name}" for name in selected_agents)
            )

        except Exception as e:
            logger.error(f"Error applying multi-agent selection: {e}")
            messagebox.showerror("Error", f"Failed to set active agents: {e}")

    def _on_agent_selected(self, event):
        """Handle agent selection from dropdown (single-agent mode)."""
        try:
            selected_agent = self.agent_var.get()
            if not selected_agent:
                return

            # Get current active agent
            current_agent = self.agent_manager.get_active_agent()
            current_agent_name = current_agent.get_name() if current_agent else None

            # Check if actually switching (not just re-selecting same agent)
            if current_agent_name == selected_agent:
                return

            # Warn if trading is active
            if self.config_manager.get("trading.enabled", False):
                # First, ask if they want to close positions
                close_positions_response = messagebox.askyesnocancel(
                    "Switch Trading Agent",
                    f"You are switching from '{current_agent_name}' to '{selected_agent}'\n\n"
                    "⚠️  WARNING: The new agent won't know about positions opened by the old agent.\n\n"
                    "Would you like to close all open positions before switching?\n\n"
                    "• YES: Close positions, then switch agent (RECOMMENDED)\n"
                    "• NO: Keep positions open and switch anyway (RISKY)\n"
                    "• CANCEL: Don't switch agents",
                    icon='warning'
                )

                if close_positions_response is None:  # Cancel
                    # User clicked Cancel - revert to previous agent
                    if current_agent_name:
                        self.agent_var.set(current_agent_name)
                    return

                if close_positions_response:  # Yes - close positions first
                    self._log_message(f"Closing positions before switching agents...", "signal")
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(self.trading_engine.close_all_positions())
                        loop.close()

                        # Display results
                        if result['closed_count'] > 0:
                            self._log_message(f"✅ Closed {result['closed_count']} positions", "order")
                            for detail in result['details']:
                                self._log_message(f"  {detail}", "order")
                        else:
                            self._log_message("No open positions to close", "info")

                        if result['failed_count'] > 0:
                            self._log_message(f"⚠️  Failed to close {result['failed_count']} positions", "error")
                            # Show detailed error messages for each failed position
                            failed_details = [d for d in result['details'] if d.startswith("Failed")]
                            for detail in failed_details:
                                self._log_message(f"  {detail}", "error")

                            messagebox.showwarning(
                                "Partial Success",
                                f"⚠️  Failed to close {result['failed_count']} positions.\n\n"
                                "Continue with agent switch?"
                            )

                    except Exception as e:
                        logger.error(f"Error closing positions: {e}")
                        self._log_message(f"❌ Error: {e}", "error")
                        if not messagebox.askyesno(
                            "Error Closing Positions",
                            f"Failed to close positions: {e}\n\n"
                            "Continue with agent switch anyway?",
                            icon='error'
                        ):
                            if current_agent_name:
                                self.agent_var.set(current_agent_name)
                            return
                else:  # No - keep positions open
                    # Confirm they really want to do this
                    confirm = messagebox.askyesno(
                        "Confirm Risky Switch",
                        "⚠️  Are you sure you want to switch agents WITHOUT closing positions?\n\n"
                        "This may result in:\n"
                        "• Conflicting trading signals\n"
                        "• Unexpected position sizes\n"
                        "• Risk management issues",
                        icon='warning'
                    )

                    if not confirm:
                        # User changed their mind - revert
                        if current_agent_name:
                            self.agent_var.set(current_agent_name)
                        return

            # Set the active agent
            self.agent_manager.set_active_agent(selected_agent)

            # Update the display
            self._update_status()

            # Log the change
            self._log_message(f"🔄 Active agent changed: {current_agent_name} → {selected_agent}", "signal")

            logger.info(f"User switched agent from '{current_agent_name}' to '{selected_agent}'")
        except Exception as e:
            logger.error(f"Error selecting agent: {e}")
            messagebox.showerror("Error", f"Failed to select agent: {e}")

    def _log_message(self, message: str, message_type: str = "info"):
        """Add message to log display with optional color coding."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # Color code based on message type
        color_map = {
            "signal": "#00BFFF",    # Deep sky blue for signals
            "order": "#00FF00",     # Green for orders
            "cancelled": "#FFA500", # Orange for cancellations
            "error": "#FF4444",     # Red for errors
            "info": "#FFFFFF"       # White for info
        }

        color = color_map.get(message_type, "#FFFFFF")

        # Insert with color
        self.log_text.insert(tk.END, log_entry, message_type)
        self.log_text.tag_config(message_type, foreground=color)
        self.log_text.see(tk.END)

    def _on_exit(self):
        """Handle application exit."""
        # First confirmation
        if not messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            return

        # Check if there are open positions (when connected to exchange)
        # For now, we'll add a warning about positions
        if self.config_manager.get("trading.enabled", False):
            response = messagebox.askyesnocancel(
                "Open Positions Warning",
                "⚠️  WARNING: Any open positions will remain on the exchange!\n\n"
                "• Click YES to exit and KEEP positions open\n"
                "• Click NO to cancel exit and manage positions manually\n"
                "• Click CANCEL to stay in the application\n\n"
                "Note: Pending orders will be automatically cancelled."
            )

            if response is None:  # Cancel clicked
                return
            elif response is False:  # No clicked
                messagebox.showinfo(
                    "Exit Cancelled",
                    "Please manually close your positions via the exchange website:\n"
                    f"https://{'demo-trading.' if self.config_manager.get('exchange.sandbox', True) else ''}blofin.com"
                )
                return

        try:
            self._log_message("Shutting down...")

            # Disable trading first
            if self.config_manager.get("trading.enabled", False):
                self.config_manager.set("trading.enabled", False)
                self.config_manager.save()
                self._log_message("Trading disabled")

            # Stop trading engine (cancels pending orders, disconnects from exchange)
            if self.trading_engine:
                self._log_message("Cancelling pending orders...")
                import asyncio
                try:
                    # Run async stop in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.trading_engine.stop())
                    loop.close()
                    self._log_message("✅ Pending orders cancelled")
                    self._log_message("✅ Disconnected from exchange")
                except Exception as e:
                    logger.error(f"Error stopping trading engine: {e}")
                    self._log_message(f"⚠️  Warning: {e}", "error")

            self._log_message("Shutdown complete")
            self.is_running = False
            self.root.quit()
            self.root.destroy()
            logger.info("GUI shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _periodic_positions_update(self):
        """Periodically update positions and agent status displays."""
        if self.is_running:
            try:
                self._update_positions_display()
                self._update_agent_status_display()
                self._update_status_indicators()
            except Exception as e:
                logger.error(f"Error in periodic positions update: {e}")

            # Schedule next update
            self.root.after(self.positions_update_interval, self._periodic_positions_update)

    def run(self):
        """Run the GUI main loop."""
        try:
            self.is_running = True
            self._log_message("GUI ready - configure API keys in config/trading_config.json to enable trading")
            logger.info("Starting GUI main loop")

            # Start periodic positions update
            self.root.after(1000, self._periodic_positions_update)  # Start after 1 second

            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in GUI main loop: {e}")
            raise
        finally:
            self.is_running = False
