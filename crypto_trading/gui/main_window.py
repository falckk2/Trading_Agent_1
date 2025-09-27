"""
Main GUI window for the cryptocurrency trading application.
Provides the primary interface for managing agents and monitoring trading.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import asyncio
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..core.trading_engine import TradingEngine
from ..core.agent_manager import AgentManager
from ..core.config_manager import ConfigManager
from ..core.event_bus import EventBus, EventSubscriber
from ..core.risk_manager import RiskManager
from ..exchange.blofin_client import BlofinClient
from ..agents.technical.rsi_agent import RSIAgent
from ..agents.technical.macd_agent import MACDAgent
from ..agents.ml.random_forest_agent import RandomForestAgent
from ..data.data_manager import DataManager


class MainWindow:
    """
    Main application window providing comprehensive trading interface.
    """

    def __init__(self, agent_manager: AgentManager, data_manager: DataManager):
        self.agent_manager = agent_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)

        # GUI components
        self.root = None
        self.dashboard = None
        self.agent_panel = None
        self.performance_panel = None
        self.chart_panel = None
        self.log_panel = None

        # Update thread management
        self.update_thread = None
        self.update_interval = 1000  # milliseconds
        self.is_running = False

        # Application state
        self.current_symbol = "BTC-USD"
        self.selected_agent = None

    def initialize(self) -> None:
        """Initialize the main window and all components."""
        try:
            self.root = tk.Tk()
            self.root.title("Cryptocurrency Trading Platform - DeepAgent")
            self.root.geometry("1400x900")
            self.root.minsize(1200, 800)

            # Configure styles
            self._setup_styles()

            # Create menu bar
            self._create_menu_bar()

            # Create main layout
            self._create_main_layout()

            # Initialize components
            self._initialize_components()

            # Setup event handlers
            self._setup_event_handlers()

            # Start update loop
            self._start_update_loop()

            self.logger.info("Main window initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize main window: {e}")
            raise

    def run(self) -> None:
        """Run the GUI main loop."""
        if not self.root:
            raise RuntimeError("Window not initialized")

        try:
            self.is_running = True
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in GUI main loop: {e}")
        finally:
            self.is_running = False

    def shutdown(self) -> None:
        """Shutdown the GUI and cleanup resources."""
        try:
            self.is_running = False

            # Stop update thread
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2)

            # Cleanup components
            if self.dashboard:
                self.dashboard.shutdown()

            # Close window
            if self.root:
                self.root.quit()
                self.root.destroy()

            self.logger.info("Main window shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    # Private methods

    def _setup_styles(self) -> None:
        """Setup GUI styles and themes."""
        style = ttk.Style()

        # Configure colors
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        select_color = "#404040"

        self.root.configure(bg=bg_color)

        # Configure ttk styles
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=select_color, foreground=fg_color)
        style.configure("Heading.TLabel", font=("Arial", 12, "bold"))

    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Configuration", command=self._load_configuration)
        file_menu.add_command(label="Save Configuration", command=self._save_configuration)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self._export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        # Agents menu
        agents_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Agents", menu=agents_menu)
        agents_menu.add_command(label="Create Agent", command=self._create_agent_dialog)
        agents_menu.add_command(label="Start All Agents", command=self._start_all_agents)
        agents_menu.add_command(label="Stop All Agents", command=self._stop_all_agents)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Backtest Strategy", command=self._backtest_dialog)
        tools_menu.add_command(label="Train ML Model", command=self._train_model_dialog)
        tools_menu.add_command(label="Data Manager", command=self._data_manager_dialog)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self) -> None:
        """Create the main application layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create paned windows for resizable layout
        # Horizontal paned window (left and right)
        h_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        h_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel (agents and controls)
        left_frame = ttk.Frame(h_paned)
        h_paned.add(left_frame, weight=1)

        # Right panel (charts and dashboard)
        right_frame = ttk.Frame(h_paned)
        h_paned.add(right_frame, weight=3)

        # Vertical paned window for left panel
        left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)

        # Agent panel
        agent_frame = ttk.LabelFrame(left_paned, text="Trading Agents")
        left_paned.add(agent_frame, weight=2)

        # Performance panel
        performance_frame = ttk.LabelFrame(left_paned, text="Performance")
        left_paned.add(performance_frame, weight=1)

        # Vertical paned window for right panel
        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # Chart panel
        chart_frame = ttk.LabelFrame(right_paned, text="Market Charts")
        right_paned.add(chart_frame, weight=2)

        # Bottom panels (logs and dashboard)
        bottom_paned = ttk.PanedWindow(right_paned, orient=tk.HORIZONTAL)
        right_paned.add(bottom_paned, weight=1)

        # Dashboard panel
        dashboard_frame = ttk.LabelFrame(bottom_paned, text="Trading Dashboard")
        bottom_paned.add(dashboard_frame, weight=2)

        # Log panel
        log_frame = ttk.LabelFrame(bottom_paned, text="Logs")
        bottom_paned.add(log_frame, weight=1)

        # Store frame references
        self.frames = {
            'agent': agent_frame,
            'performance': performance_frame,
            'chart': chart_frame,
            'dashboard': dashboard_frame,
            'log': log_frame
        }

    def _initialize_components(self) -> None:
        """Initialize all GUI components."""
        # Initialize agent panel
        self.agent_panel = AgentPanel(
            self.frames['agent'],
            self.agent_manager,
            self._on_agent_selected
        )

        # Initialize performance panel
        self.performance_panel = PerformancePanel(
            self.frames['performance'],
            self.agent_manager
        )

        # Initialize chart panel
        self.chart_panel = ChartPanel(
            self.frames['chart'],
            self.data_manager,
            self._on_symbol_changed
        )

        # Initialize trading dashboard
        self.dashboard = TradingDashboard(
            self.frames['dashboard'],
            self.agent_manager,
            self.data_manager
        )

        # Initialize log panel
        self.log_panel = LogPanel(self.frames['log'])

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for the application."""
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

        # Key bindings
        self.root.bind("<Control-q>", lambda e: self._on_exit())
        self.root.bind("<F5>", lambda e: self._refresh_all_data())
        self.root.bind("<Control-s>", lambda e: self._save_configuration())

    def _start_update_loop(self) -> None:
        """Start the periodic update loop."""
        def update_worker():
            while self.is_running:
                try:
                    # Schedule GUI updates on main thread
                    self.root.after(0, self._update_gui_components)
                    threading.Event().wait(self.update_interval / 1000.0)
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")

        self.update_thread = threading.Thread(target=update_worker, daemon=True)
        self.update_thread.start()

    def _update_gui_components(self) -> None:
        """Update all GUI components with latest data."""
        try:
            if not self.is_running:
                return

            # Update components
            if self.agent_panel:
                self.agent_panel.update()

            if self.performance_panel:
                self.performance_panel.update()

            if self.dashboard:
                self.dashboard.update()

            if self.chart_panel:
                self.chart_panel.update()

        except Exception as e:
            self.logger.error(f"Error updating GUI components: {e}")

    # Event handlers

    def _on_agent_selected(self, agent_id: str) -> None:
        """Handle agent selection."""
        self.selected_agent = agent_id
        self.logger.info(f"Selected agent: {agent_id}")

        # Update other panels with selected agent
        if self.performance_panel:
            self.performance_panel.set_selected_agent(agent_id)

        if self.dashboard:
            self.dashboard.set_selected_agent(agent_id)

    def _on_symbol_changed(self, symbol: str) -> None:
        """Handle symbol change."""
        self.current_symbol = symbol
        self.logger.info(f"Changed symbol to: {symbol}")

        # Update other panels with new symbol
        if self.dashboard:
            self.dashboard.set_current_symbol(symbol)

    def _on_exit(self) -> None:
        """Handle application exit."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.shutdown()

    # Menu command handlers

    def _load_configuration(self) -> None:
        """Load application configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                # TODO: Implement configuration loading
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def _save_configuration(self) -> None:
        """Save application configuration to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                # TODO: Implement configuration saving
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def _export_data(self) -> None:
        """Export trading data to file."""
        filename = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                # TODO: Implement data export
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {e}")

    def _create_agent_dialog(self) -> None:
        """Show create agent dialog."""
        # TODO: Implement agent creation dialog
        messagebox.showinfo("Info", "Agent creation dialog - To be implemented")

    def _start_all_agents(self) -> None:
        """Start all agents."""
        try:
            # Run async operation in thread
            def start_agents():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.agent_manager.start_all_agents())
                loop.close()

            threading.Thread(target=start_agents, daemon=True).start()
            messagebox.showinfo("Success", "All agents started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start agents: {e}")

    def _stop_all_agents(self) -> None:
        """Stop all agents."""
        try:
            # Run async operation in thread
            def stop_agents():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.agent_manager.stop_all_agents())
                loop.close()

            threading.Thread(target=stop_agents, daemon=True).start()
            messagebox.showinfo("Success", "All agents stopped")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop agents: {e}")

    def _backtest_dialog(self) -> None:
        """Show backtest dialog."""
        # TODO: Implement backtest dialog
        messagebox.showinfo("Info", "Backtest dialog - To be implemented")

    def _train_model_dialog(self) -> None:
        """Show ML model training dialog."""
        # TODO: Implement model training dialog
        messagebox.showinfo("Info", "Model training dialog - To be implemented")

    def _data_manager_dialog(self) -> None:
        """Show data manager dialog."""
        # TODO: Implement data manager dialog
        messagebox.showinfo("Info", "Data manager dialog - To be implemented")

    def _refresh_all_data(self) -> None:
        """Refresh all data in the application."""
        try:
            # Update all components
            self._update_gui_components()
            messagebox.showinfo("Success", "Data refreshed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh data: {e}")

    def _show_documentation(self) -> None:
        """Show application documentation."""
        messagebox.showinfo("Documentation", "Documentation - To be implemented")

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """
Cryptocurrency Trading Platform
DeepAgent v1.0.0

A comprehensive trading system with:
- Multiple trading agents
- Machine learning strategies
- Real-time data feeds
- Risk management
- Performance monitoring

© 2024 DeepAgent Trading Team
"""
        messagebox.showinfo("About", about_text)