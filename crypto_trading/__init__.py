"""
Modular Cryptocurrency Trading System

A comprehensive trading platform supporting multiple trading agents,
technical analysis, machine learning, and real-time trading on Blofin exchange.
"""

__version__ = "1.0.0"
__author__ = "Crypto Trading System"

from .core import AgentManager, TradingEngine
from .core.order_manager import OrderManager
from .core.connection_manager import ConnectionManager
from .core.portfolio_tracker import PortfolioTracker
from .exchange.blofin_client import BlofinClient
from .data.persistence import TradingDataPersistence, create_persistence_layer
from .security import SecureCredentialManager, HybridCredentialManager

__all__ = [
    "AgentManager",
    "TradingEngine",
    "OrderManager",
    "ConnectionManager",
    "PortfolioTracker",
    "BlofinClient",
    "TradingDataPersistence",
    "create_persistence_layer",
    "SecureCredentialManager",
    "HybridCredentialManager"
]