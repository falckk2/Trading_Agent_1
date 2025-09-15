"""
Modular Cryptocurrency Trading System

A comprehensive trading platform supporting multiple trading agents,
technical analysis, machine learning, and real-time trading on Blofin exchange.
"""

__version__ = "1.0.0"
__author__ = "Crypto Trading System"

from .core import AgentManager, TradingEngine
from .exchange.blofin_client import BlofinClient

__all__ = ["AgentManager", "TradingEngine", "BlofinClient"]