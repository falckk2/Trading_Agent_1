"""
Core module containing base classes and interfaces for the trading system.
"""

from .interfaces import (
    IExchangeClient,
    ITradingAgent,
    IRiskManager,
    IDataCollector,
    IDataProcessor,
    IEventBus,
    IConfigManager,
    INotificationService,
    MarketData,
    Order,
    Position,
    TradingSignal,
    Event,
    EventType,
    OrderType,
    OrderSide,
    OrderStatus
)

from .agent_manager import AgentManager
from .trading_engine import TradingEngine
from .risk_manager import RiskManager
from .event_bus import EventBus
from .config_manager import ConfigManager

__all__ = [
    # Interfaces
    'IExchangeClient',
    'ITradingAgent',
    'IRiskManager',
    'IDataCollector',
    'IDataProcessor',
    'IEventBus',
    'IConfigManager',
    'INotificationService',

    # Data Models
    'MarketData',
    'Order',
    'Position',
    'TradingSignal',
    'Event',
    'EventType',
    'OrderType',
    'OrderSide',
    'OrderStatus',

    # Core Components
    'AgentManager',
    'TradingEngine',
    'RiskManager',
    'EventBus',
    'ConfigManager'
]