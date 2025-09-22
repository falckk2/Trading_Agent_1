"""
Data management module for handling historical and real-time market data.
"""

from .data_manager import DataManager
from .historical.data_provider import HistoricalDataProvider
from .realtime.realtime_feed import RealtimeFeed
from .preprocessing.data_preprocessor import DataPreprocessor
from .storage.data_storage import DataStorage
from .persistence import TradingDataPersistence, create_persistence_layer

__all__ = [
    'DataManager',
    'HistoricalDataProvider',
    'RealtimeFeed',
    'DataPreprocessor',
    'DataStorage',
    'TradingDataPersistence',
    'create_persistence_layer'
]