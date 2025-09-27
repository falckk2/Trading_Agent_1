"""
Machine Learning agents module.
"""

from .ml_strategy import MLStrategy
from .random_forest_agent import RandomForestAgent
from .random_forest_strategy import RandomForestStrategy
from .ml_model_trainer import MLModelTrainer

# Optional import for LSTM agent (requires tensorflow)
try:
    from .lstm_agent import LSTMAgent
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False
    LSTMAgent = None

__all__ = [
    'MLStrategy',
    'RandomForestAgent',
    'RandomForestStrategy',
    'MLModelTrainer'
]

if _LSTM_AVAILABLE:
    __all__.append('LSTMAgent')