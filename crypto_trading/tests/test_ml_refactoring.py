"""
Unit tests for the ML architecture refactoring.
Tests the RandomForestStrategy and RandomForestAgent composition pattern.
"""

import pytest
import numpy as np
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from decimal import Decimal

# Mock all problematic dependencies
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Mock the RandomForestClassifier
mock_rf = MagicMock()
mock_rf.return_value.fit = MagicMock()
mock_rf.return_value.predict = MagicMock(return_value=[1])
mock_rf.return_value.predict_proba = MagicMock(return_value=[[0.3, 0.7]])
mock_rf.return_value.feature_importances_ = [0.5, 0.3, 0.2]
sys.modules['sklearn.ensemble'].RandomForestClassifier = mock_rf

# Mock other sklearn components
sys.modules['sklearn.model_selection'].train_test_split = MagicMock(return_value=([], [], [], []))
sys.modules['sklearn.preprocessing'].StandardScaler = MagicMock()
sys.modules['sklearn.metrics'].accuracy_score = MagicMock(return_value=0.85)

from crypto_trading.core.interfaces import MarketData
from crypto_trading.core.models import TradingSignal, SignalType
from crypto_trading.agents.ml.ml_strategy import MLStrategy
from crypto_trading.agents.ml.random_forest_strategy import RandomForestStrategy
from crypto_trading.agents.ml.random_forest_agent import RandomForestAgent


class TestMLStrategyRefactoring:
    """Test suite for ML strategy refactoring."""

    def test_random_forest_strategy_inheritance(self):
        """Test that RandomForestStrategy properly inherits from MLStrategy."""
        strategy = RandomForestStrategy()

        # Test inheritance
        assert isinstance(strategy, MLStrategy)
        assert isinstance(strategy, object)

        # Test that it's not the old BaseMLAgent
        assert strategy.__class__.__name__ == "RandomForestStrategy"

    def test_random_forest_strategy_instantiation(self):
        """Test RandomForestStrategy can be instantiated with parameters."""
        params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }

        strategy = RandomForestStrategy(params)

        assert strategy.parameters['n_estimators'] == 50
        assert strategy.parameters['max_depth'] == 5
        assert strategy.parameters['random_state'] == 42
        assert strategy.name == "RandomForestStrategy"
        assert not strategy.is_trained

    def test_random_forest_strategy_implements_abstract_methods(self):
        """Test that RandomForestStrategy implements all required abstract methods."""
        strategy = RandomForestStrategy()

        # Test abstract methods are implemented
        assert hasattr(strategy, '_create_model')
        assert hasattr(strategy, '_train_model_impl')
        assert hasattr(strategy, '_predict')
        assert hasattr(strategy, '_evaluate_model')

        # Test methods are callable
        assert callable(strategy._create_model)
        assert callable(strategy._train_model_impl)
        assert callable(strategy._predict)
        assert callable(strategy._evaluate_model)

    def test_random_forest_strategy_model_creation(self):
        """Test model creation in RandomForestStrategy."""
        strategy = RandomForestStrategy({'n_estimators': 10})

        model = strategy._create_model()

        # Should create a RandomForestClassifier mock
        assert model is not None
        mock_rf.assert_called_with(
            n_estimators=10,
            max_depth=strategy.parameters.get('max_depth', 10),
            min_samples_split=strategy.parameters.get('min_samples_split', 5),
            min_samples_leaf=strategy.parameters.get('min_samples_leaf', 2),
            random_state=strategy.parameters.get('random_state', 42),
            class_weight=strategy.parameters.get('class_weight', 'balanced'),
            n_jobs=-1
        )

    def test_random_forest_agent_composition(self):
        """Test that RandomForestAgent uses composition correctly."""
        agent = RandomForestAgent({'n_estimators': 25})

        # Test composition
        assert hasattr(agent, 'strategy')
        assert isinstance(agent.strategy, RandomForestStrategy)
        assert agent.strategy.parameters['n_estimators'] == 25

        # Test agent properties
        assert agent.name == "Random Forest Agent"

    def test_random_forest_agent_delegation(self):
        """Test that RandomForestAgent properly delegates to strategy."""
        agent = RandomForestAgent()

        # Test delegation methods exist
        delegation_methods = [
            'is_trained', 'save_model', 'load_model',
            'needs_retraining', 'get_feature_importance', 'get_model_info'
        ]

        for method in delegation_methods:
            assert hasattr(agent, method)
            assert callable(getattr(agent, method))

        # Test delegation works
        assert agent.is_trained() == agent.strategy.is_trained
        assert agent.needs_retraining() == agent.strategy.needs_retraining()

    def test_no_baseml_agent_dependency(self):
        """Test that BaseMLAgent is not used anywhere."""
        # Test RandomForestAgent doesn't inherit from BaseMLAgent
        from crypto_trading.agents.base_agent import BaseAgent

        agent = RandomForestAgent()
        assert isinstance(agent, BaseAgent)

        # Ensure BaseMLAgent is not in the inheritance chain
        mro = [cls.__name__ for cls in agent.__class__.__mro__]
        assert 'BaseMLAgent' not in mro

    @patch('crypto_trading.agents.ml.random_forest_strategy.np')
    def test_prediction_delegation(self, mock_np):
        """Test that agent analyze delegates to strategy."""
        # Setup mocks
        mock_np.ndarray = list
        mock_np.array = lambda x: x

        # Create test data
        market_data = [
            MarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("1000")
            )
        ]

        agent = RandomForestAgent()

        # Mock the strategy's analyze method
        mock_signal = TradingSignal(
            symbol="BTC-USD",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=50500,
            strategy_name="RandomForestStrategy",
            confidence=0.7
        )

        agent.strategy.analyze = Mock(return_value=mock_signal)

        # Test delegation
        result = agent.analyze(market_data)

        # Verify strategy analyze was called
        agent.strategy.analyze.assert_called_once_with(market_data)

        # Verify result includes agent metadata
        assert "agent_name" in result.metadata
        assert result.metadata["agent_name"] == agent.name

    def test_parameter_management(self):
        """Test parameter management in the new architecture."""
        strategy_params = {'n_estimators': 75, 'max_depth': 8}
        agent = RandomForestAgent(strategy_params)

        # Test strategy parameters are set
        assert agent.strategy.parameters['n_estimators'] == 75
        assert agent.strategy.parameters['max_depth'] == 8

        # Test agent can get default parameters
        defaults = agent.get_default_parameters()
        assert 'strategy_parameters' in defaults

        # Test parameter updates
        agent.initialize({'strategy_parameters': {'n_estimators': 100}})
        assert agent.strategy.parameters['n_estimators'] == 100

    def test_training_delegation(self):
        """Test that training operations are properly delegated."""
        agent = RandomForestAgent()

        # Mock the strategy's train_model method
        mock_metrics = {
            'training_score': 0.85,
            'validation_score': 0.82,
            'test_score': 0.80
        }

        # Use async mock for train_model
        async def mock_train():
            return mock_metrics

        agent.strategy.train_model = Mock(return_value=mock_train())

        # Test training delegation exists
        assert hasattr(agent, 'train_model')
        assert callable(agent.train_model)

    def test_interface_compliance(self):
        """Test that both strategy and agent comply with their interfaces."""
        strategy = RandomForestStrategy()
        agent = RandomForestAgent()

        # Test strategy interface
        strategy_methods = ['analyze', 'get_parameters', 'set_parameters', 'validate_signal']
        for method in strategy_methods:
            assert hasattr(strategy, method), f"Strategy missing {method}"

        # Test agent interface
        agent_methods = ['initialize', 'analyze', 'get_name', 'get_description']
        for method in agent_methods:
            assert hasattr(agent, method), f"Agent missing {method}"

        # Test method return types
        assert isinstance(agent.get_name(), str)
        assert isinstance(agent.get_description(), str)
        assert isinstance(strategy.get_parameters(), dict)

    def test_architecture_separation(self):
        """Test that ML logic is properly separated between strategy and agent."""
        agent = RandomForestAgent()
        strategy = RandomForestStrategy()

        # Agent should not have ML implementation methods
        ml_methods = ['_create_model', '_train_model_impl', '_predict', '_evaluate_model']
        for method in ml_methods:
            assert not hasattr(agent, method), f"Agent should not have {method}"
            assert hasattr(strategy, method), f"Strategy should have {method}"

        # Agent should have delegation and orchestration methods
        agent_methods = ['train_model', 'is_trained', 'save_model', 'load_model']
        for method in agent_methods:
            assert hasattr(agent, method), f"Agent should have delegation method {method}"


class TestMLRefactoringIntegration:
    """Integration tests for the refactored ML architecture."""

    def test_strategy_agent_integration(self):
        """Test that strategy and agent work together correctly."""
        # Create agent with specific parameters
        agent = RandomForestAgent({'n_estimators': 30})

        # Test that agent uses strategy correctly
        assert agent.strategy.parameters['n_estimators'] == 30

        # Test that agent delegates to strategy
        assert agent.is_trained() == agent.strategy.is_trained

        # Test that strategy is independent
        strategy2 = RandomForestStrategy({'n_estimators': 50})
        assert strategy2.parameters['n_estimators'] == 50
        assert strategy2 is not agent.strategy

    def test_multiple_agents_different_strategies(self):
        """Test that multiple agents can have different strategy configurations."""
        agent1 = RandomForestAgent({'n_estimators': 10})
        agent2 = RandomForestAgent({'n_estimators': 20})

        # Test they have different strategies
        assert agent1.strategy is not agent2.strategy
        assert agent1.strategy.parameters['n_estimators'] == 10
        assert agent2.strategy.parameters['n_estimators'] == 20

        # Test they have the same agent interface
        assert agent1.get_name() == agent2.get_name()
        assert type(agent1.strategy) == type(agent2.strategy)