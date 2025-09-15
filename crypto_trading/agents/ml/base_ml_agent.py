"""
Base machine learning agent implementation.
Provides common ML functionality for all ML-based trading agents.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
import joblib

from ...core.interfaces import MarketData, TradingSignal, OrderSide
from ..base_agent import BaseAgent
from ...data.processors.feature_engineer import FeatureEngineer
from ...utils.exceptions import AgentInitializationError


class BaseMLAgent(BaseAgent, ABC):
    """Base class for machine learning trading agents."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.model_path: Optional[Path] = None
        self.feature_columns: List[str] = []
        self.scaler = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the ML agent with configuration."""
        super().initialize(config)

        # Set model path if provided
        if "model_path" in config:
            self.model_path = Path(config["model_path"])

        # Initialize feature engineer config
        if "feature_config" in config:
            self.feature_engineer.features_config.update(config["feature_config"])

        # Load pre-trained model if available
        if self.model_path and self.model_path.exists():
            self.load_model()

    def get_required_parameters(self) -> List[str]:
        return ["prediction_threshold"]

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "prediction_threshold": 0.6,
            "lookback_periods": 20,
            "feature_config": {
                "price_features": True,
                "volume_features": True,
                "technical_indicators": True,
                "momentum_indicators": True,
                "volatility_indicators": True,
                "trend_indicators": True
            },
            "model_path": None,
            "retrain_interval": 24  # hours
        }

    def _get_minimum_data_points(self) -> int:
        return max(self._get_config_value("lookback_periods", 20) + 50, 100)

    async def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """Analyze market data using machine learning model."""
        self._ensure_initialized()
        self._validate_market_data(market_data)

        try:
            if not self.is_trained and not self._should_train_first():
                return self._create_neutral_signal(market_data)

            # Prepare features
            features = self._prepare_features(market_data)
            if features is None or len(features) == 0:
                return self._create_neutral_signal(market_data)

            # Make prediction
            prediction, confidence = self._make_prediction(features)

            # Convert prediction to trading signal
            signal = self._prediction_to_signal(prediction, confidence, market_data)

            # Add ML-specific metadata
            signal.metadata.update({
                "model_type": self._get_model_type(),
                "feature_count": len(self.feature_columns),
                "prediction_raw": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                "model_confidence": float(confidence),
                "is_trained": self.is_trained
            })

            logger.debug(f"ML Analysis ({self.name}): Prediction={prediction}, "
                        f"Confidence={confidence:.2f}, Signal={signal.action}")
            return signal

        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            return self._create_neutral_signal(market_data)

    def _prepare_features(self, market_data: List[MarketData]) -> Optional[np.ndarray]:
        """Prepare features for ML model."""
        try:
            # Calculate features using feature engineer
            features_dict = self.feature_engineer.calculate_features(market_data)
            if not features_dict:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(features_dict)

            # Handle missing values
            df = df.fillna(method='ffill').fillna(0)

            # Select and order features
            if self.feature_columns:
                # Use saved feature columns (from training)
                available_features = [col for col in self.feature_columns if col in df.columns]
                if len(available_features) != len(self.feature_columns):
                    logger.warning(f"Missing features: {set(self.feature_columns) - set(available_features)}")
                df = df[available_features]
            else:
                # Use all available features
                self.feature_columns = list(df.columns)

            # Create feature matrix with lookback
            lookback_periods = self._get_config_value("lookback_periods", 20)
            feature_matrix = self._create_feature_matrix(df, lookback_periods)

            # Apply scaling if available
            if self.scaler is not None:
                feature_matrix = self.scaler.transform(feature_matrix)

            return feature_matrix

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def _create_feature_matrix(self, df: pd.DataFrame, lookback_periods: int) -> np.ndarray:
        """Create feature matrix with lookback periods."""
        if len(df) < lookback_periods:
            # If not enough data, use what we have
            return df.values[-1:] if len(df) > 0 else np.array([])

        # Create sequences for time series
        sequences = []
        for i in range(lookback_periods, len(df)):
            sequence = df.iloc[i-lookback_periods:i].values.flatten()
            sequences.append(sequence)

        if sequences:
            return np.array(sequences)
        else:
            # Fallback to last row
            return df.iloc[-1:].values

    @abstractmethod
    def _get_model_type(self) -> str:
        """Get the type of ML model. Override in subclasses."""
        pass

    @abstractmethod
    def _make_prediction(self, features: np.ndarray) -> Tuple[Any, float]:
        """Make prediction using the ML model. Returns (prediction, confidence)."""
        pass

    @abstractmethod
    def _prediction_to_signal(self, prediction: Any, confidence: float, market_data: List[MarketData]) -> TradingSignal:
        """Convert model prediction to trading signal."""
        pass

    def _should_train_first(self) -> bool:
        """Check if model should be trained before making predictions."""
        return False  # Override in subclasses if needed

    def train(self, training_data: List[MarketData], labels: List[int]) -> None:
        """Train the ML model with provided data."""
        try:
            logger.info(f"Training {self.name} with {len(training_data)} samples")

            # Prepare features
            features = self._prepare_training_features(training_data)
            if features is None:
                raise ValueError("Failed to prepare training features")

            # Prepare labels
            target = np.array(labels)

            # Split into train/validation if needed
            train_features, train_target = self._prepare_training_data(features, target)

            # Train the model
            self._train_model(train_features, train_target)

            self.is_trained = True
            logger.info(f"Model training completed for {self.name}")

            # Save model if path is configured
            if self.model_path:
                self.save_model()

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def _prepare_training_features(self, training_data: List[MarketData]) -> Optional[np.ndarray]:
        """Prepare features for training."""
        # This is similar to _prepare_features but handles the full dataset
        features_dict = self.feature_engineer.calculate_features(training_data)
        if not features_dict:
            return None

        df = pd.DataFrame(features_dict)
        df = df.fillna(method='ffill').fillna(0)

        # Store feature columns for consistency
        self.feature_columns = list(df.columns)

        # Create feature matrix
        lookback_periods = self._get_config_value("lookback_periods", 20)
        feature_matrix = self._create_feature_matrix(df, lookback_periods)

        # Fit and apply scaling
        if self._should_use_scaling():
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            feature_matrix = self.scaler.fit_transform(feature_matrix)

        return feature_matrix

    def _should_use_scaling(self) -> bool:
        """Check if feature scaling should be applied."""
        return True  # Most ML models benefit from scaling

    def _prepare_training_data(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data (split, shuffle, etc.)."""
        # Ensure target has same length as features
        min_length = min(len(features), len(target))
        return features[:min_length], target[:min_length]

    @abstractmethod
    def _train_model(self, features: np.ndarray, target: np.ndarray) -> None:
        """Train the actual ML model. Override in subclasses."""
        pass

    def save_model(self) -> None:
        """Save the trained model to disk."""
        if not self.is_trained or self.model is None:
            logger.warning("No trained model to save")
            return

        try:
            if self.model_path:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)

                # Save model and associated data
                model_data = {
                    "model": self.model,
                    "feature_columns": self.feature_columns,
                    "scaler": self.scaler,
                    "config": self.config,
                    "is_trained": self.is_trained
                }

                joblib.dump(model_data, self.model_path)
                logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self) -> None:
        """Load a pre-trained model from disk."""
        try:
            if not self.model_path or not self.model_path.exists():
                logger.warning("No model file found to load")
                return

            model_data = joblib.load(self.model_path)

            self.model = model_data["model"]
            self.feature_columns = model_data.get("feature_columns", [])
            self.scaler = model_data.get("scaler")
            self.is_trained = model_data.get("is_trained", False)

            logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_type": self._get_model_type(),
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns.copy(),
            "model_path": str(self.model_path) if self.model_path else None,
            "has_scaler": self.scaler is not None
        }

    def _create_neutral_signal(self, market_data: List[MarketData]) -> TradingSignal:
        """Create a neutral signal when analysis fails."""
        symbol = self._get_symbol_from_data(market_data)
        current_price = float(market_data[-1].close) if market_data else None

        return self._create_signal(
            symbol=symbol,
            action=OrderSide.BUY,
            confidence=0.0,
            price=current_price,
            metadata={"signal_reason": "ml_analysis_failed"}
        )