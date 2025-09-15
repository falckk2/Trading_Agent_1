"""
Base machine learning strategy implementation.
Provides common ML functionality and interfaces.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

from ...core.interfaces import IStrategy
from ...core.models import MarketData, TradingSignal, SignalType
from ...core.exceptions import StrategyException
from ...data.preprocessing.data_preprocessor import DataPreprocessor


class MLStrategy(IStrategy, ABC):
    """
    Base class for machine learning trading strategies.
    Provides common ML functionality and enforces consistent behavior.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.name = self.__class__.__name__

        # ML parameters
        self.lookback_window = self.parameters.get("lookback_window", 60)
        self.prediction_horizon = self.parameters.get("prediction_horizon", 1)
        self.feature_columns = self.parameters.get("feature_columns", [])
        self.target_column = self.parameters.get("target_column", "target")

        # Model parameters
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_path = self.parameters.get("model_path", "")

        # Training parameters
        self.test_size = self.parameters.get("test_size", 0.2)
        self.validation_size = self.parameters.get("validation_size", 0.2)
        self.retrain_interval = self.parameters.get("retrain_interval", 24)  # hours

        # Signal generation parameters
        self.confidence_threshold = self.parameters.get("confidence_threshold", 0.6)
        self.signal_strength_multiplier = self.parameters.get("signal_strength_multiplier", 1.0)

        # Data preprocessing
        self.preprocessor = DataPreprocessor()
        self.scaler_type = self.parameters.get("scaler_type", "standard")  # standard, minmax

        # Performance tracking
        self.last_training_time = None
        self.training_score = 0.0
        self.validation_score = 0.0

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def analyze(self, market_data: List[MarketData]) -> TradingSignal:
        """
        Analyze market data using ML model and generate trading signals.
        """
        if not self.is_trained:
            return self._create_hold_signal(market_data[-1] if market_data else None)

        try:
            # Prepare features for prediction
            features = self._prepare_features(market_data)

            if features is None or len(features) == 0:
                return self._create_hold_signal(market_data[-1])

            # Make prediction
            prediction = self._predict(features)

            # Convert prediction to trading signal
            signal = self._prediction_to_signal(market_data[-1], prediction, features)

            return signal

        except Exception as e:
            self.logger.error(f"Error in ML analysis: {e}")
            raise StrategyException(f"ML analysis failed: {e}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        params = self.parameters.copy()
        params.update({
            "is_trained": self.is_trained,
            "last_training_time": self.last_training_time,
            "training_score": self.training_score,
            "validation_score": self.validation_score
        })
        return params

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        self.parameters.update(parameters)

    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal."""
        if not signal:
            return False

        if signal.confidence < self.confidence_threshold:
            return False

        return self._validate_signal_specific(signal)

    async def train_model(
        self,
        training_data: List[MarketData],
        retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train the ML model with provided data.
        """
        try:
            self.logger.info(f"Starting model training with {len(training_data)} data points")

            # Prepare training dataset
            features_df = self._prepare_training_features(training_data)

            if features_df.empty:
                raise StrategyException("No features could be generated from training data")

            # Split data
            X, y = self._prepare_training_data(features_df)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.test_size + self.validation_size, random_state=42
            )

            if self.validation_size > 0:
                val_ratio = self.validation_size / (self.test_size + self.validation_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=1-val_ratio, random_state=42
                )
            else:
                X_val, y_val = None, None
                X_test, y_test = X_temp, y_temp

            # Scale features
            self.scaler = self._create_scaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_val_scaled = None

            # Train model
            self.model = self._create_model()
            training_result = self._train_model_impl(
                X_train_scaled, y_train, X_val_scaled, y_val
            )

            # Evaluate model
            self.training_score = self._evaluate_model(X_train_scaled, y_train)
            test_score = self._evaluate_model(X_test_scaled, y_test)

            if X_val_scaled is not None:
                self.validation_score = self._evaluate_model(X_val_scaled, y_val)
            else:
                self.validation_score = test_score

            self.is_trained = True
            self.last_training_time = datetime.now()

            # Save model if path provided
            if self.model_path:
                self.save_model(self.model_path)

            training_metrics = {
                "training_score": self.training_score,
                "validation_score": self.validation_score,
                "test_score": test_score,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": X_train.shape[1],
                "training_time": datetime.now(),
                **training_result
            }

            self.logger.info(f"Model training completed. Training score: {self.training_score:.4f}, "
                           f"Validation score: {self.validation_score:.4f}")

            return training_metrics

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise StrategyException(f"Training failed: {e}")

    def save_model(self, filepath: str) -> None:
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise StrategyException("No trained model to save")

        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "parameters": self.parameters,
                "feature_columns": self.feature_columns,
                "training_time": self.last_training_time,
                "training_score": self.training_score,
                "validation_score": self.validation_score
            }

            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise StrategyException(f"Model save failed: {e}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model and scaler."""
        try:
            model_data = joblib.load(filepath)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data.get("feature_columns", [])
            self.last_training_time = model_data.get("training_time")
            self.training_score = model_data.get("training_score", 0.0)
            self.validation_score = model_data.get("validation_score", 0.0)

            # Update parameters from saved model
            saved_params = model_data.get("parameters", {})
            self.parameters.update(saved_params)

            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise StrategyException(f"Model load failed: {e}")

    def needs_retraining(self) -> bool:
        """Check if model needs retraining."""
        if not self.is_trained or not self.last_training_time:
            return True

        time_since_training = datetime.now() - self.last_training_time
        return time_since_training.total_seconds() > (self.retrain_interval * 3600)

    # Abstract methods for subclasses

    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return a new model instance."""
        pass

    @abstractmethod
    def _train_model_impl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Train the model implementation."""
        pass

    @abstractmethod
    def _predict(self, features: np.ndarray) -> Union[float, np.ndarray]:
        """Make prediction using the trained model."""
        pass

    @abstractmethod
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance."""
        pass

    # Protected methods for subclasses

    def _validate_signal_specific(self, signal: TradingSignal) -> bool:
        """Strategy-specific signal validation."""
        return True

    def _prepare_features(self, market_data: List[MarketData]) -> Optional[np.ndarray]:
        """Prepare features for prediction."""
        if len(market_data) < self.lookback_window:
            return None

        try:
            # Create features DataFrame
            features_df = self.preprocessor.create_features_for_ml(market_data)

            if features_df.empty:
                return None

            # Select relevant features
            if self.feature_columns:
                available_columns = [col for col in self.feature_columns if col in features_df.columns]
                if not available_columns:
                    self.logger.warning("No configured feature columns found in data")
                    return None
                features_df = features_df[available_columns]

            # Get the latest features for prediction
            latest_features = features_df.iloc[-1:].values

            # Scale features if scaler is available
            if self.scaler is not None:
                latest_features = self.scaler.transform(latest_features)

            return latest_features

        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None

    def _prepare_training_features(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Prepare features for training."""
        return self.preprocessor.create_features_for_ml(market_data)

    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data (X, y) from features DataFrame."""
        # Remove rows with NaN values
        features_df = features_df.dropna()

        if self.target_column not in features_df.columns:
            raise StrategyException(f"Target column '{self.target_column}' not found")

        # Separate features and target
        y = features_df[self.target_column].values

        # Select feature columns
        if self.feature_columns:
            available_columns = [col for col in self.feature_columns
                               if col in features_df.columns and col != self.target_column]
            if not available_columns:
                raise StrategyException("No valid feature columns found")
            X = features_df[available_columns].values
            self.feature_columns = available_columns  # Update with available columns
        else:
            # Use all columns except target
            feature_cols = [col for col in features_df.columns if col != self.target_column]
            X = features_df[feature_cols].values
            self.feature_columns = feature_cols

        return X, y

    def _create_scaler(self):
        """Create and return a feature scaler."""
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

    def _prediction_to_signal(
        self,
        current_data: MarketData,
        prediction: Union[float, np.ndarray],
        features: np.ndarray
    ) -> TradingSignal:
        """Convert model prediction to trading signal."""
        # Extract prediction value
        if isinstance(prediction, np.ndarray):
            if len(prediction.shape) > 1:
                pred_value = prediction[0, 0] if prediction.shape[0] > 0 else 0.0
            else:
                pred_value = prediction[0] if len(prediction) > 0 else 0.0
        else:
            pred_value = float(prediction)

        # Determine signal type based on prediction
        signal_type = SignalType.HOLD
        strength = 0.0
        confidence = 0.0

        # Convert prediction to signal (assuming prediction is price return)
        if pred_value > 0.001:  # Positive return threshold
            signal_type = SignalType.BUY
            strength = min(1.0, abs(pred_value) * self.signal_strength_multiplier)
        elif pred_value < -0.001:  # Negative return threshold
            signal_type = SignalType.SELL
            strength = min(1.0, abs(pred_value) * self.signal_strength_multiplier)

        # Calculate confidence based on prediction magnitude and model performance
        if signal_type != SignalType.HOLD:
            base_confidence = min(1.0, abs(pred_value) * 10)  # Scale prediction magnitude
            model_confidence = (self.validation_score + self.training_score) / 2
            confidence = base_confidence * model_confidence

        return TradingSignal(
            symbol=current_data.symbol,
            signal_type=signal_type,
            strength=strength,
            price=current_data.close,
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "prediction": pred_value,
                "model_score": self.validation_score,
                "features_used": len(self.feature_columns) if self.feature_columns else 0
            }
        )

    def _create_hold_signal(self, current_data: Optional[MarketData]) -> TradingSignal:
        """Create a HOLD signal."""
        if not current_data:
            return TradingSignal(
                symbol="UNKNOWN",
                signal_type=SignalType.HOLD,
                strength=0.0,
                price=0.0,
                strategy_name=self.name,
                confidence=0.0
            )

        return TradingSignal(
            symbol=current_data.symbol,
            signal_type=SignalType.HOLD,
            strength=0.0,
            price=current_data.close,
            strategy_name=self.name,
            confidence=1.0,
            metadata={"reason": "model_not_trained"}
        )

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported by the model."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None

        try:
            importances = self.model.feature_importances_
            if self.feature_columns and len(importances) == len(self.feature_columns):
                return dict(zip(self.feature_columns, importances))
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")

        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "training_score": self.training_score,
            "validation_score": self.validation_score,
            "last_training_time": self.last_training_time,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "lookback_window": self.lookback_window,
            "prediction_horizon": self.prediction_horizon
        }