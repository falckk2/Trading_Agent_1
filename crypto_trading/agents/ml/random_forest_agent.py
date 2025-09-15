"""
Random Forest-based trading agent.
Uses Random Forest classifier for price direction prediction.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger

from ...core.interfaces import MarketData, TradingSignal, OrderSide
from .base_ml_agent import BaseMLAgent


class RandomForestAgent(BaseMLAgent):
    """Trading agent using Random Forest for price direction prediction."""

    def __init__(self):
        super().__init__(
            name="Random Forest Agent",
            description="Uses Random Forest classifier to predict price direction and generate trading signals"
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        base_params = super().get_default_parameters()
        base_params.update({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "class_weight": "balanced"
        })
        return base_params

    def _get_model_type(self) -> str:
        return "RandomForest"

    def _train_model(self, features: np.ndarray, target: np.ndarray) -> None:
        """Train the Random Forest model."""
        try:
            # Initialize Random Forest with configured parameters
            self.model = RandomForestClassifier(
                n_estimators=self._get_config_value("n_estimators", 100),
                max_depth=self._get_config_value("max_depth", 10),
                min_samples_split=self._get_config_value("min_samples_split", 5),
                min_samples_leaf=self._get_config_value("min_samples_leaf", 2),
                random_state=self._get_config_value("random_state", 42),
                class_weight=self._get_config_value("class_weight", "balanced"),
                n_jobs=-1
            )

            # Train the model
            self.model.fit(features, target)

            # Log training metrics if possible
            if len(features) > 100:  # Only if we have enough data
                train_predictions = self.model.predict(features)
                accuracy = accuracy_score(target, train_predictions)
                logger.info(f"Random Forest training accuracy: {accuracy:.3f}")

        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            raise

    def _make_prediction(self, features: np.ndarray) -> Tuple[int, float]:
        """Make prediction using the Random Forest model."""
        if self.model is None:
            raise ValueError("Model is not trained")

        try:
            # Use the last row of features for prediction
            if len(features.shape) == 2:
                prediction_features = features[-1:] if len(features) > 0 else features
            else:
                prediction_features = features.reshape(1, -1)

            # Get prediction and probability
            prediction = self.model.predict(prediction_features)[0]
            probabilities = self.model.predict_proba(prediction_features)[0]

            # Calculate confidence as the maximum probability
            confidence = float(np.max(probabilities))

            return int(prediction), confidence

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0, 0.0

    def _prediction_to_signal(self, prediction: int, confidence: float, market_data: List[MarketData]) -> TradingSignal:
        """Convert Random Forest prediction to trading signal."""
        symbol = self._get_symbol_from_data(market_data)
        current_price = float(market_data[-1].close) if market_data else None

        # Apply prediction threshold
        prediction_threshold = self._get_config_value("prediction_threshold", 0.6)

        if confidence < prediction_threshold:
            # Low confidence, no signal
            return self._create_signal(
                symbol=symbol,
                action=OrderSide.BUY,  # Default action
                confidence=0.0,
                price=current_price,
                metadata={
                    "signal_reason": "low_confidence",
                    "prediction_class": int(prediction),
                    "model_confidence": confidence,
                    "threshold": prediction_threshold
                }
            )

        # Convert prediction to action
        # Assuming: 0 = sell/short, 1 = hold, 2 = buy/long
        # Or binary: 0 = sell, 1 = buy
        action = self._prediction_to_action(prediction)

        # Adjust confidence based on market conditions
        adjusted_confidence = self._adjust_confidence(confidence, market_data)

        return self._create_signal(
            symbol=symbol,
            action=action,
            confidence=adjusted_confidence,
            price=current_price,
            metadata={
                "signal_reason": "rf_prediction",
                "prediction_class": int(prediction),
                "model_confidence": confidence,
                "adjusted_confidence": adjusted_confidence
            }
        )

    def _prediction_to_action(self, prediction: int) -> OrderSide:
        """Convert prediction class to trading action."""
        # Binary classification: 0 = sell, 1 = buy
        if prediction == 0:
            return OrderSide.SELL
        else:
            return OrderSide.BUY

    def _adjust_confidence(self, base_confidence: float, market_data: List[MarketData]) -> float:
        """Adjust confidence based on market conditions."""
        try:
            # Calculate volatility from recent data
            if len(market_data) < 10:
                return base_confidence

            recent_closes = [float(data.close) for data in market_data[-10:]]
            returns = np.diff(recent_closes) / recent_closes[:-1]
            volatility = np.std(returns)

            # Reduce confidence in high volatility periods
            volatility_adjustment = 1 / (1 + volatility * 10)  # Scale volatility impact

            # Calculate volume trend
            if len(market_data) >= 5:
                recent_volumes = [float(data.volume) for data in market_data[-5:]]
                avg_volume = np.mean(recent_volumes)
                current_volume = recent_volumes[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Increase confidence with higher volume
                volume_adjustment = min(1.2, 1 + (volume_ratio - 1) * 0.1)
            else:
                volume_adjustment = 1.0

            adjusted_confidence = base_confidence * volatility_adjustment * volume_adjustment
            return max(0.1, min(adjusted_confidence, 0.95))

        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
            return base_confidence

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained Random Forest model."""
        if self.model is None or not self.is_trained:
            return {}

        try:
            importances = self.model.feature_importances_
            feature_importance = {}

            for i, importance in enumerate(importances):
                if i < len(self.feature_columns):
                    feature_importance[self.feature_columns[i]] = float(importance)

            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def train_with_labels(self, market_data: List[MarketData]) -> None:
        """Train the model by generating labels from market data."""
        try:
            if len(market_data) < self._get_minimum_data_points():
                raise ValueError("Insufficient data for training")

            # Generate labels based on future price movement
            labels = self._generate_labels(market_data)

            # Remove the last few samples where we can't generate labels
            training_data = market_data[:len(labels)]

            # Train the model
            self.train(training_data, labels)

        except Exception as e:
            logger.error(f"Error in automated training: {e}")
            raise

    def _generate_labels(self, market_data: List[MarketData]) -> List[int]:
        """Generate training labels from market data."""
        try:
            labels = []
            lookforward = self._get_config_value("label_lookforward", 5)  # Look 5 periods ahead

            for i in range(len(market_data) - lookforward):
                current_price = float(market_data[i].close)
                future_price = float(market_data[i + lookforward].close)

                # Calculate price change
                price_change = (future_price - current_price) / current_price

                # Generate binary labels based on price movement threshold
                threshold = self._get_config_value("label_threshold", 0.01)  # 1% threshold

                if price_change > threshold:
                    labels.append(1)  # Buy signal
                else:
                    labels.append(0)  # Sell signal

            return labels

        except Exception as e:
            logger.error(f"Error generating labels: {e}")
            return []

    def evaluate_model(self, test_data: List[MarketData]) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained")

        try:
            # Generate test labels
            test_labels = self._generate_labels(test_data)
            test_features = self._prepare_training_features(test_data[:len(test_labels)])

            if test_features is None or len(test_labels) == 0:
                raise ValueError("Failed to prepare test data")

            # Make predictions
            predictions = self.model.predict(test_features)

            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)

            # Calculate additional metrics
            unique_labels = np.unique(test_labels)
            if len(unique_labels) > 1:
                report = classification_report(test_labels, predictions, output_dict=True)
                precision = report['weighted avg']['precision']
                recall = report['weighted avg']['recall']
                f1_score = report['weighted avg']['f1-score']
            else:
                precision = recall = f1_score = 0.0

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "test_samples": len(test_labels)
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}