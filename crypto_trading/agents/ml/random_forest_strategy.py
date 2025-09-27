"""
Random Forest-based trading strategy.
Uses Random Forest classifier for price direction prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Union, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

from .ml_strategy import MLStrategy
from ...core.models import MarketData, TradingSignal, SignalType
from ...core.exceptions import StrategyException


class RandomForestStrategy(MLStrategy):
    """
    Random Forest-based trading strategy for price direction prediction.
    Uses ensemble learning to capture complex market patterns.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        # Set Random Forest specific defaults
        rf_defaults = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "class_weight": "balanced",
            "label_lookforward": 5,  # Look 5 periods ahead for labels
            "label_threshold": 0.01,  # 1% price movement threshold
            "prediction_type": "classification"  # vs regression
        }

        # Merge with provided parameters
        if parameters:
            rf_defaults.update(parameters)

        super().__init__(rf_defaults)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def _create_model(self) -> RandomForestClassifier:
        """Create and return a new Random Forest model instance."""
        return RandomForestClassifier(
            n_estimators=self.parameters.get("n_estimators", 100),
            max_depth=self.parameters.get("max_depth", 10),
            min_samples_split=self.parameters.get("min_samples_split", 5),
            min_samples_leaf=self.parameters.get("min_samples_leaf", 2),
            random_state=self.parameters.get("random_state", 42),
            class_weight=self.parameters.get("class_weight", "balanced"),
            n_jobs=-1
        )

    def _train_model_impl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Train the Random Forest model implementation."""
        try:
            # Train the model
            self.model.fit(X_train, y_train)

            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)

            training_result = {
                "train_accuracy": train_accuracy,
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth
            }

            # Validation metrics if validation data is provided
            if X_val is not None and y_val is not None:
                val_predictions = self.model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_predictions)
                training_result["val_accuracy"] = val_accuracy

            self.logger.info(f"Random Forest training completed. Train accuracy: {train_accuracy:.3f}")

            return training_result

        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
            raise StrategyException(f"Random Forest training failed: {e}")

    def _predict(self, features: np.ndarray) -> Union[float, np.ndarray]:
        """Make prediction using the trained Random Forest model."""
        if self.model is None:
            raise StrategyException("Model is not trained")

        try:
            # Ensure proper shape for prediction
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            elif len(features.shape) == 2 and features.shape[0] > 1:
                # Use the last row for prediction
                features = features[-1:]

            # Get prediction and probabilities
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            # Return prediction with confidence (max probability)
            confidence = float(np.max(probabilities))

            # For classification, return the class prediction
            return prediction

        except Exception as e:
            self.logger.error(f"Error making Random Forest prediction: {e}")
            raise StrategyException(f"Prediction failed: {e}")

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate Random Forest model performance."""
        if self.model is None:
            return 0.0

        try:
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            return accuracy

        except Exception as e:
            self.logger.error(f"Error evaluating Random Forest model: {e}")
            return 0.0

    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with generated labels for Random Forest."""
        try:
            # Generate labels for classification
            labels = self._generate_labels_from_features(features_df)

            # Align features with labels (remove rows where labels couldn't be generated)
            features_df_aligned = features_df.iloc[:len(labels)].copy()

            # Remove NaN values
            features_df_aligned = features_df_aligned.dropna()
            labels = labels[:len(features_df_aligned)]

            # Prepare features
            if self.feature_columns:
                available_columns = [col for col in self.feature_columns
                                   if col in features_df_aligned.columns]
                if not available_columns:
                    raise StrategyException("No valid feature columns found")
                X = features_df_aligned[available_columns].values
                self.feature_columns = available_columns
            else:
                # Use all columns as features
                X = features_df_aligned.values
                self.feature_columns = list(features_df_aligned.columns)

            y = np.array(labels)

            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]

            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing Random Forest training data: {e}")
            raise StrategyException(f"Data preparation failed: {e}")

    def _generate_labels_from_features(self, features_df: pd.DataFrame) -> List[int]:
        """Generate binary classification labels from price data."""
        try:
            labels = []
            lookforward = self.parameters.get("label_lookforward", 5)
            threshold = self.parameters.get("label_threshold", 0.01)

            # Assume the features DataFrame has a 'close' column
            if 'close' not in features_df.columns:
                # Try common price column names
                price_cols = ['close_price', 'price', 'Close']
                price_col = None
                for col in price_cols:
                    if col in features_df.columns:
                        price_col = col
                        break

                if price_col is None:
                    raise StrategyException("No price column found for label generation")
            else:
                price_col = 'close'

            prices = features_df[price_col].values

            for i in range(len(prices) - lookforward):
                current_price = prices[i]
                future_price = prices[i + lookforward]

                # Calculate price change
                price_change = (future_price - current_price) / current_price

                # Generate binary labels
                if price_change > threshold:
                    labels.append(1)  # Buy signal
                else:
                    labels.append(0)  # Sell signal

            return labels

        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            raise StrategyException(f"Label generation failed: {e}")

    def _prediction_to_signal(
        self,
        current_data: MarketData,
        prediction: Union[float, np.ndarray],
        features: np.ndarray
    ) -> TradingSignal:
        """Convert Random Forest prediction to trading signal."""
        try:
            # Get prediction confidence from model
            if self.model is not None:
                # Reshape features for prediction if needed
                if len(features.shape) == 1:
                    features_reshaped = features.reshape(1, -1)
                else:
                    features_reshaped = features[-1:] if len(features.shape) == 2 else features

                probabilities = self.model.predict_proba(features_reshaped)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 0.5

            # Convert prediction to signal type
            if isinstance(prediction, np.ndarray):
                pred_class = int(prediction[0]) if len(prediction) > 0 else 0
            else:
                pred_class = int(prediction)

            # Map prediction to signal type
            if pred_class == 1:
                signal_type = SignalType.BUY
                strength = confidence
            else:
                signal_type = SignalType.SELL
                strength = confidence

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                signal_type = SignalType.HOLD
                strength = 0.0
                confidence = 0.0

            return TradingSignal(
                symbol=current_data.symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_data.close,
                strategy_name=self.name,
                confidence=confidence,
                metadata={
                    "prediction_class": pred_class,
                    "model_confidence": confidence,
                    "model_type": "RandomForest",
                    "features_used": len(self.feature_columns) if self.feature_columns else 0
                }
            )

        except Exception as e:
            self.logger.error(f"Error converting prediction to signal: {e}")
            return self._create_hold_signal(current_data)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained Random Forest model."""
        if not self.is_trained or self.model is None:
            return None

        try:
            importances = self.model.feature_importances_
            if self.feature_columns and len(importances) == len(self.feature_columns):
                importance_dict = dict(zip(self.feature_columns, importances))
                # Sort by importance
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            self.logger.error(f"Failed to get Random Forest feature importance: {e}")

        return None

    def evaluate_model_detailed(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get detailed evaluation metrics for Random Forest."""
        if not self.is_trained or self.model is None:
            return {}

        try:
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Get detailed classification report
            unique_labels = np.unique(y_test)
            if len(unique_labels) > 1:
                report = classification_report(y_test, predictions, output_dict=True)
                return {
                    "accuracy": accuracy,
                    "precision": report['weighted avg']['precision'],
                    "recall": report['weighted avg']['recall'],
                    "f1_score": report['weighted avg']['f1-score'],
                    "support": len(y_test),
                    "classification_report": report
                }
            else:
                return {"accuracy": accuracy, "support": len(y_test)}

        except Exception as e:
            self.logger.error(f"Error in detailed evaluation: {e}")
            return {}