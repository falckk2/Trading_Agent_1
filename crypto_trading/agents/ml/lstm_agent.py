"""
LSTM (Long Short-Term Memory) trading agent implementation.
Uses deep learning for time series prediction and signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score

from ...core.models import MarketData, TradingSignal, SignalType
from ..base_agent import BaseAgent as BaseTradingAgent
from .ml_strategy import MLStrategy


class LSTMStrategy(MLStrategy):
    """
    LSTM-based trading strategy for time series prediction.
    Uses recurrent neural networks to capture temporal patterns.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            "sequence_length": 60,
            "lstm_units": [50, 50],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "validation_split": 0.2,
            "prediction_type": "return"  # "return" or "price"
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(default_params)

        # LSTM-specific parameters
        self.sequence_length = self.parameters.get("sequence_length", 60)
        self.lstm_units = self.parameters.get("lstm_units", [50, 50])
        self.dropout_rate = self.parameters.get("dropout_rate", 0.2)
        self.learning_rate = self.parameters.get("learning_rate", 0.001)
        self.epochs = self.parameters.get("epochs", 100)
        self.batch_size = self.parameters.get("batch_size", 32)

        # Training callbacks
        self.callbacks = []
        self.training_history = None

    def _create_model(self) -> tf.keras.Model:
        """Create LSTM model architecture."""
        model = Sequential()

        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            if i == 0:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=(self.sequence_length, len(self.feature_columns) if self.feature_columns else 1)
                ))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))

            # Add dropout for regularization
            model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(1, activation='linear'))

        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def _train_model_impl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Train LSTM model."""
        # Reshape data for LSTM (samples, time steps, features)
        X_train_reshaped = self._reshape_for_lstm(X_train)
        y_train_reshaped = y_train.reshape(-1, 1)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_reshaped = self._reshape_for_lstm(X_val)
            y_val_reshaped = y_val.reshape(-1, 1)
            validation_data = (X_val_reshaped, y_val_reshaped)

        # Setup callbacks
        self.callbacks = self._setup_callbacks()

        # Train model
        self.training_history = self.model.fit(
            X_train_reshaped,
            y_train_reshaped,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=self.callbacks,
            verbose=0
        )

        # Extract training metrics
        final_loss = self.training_history.history['loss'][-1]
        final_val_loss = self.training_history.history['val_loss'][-1] if validation_data else None

        return {
            "final_loss": final_loss,
            "final_val_loss": final_val_loss,
            "epochs_trained": len(self.training_history.history['loss']),
            "best_epoch": self._find_best_epoch()
        }

    def _predict(self, features: np.ndarray) -> Union[float, np.ndarray]:
        """Make prediction using LSTM model."""
        # Reshape for LSTM prediction
        features_reshaped = self._reshape_for_lstm(features)
        prediction = self.model.predict(features_reshaped, verbose=0)
        return prediction[0, 0] if prediction.shape[0] > 0 else 0.0

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate LSTM model performance."""
        X_reshaped = self._reshape_for_lstm(X)
        y_reshaped = y.reshape(-1, 1)

        # Get predictions
        predictions = self.model.predict(X_reshaped, verbose=0)

        # Calculate R² score
        r2 = r2_score(y_reshaped, predictions)
        return max(0.0, r2)  # Ensure non-negative score

    def _prepare_features(self, market_data: List[MarketData]) -> Optional[np.ndarray]:
        """Prepare sequential features for LSTM prediction."""
        if len(market_data) < self.sequence_length + self.lookback_window:
            return None

        try:
            # Create features DataFrame
            features_df = self.preprocessor.create_features_for_ml(market_data)

            if features_df.empty or len(features_df) < self.sequence_length:
                return None

            # Select relevant features
            if self.feature_columns:
                available_columns = [col for col in self.feature_columns if col in features_df.columns]
                if not available_columns:
                    return None
                features_df = features_df[available_columns]

            # Get the last sequence_length rows
            sequence_data = features_df.iloc[-self.sequence_length:].values

            # Scale features if scaler is available
            if self.scaler is not None:
                sequence_data = self.scaler.transform(sequence_data)

            # Reshape for LSTM: (1, sequence_length, features)
            return sequence_data.reshape(1, self.sequence_length, -1)

        except Exception as e:
            self.logger.error(f"LSTM feature preparation failed: {e}")
            return None

    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential training data for LSTM."""
        # Remove rows with NaN values
        features_df = features_df.dropna()

        if len(features_df) < self.sequence_length + 1:
            raise ValueError("Insufficient data for sequence creation")

        if self.target_column not in features_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Select feature columns
        if self.feature_columns:
            available_columns = [col for col in self.feature_columns
                               if col in features_df.columns and col != self.target_column]
            if not available_columns:
                raise ValueError("No valid feature columns found")
            feature_data = features_df[available_columns].values
            self.feature_columns = available_columns
        else:
            feature_cols = [col for col in features_df.columns if col != self.target_column]
            feature_data = features_df[feature_cols].values
            self.feature_columns = feature_cols

        target_data = features_df[self.target_column].values

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(feature_data)):
            X_sequences.append(feature_data[i-self.sequence_length:i])
            y_sequences.append(target_data[i])

        return np.array(X_sequences), np.array(y_sequences)

    def _reshape_for_lstm(self, data: np.ndarray) -> np.ndarray:
        """Reshape data for LSTM input."""
        if len(data.shape) == 2:
            # Assume data is (samples, features) and needs to be (samples, sequence_length, features)
            n_samples = data.shape[0]
            n_features = data.shape[1] // self.sequence_length

            if data.shape[1] % self.sequence_length != 0:
                # If not evenly divisible, truncate
                truncate_cols = (data.shape[1] // self.sequence_length) * self.sequence_length
                data = data[:, :truncate_cols]
                n_features = truncate_cols // self.sequence_length

            return data.reshape(n_samples, self.sequence_length, n_features)

        elif len(data.shape) == 3:
            # Already in correct shape
            return data

        else:
            raise ValueError(f"Unexpected data shape for LSTM: {data.shape}")

    def _setup_callbacks(self) -> List:
        """Setup training callbacks."""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.parameters.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)

        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.parameters.get('reduce_lr_patience', 5),
            min_lr=1e-7,
            verbose=0
        )
        callbacks.append(reduce_lr)

        return callbacks

    def _find_best_epoch(self) -> int:
        """Find the epoch with the best validation loss."""
        if not self.training_history or 'val_loss' not in self.training_history.history:
            return len(self.training_history.history['loss']) if self.training_history else 0

        val_losses = self.training_history.history['val_loss']
        return np.argmin(val_losses) + 1

    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Get training history."""
        if self.training_history:
            return self.training_history.history
        return None

    def plot_training_history(self):
        """Plot training history (requires matplotlib)."""
        if not self.training_history:
            self.logger.warning("No training history available")
            return

        try:
            import matplotlib.pyplot as plt

            history = self.training_history.history

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot loss
            ax1.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            # Plot metrics
            if 'mae' in history:
                ax2.plot(history['mae'], label='Training MAE')
                if 'val_mae' in history:
                    ax2.plot(history['val_mae'], label='Validation MAE')
                ax2.set_title('Model MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")


class LSTMAgent(BaseTradingAgent):
    """
    LSTM-based trading agent using deep learning for time series prediction.
    """

    def __init__(self, strategy=None, exchange=None, risk_manager=None,
                 portfolio_manager=None, agent_config=None):
        # Create LSTM strategy if not provided
        if strategy is None:
            strategy_params = agent_config.get('strategy_params', {}) if agent_config else {}
            strategy = LSTMStrategy(strategy_params)

        super().__init__(strategy, exchange, risk_manager, portfolio_manager, agent_config)

    async def _initialize_agent(self) -> None:
        """LSTM agent specific initialization."""
        self.logger.info("Initializing LSTM trading agent")

        # Validate LSTM-specific configuration
        strategy_params = self.strategy.get_parameters()

        sequence_length = strategy_params.get('sequence_length', 60)
        lstm_units = strategy_params.get('lstm_units', [50, 50])

        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")

        if not lstm_units or any(units < 1 for units in lstm_units):
            raise ValueError("LSTM units must be positive")

        # Check if model needs loading
        model_path = strategy_params.get('model_path', '')
        if model_path and not self.strategy.is_trained:
            try:
                self.strategy.load_model(model_path)
                self.logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load model from {model_path}: {e}")

        self.logger.info(f"LSTM agent initialized with sequence_length={sequence_length}, "
                        f"lstm_units={lstm_units}")

    async def _process_market_data(self, market_data: List[MarketData]) -> Optional[TradingSignal]:
        """Process market data using LSTM strategy."""
        try:
            signal = self.strategy.analyze(market_data)

            # Log signal details
            if signal.signal_type != SignalType.HOLD:
                metadata = signal.metadata
                self.logger.info(
                    f"LSTM Signal: {signal.symbol} {signal.signal_type.value} "
                    f"strength={signal.strength:.2f} confidence={signal.confidence:.2f} "
                    f"prediction={metadata.get('prediction', 'N/A'):.4f} "
                    f"model_score={metadata.get('model_score', 'N/A'):.3f}"
                )

            return signal

        except Exception as e:
            self.logger.error(f"Error processing market data with LSTM strategy: {e}")
            return None

    def _get_strategy_metrics(self) -> Dict[str, float]:
        """Get LSTM-specific metrics."""
        strategy_params = self.strategy.get_parameters()

        metrics = {
            "sequence_length": strategy_params.get('sequence_length', 60),
            "lstm_layers": len(strategy_params.get('lstm_units', [50, 50])),
            "dropout_rate": strategy_params.get('dropout_rate', 0.2),
            "learning_rate": strategy_params.get('learning_rate', 0.001),
            "batch_size": strategy_params.get('batch_size', 32),
            "is_trained": 1.0 if strategy_params.get('is_trained', False) else 0.0,
            "training_score": strategy_params.get('training_score', 0.0),
            "validation_score": strategy_params.get('validation_score', 0.0)
        }

        # Add feature importance if available
        feature_importance = self.strategy.get_feature_importance()
        if feature_importance:
            for feature, importance in feature_importance.items():
                metrics[f"feature_importance_{feature}"] = importance

        return metrics

    async def train_model_with_data(self, training_data: List[MarketData]) -> Dict[str, Any]:
        """Train the LSTM model with provided market data."""
        try:
            self.logger.info(f"Starting LSTM model training with {len(training_data)} data points")
            training_result = await self.strategy.train_model(training_data)

            self.logger.info("LSTM model training completed")
            return training_result

        except Exception as e:
            self.logger.error(f"LSTM model training failed: {e}")
            raise