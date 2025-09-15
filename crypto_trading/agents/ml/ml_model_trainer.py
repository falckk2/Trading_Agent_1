"""
ML Model Trainer utility for automated training and validation.
Provides centralized training orchestration for ML strategies.
"""

import asyncio
from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
import logging

from ...core.models import MarketData
from ...data.data_manager import DataManager
from .ml_strategy import MLStrategy


class MLModelTrainer:
    """
    Centralized ML model trainer for automated training and validation.
    Supports multiple strategies and data sources.
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)

        # Training configuration
        self.training_config = {
            "min_training_samples": 1000,
            "validation_split": 0.2,
            "test_split": 0.1,
            "lookback_days": 90,
            "symbols": ["BTC-USD", "ETH-USD"],
            "timeframe": "1h"
        }

        # Training history
        self.training_history: Dict[str, List[Dict[str, Any]]] = {}

    async def train_strategy(
        self,
        strategy: MLStrategy,
        symbol: str,
        training_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Train a single ML strategy."""
        config = self.training_config.copy()
        if training_config:
            config.update(training_config)

        try:
            self.logger.info(f"Starting training for {strategy.name} on {symbol}")

            # Get training data
            training_data = await self._get_training_data(symbol, config)

            if len(training_data) < config["min_training_samples"]:
                raise ValueError(f"Insufficient training data: {len(training_data)} < {config['min_training_samples']}")

            # Train the model
            training_result = await strategy.train_model(training_data)

            # Record training history
            self._record_training_history(strategy.name, symbol, training_result)

            self.logger.info(f"Training completed for {strategy.name} on {symbol}")
            return training_result

        except Exception as e:
            self.logger.error(f"Training failed for {strategy.name} on {symbol}: {e}")
            raise

    async def train_multiple_strategies(
        self,
        strategies: List[MLStrategy],
        symbols: List[str] = None,
        training_config: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Train multiple strategies on multiple symbols."""
        symbols = symbols or self.training_config["symbols"]
        results = {}

        for strategy in strategies:
            strategy_results = {}

            for symbol in symbols:
                try:
                    result = await self.train_strategy(strategy, symbol, training_config)
                    strategy_results[symbol] = result

                    # Small delay between trainings to prevent resource exhaustion
                    await asyncio.sleep(1)

                except Exception as e:
                    self.logger.error(f"Failed to train {strategy.name} on {symbol}: {e}")
                    strategy_results[symbol] = {"error": str(e)}

            results[strategy.name] = strategy_results

        return results

    async def automated_training_schedule(
        self,
        strategies: List[MLStrategy],
        symbols: List[str] = None,
        interval_hours: int = 24
    ) -> None:
        """Run automated training on a schedule."""
        symbols = symbols or self.training_config["symbols"]

        self.logger.info(f"Starting automated training schedule (every {interval_hours} hours)")

        while True:
            try:
                self.logger.info("Starting scheduled training cycle")

                results = await self.train_multiple_strategies(strategies, symbols)

                # Log summary
                total_success = 0
                total_attempts = 0

                for strategy_name, strategy_results in results.items():
                    for symbol, result in strategy_results.items():
                        total_attempts += 1
                        if "error" not in result:
                            total_success += 1

                self.logger.info(f"Training cycle completed: {total_success}/{total_attempts} successful")

                # Wait for next training cycle
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                self.logger.error(f"Error in automated training schedule: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying

    async def validate_strategy(
        self,
        strategy: MLStrategy,
        symbol: str,
        validation_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate a trained strategy on out-of-sample data."""
        if not strategy.is_trained:
            raise ValueError("Strategy must be trained before validation")

        config = validation_config or {}
        validation_days = config.get("validation_days", 30)

        try:
            # Get validation data (recent data not used in training)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=validation_days)

            validation_data = await self.data_manager.get_historical_data(
                symbol=symbol,
                timeframe=self.training_config["timeframe"],
                start_date=start_date,
                end_date=end_date
            )

            if not validation_data:
                raise ValueError("No validation data available")

            # Run validation
            validation_results = await self._run_validation(strategy, validation_data)

            self.logger.info(f"Validation completed for {strategy.name} on {symbol}")
            return validation_results

        except Exception as e:
            self.logger.error(f"Validation failed for {strategy.name} on {symbol}: {e}")
            raise

    async def compare_strategies(
        self,
        strategies: List[MLStrategy],
        symbol: str,
        comparison_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Compare multiple strategies on the same data."""
        config = comparison_config or {}
        comparison_days = config.get("comparison_days", 30)

        try:
            # Get comparison data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=comparison_days)

            comparison_data = await self.data_manager.get_historical_data(
                symbol=symbol,
                timeframe=self.training_config["timeframe"],
                start_date=start_date,
                end_date=end_date
            )

            if not comparison_data:
                raise ValueError("No comparison data available")

            # Compare strategies
            comparison_results = {}

            for strategy in strategies:
                if strategy.is_trained:
                    try:
                        result = await self._run_validation(strategy, comparison_data)
                        comparison_results[strategy.name] = result
                    except Exception as e:
                        comparison_results[strategy.name] = {"error": str(e)}
                else:
                    comparison_results[strategy.name] = {"error": "Strategy not trained"}

            # Calculate rankings
            rankings = self._calculate_strategy_rankings(comparison_results)

            return {
                "symbol": symbol,
                "comparison_period": f"{comparison_days} days",
                "results": comparison_results,
                "rankings": rankings
            }

        except Exception as e:
            self.logger.error(f"Strategy comparison failed for {symbol}: {e}")
            raise

    async def get_training_history(self, strategy_name: str = None) -> Dict[str, Any]:
        """Get training history for strategies."""
        if strategy_name:
            return self.training_history.get(strategy_name, [])
        return self.training_history

    def set_training_config(self, config: Dict[str, Any]) -> None:
        """Update training configuration."""
        self.training_config.update(config)

    # Private methods

    async def _get_training_data(
        self,
        symbol: str,
        config: Dict[str, Any]
    ) -> List[MarketData]:
        """Get training data for a symbol."""
        lookback_days = config.get("lookback_days", 90)
        timeframe = config.get("timeframe", "1h")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        training_data = await self.data_manager.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        return training_data

    async def _run_validation(
        self,
        strategy: MLStrategy,
        validation_data: List[MarketData]
    ) -> Dict[str, Any]:
        """Run validation on strategy."""
        signals = []
        correct_predictions = 0
        total_predictions = 0

        # Sliding window validation
        lookback_window = strategy.lookback_window

        for i in range(lookback_window, len(validation_data)):
            try:
                # Get data for prediction
                prediction_data = validation_data[max(0, i-lookback_window):i]

                # Generate signal
                signal = strategy.analyze(prediction_data)

                if signal.signal_type.value != "hold":
                    signals.append(signal)

                    # Check if prediction was correct (simplified)
                    if i < len(validation_data) - 1:
                        current_price = validation_data[i].close
                        next_price = validation_data[i + 1].close
                        actual_return = (next_price - current_price) / current_price

                        predicted_direction = 1 if signal.signal_type.value == "buy" else -1
                        actual_direction = 1 if actual_return > 0 else -1

                        if predicted_direction == actual_direction:
                            correct_predictions += 1

                        total_predictions += 1

            except Exception as e:
                self.logger.warning(f"Error in validation step {i}: {e}")

        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        signal_count = len(signals)
        avg_confidence = sum(s.confidence for s in signals) / signal_count if signals else 0
        avg_strength = sum(s.strength for s in signals) / signal_count if signals else 0

        return {
            "accuracy": accuracy,
            "total_signals": signal_count,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "avg_confidence": avg_confidence,
            "avg_strength": avg_strength,
            "validation_samples": len(validation_data)
        }

    def _record_training_history(
        self,
        strategy_name: str,
        symbol: str,
        training_result: Dict[str, Any]
    ) -> None:
        """Record training history."""
        if strategy_name not in self.training_history:
            self.training_history[strategy_name] = []

        history_entry = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "result": training_result
        }

        self.training_history[strategy_name].append(history_entry)

        # Keep only last 100 training records per strategy
        if len(self.training_history[strategy_name]) > 100:
            self.training_history[strategy_name] = self.training_history[strategy_name][-100:]

    def _calculate_strategy_rankings(
        self,
        comparison_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate strategy rankings based on performance metrics."""
        # Calculate composite scores
        scores = {}

        for strategy_name, result in comparison_results.items():
            if "error" in result:
                scores[strategy_name] = 0
                continue

            # Weighted score calculation
            accuracy = result.get("accuracy", 0)
            avg_confidence = result.get("avg_confidence", 0)
            total_signals = result.get("total_signals", 0)

            # Normalize signal count (more signals can be good up to a point)
            signal_score = min(1.0, total_signals / 100)

            composite_score = (
                accuracy * 0.5 +
                avg_confidence * 0.3 +
                signal_score * 0.2
            )

            scores[strategy_name] = composite_score

        # Rank strategies
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {strategy: rank + 1 for rank, (strategy, _) in enumerate(sorted_strategies)}

        return rankings