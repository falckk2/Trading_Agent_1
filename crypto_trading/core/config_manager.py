"""
Configuration management for the trading system.
Supports JSON and YAML formats with hot reloading.
"""

import json
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from .interfaces import IConfigManager
from ..core.exceptions import ConfigurationError


class ConfigManager(IConfigManager):
    """Configuration manager with file-based storage and hot reloading."""

    def __init__(self, config_path: str = "config/trading_config.json"):
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self.default_config = self._get_default_config()
        self._last_modified = None

        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create default
        if self.config_path.exists():
            self.load()
        else:
            self.config_data = self.default_config.copy()
            self.save()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            "trading": {
                "enabled": False,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "default_agent": "RSI Agent",
                "loop_interval": 10,
                "analysis_lookback_hours": 24
            },
            "exchange": {
                "name": "blofin",
                "sandbox": True,
                "api_key": "",
                "api_secret": "",
                "passphrase": "",
                "rate_limit": 10,
                "timeout": 30
            },
            "risk": {
                "max_position_size_pct": 0.1,
                "max_daily_loss_pct": 0.05,
                "max_total_exposure_pct": 0.5,
                "min_confidence_threshold": 0.6,
                "max_positions_per_symbol": 1,
                "max_total_positions": 5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.06,
                "min_order_size": 10.0,
                "min_order_amount": 0.01,
                "max_leverage": 1.0,
                "portfolio_value": 50000.0
            },
            "agents": {
                "rsi": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "minimum_confidence": 0.6
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "minimum_confidence": 0.5,
                    "use_histogram": True
                },
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "prediction_threshold": 0.6,
                    "lookback_periods": 20,
                    "retrain_interval": 24
                }
            },
            "data": {
                "collection_enabled": True,
                "collection_interval": 60,
                "storage_path": "data/market_data.db",
                "backup_enabled": True,
                "backup_interval": 3600,
                "cleanup_older_than_days": 90
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/trading.log",
                "max_file_size": "10MB",
                "backup_count": 5,
                "console_output": True
            },
            "gui": {
                "theme": "dark",
                "update_interval": 1000,
                "chart_timeframe": "1h",
                "default_window_size": [1200, 800],
                "auto_refresh": True
            },
            "notifications": {
                "enabled": False,
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "to_address": ""
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "timeout": 10
                }
            },
            "system": {
                "auto_restart": True,
                "max_memory_mb": 1024,
                "performance_monitoring": True,
                "health_check_interval": 300
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            # Check if config needs reloading
            self._check_and_reload()

            keys = key.split('.')
            value = self.config_data

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

        except Exception as e:
            logger.error(f"Error getting config value '{key}': {e}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        try:
            keys = key.split('.')
            current = self.config_data

            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the value
            current[keys[-1]] = value
            logger.debug(f"Set config value '{key}' = {value}")

        except Exception as e:
            logger.error(f"Error setting config value '{key}': {e}")
            raise ConfigurationError(f"Failed to set config value: {e}")

    def save(self) -> None:
        """Save configuration to file."""
        try:
            # Determine file format based on extension
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                self._save_yaml()
            else:
                self._save_json()

            self._update_last_modified()
            logger.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def _save_json(self) -> None:
        """Save configuration as JSON."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2, default=str)

    def _save_yaml(self) -> None:
        """Save configuration as YAML."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)

    def load(self) -> None:
        """Load configuration from file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self.config_data = self.default_config.copy()
                return

            # Determine file format and load
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                self._load_yaml()
            else:
                self._load_json()

            # Merge with defaults to ensure all keys exist
            self._merge_with_defaults()
            self._update_last_modified()

            logger.info(f"Configuration loaded from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
            self.config_data = self.default_config.copy()

    def _load_json(self) -> None:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)

    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config_data = yaml.safe_load(f) or {}

    def _merge_with_defaults(self) -> None:
        """Merge loaded config with defaults to ensure all keys exist."""
        def merge_dicts(default: dict, loaded: dict) -> dict:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        self.config_data = merge_dicts(self.default_config, self.config_data)

    def _check_and_reload(self) -> None:
        """Check if config file has been modified and reload if necessary."""
        if not self.config_path.exists():
            return

        try:
            current_modified = self.config_path.stat().st_mtime
            if self._last_modified is None or current_modified > self._last_modified:
                logger.info("Config file modified, reloading...")
                self.load()

        except Exception as e:
            logger.error(f"Error checking config file modification: {e}")

    def _update_last_modified(self) -> None:
        """Update the last modified timestamp."""
        if self.config_path.exists():
            self._last_modified = self.config_path.stat().st_mtime

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self.get(section, {})

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update multiple values in a configuration section."""
        try:
            current_section = self.get_section(section)
            current_section.update(values)
            self.set(section, current_section)
            logger.debug(f"Updated section '{section}' with {len(values)} values")

        except Exception as e:
            logger.error(f"Error updating section '{section}': {e}")
            raise ConfigurationError(f"Failed to update section: {e}")

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config_data = self.default_config.copy()
        self.save()
        logger.info("Configuration reset to defaults")

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the current configuration."""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config_path.stem}_backup_{timestamp}{self.config_path.suffix}"

        backup_file = self.config_path.parent / backup_path

        try:
            # Save current in-memory config to backup location
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_file, 'w') as f:
                json.dump(self.config_data, f, indent=2, default=str)
            logger.info(f"Configuration backed up to {backup_file}")
            return str(backup_file)

        except Exception as e:
            logger.error(f"Error creating config backup: {e}")
            raise ConfigurationError(f"Failed to create backup: {e}")

    def validate(self) -> Dict[str, Any]:
        """Validate configuration values and return validation results."""
        errors = []
        warnings = []

        try:
            # Validate trading settings
            if self.get("trading.loop_interval", 0) < 1:
                errors.append("trading.loop_interval must be at least 1 second")

            # Validate risk settings
            risk_settings = [
                ("risk.max_position_size_pct", 0, 1),
                ("risk.max_daily_loss_pct", 0, 1),
                ("risk.max_total_exposure_pct", 0, 1),
                ("risk.min_confidence_threshold", 0, 1),
                ("risk.stop_loss_pct", 0, 1),
                ("risk.take_profit_pct", 0, 1)
            ]

            for setting, min_val, max_val in risk_settings:
                value = self.get(setting, 0)
                if not (min_val <= value <= max_val):
                    errors.append(f"{setting} must be between {min_val} and {max_val}")

            # Validate agent settings
            rsi_period = self.get("agents.rsi.rsi_period", 14)
            if rsi_period < 2 or rsi_period > 100:
                warnings.append("RSI period should typically be between 2 and 100")

            # Validate exchange settings
            if self.get("trading.enabled", False):
                required_exchange_fields = ["api_key", "api_secret", "passphrase"]
                for field in required_exchange_fields:
                    if not self.get(f"exchange.{field}"):
                        errors.append(f"exchange.{field} is required when trading is enabled")

            # Validate file paths
            data_path = Path(self.get("data.storage_path", ""))
            if data_path and not data_path.parent.exists():
                warnings.append(f"Data storage directory does not exist: {data_path.parent}")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "validation_time": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
                "validation_time": datetime.now().isoformat()
            }

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the configuration."""
        return {
            "config_path": str(self.config_path),
            "config_exists": self.config_path.exists(),
            "last_modified": self._last_modified,
            "format": "yaml" if self.config_path.suffix.lower() in ['.yaml', '.yml'] else "json",
            "sections": list(self.config_data.keys()) if isinstance(self.config_data, dict) else [],
            "size_bytes": self.config_path.stat().st_size if self.config_path.exists() else 0
        }