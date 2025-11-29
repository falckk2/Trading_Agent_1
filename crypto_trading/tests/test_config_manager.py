"""
Test suite for Configuration Manager functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path

from crypto_trading.core.config_manager import ConfigManager
from crypto_trading.core.exceptions import ConfigurationError


class TestConfigManager:
    """Test Configuration Manager functionality."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "trading": {
                    "enabled": True,
                    "symbols": ["BTC/USDT", "ETH/USDT"]
                },
                "risk": {
                    "max_position_size_pct": 0.1,
                    "max_daily_loss_pct": 0.05
                }
            }
            json.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def config_manager(self, temp_config_file):
        """Create ConfigManager instance with temporary file."""
        return ConfigManager(temp_config_file)

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            config_manager = ConfigManager(str(config_path))

            # Should create file with defaults
            assert config_path.exists()
            assert config_manager.config_data is not None

    def test_get_config_value(self, config_manager):
        """Test getting configuration values."""
        # Test existing value
        enabled = config_manager.get("trading.enabled")
        assert enabled is True

        # Test nested value
        symbols = config_manager.get("trading.symbols")
        assert isinstance(symbols, list)
        assert "BTC/USDT" in symbols

        # Test non-existent value with default
        non_existent = config_manager.get("non.existent.key", "default_value")
        assert non_existent == "default_value"

        # Test non-existent value without default
        non_existent = config_manager.get("non.existent.key")
        assert non_existent is None

    def test_set_config_value(self, config_manager):
        """Test setting configuration values."""
        # Set new value
        config_manager.set("new.setting", "test_value")
        assert config_manager.get("new.setting") == "test_value"

        # Update existing value
        config_manager.set("trading.enabled", False)
        assert config_manager.get("trading.enabled") is False

        # Set nested value
        config_manager.set("agents.rsi.period", 21)
        assert config_manager.get("agents.rsi.period") == 21

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_save_load.json"

            # Create and configure
            config_manager = ConfigManager(str(config_path))
            config_manager.set("test.value", "saved_value")
            config_manager.save()

            # Create new instance and load
            new_config_manager = ConfigManager(str(config_path))
            assert new_config_manager.get("test.value") == "saved_value"

    def test_get_section(self, config_manager):
        """Test getting entire configuration sections."""
        trading_section = config_manager.get_section("trading")
        assert isinstance(trading_section, dict)
        assert "enabled" in trading_section
        assert "symbols" in trading_section

        # Test non-existent section
        empty_section = config_manager.get_section("non_existent")
        assert empty_section == {}

    def test_update_section(self, config_manager):
        """Test updating configuration sections."""
        new_values = {
            "enabled": False,
            "new_setting": "test"
        }

        config_manager.update_section("trading", new_values)

        assert config_manager.get("trading.enabled") is False
        assert config_manager.get("trading.new_setting") == "test"
        # Original symbols should still be there
        assert config_manager.get("trading.symbols") is not None

    def test_reset_to_defaults(self, config_manager):
        """Test resetting configuration to defaults."""
        # Modify some values
        config_manager.set("trading.enabled", False)
        config_manager.set("custom.setting", "custom_value")

        # Reset to defaults
        config_manager.reset_to_defaults()

        # Should have default values
        assert config_manager.get("trading.enabled") is False  # Default value
        assert config_manager.get("custom.setting") is None  # Should be gone

    def test_backup_config(self, config_manager):
        """Test configuration backup."""
        backup_path = config_manager.backup()
        assert Path(backup_path).exists()

        # Backup should contain same data
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        original_data = config_manager.config_data
        assert backup_data == original_data

        # Cleanup
        Path(backup_path).unlink(missing_ok=True)

    def test_validation(self, config_manager):
        """Test configuration validation."""
        # Valid configuration
        validation = config_manager.validate()
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation

    def test_validation_with_errors(self):
        """Test validation with configuration errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_config.json"
            config_manager = ConfigManager(str(config_path))

            # Set invalid values
            config_manager.set("risk.max_position_size_pct", 2.0)  # Invalid: > 1
            config_manager.set("risk.max_daily_loss_pct", -0.1)   # Invalid: < 0

            validation = config_manager.validate()
            assert not validation["valid"]
            assert len(validation["errors"]) > 0

    def test_config_info(self, config_manager):
        """Test getting configuration information."""
        info = config_manager.get_config_info()

        assert "config_path" in info
        assert "config_exists" in info
        assert "format" in info
        assert "sections" in info

        assert info["config_exists"] is True
        assert info["format"] == "json"
        assert isinstance(info["sections"], list)

    def test_yaml_format(self):
        """Test YAML configuration format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Create YAML config
            config_manager = ConfigManager(str(config_path))
            config_manager.set("test.yaml.setting", "yaml_value")
            config_manager.save()

            # Verify file exists and format is detected
            assert config_path.exists()
            info = config_manager.get_config_info()
            assert info["format"] == "yaml"

            # Load and verify
            new_manager = ConfigManager(str(config_path))
            assert new_manager.get("test.yaml.setting") == "yaml_value"

    def test_merge_with_defaults(self):
        """Test merging loaded config with defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "partial_config.json"

            # Create partial config (missing some default sections)
            partial_config = {"trading": {"enabled": True}}
            with open(config_path, 'w') as f:
                json.dump(partial_config, f)

            # Load config
            config_manager = ConfigManager(str(config_path))

            # Should have both loaded and default values
            assert config_manager.get("trading.enabled") is True  # From file
            assert config_manager.get("risk.max_position_size_pct") is not None  # From defaults

    def test_hot_reload(self):
        """Test hot reloading of configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "hot_reload.json"

            # Create initial config
            config_manager = ConfigManager(str(config_path))
            config_manager.set("test.value", "initial")
            config_manager.save()

            # Modify file externally
            import time
            time.sleep(0.1)  # Ensure timestamp difference

            external_config = {"test": {"value": "modified"}}
            with open(config_path, 'w') as f:
                json.dump(external_config, f)

            # Access config (should trigger reload)
            value = config_manager.get("test.value")
            assert value == "modified"

    def test_error_handling(self):
        """Test error handling in configuration operations."""
        # Test with invalid file path
        with pytest.raises(Exception):
            invalid_path = "/invalid/path/that/does/not/exist/config.json"
            config_manager = ConfigManager(invalid_path)
            config_manager.load()

    def test_default_configuration_completeness(self):
        """Test that default configuration contains all required sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "default_test.json"
            config_manager = ConfigManager(str(config_path))

            # Check required sections exist
            required_sections = [
                "trading", "exchange", "risk", "agents",
                "data", "logging", "gui", "notifications", "system"
            ]

            for section in required_sections:
                section_data = config_manager.get_section(section)
                assert isinstance(section_data, dict)
                assert len(section_data) > 0

            # Check specific required settings
            assert config_manager.get("trading.symbols") is not None
            assert config_manager.get("risk.max_position_size_pct") is not None
            assert config_manager.get("agents.rsi.rsi_period") is not None

    def test_configuration_validation_warnings(self):
        """Test configuration validation with warnings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "warning_config.json"
            config_manager = ConfigManager(str(config_path))

            # Set values that should generate warnings
            config_manager.set("agents.rsi.rsi_period", 200)  # Unusual but valid
            config_manager.set("data.storage_path", "/nonexistent/path/data.db")

            validation = config_manager.validate()
            # Should be valid but with warnings
            assert validation["valid"] or len(validation["warnings"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])