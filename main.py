#!/usr/bin/env python3
"""
Main entry point for the cryptocurrency trading system.
Provides both CLI and GUI interfaces.
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_trading.core.trading_engine import TradingEngine
from crypto_trading.core.config_manager import ConfigManager
from crypto_trading.core.event_bus import EventBus
from crypto_trading.core.risk_manager import RiskManager
from crypto_trading.core.agent_manager import AgentManager
from crypto_trading.exchange.blofin_client import BlofinClient
from loguru import logger


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>"
    )
    logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )


def run_gui():
    """Run the GUI interface."""
    try:
        logger.info("Starting GUI interface...")
        from crypto_trading.gui.main_window import TradingGUI
        app = TradingGUI()
        app.run()
    except Exception as e:
        logger.error(f"Error running GUI: {e}")
        return 1
    return 0


async def run_cli():
    """Run the CLI interface."""
    try:
        logger.info("Starting CLI interface...")

        # Initialize components
        config_manager = ConfigManager()
        event_bus = EventBus()
        risk_manager = RiskManager(config_manager)
        agent_manager = AgentManager()

        # Get exchange configuration
        exchange_config = config_manager.get_section('exchange')
        exchange_client = BlofinClient(
            api_key=exchange_config.get('api_key', ''),
            api_secret=exchange_config.get('api_secret', ''),
            passphrase=exchange_config.get('passphrase', ''),
            sandbox=exchange_config.get('sandbox', True)
        )

        # Initialize trading engine
        trading_engine = TradingEngine(
            exchange_client,
            risk_manager,
            event_bus,
            config_manager,
            agent_manager
        )

        logger.info("Trading system initialized successfully")

        # For CLI mode, we would typically run in a loop
        # For now, just demonstrate the initialization
        logger.info("CLI mode - trading engine ready")
        logger.info("Use Ctrl+C to exit")

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await trading_engine.stop()

    except Exception as e:
        logger.error(f"Error in CLI mode: {e}")
        return 1

    return 0


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'requests', 'aiohttp',
        'loguru', 'pandas_ta', 'sklearn'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run GUI interface
  %(prog)s --gui              # Run GUI interface
  %(prog)s --cli              # Run CLI interface
  %(prog)s --check-deps       # Check dependencies
  %(prog)s --debug            # Run with debug logging
        """
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Run GUI interface (default)"
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run CLI interface"
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else args.log_level
    setup_logging(log_level)

    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are installed")
            return 0
        else:
            return 1

    # Check dependencies before running
    if not check_dependencies():
        return 1

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Determine interface to run
    if args.cli:
        # Run CLI interface
        return asyncio.run(run_cli())
    else:
        # Run GUI interface (default)
        return run_gui()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)