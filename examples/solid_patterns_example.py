"""
Example demonstrating SOLID pattern improvements.
Shows how to use all the new patterns together.
"""

import asyncio
import pandas as pd
from decimal import Decimal
from typing import Dict, Any

# Import new SOLID improvements
from crypto_trading.utils.logging import create_logger, LoguruAdapter
from crypto_trading.strategies.ma_calculators import MACalculatorFactory
from crypto_trading.core.risk_validators import RiskValidatorChain
from crypto_trading.agents.agent_factory import AgentFactory, AgentBuilder, create_agent
from crypto_trading.utils.validators import (
    validate_ma_parameters,
    validate_rsi_parameters,
    ValidationError
)
from crypto_trading.core.order_decorators import OrderExecutorBuilder

# Import existing interfaces
from crypto_trading.core.interfaces import Order, OrderSide, OrderType
from crypto_trading.core.models import MarketData


# ====================================================================================
# Example 1: Using ILogger Interface
# ====================================================================================

def example_1_logger():
    """Demonstrate ILogger usage with dependency injection."""
    print("\n=== Example 1: ILogger Interface ===\n")

    # Create logger using factory
    logger = create_logger("MyTradingComponent")

    # Use logger - implementation is abstracted
    logger.info("Trading component initialized")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")

    # Easy to swap implementations for testing
    from crypto_trading.utils.logging import NullLogger
    test_logger = NullLogger()
    test_logger.info("This won't be logged - useful for testing")

    print("✓ Logger example completed")


# ====================================================================================
# Example 2: MA Calculation Strategy Pattern
# ====================================================================================

def example_2_ma_calculators():
    """Demonstrate MA calculation strategy pattern."""
    print("\n=== Example 2: MA Calculation Strategy ===\n")

    # Sample price data
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    # Get available MA types
    available_types = MACalculatorFactory.get_supported_types()
    print(f"Available MA types: {available_types}")

    # Calculate different MA types using strategy pattern
    for ma_type in ['sma', 'ema', 'wma']:
        calculator = MACalculatorFactory.create(ma_type)
        ma_values = calculator.calculate(prices, period=3)
        print(f"{ma_type.upper()}: {ma_values.iloc[-1]:.2f}")

    # Register custom MA type (example)
    from crypto_trading.strategies.ma_calculators import MACalculationStrategy

    class TripleEMACalculator(MACalculationStrategy):
        def calculate(self, data: pd.Series, period: int) -> pd.Series:
            # Triple exponential moving average
            ema1 = data.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3

        def get_name(self) -> str:
            return "tema"

    MACalculatorFactory.register('tema', TripleEMACalculator)
    tema = MACalculatorFactory.create('tema')
    tema_values = tema.calculate(prices, period=3)
    print(f"TEMA (custom): {tema_values.iloc[-1]:.2f}")

    print("✓ MA calculator example completed")


# ====================================================================================
# Example 3: Chain of Responsibility for Risk Validation
# ====================================================================================

def example_3_risk_validators():
    """Demonstrate Chain of Responsibility pattern for risk validation."""
    print("\n=== Example 3: Risk Validation Chain ===\n")

    from crypto_trading.core.risk_validators import (
        RiskValidatorChain,
        OrderBasicsValidator,
        PositionLimitValidator,
        RiskValidator
    )

    # Create a test order
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        amount=Decimal('0.1'),
        price=Decimal('50000')
    )

    # Create custom validator
    class CustomPriceValidator(RiskValidator):
        def _check(self, order, positions, context):
            max_price = context.get('max_price', Decimal('100000'))
            if order.price and order.price > max_price:
                self.logger.warning(f"Price {order.price} exceeds max {max_price}")
                return False
            return True

    # Build validation chain
    logger = create_logger("RiskValidation")
    chain = (RiskValidatorChain(logger)
             .add_validator(OrderBasicsValidator(logger))
             .add_validator(CustomPriceValidator(logger))
             .add_validator(PositionLimitValidator(logger))
             .build())

    # Validate order
    context = {
        'config': {
            'max_positions_per_symbol': 2,
            'max_total_positions': 10,
            'max_price': Decimal('100000')
        },
        'daily_losses': {}
    }

    is_valid = chain.validate(order, [], context)
    print(f"Order validation result: {is_valid}")
    print("✓ Risk validation example completed")


# ====================================================================================
# Example 4: Agent Factory Pattern
# ====================================================================================

def example_4_agent_factory():
    """Demonstrate agent factory pattern."""
    print("\n=== Example 4: Agent Factory ===\n")

    # Register agent types (normally done at startup)
    def create_mock_agent(config: Dict[str, Any]):
        """Mock agent creator for demonstration."""
        class MockAgent:
            def __init__(self, config):
                self.config = config
                self.name = config.get('name', 'MockAgent')

            def get_name(self):
                return self.name

            def get_parameters(self):
                return self.config

        return MockAgent(config)

    AgentFactory.register_agent_type('mock', create_mock_agent)

    # Create agent using factory
    agent1 = AgentFactory.create_agent('mock', {'name': 'TestAgent1', 'param': 'value'})
    print(f"Created agent: {agent1.get_name()}")
    print(f"Agent parameters: {agent1.get_parameters()}")

    # Get available agent types
    types = AgentFactory.get_available_types()
    print(f"Available agent types: {types}")

    print("✓ Agent factory example completed")


# ====================================================================================
# Example 5: Parameter Validators
# ====================================================================================

def example_5_parameter_validators():
    """Demonstrate parameter validation."""
    print("\n=== Example 5: Parameter Validators ===\n")

    # Validate MA parameters
    ma_params = {
        'fast_period': 10,
        'slow_period': 20,
        'ma_type': 'ema'
    }

    try:
        validate_ma_parameters(ma_params)
        print(f"✓ MA parameters valid: {ma_params}")
    except ValidationError as e:
        print(f"✗ MA validation failed: {e}")

    # Invalid parameters
    invalid_params = {
        'fast_period': 25,  # Greater than slow_period!
        'slow_period': 20,
        'ma_type': 'sma'
    }

    try:
        validate_ma_parameters(invalid_params)
        print("✗ Should have failed!")
    except ValidationError as e:
        print(f"✓ Correctly rejected invalid params: {e}")

    # Validate RSI parameters
    rsi_params = {'period': 14, 'overbought': 70, 'oversold': 30}
    try:
        validate_rsi_parameters(rsi_params)
        print(f"✓ RSI parameters valid: {rsi_params}")
    except ValidationError as e:
        print(f"✗ RSI validation failed: {e}")

    print("✓ Parameter validation example completed")


# ====================================================================================
# Example 6: Order Execution Decorators
# ====================================================================================

async def example_6_order_decorators():
    """Demonstrate order execution decorators."""
    print("\n=== Example 6: Order Execution Decorators ===\n")

    from crypto_trading.core.order_decorators import (
        LoggingOrderExecutorDecorator,
        MetricsOrderExecutorDecorator,
        ValidationOrderExecutorDecorator
    )
    from crypto_trading.core.interfaces import IOrderExecutor

    # Create mock base executor
    class MockOrderExecutor(IOrderExecutor):
        async def place_order(self, order: Order) -> Order:
            # Simulate order placement
            order.id = "ORDER123"
            from crypto_trading.core.models import OrderStatus
            order.status = OrderStatus.FILLED
            return order

        async def cancel_order(self, order_id: str) -> bool:
            return True

        async def get_order_status(self, order_id: str) -> Order:
            return Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=Decimal('0.1')
            )

    # Build decorated executor
    logger = create_logger("OrderExecution")
    base_executor = MockOrderExecutor()

    # Manual decoration
    executor = base_executor
    executor = ValidationOrderExecutorDecorator(executor, logger)
    executor = LoggingOrderExecutorDecorator(executor, logger)
    executor = MetricsOrderExecutorDecorator(executor, logger)

    # Use decorated executor
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        amount=Decimal('0.1'),
        price=Decimal('50000')
    )

    result = await executor.place_order(order)
    print(f"Order placed: {result.id}")

    # Get metrics
    if hasattr(executor, 'get_metrics'):
        metrics = executor.get_metrics()
        print(f"Metrics: {metrics}")

    # Using builder pattern
    from crypto_trading.core.order_decorators import OrderExecutorBuilder

    executor2 = (OrderExecutorBuilder(MockOrderExecutor())
                 .with_validation(logger)
                 .with_logging(logger)
                 .with_metrics(logger)
                 .build())

    result2 = await executor2.place_order(order)
    print(f"Order placed via builder: {result2.id}")

    print("✓ Order decorator example completed")


# ====================================================================================
# Example 7: Complete Integration
# ====================================================================================

async def example_7_complete_integration():
    """Demonstrate all patterns working together."""
    print("\n=== Example 7: Complete Integration ===\n")

    # 1. Set up logger
    logger = create_logger("TradingSystem")
    logger.info("Initializing trading system with SOLID patterns")

    # 2. Validate configuration
    config = {
        'fast_period': 10,
        'slow_period': 20,
        'ma_type': 'ema',
        'risk': {
            'max_position_size_pct': 0.1,
            'max_daily_loss_pct': 0.05
        }
    }

    try:
        validate_ma_parameters({
            'fast_period': config['fast_period'],
            'slow_period': config['slow_period'],
            'ma_type': config['ma_type']
        })
        logger.info("✓ Configuration validated")
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return

    # 3. Create MA calculator
    ma_calculator = MACalculatorFactory.create(config['ma_type'])
    logger.info(f"✓ Created {ma_calculator.get_name().upper()} calculator")

    # 4. Set up risk validation chain
    risk_chain = RiskValidatorChain.create_default_chain(logger)
    logger.info("✓ Risk validation chain created")

    # 5. Create decorated order executor
    from crypto_trading.core.order_decorators import OrderExecutorBuilder

    class MockExecutor:
        async def place_order(self, order):
            order.id = "ORDER_001"
            from crypto_trading.core.models import OrderStatus
            order.status = OrderStatus.FILLED
            return order

        async def cancel_order(self, order_id):
            return True

        async def get_order_status(self, order_id):
            return Order(symbol="BTC/USDT", side=OrderSide.BUY,
                        type=OrderType.LIMIT, amount=Decimal('0.1'))

    executor = (OrderExecutorBuilder(MockExecutor())
                .with_validation(logger)
                .with_logging(logger)
                .with_metrics(logger)
                .build())

    logger.info("✓ Order executor created with decorators")

    # 6. Simulate trading flow
    logger.info("Simulating trading flow...")

    # Calculate MA
    sample_prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    ma = ma_calculator.calculate(sample_prices, config['fast_period'])
    logger.info(f"MA calculated: {ma.iloc[-1]:.2f}")

    # Create order
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        amount=Decimal('0.1'),
        price=Decimal('50000')
    )

    # Validate with risk chain
    risk_context = {
        'config': config['risk'],
        'daily_losses': {}
    }

    if risk_chain.validate(order, [], risk_context):
        logger.info("✓ Order passed risk validation")

        # Execute order
        result = await executor.place_order(order)
        logger.info(f"✓ Order executed: {result.id}")
    else:
        logger.warning("✗ Order failed risk validation")

    print("\n✓ Complete integration example finished")
    print("\nAll SOLID pattern improvements demonstrated successfully!")


# ====================================================================================
# Main
# ====================================================================================

def main():
    """Run all examples."""
    print("=" * 80)
    print("SOLID Pattern Improvements - Examples")
    print("=" * 80)

    # Run synchronous examples
    example_1_logger()
    example_2_ma_calculators()
    example_3_risk_validators()
    example_4_agent_factory()
    example_5_parameter_validators()

    # Run async examples
    asyncio.run(example_6_order_decorators())
    asyncio.run(example_7_complete_integration())

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
