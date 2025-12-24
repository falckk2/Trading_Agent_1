# SOLID Principles Improvements

This document describes the SOLID principle improvements implemented in the codebase and how to use them.

## Overview

The following improvements have been implemented to enhance SOLID principles adherence:

1. **ILogger Interface and Adapters** - Dependency Inversion Principle
2. **MA Calculation Strategy Pattern** - Open/Closed Principle
3. **Chain of Responsibility for Risk Validators** - Open/Closed & Single Responsibility
4. **Agent Factory Pattern** - Open/Closed Principle
5. **Parameter Validators** - Single Responsibility Principle
6. **Order Execution Decorators** - Open/Closed & Single Responsibility

## 1. ILogger Interface and Logging Adapters

**Location**: `crypto_trading/utils/logging.py`

**Problem Solved**: Direct dependency on `loguru` library violates Dependency Inversion Principle.

**Solution**: Use `ILogger` interface with multiple implementations.

### Usage:

```python
from crypto_trading.utils.logging import create_logger, LoguruAdapter, StandardLibraryAdapter, NullLogger

# Create a Loguru logger (default)
logger = create_logger("MyComponent")

# Or explicitly choose adapter
logger = LoguruAdapter("MyComponent")
logger = StandardLibraryAdapter("my.module")
logger = NullLogger()  # For testing

# Use the logger
logger.info("This is an info message")
logger.error("This is an error message")
```

### Benefits:
- Easy to swap logging implementations
- Testable with `NullLogger`
- Follows Dependency Inversion Principle

---

## 2. MA Calculation Strategy Pattern

**Location**: `crypto_trading/strategies/ma_calculators.py`

**Problem Solved**: Hardcoded if/elif statements for different MA types violate Open/Closed Principle.

**Solution**: Strategy pattern with factory for MA calculation.

### Usage:

```python
from crypto_trading.strategies.ma_calculators import MACalculatorFactory, SMACalculator
import pandas as pd

# Get a calculator using the factory
calculator = MACalculatorFactory.create('sma')
ma_values = calculator.calculate(price_series, period=20)

# Or create directly
calculator = SMACalculator()
ma_values = calculator.calculate(price_series, period=20)

# Register custom MA type
class CustomMACalculator(MACalculationStrategy):
    def calculate(self, data: pd.Series, period: int) -> pd.Series:
        # Your custom logic here
        return data.rolling(period).mean()

    def get_name(self) -> str:
        return "custom"

MACalculatorFactory.register('custom', CustomMACalculator)

# Available types
types = MACalculatorFactory.get_supported_types()  # ['sma', 'ema', 'wma', 'hma']
```

### Updating MovingAverageStrategy:

**Before:**
```python
if ma_type == 'sma':
    df['ma_fast'] = self._calculate_sma(df['close'], fast_period)
elif ma_type == 'ema':
    df['ma_fast'] = self._calculate_ema(df['close'], fast_period)
```

**After:**
```python
from crypto_trading.strategies.ma_calculators import MACalculatorFactory

calculator = MACalculatorFactory.create(ma_type)
df['ma_fast'] = calculator.calculate(df['close'], fast_period)
df['ma_slow'] = calculator.calculate(df['close'], slow_period)
```

### Benefits:
- New MA types can be added without modifying existing code
- Each calculator has single responsibility
- Easy to test individual calculators

---

## 3. Chain of Responsibility for Risk Validators

**Location**: `crypto_trading/core/risk_validators.py`

**Problem Solved**: Multiple if statements in `RiskManager.validate_order()` make it hard to extend.

**Solution**: Chain of Responsibility pattern for risk validation.

### Usage:

```python
from crypto_trading.core.risk_validators import (
    RiskValidatorChain,
    OrderBasicsValidator,
    DailyLossLimitValidator,
    PositionLimitValidator
)

# Create a chain with standard validators
chain = RiskValidatorChain.create_default_chain(logger)

# Or build custom chain
chain = (RiskValidatorChain(logger)
         .add_validator(OrderBasicsValidator())
         .add_validator(DailyLossLimitValidator())
         .add_validator(CustomValidator())  # Your custom validator
         .build())

# Use the chain
context = {
    'config': config_dict,
    'daily_losses': daily_losses_dict
}

is_valid = chain.validate(order, positions, context)
```

### Creating Custom Validator:

```python
from crypto_trading.core.risk_validators import RiskValidator

class MyCustomValidator(RiskValidator):
    def _check(self, order: Order, positions: List[Position], context: Dict) -> bool:
        # Your validation logic
        if some_condition:
            self.logger.warning("Custom validation failed")
            return False
        return True
```

### Updating RiskManager:

**Before:**
```python
def validate_order(self, order, positions):
    if not self._validate_order_basics(order):
        return False
    if not self._check_daily_loss_limit():
        return False
    # More checks...
```

**After:**
```python
from crypto_trading.core.risk_validators import RiskValidatorChain

def __init__(self, config_manager):
    self.validator_chain = RiskValidatorChain.create_default_chain()
    # ...

def validate_order(self, order, positions):
    context = {
        'config': self.default_config,
        'daily_losses': self.daily_losses
    }
    return self.validator_chain.validate(order, positions, context)
```

### Benefits:
- New validators can be added without modifying RiskManager
- Each validator has single responsibility
- Validators can be reordered or removed easily

---

## 4. Agent Factory Pattern

**Location**: `crypto_trading/agents/agent_factory.py`

**Problem Solved**: Agent creation scattered across codebase, hard to extend with new agent types.

**Solution**: Factory pattern with registration system.

### Usage:

```python
from crypto_trading.agents.agent_factory import AgentFactory, AgentBuilder, create_agent

# Register agent types (do this once at startup)
def create_rsi_agent(config):
    from crypto_trading.agents.technical.rsi_agent import RSIAgent
    return RSIAgent(**config)

AgentFactory.register_agent_type('rsi', create_rsi_agent)

# Create agents using factory
agent = AgentFactory.create_agent('rsi', {'period': 14, 'overbought': 70})

# Or use convenience function
agent = create_agent('rsi', period=14, overbought=70)

# Or use builder for complex construction
agent = (AgentBuilder('rsi')
         .with_strategy_params({'period': 14})
         .with_exchange(exchange_client)
         .with_risk_manager(risk_manager)
         .with_logger(logger)
         .build())

# List available types
types = AgentFactory.get_available_types()
```

### Benefits:
- Centralized agent creation
- Easy to add new agent types
- Builder pattern for complex agent construction

---

## 5. Parameter Validators

**Location**: `crypto_trading/utils/validators.py`

**Problem Solved**: Parameter validation logic scattered and duplicated across agents.

**Solution**: Reusable validator classes with single responsibility.

### Usage:

```python
from crypto_trading.utils.validators import (
    ValidationError,
    RangeValidator,
    ChoiceValidator,
    PositiveValidator,
    validate_ma_parameters,
    validate_rsi_parameters
)

# Use convenience functions
try:
    validate_ma_parameters({'fast_period': 10, 'slow_period': 20, 'ma_type': 'sma'})
    validate_rsi_parameters({'period': 14, 'overbought': 70, 'oversold': 30})
except ValidationError as e:
    print(f"Validation failed: {e}")

# Or use validators directly
validator = RangeValidator('period', min_value=2, max_value=100)
validator.validate(14)  # OK

validator = ChoiceValidator('ma_type', ['sma', 'ema', 'wma', 'hma'], case_sensitive=False)
validator.validate('SMA')  # OK

# Compose validators
validators = [
    PositiveValidator('amount'),
    RangeValidator('amount', min_value=0.001, max_value=1000)
]

for validator in validators:
    validator.validate(amount_value)
```

### Creating Custom Validator:

```python
from crypto_trading.utils.validators import ParameterValidator, ValidationError

class MinGreaterThanMaxValidator(ParameterValidator):
    def __init__(self, min_param, max_param):
        super().__init__(f"{min_param}_vs_{max_param}")
        self.min_param = min_param
        self.max_param = max_param

    def validate(self, params: dict) -> bool:
        if params[self.min_param] >= params[self.max_param]:
            raise ValidationError(
                f"{self.min_param} must be less than {self.max_param}"
            )
        return True
```

### Benefits:
- Reusable validation logic
- Each validator has single responsibility
- Easy to compose multiple validators

---

## 6. Order Execution Decorators

**Location**: `crypto_trading/core/order_decorators.py`

**Problem Solved**: Cross-cutting concerns (logging, retry, metrics) mixed with order execution logic.

**Solution**: Decorator pattern for adding behaviors to order execution.

### Usage:

```python
from crypto_trading.core.order_decorators import (
    OrderExecutorBuilder,
    LoggingOrderExecutorDecorator,
    RetryOrderExecutorDecorator,
    MetricsOrderExecutorDecorator
)

# Get base order executor (e.g., from OrderExecutor class)
base_executor = OrderExecutor(exchange_client, risk_manager)

# Build decorated executor with fluent interface
executor = (OrderExecutorBuilder(base_executor)
            .with_validation()     # Add validation
            .with_logging(logger)  # Add logging
            .with_retry(max_retries=3, initial_delay=1.0)  # Add retry logic
            .with_metrics(logger)  # Add metrics collection
            .with_rate_limiting(max_orders_per_second=5.0)  # Add rate limiting
            .build())

# Use the decorated executor
order = await executor.place_order(order)

# Get metrics (if metrics decorator was added)
if hasattr(executor, 'get_metrics'):
    metrics = executor.get_metrics()
    print(f"Success rate: {metrics['success_rate']:.2%}")
```

### Manual Decoration:

```python
# Wrap executors manually for more control
executor = base_executor
executor = ValidationOrderExecutorDecorator(executor, logger)
executor = LoggingOrderExecutorDecorator(executor, logger)
executor = RetryOrderExecutorDecorator(executor, max_retries=3)
executor = MetricsOrderExecutorDecorator(executor, logger)

# Now executor has all behaviors
await executor.place_order(order)
```

### Benefits:
- Separation of concerns
- Behaviors can be added/removed without modifying core logic
- Easy to test each decorator independently
- Flexible composition

---

## Integration Guide

### Step 1: Update Imports

Add the new modules to your imports:

```python
# Logging
from crypto_trading.utils.logging import create_logger

# MA Calculators
from crypto_trading.strategies.ma_calculators import MACalculatorFactory

# Risk Validators
from crypto_trading.core.risk_validators import RiskValidatorChain

# Agent Factory
from crypto_trading.agents.agent_factory import AgentFactory, create_agent

# Validators
from crypto_trading.utils.validators import validate_ma_parameters, ValidationError

# Order Decorators
from crypto_trading.core.order_decorators import OrderExecutorBuilder
```

### Step 2: Update Logger Usage

Replace direct logger imports with ILogger interface:

```python
# Old
from loguru import logger
self.logger = logger

# New
from crypto_trading.utils.logging import create_logger
self.logger = create_logger(self.__class__.__name__)
```

### Step 3: Update MA Calculations

Replace hardcoded MA calculations with strategy pattern:

```python
# Old
if ma_type == 'sma':
    ma = df['close'].rolling(period).mean()
elif ma_type == 'ema':
    ma = df['close'].ewm(span=period).mean()

# New
from crypto_trading.strategies.ma_calculators import MACalculatorFactory
calculator = MACalculatorFactory.create(ma_type)
ma = calculator.calculate(df['close'], period)
```

### Step 4: Update Risk Validation

Replace if-chain validation with Chain of Responsibility:

```python
# In RiskManager.__init__
from crypto_trading.core.risk_validators import RiskValidatorChain
self.validator_chain = RiskValidatorChain.create_default_chain(self.logger)

# In validate_order method
context = {
    'config': self.default_config,
    'daily_losses': self.daily_losses
}
return self.validator_chain.validate(order, positions, context)
```

### Step 5: Use Agent Factory

Register and use agent factory for agent creation:

```python
# At application startup
from crypto_trading.agents.agent_factory import AgentFactory

def create_rsi_agent(config):
    from crypto_trading.agents.technical.rsi_agent import RSIAgent
    return RSIAgent(**config)

AgentFactory.register_agent_type('rsi', create_rsi_agent)
# Register other types...

# Create agents
agent = AgentFactory.create_agent('rsi', {'period': 14})
```

### Step 6: Add Order Execution Decorators

Wrap order executor with decorators:

```python
from crypto_trading.core.order_decorators import OrderExecutorBuilder

# In TradingEngine or wherever OrderExecutor is created
base_executor = OrderExecutor(exchange_client, risk_manager)

executor = (OrderExecutorBuilder(base_executor)
            .with_logging(logger)
            .with_retry(max_retries=3)
            .with_metrics(logger)
            .build())

# Use decorated executor
self.order_executor = executor
```

---

## Testing

All new components are designed to be easily testable:

```python
# Test with NullLogger
from crypto_trading.utils.logging import NullLogger
component = MyComponent(logger=NullLogger())

# Test validators
from crypto_trading.utils.validators import RangeValidator
validator = RangeValidator('test', min_value=0, max_value=100)
assert validator.validate(50) == True

# Test MA calculators
from crypto_trading.strategies.ma_calculators import SMACalculator
calculator = SMACalculator()
result = calculator.calculate(test_data, period=10)

# Test order decorators
from crypto_trading.core.order_decorators import MetricsOrderExecutorDecorator
executor = MetricsOrderExecutorDecorator(mock_executor)
metrics = executor.get_metrics()
```

---

## Backward Compatibility

All improvements maintain backward compatibility:

- Existing code continues to work without changes
- New patterns can be adopted gradually
- Old and new approaches can coexist during migration

---

## Summary

These improvements enhance the codebase's adherence to SOLID principles:

- **S**ingle Responsibility: Each validator, calculator, and decorator has one job
- **O**pen/Closed: New types can be added via registration without modifying existing code
- **L**iskov Substitution: All implementations properly substitute their interfaces
- **I**nterface Segregation: ILogger, validators, and calculators are focused interfaces
- **D**ependency Inversion: Components depend on abstractions (ILogger, IOrderExecutor, etc.)

The result is a more maintainable, testable, and extensible trading system.
