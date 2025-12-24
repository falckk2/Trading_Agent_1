# Trading Agent System - Architecture Documentation
## UML Diagrams and System Design

**Project:** Crypto Trading Agent System
**Date:** 2025-12-18
**Architecture Style:** Layered Architecture with SOLID Principles

---

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Component Diagram](#high-level-component-diagram)
3. [Core Layer Class Diagram](#core-layer-class-diagram)
4. [Agent Layer Class Diagram](#agent-layer-class-diagram)
5. [Exchange Layer Class Diagram](#exchange-layer-class-diagram)
6. [Data Layer Class Diagram](#data-layer-class-diagram)
7. [Sequence Diagrams](#sequence-diagrams)
8. [Design Patterns Applied](#design-patterns-applied)
9. [SOLID Principles Analysis](#solid-principles-analysis)

---

## System Overview

The Crypto Trading Agent System is a comprehensive platform for algorithmic cryptocurrency trading. It supports multiple trading strategies (technical analysis and machine learning), multiple exchanges, risk management, and both paper and live trading modes.

### Key Architectural Characteristics:
- **Layered Architecture**: Clear separation between Core, Agents, Exchange, Data, and UI layers
- **Interface Segregation**: Small, focused interfaces for each responsibility
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Event-Driven**: Asynchronous event bus for loose coupling
- **Strategy Pattern**: Pluggable trading strategies and execution modes
- **Factory Pattern**: Dynamic agent and strategy creation

---

## High-Level Component Diagram

```plantuml
@startuml
package "Trading Agent System" {

    [GUI Layer] as GUI
    [Core Engine] as Core
    [Agent Layer] as Agents
    [Exchange Layer] as Exchange
    [Data Layer] as Data
    [Security Layer] as Security

    GUI --> Core : uses
    Core --> Agents : manages
    Core --> Exchange : executes orders
    Core --> Data : retrieves market data
    Agents --> Data : analyzes
    Exchange --> Security : authenticates

    package "Core Components" {
        [TradingEngine]
        [OrderExecutor]
        [RiskManager]
        [EventBus]
        [ConfigManager]
        [AccountStateManager]
    }

    package "Agent Types" {
        [Technical Agents]
        [ML Agents]
        [DL Agents]
    }

    package "Data Components" {
        [DataManager]
        [HistoricalProvider]
        [RealtimeFeed]
        [DataStorage]
        [DataPreprocessor]
    }

    package "Exchange Components" {
        [BaseExchange]
        [BlofinExchange]
        [TypeConverter]
    }
}

database "PostgreSQL/SQLite" as DB
cloud "Exchange APIs" as ExchangeAPI

Data --> DB : persists
Exchange --> ExchangeAPI : connects
@enduml
```

**Component Responsibilities:**

1. **GUI Layer**: PyQt-based user interface for monitoring and control
2. **Core Engine**: Orchestrates all trading activities, manages lifecycle
3. **Agent Layer**: Implements trading strategies and signal generation
4. **Exchange Layer**: Handles exchange connectivity and order execution
5. **Data Layer**: Manages historical and real-time market data
6. **Security Layer**: Manages credentials and API keys securely

---

## Core Layer Class Diagram

```plantuml
@startuml
title Core Layer - Interfaces and Main Components

' Core Interfaces
interface IExchangeConnection {
    + connect(): bool
    + disconnect(): None
    + is_connected(): bool
}

interface IMarketDataProvider {
    + get_market_data(symbol: str): MarketData
    + get_historical_data(symbol, timeframe, start, end): List[MarketData]
}

interface IOrderExecutor {
    + place_order(order: Order): Order
    + cancel_order(order_id: str): bool
    + get_order_status(order_id: str): Order
}

interface IAccountDataProvider {
    + get_positions(): List[Position]
    + get_balance(): Dict[str, Decimal]
}

interface IExchangeClient {
}

IExchangeClient --|> IExchangeConnection
IExchangeClient --|> IMarketDataProvider
IExchangeClient --|> IOrderExecutor
IExchangeClient --|> IAccountDataProvider

interface IRiskManager {
    + validate_order(order: Order, positions: List[Position]): bool
    + calculate_position_size(signal: TradingSignal, balance: Dict): Decimal
}

interface ISignalGenerator {
    + analyze(market_data: List[MarketData]): TradingSignal
}

interface IConfigurableAgent {
    + initialize(config: Dict): None
    + get_parameters(): Dict
}

interface ITradingAgent {
    + get_name(): str
    + get_description(): str
}

ITradingAgent --|> ISignalGenerator
ITradingAgent --|> IConfigurableAgent

interface IEventBus {
    + subscribe(event_type: EventType, callback: Callable): None
    + unsubscribe(event_type: EventType, callback: Callable): None
    + publish(event: Event): None
}

interface IConfigManager {
    + get(key: str, default: Any): Any
    + set(key: str, value: Any): None
    + save(): None
    + load(): None
}

interface IAgentManager {
    + register_agent(agent: ITradingAgent): None
    + get_agent(name: str): ITradingAgent
    + get_active_agent(): ITradingAgent
    + set_active_agent(name: str): bool
    + list_agents(): List[str]
}

' Core Implementation Classes
class TradingEngine {
    - exchange_client: IExchangeClient
    - risk_manager: IRiskManager
    - event_bus: IEventBus
    - config_manager: IConfigManager
    - agent_manager: IAgentManager
    - order_executor: OrderExecutor
    - account_state_manager: AccountStateManager
    - _is_running: bool
    - _trading_enabled: bool

    + start(): None
    + stop(): None
    + enable_trading(): None
    + disable_trading(): None
    + get_status(): Dict
    + get_positions(): List[Position]
    + get_balance(): Dict
    - _run_trading_loop(): None
    - _execute_trading_cycle(): None
    - _execute_signal(signal: TradingSignal): None
}

class OrderExecutor {
    - exchange_client: IExchangeClient
    - risk_manager: IRiskManager

    + execute_signal(signal, balance, positions): Order
    - _create_order_from_signal(signal, size): Order
}

abstract class SignalExecutionStrategy {
    + {abstract} execute(signal, executor, balance, positions): Order
}

class ImmediateExecutionStrategy {
    + execute(signal, executor, balance, positions): Order
}

class ConservativeExecutionStrategy {
    - min_confidence: float
    + execute(signal, executor, balance, positions): Order
}

SignalExecutionStrategy <|-- ImmediateExecutionStrategy
SignalExecutionStrategy <|-- ConservativeExecutionStrategy

class RiskManager {
    - config_manager: IConfigManager
    - daily_losses: Dict[str, Decimal]
    - default_config: Dict

    + validate_order(order, positions): bool
    + calculate_position_size(signal, balance): Decimal
    + get_risk_metrics(positions): Dict
    + check_stop_loss(position): bool
    + check_take_profit(position): bool
    - _check_daily_loss_limit(): bool
    - _check_position_limits(order, positions): bool
    - _check_exposure_limits(order, positions): bool
}

RiskManager ..|> IRiskManager

class EventBus {
    - _subscribers: Dict[EventType, List[Callable]]
    - _event_history: List[Event]
    - _is_running: bool

    + subscribe(event_type, callback): None
    + unsubscribe(event_type, callback): None
    + publish(event: Event): None
    + get_event_history(event_type, since, limit): List[Event]
    + get_stats(): Dict
}

EventBus ..|> IEventBus

class AccountStateManager {
    - exchange_client: IExchangeClient
    - event_bus: IEventBus
    - _balance: Dict[str, Decimal]
    - _positions: List[Position]
    - _active_orders: List[Order]

    + update_account_info(): None
    + get_balance(): Dict
    + get_positions(): List[Position]
    + add_active_order(order): None
    + clear_active_orders(): None
}

' Core Data Models
class Order <<dataclass>> {
    + id: str
    + symbol: str
    + side: OrderSide
    + type: OrderType
    + amount: Decimal
    + price: Decimal
    + status: OrderStatus
    + timestamp: datetime
    + filled_amount: Decimal
    + is_filled: bool
    + is_active: bool
    + remaining_amount: Decimal
}

class Position <<dataclass>> {
    + symbol: str
    + side: OrderSide
    + amount: Decimal
    + entry_price: Decimal
    + current_price: Decimal
    + pnl: Decimal
    + timestamp: datetime
    + market_value: Decimal
    + unrealized_pnl: Decimal
    + total_pnl: Decimal
    + update_price(new_price): None
}

class TradingSignal <<dataclass>> {
    + symbol: str
    + action: OrderSide
    + confidence: float
    + price: Decimal
    + amount: Decimal
    + timestamp: datetime
    + metadata: Dict
    + is_actionable: bool
}

class MarketData <<dataclass>> {
    + symbol: str
    + timestamp: datetime
    + open: Decimal
    + high: Decimal
    + low: Decimal
    + close: Decimal
    + volume: Decimal
    + bid: Decimal
    + ask: Decimal
    + ohlc: tuple
}

enum OrderType {
    MARKET
    LIMIT
    STOP
    STOP_LIMIT
}

enum OrderSide {
    BUY
    SELL
}

enum OrderStatus {
    PENDING
    OPEN
    FILLED
    PARTIALLY_FILLED
    CANCELLED
    REJECTED
}

enum EventType {
    ORDER_FILLED
    ORDER_CANCELLED
    SIGNAL_GENERATED
    ERROR_OCCURRED
}

' Relationships
TradingEngine --> IExchangeClient : uses
TradingEngine --> IRiskManager : uses
TradingEngine --> IEventBus : uses
TradingEngine --> IConfigManager : uses
TradingEngine --> IAgentManager : uses
TradingEngine --> OrderExecutor : uses
TradingEngine --> AccountStateManager : uses

OrderExecutor --> IExchangeClient : uses
OrderExecutor --> IRiskManager : uses
OrderExecutor --> SignalExecutionStrategy : uses

TradingEngine ..> TradingSignal : receives
OrderExecutor ..> Order : creates
OrderExecutor ..> TradingSignal : processes

@enduml
```

**Key Design Decisions:**

1. **Interface Segregation**: Separated `IExchangeClient` into smaller focused interfaces
2. **Strategy Pattern**: `SignalExecutionStrategy` allows different execution approaches
3. **Single Responsibility**: Each class has one clear purpose
4. **Dependency Inversion**: All dependencies are on interfaces, not concrete classes

---

## Agent Layer Class Diagram

```plantuml
@startuml
title Agent Layer - Trading Strategies and Agents

' Base Interfaces
interface ITradingAgent {
    + analyze(market_data): TradingSignal
    + initialize(config): None
    + get_name(): str
    + get_description(): str
    + get_parameters(): Dict
}

interface IStrategy {
    + analyze(market_data): TradingSignal
    + get_parameters(): Dict
    + set_parameters(parameters): None
    + validate_signal(signal): bool
}

' Base Agent Implementation
abstract class BaseAgent {
    # name: str
    # description: str
    # config: Dict
    # is_initialized: bool

    + initialize(config): None
    + get_name(): str
    + get_description(): str
    + get_parameters(): Dict
    # _validate_config(): None
    # _create_signal(symbol, action, confidence, ...): TradingSignal
    # _ensure_initialized(): None
    # _validate_market_data(data): None
    # _get_minimum_data_points(): int
    # _calculate_confidence(strength, volatility): float
}

BaseAgent ..|> ITradingAgent

' Strategy Base Classes
abstract class TechnicalStrategy {
    # parameters: Dict
    # name: str
    # min_data_points: int
    # signal_threshold: float

    + analyze(market_data): TradingSignal
    + get_parameters(): Dict
    + set_parameters(parameters): None
    + validate_signal(signal): bool
    # {abstract} _calculate_indicators(df): DataFrame
    # {abstract} _generate_signal(df): Dict
    # _market_data_to_dataframe(data): DataFrame
    # _create_signal(current_data, signal_info): TradingSignal
    # _calculate_sma(prices, period): Series
    # _calculate_ema(prices, period): Series
    # _calculate_rsi(prices, period): Series
    # _calculate_macd(prices, fast, slow, signal): Tuple
    # _calculate_bollinger_bands(prices, period, std): Tuple
}

TechnicalStrategy ..|> IStrategy

abstract class MLStrategy {
    # parameters: Dict
    # model: Any
    # scaler: Any
    # is_trained: bool
    # lookback_window: int
    # feature_columns: List[str]

    + analyze(market_data): TradingSignal
    + train_model(training_data, retrain): Dict
    + save_model(filepath): None
    + load_model(filepath): None
    + needs_retraining(): bool
    # {abstract} _create_model(): Any
    # {abstract} _train_model_impl(X_train, y_train, X_val, y_val): Dict
    # {abstract} _predict(features): float
    # {abstract} _evaluate_model(X, y): float
    # _prepare_features(market_data): ndarray
    # _prediction_to_signal(current_data, prediction, features): TradingSignal
}

MLStrategy ..|> IStrategy

' Technical Analysis Agents
class RSIAgent {
    - strategy: TechnicalStrategy
    + analyze(market_data): TradingSignal
}

class MACDAgent {
    - strategy: TechnicalStrategy
    + analyze(market_data): TradingSignal
}

class MovingAverageAgent {
    - strategy: TechnicalStrategy
    + analyze(market_data): TradingSignal
}

class BollingerBandsAgent {
    - strategy: TechnicalStrategy
    + analyze(market_data): TradingSignal
}

BaseAgent <|-- RSIAgent
BaseAgent <|-- MACDAgent
BaseAgent <|-- MovingAverageAgent
BaseAgent <|-- BollingerBandsAgent

RSIAgent --> TechnicalStrategy : uses
MACDAgent --> TechnicalStrategy : uses
MovingAverageAgent --> TechnicalStrategy : uses
BollingerBandsAgent --> TechnicalStrategy : uses

' Machine Learning Agents
class RandomForestAgent {
    - strategy: MLStrategy
    + analyze(market_data): TradingSignal
}

class LSTMAgent {
    - strategy: MLStrategy
    + analyze(market_data): TradingSignal
}

BaseAgent <|-- RandomForestAgent
BaseAgent <|-- LSTMAgent

RandomForestAgent --> MLStrategy : uses
LSTMAgent --> MLStrategy : uses

' Specific Strategy Implementations
class RandomForestStrategy {
    - n_estimators: int
    - max_depth: int
    - model: RandomForestClassifier

    - _create_model(): RandomForestClassifier
    - _train_model_impl(X_train, y_train, X_val, y_val): Dict
    - _predict(features): float
    - _evaluate_model(X, y): float
}

MLStrategy <|-- RandomForestStrategy

class LSTMStrategy {
    - hidden_size: int
    - num_layers: int
    - model: Sequential

    - _create_model(): Sequential
    - _train_model_impl(X_train, y_train, X_val, y_val): Dict
    - _predict(features): float
    - _evaluate_model(X, y): float
}

MLStrategy <|-- LSTMStrategy

' Agent Management
class AgentManager {
    - agents: Dict[str, ITradingAgent]
    - active_agent: str

    + register_agent(agent): None
    + get_agent(name): ITradingAgent
    + get_active_agent(): ITradingAgent
    + set_active_agent(name): bool
    + list_agents(): List[str]
}

AgentManager --> ITradingAgent : manages

class AgentFactory {
    - {static} _agent_types: Dict[str, Callable]

    + {static} register_agent_type(type, creator): None
    + {static} create_agent(type, config): ITradingAgent
    + {static} get_available_types(): List[str]
}

AgentFactory ..> ITradingAgent : creates

class AgentBuilder {
    - agent_type: str
    - _config: Dict
    - _strategy: IStrategy
    - _exchange: IExchangeClient
    - _risk_manager: IRiskManager

    + with_config(config): AgentBuilder
    + with_strategy(strategy): AgentBuilder
    + with_exchange(exchange): AgentBuilder
    + with_risk_manager(risk_manager): AgentBuilder
    + build(): ITradingAgent
}

AgentBuilder --> AgentFactory : uses
AgentBuilder ..> ITradingAgent : builds

note right of MLStrategy
  Strategy Pattern:
  - Encapsulates ML algorithms
  - Template Method for training
  - Easily extensible for new models
end note

note right of TechnicalStrategy
  Template Method Pattern:
  - Common flow in analyze()
  - Subclasses implement indicators
  - Reusable indicator calculations
end note

@enduml
```

**Agent Layer Architecture:**

1. **Base Agent**: Provides common functionality for all agents
2. **Strategy Separation**: Agents delegate to strategy objects (Strategy Pattern)
3. **Technical vs ML**: Clear separation between technical and ML approaches
4. **Factory Creation**: AgentFactory for dynamic agent instantiation
5. **Builder Pattern**: AgentBuilder for complex agent configuration

---

## Exchange Layer Class Diagram

```plantuml
@startuml
title Exchange Layer - Exchange Integration

interface IExchangeClient {
    + connect(): bool
    + disconnect(): None
    + is_connected(): bool
    + get_market_data(symbol): MarketData
    + get_historical_data(symbol, timeframe, start, end): List[MarketData]
    + place_order(order): Order
    + cancel_order(order_id): bool
    + get_order_status(order_id): Order
    + get_positions(): List[Position]
    + get_balance(): Dict[str, Decimal]
}

abstract class BaseExchange {
    # api_key: str
    # api_secret: str
    # sandbox: bool
    # _connected: bool
    # rate_limit_delay: float
    # timeout: int
    # max_retries: int
    # _market_data_cache: Dict

    + connect(): bool
    + disconnect(): None
    + is_connected(): bool
    + get_balance(): Dict[str, Decimal]
    + place_order(order): Order
    + cancel_order(order_id): bool
    + get_order_status(order_id): Order
    + get_market_data(symbol): MarketData
    + get_historical_data(symbol, timeframe, start, end): List[MarketData]
    + get_positions(): List[Position]

    # {abstract} _authenticate(): None
    # {abstract} _initialize_connection(): None
    # {abstract} _cleanup_connection(): None
    # {abstract} _get_balance_impl(): Dict
    # {abstract} _place_order_impl(order): str
    # {abstract} _cancel_order_impl(order_id): bool
    # {abstract} _get_order_status_impl(order_id): Order
    # {abstract} _get_market_data_impl(symbol): MarketData
    # {abstract} _get_historical_data_impl(symbol, timeframe, start, end, limit): List[MarketData]
    # {abstract} _get_positions_impl(): List[Position]

    # _ensure_connected(): None
    # _validate_order(order): None
    # _is_cache_valid(symbol): bool
    # _update_cache(symbol, data): None
    # _rate_limit(): None
    # _retry_request(func, *args, **kwargs): Any
}

BaseExchange ..|> IExchangeClient

class BlofinExchange {
    - client: BlofinClient
    - instrument_cache: Dict

    # _authenticate(): None
    # _initialize_connection(): None
    # _cleanup_connection(): None
    # _get_balance_impl(): Dict
    # _place_order_impl(order): str
    # _cancel_order_impl(order_id): bool
    # _get_order_status_impl(order_id): Order
    # _get_market_data_impl(symbol): MarketData
    # _get_historical_data_impl(symbol, timeframe, start, end, limit): List[MarketData]
    # _get_positions_impl(): List[Position]
    - _load_instruments(): None
    - _normalize_symbol(symbol): str
}

BaseExchange <|-- BlofinExchange

class BlofinClient {
    - api_key: str
    - api_secret: str
    - passphrase: str
    - base_url: str

    + get_account_balance(): Dict
    + place_order(params): Dict
    + cancel_order(order_id): Dict
    + get_order(order_id): Dict
    + get_ticker(symbol): Dict
    + get_candles(symbol, timeframe, start, end): List
    + get_positions(): List
    - _sign_request(method, endpoint, params): str
    - _request(method, endpoint, params): Dict
}

BlofinExchange --> BlofinClient : uses

class TypeConverter {
    + {static} to_market_data(exchange_data, symbol): MarketData
    + {static} to_order(exchange_order): Order
    + {static} to_position(exchange_position): Position
    + {static} from_order(order): Dict
    + {static} to_order_type(exchange_type): OrderType
    + {static} to_order_side(exchange_side): OrderSide
    + {static} to_order_status(exchange_status): OrderStatus
    + {static} from_order_type(order_type): str
    + {static} from_order_side(side): str
}

BlofinExchange --> TypeConverter : uses

note right of BaseExchange
  Template Method Pattern:
  - Public methods define workflow
  - Protected abstract methods for
    exchange-specific implementation
  - Common functionality shared
  - Caching and rate limiting built-in
end note

note right of TypeConverter
  Adapter Pattern:
  - Converts exchange-specific formats
  - To/from domain models
  - Isolates exchange API changes
  - Type-safe conversions
end note

@enduml
```

**Exchange Layer Features:**

1. **Template Method**: BaseExchange defines common workflow
2. **Adapter Pattern**: TypeConverter normalizes exchange data
3. **Caching**: Built-in market data caching with expiry
4. **Rate Limiting**: Automatic request throttling
5. **Retry Logic**: Exponential backoff for failed requests
6. **Open/Closed**: New exchanges extend BaseExchange without modification

---

## Data Layer Class Diagram

```plantuml
@startuml
title Data Layer - Market Data Management

interface IDataProvider {
    + get_historical_data(symbol, timeframe, start, end): List[MarketData]
    + get_realtime_data(symbol): MarketData
    + get_data_range(symbol): Dict[str, datetime]
}

class DataManager {
    - historical_provider: HistoricalDataProvider
    - realtime_feed: RealtimeFeed
    - preprocessor: DataPreprocessor
    - storage: DataStorage
    - _subscriptions: Dict[str, List[Callable]]
    - _data_cache: Dict[str, List[MarketData]]

    + initialize(): None
    + shutdown(): None
    + get_historical_data(symbol, timeframe, start, end, use_cache): List[MarketData]
    + get_realtime_data(symbol): MarketData
    + subscribe_to_data(symbols, callback): None
    + unsubscribe_from_data(symbols): None
    + get_cached_data(symbol, limit): List[MarketData]
    + preload_data(symbols, timeframe, days_back): None
    + get_data_statistics(symbol): Dict
    + cleanup_old_data(days_to_keep): None
    - _data_callback(symbol, data): None
    - _update_cache(symbol, data): None
}

DataManager ..|> IDataProvider

class HistoricalDataProvider {
    - exchange_client: IExchangeClient
    - cache: Dict[str, List[MarketData]]

    + initialize(): None
    + shutdown(): None
    + get_historical_data(symbol, timeframe, start, end): List[MarketData]
    + get_data_range(symbol): Dict[str, datetime]
    - _fetch_from_exchange(symbol, timeframe, start, end): List[MarketData]
    - _cache_data(symbol, data): None
}

class RealtimeFeed {
    - exchange_client: IExchangeClient
    - _subscriptions: Dict[str, List[Callable]]
    - _websocket_connection: Any
    - _is_running: bool

    + initialize(): None
    + shutdown(): None
    + subscribe(symbol, callback): None
    + unsubscribe(symbol): None
    + get_current_data(symbol): MarketData
    - _start_websocket(): None
    - _handle_message(message): None
    - _notify_subscribers(symbol, data): None
}

class DataPreprocessor {
    + process_historical_data(data): List[MarketData]
    + process_realtime_data(data): MarketData
    + create_features_for_ml(data): DataFrame
    + normalize_data(data): List[MarketData]
    + handle_missing_values(data): List[MarketData]
    + remove_outliers(data, threshold): List[MarketData]
    + calculate_technical_indicators(df): DataFrame
}

class DataStorage {
    - db_connection: Any
    - db_path: str

    + initialize(): None
    + shutdown(): None
    + store_historical_data(data): None
    + store_realtime_data(data): None
    + get_historical_data(symbol, timeframe, start, end): List[MarketData]
    + get_data_statistics(symbol): Dict
    + cleanup_old_data(cutoff_date): None
    - _create_tables(): None
    - _insert_market_data(data): None
    - _query_market_data(symbol, timeframe, start, end): List[MarketData]
}

class FeatureEngineer {
    + create_price_features(df): DataFrame
    + create_volume_features(df): DataFrame
    + create_momentum_features(df): DataFrame
    + create_volatility_features(df): DataFrame
    + create_trend_features(df): DataFrame
    + create_lagged_features(df, lags): DataFrame
    + create_rolling_statistics(df, windows): DataFrame
}

' Relationships
DataManager --> HistoricalDataProvider : uses
DataManager --> RealtimeFeed : uses
DataManager --> DataPreprocessor : uses
DataManager --> DataStorage : uses

DataPreprocessor --> FeatureEngineer : uses

HistoricalDataProvider --> IExchangeClient : uses
RealtimeFeed --> IExchangeClient : uses

' Storage implementations
class SQLiteStorage {
    + _create_tables(): None
    + _insert_market_data(data): None
}

class PostgreSQLStorage {
    + _create_tables(): None
    + _insert_market_data(data): None
}

DataStorage <|-- SQLiteStorage
DataStorage <|-- PostgreSQLStorage

note right of DataManager
  Facade Pattern:
  - Unified interface for all data operations
  - Coordinates multiple data sources
  - Manages subscriptions and caching
  - Simplifies complex subsystem
end note

note right of DataPreprocessor
  Single Responsibility:
  - Only handles data transformation
  - Separate concerns for cleaning
  - Feature engineering delegated
  - Reusable preprocessing pipeline
end note

@enduml
```

**Data Layer Architecture:**

1. **Facade Pattern**: DataManager provides unified interface
2. **Separation of Concerns**: Historical, Realtime, Storage separated
3. **Preprocessing Pipeline**: Clean separation of data transformation
4. **Feature Engineering**: Dedicated component for ML features
5. **Multiple Storage Options**: SQLite for development, PostgreSQL for production

---

## Sequence Diagrams

### 1. Complete Trading Cycle Sequence

```plantuml
@startuml
title Trading Cycle - From Signal Generation to Order Execution

actor User
participant TradingEngine
participant AgentManager
participant Agent
participant DataManager
participant RiskManager
participant OrderExecutor
participant Exchange
participant EventBus

User -> TradingEngine: start()
activate TradingEngine

TradingEngine -> Exchange: connect()
activate Exchange
Exchange --> TradingEngine: connected
deactivate Exchange

TradingEngine -> TradingEngine: _run_trading_loop()

loop Every trading cycle (e.g., 10 seconds)

    TradingEngine -> AgentManager: get_active_agent()
    activate AgentManager
    AgentManager --> TradingEngine: active_agent
    deactivate AgentManager

    TradingEngine -> DataManager: get_market_data("BTC/USDT")
    activate DataManager
    DataManager -> Exchange: get_market_data("BTC/USDT")
    activate Exchange
    Exchange --> DataManager: market_data
    deactivate Exchange
    DataManager --> TradingEngine: processed_data
    deactivate DataManager

    TradingEngine -> Agent: analyze(market_data)
    activate Agent
    Agent -> Agent: calculate_indicators()
    Agent -> Agent: generate_signal()
    Agent --> TradingEngine: trading_signal
    deactivate Agent

    TradingEngine -> EventBus: publish(SIGNAL_GENERATED)
    activate EventBus
    EventBus -> EventBus: notify_subscribers()
    deactivate EventBus

    alt Trading is enabled
        TradingEngine -> TradingEngine: _execute_signal(signal)

        TradingEngine -> Exchange: get_balance()
        activate Exchange
        Exchange --> TradingEngine: balance
        deactivate Exchange

        TradingEngine -> Exchange: get_positions()
        activate Exchange
        Exchange --> TradingEngine: positions
        deactivate Exchange

        TradingEngine -> OrderExecutor: execute_signal(signal, balance, positions)
        activate OrderExecutor

        OrderExecutor -> RiskManager: calculate_position_size(signal, balance)
        activate RiskManager
        RiskManager --> OrderExecutor: position_size
        deactivate RiskManager

        OrderExecutor -> RiskManager: validate_order(order, positions)
        activate RiskManager
        RiskManager -> RiskManager: check_daily_loss_limit()
        RiskManager -> RiskManager: check_position_limits()
        RiskManager -> RiskManager: check_exposure_limits()
        RiskManager --> OrderExecutor: valid
        deactivate RiskManager

        alt Order is valid
            OrderExecutor -> Exchange: place_order(order)
            activate Exchange
            Exchange --> OrderExecutor: placed_order
            deactivate Exchange

            OrderExecutor --> TradingEngine: placed_order
            deactivate OrderExecutor

            TradingEngine -> EventBus: publish(ORDER_PLACED)
            activate EventBus
            EventBus -> EventBus: notify_subscribers()
            deactivate EventBus
        else Order rejected by risk
            OrderExecutor --> TradingEngine: RiskManagementError
            deactivate OrderExecutor

            TradingEngine -> EventBus: publish(ERROR_OCCURRED)
            activate EventBus
            deactivate EventBus
        end
    end

    TradingEngine -> TradingEngine: sleep(interval)
end

User -> TradingEngine: stop()
TradingEngine -> Exchange: disconnect()
activate Exchange
deactivate Exchange

deactivate TradingEngine

@enduml
```

### 2. ML Agent Training Sequence

```plantuml
@startuml
title ML Agent Training Workflow

actor User
participant MLAgent
participant MLStrategy
participant DataManager
participant DataPreprocessor
participant FeatureEngineer
participant Model
database Storage

User -> MLAgent: train_model()
activate MLAgent

MLAgent -> DataManager: get_historical_data(symbol, timeframe, start, end)
activate DataManager

DataManager -> Storage: query_historical_data()
activate Storage
Storage --> DataManager: raw_data
deactivate Storage

DataManager -> DataPreprocessor: process_historical_data(raw_data)
activate DataPreprocessor

DataPreprocessor -> DataPreprocessor: normalize_data()
DataPreprocessor -> DataPreprocessor: handle_missing_values()
DataPreprocessor -> DataPreprocessor: remove_outliers()

DataPreprocessor --> DataManager: processed_data
deactivate DataPreprocessor

DataManager --> MLAgent: training_data
deactivate DataManager

MLAgent -> MLStrategy: train_model(training_data)
activate MLStrategy

MLStrategy -> FeatureEngineer: create_features_for_ml(training_data)
activate FeatureEngineer

FeatureEngineer -> FeatureEngineer: create_price_features()
FeatureEngineer -> FeatureEngineer: create_volume_features()
FeatureEngineer -> FeatureEngineer: create_momentum_features()
FeatureEngineer -> FeatureEngineer: create_volatility_features()
FeatureEngineer -> FeatureEngineer: create_lagged_features()

FeatureEngineer --> MLStrategy: features_df
deactivate FeatureEngineer

MLStrategy -> MLStrategy: prepare_training_data(features_df)
MLStrategy -> MLStrategy: train_test_split(X, y)
MLStrategy -> MLStrategy: scale_features(X_train)

MLStrategy -> Model: create_model()
activate Model
Model --> MLStrategy: model_instance
deactivate Model

MLStrategy -> Model: fit(X_train, y_train)
activate Model
Model -> Model: train_iterations()
Model --> MLStrategy: trained_model
deactivate Model

MLStrategy -> Model: evaluate(X_test, y_test)
activate Model
Model --> MLStrategy: test_score
deactivate Model

MLStrategy -> Storage: save_model(model, scaler, metadata)
activate Storage
Storage --> MLStrategy: saved
deactivate Storage

MLStrategy --> MLAgent: training_metrics
deactivate MLStrategy

MLAgent --> User: training_results
deactivate MLAgent

@enduml
```

### 3. Risk Management Validation Sequence

```plantuml
@startuml
title Risk Management - Order Validation Flow

participant OrderExecutor
participant RiskManager
participant ConfigManager
database PositionDB

OrderExecutor -> RiskManager: validate_order(order, positions)
activate RiskManager

RiskManager -> RiskManager: _validate_order_basics(order)
note right: Check amount > 0\nCheck price > 0\nCheck symbol exists

alt Basic validation fails
    RiskManager --> OrderExecutor: false
else Basic validation passes

    RiskManager -> ConfigManager: get("risk.max_daily_loss_pct")
    activate ConfigManager
    ConfigManager --> RiskManager: max_daily_loss_pct
    deactivate ConfigManager

    RiskManager -> RiskManager: _check_daily_loss_limit()
    note right: Calculate today's losses\nCompare with limit\nReturn true/false

    alt Daily loss limit exceeded
        RiskManager --> OrderExecutor: false
    else Daily loss ok

        RiskManager -> ConfigManager: get("risk.max_positions_per_symbol")
        activate ConfigManager
        ConfigManager --> RiskManager: max_positions
        deactivate ConfigManager

        RiskManager -> RiskManager: _check_position_limits(order, positions)
        note right: Count positions for symbol\nCount total positions\nCheck against limits

        alt Position limits exceeded
            RiskManager --> OrderExecutor: false
        else Position limits ok

            RiskManager -> ConfigManager: get("risk.max_total_exposure_pct")
            activate ConfigManager
            ConfigManager --> RiskManager: max_exposure_pct
            deactivate ConfigManager

            RiskManager -> RiskManager: _check_exposure_limits(order, positions)
            note right: Calculate current exposure\nCalculate new exposure\nCheck against limit

            alt Exposure limits exceeded
                RiskManager --> OrderExecutor: false
            else All checks passed

                RiskManager -> RiskManager: _check_order_size_limits(order)
                note right: Check minimum order size\nCheck minimum order amount

                RiskManager --> OrderExecutor: validation_result
            end
        end
    end
end

deactivate RiskManager

@enduml
```

---

## Design Patterns Applied

### 1. Strategy Pattern
**Location**: Agent strategies, Order execution strategies
**Purpose**: Allow algorithms to be selected at runtime
**Implementation**:
- `IStrategy` interface with `TechnicalStrategy` and `MLStrategy` implementations
- `SignalExecutionStrategy` with `ImmediateExecutionStrategy` and `ConservativeExecutionStrategy`
- Agents delegate to strategy objects for signal generation

**Benefits**:
- Easy to add new strategies without modifying existing code
- Strategies can be swapped at runtime
- Clean separation between algorithm and context

### 2. Template Method Pattern
**Location**: `BaseExchange`, `BaseAgent`, `TechnicalStrategy`, `MLStrategy`
**Purpose**: Define skeleton of algorithm, let subclasses implement specific steps
**Implementation**:
- `BaseExchange` defines connection flow, subclasses implement exchange-specific details
- `TechnicalStrategy.analyze()` defines analysis flow, subclasses implement indicators
- `MLStrategy.train_model()` defines training flow, subclasses implement model specifics

**Benefits**:
- Code reuse through inheritance
- Consistent workflow across implementations
- Easy to add new implementations

### 3. Factory Pattern
**Location**: `AgentFactory`
**Purpose**: Create objects without specifying exact class
**Implementation**:
- `AgentFactory.register_agent_type()` registers creators
- `AgentFactory.create_agent()` creates instances
- `AgentBuilder` provides fluent interface for complex construction

**Benefits**:
- Centralized object creation
- Runtime type selection
- Extensible without modification (Open/Closed Principle)

### 4. Observer Pattern (Publish-Subscribe)
**Location**: `EventBus`
**Purpose**: Notify multiple objects about events
**Implementation**:
- `EventBus.subscribe()` registers callbacks
- `EventBus.publish()` notifies all subscribers
- Events: `ORDER_FILLED`, `SIGNAL_GENERATED`, `ERROR_OCCURRED`

**Benefits**:
- Loose coupling between components
- Asynchronous communication
- Easy to add new event listeners

### 5. Facade Pattern
**Location**: `TradingEngine`, `DataManager`
**Purpose**: Provide simplified interface to complex subsystem
**Implementation**:
- `TradingEngine` coordinates agents, risk, orders, exchange
- `DataManager` unifies historical, realtime, storage, preprocessing

**Benefits**:
- Simplified API for clients
- Hides subsystem complexity
- Single point of control

### 6. Adapter Pattern
**Location**: `TypeConverter`
**Purpose**: Convert between incompatible interfaces
**Implementation**:
- Converts exchange-specific data formats to domain models
- `to_market_data()`, `to_order()`, `to_position()`
- Isolates exchange API changes

**Benefits**:
- Decouples domain from external APIs
- Easy to support new exchanges
- Type-safe conversions

### 7. Dependency Injection
**Location**: Throughout system
**Purpose**: Invert control of dependencies
**Implementation**:
- All major components receive dependencies via constructor
- `TradingEngine(exchange_client, risk_manager, event_bus, ...)`
- Enables testing with mocks

**Benefits**:
- Loose coupling
- Easy testing
- Flexible configuration

### 8. Builder Pattern
**Location**: `AgentBuilder`
**Purpose**: Construct complex objects step by step
**Implementation**:
```python
agent = (AgentBuilder('rsi')
    .with_config({'period': 14})
    .with_exchange(exchange)
    .with_risk_manager(risk_manager)
    .build())
```

**Benefits**:
- Fluent interface
- Complex object construction
- Validation before build

---

## SOLID Principles Analysis

### Single Responsibility Principle (SRP)
Each class has one reason to change:

| Class | Single Responsibility |
|-------|----------------------|
| `TradingEngine` | Orchestrate trading lifecycle |
| `OrderExecutor` | Execute orders |
| `RiskManager` | Validate risk constraints |
| `EventBus` | Manage event subscriptions and publishing |
| `DataManager` | Coordinate data operations |
| `BaseAgent` | Provide common agent functionality |
| `TechnicalStrategy` | Implement technical analysis |
| `MLStrategy` | Implement ML-based analysis |

**Violations Avoided**:
- Separated `OrderExecutor` from `TradingEngine` (was initially combined)
- Split `IExchangeClient` into smaller interfaces
- Extracted `AccountStateManager` from `TradingEngine`

### Open/Closed Principle (OCP)
Open for extension, closed for modification:

**Extension Points**:
1. **New Trading Strategies**: Extend `TechnicalStrategy` or `MLStrategy`
2. **New Exchanges**: Extend `BaseExchange`
3. **New Execution Strategies**: Implement `SignalExecutionStrategy`
4. **New Agents**: Register with `AgentFactory`
5. **New Event Types**: Add to `EventType` enum, subscribe to `EventBus`

**No Modification Required**:
- Adding new agent doesn't modify `AgentManager`
- Adding new strategy doesn't modify `Agent`
- Adding new exchange doesn't modify `TradingEngine`

### Liskov Substitution Principle (LSP)
Derived classes are substitutable for base classes:

**Correct Substitutions**:
- Any `BaseExchange` subclass can replace `IExchangeClient`
- Any `TechnicalStrategy` can replace `IStrategy`
- Any `MLStrategy` can replace `IStrategy`
- Any `SignalExecutionStrategy` implementation is interchangeable

**Contracts Maintained**:
- All implementations respect interface preconditions/postconditions
- No exceptions thrown that violate contracts
- Return types match expectations

### Interface Segregation Principle (ISP)
Clients depend only on interfaces they use:

**Well-Segregated Interfaces**:
```
IExchangeClient split into:
├── IExchangeConnection (connect, disconnect)
├── IMarketDataProvider (get_market_data, get_historical_data)
├── IOrderExecutor (place_order, cancel_order)
└── IAccountDataProvider (get_positions, get_balance)
```

**Benefits**:
- Risk manager only needs `IRiskManager`, not full exchange interface
- Data manager only needs `IDataProvider`
- Agents only need `ISignalGenerator`

### Dependency Inversion Principle (DIP)
Depend on abstractions, not concretions:

**High-Level Modules Depend on Abstractions**:
- `TradingEngine` depends on `IExchangeClient`, not `BlofinExchange`
- `OrderExecutor` depends on `IRiskManager`, not `RiskManager`
- `Agent` depends on `IStrategy`, not specific strategy

**Dependency Injection**:
```python
class TradingEngine:
    def __init__(
        self,
        exchange_client: IExchangeClient,  # Interface, not concrete
        risk_manager: IRiskManager,        # Interface, not concrete
        event_bus: IEventBus,              # Interface, not concrete
        ...
    ):
```

**Benefits**:
- Easy to swap implementations
- Testing with mocks
- Loose coupling

---

## Architecture Quality Metrics

### Coupling Metrics
- **Low Coupling**: Components interact through interfaces
- **High Cohesion**: Related functionality grouped together
- **Dependency Direction**: All dependencies point toward abstractions

### Extensibility Points
1. New trading strategies: Extend base strategy classes
2. New exchanges: Implement `IExchangeClient` interface
3. New risk rules: Extend `RiskManager`
4. New data sources: Implement `IDataProvider`
5. New event types: Add to `EventType` enum
6. New execution strategies: Implement `SignalExecutionStrategy`

### Testability
- All dependencies injectable
- Interfaces allow mocking
- Single responsibility enables isolated testing
- Event-driven enables integration testing

### Maintainability
- Clear separation of concerns
- Consistent naming conventions
- Well-documented interfaces
- Design patterns aid understanding

---

## Future Architecture Enhancements

### Recommended Improvements

1. **Command Pattern for Order Management**
   - Encapsulate order operations as commands
   - Enable undo/redo functionality
   - Queue commands for batch processing

2. **State Pattern for Trading Modes**
   - Paper Trading State
   - Live Trading State
   - Backtesting State
   - Each state with different behavior

3. **Repository Pattern for Data Access**
   - Abstract data persistence layer
   - Support multiple databases seamlessly
   - Easier testing with in-memory repository

4. **Circuit Breaker for Exchange Calls**
   - Prevent cascading failures
   - Auto-recovery after exchange downtime
   - Fallback to cached data

5. **Decorator Pattern for Order Enhancement**
   - Add stop-loss automatically
   - Add take-profit automatically
   - Add trailing stops
   - Stack multiple decorators

---

## Conclusion

This architecture demonstrates a well-designed, SOLID-compliant trading system with:

- **Clear Layering**: GUI → Core → Agents → Exchange → Data
- **Strong Abstraction**: Interfaces define contracts, implementations vary
- **High Extensibility**: Easy to add strategies, exchanges, agents
- **Loose Coupling**: Components interact through interfaces
- **Event-Driven**: Asynchronous communication via EventBus
- **Testable**: Dependency injection enables comprehensive testing

The system is production-ready with room for enhancement while maintaining architectural integrity.

---

**Document Version**: 1.0
**Last Updated**: 2025-12-18
**Architect**: Claude Sonnet 4.5
