# UML Diagrams - Trading Agent System

This directory contains PlantUML diagrams documenting the system architecture.

## Viewing the Diagrams

### Online Viewers
1. **PlantUML Web Server**: http://www.plantuml.com/plantuml/uml/
   - Copy and paste the `.puml` file contents
   - View rendered diagram instantly

2. **VS Code Extension**:
   - Install "PlantUML" extension by jebbs
   - Open any `.puml` file
   - Press `Alt+D` to preview

3. **IntelliJ IDEA**:
   - Install "PlantUML integration" plugin
   - Right-click `.puml` file → "Show PlantUML Diagram"

### Generating Images Locally

```bash
# Install PlantUML (requires Java)
# macOS
brew install plantuml

# Ubuntu/Debian
sudo apt-get install plantuml

# Generate PNG images
plantuml diagrams/*.puml

# Generate SVG (better quality)
plantuml -tsvg diagrams/*.puml

# Generate all formats
plantuml -tpng -tsvg diagrams/*.puml
```

## Available Diagrams

### 1. system_overview.puml
High-level component diagram showing:
- System layers (UI, Core, Agent, Exchange, Data)
- Major components in each layer
- Dependencies between components
- External systems (Database, Exchange APIs)

**Use this to**: Understand overall system structure and component relationships

### 2. core_layer_detailed.puml
Detailed class diagram of the Core Layer:
- All core interfaces (IExchangeClient, IRiskManager, etc.)
- TradingEngine (main orchestrator)
- OrderExecutor and execution strategies
- RiskManager with validation logic
- EventBus for event-driven architecture
- AccountStateManager
- Core data models (Order, Position, TradingSignal, MarketData)

**Use this to**: Understand the core trading engine and risk management

### 3. agent_layer_detailed.puml
Detailed class diagram of the Agent Layer:
- Base agent and strategy abstractions
- Technical analysis strategies (RSI, MACD, MA, Bollinger Bands)
- Machine learning strategies (Random Forest, LSTM)
- Agent factory and builder patterns
- Strategy pattern implementation

**Use this to**: Understand how trading strategies are implemented and extended

### 4. exchange_layer_detailed.puml
Detailed class diagram of the Exchange Layer:
- BaseExchange template method pattern
- BlofinExchange concrete implementation
- TypeConverter adapter pattern
- Exchange client API wrapper
- Connection pooling and factory

**Use this to**: Understand exchange integration and how to add new exchanges

### 5. data_layer_detailed.puml
Detailed class diagram of the Data Layer:
- DataManager facade
- Historical and realtime data providers
- Data preprocessing pipeline
- Feature engineering for ML
- Storage abstraction (SQLite, PostgreSQL)
- Caching strategy

**Use this to**: Understand data flow and feature engineering

### 6. trading_sequence.puml
Complete sequence diagram showing:
- Full trading cycle from start to finish
- Signal generation by agents
- Risk validation process
- Order execution flow
- Event publishing
- Order monitoring
- Error handling

**Use this to**: Understand the runtime behavior and interactions

### 7. ml_training_sequence.puml
Machine Learning training workflow:
- Data collection and preprocessing
- Feature engineering process
- Model training steps
- Model evaluation and validation
- Model persistence
- Inference workflow

**Use this to**: Understand how ML models are trained and used

### 8. event_driven_architecture.puml
Event-driven architecture details:
- EventBus implementation
- Event publishers and subscribers
- System components as event producers
- Dashboard, logging, monitoring as consumers
- Event flow examples

**Use this to**: Understand event-driven communication

### 9. deployment_diagram.puml
Production deployment architecture:
- Application server setup
- Database configuration
- Caching layer (Redis)
- Monitoring stack (Prometheus, Grafana)
- Log aggregation (ELK stack)
- Network zones and security

**Use this to**: Understand production deployment

### 10. state_diagram.puml
Trading Engine state machine:
- All system states
- State transitions
- Error handling states
- Recovery mechanisms
- Order lifecycle states

**Use this to**: Understand system behavior and states

### 11. package_diagram.puml
Package dependencies:
- All Python packages/modules
- Internal dependencies
- External library dependencies
- Dependency direction and flow

**Use this to**: Understand module organization and dependencies

### 12. solid_principles.puml
SOLID principles demonstration:
- Single Responsibility examples
- Open/Closed examples
- Liskov Substitution examples
- Interface Segregation examples
- Dependency Inversion examples
- Anti-patterns to avoid

**Use this to**: Understand SOLID principles application

## Diagram Relationships

```
system_overview.puml
    ├─ Shows WHERE components fit
    │
    ├─ core_layer_detailed.puml
    │  └─ Details WHAT the core layer does
    │
    ├─ agent_layer_detailed.puml
    │  └─ Details HOW strategies work
    │
    └─ trading_sequence.puml
       └─ Shows WHEN things happen
```

## Design Patterns Illustrated

Each diagram highlights different design patterns:

### system_overview.puml
- Layered Architecture
- Separation of Concerns

### core_layer_detailed.puml
- Facade Pattern (TradingEngine)
- Strategy Pattern (SignalExecutionStrategy)
- Observer Pattern (EventBus)
- Interface Segregation Principle

### agent_layer_detailed.puml
- Template Method Pattern (TechnicalStrategy, MLStrategy)
- Strategy Pattern (Agent uses Strategy)
- Factory Pattern (AgentFactory)
- Builder Pattern (AgentBuilder)

### trading_sequence.puml
- Sequence of operations
- Error handling flow
- Asynchronous event publishing

## Extending the Diagrams

When adding new features:

1. **New Component**: Add to `system_overview.puml`
2. **New Class**: Add to appropriate layer diagram
3. **New Interaction**: Update `trading_sequence.puml`

### Example: Adding a New Agent Type

1. Add class to `agent_layer_detailed.puml`:
```plantuml
class MyNewAgent {
    - strategy: MyStrategy
    + analyze(market_data): TradingSignal
}

BaseAgent <|-- MyNewAgent
MyNewAgent --> MyStrategy : delegates
```

2. Update factory registration in code documentation

## SOLID Principles in Diagrams

Look for these SOLID principle applications:

- **S**ingle Responsibility: Each class has one job
  - OrderExecutor only executes orders
  - RiskManager only validates risk
  - EventBus only handles events

- **O**pen/Closed: Extension without modification
  - Add new agents by extending BaseAgent
  - Add new strategies by implementing IStrategy
  - Add new exchanges by extending BaseExchange

- **L**iskov Substitution: Subtypes are interchangeable
  - Any IExchangeClient works with TradingEngine
  - Any IStrategy works with Agent
  - Any SignalExecutionStrategy works with OrderExecutor

- **I**nterface Segregation: Small, focused interfaces
  - IExchangeClient split into 4 interfaces
  - ITradingAgent split into ISignalGenerator + IConfigurableAgent

- **D**ependency Inversion: Depend on abstractions
  - TradingEngine depends on interfaces, not concrete classes
  - All arrows point to interfaces in diagrams

## Color Coding

The diagrams use color coding to indicate component types:

- **Light Blue**: Core layer components
- **Light Green**: Agent layer components
- **Light Coral**: Exchange layer components
- **Light Yellow**: Data layer components
- **Plum**: UI layer components

## Notes and Annotations

Diagrams include notes explaining:
- Design pattern usage
- Component responsibilities
- Architectural decisions
- Important relationships

## Updating Diagrams

When code changes:
1. Update the relevant `.puml` file
2. Regenerate images
3. Update this README if structure changes
4. Commit both `.puml` and generated images

## Tools and Resources

- **PlantUML Documentation**: https://plantuml.com/
- **PlantUML Cheat Sheet**: https://plantuml.com/commons
- **Real World PlantUML**: https://real-world-plantuml.com/

## Questions?

For questions about:
- **System Architecture**: See `../ARCHITECTURE_UML.md`
- **SOLID Principles**: See SOLID Principles section in `../ARCHITECTURE_UML.md`
- **Design Patterns**: See Design Patterns section in `../ARCHITECTURE_UML.md`
