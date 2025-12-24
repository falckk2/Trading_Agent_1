# Architecture Documentation Index

This document helps you navigate the complete architecture documentation for the Crypto Trading Agent System.

---

## Documentation Structure

```
Architecture Documentation
│
├── Quick Start (You are here)
│   └── ARCHITECTURE_INDEX.md
│
├── Executive Summary
│   └── ARCHITECTURE_SUMMARY.md (15-minute read)
│       ├── System overview
│       ├── Key components
│       ├── Quick reference tables
│       └── Extension points
│
├── Detailed Architecture
│   └── ARCHITECTURE_UML.md (45-minute read)
│       ├── Complete UML diagrams (PlantUML syntax)
│       ├── Design patterns analysis
│       ├── SOLID principles detailed
│       └── Sequence diagrams
│
└── Visual Diagrams
    └── diagrams/ directory
        ├── system_overview.puml
        ├── core_layer_detailed.puml
        ├── agent_layer_detailed.puml
        ├── trading_sequence.puml
        └── README.md (how to view diagrams)
```

---

## What to Read Based on Your Role

### Software Developer (New to Project)
**Goal**: Understand how to add features

1. Start: `ARCHITECTURE_SUMMARY.md` (read sections: System Layers, Core Interfaces, Trading Flow)
2. Then: `diagrams/system_overview.puml` (view the rendered diagram)
3. Then: `ARCHITECTURE_UML.md` (read: Extension Points section)
4. Deep dive: Specific layer diagrams based on what you're working on

**Time**: 30 minutes to get started, 2 hours for deep understanding

### System Architect / Tech Lead
**Goal**: Understand design decisions and trade-offs

1. Start: `ARCHITECTURE_SUMMARY.md` (complete read)
2. Then: `ARCHITECTURE_UML.md` (focus on: Design Patterns, SOLID Principles)
3. Review: All diagrams in `diagrams/` directory
4. Analyze: Sequence diagrams for runtime behavior

**Time**: 2-3 hours for comprehensive understanding

### DevOps / Infrastructure
**Goal**: Understand deployment and dependencies

1. Start: `ARCHITECTURE_SUMMARY.md` (sections: System Layers, Deployment, Security)
2. Then: `diagrams/system_overview.puml` (understand external dependencies)
3. Review: Database schema in `ARCHITECTURE_UML.md`
4. Check: Configuration requirements

**Time**: 45 minutes

### QA / Test Engineer
**Goal**: Understand testing strategy and test points

1. Start: `ARCHITECTURE_SUMMARY.md` (sections: Core Interfaces, Trading Flow, Testing Strategy)
2. Then: `diagrams/trading_sequence.puml` (understand complete flow)
3. Review: `ARCHITECTURE_UML.md` (Testability section)
4. Examine: Existing tests in `/tests` directory

**Time**: 1 hour

### Product Manager / Business Analyst
**Goal**: Understand capabilities and limitations

1. Start: `ARCHITECTURE_SUMMARY.md` (sections: What It Does, Trading Flow, Risk Management)
2. Then: `diagrams/system_overview.puml` (high-level view)
3. Review: Extension Points in `ARCHITECTURE_SUMMARY.md`
4. Check: Configuration options

**Time**: 30 minutes

---

## Quick Navigation by Topic

### Understanding System Structure
- **High-Level**: `diagrams/system_overview.puml`
- **Detailed Layers**: Each layer has dedicated section in `ARCHITECTURE_UML.md`
- **Component Relationships**: All diagrams show dependencies

### Understanding Runtime Behavior
- **Complete Trading Cycle**: `diagrams/trading_sequence.puml`
- **ML Training Flow**: `ARCHITECTURE_UML.md` → Sequence Diagrams section
- **Risk Validation**: `ARCHITECTURE_UML.md` → Sequence Diagrams section

### Understanding Design Decisions
- **SOLID Principles**: `ARCHITECTURE_UML.md` → SOLID Principles Analysis
- **Design Patterns**: `ARCHITECTURE_UML.md` → Design Patterns Applied
- **Trade-offs**: `ARCHITECTURE_UML.md` → Architecture Quality Metrics

### Adding New Features

#### Adding a New Trading Strategy
1. Read: `ARCHITECTURE_SUMMARY.md` → Extension Points → Adding a New Trading Strategy
2. View: `diagrams/agent_layer_detailed.puml`
3. Example: `examples/solid_patterns_example.py`
4. Template: Extend `TechnicalStrategy` or `MLStrategy`

#### Adding a New Exchange
1. Read: `ARCHITECTURE_SUMMARY.md` → Extension Points → Adding a New Exchange
2. View: `diagrams/core_layer_detailed.puml` (Exchange section)
3. Review: `/crypto_trading/exchange/base_exchange.py`
4. Template: Extend `BaseExchange`

#### Adding a New Risk Rule
1. Read: `ARCHITECTURE_SUMMARY.md` → Risk Management
2. View: `/crypto_trading/core/risk_manager.py`
3. Add: New validation method in `RiskManager`

### Understanding Specific Components

| Component | Quick Reference | Detailed Diagram | Implementation |
|-----------|----------------|------------------|----------------|
| Trading Engine | `ARCHITECTURE_SUMMARY.md` → Trading Flow | `core_layer_detailed.puml` | `/crypto_trading/core/trading_engine.py` |
| Risk Manager | `ARCHITECTURE_SUMMARY.md` → Risk Management | `core_layer_detailed.puml` | `/crypto_trading/core/risk_manager.py` |
| Agents | `ARCHITECTURE_SUMMARY.md` → Agent Layer | `agent_layer_detailed.puml` | `/crypto_trading/agents/` |
| Data Pipeline | `ARCHITECTURE_SUMMARY.md` → Machine Learning Pipeline | `ARCHITECTURE_UML.md` | `/crypto_trading/data/` |
| Event System | `ARCHITECTURE_SUMMARY.md` → Event-Driven Architecture | `core_layer_detailed.puml` | `/crypto_trading/core/event_bus.py` |

---

## Common Questions → Where to Find Answers

### How does the system execute a trade?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Trading Flow (step-by-step)
- `diagrams/trading_sequence.puml` (visual sequence)

### How do I add a new technical indicator?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Extension Points
- `diagrams/agent_layer_detailed.puml` → TechnicalStrategy section
- Code: `/crypto_trading/agents/technical/technical_strategy.py`

### What design patterns are used and why?
**Answer in**:
- `ARCHITECTURE_UML.md` → Design Patterns Applied (detailed)
- `ARCHITECTURE_SUMMARY.md` → Design Patterns Quick Reference (table)

### How is risk managed?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Risk Management (comprehensive)
- `diagrams/trading_sequence.puml` → Risk Management Phase
- Code: `/crypto_trading/core/risk_manager.py`

### How do ML models integrate with the system?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Machine Learning Pipeline
- `diagrams/agent_layer_detailed.puml` → MLStrategy section
- `ARCHITECTURE_UML.md` → ML Agent Training Sequence

### How do events work?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Event-Driven Architecture
- `diagrams/core_layer_detailed.puml` → EventBus
- Code: `/crypto_trading/core/event_bus.py`

### How are SOLID principles applied?
**Answer in**:
- `ARCHITECTURE_UML.md` → SOLID Principles Analysis (detailed)
- `ARCHITECTURE_SUMMARY.md` → SOLID Principles Applied (summary)
- Examples: Throughout all diagrams (noted with comments)

### How do I configure the system?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Configuration section
- Example configs: `/config/` directory
- Code: `/crypto_trading/core/config_manager.py`

### How is testing structured?
**Answer in**:
- `ARCHITECTURE_SUMMARY.md` → Testing Strategy
- `ARCHITECTURE_UML.md` → Testability section
- Code: `/tests/` directory

### What databases are supported?
**Answer in**:
- `ARCHITECTURE_UML.md` → Data Layer Class Diagram
- Code: `/crypto_trading/data/storage/`

---

## Reading Paths for Common Tasks

### Task: Implement a new RSI-based strategy
**Path**:
1. Read: `ARCHITECTURE_SUMMARY.md` → Agent Layer → Technical Analysis Agents
2. View: `diagrams/agent_layer_detailed.puml` → RSIStrategy
3. Review: `/crypto_trading/agents/technical/rsi_agent.py`
4. Copy and modify: Create your variant
5. Register: Use `AgentFactory.register_agent_type()`

### Task: Add support for Binance exchange
**Path**:
1. Read: `ARCHITECTURE_SUMMARY.md` → Exchange Layer
2. View: `diagrams/system_overview.puml` → Exchange Layer
3. Review: `/crypto_trading/exchange/base_exchange.py` (template)
4. Review: `/crypto_trading/exchange/blofin_exchange.py` (example)
5. Implement: Extend `BaseExchange` for Binance
6. Create: Type converter methods

### Task: Add a new risk validation rule
**Path**:
1. Read: `ARCHITECTURE_SUMMARY.md` → Risk Management
2. View: `diagrams/trading_sequence.puml` → Risk Management Phase
3. Review: `/crypto_trading/core/risk_manager.py`
4. Add: New validation method (e.g., `_check_volatility_limit()`)
5. Call: From `validate_order()` method
6. Test: Add unit tests in `/tests/test_risk_manager.py`

### Task: Implement LSTM model for prediction
**Path**:
1. Read: `ARCHITECTURE_SUMMARY.md` → Machine Learning Pipeline
2. View: `diagrams/agent_layer_detailed.puml` → LSTMStrategy
3. Review: `/crypto_trading/agents/ml/lstm_agent.py`
4. Review: `/crypto_trading/agents/ml/ml_strategy.py` (base class)
5. Implement: Your LSTM variant
6. Train: Use training pipeline
7. Integrate: Register with `AgentFactory`

### Task: Add monitoring dashboard widget
**Path**:
1. Read: `ARCHITECTURE_SUMMARY.md` → Monitoring and Observability
2. View: `diagrams/system_overview.puml` → GUI Layer
3. Review: `/crypto_trading/gui/main_window.py`
4. Subscribe: To relevant events via `EventBus`
5. Display: Real-time data in PyQt widget

---

## Documentation Maintenance

### When adding new features:
1. Update relevant section in `ARCHITECTURE_SUMMARY.md`
2. Update corresponding PlantUML diagram in `diagrams/`
3. Regenerate diagram images
4. Update this index if new documentation added

### When refactoring:
1. Check all diagrams for accuracy
2. Update class diagrams if interfaces changed
3. Update sequence diagrams if flow changed
4. Update SOLID analysis if principles affected

### When fixing bugs:
- Usually no documentation update needed
- Unless bug revealed architectural flaw

---

## External Resources

### Learning Materials
- **SOLID Principles**: https://en.wikipedia.org/wiki/SOLID
- **Design Patterns**: https://refactoring.guru/design-patterns
- **PlantUML**: https://plantuml.com/
- **Clean Architecture**: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html

### Tools
- **PlantUML Online**: http://www.plantuml.com/plantuml/uml/
- **VS Code PlantUML Extension**: jebbs.plantuml
- **Draw.io**: https://app.diagrams.net/ (for quick sketches)

---

## Getting Help

### For Architecture Questions:
1. Check this index for relevant documentation
2. Review the appropriate diagram
3. Read the detailed section in `ARCHITECTURE_UML.md`
4. Examine the actual code implementation

### For Code Questions:
1. Start with `ARCHITECTURE_SUMMARY.md` to understand context
2. Review the class diagram for the component
3. Look at the implementation in `/crypto_trading`
4. Check tests in `/tests` for usage examples

### For Design Questions:
1. Review `ARCHITECTURE_UML.md` → Design Patterns section
2. Review `ARCHITECTURE_UML.md` → SOLID Principles section
3. Examine how existing features are implemented
4. Consider consistency with existing patterns

---

## Document Versions

| Document | Version | Last Updated |
|----------|---------|--------------|
| ARCHITECTURE_INDEX.md | 1.0 | 2025-12-18 |
| ARCHITECTURE_SUMMARY.md | 1.0 | 2025-12-18 |
| ARCHITECTURE_UML.md | 1.0 | 2025-12-18 |
| diagrams/*.puml | 1.0 | 2025-12-18 |

---

## Quick Start Checklist

- [ ] Read `ARCHITECTURE_SUMMARY.md` (15 minutes)
- [ ] View `diagrams/system_overview.puml` (5 minutes)
- [ ] Understand your component's layer and responsibilities
- [ ] Review relevant detailed diagram
- [ ] Check code implementation
- [ ] Ready to contribute!

**Total Time to Productivity**: ~30 minutes

---

**Happy Coding!**

For questions or suggestions about this documentation, please update this index and submit a pull request.
