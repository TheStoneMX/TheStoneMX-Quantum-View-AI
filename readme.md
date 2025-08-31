
Quantum Vision AI/
├── main_crewai.py
├── config.py (your existing)
├── market_context.py (your existing)
├── strategy_engine.py (your existing)
├── execution_engine.py (your existing)
├── crewai_system/
│   ├── config/
│   │   ├── trading_config.yaml
│   │   ├── config_loader.py
│   │   ├── timezone_handler.py
│   │   └── ollama_config.py
│   ├── agents/
│   │   ├── market_analyst.py
│   │   ├── strategy_architect.py
│   │   ├── risk_manager.py
│   │   └── chief_trading_officer.py
│   ├── tools/
│   │   ├── base_tools.py
│   │   ├── market_tools.py
│   │   └── execution_tools.py
│   ├── crews/
│   │   └── trading_crew.py
│   └── memory/
│       ├── persistence.py
│       └── trading_journal.py