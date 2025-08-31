# crewai_system/tools/base_tools.py
"""
Base Tool Wrapper Infrastructure
=================================
Foundation for wrapping your existing modules as CrewAI tools.
These wrappers make your proven logic accessible to agents.

Design Philosophy:
- Thin wrappers that preserve existing functionality
- Error handling that agents can interpret
- Structured responses for JSON parsing

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from crewai.tools import tool

# Add parent directory to path to import your existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import your existing modules
try:
    from market_context import MarketContext
    from strategy_engine import StrategyEngine
    from nexus_core.execution_engine import ExecutionEngine
    from nexus_core.position_monitor import PositionMonitor
    from config import get_config
    from models import StrategyType, MarketRegime
    from exceptions import (
        TradingSystemError, InsufficientCreditError,
        StrikeTooCloseError, NoViableStrategyError
    )
except ImportError as e:
    logging.error(f"Failed to import existing modules: {e}")
    raise

# Initialize shared instances (singleton pattern)
_market_context: Optional[MarketContext] = None
_strategy_engine: Optional[StrategyEngine] = None
_execution_engine: Optional[ExecutionEngine] = None
_position_monitor: Optional[PositionMonitor] = None

def initialize_tool_infrastructure() -> bool:
    """
    Initialize all tool infrastructure with your existing modules.
    
    Must be called before agents can use tools.
    
    Returns:
        True if initialization successful
    """
    global _market_context, _strategy_engine, _execution_engine, _position_monitor
    
    try:
        # Initialize your existing components
        _market_context = MarketContext()
        _strategy_engine = StrategyEngine(_market_context)
        _execution_engine = ExecutionEngine()
        _position_monitor = PositionMonitor(_market_context, _execution_engine)
        
        logging.info("Tool infrastructure initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize tool infrastructure: {e}")
        return False


# Example tool wrapper - pattern for all tools
@tool
def analyze_market_conditions() -> str:
    """
    Analyze current market conditions using MarketContext.
    
    Returns comprehensive market analysis including:
    - Market regime (trending/ranging/volatile)
    - Trend strength and direction
    - VIX level and implications
    - Movement statistics
    - Recommended strategy
    
    Returns:
        JSON string with market analysis
    """
    if not _market_context:
        return json.dumps({
            "error": "Market context not initialized",
            "success": False
        })
    
    try:
        # Get current market analysis
        analysis = _market_context.analyze_market()
        
        # Get trading parameters
        params = _market_context.get_trading_parameters()
        
        # Structure response for agent consumption
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "market_analysis": {
                "regime": analysis.market_regime.value,
                "regime_confidence": analysis.regime_confidence,
                "trend_state": analysis.trend_state.value,
                "trend_strength": analysis.trend_strength,
                "vix": _market_context.current_data.vix if _market_context.current_data else None,
                "underlying_price": _market_context.current_data.underlying_price if _market_context.current_data else None,
            },
            "movement_stats": {
                "atr_points": analysis.movement_stats.atr_points,
                "atr_percent": analysis.movement_stats.atr_percent,
                "expected_move": analysis.movement_stats.expected_move_1std,
            },
            "parameters": {
                "wing_width": params["wing_width"],
                "min_credit_ic": params["min_credit_ic"],
                "strike_distance_pct": params["strike_distance_pct"],
                "can_trade": params["can_trade"],
            },
            "recommendation": {
                "strategy": analysis.recommended_strategy.value,
                "confidence": analysis.confidence_level,
                "risk_level": analysis.risk_assessment.value,
                "rationale": analysis.rationale
            }
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "success": False
        })


@tool 
def check_execution_connection() -> str:
    """
    Check Interactive Brokers connection status.
    
    Returns:
        JSON string with connection status and account details
    """
    if not _execution_engine:
        return json.dumps({
            "error": "Execution engine not initialized",
            "success": False
        })
    
    try:
        is_connected = _execution_engine.is_connected()
        
        response = {
            "success": True,
            "connected": is_connected,
            "broker": "Interactive Brokers",
            "mode": "Paper Trading",
        }
        
        if is_connected:
            # Get account summary if connected
            account = _execution_engine.get_account_summary()
            response["account"] = {
                "net_liquidation": account.get("NetLiquidation", 0),
                "available_funds": account.get("AvailableFunds", 0),
                "buying_power": account.get("BuyingPower", 0),
            }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "success": False
        })
        
        