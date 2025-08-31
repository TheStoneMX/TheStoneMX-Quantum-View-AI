# crewai_system/tools/execution_tools.py
"""
Execution Tools - REAL trades via Interactive Brokers
======================================================
Direct connection to your execution_engine.py for actual trading.
"""

from crewai.tools import tool

import json
from typing import Dict, Any

@tool
async def execute_real_trade(trade_setup: Dict[str, Any]) -> str:
    """
    Execute a REAL trade through Interactive Brokers.
    
    Args:
        trade_setup: Complete trade specification
        
    Returns:
        JSON with REAL execution results from IB
    """
    try:
        # Convert dict to your TradeSetup object
        from models import TradeSetup, StrikeSelection, StrategyType
        
        strikes = StrikeSelection(
            short_put_strike=trade_setup["strikes"]["short_put"],
            long_put_strike=trade_setup["strikes"]["long_put"],
            short_call_strike=trade_setup["strikes"]["short_call"],
            long_call_strike=trade_setup["strikes"]["long_call"]
        )
        
        setup = TradeSetup(
            strategy=StrategyType[trade_setup["strategy_type"].upper()],
            strikes=strikes,
            contracts=trade_setup["contracts"],
            credit_per_contract=trade_setup["expected_credit"],
            max_risk_per_contract=trade_setup["max_risk"]
        )
        
        # Execute REAL trade via your execution engine
        result = await _execution_engine.execute_trade(setup)
        
        return json.dumps({
            "success": result.success,
            "position_id": result.position_id,
            "fill_prices": result.fill_prices,
            "total_credit": result.total_fill_price,
            "error": result.error_message
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })