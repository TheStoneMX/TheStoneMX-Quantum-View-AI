# crewai_system/tools/market_tools.py
"""
Market Data Tools - REAL DATA from your existing system
========================================================
These tools connect directly to your market_context.py and 
execution_engine.py to get live data from Interactive Brokers.

NO MOCK DATA - Everything comes from your production modules.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from crewai.tools import tool

# Add parent directory to import YOUR EXISTING modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_context import MarketContext
from nexus_core.execution_engine import ExecutionEngine
from models import OptionContract
from config import get_config
from ib_insync import Index

# Initialize your REAL components
_market_context = MarketContext()
_execution_engine = ExecutionEngine()


@tool
async def get_real_market_data() -> str:
    """Get REAL market data from Interactive Brokers."""
    try:
        if not _execution_engine.connected:
            await _execution_engine.connect(paper=True)

        # Get full ticker data from IB (not just price)
        ticker = _execution_engine.ib.reqMktData(
            _execution_engine.underlying_contract,
            '',
            False,
            False
        )
        _execution_engine.ib.sleep(2)

        # Get VIX separately
        vix_contract = Index('VIX', 'CBOE')
        vix_ticker = _execution_engine.ib.reqMktData(vix_contract, '', False, False)
        _execution_engine.ib.sleep(1)

        market_data = {
            "underlying_price": ticker.last or ticker.close,
            "vix": vix_ticker.last or vix_ticker.close,
            "volume": ticker.volume,
            "high": ticker.high,
            "low": ticker.low,
            "open": ticker.open,
            "previous_close": ticker.close,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "timestamp": datetime.now().isoformat()
        }

        # Cancel market data subscriptions
        _execution_engine.ib.cancelMktData(ticker.contract)
        _execution_engine.ib.cancelMktData(vix_contract)

        return json.dumps(market_data)
        
    except Exception as e:
        logging.error(f"Failed to get real market data: {e}")
        return json.dumps({"error": str(e), "success": False})
    
# async def get_real_market_data() -> str:
#     """
#     Get REAL market data from Interactive Brokers via your execution engine.
    
#     Returns:
#         JSON string with actual NQ price, VIX, and other market data
#     """
#     try:
#         # Connect to IB if not connected
#         if not _execution_engine.connected:
#             await _execution_engine.connect(paper=True)
        
#         # Get REAL underlying price from IB
#         current_price = await _execution_engine.get_underlying_price()
        
#         # Get REAL market data from your market context
#         # This assumes you've updated market context with IB data
#         market_data = {
#             "underlying_price": current_price,
#             "vix": _market_context.current_data.vix if _market_context.current_data else None,
#             "volume": _market_context.current_data.underlying_volume if _market_context.current_data else None,
#             "high": _market_context.current_data.high_price if _market_context.current_data else None,
#             "low": _market_context.current_data.low_price if _market_context.current_data else None,
#             "open": _market_context.current_data.open_price if _market_context.current_data else None,
#             "previous_close": _market_context.current_data.previous_close if _market_context.current_data else None,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         return json.dumps(market_data)
        
#     except Exception as e:
#         logging.error(f"Failed to get real market data: {e}")
#         return json.dumps({"error": str(e), "success": False})


@tool
async def get_real_option_chain() -> str:
    """
    Get REAL option chain from Interactive Brokers.
    
    Returns:
        JSON string with actual available options for today's expiration
    """
    try:
        # Ensure connected to IB
        if not _execution_engine.connected:
            await _execution_engine.connect(paper=True)
        
        # Fetch REAL 0DTE option chain from IB
        options = await _execution_engine.fetch_option_chain()
        
        # Convert to serializable format
        option_data = []
        for opt in options:
            option_data.append({
                "strike": opt.strike,
                "right": opt.right,
                "bid": opt.bid,
                "ask": opt.ask,
                "mid": opt.mid,
                "volume": opt.volume,
                "open_interest": opt.open_interest,
                "delta": opt.greeks.delta if opt.greeks else None,
                "gamma": opt.greeks.gamma if opt.greeks else None,
                "theta": opt.greeks.theta if opt.greeks else None,
                "vega": opt.greeks.vega if opt.greeks else None,
                "iv": opt.greeks.iv if opt.greeks else None
            })
        
        return json.dumps({
            "success": True,
            "options_count": len(option_data),
            "options": option_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Failed to get option chain: {e}")
        return json.dumps({"error": str(e), "success": False})


@tool
def analyze_market_with_real_data() -> str:
    """
    Perform market analysis using REAL data from your market_context.
    
    Returns:
        JSON string with actual market analysis based on live data
    """
    try:
        # Ensure market context has current data
        if not _market_context.current_data:
            return json.dumps({
                "error": "No market data available - need to update from IB first",
                "success": False
            })
        
        # Get REAL analysis from your market context
        analysis = _market_context.analyze_market()
        
        # Get REAL trading parameters
        params = _market_context.get_trading_parameters()
        
        # Structure the REAL data for agent consumption
        result = {
            "success": True,
            "underlying_price": _market_context.current_data.underlying_price,
            "vix": _market_context.current_data.vix,
            "regime": analysis.market_regime.value,
            "regime_confidence": analysis.regime_confidence,
            "trend_state": analysis.trend_state.value,
            "trend_strength": analysis.trend_strength,
            "movement_stats": {
                "atr_points": analysis.movement_stats.atr_points,
                "atr_percent": analysis.movement_stats.atr_percent,
                "expected_move": analysis.movement_stats.expected_move_1std
            },
            "parameters": {
                "wing_width": params["wing_width"],
                "min_credit_ic": params["min_credit_ic"],
                "min_credit_put": params["min_credit_put"],
                "min_credit_call": params["min_credit_call"],
                "strike_distance_pct": params["strike_distance_pct"],
                "can_trade": params["can_trade"]
            },
            "recommendation": {
                "strategy": analysis.recommended_strategy.value,
                "confidence": analysis.confidence_level,
                "rationale": analysis.rationale
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logging.error(f"Market analysis failed: {e}")
        return json.dumps({"error": str(e), "success": False})


@tool
async def get_account_status() -> str:
    """
    Get REAL account status from Interactive Brokers.
    
    Returns:
        JSON with actual account balance, buying power, positions
    """
    try:
        if not _execution_engine.connected:
            await _execution_engine.connect(paper=True)
        
        # Get REAL account summary from IB
        account_summary = await _execution_engine.get_account_summary()
        
        # Get REAL positions from IB
        positions = await _execution_engine.get_positions()
        
        result = {
            "success": True,
            "account": {
                "net_liquidation": account_summary.get("NetLiquidation", 0),
                "available_funds": account_summary.get("AvailableFunds", 0),
                "buying_power": account_summary.get("BuyingPower", 0),
                "unrealized_pnl": account_summary.get("UnrealizedPnL", 0),
                "realized_pnl": account_summary.get("RealizedPnL", 0)
            },
            "positions_count": len(positions),
            "positions": [
                {
                    "symbol": pos.contract.localSymbol,
                    "position": pos.position,
                    "avg_cost": pos.avgCost,
                    "unrealized_pnl": pos.unrealizedPNL
                } for pos in positions
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logging.error(f"Failed to get account status: {e}")
        return json.dumps({"error": str(e), "success": False})