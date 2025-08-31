# debug_agents.py
"""
Debug Script - System Component Test
Tests each component individually to find where the agent execution is failing
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger("DEBUG")

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules import correctly."""
    logger.info("Testing imports...")
    
    try:
        from crewai_system.crews.trading_crew import TradingCrew
        logger.info("✓ TradingCrew imported")
    except Exception as e:
        logger.error(f"✗ TradingCrew import failed: {e}")
        return False
    
    try:
        from crewai_system.agents.market_analyst import MarketAnalystAgent
        logger.info("✓ MarketAnalystAgent imported")
    except Exception as e:
        logger.error(f"✗ MarketAnalystAgent import failed: {e}")
        return False
    
    try:
        from crewai_system.agents.strategy_architect import StrategyArchitectAgent
        logger.info("✓ StrategyArchitectAgent imported")
    except Exception as e:
        logger.error(f"✗ StrategyArchitectAgent import failed: {e}")
        return False
    
    try:
        from crewai_system.agents.risk_manager import RiskManagerAgent
        logger.info("✓ RiskManagerAgent imported")
    except Exception as e:
        logger.error(f"✗ RiskManagerAgent import failed: {e}")
        return False
    
    try:
        from crewai_system.agents.chief_trading_officer import ChiefTradingOfficer
        logger.info("✓ ChiefTradingOfficer imported")
    except Exception as e:
        logger.error(f"✗ ChiefTradingOfficer import failed: {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test if agents initialize correctly."""
    logger.info("\nTesting agent initialization...")
    
    try:
        from crewai_system.agents.market_analyst import MarketAnalystAgent
        
        # Create minimal tools list
        tools = []
        
        # Try to create agent
        analyst = MarketAnalystAgent(tools)
        logger.info("✓ MarketAnalystAgent created")
        
        # Check if agent exists
        if hasattr(analyst, 'agent'):
            logger.info("✓ Agent object exists")
            
            # Check agent properties
            logger.info(f"  Role: {analyst.agent.role}")
            logger.info(f"  LLM: {type(analyst.agent.llm)}")
        else:
            logger.error("✗ No agent object found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_market_analysis():
    """Test if market analysis works with mock data."""
    logger.info("\nTesting market analysis...")
    
    try:
        from crewai_system.agents.market_analyst import MarketAnalystAgent
        
        # Create agent
        analyst = MarketAnalystAgent([])
        
        # Create mock market data
        mock_data = {
            "success": True,
            "underlying_price": 375.50,
            "vix": 18.5,
            "volume": 50000000,
            "high": 377.0,
            "low": 374.0,
            "open": 375.0,
            "trend": "neutral",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Calling analyze_market...")
        
        # Try to analyze
        result = analyst.analyze_market(mock_data)
        
        if result:
            logger.info("✓ Analysis returned result")
            logger.info(f"  Regime: {result.get('regime', 'N/A')}")
            logger.info(f"  Confidence: {result.get('confidence_score', 0)}")
            logger.info(f"  VIX Regime: {result.get('vix_analysis', {}).get('regime', 'N/A')}")
            
            # Check if it's defaulting
            if result.get('confidence_score', 0) == 0:
                logger.warning("⚠ Analysis defaulted to safe mode")
        else:
            logger.error("✗ Analysis returned None")
            return False
            
    except Exception as e:
        logger.error(f"✗ Market analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_crew_initialization():
    """Test if TradingCrew initializes and has required methods."""
    logger.info("\nTesting TradingCrew...")
    
    try:
        from crewai_system.crews.trading_crew import TradingCrew
        
        # Create crew
        crew = TradingCrew()
        logger.info("✓ TradingCrew created")
        
        # Check for expected methods
        if hasattr(crew, 'analyze_and_trade'):
            logger.info("✓ analyze_and_trade method exists")
        else:
            logger.error("✗ analyze_and_trade method missing")
            
        # Check for agents
        if hasattr(crew, 'market_analyst'):
            logger.info("✓ market_analyst exists")
        if hasattr(crew, 'strategy_architect'):
            logger.info("✓ strategy_architect exists")
        if hasattr(crew, 'risk_manager'):
            logger.info("✓ risk_manager exists")
        if hasattr(crew, 'chief_trading_officer'):
            logger.info("✓ chief_trading_officer exists")
            
        # Try to execute
        logger.info("\nAttempting to execute analyze_and_trade...")
        result = crew.analyze_and_trade()
        
        if result:
            logger.info(f"✓ Got result: {result.get('decision', 'N/A')}")
        else:
            logger.error("✗ No result returned")
            
    except AttributeError as e:
        logger.error(f"✗ Missing attribute: {e}")
    except Exception as e:
        logger.error(f"✗ TradingCrew test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def check_llm_configuration():
    """Check if LLM is configured correctly."""
    logger.info("\nChecking LLM configuration...")
    
    try:
        from crewai_system.config.llm_factory import LLMFactory
        from crewai_system.config.config_loader import TRADING_CONFIG
        
        # Try to create LLM
        llm = LLMFactory.make(TRADING_CONFIG)
        logger.info(f"✓ LLM created: {type(llm)}")
        
        # Check if it's the right type
        if 'ChatOpenAI' in str(type(llm)):
            logger.info("✓ Using OpenAI LLM")
        elif 'Anthropic' in str(type(llm)):
            logger.info("✓ Using Anthropic LLM")
        else:
            logger.warning(f"⚠ Unknown LLM type: {type(llm)}")
            
    except Exception as e:
        logger.error(f"✗ LLM configuration failed: {e}")
        return False
    
    return True

def main():
    """Run all debug tests."""
    logger.info("="*60)
    logger.info("AGENT SYSTEM DEBUG")
    logger.info("="*60)
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Agent Initialization", test_agent_initialization),
        ("LLM Configuration", check_llm_configuration),
        ("Market Analysis", test_market_analysis),
        ("Crew System", test_crew_initialization)
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*40}")
        
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name}: {status}")
    
    # Diagnosis
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSIS")
    logger.info("="*60)
    
    if not results.get("Imports", False):
        logger.error("Critical: Module imports failing - check file paths and dependencies")
    elif not results.get("LLM Configuration", False):
        logger.error("Critical: LLM not configured - check API keys and config")
    elif not results.get("Agent Initialization", False):
        logger.error("Critical: Agents not initializing - check CrewAI installation")
    elif not results.get("Market Analysis", False):
        logger.error("Critical: Analysis failing - agents may be timing out or erroring")
    elif not results.get("Crew System", False):
        logger.error("Critical: Crew orchestration broken - check TradingCrew implementation")
    else:
        logger.info("All basic systems working - issue may be in execution flow")

if __name__ == "__main__":
    main()