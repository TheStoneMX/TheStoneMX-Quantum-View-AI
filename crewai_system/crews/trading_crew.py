# crewai_system/crews/trading_crew.py
"""
Trading Crew Configuration
===========================
Assembles all agents into a functioning crew.
"""

from crewai import Crew, Process
from typing import Dict, Any
import logging
import datetime

from ..agents.market_analyst import MarketAnalystAgent
from ..agents.strategy_architect import StrategyArchitectAgent
from ..agents.risk_manager import RiskManagerAgent
from ..agents.chief_trading_officer import ChiefTradingOfficer
from ..tools.market_tools import (
    get_real_market_data,
    get_real_option_chain,
    analyze_market_with_real_data,
    get_account_status
)
from ..tools.execution_tools import execute_real_trade
from ..config.config_loader import TRADING_CONFIG


class TradingCrew:
    """
    Assembles and manages the complete trading crew.
    """
    
    def __init__(self):
        """Initialize the trading crew with all agents and tools."""
        self.logger = logging.getLogger("TradingCrew")
        
        # Initialize tools
        self.market_tools = [
            get_real_market_data,
            analyze_market_with_real_data,
            get_account_status
        ]
        
        self.strategy_tools = [
            get_real_option_chain,
            analyze_market_with_real_data
        ]
        
        self.execution_tools = [
            execute_real_trade,
            get_account_status
        ]
        
        # Initialize agents
        self.logger.info("Initializing agents...")
        self.market_analyst = MarketAnalystAgent(tools=self.market_tools)
        self.strategy_architect = StrategyArchitectAgent(tools=self.strategy_tools)
        self.risk_manager = RiskManagerAgent(tools=self.market_tools)
        
        try:
            self.cto = ChiefTradingOfficer(
            market_analyst=self.market_analyst,
            strategy_architect=self.strategy_architect,
            risk_manager=self.risk_manager,
            execution_specialist=None,  # We're not using this
            tools=self.execution_tools
        )
        except Exception as e:
            print(e)
        # Initialize CTO with references to all specialists

        
        # Create the crew
        self.crew = Crew(
            agents=[
                self.market_analyst.agent,
                self.strategy_architect.agent,
                self.risk_manager.agent,
                self.cto.agent
            ],
            process=Process.sequential,  # For 0DTE, sequential is safest
            verbose=TRADING_CONFIG.crew_verbose,
            memory=TRADING_CONFIG.crew_memory,
            cache=True,
            max_iter=TRADING_CONFIG.max_iterations
        )
        
        self.logger.info("Trading crew assembled and ready")
    
    def analyze_and_trade(self) -> Dict[str, Any]:
        """
        Run the complete analysis and trading workflow.
        
        Returns:
            Results of the trading decision and execution
        """
        # The CTO orchestrates everything
        decision, details = self.cto.orchestrate_trading_decision()
        
        return {
            "decision": decision.value,
            "details": details,
            "timestamp": datetime.datetime.now().isoformat()
        }