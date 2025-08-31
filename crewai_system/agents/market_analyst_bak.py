# crewai_system/agents/market_analyst.py
"""
Market Analyst Agent
====================
Specializes in market condition analysis and regime identification.
This agent interprets market data to provide actionable intelligence
for strategy selection.

Key Responsibilities:
- Analyze market regime (trending/ranging/volatile)
- Assess trend strength and confidence
- Monitor VIX levels and implications
- Identify support/resistance levels
- Provide movement statistics

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

from crewai import Agent
from langchain_ollama import OllamaLLM
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from ..config.config_loader import TRADING_CONFIG
from ..config.timezone_handler import TIMEZONE_HANDLER
from ..memory.persistence import TRADING_MEMORY

from crewai_system.config.llm_factory import LLMFactory

class MarketAnalystAgent:
    """
    Market Analyst - The eyes of the trading system.
    
    This agent continuously analyzes market conditions and provides
    structured assessments for other agents to consume.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Market Analyst agent.
        
        Args:
            tools: List of tools for market analysis
        """
        self.logger = logging.getLogger("MarketAnalyst")
        self.config = TRADING_CONFIG
        
       
        self.llm = LLMFactory.make(self.config, timeout_key="market_analysis")
               
        # Create the agent with clear role definition
        self.agent = Agent(
            role="Senior Market Analyst",
            
            goal="""Analyze market conditions for 0DTE options trading opportunities.
            Provide precise, actionable intelligence about market regime, trend strength,
            and volatility conditions. Focus on NQ futures and factors affecting
            intraday option pricing.""",
            
            backstory="""You are a veteran market analyst with 20 years of experience
            in derivatives trading, specializing in 0DTE options on index futures.
            You excel at identifying market regimes and understanding how volatility,
            trend, and time decay interact. You've seen every market condition and
            know that 0DTE options require precise timing and careful analysis.
            
            You work for a trader based in Spain who trades US markets, so you
            understand the importance of clear communication across time zones.
            Your analysis must be structured, quantitative, and actionable.""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,  # Analysts don't delegate
            max_iter=2,  # Limit iterations for speed
            cache=True
        )
        
        # Analysis prompt template for consistent output
        self.analysis_prompt = """
        Analyze the current market conditions and provide a structured assessment.
        
        Current Data:
        {market_data}
        
        Recent Performance:
        {recent_performance}
        
        Time Factors:
        {time_factors}
        
        Provide your analysis in the following JSON structure:
        {{
            "regime": "trending|ranging|volatile|squeeze|breakout",
            "regime_confidence": 0-100,
            "trend_direction": "bullish|bearish|neutral",
            "trend_strength": -100 to +100,
            "volatility_assessment": {{
                "current_vix": number,
                "vix_regime": "low|normal|high|extreme",
                "implications": "string"
            }},
            "key_levels": {{
                "support": number,
                "resistance": number,
                "distance_to_support_pct": number,
                "distance_to_resistance_pct": number
            }},
            "movement_expectations": {{
                "expected_range": number,
                "directional_bias": "up|down|neutral",
                "confidence": 0-100
            }},
            "trading_recommendation": {{
                "favorable_strategies": ["iron_condor", "put_spread", "call_spread"],
                "avoid_strategies": [],
                "rationale": "string"
            }},
            "risk_factors": ["string"],
            "confidence_score": 0-100
        }}
        """
    
    def analyze_market(self, 
                      market_data: Dict[str, Any],
                      lookback_trades: int = 10) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis.
        
        Args:
            market_data: Current market data from tools
            lookback_trades: Number of recent trades to consider
            
        Returns:
            Structured market analysis for consumption by other agents
        """
        try:
            # Get recent trading performance for context
            recent_performance = TRADING_MEMORY.get_performance_stats(days=7)
            
            # Get time factors
            time_factors = {
                "minutes_to_close": TIMEZONE_HANDLER.minutes_to_close(),
                "market_phase": TIMEZONE_HANDLER.get_market_phase()[0],
                "spain_window": TIMEZONE_HANDLER.get_spain_trading_window(),
                "time_display": TIMEZONE_HANDLER.format_times_for_logging()
            }
            
            # Format the analysis request
            analysis_request = self.analysis_prompt.format(
                market_data=json.dumps(market_data, indent=2),
                recent_performance=json.dumps(recent_performance, indent=2),
                time_factors=json.dumps(time_factors, indent=2)
            )
            
            # Get analysis from agent
            self.logger.info("Performing market analysis...")
            response = self.agent.execute(analysis_request)
            
            # Parse JSON response
            try:
                analysis = json.loads(response)
                analysis["timestamp"] = datetime.now().isoformat()
                analysis["analyst"] = "market_analyst"
                
                # Log key findings
                self.logger.info(
                    f"Market Analysis Complete: "
                    f"Regime={analysis['regime']} ({analysis['regime_confidence']}%), "
                    f"Trend={analysis['trend_strength']:+.0f}, "
                    f"VIX={analysis['volatility_assessment']['current_vix']}"
                )
                
                return analysis
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse agent response: {e}")
                # Return a safe default
                return self._default_analysis(market_data)
                
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return self._default_analysis(market_data)
    
    def _default_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide safe default analysis when agent fails.
        
        This ensures the system can continue even if analysis fails.
        
        Args:
            market_data: Current market data
            
        Returns:
            Conservative default analysis
        """
        return {
            "regime": "unknown",
            "regime_confidence": 0,
            "trend_direction": "neutral",
            "trend_strength": 0,
            "volatility_assessment": {
                "current_vix": market_data.get("vix", 20),
                "vix_regime": "normal",
                "implications": "Unable to assess - using conservative parameters"
            },
            "trading_recommendation": {
                "favorable_strategies": [],
                "avoid_strategies": ["all"],
                "rationale": "Analysis failed - avoiding trades for safety"
            },
            "risk_factors": ["Analysis system offline"],
            "confidence_score": 0,
            "timestamp": datetime.now().isoformat(),
            "analyst": "market_analyst_default"
        }
        
        