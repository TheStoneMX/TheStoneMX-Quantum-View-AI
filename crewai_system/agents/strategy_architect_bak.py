# crewai_system/agents/strategy_architect.py
"""
Strategy Architect Agent
========================
Designs and selects optimal trading strategies based on market conditions.
This agent translates market analysis into specific, executable trade setups.

Key Responsibilities:
- Select appropriate strategy (IC, put spread, call spread)
- Calculate optimal strike prices
- Determine position sizing
- Validate trade viability
- Handle IB rejection recovery

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

from crewai import Agent
from langchain_ollama import OllamaLLM  as Ollama
from typing import Any
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from ..config.config_loader import TRADING_CONFIG
from ..memory.persistence import TRADING_MEMORY, TradeRecord

from crewai_system.config.llm_factory import LLMFactory

class StrategyArchitectAgent:
    """
    Strategy Architect - The brain of trade construction.
    
    This agent takes market analysis and constructs specific,
    executable trading strategies optimized for current conditions.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Strategy Architect agent.
        
        Args:
            tools: List of strategy-related tools
        """
        self.logger = logging.getLogger("StrategyArchitect")
        self.config = TRADING_CONFIG
                
        self.llm = LLMFactory.make(self.config, timeout_key="strategy_selection")
               
        # Create the agent
        self.agent = Agent(
            role="Senior Options Strategy Architect",
            
            goal="""Design optimal 0DTE option strategies that maximize probability
            of profit while managing risk. Select appropriate strategies, calculate
            precise strike prices, and determine optimal position sizing based on
            market conditions and risk parameters.""",
            
            backstory="""You are a quantitative strategist with deep expertise in
            options pricing and strategy construction. You've designed thousands of
            successful 0DTE trades and understand the delicate balance between
            collecting premium and managing gamma risk. 
            
            You know that 0DTE options decay rapidly but can move violently on
            price changes. Your strategies account for time decay, volatility skew,
            and the unique risks of expiration day. You're particularly skilled at
            adapting strategies when initial attempts are rejected by the broker.
            
            You work with a Spain-based trader, so you're aware that late-day
            trades require extra caution due to limited time for adjustments.""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=self.config.max_iterations,
        )
        
        # Strategy selection prompt
        self.strategy_prompt = """
        Design an optimal options strategy based on current conditions.
        
        Market Analysis:
        {market_analysis}
        
        Available Options Chain:
        {options_chain}
        
        Risk Parameters:
        - Max contracts: {max_contracts}
        - Min credit IC: ${min_credit_ic}
        - Min credit spreads: ${min_credit_spread}
        - Wing width bounds: {wing_width_min}-{wing_width_max}
        
        Recent Similar Trades:
        {similar_trades}
        
        Provide your strategy in the following JSON structure:
        {{
            "strategy_type": "iron_condor|put_spread|call_spread|skip",
            "strikes": {{
                "short_put": number or null,
                "long_put": number or null,
                "short_call": number or null,
                "long_call": number or null
            }},
            "contracts": number,
            "expected_credit": number,
            "max_risk": number,
            "probability_profit": 0-100,
            "breakeven_points": [numbers],
            "rationale": "string explaining the selection",
            "key_risks": ["string"],
            "alternative_considered": "string",
            "confidence": 0-100,
            "adjustments_if_rejected": {{
                "widen_strikes_by": number,
                "reduce_contracts_by": number,
                "alternative_strategy": "string"
            }}
        }}
        """
    
    def design_strategy(self,
                       market_analysis: Dict[str, Any],
                       options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Design optimal trading strategy based on conditions.
        
        Args:
            market_analysis: Analysis from Market Analyst
            options_chain: Available options from market data
            
        Returns:
            Complete strategy specification
        """
        try:
            # Get similar historical trades for learning
            similar_trades = self._get_similar_trades(market_analysis)
            
            # Prepare strategy request
            strategy_request = self.strategy_prompt.format(
                market_analysis=json.dumps(market_analysis, indent=2),
                options_chain=json.dumps(options_chain[:20], indent=2),  # Limit for context
                max_contracts=self.config.max_contracts_per_trade,
                min_credit_ic=self.config.min_credit_ic,
                min_credit_spread=self.config.min_credit_put,
                wing_width_min=self.config.wing_width_min,
                wing_width_max=self.config.wing_width_max,
                similar_trades=json.dumps(similar_trades, indent=2)
            )
            
            # Get strategy from agent
            self.logger.info("Designing trading strategy...")
            response = self.agent.execute(strategy_request)
            
            # Parse response
            strategy = json.loads(response)
            strategy["timestamp"] = datetime.now().isoformat()
            strategy["architect"] = "strategy_architect"
            
            # Validate strategy makes sense
            if self._validate_strategy(strategy, market_analysis):
                self.logger.info(
                    f"Strategy Designed: {strategy['strategy_type']} "
                    f"with {strategy['contracts']} contracts, "
                    f"Credit=${strategy['expected_credit']:.2f}, "
                    f"Confidence={strategy['confidence']}%"
                )
                return strategy
            else:
                self.logger.warning("Strategy validation failed, returning skip")
                return self._skip_strategy("Failed validation")
                
        except Exception as e:
            self.logger.error(f"Strategy design failed: {e}")
            return self._skip_strategy(str(e))
    
    def propose_alternative(self,
                           original_strategy: Dict[str, Any],
                           rejection_reason: str,
                           market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose alternative strategy after IB rejection.
        
        This is called when IB rejects our original trade.
        
        Args:
            original_strategy: The rejected strategy
            rejection_reason: Why IB rejected it
            market_analysis: Current market conditions
            
        Returns:
            Alternative strategy specification
        """
        alternative_prompt = f"""
        The broker rejected our trade. Design an alternative.
        
        Original Strategy: {json.dumps(original_strategy, indent=2)}
        Rejection Reason: {rejection_reason}
        Market Analysis: {json.dumps(market_analysis, indent=2)}
        
        Provide alternative with same JSON structure as before.
        Consider the rejection reason and adjust accordingly.
        """
        
        try:
            response = self.agent.execute(alternative_prompt)
            alternative = json.loads(response)
            
            self.logger.info(
                f"Alternative proposed: {alternative['strategy_type']} "
                f"with adjusted parameters"
            )
            
            return alternative
            
        except Exception as e:
            self.logger.error(f"Failed to propose alternative: {e}")
            return self._skip_strategy("Cannot create alternative")
    
    def _validate_strategy(self,
                          strategy: Dict[str, Any],
                          market_analysis: Dict[str, Any]) -> bool:
        """
        Validate strategy makes sense given conditions.
        
        Safety check to ensure agent's strategy is rational.
        
        Args:
            strategy: Proposed strategy
            market_analysis: Current market conditions
            
        Returns:
            True if strategy is valid
        """
        # Basic sanity checks
        if strategy["strategy_type"] == "skip":
            return True
        
        if strategy["contracts"] <= 0 or strategy["contracts"] > self.config.max_contracts_per_trade:
            return False
        
        if strategy["expected_credit"] <= 0:
            return False
        
        if strategy["probability_profit"] < 30:  # Less than 30% chance is too risky
            return False
        
        # Strategy-specific validation
        if strategy["strategy_type"] == "iron_condor":
            if not all([
                strategy["strikes"]["short_put"],
                strategy["strikes"]["long_put"],
                strategy["strikes"]["short_call"],
                strategy["strikes"]["long_call"]
            ]):
                return False
        
        return True
    
    def _get_similar_trades(self, market_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get similar historical trades for learning.
        
        Args:
            market_analysis: Current market conditions
            
        Returns:
            List of similar trade outcomes
        """
        # Get similar trades from memory
        df = TRADING_MEMORY.get_similar_trades(
            market_regime=market_analysis.get("regime", "unknown"),
            vix_level=market_analysis.get("volatility_assessment", {}).get("current_vix", 20),
            lookback_days=30
        )
        
        if df.empty:
            return []
        
        # Convert to list of dicts, focusing on key info
        trades = []
        for _, row in df.head(5).iterrows():  # Top 5 most recent similar
            trades.append({
                "strategy": row["strategy"],
                "credit": row["credit_received"],
                "outcome": row["realized_pnl"],
                "win": row["win"],
                "exit_reason": row["exit_reason"]
            })
        
        return trades
    
    def _skip_strategy(self, reason: str) -> Dict[str, Any]:
        """
        Return a skip strategy with reason.
        
        Args:
            reason: Why we're skipping
            
        Returns:
            Skip strategy specification
        """
        return {
            "strategy_type": "skip",
            "strikes": {
                "short_put": None,
                "long_put": None,
                "short_call": None,
                "long_call": None
            },
            "contracts": 0,
            "expected_credit": 0,
            "max_risk": 0,
            "probability_profit": 0,
            "rationale": f"Skipping trade: {reason}",
            "confidence": 0,
            "timestamp": datetime.now().isoformat(),
            "architect": "strategy_architect"
        }