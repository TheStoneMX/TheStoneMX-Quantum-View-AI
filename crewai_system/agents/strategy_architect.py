# crewai_system/agents/strategy_architect.py
"""
Strategy Architect Agent - The Master Builder
==============================================
Veteran strategist with 30 years building 0DTE option structures.
Knows intuitively which strategies thrive in each market condition.

Key Enhancements:
- VIX-regime aware strategy selection (learned, not hardcoded)
- Pattern-based strategy optimization
- Aggressive learning from outcomes
- Professional execution focus

Author: Quantum View AI Trading System
Version: 2.0
Date: December 2024
"""

from crewai import Agent, Task, Crew 
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
import numpy as np

from ..config.config_loader import TRADING_CONFIG
from ..memory.persistence import TRADING_MEMORY
from ..config.llm_factory import LLMFactory


class StrategyArchitectAgent:
    """
    Strategy Architect - The master builder of option structures.
    
    30-year veteran who knows every nuance of 0DTE option strategies.
    Has built thousands of trades and learned which structures work
    in each VIX regime through experience, not rules.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Strategy Architect agent.
        
        Args:
            tools: List of strategy-related tools
        """
        self.logger = logging.getLogger("StrategyArchitect")
        self.config = TRADING_CONFIG
        
        # Initialize LLM
        self.llm = LLMFactory.make(self.config, timeout_key="strategy_selection")
        
        # Load strategy performance patterns
        self._load_strategy_patterns()
        
        # Create the veteran architect agent
        self.agent = Agent(
            role="Master Options Strategy Architect - 30 Year Veteran",
            
            goal="""Design optimal 0DTE option strategies that exploit current market 
            conditions. Select structures based on deep pattern recognition and historical 
            edge, not mechanical rules. Maximize probability of profit while respecting risk.""",
            
            backstory="""You've been building option strategies since the CBOE floor days. 
            You've seen every market regime and know which strategies actually work versus 
            which just look good on paper.
            
            You learned the hard way that VIX below 15 doesn't mean "sell premium blindly" - 
            you were there in February 2018 when short vol strategies blew up. You know that 
            VIX 20 after a spike down from 35 is totally different from VIX 20 grinding up 
            from 12.
            
            Your expertise isn't from textbooks but from thousands of trades:
            - You know iron condors work brilliantly in VIX 16-22 with range-bound markets
            - You've learned butterflies excel when VIX is compressed below 14 for days
            - You recognize when to use broken-wing structures versus balanced ones
            - You know exactly when put spreads outperform calls (it's not just direction)
            
            You don't think "VIX is 14, use strategy X" - you think "VIX at 14 with this 
            momentum, after this pattern, with these levels nearby... perfect for a skip-strike 
            butterfly with 2:1 put skew."
            
            You size positions based on regime volatility, not fixed rules. You know when 
            1 contract is aggressive and when 5 contracts is conservative. Most importantly, 
            you know when NOT to trade - sometimes the best strategy is patience.
            
            You understand market maker positioning for 0DTE:
            - Wide spreads ($0.50+) = MMs are scared, reduce size or skip
            - Tight spreads at round strikes = MM inventory build, good liquidity
            - Skew changes tell you where MMs expect moves (put skew = downside fear)
            - Unusual volume at specific strikes = MM hedging, potential pin
            - Spreads widening into close = MMs pulling quotes, exit immediately""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=self.config.max_iterations,
            memory=True  # Enable memory for pattern learning
        )
        
        # Enhanced strategy design prompt
        self.strategy_prompt = """
        Design the optimal 0DTE strategy based on pattern recognition and experience.
        
        Market Analysis from Analyst:
        {market_analysis}
        
        Available Options Chain:
        {options_chain}
        
        Historical Performance in Similar Conditions:
        {performance_patterns}
        
        Risk Parameters:
        - Max contracts: {max_contracts}
        - Min credit IC: ${min_credit_ic}
        - Min credit spreads: ${min_credit_spread}
        - Wing width: {wing_width_min}-{wing_width_max}
        
        Recent Strategy Performance (last 30 days):
        {recent_performance}
        
        CRITICAL: Return ONLY valid JSON. No commentary before or after.
        Validate your response is parseable JSON before returning.
        
        Provide strategy specification in this EXACT JSON structure:
        {{
            "strategy_type": "iron_condor|put_spread|call_spread|butterfly|skip",
            "strategy_variant": "balanced|skewed|broken_wing|skip_strike|etc",
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
            "expected_value": number,
            "breakeven_points": [numbers],
            "regime_alignment": {{
                "vix_regime": "current VIX context",
                "strategy_fit": "why this strategy fits",
                "historical_edge": "specific edge in this setup"
            }},
            "position_sizing_rationale": {{
                "base_size": number,
                "regime_adjustment": number,
                "confidence_adjustment": number,
                "final_size": number
            }},
            "key_risks": ["specific risks"],
            "adjustment_plan": {{
                "trigger_conditions": ["condition1"],
                "adjustment_strategy": "description"
            }},
            "alternative_if_rejected": {{
                "strategy": "backup strategy type",
                "modification": "how to modify"
            }},
            "context_for_risk_manager": {{
                "critical_levels": [numbers],
                "max_adverse_excursion": number,
                "recommended_stops": "description",
                "time_decay_schedule": "description"
            }},
            "confidence": 0-100,
            "architect_notes": "Key insight about this trade"
        }}
        """
    
    def design_strategy(self,
                        market_analysis: Dict[str, Any],
                        options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Design optimal trading strategy based on pattern recognition.
        
        Args:
            market_analysis: Analysis from Market Analyst (with context)
            options_chain: Available options from market data
            
        Returns:
            Complete strategy specification with context
        """
        try:
            # Extract VIX regime and context from analyst
            vix_context = market_analysis.get("vix_analysis", {})
            analyst_context = market_analysis.get("context_for_strategy_architect", {})
            
            # Get performance patterns for this regime
            performance_patterns = self._get_regime_performance(
                vix_level=vix_context.get("level", 20),
                vix_regime=vix_context.get("regime", "normal"),
                trend=market_analysis.get("trend_analysis", {}).get("primary_trend", "neutral")
            )
            
            # Get recent performance for adaptation
            recent_performance = self._get_recent_strategy_performance()
            
            # Prepare strategy request with full context
            strategy_request = self.strategy_prompt.format(
                market_analysis=json.dumps(market_analysis, indent=2),
                options_chain=json.dumps(options_chain[:20], indent=2),
                performance_patterns=json.dumps(performance_patterns, indent=2),
                recent_performance=json.dumps(recent_performance, indent=2),
                max_contracts=self.config.max_contracts_per_trade,
                min_credit_ic=self.config.min_credit_ic,
                min_credit_spread=self.config.min_credit_put,
                wing_width_min=self.config.wing_width_min,
                wing_width_max=self.config.wing_width_max
            )
            
            # Get strategy from veteran architect
            self.logger.info("Designing strategy based on pattern recognition...")
            # response = self.agent.execute(strategy_request)
            #---------------------------------------------------------------
            task = Task(
                description=strategy_request,
                agent=self.agent,
                expected_output="A valid JSON object with strategy_type, strikes, contracts, expected_credit, and other strategy design fields as specified in the prompt"
            )

            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=self.config.crew_verbose
            )

            response = crew.kickoff()
            #---------------------------------------------------------------            
            # response is a CrewOutput    
            response_text = getattr(response, "raw", None)
            if not response_text and hasattr(response, "tasks_output") and response.tasks_output:
                response_text = getattr(response.tasks_output[0], "raw", None)
                
            if response_text is None:
                self.logger.error("Risk assessment response is empty.")
                return self._reject_trade("Risk assessment response is empty", critical=True)
            try:
                strategy = json.loads(response_text) # <-- Response text to JSON 
            except Exception as e:
                self.logger.error(f"Failed to parse risk assessment JSON: {e}")
                return self._reject_trade("Risk assessment response invalid JSON", critical=True)          
            
            # Parse and enhance response
            # strategy = json.loads(response)
            
            # Validate that required fields exist
            if "strategy_type" not in strategy:
                self.logger.error(f"Strategy missing required field 'strategy_type'. Got: {strategy.keys()}")
                return self._skip_strategy("Invalid strategy response from agent - missing strategy_type")
            
            strategy["timestamp"] = datetime.now().isoformat()
            strategy["architect"] = "strategy_architect_veteran"
            strategy["market_context"] = {
                "vix_level": vix_context.get("level"),
                "vix_regime": vix_context.get("regime"),
                "pattern": market_analysis.get("pattern_recognition", {}).get("current_pattern")
            }
            
            # Validate against learned patterns
            if self._validate_strategy(strategy, market_analysis, performance_patterns):
                self.logger.info(
                    f"Strategy: {strategy['strategy_type']} "
                    f"({strategy.get('strategy_variant', 'standard')}), "
                    f"Size: {strategy['contracts']}, "
                    f"Credit: ${strategy['expected_credit']:.2f}, "
                    f"Confidence: {strategy['confidence']}%"
                )
                return strategy
            else:
                self.logger.warning("Strategy failed pattern validation, recommending skip")
                return self._skip_strategy("Failed pattern-based validation")
                
        except Exception as e:
            self.logger.error(f"Strategy design failed: {e}")
            return self._skip_strategy(str(e))

    def _reject_trade(self, reason: str, critical: bool = False) -> Dict[str, Any]:
        """
        Create a rejection response with full context.
        
        Args:
            reason: Why trade is rejected
            critical: Whether this is a critical rejection
            
        Returns:
            Rejection assessment
        """
        return {
            "approval": "rejected",
            "risk_score": 100 if critical else 80,
            "pattern_recognition": {
                "risk_regime": "dangerous" if critical else "elevated"
            },
            "position_sizing": {
                "final_contracts": 0,
                "sizing_rationale": "Trade rejected"
            },
            "risk_factors": [
                {
                    "factor": reason,
                    "severity": "critical" if critical else "high",
                    "mitigation": "Skip this trade"
                }
            ],
            "context_for_cto": {
                "key_concerns": [reason],
                "override_recommendation": "skip"
            },
            "confidence": 100,
            "risk_manager_notes": f"Rejected: {reason}",
            "timestamp": datetime.now().isoformat()
        }
            
    def propose_alternative(self,
                            original_strategy: Dict[str, Any],
                            rejection_reason: str,
                            market_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose alternative strategy after IB rejection.
        
        Uses learned patterns to adapt strategy.
        
        Args:
            original_strategy: The rejected strategy
            rejection_reason: Why IB rejected it
            market_snapshot: Current market conditions
            
        Returns:
            Alternative strategy specification
        """
        # Analyze rejection pattern
        rejection_patterns = self._analyze_rejection_patterns(rejection_reason)
        
        alternative_prompt = f"""
        The broker rejected our trade. Design an alternative using your experience.
        
        Original Strategy: {json.dumps(original_strategy, indent=2)}
        Rejection Reason: {rejection_reason}
        Rejection Pattern Analysis: {json.dumps(rejection_patterns, indent=2)}
        Current Market: {json.dumps(market_snapshot, indent=2)}
        
        Based on similar rejections, design alternative with same structure as original.
        Apply lessons learned from past rejections.
        """
        
        try:
            response = self.agent.execute(alternative_prompt)
            alternative = json.loads(response)
            
            self.logger.info(
                f"Alternative: {alternative.get('strategy_variant', 'modified')} "
                f"{alternative['strategy_type']}"
            )
            
            return alternative
            
        except Exception as e:
            self.logger.error(f"Failed to propose alternative: {e}")
            return self._skip_strategy("Cannot create viable alternative")
    
    def _load_strategy_patterns(self) -> None:
        """Load historical strategy performance patterns."""
        self.patterns = TRADING_MEMORY.get_strategy_patterns()
        self.logger.info(f"Loaded patterns for {len(self.patterns)} strategy-regime combinations")
    
    def _get_regime_performance(self, 
                                vix_level: float,
                                vix_regime: str,
                                trend: str) -> Dict[str, Any]:
        """
        Get historical performance for strategies in this regime.
        
        Args:
            vix_level: Current VIX level
            vix_regime: VIX regime classification
            trend: Market trend direction
            
        Returns:
            Performance statistics by strategy
        """
        # Query for similar regime performance
        regime_data = TRADING_MEMORY.get_regime_performance(
            vix_range=(vix_level - 3, vix_level + 3),
            trend=trend,
            lookback_days=180  # 6 months of patterns
        )
        
        if not regime_data:
            return {"status": "no_regime_data"}
        
        # Aggregate by strategy type
        strategy_stats = {}
        for strategy_type in ["iron_condor", "put_spread", "call_spread", "butterfly"]:
            strategy_data = regime_data.get(strategy_type, {})
            if strategy_data:
                strategy_stats[strategy_type] = {
                    "trades": strategy_data["count"],
                    "win_rate": strategy_data["win_rate"],
                    "avg_return": strategy_data["avg_return"],
                    "avg_credit": strategy_data["avg_credit"],
                    "sharpe": strategy_data.get("sharpe", 0),
                    "max_drawdown": strategy_data.get("max_dd", 0),
                    "best_variant": strategy_data.get("best_variant", "standard")
                }
        
        # Rank strategies by expected value
        ranked = sorted(
            strategy_stats.items(),
            key=lambda x: x[1]["avg_return"] * x[1]["win_rate"],
            reverse=True
        )
        
        return {
            "regime_description": f"VIX {vix_regime} ({vix_level:.1f}) with {trend} trend",
            "strategy_rankings": ranked,
            "best_historical": ranked[0][0] if ranked else None,
            "avoid_strategy": ranked[-1][0] if len(ranked) > 1 else None
        }
    
    def _get_recent_strategy_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get recent strategy performance for aggressive learning.
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            Recent performance metrics
        """
        recent = TRADING_MEMORY.get_recent_trades(days=days)
        
        if recent.empty:
            return {"status": "no_recent_trades"}
        
        # Weight more recent trades higher (aggressive learning)
        weights = np.linspace(0.5, 2.0, len(recent))  # 2x weight for most recent
        
        performance = {}
        for strategy in recent["strategy"].unique():
            strategy_trades = recent[recent["strategy"] == strategy]
            strategy_weights = weights[-len(strategy_trades):]
            
            weighted_returns = strategy_trades["return"].values * strategy_weights
            weighted_wins = (strategy_trades["return"] > 0).values * strategy_weights
            
            performance[strategy] = {
                "recent_trades": len(strategy_trades),
                "weighted_win_rate": float(weighted_wins.sum() / strategy_weights.sum()),
                "weighted_avg_return": float(weighted_returns.sum() / strategy_weights.sum()),
                "momentum": "improving" if weighted_returns[-3:].mean() > weighted_returns.mean() else "declining"
            }
        
        return performance
    
    def _validate_strategy(self,
                           strategy: Dict[str, Any],
                           market_analysis: Dict[str, Any],
                           performance_patterns: Dict[str, Any]) -> bool:
        """
        Validate strategy against learned patterns.
        
        Args:
            strategy: Proposed strategy
            market_analysis: Current market conditions
            performance_patterns: Historical performance data
            
        Returns:
            True if strategy aligns with successful patterns
        """
        # Skip is always valid
        if strategy["strategy_type"] == "skip":
            return True
        
        # Basic sanity checks
        if strategy["contracts"] <= 0 or strategy["contracts"] > self.config.max_contracts_per_trade:
            return False
        
        if strategy["expected_credit"] <= 0:
            return False
        
        # Pattern-based validation
        best_historical = performance_patterns.get("best_historical")
        avoid_strategy = performance_patterns.get("avoid_strategy")
        
        # Warn if using historically poor strategy
        if strategy["strategy_type"] == avoid_strategy:
            self.logger.warning(
                f"Strategy {avoid_strategy} has poor historical performance in this regime"
            )
            # Don't hard reject, but reduce confidence
            strategy["confidence"] = min(strategy["confidence"], 40)
        
        # Boost confidence if using best historical
        if strategy["strategy_type"] == best_historical:
            strategy["confidence"] = min(100, strategy["confidence"] + 10)
        
        # Validate probability of profit makes sense
        if strategy["probability_profit"] < 25:  # Too unlikely
            return False
        
        return True
    
    def _analyze_rejection_patterns(self, rejection_reason: str) -> Dict[str, Any]:
        """
        Analyze rejection patterns to learn from them.
        
        Args:
            rejection_reason: IB's rejection message
            
        Returns:
            Analysis of rejection pattern
        """
        # Get similar rejections from memory
        similar_rejections = TRADING_MEMORY.get_rejection_patterns(rejection_reason)
        
        if not similar_rejections:
            return {"status": "novel_rejection"}
        
        # Find what worked after similar rejections
        successful_adaptations = []
        for rejection in similar_rejections:
            if rejection.get("retry_successful"):
                successful_adaptations.append({
                    "original": rejection["original_strategy"],
                    "successful": rejection["successful_strategy"],
                    "change": rejection["key_change"]
                })
        
        return {
            "rejection_type": self._classify_rejection(rejection_reason),
            "similar_cases": len(similar_rejections),
            "successful_adaptations": successful_adaptations,
            "recommended_change": successful_adaptations[0]["change"] if successful_adaptations else "widen_strikes"
        }
    
    def _classify_rejection(self, reason: str) -> str:
        """Classify rejection reason into categories."""
        reason_lower = reason.lower()
        
        if "margin" in reason_lower:
            return "insufficient_margin"
        elif "width" in reason_lower or "strike" in reason_lower:
            return "invalid_strikes"
        elif "price" in reason_lower or "credit" in reason_lower:
            return "price_moved"
        elif "size" in reason_lower or "contract" in reason_lower:
            return "size_issue"
        else:
            return "other"
    
    def _skip_strategy(self, reason: str) -> Dict[str, Any]:
        """
        Return a skip strategy with full context.
        
        Args:
            reason: Why we're skipping
            
        Returns:
            Skip strategy specification
        """
        return {
            "strategy_type": "skip",
            "strategy_variant": "none",
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
            "regime_alignment": {
                "vix_regime": "not_applicable",
                "strategy_fit": reason,
                "historical_edge": "none"
            },
            "context_for_risk_manager": {
                "critical_levels": [],
                "recommended_stops": "not_applicable"
            },
            "confidence": 0,
            "architect_notes": f"Skipping: {reason}",
            "timestamp": datetime.now().isoformat()
        }