# crewai_system/agents/market_analyst.py
"""
Market Analyst Agent - The Market Reader
=========================================
Veteran analyst with 30 years of experience reading market patterns.
Specializes in recognizing regime transitions and volatility patterns
that precede major moves.

Key Enhancements:
- Pattern recognition from historical VIX behavior
- Context-rich analysis for downstream agents
- Regime transition detection
- Professional concise communication

Author: Quantum View AI Trading System
Version: 2.0
Date: December 2024
"""

from crewai import Agent, Task, Crew 
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
import numpy as np

from ..config.config_loader import TRADING_CONFIG
from ..config.timezone_handler import TIMEZONE_HANDLER 
from ..memory.persistence import TRADING_MEMORY
from ..config.llm_factory import LLMFactory
from ..utils.prompt_enhancer import PROMPT_ENHANCER

import litellm

class MarketAnalystAgent:
    """
    Market Analyst - The eyes of the trading system.
    
    30-year veteran who has seen every market condition and recognizes
    patterns before they fully develop. Doesn't just report numbers,
    but interprets what they mean in context.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Market Analyst agent.
        
        Args:
            tools: List of tools for market analysis
        """
        self.logger = logging.getLogger("MarketAnalyst")
        self.config = TRADING_CONFIG
        
        # Initialize LLM with appropriate settings
        self.llm = LLMFactory.make(self.config, timeout_key="market_analysis")
        
        # Load historical patterns for pattern recognition
        self._load_market_patterns()
        
        # Create the veteran analyst agent
        self.agent = Agent(
            role="Senior Market Analyst - 30 Year Veteran",
            
            goal="""Provide deep market insight for 0DTE options trading by recognizing 
            patterns others miss. Focus on regime transitions, volatility compression/expansion 
            patterns, and microstructure signals that precede major moves.""",
            
            backstory="""You've been reading markets since the 1990s. You were there for 
            the dot-com bubble, 2008 crisis, COVID crash, and meme stock mania. You've seen 
            VIX at 9 and VIX at 80. You know that numbers without context are meaningless.
            
            You've learned that VIX below 15 isn't automatically bullish - you remember 
            February 5, 2018 when XIV imploded. You know VIX 20 in a grinding uptrend 
            is different from VIX 20 after a spike. You recognize when the market is 
            coiling for a move versus when it's genuinely calm.
            
            You don't just report that VIX is at 14 - you explain it's been compressed 
            for 8 days with declining volume, similar to August 2017 before the volatility 
            explosion. You see patterns in the tape that algorithms miss.
            
            Your analysis is concise but rich with context. You communicate in professional 
            trading desk language - no fluff, just signal.""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,
            memory=True  # Enable memory for pattern recognition
        )
        
        # Enhanced analysis prompt for pattern recognition
        self.analysis_prompt = """
        Analyze current market conditions with your 30 years of pattern recognition.
        
        Current Data:
        {market_data}
        
        Historical Patterns in Similar Conditions:
        {similar_patterns}
        
        Recent Market Behavior (last 10 days):
        {recent_behavior}
        
        Time Factors:
        {time_factors}
        
        Time Context Awareness for 0DTE:
        - Opening 30 min (9:30-10:00): Expect volatility, wider spreads, wait for clarity
        - Mid-morning (10-11 AM): Cleaner trends emerge, best entry window
        - Lunch hours (12-1 PM): Reduced volume, avoid new positions
        - Afternoon (2-3 PM): Theta acceleration zone, premium decay accelerates
        - Power Hour (3-4 PM): Gamma risk explodes, exit or defend only
        - Final 30 min: Close positions, assignment risk too high
        
        CRITICAL: Return ONLY valid JSON. No commentary before or after.
        Validate your response is parseable JSON before returning.
        
        Provide analysis in this EXACT JSON structure:
        {{
            "regime": "trending|ranging|volatile|compression|expansion",
            "regime_confidence": 0-100,
            "pattern_recognition": {{
                "current_pattern": "description",
                "similar_historical_setups": ["date1", "date2"],
                "pattern_implications": "what typically follows"
            }},
            "vix_analysis": {{
                "level": number,
                "regime": "crushed|low|normal|elevated|spiking",
                "context": "compression before expansion|post-spike normalization|grinding higher|etc",
                "days_in_regime": number,
                "percentile_30d": number,
                "percentile_90d": number
            }},
            "trend_analysis": {{
                "primary_trend": "bullish|bearish|neutral",
                "strength": -100 to +100,
                "momentum": "accelerating|steady|decelerating|reversing",
                "key_levels": {{
                    "support": number,
                    "resistance": number,
                    "pivot": number
                }}
            }},
            "microstructure_signals": [
                "signal description"
            ],
            "trading_implications": {{
                "favorable_setups": ["setup1", "setup2"],
                "setups_to_avoid": ["setup1"],
                "key_risks": ["risk1", "risk2"]
            }},
            "context_for_strategy_architect": {{
                "critical_observations": ["obs1", "obs2"],
                "regime_transition_risk": "low|medium|high",
                "recommended_approach": "aggressive|normal|cautious",
                "similar_profitable_setups": ["date1: strategy", "date2: strategy"]
            }},
            "confidence_score": 0-100,
            "analyst_notes": "Concise key insight"
        }}
        """
    
    def analyze_market(self, 
                       market_data: Dict[str, Any],
                       lookback_days: int = 10) -> Dict[str, Any]:
        """
        Perform veteran-level market analysis with pattern recognition.
        
        Args:
            market_data: Current market data from tools
            lookback_days: Days of history to consider for patterns
            
        Returns:
            Rich contextual analysis for downstream agents
        """
        try:
            # Get historical patterns similar to current setup
            similar_patterns = self._find_similar_historical_patterns(market_data)
            
            # Get recent market behavior for context
            recent_behavior = self._analyze_recent_behavior(lookback_days)
            
            # Get time factors
            time_factors = self._get_enhanced_time_factors()
            
            # Format the analysis request
            base_request = self.analysis_prompt.format(
                market_data=json.dumps(market_data, indent=2),
                similar_patterns=json.dumps(similar_patterns, indent=2),
                recent_behavior=json.dumps(recent_behavior, indent=2),
                time_factors=json.dumps(time_factors, indent=2)
            )
            
            # Enhance with current regime context
            minutes_to_close = TIMEZONE_HANDLER.get_minutes_to_close()
            vix_level = market_data.get("vix", 20)  # Default to 20 if not available
            
            analysis_request = PROMPT_ENHANCER.enhance_with_regime(
                base_prompt=base_request,
                vix_level=vix_level,
                minutes_to_close=minutes_to_close,
                current_pnl=0,  # Could get from execution engine
                positions_open=0  # Could get from execution engine
            )
            
            # Get veteran analysis from agent
            self.logger.info("Performing pattern-based market analysis...")
            # response = self.agent.execute(analysis_request)
            #---------------------------------------------------------------
            task = Task(
                description=analysis_request,
                agent=self.agent,
                expected_output="A valid JSON object with the requested market analysis"
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

            if not isinstance(response_text, str) or not response_text.strip():
                self.logger.error("CrewOutput contained no raw text; falling back to default.")
                return self._default_analysis(market_data)

            analysis = json.loads(response_text)
                        
            # Parse and enhance response
            # analysis = json.loads(response)
            analysis["timestamp"] = datetime.now().isoformat()
            analysis["analyst"] = "market_analyst_veteran"
            
            # Add aggregated pattern statistics
            analysis["pattern_statistics"] = self._get_pattern_statistics(analysis["pattern_recognition"]["current_pattern"])
            
            # Log concise summary
            self.logger.info(
                f"Market Analysis: {analysis['regime']} "
                f"(VIX: {analysis['vix_analysis']['level']:.1f} - {analysis['vix_analysis']['context']}), "
                f"Pattern: {analysis['pattern_recognition']['current_pattern'][:50]}..."
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return self._default_analysis(market_data)
    
    def _load_market_patterns(self) -> None:
        """Load historical market patterns for recognition."""
        # Load aggregated patterns from memory
        self.patterns = TRADING_MEMORY.get_aggregated_patterns()
        self.logger.info(f"Loaded {len(self.patterns)} historical patterns")
    
    def _find_similar_historical_patterns(self, 
                                          current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find historical patterns similar to current market setup.
        
        Args:
            current_data: Current market conditions
            
        Returns:
            Similar historical patterns with outcomes
        """
        vix = current_data.get("vix", 20)
        trend = current_data.get("trend_strength", 0)
        
        # Find similar VIX/trend combinations in history
        similar = TRADING_MEMORY.find_similar_setups(
            vix_range=(vix - 2, vix + 2),
            trend_range=(trend - 20, trend + 20),
            min_similarity=0.7
        )
        
        # Aggregate outcomes by strategy
        pattern_outcomes = {}
        for setup in similar:
            date = setup["date"]
            strategy = setup["strategy"]
            outcome = setup["outcome"]
            
            if strategy not in pattern_outcomes:
                pattern_outcomes[strategy] = {
                    "instances": 0,
                    "wins": 0,
                    "avg_return": 0,
                    "best_setup": None
                }
            
            pattern_outcomes[strategy]["instances"] += 1
            if outcome > 0:
                pattern_outcomes[strategy]["wins"] += 1
            pattern_outcomes[strategy]["avg_return"] += outcome
            
            if not pattern_outcomes[strategy]["best_setup"] or \
               outcome > pattern_outcomes[strategy]["best_setup"]["return"]:
                pattern_outcomes[strategy]["best_setup"] = {
                    "date": date,
                    "return": outcome
                }
        
        # Calculate win rates and average returns
        for strategy in pattern_outcomes:
            data = pattern_outcomes[strategy]
            data["win_rate"] = data["wins"] / data["instances"]
            data["avg_return"] = data["avg_return"] / data["instances"]
        
        return {
            "similar_setups_found": len(similar),
            "pattern_outcomes": pattern_outcomes,
            "most_successful_strategy": max(
                pattern_outcomes.items(),
                key=lambda x: x[1]["avg_return"]
            )[0] if pattern_outcomes else None
        }
    
    def _analyze_recent_behavior(self, days: int) -> Dict[str, Any]:
        """
        Analyze recent market behavior for regime context.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Recent behavior analysis
        """
        # Get recent data from memory
        recent = TRADING_MEMORY.get_market_data(days=days)
        
        if recent.empty:
            return {"status": "no_recent_data"}
        
        # Calculate behavior metrics
        vix_data = recent["vix"].values
        price_data = recent["close"].values
        
        return {
            "vix_trend": "rising" if vix_data[-1] > vix_data[0] else "falling",
            "vix_mean": float(np.mean(vix_data)),
            "vix_std": float(np.std(vix_data)),
            "days_below_15": int(sum(vix_data < 15)),
            "days_above_20": int(sum(vix_data > 20)),
            "largest_daily_move": float(np.max(np.abs(np.diff(price_data)))),
            "trend_consistency": self._calculate_trend_consistency(price_data),
            "compression_days": self._count_compression_days(vix_data)
        }
    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate how consistent the trend has been (0-1)."""
        if len(prices) < 2:
            return 0.5
        
        daily_returns = np.diff(prices) / prices[:-1]
        positive_days = sum(daily_returns > 0)
        consistency = abs(positive_days / len(daily_returns) - 0.5) * 2
        return float(consistency)
    
    def _count_compression_days(self, vix_data: np.ndarray) -> int:
        """Count consecutive days of volatility compression."""
        if len(vix_data) < 2:
            return 0
        
        # Count days where VIX range is < 1 point
        compressed_days = 0
        for i in range(1, len(vix_data)):
            if abs(vix_data[i] - vix_data[i-1]) < 1.0:
                compressed_days += 1
            else:
                compressed_days = 0  # Reset on expansion
        
        return compressed_days
    
    def _get_enhanced_time_factors(self) -> Dict[str, Any]:
        """Get enhanced time factors including pattern-relevant timing."""
        base_time = {
            "minutes_to_close": TIMEZONE_HANDLER.minutes_to_close(),
            "market_phase": TIMEZONE_HANDLER.get_market_phase()[0],
            "spain_window": TIMEZONE_HANDLER.get_spain_trading_window()
        }
        
        # Add pattern-specific timing
        now = datetime.now()
        base_time.update({
            "day_of_week": now.strftime("%A"),
            "days_to_expiry": 0,  # Always 0 for 0DTE
            "vix_expiry_week": (now.isocalendar()[1] % 4) == 3,  # VIX expiry week
            "monthly_expiry_week": 15 <= now.day <= 21 and now.weekday() == 4,  # Monthly opex
            "fed_week": self._is_fed_week(),  # Simplified - would need real calendar
            "earnings_season": self._is_earnings_season()  # Simplified
        })
        
        return base_time
    
    def _is_fed_week(self) -> bool:
        """Check if this is an FOMC week (simplified)."""
        # In production, would check actual Fed calendar
        # For now, assume 3rd week of March, June, Sept, Dec
        now = datetime.now()
        return (now.month in [3, 6, 9, 12]) and (15 <= now.day <= 21)
    
    def _is_earnings_season(self) -> bool:
        """Check if we're in earnings season (simplified)."""
        # Roughly: mid-Jan to mid-Feb, mid-Apr to mid-May, 
        # mid-Jul to mid-Aug, mid-Oct to mid-Nov
        now = datetime.now()
        month_day = (now.month, now.day)
        
        earnings_periods = [
            ((1, 15), (2, 15)),
            ((4, 15), (5, 15)),
            ((7, 15), (8, 15)),
            ((10, 15), (11, 15))
        ]
        
        for start, end in earnings_periods:
            if start <= month_day <= end:
                return True
        return False
    
    def _get_pattern_statistics(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get historical statistics for identified pattern.
        
        Args:
            pattern_name: Name of identified pattern
            
        Returns:
            Historical performance statistics
        """
        # Query memory for this specific pattern
        stats = TRADING_MEMORY.get_pattern_performance(pattern_name)
        
        if not stats:
            return {"status": "no_historical_data"}
        
        # Handle new patterns that don't have historical data yet
        if stats.get("status") == "new_pattern":
            return {
                "status": "new_pattern",
                "occurrences": 0,
                "avg_vix_move_next_day": 0,
                "avg_price_move_next_day": 0,
                "best_strategy_historical": None,
                "worst_strategy_historical": None
            }
        
        return {
            "occurrences": stats.get("count", 0),
            "avg_vix_move_next_day": stats.get("avg_vix_change", 0),
            "avg_price_move_next_day": stats.get("avg_price_change", 0),
            "best_strategy_historical": stats.get("best_strategy"),
            "worst_strategy_historical": stats.get("worst_strategy")
        }
    
    def _default_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide safe default analysis when agent fails.
        
        Args:
            market_data: Current market data
            
        Returns:
            Conservative default analysis
        """
        return {
            "regime": "unknown",
            "regime_confidence": 0,
            "vix_analysis": {
                "level": market_data.get("vix", 20),
                "regime": "uncertain",
                "context": "Unable to assess - using conservative parameters"
            },
            "trading_implications": {
                "favorable_setups": [],
                "setups_to_avoid": ["all"],
                "key_risks": ["Analysis system offline"]
            },
            "context_for_strategy_architect": {
                "critical_observations": ["System in safe mode"],
                "regime_transition_risk": "unknown",
                "recommended_approach": "skip"
            },
            "confidence_score": 0,
            "timestamp": datetime.now().isoformat(),
            "analyst": "market_analyst_default"
        }