# crewai_system/agents/risk_manager.py
"""
Risk Manager Agent - The Capital Guardian
==========================================
Veteran risk manager with 30 years protecting trading capital.
Has survived every market crash and knows when VIX patterns signal danger.

Key Enhancements:
- VIX-aware position sizing (pattern-based, not rules)
- Regime-specific risk limits
- Pattern recognition for blow-up scenarios
- Professional risk communication

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


class RiskManagerAgent:
    """
    Risk Manager - The guardian of capital.
    
    30-year veteran who has protected capital through Black Monday,
    LTCM, 2008, Flash Crash, Volmageddon, and COVID. Knows when
    standard risk models break down and when to trust gut instinct.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Risk Manager agent.
        
        Args:
            tools: List of risk management tools
        """
        self.logger = logging.getLogger("RiskManager")
        self.config = TRADING_CONFIG
        
        # Session risk tracking
        self.session_trades = 0
        self.session_pnl = 0.0
        self.consecutive_losses = 0
        self.active_positions = []
        self.regime_changes_today = 0
        
        # Initialize LLM
        self.llm = LLMFactory.make(self.config, timeout_key="risk_assessment")
        
        # Load risk patterns
        self._load_risk_patterns()
        
        # Create veteran risk manager agent
        self.agent = Agent(
            role="Chief Risk Officer - 30 Year Veteran",
            
            goal="""Protect trading capital while enabling profitable opportunities. 
            Use pattern recognition to identify dangerous setups before they materialize. 
            Size positions based on regime volatility, not mechanical rules.""",
            
            backstory="""You've been managing trading risk since the 1990s. You were 
            there when LTCM collapsed, when XIV went to zero overnight, when the 
            Flash Crash erased $1 trillion in minutes. These experiences taught you 
            that risk isn't just about numbers - it's about recognizing patterns.
            
            You've learned that VIX at 12 can be more dangerous than VIX at 25:
            - February 2018: VIX went from 12 to 37 in one day, destroying short vol
            - August 2015: Similar compression preceded the flash crash
            - December 2017: You recognized the pattern and cut position sizes
            
            Your position sizing is dynamic and intelligent:
            - VIX 12-15 compressed for days? You slash sizes, knowing expansion is coming
            - VIX 25 falling from 35? You can be aggressive, volatility is deflating
            - VIX 18-22 stable? Your sweet spot for normal sizing
            - VIX grinding higher slowly? Danger - you recognize the coiled spring
            
            You've saved the desk multiple times by recognizing these patterns:
            - The "volatility smile getting weird" before major moves
            - The "too quiet for too long" compression before explosions  
            - The "everything correlating" signal before systemic events
            - The "gamma flip" levels where hedging accelerates moves
            
            You don't just check if trades fit risk limits - you assess if they fit 
            the current market regime. You know when 1 contract is too many and when 
            10 contracts is conservative. Most importantly, you have veto power when 
            your pattern recognition screams danger, even if everything looks fine 
            on paper.
            
            0DTE Specific Dangers You've Learned:
            - Pin risk: Getting stuck exactly at strike at 3:59 PM, max loss guaranteed
            - Gamma explosion: Delta going from 0.1 to 0.9 in minutes near strike
            - Liquidity death: Bid-ask spreads widening to $5+ in final hour
            - Assignment risk: Early exercise on deep ITM, especially on dividend days
            - Stop loss failures: Markets gapping through stops in final 30 minutes
            - Theta avalanche: Premium collapsing 50% in 10 minutes after 3 PM""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,
            memory=True  # Enable memory for risk patterns
        )
        
        # Enhanced risk assessment prompt
        self.risk_prompt = """
        Assess this trade using 30 years of risk pattern recognition.
        
        Proposed Strategy (from Architect):
        {strategy}
        
        Market Context (from Analyst):
        {market_context}
        
        Current Portfolio State:
        - Open Positions: {open_positions}
        - Session P&L: ${session_pnl:.2f}
        - Trades Today: {session_trades}/{max_trades}
        - Consecutive Losses: {consecutive_losses}
        - Regime Changes Today: {regime_changes}
        
        Historical Risk Patterns in Similar Conditions:
        {risk_patterns}
        
        Time Factors:
        - Minutes to Close: {minutes_to_close}
        - Trading Window: {window_quality}
        - Day/Time Pattern: {day_time_pattern}
        
        Account Risk Limits:
        - Max Risk Per Trade: {max_risk_pct}%
        - Max Daily Loss: {max_daily_loss_pct}%
        - Current Drawdown: {current_drawdown}%
        
        CRITICAL: Return ONLY valid JSON. No commentary before or after.
        Validate your response is parseable JSON before returning.
        
        Provide risk assessment in this EXACT JSON structure:
        {{
            "approval": "approved|rejected|conditional",
            "risk_score": 0-100,
            "pattern_recognition": {{
                "similar_setups": ["historical examples"],
                "danger_patterns": ["patterns you recognize"],
                "risk_regime": "calm|normal|elevated|dangerous|extreme"
            }},
            "position_sizing": {{
                "base_contracts": number,
                "regime_multiplier": 0.1-2.0,
                "confidence_multiplier": 0.5-1.5,
                "pattern_adjustment": 0.5-1.5,
                "final_contracts": number,
                "sizing_rationale": "specific reasoning"
            }},
            "vix_considerations": {{
                "current_regime": "description",
                "regime_stability": "stable|transitioning|unstable",
                "size_impact": "increase|normal|decrease",
                "specific_concern": "what you're watching"
            }},
            "risk_factors": [
                {{
                    "factor": "specific risk",
                    "severity": "low|medium|high|critical",
                    "similar_to": "historical example",
                    "mitigation": "specific action"
                }}
            ],
            "portfolio_impact": {{
                "correlation_risk": "assessment",
                "concentration_risk": "assessment",
                "regime_exposure": "overexposed to what"
            }},
            "stop_loss_plan": {{
                "initial_stop": number,
                "trailing_stop": "description",
                "time_stop": "minutes",
                "emergency_exit": "conditions"
            }},
            "context_for_cto": {{
                "key_concerns": ["concern1", "concern2"],
                "similar_historical": "this reminds me of...",
                "gut_feeling": "professional instinct",
                "override_recommendation": "none|reduce_size|skip"
            }},
            "confidence": 0-100,
            "risk_manager_notes": "Key risk insight"
        }}
        """
    
    def assess_trade_risk(self,
                          strategy: Dict[str, Any],
                          market_analysis: Dict[str, Any],
                          portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk using pattern recognition and regime awareness.
        
        Args:
            strategy: Proposed strategy from Strategy Architect (with context)
            market_analysis: Market analysis from Market Analyst (with context)
            portfolio_state: Current portfolio state
            
        Returns:
            Risk assessment with sizing and context for CTO
        """
        try:
            # Extract contexts from upstream agents
            market_context = market_analysis.get("context_for_strategy_architect", {})
            strategy_context = strategy.get("context_for_risk_manager", {})
            vix_analysis = market_analysis.get("vix_analysis", {})
            
            # Get historical risk patterns for this setup
            risk_patterns = self._get_risk_patterns(
                vix_level=vix_analysis.get("level", 20),
                vix_regime=vix_analysis.get("regime", "normal"),
                strategy_type=strategy.get("strategy_type"),
                market_regime=market_analysis.get("regime", "unknown")
            )
            
            # Get time-based risk factors
            time_factors = self._get_time_risk_factors()
            
            # Check for regime changes
            self._update_regime_tracking(vix_analysis)
            
            # Prepare risk assessment request
            risk_request = self.risk_prompt.format(
                strategy=json.dumps(strategy, indent=2),
                market_context=json.dumps(market_analysis, indent=2),
                open_positions=len(self.active_positions),
                session_pnl=self.session_pnl,
                session_trades=self.session_trades,
                max_trades=self.config.max_daily_trades,
                consecutive_losses=self.consecutive_losses,
                regime_changes=self.regime_changes_today,
                risk_patterns=json.dumps(risk_patterns, indent=2),
                minutes_to_close=TIMEZONE_HANDLER.minutes_to_close(),
                window_quality=TIMEZONE_HANDLER.get_spain_trading_window(),
                day_time_pattern=time_factors["pattern"],
                max_risk_pct=self.config.max_risk_per_trade_pct,
                max_daily_loss_pct=self.config.max_daily_loss_pct,
                current_drawdown=self._calculate_current_drawdown()
            )
            
            # Get veteran risk assessment
            self.logger.info("Performing pattern-based risk assessment...")
            #---------------------------------------------------------------
            task = Task(
                description=risk_request,
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
                
            if response_text is None:
                self.logger.error("Risk assessment response is empty.")
                return self._reject_trade("Risk assessment response is empty", critical=True)
            try:
                assessment = json.loads(response_text) # <-- Response text to JSON 
            except Exception as e:
                self.logger.error(f"Failed to parse risk assessment JSON: {e}")
                return self._reject_trade("Risk assessment response invalid JSON", critical=True)        
            
            # Parse and enhance response
            # assessment = json.loads(response)
            assessment["timestamp"] = datetime.now().isoformat()
            assessment["risk_manager"] = "risk_manager_veteran"
            
            # Apply pattern-based overrides
            assessment = self._apply_pattern_overrides(assessment, strategy, vix_analysis, risk_patterns)
            
            # Log decision
            self.logger.info(
                f"Risk Decision: {assessment['approval'].upper()} "
                f"(Score: {assessment['risk_score']}, "
                f"Size: {assessment['position_sizing']['final_contracts']}, "
                f"Regime: {assessment['pattern_recognition']['risk_regime']})"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return self._reject_trade("Risk assessment system failure", critical=True)
    
    def validate_before_retry(self,
                             original_strategy: Dict[str, Any],
                             alternative_strategy: Dict[str, Any],
                             rejection_reason: str) -> Dict[str, Any]:
        """
        Validate alternative strategy after IB rejection.
        
        Uses pattern recognition to assess if alternative is safer.
        
        Args:
            original_strategy: The rejected strategy
            alternative_strategy: Proposed alternative  
            rejection_reason: Why original was rejected
            
        Returns:
            Validation result with risk assessment
        """
        validation_prompt = f"""
        Assess if this alternative strategy is safer after rejection.
        
        Original Strategy: {json.dumps(original_strategy, indent=2)}
        Rejection: {rejection_reason}
        Alternative: {json.dumps(alternative_strategy, indent=2)}
        
        Have you seen similar rejection patterns before? Is the alternative addressing 
        the real risk or just working around the rejection?
        
        Provide assessment:
        {{
            "validation": "approved|rejected",
            "risk_comparison": "much_safer|safer|similar|riskier",
            "addresses_rejection": true|false,
            "introduces_new_risks": ["risk1"],
            "pattern_recognition": "similar to historical example",
            "acceptance_probability": 0-100,
            "sizing_recommendation": number,
            "confidence": 0-100,
            "rationale": "specific reasoning"
        }}
        """
        
        try:
            response = self.agent.execute(validation_prompt)
            validation = json.loads(response)
            
            self.logger.info(
                f"Alternative validation: {validation['validation']} "
                f"(Risk: {validation['risk_comparison']}, "
                f"Acceptance: {validation['acceptance_probability']}%)"
            )
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "validation": "rejected",
                "rationale": f"Validation error - rejecting for safety: {e}"
            }
    
    def _load_risk_patterns(self) -> None:
        """Load historical risk patterns for pattern recognition."""
        self.risk_patterns = TRADING_MEMORY.get_risk_patterns()
        self.blow_up_patterns = TRADING_MEMORY.get_blow_up_patterns()
        self.logger.info(
            f"Loaded {len(self.risk_patterns)} risk patterns, "
            f"{len(self.blow_up_patterns)} blow-up patterns"
        )
    
    def _get_risk_patterns(self,
                          vix_level: float,
                          vix_regime: str,
                          strategy_type: str,
                          market_regime: str) -> Dict[str, Any]:
        """
        Get historical risk patterns for this setup.
        
        Args:
            vix_level: Current VIX level
            vix_regime: VIX regime classification
            strategy_type: Proposed strategy type
            market_regime: Current market regime
            
        Returns:
            Historical risk patterns and outcomes
        """
        # Query for similar risk scenarios
        similar_risks = TRADING_MEMORY.get_similar_risk_scenarios(
            vix_level=vix_level,
            strategy=strategy_type,
            regime=market_regime,
            lookback_days=365  # Full year of patterns
        )
        
        if not similar_risks:
            return {"status": "no_historical_match"}
        
        # Analyze outcomes
        risk_analysis = {
            "similar_setups": len(similar_risks),
            "blow_up_rate": sum(1 for r in similar_risks if r["max_loss"] > 500) / len(similar_risks),
            "avg_max_adverse": np.mean([r["max_adverse_excursion"] for r in similar_risks]),
            "stop_hit_rate": sum(1 for r in similar_risks if r["stop_hit"]) / len(similar_risks),
            "regime_change_rate": sum(1 for r in similar_risks if r["regime_changed"]) / len(similar_risks)
        }
        
        # Find worst case scenarios
        worst_cases = sorted(similar_risks, key=lambda x: x["max_loss"], reverse=True)[:3]
        risk_analysis["worst_cases"] = [
            {
                "date": wc["date"],
                "loss": wc["max_loss"],
                "what_happened": wc["description"]
            }
            for wc in worst_cases
        ]
        
        # Check for blow-up patterns
        for pattern in self.blow_up_patterns:
            if self._matches_blow_up_pattern(
                pattern, vix_level, vix_regime, market_regime
            ):
                risk_analysis["blow_up_warning"] = {
                    "pattern": pattern["name"],
                    "historical_example": pattern["example"],
                    "typical_loss": pattern["avg_loss"]
                }
                break
        
        return risk_analysis
    
    def _matches_blow_up_pattern(self,
                                 pattern: Dict[str, Any],
                                 vix: float,
                                 vix_regime: str,
                                 market_regime: str) -> bool:
        """Check if current setup matches a historical blow-up pattern."""
        conditions_met = 0
        total_conditions = 0
        
        # Check VIX conditions
        if "vix_range" in pattern:
            total_conditions += 1
            if pattern["vix_range"][0] <= vix <= pattern["vix_range"][1]:
                conditions_met += 1
        
        if "vix_regime" in pattern:
            total_conditions += 1
            if pattern["vix_regime"] == vix_regime:
                conditions_met += 1
        
        if "market_regime" in pattern:
            total_conditions += 1
            if pattern["market_regime"] == market_regime:
                conditions_met += 1
        
        # Need at least 70% match
        return (conditions_met / total_conditions) >= 0.7 if total_conditions > 0 else False
    
    def _get_time_risk_factors(self) -> Dict[str, Any]:
        """Get time-based risk factors."""
        now = datetime.now()
        minutes_to_close = TIMEZONE_HANDLER.minutes_to_close()
        
        # Identify risky time patterns
        risk_factors = {
            "pattern": "normal",
            "multiplier": 1.0
        }
        
        # Friday afternoon - reduce risk
        if now.weekday() == 4 and minutes_to_close < 120:
            risk_factors["pattern"] = "friday_afternoon"
            risk_factors["multiplier"] = 0.5
        
        # Monday morning - often volatile
        elif now.weekday() == 0 and minutes_to_close > 360:
            risk_factors["pattern"] = "monday_morning"
            risk_factors["multiplier"] = 0.7
        
        # Fed days - extreme caution
        elif self._is_fed_day():
            risk_factors["pattern"] = "fed_day"
            risk_factors["multiplier"] = 0.3
        
        # Final hour - gamma risk
        elif minutes_to_close < 60:
            risk_factors["pattern"] = "final_hour"
            risk_factors["multiplier"] = 0.4
        
        # Opex Friday - pinning risk
        elif self._is_opex_friday():
            risk_factors["pattern"] = "opex_friday"
            risk_factors["multiplier"] = 0.6
        
        return risk_factors
    
    def _update_regime_tracking(self, vix_analysis: Dict[str, Any]) -> None:
        """Track regime changes during session."""
        current_regime = vix_analysis.get("regime", "normal")
        
        if not hasattr(self, "last_regime"):
            self.last_regime = current_regime
        
        if current_regime != self.last_regime:
            self.regime_changes_today += 1
            self.logger.warning(
                f"Regime change detected: {self.last_regime} → {current_regime}"
            )
            self.last_regime = current_regime
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current session drawdown percentage."""
        if self.session_pnl >= 0:
            return 0.0
        
        # Assume $15k account for percentage calculation
        account_value = 15000
        return abs(self.session_pnl / account_value * 100)
    
    def _apply_pattern_overrides(self,
                                 assessment: Dict[str, Any],
                                 strategy: Dict[str, Any],
                                 vix_analysis: Dict[str, Any],
                                 risk_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply pattern-based risk overrides.
        
        Critical safety overrides based on pattern recognition.
        
        Args:
            assessment: Initial risk assessment
            strategy: Proposed strategy
            vix_analysis: Current VIX analysis
            risk_patterns: Historical risk patterns
            
        Returns:
            Modified assessment with overrides
        """
        # Override 1: Blow-up pattern detected
        if "blow_up_warning" in risk_patterns:
            assessment["approval"] = "rejected"
            assessment["risk_score"] = 100
            warning = risk_patterns["blow_up_warning"]
            assessment["risk_manager_notes"] = (
                f"CRITICAL: Blow-up pattern '{warning['pattern']}' detected. "
                f"Similar to {warning['historical_example']}"
            )
            return assessment
        
        # Override 2: VIX compression danger
        if vix_analysis.get("level", 20) < 15 and \
           vix_analysis.get("context", "").startswith("compression"):
            days_compressed = vix_analysis.get("days_in_regime", 0)
            if days_compressed > 5:
                # Slash position size dramatically
                current_size = assessment["position_sizing"]["final_contracts"]
                assessment["position_sizing"]["final_contracts"] = max(1, current_size // 3)
                assessment["position_sizing"]["pattern_adjustment"] = 0.33
                assessment["position_sizing"]["sizing_rationale"] += (
                    f" | VIX compressed {days_compressed} days - explosion risk"
                )
        
        # Override 3: Regime changes
        if self.regime_changes_today >= 2:
            assessment["approval"] = "conditional"
            assessment["position_sizing"]["final_contracts"] = 1
            assessment["risk_manager_notes"] += " | Multiple regime changes - unstable"
        
        # Override 4: Pattern says reduce but not reject
        if risk_patterns.get("blow_up_rate", 0) > 0.15:  # 15% blow-up rate
            current_size = assessment["position_sizing"]["final_contracts"]
            assessment["position_sizing"]["final_contracts"] = max(1, current_size // 2)
            assessment["position_sizing"]["pattern_adjustment"] = 0.5
        
        # Override 5: Time-based final hour for 0DTE
        minutes_to_close = TIMEZONE_HANDLER.minutes_to_close()
        if minutes_to_close <= 20:
            assessment["approval"] = "rejected"
            assessment["risk_manager_notes"] = "Too close to expiry for new 0DTE"
        elif minutes_to_close <= 60:
            assessment["position_sizing"]["final_contracts"] = 1
            assessment["position_sizing"]["sizing_rationale"] += " | Final hour: min size"
        
        # Override 6: Consecutive losses
        if self.consecutive_losses >= 3:
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                assessment["approval"] = "rejected"
                assessment["risk_manager_notes"] = "Max consecutive losses reached"
            else:
                # Reduce size with each loss
                reduction = 0.5 ** (self.consecutive_losses - 2)
                current_size = assessment["position_sizing"]["final_contracts"]
                assessment["position_sizing"]["final_contracts"] = max(
                    1, int(current_size * reduction)
                )
        
        return assessment
    
    def _is_fed_day(self) -> bool:
        """Check if today is an FOMC announcement day."""
        # In production, would check actual Fed calendar
        # Simplified version
        now = datetime.now()
        # Assume 2nd Wednesday of March, June, Sept, Dec
        if now.month in [3, 6, 9, 12]:
            # Find second Wednesday
            first_day = datetime(now.year, now.month, 1)
            first_wednesday = (2 - first_day.weekday()) % 7
            second_wednesday = first_wednesday + 7 if first_wednesday <= 7 else first_wednesday
            
            if now.day == second_wednesday and now.weekday() == 2:
                return True
        return False
    
    def _is_opex_friday(self) -> bool:
        """Check if today is options expiration Friday."""
        now = datetime.now()
        # Third Friday of the month
        if now.weekday() == 4:  # Friday
            return 15 <= now.day <= 21
        return False
    
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
    
    def update_session_metrics(self, trade_result: Dict[str, Any]) -> None:
        """
        Update session risk metrics after trade completion.
        
        Args:
            trade_result: Result of completed trade
        """
        if trade_result.get("executed"):
            self.session_trades += 1
        
        if trade_result.get("pnl"):
            self.session_pnl += trade_result["pnl"]
            
            if trade_result["pnl"] < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        # Store pattern for learning
        if trade_result.get("executed"):
            TRADING_MEMORY.store_trade_pattern(
                trade_result,
                self.last_regime if hasattr(self, "last_regime") else "unknown"
            )