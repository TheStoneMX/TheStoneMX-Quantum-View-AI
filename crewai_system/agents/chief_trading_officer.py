# crewai_system/agents/chief_trading_officer.py
"""
Chief Trading Officer Agent - The Final Authority
==================================================
Master orchestrator with 30 years leading trading desks.
Makes final decisions considering all perspectives and patterns.

Key Enhancements:
- Holistic decision making with VIX awareness
- Pattern-based judgment calls
- Authority to override but respects expertise
- Aggressive learning from outcomes

Author: Quantum View AI Trading System
Version: 2.0
Date: December 2024
"""

from crewai import Agent, Task, Crew 
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
from enum import Enum
import numpy as np

from ..config.config_loader import TRADING_CONFIG
from ..config.timezone_handler import TIMEZONE_HANDLER
from ..memory.persistence import TRADING_MEMORY
from ..config.llm_factory import LLMFactory
from ..tools.market_tools import get_real_market_data, get_real_option_chain


class TradingDecision(Enum):
    """Possible trading decisions from the CTO."""
    EXECUTE = "execute"
    SKIP = "skip"
    WAIT = "wait"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"


class ChiefTradingOfficer:
    """
    Chief Trading Officer - The decision maker and orchestrator.
    
    30-year veteran who has led trading desks through every market
    condition. Makes final calls balancing all perspectives while
    trusting the expertise of specialists.
    """
    
    def __init__(self, 
                 market_analyst: Any,
                 strategy_architect: Any,
                 risk_manager: Any,
                 execution_specialist: Any,
                 tools: List[Any]):
        """
        Initialize the Chief Trading Officer.
        
        Args:
            market_analyst: Market Analyst agent instance
            strategy_architect: Strategy Architect agent instance
            risk_manager: Risk Manager agent instance
            execution_specialist: Execution Specialist agent instance
            tools: List of coordination tools
        """
        self.logger = logging.getLogger("ChiefTradingOfficer")
        self.config = TRADING_CONFIG
        
        # Store references to specialist agents
        self.market_analyst = market_analyst
        self.strategy_architect = strategy_architect
        self.risk_manager = risk_manager
        self.execution_specialist = execution_specialist
        
        # Session management
        self.session_active = False
        self.emergency_stop = False
        self.trades_today = 0
        self.session_patterns = []  # Track patterns seen today
        self.decision_history = []  # For learning
        
        # Initialize LLM
        self.llm = LLMFactory.make(self.config, timeout_key="chief_trading_officer")
        
        # Load CTO patterns
        self._load_decision_patterns()
        
        # Create veteran CTO agent
        self.agent = Agent(
            role="Chief Trading Officer - 30 Year Veteran",
            
            goal="""Orchestrate trading operations with the wisdom of three decades. 
            Make final decisions that balance opportunity with survival. Trust your 
            specialists but apply holistic judgment they might miss.""",
            
            backstory="""You've led trading desks since the 1990s. You've built teams, 
            managed billions, and navigated every crisis. Your greatest skill isn't 
            trading - it's orchestrating talented specialists and making tough calls 
            under pressure.
            
            You've learned when to trust your team and when to override:
            - October 2008: Your risk manager said "skip" but you saw opportunity in chaos
            - February 2018: Your strategist loved short vol, you overruled and saved millions
            - March 2020: You went to cash before the circuit breakers, pattern recognition
            - GameStop 2021: You recognized the new dynamics and adapted strategies
            
            Your decision framework isn't mechanical - it's pattern-based:
            - When analyst, architect, and risk all align: Execute with conviction
            - When risk says danger but you see opportunity: Trust risk (usually)
            - When VIX patterns conflict with team consensus: Dig deeper
            - When something feels off despite good numbers: Trust your gut
            
            You understand VIX regimes deeply:
            - VIX 12 after compression: Maximum caution, regardless of team optimism
            - VIX 35 stabilizing: Opportunity when others are fearful
            - VIX 18-22 grinding: Your profit zone, let the team work
            - VIX transitions: The most dangerous and profitable moments
            
            You respect your specialists' expertise:
            - Market Analyst: Trust their pattern recognition
            - Strategy Architect: Respect their structure expertise
            - Risk Manager: Override only when truly necessary
            
            Your philosophy: "Great traders make money. Great CTOs keep it."
            
            You coordinate everything with professional efficiency - no time for long 
            discussions when 0DTE options are decaying by the minute.
            
            In 0DTE, analysis paralysis kills profits. Your decision framework:
            - Green light (5 seconds): All three specialists agree = EXECUTE
            - Yellow light (10 seconds): 2/3 agree = EXECUTE with reduced size
            - Red light (instant): Risk Manager says no = VETO, no debate
            - Market close < 30 min: NO NEW TRADES, only exits
            Time is literally money with 0DTE theta burn - decide fast or skip.""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=True,
            max_iter=self.config.max_iterations,
            memory=True  # Enable memory for decision patterns
        )
        
        # Enhanced decision prompt
        self.decision_prompt = """
        Make final trading decision using 30 years of pattern recognition.
        
        Market Analysis (from veteran analyst):
        {market_analysis}
        
        Strategy Design (from veteran architect):
        {strategy}
        
        Risk Assessment (from veteran risk manager):
        {risk_assessment}
        
        Current Session State:
        - Trades Today: {trades_today}
        - Active Positions: {active_positions}
        - Session P&L: ${session_pnl:.2f}
        - Time to Close: {time_to_close} minutes
        - Patterns Seen Today: {session_patterns}
        
        Historical Decision Patterns:
        {decision_patterns}
        
        Team Alignment Analysis:
        {team_alignment}
        
        CRITICAL: Return ONLY valid JSON. No commentary before or after.
        Validate your response is parseable JSON before returning.
        
        Make your decision in this EXACT JSON structure:
        {{
            "decision": "execute|skip|wait|close_positions",
            "rationale": "specific reasoning referencing patterns",
            "confidence": 0-100,
            "pattern_recognition": {{
                "current_setup": "description",
                "similar_historical": ["example1", "example2"],
                "key_difference": "what's different this time"
            }},
            "team_assessment": {{
                "alignment_level": "full|partial|conflicted",
                "trust_score": 0-100,
                "override_reason": "none|risk_too_high|opportunity_too_good|pattern_mismatch"
            }},
            "vix_consideration": {{
                "regime_assessment": "your read on VIX",
                "agrees_with_team": true|false,
                "size_adjustment": "none|increase|decrease",
                "specific_concern": "what you're watching"
            }},
            "execution_parameters": {{
                "strategy_type": "from architect",
                "final_contracts": number,
                "urgency": "immediate|normal|patient",
                "special_instructions": ["instruction1"]
            }},
            "risk_overrides": {{
                "accept_risk_rejection": true|false,
                "modify_stops": false,
                "reason": "specific reason if overriding"
            }},
            "next_review": "seconds until next decision",
            "cto_notes": "Key insight for session log"
        }}
        """
    
    def orchestrate_trading_decision(self) -> Tuple[TradingDecision, Dict[str, Any]]:
        """
        Orchestrate complete trading decision with veteran judgment.
        
        Coordinates all agents and makes final call based on pattern
        recognition and holistic assessment.
        
        Returns:
            Tuple of (decision_type, execution_details)
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("CTO: ORCHESTRATING TRADING DECISION")
            self.logger.info(TIMEZONE_HANDLER.format_times_for_logging())
            self.logger.info("=" * 60)
            
            # Pre-flight checks
            if not self._pre_flight_checks():
                return (TradingDecision.SKIP, {"reason": "Pre-flight checks failed"})
            
            # Step 1: Market Analysis with context
            self.logger.info("CTO: Requesting veteran market analysis...")
            market_analysis = self._get_market_analysis()
            
            self.logger.info(f"DEBUG - Market Analysis: confidence={market_analysis.get('confidence_score', 0)}, regime={market_analysis.get('regime', 'unknown')}")
            
            if not market_analysis or market_analysis.get("confidence_score", 0) < 20:
                self.logger.warning("CTO: Market analysis confidence too low")
                return (TradingDecision.WAIT, {"reason": "Insufficient market clarity"})
            
            # Step 2: Strategy Design if favorable
            if self._should_design_strategy(market_analysis):
                self.logger.info("CTO: Requesting strategy design...")
                strategy = self._get_strategy_design(market_analysis)
                
                if not strategy or strategy.get("strategy_type") == "skip":
                    return (TradingDecision.SKIP, {
                        "reason": strategy.get("architect_notes", "No viable strategy")
                    })
            else:
                return (TradingDecision.WAIT, {
                    "reason": "Market conditions unfavorable for strategy design"
                })
            
            # Step 3: Risk Validation with context
            self.logger.info("CTO: Requesting risk validation...")
            risk_assessment = self._get_risk_assessment(strategy, market_analysis)
            
            # Step 4: Make veteran CTO decision
            final_decision = self._make_final_decision(
                market_analysis, strategy, risk_assessment
            )
            
            # Step 5: Execute or skip based on decision
            if final_decision["decision"] == "execute":
                # Check if we're overriding risk rejection
                if risk_assessment.get("approval") == "rejected" and \
                   not final_decision.get("risk_overrides", {}).get("accept_risk_rejection"):
                    self.logger.warning("CTO: Risk rejection stands - not overriding")
                    return (TradingDecision.SKIP, {
                        "reason": "Risk rejection accepted by CTO"
                    })
                
                return self._coordinate_execution(final_decision, strategy, risk_assessment)
            else:
                return (TradingDecision[final_decision["decision"].upper()], final_decision)
                
        except Exception as e:
            self.logger.error(f"CTO: Orchestration error - {e}")
            return (TradingDecision.SKIP, {"reason": f"System error: {e}"})
    
    def handle_ib_rejection(self,
                            original_strategy: Dict[str, Any],
                            rejection_reason: str,
                            attempt_number: int) -> Tuple[TradingDecision, Dict[str, Any]]:
        """
        Handle IB rejection with pattern-based recovery.
        
        Args:
            original_strategy: The rejected strategy
            rejection_reason: IB's rejection reason
            attempt_number: Which attempt this is
            
        Returns:
            Tuple of (decision_type, execution_details)
        """
        self.logger.warning(f"CTO: Handling IB rejection (attempt {attempt_number})")
        
        # Check retry limits
        if attempt_number >= 3:
            self.logger.error("CTO: Max retries reached - moving on")
            self._record_rejection_pattern(original_strategy, rejection_reason)
            return (TradingDecision.SKIP, {"reason": "Max retries exceeded"})
        
        # Check time constraints
        if TIMEZONE_HANDLER.minutes_to_close() < 15:
            self.logger.warning("CTO: Insufficient time for retry")
            return (TradingDecision.SKIP, {"reason": "Too close to close"})
        
        try:
            # Get alternative from Strategy Architect
            self.logger.info("CTO: Requesting alternative strategy...")
            alternative = self.strategy_architect.propose_alternative(
                original_strategy,
                rejection_reason,
                self._get_current_market_snapshot()
            )
            
            if not alternative or alternative.get("strategy_type") == "skip":
                return (TradingDecision.SKIP, {"reason": "No viable alternative"})
            
            # Validate with Risk Manager
            self.logger.info("CTO: Risk validating alternative...")
            validation = self.risk_manager.validate_before_retry(
                original_strategy,
                alternative,
                rejection_reason
            )
            
            # CTO decision on retry
            retry_decision = self._decide_on_retry(
                original_strategy,
                alternative,
                validation,
                rejection_reason,
                attempt_number
            )
            
            if retry_decision["approve_retry"]:
                self.logger.info(
                    f"CTO: Approving retry - {retry_decision['modification']}"
                )
                return (TradingDecision.EXECUTE, {
                    "strategy": alternative,
                    "is_retry": True,
                    "attempt": attempt_number + 1,
                    "modifications": retry_decision["modifications"]
                })
            else:
                return (TradingDecision.SKIP, {
                    "reason": retry_decision["reason"]
                })
                
        except Exception as e:
            self.logger.error(f"CTO: Rejection handling failed - {e}")
            return (TradingDecision.SKIP, {"reason": f"Recovery failed: {e}"})
    
    def _load_decision_patterns(self) -> None:
        """Load historical decision patterns for CTO judgment."""
        self.decision_patterns = TRADING_MEMORY.get_cto_decision_patterns()
        self.override_patterns = TRADING_MEMORY.get_override_patterns()
        self.logger.info(
            f"Loaded {len(self.decision_patterns)} decision patterns, "
            f"{len(self.override_patterns)} override patterns"
        )
    
    def _make_final_decision(self,
                            market_analysis: Dict[str, Any],
                            strategy: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final trading decision with veteran CTO judgment.
        
        Args:
            market_analysis: Full market analysis with context
            strategy: Strategy design with context
            risk_assessment: Risk assessment with context
            
        Returns:
            Final decision with execution details
        """
        # Analyze team alignment
        team_alignment = self._analyze_team_alignment(
            market_analysis, strategy, risk_assessment
        )
        
        # Get relevant decision patterns
        decision_patterns = self._get_relevant_decision_patterns(
            market_analysis, strategy, risk_assessment
        )
        
        # Prepare decision context
        decision_context = self.decision_prompt.format(
            market_analysis=json.dumps(market_analysis, indent=2),
            strategy=json.dumps(strategy, indent=2),
            risk_assessment=json.dumps(risk_assessment, indent=2),
            trades_today=self.trades_today,
            active_positions=len(self.risk_manager.active_positions),
            session_pnl=self.risk_manager.session_pnl,
            time_to_close=TIMEZONE_HANDLER.minutes_to_close(),
            session_patterns=json.dumps(self.session_patterns[-5:]),  # Last 5 patterns
            decision_patterns=json.dumps(decision_patterns, indent=2),
            team_alignment=json.dumps(team_alignment, indent=2)
        )
        
        try:
            # Get veteran analysis from agent
            self.logger.info("CTO: Making final decision...")          
           #---------------------------------------------------------------
            task = Task(
                description=decision_context,
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
            
            # response is a CrewOutput    
            response_text = getattr(response, "raw", None)
            if not response_text and hasattr(response, "tasks_output") and response.tasks_output:
                response_text = getattr(response.tasks_output[0], "raw", None)
                
            if response_text is None:
                self.logger.error("Risk assessment response is empty.")
                return self._reject_trade("Risk assessment response is empty", critical=True)
            try:
                decision = json.loads(response_text) # <-- Response text to JSON 
            except Exception as e:
                self.logger.error(f"Failed to parse risk assessment JSON: {e}")
                return self._reject_trade("Risk assessment response invalid JSON", critical=True)          
            
            # Apply CTO judgment overrides
            decision = self._apply_cto_judgment(decision, market_analysis, strategy, risk_assessment)
            
            # Record decision pattern
            self._record_decision_pattern(decision, market_analysis)
            
            self.logger.info(
                f"CTO DECISION: {decision['decision'].upper()} "
                f"(Confidence: {decision['confidence']}%, "
                f"Team Trust: {decision['team_assessment']['trust_score']}%)"
            )
            
            if decision.get("risk_overrides", {}).get("accept_risk_rejection"):
                self.logger.warning("CTO: Overriding risk rejection")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"CTO decision failed: {e}")
            return {
                "decision": "skip",
                "rationale": f"Decision system error: {e}",
                "confidence": 0
            }
    
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

    def _analyze_team_alignment(self,
                               market_analysis: Dict[str, Any],
                               strategy: Dict[str, Any],
                               risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how well the team's assessments align.
        
        Args:
            market_analysis: Market analyst output
            strategy: Strategy architect output
            risk_assessment: Risk manager output
            
        Returns:
            Team alignment analysis
        """
        # Extract confidence scores
        market_confidence = market_analysis.get("confidence_score", 0)
        strategy_confidence = strategy.get("confidence", 0)
        risk_confidence = risk_assessment.get("confidence", 0)
        
        # Calculate alignment
        avg_confidence = np.mean([market_confidence, strategy_confidence, risk_confidence])
        confidence_std = np.std([market_confidence, strategy_confidence, risk_confidence])
        
        # Check for conflicts
        conflicts = []
        
        # Market wants trade but risk says no
        if market_analysis.get("trading_implications", {}).get("favorable_setups") and \
           risk_assessment.get("approval") == "rejected":
            conflicts.append("market_bullish_risk_bearish")
        
        # Strategy confident but risk worried
        if strategy_confidence > 70 and risk_assessment.get("risk_score", 0) > 70:
            conflicts.append("strategy_risk_disagreement")
        
        # VIX regime conflicts
        market_vix = market_analysis.get("vix_analysis", {}).get("regime")
        risk_vix = risk_assessment.get("vix_considerations", {}).get("current_regime")
        if market_vix and risk_vix and market_vix != risk_vix:
            conflicts.append("vix_interpretation_mismatch")
        
        return {
            "average_confidence": float(avg_confidence),
            "confidence_deviation": float(confidence_std),
            "alignment_score": float(100 - confidence_std),  # Higher = more aligned
            "conflicts": conflicts,
            "unanimous": len(conflicts) == 0 and confidence_std < 10,
            "split": confidence_std > 30
        }
    
    def _get_relevant_decision_patterns(self,
                                       market_analysis: Dict[str, Any],
                                       strategy: Dict[str, Any],
                                       risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get historical patterns relevant to this decision.
        
        Args:
            market_analysis: Market conditions
            strategy: Proposed strategy
            risk_assessment: Risk assessment
            
        Returns:
            Relevant historical patterns
        """
        vix_level = market_analysis.get("vix_analysis", {}).get("level", 20)
        vix_regime = market_analysis.get("vix_analysis", {}).get("regime", "normal")
        
        # Find similar historical decisions
        similar_decisions = TRADING_MEMORY.get_similar_cto_decisions(
            vix_level=vix_level,
            strategy_type=strategy.get("strategy_type"),
            risk_score=risk_assessment.get("risk_score", 50),
            lookback_days=90
        )
        
        if not similar_decisions:
            return {"status": "no_similar_decisions"}
        
        # Analyze outcomes
        outcomes = {
            "total_similar": len(similar_decisions),
            "executed": sum(1 for d in similar_decisions if d["executed"]),
            "profitable": sum(1 for d in similar_decisions if d["profitable"]),
            "risk_overrides": sum(1 for d in similar_decisions if d["risk_overridden"])
        }
        
        # Success rates
        if outcomes["executed"] > 0:
            outcomes["success_rate"] = outcomes["profitable"] / outcomes["executed"]
            outcomes["override_success"] = sum(
                1 for d in similar_decisions 
                if d["risk_overridden"] and d["profitable"]
            ) / max(1, outcomes["risk_overrides"])
        
        # Best and worst examples
        sorted_decisions = sorted(similar_decisions, key=lambda x: x.get("pnl", 0))
        outcomes["worst_example"] = sorted_decisions[0] if sorted_decisions else None
        outcomes["best_example"] = sorted_decisions[-1] if sorted_decisions else None
        
        return outcomes
    
    def _apply_cto_judgment(self,
                           decision: Dict[str, Any],
                           market_analysis: Dict[str, Any],
                           strategy: Dict[str, Any],
                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply veteran CTO judgment overrides.
        
        Args:
            decision: Initial CTO decision
            market_analysis: Market conditions
            strategy: Proposed strategy
            risk_assessment: Risk assessment
            
        Returns:
            Modified decision with CTO overrides
        """
        vix_level = market_analysis.get("vix_analysis", {}).get("level", 20)
        vix_context = market_analysis.get("vix_analysis", {}).get("context", "")
        
        # Override 1: Never override risk in dangerous VIX patterns
        if risk_assessment.get("approval") == "rejected":
            risk_pattern = risk_assessment.get("pattern_recognition", {}).get("risk_regime")
            if risk_pattern in ["dangerous", "extreme"]:
                decision["risk_overrides"]["accept_risk_rejection"] = False
                decision["rationale"] += " | Risk pattern too dangerous to override"
        
        # Override 2: VIX regime transitions
        if "transition" in vix_context or "changing" in vix_context:
            if decision["decision"] == "execute":
                # Reduce size during transitions
                current_size = decision["execution_parameters"]["final_contracts"]
                decision["execution_parameters"]["final_contracts"] = max(1, current_size // 2)
                decision["vix_consideration"]["size_adjustment"] = "decrease"
                decision["rationale"] += " | VIX transitioning - reduced size"
        
        # Override 3: Team conflict resolution
        team_trust = decision.get("team_assessment", {}).get("trust_score", 100)
        if team_trust < 50 and decision["decision"] == "execute":
            decision["decision"] = "wait"
            decision["rationale"] = "Team alignment too low - waiting for clarity"
        
        # Override 4: Pattern says skip despite team optimism
        pattern = decision.get("pattern_recognition", {})
        if pattern.get("similar_historical"):
            # Check if similar patterns ended badly
            for historical in pattern["similar_historical"]:
                if "2018-02" in historical or "flash" in historical.lower():
                    if vix_level < 15:
                        decision["decision"] = "skip"
                        decision["rationale"] = f"Pattern similar to {historical} - too dangerous"
                        break
        
        # Override 5: Friday afternoon special rules
        if datetime.now().weekday() == 4:
            minutes_to_close = TIMEZONE_HANDLER.minutes_to_close()
            if minutes_to_close < 90:
                if decision["decision"] == "execute":
                    decision["execution_parameters"]["final_contracts"] = 1
                    decision["execution_parameters"]["urgency"] = "patient"
                    decision["cto_notes"] = "Friday afternoon - minimum size and patience"
        
        # Override 6: Consecutive losses
        if self.risk_manager.consecutive_losses >= 2:
            if decision["decision"] == "execute":
                decision["execution_parameters"]["urgency"] = "patient"
                decision["cto_notes"] = "Consecutive losses - extra patience required"
        
        return decision
    
    def _decide_on_retry(self,
                        original: Dict[str, Any],
                        alternative: Dict[str, Any],
                        validation: Dict[str, Any],
                        rejection_reason: str,
                        attempt: int) -> Dict[str, Any]:
        """
        Decide whether to retry with alternative strategy.
        
        Args:
            original: Original rejected strategy
            alternative: Proposed alternative
            validation: Risk validation of alternative
            rejection_reason: Original rejection reason
            attempt: Current attempt number
            
        Returns:
            Retry decision
        """
        retry_prompt = f"""
        Decide on retry using your pattern recognition.
        
        Original Strategy: {json.dumps(original, indent=2)}
        Rejection: {rejection_reason}
        Alternative: {json.dumps(alternative, indent=2)}
        Risk Validation: {json.dumps(validation, indent=2)}
        
        Attempt: {attempt} of 3
        Time to Close: {TIMEZONE_HANDLER.minutes_to_close()} minutes
        
        Have you seen this rejection pattern before? Will the alternative work?
        
        Respond with:
        {{
            "approve_retry": true|false,
            "confidence": 0-100,
            "modification": "what changed",
            "pattern_match": "similar historical example",
            "modifications": {{
                "strikes": "widened|tightened|same",
                "size": "reduced|same",
                "strategy": "changed|same"
            }},
            "reason": "specific reasoning"
        }}
        """
        
        try:
            response = self.agent.execute(retry_prompt)
            decision = json.loads(response)
            
            # Record retry pattern for learning
            self._record_retry_pattern(original, alternative, decision)
            
            return decision
            
        except Exception as e:
            return {
                "approve_retry": False,
                "reason": f"Retry decision failed: {e}"
            }
    
    def _coordinate_execution(self,
                            decision: Dict[str, Any],
                            strategy: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> Tuple[TradingDecision, Dict[str, Any]]:
        """
        Coordinate trade execution with full context.
        
        Args:
            decision: CTO's final decision
            strategy: Strategy to execute
            risk_assessment: Risk parameters
            
        Returns:
            Tuple of (EXECUTE decision, execution details)
        """
        # Build complete execution package
        execution_package = {
            "strategy": strategy,
            "risk_limits": {
                "stop_loss": risk_assessment.get("stop_loss_plan", {}).get("initial_stop"),
                "trailing_stop": risk_assessment.get("stop_loss_plan", {}).get("trailing_stop"),
                "time_stop": risk_assessment.get("stop_loss_plan", {}).get("time_stop"),
                "emergency_exit": risk_assessment.get("stop_loss_plan", {}).get("emergency_exit")
            },
            "execution_params": decision["execution_parameters"],
            "vix_context": {
                "level": strategy.get("market_context", {}).get("vix_level"),
                "regime": strategy.get("market_context", {}).get("vix_regime")
            },
            "cto_notes": decision.get("cto_notes", ""),
            "pattern": decision.get("pattern_recognition", {}).get("current_setup", "")
        }
        
        # Update session tracking
        self.trades_today += 1
        self.session_patterns.append(execution_package["pattern"])
        
        # Log execution decision
        self.logger.info("CTO: Trade approved for execution")
        self.logger.info(f"  Strategy: {strategy['strategy_type']} ({strategy.get('strategy_variant', 'standard')})")
        self.logger.info(f"  Contracts: {decision['execution_parameters']['final_contracts']}")
        self.logger.info(f"  Urgency: {decision['execution_parameters']['urgency']}")
        self.logger.info(f"  VIX Context: {execution_package['vix_context']}")
        
        return (TradingDecision.EXECUTE, execution_package)
    
    def _record_decision_pattern(self, decision: Dict[str, Any], market_analysis: Dict[str, Any]) -> None:
        """Record decision pattern for learning."""
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision["decision"],
            "confidence": decision["confidence"],
            "vix_level": market_analysis.get("vix_analysis", {}).get("level"),
            "vix_regime": market_analysis.get("vix_analysis", {}).get("regime"),
            "pattern": decision.get("pattern_recognition", {}).get("current_setup"),
            "team_alignment": decision.get("team_assessment", {}).get("alignment_level"),
            "risk_override": decision.get("risk_overrides", {}).get("accept_risk_rejection", False)
        }
        
        self.decision_history.append(pattern)
        TRADING_MEMORY.store_cto_decision(pattern)
    
    def _record_rejection_pattern(self, strategy: Dict[str, Any], rejection_reason: str) -> None:
        """Record rejection pattern for learning."""
        TRADING_MEMORY.store_rejection_pattern(strategy, rejection_reason)
    
    def _record_retry_pattern(self, original: Dict[str, Any], alternative: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """Record retry pattern for learning."""
        TRADING_MEMORY.store_retry_pattern(original, alternative, decision)
    
    # Keep all the existing helper methods from original
    def _pre_flight_checks(self) -> bool:
        """Perform pre-flight checks before initiating workflow."""
        if not TIMEZONE_HANDLER.is_market_open():
            self.logger.info("CTO: Market is closed")
            return False
        
        if self.trades_today >= self.config.max_daily_trades:
            self.logger.warning("CTO: Daily trade limit reached")
            return False
        
        if self.emergency_stop:
            self.logger.error("CTO: Emergency stop is active")
            return False
        
        minutes_to_close = TIMEZONE_HANDLER.minutes_to_close()
        if minutes_to_close < 20:
            self.logger.warning(f"CTO: Only {minutes_to_close:.0f} minutes to close - too late for new 0DTE")
            return False
        
        window_quality = TIMEZONE_HANDLER.get_spain_trading_window()
        if window_quality == "closed":
            self.logger.info("CTO: Outside Spain trading window")
            return False
        
        return True
    
    def _get_market_analysis(self) -> Dict[str, Any]:
        """Get market analysis with REAL data from IB."""
        market_data_json = get_real_market_data.run()
        market_data = json.loads(market_data_json)
        
        if "error" in market_data:
            self.logger.error(f"Failed to get market data: {market_data.get('error')}")
            return None
        
        return self.market_analyst.analyze_market(market_data)
    
    def _get_strategy_design(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy design with REAL option chain from IB."""
        options_json = get_real_option_chain.run()
        options_data = json.loads(options_json)
        
        if "error" in options_data:
            self.logger.error(f"Failed to get options: {options_data.get('error')}")
            return None
        
        return self.strategy_architect.design_strategy(
            market_analysis, 
            options_data["options"]
        )
    
    def _get_risk_assessment(self, 
                            strategy: Dict[str, Any],
                            market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk assessment from Risk Manager."""
        portfolio_state = {
            "active_positions": len(self.risk_manager.active_positions),
            "session_pnl": self.risk_manager.session_pnl,
            "margin_used": 0  # Would calculate from actual positions
        }
        
        return self.risk_manager.assess_trade_risk(
            strategy, market_analysis, portfolio_state
        )
    
    def _should_design_strategy(self, market_analysis: Dict[str, Any]) -> bool:
        """Determine if we should proceed to strategy design."""
        # Skip if regime too volatile with high VIX
        vix = market_analysis.get("vix_analysis", {})
        if vix.get("regime") == "extreme" or vix.get("level", 0) > 35:
            return False
        
        # Skip if no favorable strategies
        favorable = market_analysis.get("trading_implications", {}).get("favorable_setups", [])
        if not favorable or favorable == ["skip"]:
            return False
        
        # Skip if pattern recognition shows danger
        if "blow_up" in market_analysis.get("pattern_recognition", {}).get("current_pattern", "").lower():
            return False
        
        return True
    
    async def _get_current_market_snapshot(self) -> Dict[str, Any]:
        """Get current market snapshot with REAL data."""
        market_data_json = await get_real_market_data()
        market_data = json.loads(market_data_json)
        
        market_data["time_to_close"] = TIMEZONE_HANDLER.minutes_to_close()
        market_data["market_phase"] = TIMEZONE_HANDLER.get_market_phase()[0]
        
        return market_data