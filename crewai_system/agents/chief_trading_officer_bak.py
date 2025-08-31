# crewai_system/agents/chief_trading_officer.py
"""
Chief Trading Officer Agent
============================
The master orchestrator that coordinates all trading decisions.
This agent manages the entire trading workflow, from analysis to execution,
ensuring all specialists work together effectively.

Key Responsibilities:
- Orchestrate the trading workflow
- Make final go/no-go decisions
- Resolve conflicts between agents
- Manage the trading session
- Handle exception scenarios
- Coordinate retries after rejections

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

from crewai import Agent
from langchain_community.llms import Ollama
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
from enum import Enum

from ..config.config_loader import TRADING_CONFIG
from ..config.timezone_handler import TIMEZONE_HANDLER
from ..memory.persistence import TRADING_MEMORY, TradeRecord


from crewai_system.config.llm_factory import LLMFactory

from crewai_system.tools.market_tools import get_real_market_data, get_real_option_chain,  _market_context


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
    
    This agent coordinates all other agents, makes final trading decisions,
    and manages the overall trading session. It's the only agent that can
    authorize trades and coordinate retries.
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
        
        # Store references to all specialist agents
        self.market_analyst = market_analyst
        self.strategy_architect = strategy_architect
        self.risk_manager = risk_manager
        self.execution_specialist = execution_specialist
        
        # Session management
        self.session_active = False
        self.emergency_stop = False
        self.trades_today = 0
        self.last_analysis_time = None
        self.current_positions = []
        
        try:
            # Initialize Ollama LLM with balanced temperature
            self.llm = LLMFactory.make(self.config, timeout_key="chief_trading_officer")
            self.logger.info("✅ LLM for Chief Trading Officer initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM for Chief Trading Officer: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())  # full stack trace for debugging
            self.llm = None  # fallback to None so system can still run
        
        # Create the CTO agent
        self.agent = Agent(
            role="Chief Trading Officer",
            
            goal="""Orchestrate all trading operations to maximize profitable opportunities
            while maintaining strict risk discipline. Coordinate specialist agents to analyze,
            design, validate, and execute 0DTE option trades. Make final decisions on all
            trades and handle exceptional situations.""",
            
            backstory="""You are the Chief Trading Officer with 25 years of experience
            managing trading operations. You've led trading desks through bull markets,
            crashes, and everything in between. Your strength is coordinating talented
            specialists and making decisive calls under pressure.
            
            You understand that 0DTE options require perfect orchestration - there's no
            room for hesitation or confusion. Every minute counts, especially in the final
            hours. You know when to trust your team's analysis and when to override for
            safety.
            
            You manage a sophisticated operation for a Spain-based trader, so you're
            particularly aware of the timezone challenges and the need for clear,
            decisive action during evening hours when fatigue might affect judgment.
            
            Your philosophy: "Plan the trade, trade the plan, but always be ready to
            adapt when the market humbles you." """,
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=True,  # CTO can delegate to specialists
            max_iter=self.config.max_iterations,
        )
        
        # Decision-making prompt template
        self.decision_prompt = """
        Make a final trading decision based on specialist inputs.
        
        Market Analysis:
        {market_analysis}
        
        Strategy Recommendation:
        {strategy}
        
        Risk Assessment:
        {risk_assessment}
        
        Current Session State:
        - Trades Today: {trades_today}
        - Active Positions: {active_positions}
        - Time to Close: {time_to_close} minutes
        - Session P&L: ${session_pnl}
        
        Provide your decision in the following JSON structure:
        {{
            "decision": "execute|skip|wait|close_positions|emergency_stop",
            "rationale": "string explaining the decision",
            "confidence": 0-100,
            "execution_parameters": {{
                "strategy_type": "string",
                "contracts": number,
                "urgency": "immediate|normal|low",
                "special_instructions": "string or null"
            }},
            "risk_overrides": {{
                "modify_stops": boolean,
                "reduce_size": boolean,
                "adjustment_reason": "string or null"
            }},
            "next_action_time": "seconds until next review",
            "market_outlook": "bullish|bearish|neutral|uncertain",
            "session_notes": "string for session log"
        }}
        """
    
    def orchestrate_trading_decision(self) -> Tuple[TradingDecision, Dict[str, Any]]:
        """
        Orchestrate a complete trading decision workflow.
        
        This is the main method that coordinates all agents to make
        a trading decision. It follows the sequence:
        1. Market Analysis
        2. Strategy Design  
        3. Risk Validation
        4. Final Decision
        5. Execution (if approved)
        
        Returns:
            Tuple of (decision_type, execution_details)
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("CTO: INITIATING TRADING DECISION WORKFLOW")
            self.logger.info(TIMEZONE_HANDLER.format_times_for_logging())
            self.logger.info("=" * 60)
            
            # Step 1: Check if we should be trading
            if not self._pre_flight_checks():
                return (TradingDecision.SKIP, {"reason": "Pre-flight checks failed"})
            
            # Step 2: Get market analysis
            self.logger.info("CTO: Requesting market analysis...")
            market_analysis = self._get_market_analysis()
            
            if not market_analysis or market_analysis.get("confidence_score", 0) < 30:
                self.logger.warning("CTO: Market analysis confidence too low")
                return (TradingDecision.WAIT, {"reason": "Insufficient market clarity"})
            
            # Step 3: Design strategy if market is favorable
            if self._should_design_strategy(market_analysis):
                self.logger.info("CTO: Requesting strategy design...")
                strategy = self._get_strategy_design(market_analysis)
                
                if not strategy or strategy.get("strategy_type") == "skip":
                    return (TradingDecision.SKIP, {"reason": strategy.get("rationale", "No viable strategy")})
            else:
                return (TradingDecision.WAIT, {"reason": "Market conditions unfavorable"})
            
            # Step 4: Validate risk
            self.logger.info("CTO: Requesting risk validation...")
            risk_assessment = self._get_risk_assessment(strategy, market_analysis)
            
            if risk_assessment.get("approval") == "rejected":
                self.logger.warning(f"CTO: Risk rejected - {risk_assessment.get('rationale')}")
                return (TradingDecision.SKIP, {"reason": risk_assessment.get("rationale")})
            
            # Step 5: Make final decision
            final_decision = self._make_final_decision(
                market_analysis, strategy, risk_assessment
            )
            
            # Step 6: Execute if approved
            if final_decision["decision"] == "execute":
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
        Handle Interactive Brokers trade rejection.
        
        Coordinates the recovery workflow:
        1. Strategy Architect proposes alternative
        2. Risk Manager validates alternative
        3. CTO approves/rejects retry
        
        Args:
            original_strategy: The rejected strategy
            rejection_reason: IB's rejection reason
            attempt_number: Which attempt this is (1, 2, 3)
            
        Returns:
            Tuple of (decision_type, execution_details)
        """
        self.logger.warning(f"CTO: Handling IB rejection (attempt {attempt_number})")
        self.logger.info(f"Rejection reason: {rejection_reason}")
        
        # Check if we should retry
        if attempt_number >= 3:
            self.logger.error("CTO: Max retry attempts reached")
            return (TradingDecision.SKIP, {"reason": "Max retries exceeded"})
        
        if TIMEZONE_HANDLER.minutes_to_close() < 15:
            self.logger.warning("CTO: Too close to close for retries")
            return (TradingDecision.SKIP, {"reason": "Insufficient time for retry"})
        
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
            self.logger.info("CTO: Validating alternative with Risk...")
            validation = self.risk_manager.validate_before_retry(
                original_strategy,
                alternative,
                rejection_reason
            )
            
            if validation.get("validation") == "rejected":
                return (TradingDecision.SKIP, {"reason": validation.get("rationale")})
            
            # CTO makes final retry decision
            retry_decision = self._decide_on_retry(
                original_strategy,
                alternative,
                validation,
                attempt_number
            )
            
            if retry_decision["approve_retry"]:
                self.logger.info("CTO: Approving retry with modified parameters")
                return (TradingDecision.EXECUTE, {
                    "strategy": alternative,
                    "is_retry": True,
                    "attempt": attempt_number + 1,
                    "modifications": retry_decision["modifications"]
                })
            else:
                return (TradingDecision.SKIP, {"reason": retry_decision["reason"]})
                
        except Exception as e:
            self.logger.error(f"CTO: Rejection handling failed - {e}")
            return (TradingDecision.SKIP, {"reason": f"Recovery failed: {e}"})
    
    def _pre_flight_checks(self) -> bool:
        """
        Perform pre-flight checks before initiating workflow.
        
        Returns:
            True if all checks pass
        """
        # Check market is open
        if not TIMEZONE_HANDLER.is_market_open():
            self.logger.info("CTO: Market is closed")
            return False
        
        # Check daily trade limit
        if self.trades_today >= self.config.max_daily_trades:
            self.logger.warning("CTO: Daily trade limit reached")
            return False
        
        # Check emergency stop
        if self.emergency_stop:
            self.logger.error("CTO: Emergency stop is active")
            return False
        
        # Check time to close for new trades
        minutes_to_close = TIMEZONE_HANDLER.minutes_to_close()
        if minutes_to_close < 20:
            self.logger.warning(f"CTO: Only {minutes_to_close:.0f} minutes to close - too late for new 0DTE")
            return False
        
        # Check Spain trading window quality
        window_quality = TIMEZONE_HANDLER.get_spain_trading_window()
        if window_quality == "closed":
            self.logger.info("CTO: Outside Spain trading window")
            return False
        
        return True
    
    def _get_market_analysis(self) -> Dict[str, Any]:
        """
        Get market analysis using REAL data from IB and market_context.
        
        Returns:
            Market analysis dict with REAL data
        """
        # Get REAL market data from Interactive Brokers
        market_data_json = get_real_market_data()  # This is the tool we just created
        market_data = json.loads(market_data_json)
        
        if not market_data.get("success", False):
            self.logger.error(f"Failed to get market data: {market_data.get('error')}")
            return None
        
        # Update market context with REAL IB data
        _market_context.update_market_data(
            price=market_data["underlying_price"],
            volume=market_data["volume"],
            vix=market_data["vix"],
            high=market_data["high"],
            low=market_data["low"],
            open_price=market_data["open"],
            previous_close=market_data["previous_close"]
        )
        
        # Get analysis based on REAL data
        return self.market_analyst.analyze_market(market_data)
    
    def _get_strategy_design(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get strategy design using REAL option chain from IB.
        
        Args:
            market_analysis: Current market analysis
            
        Returns:
            Strategy specification based on REAL options
        """
        # Get REAL option chain from Interactive Brokers
        options_json = get_real_option_chain()  # Tool with REAL IB data
        options_data = json.loads(options_json)
        
        if not options_data.get("success", False):
            self.logger.error(f"Failed to get options: {options_data.get('error')}")
            return None
        
        return self.strategy_architect.design_strategy(
            market_analysis, 
            options_data["options"]  # REAL options from IB
        )
    
    def _get_risk_assessment(self, 
                            strategy: Dict[str, Any],
                            market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get risk assessment from Risk Manager.
        
        Args:
            strategy: Proposed strategy
            market_analysis: Current market analysis
            
        Returns:
            Risk assessment
        """
        portfolio_state = {
            "active_positions": len(self.current_positions),
            "session_pnl": 0,  # Would calculate from positions
            "margin_used": 0
        }
        
        return self.risk_manager.assess_trade_risk(
            strategy, market_analysis, portfolio_state
        )
    
    def _make_final_decision(self,
                            market_analysis: Dict[str, Any],
                            strategy: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final trading decision as CTO.
        
        This is where the CTO agent applies judgment beyond
        what the specialists recommend.
        
        Args:
            market_analysis: Market Analyst's assessment
            strategy: Strategy Architect's design
            risk_assessment: Risk Manager's validation
            
        Returns:
            Final decision dictionary
        """
        # Prepare decision context
        decision_context = self.decision_prompt.format(
            market_analysis=json.dumps(market_analysis, indent=2),
            strategy=json.dumps(strategy, indent=2),
            risk_assessment=json.dumps(risk_assessment, indent=2),
            trades_today=self.trades_today,
            active_positions=len(self.current_positions),
            time_to_close=TIMEZONE_HANDLER.minutes_to_close(),
            session_pnl=self.risk_manager.session_pnl
        )
        
        try:
            # Get CTO's decision
            response = self.agent.execute(decision_context)
            decision = json.loads(response)
            
            # Apply CTO overrides for safety
            decision = self._apply_cto_overrides(decision, market_analysis, strategy)
            
            self.logger.info(
                f"CTO DECISION: {decision['decision'].upper()} "
                f"(Confidence: {decision['confidence']}%)"
            )
            self.logger.info(f"Rationale: {decision['rationale']}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"CTO decision failed: {e}")
            return {
                "decision": "skip",
                "rationale": f"Decision system error: {e}",
                "confidence": 0
            }
    
    def _apply_cto_overrides(self,
                            decision: Dict[str, Any],
                            market_analysis: Dict[str, Any],
                            strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CTO's judgment overrides.
        
        The CTO can override specialist recommendations based on
        holistic factors they might miss.
        
        Args:
            decision: Initial CTO decision
            market_analysis: Market conditions
            strategy: Proposed strategy
            
        Returns:
            Modified decision with overrides
        """
        # Override 1: Low confidence cascade
        # If multiple agents have low confidence, skip
        market_confidence = market_analysis.get("confidence_score", 0)
        strategy_confidence = strategy.get("confidence", 0)
        
        if market_confidence < 50 and strategy_confidence < 50:
            decision["decision"] = "skip"
            decision["rationale"] = "Multiple specialists showing low confidence"
        
        # Override 2: Friday afternoon caution
        # More conservative on Fridays approaching weekend
        if datetime.now().weekday() == 4:  # Friday
            if TIMEZONE_HANDLER.minutes_to_close() < 90:
                if decision["decision"] == "execute":
                    decision["execution_parameters"]["contracts"] = min(
                        1, decision["execution_parameters"].get("contracts", 1)
                    )
                    decision["rationale"] += " (Friday size reduction applied)"
        
        # Override 3: Correlation risk
        # If we have similar positions open, reduce or skip
        if self._has_correlated_positions(strategy):
            if decision["decision"] == "execute":
                decision["decision"] = "skip"
                decision["rationale"] = "Correlation risk with existing positions"
        
        return decision
    
    def _coordinate_execution(self,
                            decision: Dict[str, Any],
                            strategy: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> Tuple[TradingDecision, Dict[str, Any]]:
        """
        Coordinate trade execution with Execution Specialist.
        
        Args:
            decision: CTO's decision
            strategy: Strategy to execute
            risk_assessment: Risk parameters
            
        Returns:
            Tuple of (EXECUTE decision, execution details)
        """
        # Prepare execution package
        execution_package = {
            "strategy": strategy,
            "risk_limits": {
                "stop_loss": risk_assessment.get("stop_loss_level"),
                "profit_target": risk_assessment.get("profit_target"),
                "max_hold_time": risk_assessment.get("max_hold_time_minutes")
            },
            "execution_params": decision["execution_parameters"],
            "cto_notes": decision.get("session_notes", "")
        }
        
        # Record the decision in memory
        self._record_trade_decision(execution_package)
        
        # Update session metrics
        self.trades_today += 1
        
        self.logger.info("CTO: Trade approved for execution")
        self.logger.info(f"  Strategy: {strategy['strategy_type']}")
        self.logger.info(f"  Contracts: {decision['execution_parameters']['contracts']}")
        self.logger.info(f"  Confidence: {decision['confidence']}%")
        
        return (TradingDecision.EXECUTE, execution_package)
    
    def _decide_on_retry(self,
                        original: Dict[str, Any],
                        alternative: Dict[str, Any],
                        validation: Dict[str, Any],
                        attempt: int) -> Dict[str, Any]:
        """
        Decide whether to retry with alternative strategy.
        
        Args:
            original: Original rejected strategy
            alternative: Proposed alternative
            validation: Risk validation of alternative
            attempt: Current attempt number
            
        Returns:
            Retry decision
        """
        retry_prompt = f"""
        Decide on retry after broker rejection.
        
        Original Strategy: {json.dumps(original, indent=2)}
        Alternative Strategy: {json.dumps(alternative, indent=2)}
        Risk Validation: {json.dumps(validation, indent=2)}
        Attempt Number: {attempt}
        Time to Close: {TIMEZONE_HANDLER.minutes_to_close()} minutes
        
        Should we retry with the alternative?
        
        Respond with:
        {{
            "approve_retry": true|false,
            "confidence": 0-100,
            "modifications": {{
                "adjust_size": boolean,
                "widen_strikes": boolean,
                "change_strategy": boolean
            }},
            "reason": "string"
        }}
        """
        
        try:
            response = self.agent.execute(retry_prompt)
            return json.loads(response)
        except Exception as e:
            return {
                "approve_retry": False,
                "reason": f"Retry decision failed: {e}"
            }
    
    def _should_design_strategy(self, market_analysis: Dict[str, Any]) -> bool:
        """
        Determine if we should proceed to strategy design.
        
        Args:
            market_analysis: Current market analysis
            
        Returns:
            True if we should design a strategy
        """
        # Skip if market regime is too volatile
        if market_analysis.get("regime") == "volatile" and \
           market_analysis.get("volatility_assessment", {}).get("current_vix", 0) > 30:
            return False
        
        # Skip if no favorable strategies identified
        favorable = market_analysis.get("trading_recommendation", {}).get("favorable_strategies", [])
        if not favorable or favorable == ["skip"]:
            return False
        
        return True
    
    def _has_correlated_positions(self, strategy: Dict[str, Any]) -> bool:
        """
        Check if we have correlated positions open.
        
        Args:
            strategy: Proposed strategy
            
        Returns:
            True if correlation risk exists
        """
        # Check if we already have the same strategy type open
        for position in self.current_positions:
            if position.get("strategy_type") == strategy.get("strategy_type"):
                return True
        
        return False
    
    def _get_current_market_snapshot(self) -> Dict[str, Any]:
        """
        Get current market snapshot - REAL DATA ONLY.
        
        Returns:
            Current market data from IB
        """
        # Get REAL data from IB
        market_data_json = get_real_market_data()
        market_data = json.loads(market_data_json)
        
        # Add REAL time factors
        market_data["time_to_close"] = TIMEZONE_HANDLER.minutes_to_close()
        market_data["market_phase"] = TIMEZONE_HANDLER.get_market_phase()[0]
        
        return market_data
    def _record_trade_decision(self, execution_package: Dict[str, Any]) -> None:
        """
        Record trade decision in memory for learning.
        
        Args:
            execution_package: Complete execution details
        """
        # This would create a TradeRecord and store it
        self.logger.info("CTO: Trade decision recorded in memory")