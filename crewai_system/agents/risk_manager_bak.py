# crewai_system/agents/risk_manager.py
"""
Risk Manager Agent
==================
Validates all trading decisions against risk parameters and portfolio limits.
Monitors existing positions and recommends adjustments or exits.

Key Responsibilities:
- Validate new trades against risk limits
- Monitor portfolio exposure
- Track P&L and drawdowns  
- Recommend position adjustments
- Enforce emergency stops

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

from crewai import Agent
from langchain_community.llms import Ollama
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from ..config.config_loader import TRADING_CONFIG
from ..config.timezone_handler import TIMEZONE_HANDLER
from ..memory.persistence import TRADING_MEMORY

from crewai_system.config.llm_factory import LLMFactory

class RiskManagerAgent:
    """
    Risk Manager - The guardian of capital.
    
    This agent ensures all trades comply with risk parameters
    and monitors portfolio health continuously.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Risk Manager agent.
        
        Args:
            tools: List of risk management tools
        """
        self.logger = logging.getLogger("RiskManager")
        self.config = TRADING_CONFIG
        
        # Track session risk metrics
        self.session_trades = 0
        self.session_pnl = 0.0
        self.consecutive_losses = 0
        self.active_positions = []
        
        self.llm = LLMFactory.make(self.config, timeout_key="risk_assessment")
               
        # Create the agent
        self.agent = Agent(
            role="Chief Risk Officer",
            
            goal="""Protect trading capital while enabling profitable opportunities.
            Ensure all trades comply with risk parameters, monitor portfolio exposure,
            and prevent catastrophic losses. Balance opportunity with preservation.""",
            
            backstory="""You are a seasoned risk manager who has protected trading
            capital through multiple market crashes and volatility events. You understand
            that 0DTE options can move from worthless to devastating losses in minutes.
            
            Your philosophy: "Live to trade another day." You've seen traders blow up
            from one bad position and know that consistent small wins beat sporadic
            large gains. You enforce discipline especially during the final hour when
            0DTE gamma risk explodes.
            
            You support a Spain-based trader, so you're extra vigilant during late
            evening trades when fatigue might impair judgment.""",
            
            tools=tools,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,  # Risk decisions must be fast
        )
        
        # Risk assessment prompt
        self.risk_prompt = """
        Assess the risk of this proposed trade.
        
        Proposed Strategy:
        {strategy}
        
        Current Portfolio:
        - Open Positions: {open_positions}
        - Session P&L: ${session_pnl}
        - Trades Today: {session_trades}/{max_trades}
        - Consecutive Losses: {consecutive_losses}
        
        Market Conditions:
        {market_conditions}
        
        Time Factors:
        - Minutes to Close: {minutes_to_close}
        - Trading Window Quality: {window_quality}
        
        Account Limits:
        - Max Risk Per Trade: {max_risk_pct}%
        - Max Daily Loss: {max_daily_loss_pct}%
        - Max Contracts: {max_contracts}
        
        Provide risk assessment in the following JSON structure:
        {{
            "approval": "approved|rejected|conditional",
            "risk_score": 0-100,
            "position_sizing": {{
                "recommended_contracts": number,
                "max_allowed_contracts": number,
                "sizing_rationale": "string"
            }},
            "risk_factors": [
                {{
                    "factor": "string",
                    "severity": "low|medium|high|critical",
                    "mitigation": "string"
                }}
            ],
            "portfolio_impact": {{
                "total_exposure_after": number,
                "margin_usage_pct": number,
                "correlation_risk": "low|medium|high"
            }},
            "conditions": [
                "string"  // Any conditions for approval
            ],
            "stop_loss_level": number,
            "profit_target": number,
            "max_hold_time_minutes": number,
            "rationale": "string",
            "confidence": 0-100
        }}
        """
    
    def assess_trade_risk(self,
                         strategy: Dict[str, Any],
                         market_analysis: Dict[str, Any],
                         portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk of proposed trade.
        
        Args:
            strategy: Proposed strategy from Strategy Architect
            market_analysis: Current market analysis
            portfolio_state: Current portfolio state
            
        Returns:
            Risk assessment with approval decision
        """
        try:
            # Get time factors
            minutes_to_close = TIMEZONE_HANDLER.minutes_to_close()
            window_quality = TIMEZONE_HANDLER.get_spain_trading_window()
            
            # Prepare risk assessment request
            risk_request = self.risk_prompt.format(
                strategy=json.dumps(strategy, indent=2),
                open_positions=len(self.active_positions),
                session_pnl=self.session_pnl,
                session_trades=self.session_trades,
                max_trades=self.config.max_daily_trades,
                consecutive_losses=self.consecutive_losses,
                market_conditions=json.dumps(market_analysis, indent=2),
                minutes_to_close=minutes_to_close,
                window_quality=window_quality,
                max_risk_pct=self.config.max_risk_per_trade_pct,
                max_daily_loss_pct=self.config.max_daily_loss_pct,
                max_contracts=self.config.max_contracts_per_trade
            )
            
            # Get risk assessment
            self.logger.info("Assessing trade risk...")
            response = self.agent.execute(risk_request)
            
            # Parse response
            assessment = json.loads(response)
            assessment["timestamp"] = datetime.now().isoformat()
            assessment["risk_manager"] = "risk_manager"
            
            # Apply hard overrides for safety
            assessment = self._apply_risk_overrides(assessment, strategy, minutes_to_close)
            
            # Log decision
            self.logger.info(
                f"Risk Assessment: {assessment['approval'].upper()} "
                f"(Risk Score: {assessment['risk_score']}, "
                f"Contracts: {assessment['position_sizing']['recommended_contracts']})"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return self._reject_trade("Risk assessment system failure")
    
    def validate_before_retry(self,
                            original_strategy: Dict[str, Any],
                            alternative_strategy: Dict[str, Any],
                            rejection_reason: str) -> Dict[str, Any]:
        """
        Validate alternative strategy after IB rejection.
        
        Called by Chief Trading Officer before retry.
        
        Args:
            original_strategy: The rejected strategy
            alternative_strategy: Proposed alternative
            rejection_reason: Why original was rejected
            
        Returns:
            Validation result with approval/rejection
        """
        validation_prompt = f"""
        Validate this alternative strategy after broker rejection.
        
        Original Strategy: {json.dumps(original_strategy, indent=2)}
        Rejection Reason: {rejection_reason}
        Alternative Strategy: {json.dumps(alternative_strategy, indent=2)}
        
        Is the alternative safer and likely to be accepted?
        
        Respond with:
        {{
            "validation": "approved|rejected",
            "risk_comparison": "safer|same|riskier",
            "acceptance_probability": 0-100,
            "concerns": ["string"],
            "rationale": "string"
        }}
        """
        
        try:
            response = self.agent.execute(validation_prompt)
            validation = json.loads(response)
            
            self.logger.info(
                f"Alternative validation: {validation['validation']} "
                f"(Acceptance probability: {validation['acceptance_probability']}%)"
            )
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "validation": "rejected",
                "rationale": f"Validation system error: {e}"
            }
    
    def _apply_risk_overrides(self,
                             assessment: Dict[str, Any],
                             strategy: Dict[str, Any],
                             minutes_to_close: float) -> Dict[str, Any]:
        """
        Apply hard risk rules that override agent decisions.
        
        Safety mechanism to prevent dangerous trades.
        
        Args:
            assessment: Agent's risk assessment
            strategy: Proposed strategy
            minutes_to_close: Time remaining in session
            
        Returns:
            Modified assessment with overrides applied
        """
        # Override 1: No new trades in final 20 minutes for 0DTE
        if minutes_to_close <= 20:
            assessment["approval"] = "rejected"
            assessment["rationale"] = "Too close to market close for new 0DTE positions"
        
        # Override 2: Daily loss limit
        if self.session_pnl <= -(self.config.max_daily_loss_pct / 100 * 15000):  # Assuming $15k account
            assessment["approval"] = "rejected"
            assessment["rationale"] = "Daily loss limit reached"
        
        # Override 3: Consecutive loss limit
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            assessment["approval"] = "rejected"
            assessment["rationale"] = f"Consecutive loss limit ({self.consecutive_losses}) reached"
        
        # Override 4: Max daily trades
        if self.session_trades >= self.config.max_daily_trades:
            assessment["approval"] = "rejected"
            assessment["rationale"] = "Daily trade limit reached"
        
        # Override 5: Reduce size if late in day
        if minutes_to_close <= 60 and assessment["approval"] == "approved":
            max_contracts = min(1, assessment["position_sizing"]["recommended_contracts"])
            assessment["position_sizing"]["recommended_contracts"] = max_contracts
            assessment["position_sizing"]["sizing_rationale"] += " (Reduced for final hour)"
        
        return assessment
    
    def _reject_trade(self, reason: str) -> Dict[str, Any]:
        """
        Create a rejection response.
        
        Args:
            reason: Why trade is rejected
            
        Returns:
            Rejection assessment
        """
        return {
            "approval": "rejected",
            "risk_score": 100,
            "position_sizing": {
                "recommended_contracts": 0,
                "max_allowed_contracts": 0,
                "sizing_rationale": "Trade rejected"
            },
            "risk_factors": [
                {
                    "factor": reason,
                    "severity": "critical",
                    "mitigation": "Skip this trade"
                }
            ],
            "rationale": reason,
            "confidence": 100,
            "timestamp": datetime.now().isoformat(),
            "risk_manager": "risk_manager"
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