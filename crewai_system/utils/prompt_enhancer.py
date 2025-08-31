# crewai_system/utils/prompt_enhancer.py
"""
Dynamic Prompt Enhancement Utilities
=====================================
Adds real-time context to agent prompts based on market conditions.

Author: Quantum View AI Trading System
Version: 1.0
Date: December 2024
"""

from datetime import datetime
from typing import Dict, Any, Optional
import logging


class PromptEnhancer:
    """
    Enhances agent prompts with dynamic market regime context.
    
    This helps agents understand the current market environment
    and adjust their decision-making accordingly.
    """
    
    def __init__(self):
        """Initialize the prompt enhancer."""
        self.logger = logging.getLogger("PromptEnhancer")
    
    def enhance_with_regime(self, 
                           base_prompt: str, 
                           vix_level: float,
                           minutes_to_close: int,
                           current_pnl: float = 0,
                           positions_open: int = 0) -> str:
        """
        Enhance prompt with current market regime context.
        
        Args:
            base_prompt: Original prompt to enhance
            vix_level: Current VIX level
            minutes_to_close: Minutes until market close
            current_pnl: Current day's P&L
            positions_open: Number of open positions
            
        Returns:
            Enhanced prompt with regime context
        """
        regime_context = self._get_regime_context(vix_level)
        time_context = self._get_time_context(minutes_to_close)
        risk_context = self._get_risk_context(current_pnl, positions_open)
        
        # Build enhanced prompt
        enhanced = f"""
=============== CURRENT MARKET CONTEXT ===============
{regime_context}
{time_context}
{risk_context}
=====================================================

{base_prompt}
"""
        return enhanced
    
    def _get_regime_context(self, vix_level: float) -> str:
        """
        Get VIX regime context.
        
        Args:
            vix_level: Current VIX level
            
        Returns:
            Regime context string
        """
        if vix_level < 12:
            return """🔴 REGIME: Ultra-Low VIX (<12) - EXTREME CAUTION
- Compression at dangerous levels, violent expansion likely
- Reduce all position sizes by 50%
- Avoid short volatility strategies
- Remember: Feb 5, 2018 started at VIX 11"""
            
        elif vix_level < 15:
            return """⚠️  REGIME: Low VIX (12-15) - HEIGHTENED CAUTION
- Compression phase, expansion risk elevated
- Consider reducing position sizes by 25%
- Wider strikes recommended
- Watch for sudden volatility spikes"""
            
        elif vix_level < 20:
            return """✅ REGIME: Normal VIX (15-20) - OPTIMAL ZONE
- Balanced risk/reward for 0DTE
- Normal position sizing appropriate
- All strategies available
- Sweet spot for credit spreads"""
            
        elif vix_level < 25:
            return """💰 REGIME: Elevated VIX (20-25) - OPPORTUNITY ZONE
- Higher premiums available
- Can be more aggressive with sizing
- Focus on defined risk strategies
- Volatility mean reversion plays work"""
            
        elif vix_level < 30:
            return """🚨 REGIME: High VIX (25-30) - DEFENSIVE MODE
- Market stress elevated
- Reduce position sizes
- Wider strikes mandatory
- Quick profits, don't be greedy"""
            
        else:
            return """🔥 REGIME: Extreme VIX (>30) - CRISIS MODE
- Market panic conditions
- Minimum position sizes only
- Focus on capital preservation
- Consider sitting out"""
    
    def _get_time_context(self, minutes_to_close: int) -> str:
        """
        Get time-of-day context for 0DTE.
        
        Args:
            minutes_to_close: Minutes until market close
            
        Returns:
            Time context string
        """
        if minutes_to_close > 360:  # More than 6 hours (morning)
            return """⏰ TIME: Market Open Phase
- Higher volatility expected
- Wait for initial range to establish
- Best entry after first 30 minutes
- Full position sizes available"""
            
        elif minutes_to_close > 240:  # 4-6 hours (late morning)
            return """⏰ TIME: Mid-Morning Optimal Window
- Trends establishing
- Best risk/reward period
- Full position sizes
- All strategies available"""
            
        elif minutes_to_close > 150:  # 2.5-4 hours (early afternoon)
            return """⏰ TIME: Afternoon Theta Burn
- Accelerating time decay
- Reduce new position sizes by 25%
- Focus on high probability setups
- Monitor gamma risk closely"""
            
        elif minutes_to_close > 60:  # 1-2.5 hours (late afternoon)
            return """⏰ TIME: Power Hour Approach
- Gamma risk increasing rapidly
- Reduce new position sizes by 50%
- Consider closing winners
- No new complex strategies"""
            
        elif minutes_to_close > 30:  # 30-60 minutes
            return """⏰ TIME: Final Hour - DEFENSIVE ONLY
- Extreme gamma risk
- EXIT ONLY - no new positions
- Close all positions by 3:45 PM
- Assignment risk high"""
            
        else:  # Less than 30 minutes
            return """⏰ TIME: MARKET CLOSING - NO TRADES
- CLOSE ALL POSITIONS IMMEDIATELY
- Pin risk at maximum
- Liquidity disappearing
- Assignment risk extreme"""
    
    def _get_risk_context(self, current_pnl: float, positions_open: int) -> str:
        """
        Get risk management context.
        
        Args:
            current_pnl: Current day's P&L
            positions_open: Number of open positions
            
        Returns:
            Risk context string
        """
        risk_notes = []
        
        # P&L context
        if current_pnl > 0:
            risk_notes.append(f"📈 Day P&L: +${current_pnl:.2f} (protect profits)")
        else:
            risk_notes.append(f"📉 Day P&L: -${abs(current_pnl):.2f} (manage risk)")
        
        # Position context
        if positions_open >= 3:
            risk_notes.append("⚠️  Maximum positions open - no new trades")
        elif positions_open >= 2:
            risk_notes.append("📊 Near position limit - be selective")
        else:
            risk_notes.append(f"📊 {positions_open} positions open")
        
        # Build risk context
        return "RISK STATUS: " + " | ".join(risk_notes)
    
    def get_critical_warnings(self, 
                             vix_level: float,
                             minutes_to_close: int) -> Optional[str]:
        """
        Get critical warnings that should override normal operation.
        
        Args:
            vix_level: Current VIX level
            minutes_to_close: Minutes until market close
            
        Returns:
            Critical warning if applicable, None otherwise
        """
        if minutes_to_close < 30:
            return "🚨 CRITICAL: Less than 30 minutes to close - EXIT ALL POSITIONS"
        
        if vix_level > 35:
            return "🚨 CRITICAL: VIX above 35 - Crisis mode, consider closing all positions"
        
        if vix_level < 10:
            return "🚨 CRITICAL: VIX below 10 - Extreme compression, massive spike risk"
        
        return None


# Singleton instance for easy import
PROMPT_ENHANCER = PromptEnhancer()