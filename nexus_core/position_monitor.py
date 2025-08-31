"""
Position Monitoring System
==========================
Monitors and manages active positions with sophisticated risk management.
Handles profit targets, stop losses, adjustments, and expiration.

Author: Trading Systems
Version: 2.0
Date: August 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import asyncio
import pytz

from config import get_config
from models import (
    Position, PositionState, TradeSetup, StrategyType,
    MovementStats
)
from exceptions import RiskError
from market_context import MarketContext
from nexus_core.execution_engine import ExecutionEngine


@dataclass
class PositionMetrics:
    """Real-time metrics for a position."""
    position_id: str
    current_price: float
    entry_price: float
    
    # P&L metrics
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Risk metrics
    distance_to_short_put: Optional[float]
    distance_to_short_call: Optional[float]
    delta_risk: float
    
    # Time metrics
    time_in_position_minutes: float
    minutes_to_close: float
    
    # Management flags
    profit_target_1_hit: bool = False
    profit_target_2_hit: bool = False
    stop_loss_triggered: bool = False
    adjustment_needed: bool = False
    time_exit_needed: bool = False


@dataclass
class ManagementAction:
    """Action to take on a position."""
    position_id: str
    action_type: str  # 'close_all', 'close_half', 'adjust', 'hold'
    reason: str
    urgency: str  # 'immediate', 'soon', 'normal'
    details: Dict = field(default_factory=dict)


class PositionMonitor:
    """
    Sophisticated position monitoring and management system.
    
    This class monitors all active positions and executes management
    rules based on profit targets, stop losses, and market conditions.
    
    Key features:
    - Real-time P&L tracking
    - Dynamic stop loss adjustment
    - Profit target execution
    - Time-based exits for 0DTE
    - Position adjustments when threatened
    - Emergency risk management
    """
    
    def __init__(
        self,
        context: MarketContext,
        executor: ExecutionEngine,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize position monitor.
        
        Args:
            context: MarketContext for parameters
            executor: ExecutionEngine for closing positions
            logger: Optional logger
        """
        self.context = context
        self.executor = executor
        self.logger = logger or self._setup_logger()
        self.config = get_config()
        
        # Position tracking
        self.active_positions: Dict[str, Position] = {}
        self.position_metrics: Dict[str, PositionMetrics] = {}
        
        # Management rules (from context parameters)
        self.profit_targets = []
        self.stop_loss_multiplier = 2.0
        self.time_exit_minutes = 30
        
        # Performance tracking
        self.stats = {
            "positions_monitored": 0,
            "profit_targets_hit": 0,
            "stop_losses_hit": 0,
            "time_exits": 0,
            "adjustments_made": 0,
            "total_profit": 0.0,
            "total_loss": 0.0
        }
        
        # Timezone setup
        self.et_tz = pytz.timezone("US/Eastern")
        
        self.logger.info("Position Monitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Create logger if none provided."""
        logger = logging.getLogger("PositionMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def add_position(self, position: Position) -> None:
        """
        Add a new position to monitor.
        
        Args:
            position: Position object from execution
        """
        self.active_positions[position.position_id] = position
        position.state = PositionState.ACTIVE
        
        self.logger.info(f"Added position to monitor: {position.position_id}")
        self.logger.info(f"  Strategy: {position.trade_setup.strategy.value}")
        self.logger.info(f"  Entry price: ${position.entry_price:.2f}")
        self.logger.info(f"  Credit: ${position.trade_setup.total_credit:.2f}")
    
    def remove_position(self, position_id: str) -> None:
        """Remove position from monitoring."""
        if position_id in self.active_positions:
            del self.active_positions[position_id]
            if position_id in self.position_metrics:
                del self.position_metrics[position_id]
            
            self.logger.info(f"Removed position: {position_id}")
    
    async def monitor_all_positions(self) -> List[ManagementAction]:
        """
        Monitor all active positions and determine actions.
        
        This is the main method called periodically by the orchestrator.
        
        Returns:
            List of management actions to execute
        """
        if not self.active_positions:
            return []
        
        # Update management parameters from context
        self._update_parameters()
        
        actions = []
        current_price = self.context.current_data.underlying_price if self.context.current_data else None
        
        if not current_price:
            self.logger.warning("No current price available for monitoring")
            return []
        
        # Monitor each position
        for position_id, position in self.active_positions.items():
            # Skip if not active
            if position.state != PositionState.ACTIVE:
                continue
            
            # Calculate metrics
            metrics = self._calculate_metrics(position, current_price)
            self.position_metrics[position_id] = metrics
            
            # Determine action
            action = self._determine_action(position, metrics)
            
            if action and action.action_type != 'hold':
                actions.append(action)
                self.logger.info(f"Action for {position_id}: {action.action_type} - {action.reason}")
        
        self.stats["positions_monitored"] += len(self.active_positions)
        
        return actions
    
    def _update_parameters(self) -> None:
        """Update management parameters from context."""
        params = self.context.get_trading_parameters()
        
        self.profit_targets = params["profit_targets"]
        self.stop_loss_multiplier = params["stop_loss_multiplier"]
        
        # Time exit based on market hours
        hours_status = self.context.get_market_hours_status()
        if hours_status["minutes_to_close"] < 30:
            self.time_exit_minutes = 10  # Exit sooner when very close to close
        else:
            self.time_exit_minutes = 30
    
    def _calculate_metrics(self, position: Position, current_price: float) -> PositionMetrics:
        """
        Calculate real-time metrics for a position - FIXED VERSION.
        
        Args:
            position: Position to analyze
            current_price: Current underlying price
            
        Returns:
            PositionMetrics object
        """
        # Time calculations
        time_in_position = (datetime.now() - position.entry_time).total_seconds() / 60
        
        hours_status = self.context.get_market_hours_status()
        minutes_to_close = hours_status["minutes_to_close"]
        
        # P&L calculation (simplified - would use real option prices in production)
        entry_credit = position.trade_setup.total_credit
        
        # Estimate current value based on time decay and price movement
        price_move = current_price - position.entry_price
        time_decay_factor = min(time_in_position / (6.5 * 60), 0.8)  # Max 80% decay
        
        # Calculate distances to strikes
        put_distance = None
        call_distance = None
        
        # Very simplified P&L estimation
        if position.trade_setup.strategy == StrategyType.IRON_CONDOR:
            # Check if either side is threatened
            put_distance = current_price - position.trade_setup.strikes.short_put_strike
            call_distance = position.trade_setup.strikes.short_call_strike - current_price
            
            if put_distance < 0 or call_distance < 0:
                # One side is ITM - losing position
                unrealized_pnl = -abs(price_move) * 0.5
            else:
                # Both sides OTM - winning from decay
                unrealized_pnl = entry_credit * time_decay_factor
        else:
            # Credit spread
            if position.trade_setup.strategy == StrategyType.PUT_SPREAD:
                put_distance = current_price - position.trade_setup.strikes.short_put_strike
                
                if put_distance < 0:
                    unrealized_pnl = -abs(price_move) * 0.5
                else:
                    unrealized_pnl = entry_credit * time_decay_factor
            else:  # Call spread
                call_distance = position.trade_setup.strikes.short_call_strike - current_price
                
                if call_distance < 0:
                    unrealized_pnl = -abs(price_move) * 0.5
                else:
                    unrealized_pnl = entry_credit * time_decay_factor
        
        # Calculate percentage
        max_profit = entry_credit
        unrealized_pnl_pct = (unrealized_pnl / max_profit * 100) if max_profit > 0 else 0
        
        # Check profit targets
        profit_target_1_hit = unrealized_pnl >= max_profit * self.profit_targets[0]
        profit_target_2_hit = unrealized_pnl >= max_profit * self.profit_targets[1]
        
        # Check stop loss
        stop_loss_triggered = unrealized_pnl <= -max_profit * self.stop_loss_multiplier
        
        # ==========================================================================
        # FIXED THREAT DETECTION - Much less sensitive for 0DTE
        # ==========================================================================
        adjustment_needed = False
        
        # Dynamic threat distance based on time remaining and volatility
        vix = self.context.current_data.vix if self.context.current_data else 16
        
        # Base threat distance - much larger than before
        if minutes_to_close > 180:  # More than 3 hours
            # Early in day - only threatened if VERY close
            if vix < 18:
                threat_distance = 40  # Was effectively using 60, now 40
            elif vix < 25:
                threat_distance = 50
            else:
                threat_distance = 75
        elif minutes_to_close > 60:  # 1-3 hours left
            # Mid-day - moderate threat distance
            if vix < 18:
                threat_distance = 30  # More reasonable
            elif vix < 25:
                threat_distance = 40
            else:
                threat_distance = 60
        else:  # Last hour
            # Close to expiry - tighter but still reasonable
            if vix < 18:
                threat_distance = 20  # Only if really close
            elif vix < 25:
                threat_distance = 25
            else:
                threat_distance = 35
        
        # Additional check: Only consider threatened if strike is actually at risk
        # Don't trigger on normal market movement
        if put_distance is not None:
            # Put is threatened if:
            # 1. Distance is positive but less than threat distance AND
            # 2. Market is moving toward the strike (bearish for puts)
            if 0 < put_distance < threat_distance:
                # Check if price is actually moving toward put
                price_momentum = price_move  # Negative = moving down toward put
                if price_momentum < -10:  # Only if clearly moving that way
                    adjustment_needed = True
                    self.logger.warning(f"Put strike threatened: {put_distance:.0f} points away, "
                                    f"threshold: {threat_distance:.0f}")
        
        if call_distance is not None:
            # Call is threatened if:
            # 1. Distance is positive but less than threat distance AND
            # 2. Market is moving toward the strike (bullish for calls)
            if 0 < call_distance < threat_distance:
                # Check if price is actually moving toward call
                price_momentum = price_move  # Positive = moving up toward call
                if price_momentum > 10:  # Only if clearly moving that way
                    adjustment_needed = True
                    self.logger.warning(f"Call strike threatened: {call_distance:.0f} points away, "
                                    f"threshold: {threat_distance:.0f}")
        
        # ==========================================================================
        # OVERRIDE: Don't adjust if position is already profitable
        # ==========================================================================
        if adjustment_needed and unrealized_pnl_pct > 20:
            # Position is profitable, let it ride a bit more
            self.logger.info(f"Strike close but position profitable ({unrealized_pnl_pct:.1f}%), "
                            "holding position")
            adjustment_needed = False
        
        # Check time exit (unchanged)
        time_exit_needed = minutes_to_close <= self.time_exit_minutes
        
        return PositionMetrics(
            position_id=position.position_id,
            current_price=current_price,
            entry_price=position.entry_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            distance_to_short_put=put_distance,
            distance_to_short_call=call_distance,
            delta_risk=0,  # Would calculate from real Greeks
            time_in_position_minutes=time_in_position,
            minutes_to_close=minutes_to_close,
            profit_target_1_hit=profit_target_1_hit,
            profit_target_2_hit=profit_target_2_hit,
            stop_loss_triggered=stop_loss_triggered,
            adjustment_needed=adjustment_needed,
            time_exit_needed=time_exit_needed
        )
    
    def _determine_action(self, position: Position, metrics: PositionMetrics) -> Optional[ManagementAction]:
        """
        Determine what action to take on a position.
        
        Priority order:
        1. Stop loss (immediate)
        2. Time exit (immediate if needed)
        3. Profit target 2 (close all)
        4. Profit target 1 (close half)
        5. Adjustment (if threatened)
        6. Hold
        
        Args:
            position: Position to evaluate
            metrics: Current metrics
            
        Returns:
            ManagementAction or None
        """
        # Priority 1: Stop loss
        if metrics.stop_loss_triggered:
            self.stats["stop_losses_hit"] += 1
            return ManagementAction(
                position_id=position.position_id,
                action_type="close_all",
                reason=f"Stop loss triggered: ${metrics.unrealized_pnl:.2f}",
                urgency="immediate",
                details={"pnl": metrics.unrealized_pnl}
            )
        
        # Priority 2: Time exit for 0DTE
        if metrics.time_exit_needed:
            self.stats["time_exits"] += 1
            return ManagementAction(
                position_id=position.position_id,
                action_type="close_all",
                reason=f"Time exit: {metrics.minutes_to_close:.0f} min to close",
                urgency="immediate",
                details={"minutes_to_close": metrics.minutes_to_close}
            )
        
        # Priority 3: Profit target 2 (full close)
        if metrics.profit_target_2_hit:
            self.stats["profit_targets_hit"] += 1
            return ManagementAction(
                position_id=position.position_id,
                action_type="close_all",
                reason=f"Profit target 2 hit: {metrics.unrealized_pnl_pct:.1f}%",
                urgency="soon",
                details={"profit_pct": metrics.unrealized_pnl_pct}
            )
        
        # Priority 4: Profit target 1 (partial close)
        if metrics.profit_target_1_hit and position.state == PositionState.ACTIVE:
            # Only if we haven't already closed half
            if "half_closed" not in position.adjustments:
                return ManagementAction(
                    position_id=position.position_id,
                    action_type="close_half",
                    reason=f"Profit target 1 hit: {metrics.unrealized_pnl_pct:.1f}%",
                    urgency="normal",
                    details={"profit_pct": metrics.unrealized_pnl_pct}
                )
        
        # Priority 5: Adjustment needed
        if metrics.adjustment_needed:
            # Check if we've adjusted recently
            if position.adjustments:
                last_adjustment = position.adjustments[-1]["time"]
                if (datetime.now() - last_adjustment).total_seconds() < 1800:
                    return None  # Don't adjust again within 30 minutes
            
            self.stats["adjustments_made"] += 1
            return ManagementAction(
                position_id=position.position_id,
                action_type="adjust",
                reason="Strike threatened - adjustment needed",
                urgency="soon",
                details={
                    "put_distance": metrics.distance_to_short_put,
                    "call_distance": metrics.distance_to_short_call
                }
            )
        
        # Default: Hold
        return ManagementAction(
            position_id=position.position_id,
            action_type="hold",
            reason="Within normal parameters",
            urgency="normal",
            details={}
        )
    
    # ========================================================================
    # ACTION EXECUTION
    # ========================================================================
    
    async def execute_actions(self, actions: List[ManagementAction]) -> None:
        """
        Execute management actions.
        
        Args:
            actions: List of actions to execute
        """
        # Sort by urgency
        urgent = [a for a in actions if a.urgency == "immediate"]
        soon = [a for a in actions if a.urgency == "soon"]
        normal = [a for a in actions if a.urgency == "normal"]
        
        # Execute in priority order
        for action in urgent + soon + normal:
            await self._execute_single_action(action)
    
    async def _execute_single_action(self, action: ManagementAction) -> None:
        """Execute a single management action."""
        position = self.active_positions.get(action.position_id)
        
        if not position:
            self.logger.warning(f"Position {action.position_id} not found")
            return
        
        self.logger.info(f"Executing {action.action_type} for {action.position_id}")
        
        try:
            if action.action_type == "close_all":
                await self._close_position(position, action.reason)
                
            elif action.action_type == "close_half":
                await self._close_half_position(position, action.reason)
                
            elif action.action_type == "adjust":
                await self._adjust_position(position, action.details)
                
            elif action.action_type == "hold":
                # No action needed
                pass
                
        except Exception as e:
            self.logger.error(f"Failed to execute action: {e}")
    
    async def _close_position(self, position: Position, reason: str) -> None:
        """Close entire position."""
        result = await self.executor.close_position(position, reason)
        
        if result.success:
            # Update position state
            position.state = PositionState.CLOSED
            position.exit_time = datetime.now()
            position.exit_reason = reason
            position.realized_pnl = result.total_fill_price
            
            # Update stats
            if result.total_fill_price > 0:
                self.stats["total_profit"] += result.total_fill_price
            else:
                self.stats["total_loss"] += abs(result.total_fill_price)
            
            # Remove from active monitoring
            self.remove_position(position.position_id)
            
            self.logger.info(f"✅ Position closed: {reason}")
            self.logger.info(f"  P&L: ${result.total_fill_price:.2f}")
    
    async def _close_half_position(self, position: Position, reason: str) -> None:
        """Close half of position."""
        # Temporarily reduce position size
        original_contracts = position.trade_setup.contracts
        position.trade_setup.contracts = original_contracts // 2
        
        if position.trade_setup.contracts > 0:
            result = await self.executor.close_position(position, reason)
            
            if result.success:
                # Restore remaining size
                position.trade_setup.contracts = original_contracts - position.trade_setup.contracts
                
                # Mark as partially closed
                position.state = PositionState.PARTIAL_CLOSE
                position.partial_closes.append({
                    "time": datetime.now(),
                    "contracts_closed": original_contracts // 2,
                    "reason": reason,
                    "pnl": result.total_fill_price
                })
                
                # Mark that we've taken partial profits
                position.adjustments.append({"type": "half_closed", "time": datetime.now()})
                
                self.logger.info(f"✅ Closed half position: {reason}")
    
    async def _adjust_position(self, position: Position, details: Dict) -> None:
        """Adjust threatened position."""
        # For now, just close it (rolling would be implemented here)
        self.logger.info("Position adjustment needed - closing for safety")
        await self._close_position(position, "Adjusted (closed) due to threat")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_position_summary(self) -> Dict:
        """
        Get summary of all positions.
        
        Returns:
            Dictionary with position summaries
        """
        summary = {
            "active_positions": len(self.active_positions),
            "total_unrealized_pnl": 0,
            "positions": []
        }
        
        for position_id, metrics in self.position_metrics.items():
            position = self.active_positions[position_id]
            
            summary["total_unrealized_pnl"] += metrics.unrealized_pnl
            
            summary["positions"].append({
                "id": position_id,
                "strategy": position.trade_setup.strategy.value,
                "entry_price": position.entry_price,
                "current_price": metrics.current_price,
                "unrealized_pnl": metrics.unrealized_pnl,
                "pnl_pct": metrics.unrealized_pnl_pct,
                "time_in_position": metrics.time_in_position_minutes,
                "put_distance": metrics.distance_to_short_put,
                "call_distance": metrics.distance_to_short_call
            })
        
        return summary
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        total_trades = self.stats["profit_targets_hit"] + self.stats["stop_losses_hit"]
        
        if total_trades > 0:
            win_rate = self.stats["profit_targets_hit"] / total_trades * 100
        else:
            win_rate = 0
        
        return {
            "positions_monitored": self.stats["positions_monitored"],
            "profit_targets_hit": self.stats["profit_targets_hit"],
            "stop_losses_hit": self.stats["stop_losses_hit"],
            "time_exits": self.stats["time_exits"],
            "adjustments_made": self.stats["adjustments_made"],
            "total_profit": self.stats["total_profit"],
            "total_loss": self.stats["total_loss"],
            "net_pnl": self.stats["total_profit"] - self.stats["total_loss"],
            "win_rate": win_rate
        }