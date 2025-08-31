"""
Shared Data Models
==================
Common data structures used throughout the trading system.
These models ensure type safety and consistent data handling.

Author: Trading Systems
Version: 2.0
Date: August 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


# ============================================================================
# ENUMERATIONS
# ============================================================================

class StrategyType(Enum):
    """Available trading strategies."""
    IRON_CONDOR = "iron_condor"
    PUT_SPREAD = "put_spread"
    CALL_SPREAD = "call_spread"
    IRON_BUTTERFLY = "iron_butterfly"
    SKIP = "skip"


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    SQUEEZE = "squeeze"
    VOLATILE = "volatile"


class TrendState(Enum):
    """Market trend classifications."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class PositionState(Enum):
    """Position lifecycle states."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIAL_CLOSE = "partial_close"
    ADJUSTING = "adjusting"
    CLOSING = "closing"
    CLOSED = "closed"
    EXPIRED = "expired"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# ============================================================================
# MARKET DATA MODELS
# ============================================================================

@dataclass
class Greeks:
    """Option Greeks container."""
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float  # Implied volatility
    

@dataclass
class OptionContract:
    """Single option contract details."""
    strike: float
    right: str  # 'P' or 'C'
    expiry: str
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    greeks: Optional[Greeks] = None
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        return (self.spread / self.mid * 100) if self.mid > 0 else 0


@dataclass
class MarketData:
    """Current market data snapshot."""
    timestamp: datetime
    underlying_price: float
    underlying_volume: int
    vix: float
    
    # Price levels
    open_price: float
    high_price: float
    low_price: float
    previous_close: float
    
    # Calculated metrics
    daily_change: float = field(init=False)
    daily_change_pct: float = field(init=False)
    daily_range: float = field(init=False)
    gap_pct: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.daily_change = self.underlying_price - self.previous_close
        self.daily_change_pct = (self.daily_change / self.previous_close * 100) if self.previous_close > 0 else 0
        self.daily_range = self.high_price - self.low_price
        self.gap_pct = ((self.open_price - self.previous_close) / self.previous_close * 100) if self.previous_close > 0 else 0


# ============================================================================
# TRADING SETUP MODELS
# ============================================================================

@dataclass
class StrikeSelection:
    """Selected strikes for a strategy."""
    # Put side
    short_put_strike: Optional[float] = None
    long_put_strike: Optional[float] = None
    
    # Call side
    short_call_strike: Optional[float] = None
    long_call_strike: Optional[float] = None
    
    # Metrics
    put_credit: float = 0
    call_credit: float = 0
    total_credit: float = 0
    max_risk: float = 0
    probability_profit: float = 0
    
    def validate(self) -> bool:
        """Validate strike selection logic."""
        # For iron condor, need all four strikes
        if all([self.short_put_strike, self.long_put_strike, 
                self.short_call_strike, self.long_call_strike]):
            return (self.long_put_strike < self.short_put_strike < 
                   self.short_call_strike < self.long_call_strike)
        
        # For put spread
        if self.short_put_strike and self.long_put_strike:
            return self.long_put_strike < self.short_put_strike
        
        # For call spread
        if self.short_call_strike and self.long_call_strike:
            return self.short_call_strike < self.long_call_strike
        
        return False


@dataclass
class TradeSetup:
    """Complete trade setup ready for execution."""
    strategy: StrategyType
    strikes: StrikeSelection
    contracts: int
    
    # Risk metrics
    credit_per_contract: float
    max_risk_per_contract: float
    total_credit: float = field(init=False)
    total_max_risk: float = field(init=False)
    
    # Decision factors
    confidence: float
    risk_level: RiskLevel
    rationale: str
    
    # Execution parameters
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    market_price_at_creation: Optional[float] = None
    
    def __post_init__(self):
        """Calculate totals."""
        self.total_credit = self.credit_per_contract * self.contracts * 20  # NQ multiplier
        self.total_max_risk = self.max_risk_per_contract * self.contracts * 20


# ============================================================================
# POSITION MODELS
# ============================================================================

@dataclass
class Position:
    """Active position tracking."""
    position_id: str
    trade_setup: TradeSetup
    
    # Entry details
    entry_time: datetime
    entry_price: float
    fill_prices: Dict[str, float]  # Strike -> fill price
    
    # Current state
    state: PositionState
    current_value: float = 0
    unrealized_pnl: float = 0
    
    # Management
    adjustments: List[Dict[str, Any]] = field(default_factory=list)
    partial_closes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Exit details (when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    
    def time_in_position(self) -> float:
        """Calculate time in position (minutes)."""
        end_time = self.exit_time or datetime.now()
        return (end_time - self.entry_time).total_seconds() / 60


# ============================================================================
# ANALYSIS MODELS
# ============================================================================

@dataclass
class MovementStats:
    """Market movement statistics."""
    atr_points: float
    atr_percent: float
    avg_up_move: float
    avg_down_move: float
    expected_move_1std: float
    trend_5day: float
    
    # Suggested distances based on ATR
    conservative_distance: float = field(init=False)
    balanced_distance: float = field(init=False)
    aggressive_distance: float = field(init=False)
    
    def __post_init__(self):
        """Calculate suggested distances."""
        self.conservative_distance = self.atr_points * 1.5
        self.balanced_distance = self.atr_points * 1.0
        self.aggressive_distance = self.atr_points * 0.7


@dataclass
class MarketAnalysis:
    """Complete market analysis results."""
    # Trend
    trend_state: TrendState
    trend_strength: float  # -100 to +100
    trend_confidence: float  # 0 to 100
    
    # Regime
    market_regime: MarketRegime
    regime_confidence: float
    
    # Key levels
    support: float
    resistance: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    
    # Momentum
    rsi: float
    momentum_score: float  # -100 to +100
    
    # Movement statistics
    movement_stats: MovementStats
    
    # Recommendation
    recommended_strategy: StrategyType
    confidence_level: float
    risk_assessment: RiskLevel
    rationale: str


# ============================================================================
# PERFORMANCE MODELS
# ============================================================================

@dataclass
class SessionStats:
    """Trading session statistics."""
    session_start: datetime = field(default_factory=datetime.now)
    
    # Trade counts
    trades_executed: int = 0
    trades_skipped: int = 0
    iron_condors: int = 0
    put_spreads: int = 0
    call_spreads: int = 0
    
    # Financial metrics
    total_premium_collected: float = 0
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    commissions_paid: float = 0
    
    # Risk metrics
    max_drawdown: float = 0
    consecutive_losses: int = 0
    win_rate: float = 0
    
    # Position management
    positions_adjusted: int = 0
    profit_targets_hit: int = 0
    stop_losses_hit: int = 0
    time_exits: int = 0
    
    def update_win_rate(self, wins: int, total: int) -> None:
        """Update win rate calculation."""
        self.win_rate = (wins / total * 100) if total > 0 else 0