"""
Market Context Provider
=======================
The single source of truth for all market data and derived parameters.
This module consolidates all market analysis and parameter calculation
into one authoritative location.

Author: Trading Systems
Version: 2.0
Date: August 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytz
from dataclasses import dataclass, field

from config import CONFIG, get_config
from models import (
    MarketData, MarketAnalysis, MarketRegime, TrendState,
    MovementStats, RiskLevel, StrategyType
)
from exceptions import (
    MarketDataError, NoMarketDataError, StaleDataError,
    InvalidPriceError
)


class MarketContext:
    """
    Central hub for all market data and derived parameters.
    
    This class is the single source of truth for:
    - Current market prices and data
    - Movement statistics and volatility
    - Trading parameters based on conditions
    - Time and session management
    
    Everything else in the system asks this class for parameters
    rather than calculating their own.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the market context provider.
        
        Args:
            logger: Optional logger instance for debugging
        """
        self.logger = logger or self._setup_logger()
        self.config = get_config()
        
        # Timezone setup for Spain-based trading
        self.madrid_tz = pytz.timezone("Europe/Madrid")
        self.et_tz = pytz.timezone("US/Eastern")
        
        # Market data storage
        self.current_data: Optional[MarketData] = None
        self.historical_data: Optional[pd.DataFrame] = None
        self.option_chain: List = []
        
        # Analysis results cache
        self.market_analysis: Optional[MarketAnalysis] = None
        self.movement_stats: Optional[MovementStats] = None
        self.last_update: Optional[datetime] = None
        
        # Parameter cache (calculated based on conditions)
        self._parameter_cache: Dict = {}
        self._cache_expiry: Optional[datetime] = None
        
        self.logger.info("Market Context Provider initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Create a logger instance if none provided."""
        logger = logging.getLogger("MarketContext")
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
    # DATA UPDATES
    # ========================================================================
    
    def update_market_data(
        self,
        price: float,
        volume: int,
        vix: float,
        high: float,
        low: float,
        open_price: float,
        previous_close: float
    ) -> None:
        """
        Update current market data snapshot.
        
        This is called by the orchestrator whenever new data arrives.
        All derived parameters are automatically recalculated.
        
        Args:
            price: Current underlying price
            volume: Current volume
            vix: Current VIX level
            high: Day's high price
            low: Day's low price
            open_price: Day's open price
            previous_close: Previous day's close
            
        Raises:
            InvalidPriceError: If price data is invalid
        """
        # Validate inputs
        if price <= 0 or vix <= 0:
            raise InvalidPriceError(f"Invalid price data: price={price}, vix={vix}")
        
        # Create new market data snapshot
        self.current_data = MarketData(
            timestamp=datetime.now(),
            underlying_price=price,
            underlying_volume=volume,
            vix=vix,
            open_price=open_price,
            high_price=high,
            low_price=low,
            previous_close=previous_close
        )
        
        # Mark update time
        self.last_update = datetime.now()
        
        # Invalidate parameter cache to force recalculation
        self._invalidate_cache()
        
        self.logger.info(f"Market data updated: NQ=${price:,.2f}, VIX={vix:.1f}")
    
    def update_historical_data(self, data: pd.DataFrame) -> None:
        """
        Update historical price data for analysis.
        
        Args:
            data: DataFrame with OHLCV data and VIX
        """
        self.historical_data = data
        
        # Calculate movement statistics immediately
        if self.current_data:
            self.movement_stats = self._calculate_movement_stats()
            self.logger.info("Historical data updated and movement stats calculated")
    
    def update_option_chain(self, options: List) -> None:
        """
        Update available option contracts.
        
        Args:
            options: List of option contracts with Greeks
        """
        self.option_chain = options
        self.logger.info(f"Option chain updated: {len(options)} contracts")
    
    # ========================================================================
    # MARKET ANALYSIS
    # ========================================================================
    
    def analyze_market(self) -> MarketAnalysis:
        """
        Perform comprehensive market analysis.
        
        This consolidates all the analysis logic from your various analyzers
        into one authoritative method.
        
        Returns:
            Complete market analysis with trend, regime, and recommendations
            
        Raises:
            NoMarketDataError: If insufficient data for analysis
        """
        if not self.current_data or self.historical_data is None or self.historical_data.empty:
            raise NoMarketDataError("Insufficient data for market analysis")
        
        # Check data freshness
        if self.last_update:
            age = (datetime.now() - self.last_update).total_seconds()
            if age > 60:  # Data older than 1 minute
                raise StaleDataError(age)
        
        # Calculate all components
        trend = self._analyze_trend()
        regime = self._determine_regime(trend)
        momentum = self._analyze_momentum()
        structure = self._analyze_structure()
        
        # Get movement stats if not already calculated
        if not self.movement_stats:
            self.movement_stats = self._calculate_movement_stats()
        
        # Generate recommendation based on analysis
        recommendation = self._generate_recommendation(
            trend, regime, momentum, structure
        )
        
        # Create complete analysis
        self.market_analysis = MarketAnalysis(
            trend_state=trend["state"],
            trend_strength=trend["strength"],
            trend_confidence=trend["confidence"],
            market_regime=regime,
            regime_confidence=self._calculate_regime_confidence(regime),
            support=structure["support"],
            resistance=structure["resistance"],
            distance_to_support_pct=structure["dist_to_support"],
            distance_to_resistance_pct=structure["dist_to_resistance"],
            rsi=momentum["rsi"],
            momentum_score=momentum["score"],
            movement_stats=self.movement_stats,
            recommended_strategy=recommendation["strategy"],
            confidence_level=recommendation["confidence"],
            risk_assessment=recommendation["risk_level"],
            rationale=recommendation["rationale"]
        )
        
        return self.market_analysis
    
    def _analyze_trend(self) -> Dict:
        """
        Analyze market trend using multiple timeframes.
        
        This is your existing trend analysis logic, consolidated here.
        """
        if self.historical_data is None or self.historical_data.empty:
            raise NoMarketDataError("No historical data for trend analysis")
        
        df = self.historical_data
        
        current_price = self.current_data.underlying_price
        
        # Calculate moving averages
        df['ma9'] = df['close'].rolling(9).mean()
        df['ma21'] = df['close'].rolling(21).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        
        ma9 = df['ma9'].iloc[-1]
        ma21 = df['ma21'].iloc[-1]
        ma50 = df['ma50'].iloc[-1] if len(df) >= 50 else ma21
        
        # Calculate trend strength (-100 to +100)
        strength = 0
        confidence = 50
        
        # MA alignment
        if current_price > ma9 > ma21 > ma50:
            strength += 30
            confidence += 20
        elif current_price < ma9 < ma21 < ma50:
            strength -= 30
            confidence += 20
        
        # Price distance from MAs
        dist_from_ma21 = ((current_price - ma21) / ma21) * 100
        strength += np.clip(dist_from_ma21 * 10, -20, 20)
        
        # MA slopes
        ma9_slope = (df['ma9'].iloc[-1] - df['ma9'].iloc[-5]) / df['ma9'].iloc[-5] * 100
        strength += np.clip(ma9_slope * 5, -15, 15)
        
        # Higher highs/lower lows
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        
        if (recent_highs.iloc[-1] > recent_highs.iloc[-10] and 
            recent_lows.iloc[-1] > recent_lows.iloc[-10]):
            strength += 25
            confidence += 15
        elif (recent_highs.iloc[-1] < recent_highs.iloc[-10] and 
              recent_lows.iloc[-1] < recent_lows.iloc[-10]):
            strength -= 25
            confidence += 15
        
        # Determine trend state
        if strength > 60:
            state = TrendState.STRONG_UPTREND
        elif strength > 20:
            state = TrendState.UPTREND
        elif strength < -60:
            state = TrendState.STRONG_DOWNTREND
        elif strength < -20:
            state = TrendState.DOWNTREND
        else:
            state = TrendState.NEUTRAL
        
        return {
            "state": state,
            "strength": np.clip(strength, -100, 100),
            "confidence": np.clip(confidence, 0, 100)
        }
    
    def _determine_regime(self, trend: Dict) -> MarketRegime:
        """
        Determine current market regime.
        
        This uses your existing regime detection logic.
        """
        vix = self.current_data.vix
        atr_pct = self.movement_stats.atr_percent if self.movement_stats else 1.5
        trend_strength = trend["strength"]
        
        # Clear trend with manageable volatility
        if abs(trend_strength) > 40 and vix < 25:
            return MarketRegime.TRENDING
        
        # High volatility environment
        elif vix > 25 or atr_pct > 2.5:
            return MarketRegime.VOLATILE
        
        # Low volatility squeeze
        elif vix < 14 and atr_pct < 1.0:
            return MarketRegime.SQUEEZE
        
        # Near key levels
        elif self._near_key_levels():
            return MarketRegime.BREAKOUT
        
        # Default: ranging market
        else:
            return MarketRegime.RANGING
    
    def _analyze_momentum(self) -> Dict:
        """Calculate momentum indicators."""
        
        if self.historical_data is None or self.historical_data.empty:
            return {"rsi": 50, "score": 0, "roc_5": 0}  # neutral defaults
        
        df = self.historical_data
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Momentum score
        score = 0
        if current_rsi > 70:
            score += 30
        elif current_rsi > 50:
            score += 15
        elif current_rsi < 30:
            score -= 30
        elif current_rsi < 50:
            score -= 15
        
        # Rate of change
        roc_5 = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / 
                 df['close'].iloc[-6] * 100)
        score += np.clip(roc_5 * 5, -30, 30)
        
        return {
            "rsi": current_rsi,
            "score": np.clip(score, -100, 100),
            "roc_5": roc_5
        }
    
    def _analyze_structure(self) -> Dict:
        """Identify support and resistance levels."""
        
        if self.historical_data is None or self.historical_data.empty:
            current_price = self.current_data.underlying_price
            return {
                "support": current_price * 0.99,
                "resistance": current_price * 1.01,
                "dist_to_support": 1.0,
                "dist_to_resistance": 1.0
            }
        df = self.historical_data
        
        current_price = self.current_data.underlying_price
        
        # Find recent highs and lows
        recent = df.tail(50)
        highs = recent['high'].nlargest(5).mean()
        lows = recent['low'].nsmallest(5).mean()
        
        # Calculate distances
        dist_to_resistance = ((highs - current_price) / current_price * 100)
        dist_to_support = ((current_price - lows) / current_price * 100)
        
        return {
            "support": lows,
            "resistance": highs,
            "dist_to_support": dist_to_support,
            "dist_to_resistance": dist_to_resistance
        }
    
    def _calculate_movement_stats(self) -> MovementStats:
        """
        Calculate movement statistics from historical data.
        
        This is your existing movement calculation logic.
        """
        
        if self.historical_data is None or self.historical_data.empty:
            # Return default stats
            return MovementStats(
                atr_points=100,
                atr_percent=0.5,
                avg_up_move=50,
                avg_down_move=50,
                expected_move_1std=100,
                trend_5day=0
            )

        df = self.historical_data.tail(10)
        current_price = self.current_data.underlying_price
        
        # ATR calculation
        high_low = df['high'] - df['low']
        atr_points = high_low.mean()
        atr_percent = (high_low / df['open'] * 100).mean()
        
        # Daily moves
        daily_moves = df['close'].diff()
        up_days = daily_moves[daily_moves > 0]
        down_days = daily_moves[daily_moves < 0]
        
        avg_up = up_days.mean() if len(up_days) > 0 else 0
        avg_down = abs(down_days.mean()) if len(down_days) > 0 else 0
        
        # Expected move (1 std dev)
        daily_returns = df['close'].pct_change().dropna()
        daily_std = daily_returns.std()
        expected_move = current_price * daily_std
        
        # 5-day trend
        trend_5d = ((df['close'].iloc[-1] - df['close'].iloc[-5]) / 
                    df['close'].iloc[-5] * 100)
        
        return MovementStats(
            atr_points=atr_points,
            atr_percent=atr_percent,
            avg_up_move=avg_up,
            avg_down_move=avg_down,
            expected_move_1std=expected_move,
            trend_5day=trend_5d
        )
    
    def _generate_recommendation(
        self, trend: Dict, regime: MarketRegime,
        momentum: Dict, structure: Dict
    ) -> Dict:
        """Generate trading recommendation based on analysis."""
        vix = self.current_data.vix
        confidence = 50
        
        # Determine strategy based on regime and trend
        if regime == MarketRegime.VOLATILE and vix > 30:
            strategy = StrategyType.SKIP
            rationale = f"VIX {vix:.1f} too high for safe trading"
            risk_level = RiskLevel.EXTREME
            
        elif regime == MarketRegime.TRENDING:
            if trend["strength"] > 40:
                strategy = StrategyType.PUT_SPREAD
                rationale = f"Strong uptrend ({trend['strength']:.0f}) - bullish put spread"
            elif trend["strength"] < -40:
                strategy = StrategyType.CALL_SPREAD
                rationale = f"Strong downtrend ({trend['strength']:.0f}) - bearish call spread"
            else:
                strategy = StrategyType.IRON_CONDOR
                rationale = "Moderate trend - neutral iron condor"
            risk_level = RiskLevel.MEDIUM
            confidence = 70
            
        elif regime == MarketRegime.RANGING:
            strategy = StrategyType.IRON_CONDOR
            rationale = f"Range-bound market with VIX {vix:.1f} - ideal for iron condor"
            risk_level = RiskLevel.LOW
            confidence = 75
            
        else:
            strategy = StrategyType.IRON_CONDOR
            rationale = "Default to iron condor in uncertain conditions"
            risk_level = RiskLevel.MEDIUM
            confidence = 60
        
        # Adjust confidence based on momentum alignment
        if momentum["score"] * trend["strength"] > 0:
            confidence += 10  # Momentum confirms trend
        
        return {
            "strategy": strategy,
            "confidence": np.clip(confidence, 0, 100),
            "risk_level": risk_level,
            "rationale": rationale
        }
    
    # ========================================================================
    # PARAMETER CALCULATIONS
    # ========================================================================
    def get_trading_parameters(self) -> Dict:
        """
        ADAPTIVE PARAMETER SYSTEM for all market conditions.
        Professional 0DTE approach that adjusts to any market.
        """
        if self._is_cache_valid():
            return self._parameter_cache
        
        if not self.current_data:
            raise NoMarketDataError("No market data available for parameters")
        
        vix = self.current_data.vix
        current_price = self.current_data.underlying_price
        
        # Determine market condition
        market_condition = self._classify_market_condition(vix)
        
        # Get base parameters for the condition
        params = self._get_condition_parameters(market_condition, vix)
        
        # Apply additional adjustments
        params = self._apply_time_adjustments(params)
        params = self._apply_trend_adjustments(params)
        params = self._apply_structure_adjustments(params)
        
        # Final safety checks
        params["can_trade"] = self._check_trading_window() and params.get("can_trade", True)
        
        # Log the strategy
        self.logger.info(f"📊 MARKET CONDITION: {market_condition}")
        self.logger.info(f"  VIX: {vix:.1f}")
        self.logger.info(f"  Strategy: Strike={params['min_strike_distance']}pts, "
                        f"Wing={params['wing_width']}pts")
        self.logger.info(f"  Min Credits: IC=${params['min_credit_ic']:.2f}, "
                        f"Put=${params['min_credit_put']:.2f}, "
                        f"Call=${params['min_credit_call']:.2f}")
        
        # Cache and return
        self._parameter_cache = params
        self._cache_expiry = datetime.now() + timedelta(seconds=30)
        
        return params

    
    def _calculate_wing_width(self, vix: float) -> int:
        """
        Calculate optimal wing width based on VIX.
        
        This is THE authoritative wing width calculation.
        """
        if vix < self.config["thresholds"].vix_low:
            # Low VIX: need wider wings for credit
            return self.config["strategy"].wing_widths["low_vix"]
        elif vix < self.config["thresholds"].vix_normal:
            # Normal VIX: standard wings
            return self.config["strategy"].wing_widths["normal_vix"]
        elif vix < self.config["thresholds"].vix_high:
            # High VIX: can use standard wings
            return self.config["strategy"].wing_widths["high_vix"]
        else:
            # Extreme VIX: tighter wings for protection
            return self.config["strategy"].wing_widths["extreme_vix"]
    
    def _calculate_strike_distance(self, vix: float) -> float:
        """Calculate strike distance as percentage of price."""
        # Base calculation using ATR if available
        if self.movement_stats:
            atr_pct = self.movement_stats.atr_percent
            
            # VIX-adjusted multiplier
            if vix < 16:
                multiplier = 0.8  # Closer strikes in low vol
            elif vix < 25:
                multiplier = 1.0  # Standard distance
            else:
                multiplier = 1.3  # Wider in high vol
            
            return (atr_pct * multiplier) / 100  # Convert to decimal
        
        # Fallback to fixed percentages
        if vix < 16:
            return 0.004  # 0.4%
        elif vix < 25:
            return 0.005  # 0.5%
        else:
            return 0.007  # 0.7%
    
    def _calculate_min_strike_distance(self) -> float:
        """Calculate minimum safe strike distance in points."""
        if self.movement_stats:
            # Based on ATR
            return max(100, self.movement_stats.atr_points * 0.4)
        return 100  # Default minimum
    
    def _calculate_min_credit(self, vix: float, strategy: str) -> float:
        """Calculate minimum credit requirement."""
        base_credit = self.config["strategy"].min_credit_base[strategy]
        
        # VIX-based adjustment
        if vix < 16:
            # Low VIX: drastically reduce requirements
            if strategy == "iron_condor":
                return 2.0 + (vix * 0.25)  # ~$6 at VIX 16
            else:
                return 1.0 + (vix * 0.3)   # ~$5 at VIX 16
        elif vix < 25:
            # Normal VIX
            return base_credit
        else:
            # High VIX: increase requirements
            return base_credit + ((vix - 25) * 0.5)
    
    def _calculate_stop_loss_multiplier(self, vix: float) -> float:
        """Calculate stop loss multiplier based on volatility."""
        if vix < 16:
            return self.config["strategy"].stop_loss_multiplier["low_vix"]
        elif vix < 25:
            return self.config["strategy"].stop_loss_multiplier["normal_vix"]
        else:
            return self.config["strategy"].stop_loss_multiplier["high_vix"]
    
    def _calculate_profit_targets(self, vix: float) -> List[float]:
        """Calculate profit targets based on conditions."""
        base_targets = self.config["strategy"].profit_targets.copy()
        
        # Adjust for VIX
        if vix < 15:
            # Take profits faster in low vol
            return [t * 0.8 for t in base_targets]
        elif vix > 25:
            # Let winners run in high vol
            return [t * 1.2 for t in base_targets]
        
        return base_targets
    
    def _calculate_max_contracts(self) -> int:
        """Calculate maximum contracts based on conditions."""
        base_max = self.config["account"].max_contracts_per_trade
        
        # Reduce in high volatility
        if self.current_data.vix > 25:
            return max(1, base_max - 1)
        
        # Reduce late in day
        if self._get_minutes_to_close() < 120:
            return max(1, base_max - 1)
        
        return base_max
    
    def _calculate_size_multiplier(self) -> float:
        """Calculate position size multiplier."""
        multiplier = 1.0
        
        # Time of day adjustment
        if self._get_minutes_to_close() < 180:
            multiplier *= 0.8
        
        # Volatility adjustment
        if self.current_data.vix > 20:
            multiplier *= 0.9
        
        # Day of week adjustment
        day_weight = self.config["hours"].day_weights.get(
            datetime.now().weekday(), 1.0
        )
        multiplier *= day_weight
        
        return multiplier
    
    def _calculate_min_probability(self, vix: float, strategy_type: str) -> float:
        """Calculate minimum probability requirement."""
        if strategy_type == "iron_condor":
            if vix < 16:
                return 0.50  # 50% in low vol
            elif vix < 25:
                return 0.60  # 60% normal
            else:
                return 0.70  # 70% in high vol
        else:  # Credit spreads
            if vix < 16:
                return 0.60
            elif vix < 25:
                return 0.65
            else:
                return 0.75
    
    def _calculate_max_delta(self, vix: float) -> float:
        """Calculate maximum acceptable delta."""
        if vix < 16:
            return 0.35  # Can accept higher delta in low vol
        elif vix < 25:
            return 0.30  # Standard limit
        else:
            return 0.25  # Conservative in high vol
    
    def _calculate_target_delta(self, vix: float) -> float:
        """Calculate target delta for strike selection."""
        if vix < 16:
            return 0.20  # Target 20 delta in low vol
        elif vix < 25:
            return 0.25  # Standard target
        else:
            return 0.15  # Conservative target in high vol
    
    # ========================================================================
    # TIME MANAGEMENT
    # ========================================================================
    
    def get_market_hours_status(self) -> Dict:
        """
        Get comprehensive market hours status.
        
        Returns:
            Dictionary with market hours information
        """
        madrid_now = datetime.now(self.madrid_tz)
        et_now = madrid_now.astimezone(self.et_tz)
        
        # Market hours in ET
        market_open = et_now.replace(hour=9, minute=30, second=0)
        market_close = et_now.replace(hour=16, minute=0, second=0)
        
        # Calculate times
        is_open = market_open <= et_now <= market_close
        minutes_since_open = (et_now - market_open).total_seconds() / 60 if is_open else 0
        minutes_to_close = (market_close - et_now).total_seconds() / 60 if is_open else 0
        
        # Determine trading window quality
        window_quality = self._assess_trading_window(madrid_now, minutes_to_close)
        
        return {
            "market_open": is_open,
            "madrid_time": madrid_now.strftime("%H:%M"),
            "et_time": et_now.strftime("%H:%M"),
            "minutes_since_open": max(0, minutes_since_open),
            "minutes_to_close": max(0, minutes_to_close),
            "window_quality": window_quality,
            "can_trade": window_quality not in ["closed", "closing_only"],
            "day_of_week": et_now.weekday()
        }
    
    def _assess_trading_window(self, madrid_time: datetime, minutes_to_close: float) -> str:
        """Assess current trading window quality."""
        hour = madrid_time.hour
        minute = madrid_time.minute
        
        # Check against Spain windows
        for window in self.config["hours"].spain_windows:
            start_h, start_m = window["start"]
            end_h, end_m = window["end"]
            
            # Check if current time is in window
            current_minutes = hour * 60 + minute
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            
            if start_minutes <= current_minutes < end_minutes:
                # Additional check for closing time
                if minutes_to_close < 30:
                    return "closing_only"
                return window["quality"]
        
        return "closed"
    
    def _calculate_time_multiplier(self) -> float:
        """Calculate time-based parameter multiplier."""
        minutes_to_close = self._get_minutes_to_close()
        
        if minutes_to_close > 360:  # More than 6 hours
            return 1.0
        elif minutes_to_close > 240:  # 4-6 hours
            return 0.8
        elif minutes_to_close > 120:  # 2-4 hours
            return 0.6
        else:  # Less than 2 hours
            return 0.4
    
    def _get_minutes_to_close(self) -> float:
        """Get minutes remaining to market close."""
        status = self.get_market_hours_status()
        return status["minutes_to_close"]
    
    def _check_trading_window(self) -> bool:
        """Check if current time is suitable for trading."""
        status = self.get_market_hours_status()
        return status["can_trade"]
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _near_key_levels(self) -> bool:
        """Check if price is near key support/resistance."""
        
        if self.historical_data is None or self.historical_data.empty or not self.current_data:
            return False
        
        structure = self._analyze_structure()
        
        # Within 0.3% of key level
        return (structure["dist_to_support"] < 0.3 or 
                structure["dist_to_resistance"] < 0.3)
    
    def _calculate_regime_confidence(self, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification."""
        # Base confidence
        confidence = 60.0
        
        # Adjust based on data quality
        if self.movement_stats:
            confidence += 20
        
        # Adjust based on VIX clarity
        vix = self.current_data.vix
        if vix < 12 or vix > 30:
            confidence += 10  # Clear extreme
        
        return min(100, confidence)
    
    def _is_cache_valid(self) -> bool:
        """Check if parameter cache is still valid."""
        if not self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry
    
    def _invalidate_cache(self) -> None:
        """Invalidate parameter cache."""
        self._cache_expiry = None
        self._parameter_cache = {}
    
    def get_summary(self) -> Dict:
        """
        Get a summary of current market context.
        
        Returns:
            Dictionary with key market metrics
        """
        if not self.current_data:
            return {"status": "No data"}
        
        params = self.get_trading_parameters()
        hours = self.get_market_hours_status()
        
        return {
            "price": self.current_data.underlying_price,
            "vix": self.current_data.vix,
            "wing_width": params["wing_width"],
            "min_credit_ic": params["min_credit_ic"],
            "market_open": hours["market_open"],
            "can_trade": params["can_trade"],
            "regime": self.market_analysis.market_regime.value if self.market_analysis else "Unknown",
            "trend": self.market_analysis.trend_strength if self.market_analysis else 0
        }
        
    def _classify_market_condition(self, vix: float) -> str:
        """Classify current market condition based on multiple factors."""
        
        # Primary classification by VIX
        if vix < 12:
            return "ULTRA_LOW_VOL"
        elif vix < 15:
            return "VERY_LOW_VOL"
        elif vix <= 18:  # YOUR CURRENT MARKET
            return "LOW_VOL_RANGE"
        elif vix <= 22:
            return "NORMAL"
        elif vix <= 28:
            return "ELEVATED"
        elif vix <= 35:
            return "HIGH_VOL"
        else:
            return "EXTREME"

    def _get_condition_parameters(self, condition: str, vix: float) -> Dict:
        """Get parameters for specific market condition."""
        
        # Base parameters that apply to all conditions
        base_params = {
            "stop_loss_multiplier": 2.0,
            "profit_targets": [0.25, 0.50, 0.75],
            "time_multiplier": self._calculate_time_multiplier(),
            "can_trade": True
        }
        
        # Condition-specific parameters
        if condition == "ULTRA_LOW_VOL":
            # VIX < 12: Extremely tight markets, consider skipping
            return {
                **base_params,
                "wing_width": 25,
                "strike_distance_pct": 0.003,  # 0.3% = ~70 points
                "min_strike_distance": 50,
                "min_credit_ic": 2.50,  # Very low requirements
                "min_credit_put": 1.50,
                "min_credit_call": 1.00,
                "max_contracts": 1,  # Reduce size
                "position_size_multiplier": 0.5,
                "min_probability_ic": 0.40,
                "min_probability_spread": 0.50,
                "max_delta": 0.45,
                "target_delta": 0.30,
                "can_trade": vix > 10  # Skip if VIX under 10
            }
        
        elif condition == "VERY_LOW_VOL":
            # VIX 12-15: Need tighter strikes for premium
            return {
                **base_params,
                "wing_width": 25,
                "strike_distance_pct": 0.0035,  # ~80 points
                "min_strike_distance": 60,
                "min_credit_ic": 3.50,
                "min_credit_put": 2.00,
                "min_credit_call": 1.50,
                "max_contracts": 2,
                "position_size_multiplier": 0.7,
                "min_probability_ic": 0.45,
                "min_probability_spread": 0.55,
                "max_delta": 0.40,
                "target_delta": 0.25
            }
        
        elif condition == "LOW_VOL_RANGE":
            # VIX 15-18: YOUR CURRENT MARKET
            return {
                **base_params,
                "wing_width": 25,
                "strike_distance_pct": 0.0045,  # ~93 points
                "min_strike_distance": 100,
                "min_credit_ic": 4.00,  # Matches your $5.12
                "min_credit_put": 3.00,  # Matches your $3.50
                "min_credit_call": 1.50,  # Matches your $1.62
                "max_contracts": 2,
                "position_size_multiplier": 0.6,
                "min_probability_ic": 0.45,
                "min_probability_spread": 0.60,
                "max_delta": 0.35,
                "target_delta": 0.20
            }
        
        elif condition == "NORMAL":
            # VIX 18-22: Standard conditions
            return {
                **base_params,
                "wing_width": 50,
                "strike_distance_pct": 0.006,  # ~140 points
                "min_strike_distance": 150,
                "min_credit_ic": 8.00,
                "min_credit_put": 5.00,
                "min_credit_call": 4.00,
                "max_contracts": 3,
                "position_size_multiplier": 1.0,
                "min_probability_ic": 0.55,
                "min_probability_spread": 0.65,
                "max_delta": 0.35,
                "target_delta": 0.25
            }
        
        elif condition == "ELEVATED":
            # VIX 22-28: Good premium, normal risk
            return {
                **base_params,
                "wing_width": 50,
                "strike_distance_pct": 0.008,  # ~185 points
                "min_strike_distance": 200,
                "min_credit_ic": 12.00,
                "min_credit_put": 7.00,
                "min_credit_call": 6.00,
                "max_contracts": 3,
                "position_size_multiplier": 1.0,
                "min_probability_ic": 0.60,
                "min_probability_spread": 0.70,
                "max_delta": 0.30,
                "target_delta": 0.20,
                "profit_targets": [0.30, 0.60, 0.80]  # Let winners run
            }
        
        elif condition == "HIGH_VOL":
            # VIX 28-35: Great premium, manage risk
            return {
                **base_params,
                "wing_width": 75,
                "strike_distance_pct": 0.010,  # ~230 points
                "min_strike_distance": 250,
                "min_credit_ic": 18.00,
                "min_credit_put": 10.00,
                "min_credit_call": 8.00,
                "max_contracts": 2,  # Reduce size
                "position_size_multiplier": 0.7,
                "min_probability_ic": 0.65,
                "min_probability_spread": 0.75,
                "max_delta": 0.25,
                "target_delta": 0.15,
                "stop_loss_multiplier": 1.5,  # Tighter stops
                "profit_targets": [0.35, 0.65, 0.85]
            }
        
        else:  # EXTREME (VIX > 35)
            # Extreme volatility: Very selective
            return {
                **base_params,
                "wing_width": 100,
                "strike_distance_pct": 0.015,  # Very wide
                "min_strike_distance": 300,
                "min_credit_ic": 25.00,
                "min_credit_put": 15.00,
                "min_credit_call": 12.00,
                "max_contracts": 1,
                "position_size_multiplier": 0.3,
                "min_probability_ic": 0.75,
                "min_probability_spread": 0.80,
                "max_delta": 0.20,
                "target_delta": 0.10,
                "stop_loss_multiplier": 1.0,  # Very tight stops
                "can_trade": vix < 45  # Skip if too extreme
            }

    def _apply_time_adjustments(self, params: Dict) -> Dict:
        """Adjust parameters based on time of day - FIXED VERSION."""
        minutes_to_close = self._get_minutes_to_close()
        
        if minutes_to_close < 60:  # Last hour - DON'T TRADE
            params["can_trade"] = False
            self.logger.warning("🛑 Last hour - trading disabled")
            return params
            
        elif minutes_to_close < 90:  # Last 90 minutes
            # DO NOT MODIFY STRIKE DISTANCE!
            # params["min_strike_distance"] *= 0.8  # DELETE THIS LINE IF IT EXISTS
            
            # Instead, ENFORCE minimum safe distance
            original_distance = params["min_strike_distance"]
            params["min_strike_distance"] = max(original_distance, 100)  # Never below 100
            
            # Moderate credit adjustments with floors
            params["min_credit_ic"] = max(params["min_credit_ic"] * 0.8, 3.50)
            params["min_credit_put"] = max(params["min_credit_put"] * 0.8, 2.00)
            params["min_credit_call"] = max(params["min_credit_call"] * 0.8, 1.00)
            params["max_contracts"] = 1
            
            self.logger.info(f"⏰ Late day adjustments (kept strike at {params['min_strike_distance']})")
            
        elif minutes_to_close < 180:  # Last 3 hours
            # Minor adjustments only
            params["position_size_multiplier"] *= 0.9
            # DO NOT reduce strike distance!
            
        return params

    def _apply_trend_adjustments(self, params: Dict) -> Dict:
        """Adjust for trending vs ranging markets."""
        if self.market_analysis:
            trend_strength = abs(self.market_analysis.trend_strength)
            
            if trend_strength > 50:  # Strong trend
                params["min_credit_ic"] *= 1.2
                params["min_credit_put"] *= 0.9
                params["min_credit_call"] *= 0.9
                # DO NOT modify min_strike_distance here!
                
            elif trend_strength < 20:  # Range-bound
                params["min_credit_ic"] *= 0.9
                # params["min_strike_distance"] *= 0.8  # DELETE THIS IF IT EXISTS!
        
        return params

    def _apply_structure_adjustments(self, params: Dict) -> Dict:
        """Adjust for market structure (support/resistance)."""
        if self.market_analysis:
            # Near support: favor put spreads
            if self.market_analysis.distance_to_support_pct < 0.5:
                params["min_credit_put"] *= 0.8
                self.logger.info("📊 Near support - favoring put spreads")
            
            # Near resistance: favor call spreads
            if self.market_analysis.distance_to_resistance_pct < 0.5:
                params["min_credit_call"] *= 0.8
                self.logger.info("📊 Near resistance - favoring call spreads")
        
        return params        