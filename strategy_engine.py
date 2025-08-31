"""
Strategy Engine
===============
Centralized strategy selection and strike calculation engine.
All strategy decisions are made here using parameters from MarketContext.

Author: Trading Systems
Version: 2.0
Date: August 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np

import math
from typing import Dict, Any

from config import get_config
from models import (
    StrategyType, TradeSetup, StrikeSelection, OptionContract,
    MarketAnalysis, RiskLevel
)
from exceptions import (
    StrategyError, InsufficientCreditError, StrikeTooCloseError,
    NoViableStrategyError, PositionLimitError
)
from market_context import MarketContext


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Each strategy implements its own strike selection logic
    but uses parameters from MarketContext.
    """
    
    def __init__(self, context: MarketContext, logger: Optional[logging.Logger] = None):
        """
        Initialize strategy with market context.
        
        Args:
            context: MarketContext instance for parameters
            logger: Optional logger for debugging
        """
        self.context = context
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = get_config()
    
    @abstractmethod
    def select_strikes(
        self,
        option_chain: List[OptionContract],
        style: str = "balanced"
    ) -> Optional[StrikeSelection]:
        """
        Select strikes for the strategy.
        
        Args:
            option_chain: Available options
            style: Trading style (safe/balanced/aggressive)
            
        Returns:
            StrikeSelection or None if no suitable strikes
        """
        pass
    
    @abstractmethod
    def validate_setup(self, strikes: StrikeSelection) -> bool:
        """
        Validate that the strike selection meets all requirements.
        
        Args:
            strikes: Selected strikes to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def calculate_position_size(self, strikes: StrikeSelection, style: str) -> int:
        """
        Calculate position size based on risk and conditions.
        
        This is common logic used by all strategies.
        
        Args:
            strikes: Selected strikes
            style: Trading style
            
        Returns:
            Number of contracts to trade
        """
        params = self.context.get_trading_parameters()
        account_config = self.config["account"]
        
        # Base calculation from risk
        max_risk_dollars = account_config.account_size * account_config.max_risk_per_trade
        risk_per_contract = strikes.max_risk * 20  # NQ multiplier
        
        if risk_per_contract > 0:
            contracts_by_risk = int(max_risk_dollars / risk_per_contract)
        else:
            contracts_by_risk = 1
        
        # Apply style multiplier
        style_mult = self.config["strategy"].styles[style]["size_multiplier"]
        contracts = int(contracts_by_risk * style_mult)
        
        # Apply parameter multipliers
        contracts = int(contracts * params["position_size_multiplier"])
        
        # Apply limits
        contracts = min(contracts, params["max_contracts"])
        contracts = max(1, contracts)  # At least 1 contract
        
        self.logger.info(f"Position size: {contracts} contracts "
                        f"(risk-based: {contracts_by_risk}, "
                        f"style: {style}, multiplier: {params['position_size_multiplier']:.2f})")
        
        return contracts


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor strategy implementation.
    
    This contains all your iron condor logic from the original code,
    but now uses parameters exclusively from MarketContext.
    """
    
    def select_strikes(
        self,
        option_chain: List[OptionContract],
        style: str = "balanced"
    ) -> Optional[StrikeSelection]:
        """Select iron condor strikes using centralized parameters."""
        
        # Get parameters from context (single source of truth!)
        params = self.context.get_trading_parameters()
        market_data = self.context.current_data
        
        if not market_data:
            self.logger.error("No market data available")
            return None
        
        current_price = market_data.underlying_price
        
        # USE THE PARAMETERS FROM CONTEXT!
        strike_distance = params["min_strike_distance"]
        wing_width = params["wing_width"]
        
        # Round to nearest 25 for NQ
        strike_distance = round(strike_distance / 25) * 25
        
        self.logger.info(f"Selecting {style.upper()} iron condor strikes")
        self.logger.info(f"  Strike distance: {strike_distance} points")
        self.logger.info(f"  Wing width: {wing_width} points")
        
        # Separate puts and calls
        puts = [opt for opt in option_chain if opt.right == 'P']
        calls = [opt for opt in option_chain if opt.right == 'C']
        
        # Sort for easier selection
        puts.sort(key=lambda x: x.strike, reverse=True)
        calls.sort(key=lambda x: x.strike)
        
        # Find short strikes using the CORRECT distance
        target_put_strike = current_price - strike_distance
        target_call_strike = current_price + strike_distance
        
        short_put = self._find_closest_strike(puts, target_put_strike, 'P', params)
        short_call = self._find_closest_strike(calls, target_call_strike, 'C', params)
        
        if not short_put or not short_call:
            self.logger.warning("Could not find suitable short strikes")
            return None
        
        # Find long strikes (protection)
        long_put_strike = short_put.strike - wing_width
        long_call_strike = short_call.strike + wing_width
        
        long_put = self._find_closest_strike(puts, long_put_strike, 'P')
        long_call = self._find_closest_strike(calls, long_call_strike, 'C')
        
        if not long_put or not long_call:
            self.logger.warning("Could not find suitable long strikes")
            return None
        
        # Calculate metrics
        put_credit = short_put.mid - long_put.mid
        call_credit = short_call.mid - long_call.mid
        total_credit = put_credit + call_credit
        
        # Max risk is wing width minus credit
        max_risk = wing_width - total_credit
        
        # Probability calculation
        prob_put_otm = 1 - abs(short_put.greeks.delta) if short_put.greeks else 0.7
        prob_call_otm = 1 - abs(short_call.greeks.delta) if short_call.greeks else 0.7
        prob_profit = prob_put_otm * prob_call_otm
        
        # Create strike selection
        strikes = StrikeSelection(
            short_put_strike=short_put.strike,
            long_put_strike=long_put.strike,
            short_call_strike=short_call.strike,
            long_call_strike=long_call.strike,
            put_credit=put_credit,
            call_credit=call_credit,
            total_credit=total_credit,
            max_risk=max_risk,
            probability_profit=prob_profit
        )
        
        self.logger.info(f"Iron Condor found:")
        self.logger.info(f"  Strikes: {short_put.strike}/{short_call.strike}")
        self.logger.info(f"  Protection: {long_put.strike}/{long_call.strike}")
        self.logger.info(f"  Credit: ${total_credit:.2f}")
        self.logger.info(f"  Max Risk: ${max_risk:.2f}")
        self.logger.info(f"  Win Probability: {prob_profit:.1%}")
        
        return strikes
    
    def validate_setup(self, strikes: StrikeSelection) -> bool:
        """
        Validate iron condor setup meets all requirements.
        FIXED: Add tolerance for strike distance validation.
        """
        params = self.context.get_trading_parameters()
        current_price = self.context.current_data.underlying_price
        
        # Check credit minimum
        if strikes.total_credit < params["min_credit_ic"]:
            raise InsufficientCreditError(
                strikes.total_credit,
                params["min_credit_ic"],
                "iron_condor"
            )
        
        # Check probability minimum
        if strikes.probability_profit < params["min_probability_ic"]:
            self.logger.warning(f"Probability {strikes.probability_profit:.1%} "
                            f"below minimum {params['min_probability_ic']:.1%}")
            return False
        
        # Check strike distances WITH TOLERANCE
        put_distance = current_price - strikes.short_put_strike
        call_distance = strikes.short_call_strike - current_price
        
        min_distance = params["min_strike_distance"]
        
        # ADD 20% TOLERANCE for low vol markets
        # If we're within 80% of the minimum distance, it's acceptable
        tolerance = 0.8
        adjusted_min = min_distance * tolerance
        
        if put_distance < adjusted_min:
            self.logger.warning(f"Put strike {put_distance:.0f} points away "
                            f"(min: {min_distance:.0f}, allowing {adjusted_min:.0f})")
            # In low vol, if credit is good, allow it anyway
            if params.get("min_credit_ic", 10) <= 5 and strikes.total_credit > 10:
                self.logger.info("Allowing close strike due to excellent credit in low vol")
            else:
                raise StrikeTooCloseError(
                    strikes.short_put_strike,
                    current_price,
                    min_distance
                )
        
        if call_distance < adjusted_min:
            self.logger.warning(f"Call strike {call_distance:.0f} points away "
                            f"(min: {min_distance:.0f}, allowing {adjusted_min:.0f})")
            # In low vol, if credit is good, allow it anyway
            if params.get("min_credit_ic", 10) <= 5 and strikes.total_credit > 10:
                self.logger.info("Allowing close strike due to excellent credit in low vol")
            else:
                raise StrikeTooCloseError(
                    strikes.short_call_strike,
                    current_price,
                    min_distance
                )
        
        # Validate strike relationships
        if not strikes.validate():
            self.logger.error("Strike validation failed")
            return False
        
        self.logger.info(f"✅ Setup validated: ${strikes.total_credit:.2f} credit, "
                        f"{strikes.probability_profit:.0%} win prob")
        
        return True
    
    def _find_closest_strike(
        self,
        options: List[OptionContract],
        target: float,
        right: str,
        params: Optional[Dict] = None
    ) -> Optional[OptionContract]:
        """
        Find the closest valid strike to target.
        
        Args:
            options: List of options to search
            target: Target strike price
            right: 'P' or 'C'
            params: Optional parameters for filtering
            
        Returns:
            Best matching option or None
        """
        if not options:
            return None
        
        # Filter by delta if params provided
        if params:
            max_delta = params.get("max_delta", 0.40)
            filtered = [
                opt for opt in options
                if opt.greeks and abs(opt.greeks.delta) <= max_delta
            ]
            
            # Also filter by bid-ask spread
            filtered = [
                opt for opt in filtered
                if opt.spread_pct < 30  # Max 30% spread
            ]
            
            if filtered:
                options = filtered
        
        # Find closest to target
        best = min(options, key=lambda x: abs(x.strike - target))
        
        return best


class PutSpreadStrategy(BaseStrategy):
    """
    Put Credit Spread strategy (bullish).
    
    Sells a put and buys a further OTM put for protection.
    """
    
    def select_strikes(
        self,
        option_chain: List[OptionContract],
        style: str = "balanced"
    ) -> Optional[StrikeSelection]:
        """Select put spread strikes using centralized parameters."""
        
        params = self.context.get_trading_parameters()
        current_price = self.context.current_data.underlying_price
        
        # Get puts only
        puts = [opt for opt in option_chain if opt.right == 'P']
        puts.sort(key=lambda x: x.strike, reverse=True)
        
        # USE PARAMETERS FROM CONTEXT!
        strike_distance = params["min_strike_distance"]
        wing_width = params["wing_width"]
        
        # Round to nearest 25 for NQ
        strike_distance = round(strike_distance / 25) * 25
        
        target_strike = current_price - strike_distance
        
        self.logger.info(f"Selecting {style.upper()} put spread")
        self.logger.info(f"  Target strike: {target_strike:.0f}")
        self.logger.info(f"  Wing width: {wing_width} points")
        
        # Find short put
        short_put = self._find_best_put(puts, target_strike, params)
        
        if not short_put:
            self.logger.warning("No suitable short put found")
            return None
        
        # Find long put (protection) using wing_width from params
        long_put_strike = short_put.strike - wing_width
        
        long_put = min(
            (p for p in puts if p.strike <= long_put_strike),
            key=lambda x: abs(x.strike - long_put_strike),
            default=None
        )
        
        if not long_put:
            self.logger.warning("No suitable long put found")
            return None
        
        # Calculate metrics
        credit = short_put.mid - long_put.mid
        max_risk = wing_width - credit
        prob_profit = 1 - abs(short_put.greeks.delta) if short_put.greeks else 0.7
        
        strikes = StrikeSelection(
            short_put_strike=short_put.strike,
            long_put_strike=long_put.strike,
            put_credit=credit,
            total_credit=credit,
            max_risk=max_risk,
            probability_profit=prob_profit
        )
        
        self.logger.info(f"Put spread found:")
        self.logger.info(f"  Short: {short_put.strike}P @ ${short_put.mid:.2f}")
        self.logger.info(f"  Long: {long_put.strike}P @ ${long_put.mid:.2f}")
        self.logger.info(f"  Credit: ${credit:.2f}")
        self.logger.info(f"  Max Risk: ${max_risk:.2f}")
        self.logger.info(f"  Win Probability: {prob_profit:.1%}")
        
        return strikes

    
    def validate_setup(self, strikes: StrikeSelection) -> bool:
        """Validate put spread setup with tolerance."""
        params = self.context.get_trading_parameters()
        current_price = self.context.current_data.underlying_price
        
        # Check credit minimum
        if strikes.total_credit < params["min_credit_put"]:
            raise InsufficientCreditError(
                strikes.total_credit,
                params["min_credit_put"],
                "put_spread"
            )
        
        # Check probability
        if strikes.probability_profit < params["min_probability_spread"]:
            self.logger.warning(f"Probability too low: {strikes.probability_profit:.1%}")
            return False
        
        # Check strike distance WITH TOLERANCE
        distance = current_price - strikes.short_put_strike
        min_distance = params["min_strike_distance"]
        adjusted_min = min_distance * 0.8  # 20% tolerance
        
        if distance < adjusted_min:
            # In low vol, be more flexible
            if params.get("min_credit_put", 5) <= 3 and strikes.total_credit > 2.5:
                self.logger.info("Allowing close strike in low vol market")
            else:
                raise StrikeTooCloseError(
                    strikes.short_put_strike,
                    current_price,
                    min_distance
                )
        
        return strikes.validate()
    
    def _find_best_put(
        self,
        puts: List[OptionContract],
        target: float,
        params: Dict
    ) -> Optional[OptionContract]:
        """Find best put option near target."""
        # For 0DTE, we need to be more flexible with delta
        # Original max_delta is too restrictive
        max_delta = 0.35  # Override to 0.35 for 0DTE (was probably 0.25 or less)
        
        candidates = [
            p for p in puts
            if p.greeks and abs(p.greeks.delta) <= max_delta
            and p.strike <= target + 100  # Increased from 50 to 100 for more flexibility
            and p.bid > 0.5  # Ensure there's some premium
        ]
        
        if not candidates:
            # Fallback: just find closest strike with any premium
            self.logger.warning(f"No puts with delta <= {max_delta}, using fallback")
            candidates = [
                p for p in puts
                if p.strike <= target + 200  # Even wider range
                and p.bid > 0.25  # Any premium
            ]
        
        if not candidates:
            return None
        
        # Find closest to target
        return min(candidates, key=lambda x: abs(x.strike - target))


class CallSpreadStrategy(BaseStrategy):
    """
    Call Credit Spread strategy (bearish).
    
    Sells a call and buys a further OTM call for protection.
    """
    
    def select_strikes(
        self,
        option_chain: List[OptionContract],
        style: str = "balanced"
    ) -> Optional[StrikeSelection]:
        """Select call spread strikes using centralized parameters."""
        
        params = self.context.get_trading_parameters()
        current_price = self.context.current_data.underlying_price
        
        # Get calls only
        calls = [opt for opt in option_chain if opt.right == 'C']
        calls.sort(key=lambda x: x.strike)
        
        # USE PARAMETERS FROM CONTEXT!
        strike_distance = params["min_strike_distance"]
        wing_width = params["wing_width"]
        
        # Round to nearest 25 for NQ
        strike_distance = round(strike_distance / 25) * 25
        
        target_strike = current_price + strike_distance
        
        self.logger.info(f"Selecting {style.upper()} call spread")
        self.logger.info(f"  Target strike: {target_strike:.0f}")
        self.logger.info(f"  Wing width: {wing_width} points")
        
        # Find short call
        short_call = self._find_best_call(calls, target_strike, params)
        
        if not short_call:
            self.logger.warning("No suitable short call found")
            return None
        
        # Find long call (protection) using wing_width from params
        long_call_strike = short_call.strike + wing_width
        
        long_call = min(
            (c for c in calls if c.strike >= long_call_strike),
            key=lambda x: abs(x.strike - long_call_strike),
            default=None
        )
        
        if not long_call:
            self.logger.warning("No suitable long call found")
            return None
        
        # Calculate metrics
        credit = short_call.mid - long_call.mid
        max_risk = wing_width - credit
        prob_profit = 1 - short_call.greeks.delta if short_call.greeks else 0.7
        
        strikes = StrikeSelection(
            short_call_strike=short_call.strike,
            long_call_strike=long_call.strike,
            call_credit=credit,
            total_credit=credit,
            max_risk=max_risk,
            probability_profit=prob_profit
        )
        
        self.logger.info(f"Call spread found:")
        self.logger.info(f"  Short: {short_call.strike}C @ ${short_call.mid:.2f}")
        self.logger.info(f"  Long: {long_call.strike}C @ ${long_call.mid:.2f}")
        self.logger.info(f"  Credit: ${credit:.2f}")
        self.logger.info(f"  Max Risk: ${max_risk:.2f}")
        self.logger.info(f"  Win Probability: {prob_profit:.1%}")
        
        return strikes
    
    def validate_setup(self, strikes: StrikeSelection) -> bool:
        """Validate call spread setup with tolerance."""
        params = self.context.get_trading_parameters()
        current_price = self.context.current_data.underlying_price
        
        # Check credit minimum
        if strikes.total_credit < params["min_credit_call"]:
            raise InsufficientCreditError(
                strikes.total_credit,
                params["min_credit_call"],
                "call_spread"
            )
        
        # Check probability
        if strikes.probability_profit < params["min_probability_spread"]:
            self.logger.warning(f"Probability too low: {strikes.probability_profit:.1%}")
            return False
        
        # Check strike distance WITH TOLERANCE
        distance = strikes.short_call_strike - current_price
        min_distance = params["min_strike_distance"]
        adjusted_min = min_distance * 0.8  # 20% tolerance
        
        if distance < adjusted_min:
            # In low vol, be more flexible
            if params.get("min_credit_call", 5) <= 2 and strikes.total_credit > 1.0:
                self.logger.info("Allowing close strike in low vol market")
            else:
                raise StrikeTooCloseError(
                    strikes.short_call_strike,
                    current_price,
                    min_distance
                )
        
        return strikes.validate()
    
    def _find_best_call(
        self,
        calls: List[OptionContract],
        target: float,
        params: Dict
    ) -> Optional[OptionContract]:
        """Find best call option near target."""
        # For 0DTE, we need to be more flexible with delta
        max_delta = 0.35  # Override to 0.35 for 0DTE
        
        candidates = [
            c for c in calls
            if c.greeks and c.greeks.delta <= max_delta
            and c.strike >= target - 100  # Increased from 50 to 100
            and c.bid > 0.5  # Ensure there's some premium
        ]
        
        if not candidates:
            # Fallback: just find closest strike with any premium
            self.logger.warning(f"No calls with delta <= {max_delta}, using fallback")
            candidates = [
                c for c in calls
                if c.strike >= target - 200  # Wider range
                and c.bid > 0.25  # Any premium
            ]
        
        if not candidates:
            return None
        
        # Find closest to target
        return min(candidates, key=lambda x: abs(x.strike - target))


class StrategyEngine:
    """
    Main strategy engine that coordinates all strategies.
    
    This is the central decision maker that:
    1. Analyzes market conditions
    2. Selects appropriate strategy
    3. Calculates strikes
    4. Validates setups
    5. Returns complete trade setups
    """
    
    def __init__(self, context: MarketContext, logger: Optional[logging.Logger] = None):
        """
        Initialize strategy engine.
        
        Args:
            context: MarketContext for all parameters
            logger: Optional logger
        """
        self.context = context
        self.logger = logger or self._setup_logger()
        self.config = get_config()
        
        # Initialize all strategies
        self.strategies = {
            StrategyType.IRON_CONDOR: IronCondorStrategy(context, self.logger),
            StrategyType.PUT_SPREAD: PutSpreadStrategy(context, self.logger),
            StrategyType.CALL_SPREAD: CallSpreadStrategy(context, self.logger)
        }
        
        # Track strategy performance
        self.strategy_stats = {
            strategy: {"attempts": 0, "successes": 0, "failures": 0}
            for strategy in StrategyType
        }
        
        self.logger.info("Strategy Engine initialized with all strategies")
    
    def _setup_logger(self) -> logging.Logger:
        """Create logger if none provided."""
        logger = logging.getLogger("StrategyEngine")
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
    
    def get_best_trade(
        self,
        option_chain: List[OptionContract],
        style: str = "balanced",
        override_strategy: Optional[StrategyType] = None
    ) -> Optional[TradeSetup]:
        """
        Get the best available trade based on current conditions.
        
        This is the main method that orchestrates everything:
        1. Analyzes market
        2. Selects strategy
        3. Finds strikes
        4. Validates setup
        5. Returns complete TradeSetup
        
        Args:
            option_chain: Available options
            style: Trading style (safe/balanced/aggressive)
            override_strategy: Force specific strategy (optional)
            
        Returns:
            Complete TradeSetup ready for execution, or None
        """
        self.logger.info("=" * 60)
        self.logger.info("STRATEGY ENGINE: Finding best trade")
        self.logger.info(f"Style: {style.upper()}")
        
        # Analyze market if not already done
        if not self.context.market_analysis:
            analysis = self.context.analyze_market()
        else:
            analysis = self.context.market_analysis
        
        # Select strategy
        if override_strategy:
            strategy_type = override_strategy
            self.logger.info(f"Using override strategy: {strategy_type.value}")
        else:
            strategy_type = self._select_strategy(analysis)
            self.logger.info(f"Selected strategy: {strategy_type.value}")
        
        # Check if we should skip
        if strategy_type == StrategyType.SKIP:
            self.logger.warning("Market conditions suggest skipping trade")
            return None
        
        # Get the strategy implementation
        strategy = self.strategies.get(strategy_type)
        if not strategy:
            self.logger.error(f"Strategy {strategy_type} not implemented")
            return None
        
        # Track attempt
        self.strategy_stats[strategy_type]["attempts"] += 1
        
        try:
            # Select strikes
            strikes = strategy.select_strikes(option_chain, style)
            
            if not strikes:
                self.logger.warning(f"No suitable strikes found for {strategy_type.value}")
                self.strategy_stats[strategy_type]["failures"] += 1
                
                # Try alternative strategy
                alternative = self._get_alternative_strategy(strategy_type, analysis)
                if alternative and alternative != strategy_type:
                    self.logger.info(f"Trying alternative: {alternative.value}")
                    return self.get_best_trade(option_chain, style, alternative)
                
                return None
            
            # Validate setup
            if not strategy.validate_setup(strikes):
                self.logger.warning("Setup validation failed")
                self.strategy_stats[strategy_type]["failures"] += 1
                return None
            
            # Calculate position size
            contracts = strategy.calculate_position_size(strikes, style)
            
            # Create complete trade setup
            trade_setup = self._create_trade_setup(
                strategy_type, strikes, contracts, analysis
            )
            
            self.strategy_stats[strategy_type]["successes"] += 1
            
            self.logger.info("=" * 60)
            self.logger.info("TRADE SETUP COMPLETE:")
            self.logger.info(f"  Strategy: {trade_setup.strategy.value}")
            self.logger.info(f"  Contracts: {trade_setup.contracts}")
            self.logger.info(f"  Credit: ${trade_setup.credit_per_contract:.2f}/contract")
            self.logger.info(f"  Max Risk: ${trade_setup.max_risk_per_contract:.2f}/contract")
            self.logger.info(f"  Total Credit: ${trade_setup.credit_per_contract * trade_setup.contracts * 20:.2f}")  # x20 for NQ
            self.logger.info(f"  Total Risk: ${trade_setup.max_risk_per_contract * trade_setup.contracts * 20:.2f}")  # x20 for NQ
            self.logger.info(f"  Confidence: {trade_setup.confidence:.0f}%")
            self.logger.info(f"  Risk Level: {trade_setup.risk_level.value}")
            self.logger.info(f"  Rationale: {trade_setup.rationale}")
            self.logger.info("=" * 60)
            
            return trade_setup
            
        except StrategyError as e:
            self.logger.error(f"Strategy error: {e}")
            self.strategy_stats[strategy_type]["failures"] += 1
            
            # Try alternative on specific errors
            if isinstance(e, InsufficientCreditError):
                self.logger.info("Insufficient credit - trying alternative")
                alternative = self._get_alternative_strategy(strategy_type, analysis)
                if alternative:
                    return self.get_best_trade(option_chain, style, alternative)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _select_strategy(self, analysis: MarketAnalysis) -> StrategyType:
        """
        Select strategy based on market analysis.
        
        This encapsulates your strategy selection logic.
        """
        # First, check if analysis recommends skipping
        if analysis.recommended_strategy == StrategyType.SKIP:
            return StrategyType.SKIP
        
        # Check risk level
        if analysis.risk_assessment == RiskLevel.EXTREME:
            self.logger.warning("Extreme risk - skipping trade")
            return StrategyType.SKIP
        
        # Use analysis recommendation
        strategy = analysis.recommended_strategy
        
        # Apply additional filters
        params = self.context.get_trading_parameters()
        
        # Don't trade if market closing soon
        if not params["can_trade"]:
            self.logger.warning("Outside trading window")
            return StrategyType.SKIP
        
        # Confidence filter
        if analysis.confidence_level < 40:
            self.logger.warning(f"Confidence too low: {analysis.confidence_level:.0f}%")
            return StrategyType.SKIP
        
        return strategy
    
    def _get_alternative_strategy(
        self,
        failed_strategy: StrategyType,
        analysis: MarketAnalysis
    ) -> Optional[StrategyType]:
        """
        Get alternative strategy when primary fails.
        
        Args:
            failed_strategy: Strategy that failed
            analysis: Market analysis
            
        Returns:
            Alternative strategy or None
        """
        # Define alternatives
        # alternatives = {
        #     StrategyType.IRON_CONDOR: (
        #         StrategyType.PUT_SPREAD if analysis.trend_strength > 20
        #         else StrategyType.CALL_SPREAD if analysis.trend_strength < -20
        #         else None
        #     ),
        #     StrategyType.PUT_SPREAD: StrategyType.IRON_CONDOR,
        #     StrategyType.CALL_SPREAD: StrategyType.IRON_CONDOR
        # }
        alternatives = {
        StrategyType.IRON_CONDOR: StrategyType.PUT_SPREAD,  # Always try put spread first
        StrategyType.PUT_SPREAD: StrategyType.CALL_SPREAD,   # If put fails, try call
        StrategyType.CALL_SPREAD: None  # We've exhausted all options
        }
        
        return alternatives.get(failed_strategy)
    
    def _create_trade_setup(
        self,
        strategy: StrategyType,
        strikes: StrikeSelection,
        contracts: int,
        analysis: MarketAnalysis
    ) -> TradeSetup:
        """
        Create complete trade setup from components.
        
        Args:
            strategy: Selected strategy
            strikes: Selected strikes
            contracts: Number of contracts
            analysis: Market analysis
            
        Returns:
            Complete TradeSetup object
        """
        params = self.context.get_trading_parameters()
        
        # Calculate stop loss and profit targets
        stop_loss = strikes.total_credit * params["stop_loss_multiplier"]
        profit_targets = [
            strikes.total_credit * target
            for target in params["profit_targets"]
        ]
        
        # Determine confidence (use analysis confidence or calculate)
        confidence = analysis.confidence_level if analysis else 50
        
        # Add strategy-specific confidence adjustments
        if strategy == StrategyType.IRON_CONDOR:
            # IC confidence based on probability and credit
            if strikes.probability_profit > 0.60 and strikes.total_credit > params["min_credit_ic"]:
                confidence = min(100, confidence + 10)
        elif strategy == StrategyType.PUT_SPREAD:
            # Put spread confidence based on trend alignment
            if analysis.trend_strength > 20:
                confidence = min(100, confidence + 15)
        elif strategy == StrategyType.CALL_SPREAD:
            # Call spread confidence based on trend alignment
            if analysis.trend_strength < -20:
                confidence = min(100, confidence + 15)
        
        # Create and return the TradeSetup
        rationale = f"{strategy.value} selected based on market analysis and parameters."
        return TradeSetup(
            strategy=strategy,
            strikes=strikes,
            contracts=contracts,
            credit_per_contract=strikes.total_credit,
            max_risk_per_contract=strikes.max_risk,
            confidence=analysis.confidence_level if analysis else 50,
            risk_level=analysis.risk_assessment if analysis else RiskLevel.MEDIUM,
            rationale=rationale
        )

    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all strategies.
        Returns:
            Dictionary containing strategy statistics 
        """

        stats: Dict[str, Any] = {}
        total_attempts = 0
        total_wins = 0
        total_losses = 0

        for name, strategy in self.strategies.items():
            # Pull values with safe defaults
            attempts = _num0(getattr(strategy, "attempts", getattr(strategy, "signals_count", 0)))
            wins     = _num0(getattr(strategy, "wins", 0))
            losses   = _num0(getattr(strategy, "losses", max(0, attempts - wins)))
            success_rate = (wins / attempts * 100.0) if attempts else 0.0

            stats[name] = {
                "name": name,
                "enabled": bool(getattr(strategy, "enabled", True)),
                "signals_generated": _num0(getattr(strategy, "signals_count", 0)),
                "last_signal": _str0(getattr(strategy, "last_signal_time", None)),
                "parameters": getattr(strategy, "parameters", {}) or {},
                "attempts": int(attempts),
                "wins": int(wins),
                "losses": int(losses),
                "success_rate": float(success_rate),  # always safe to format
            }

            total_attempts += attempts
            total_wins += wins
            total_losses += losses

        # Aggregate summary
        stats["summary"] = {
            "total_strategies": len(self.strategies),
            "active_strategies": sum(
                1 for s in stats.values()
                if isinstance(s, dict) and s.get("enabled", False) and s is not stats.get("summary")
            ),
            "total_signals": sum(
                s.get("signals_generated", 0)
                for s in stats.values()
                if isinstance(s, dict) and s is not stats.get("summary")
            ),
            "total_attempts": int(total_attempts),
            "total_wins": int(total_wins),
            "total_losses": int(total_losses),
            "success_rate": (total_wins / total_attempts * 100.0) if total_attempts else 0.0,
        }

        return stats
        
    # def get_strategy_stats(self) -> Dict[str, Any]:
    #     """
    #     Get statistics for all strategies.
        
    #     Returns:
    #         Dictionary containing strategy statistics
    #     """
    #     stats = {}
        
    #     for name, strategy in self.strategies.items():
    #         # Get basic stats for each strategy
    #         stats[name] = {
    #             'name': name,
    #             'enabled': getattr(strategy, 'enabled', True),
    #             'signals_generated': getattr(strategy, 'signals_count', 0),
    #             'last_signal': getattr(strategy, 'last_signal_time', None),
    #             'parameters': getattr(strategy, 'parameters', {}),
    #             # Add more stats as needed based on your strategy implementation
    #         }
        
    #     # Add aggregate statistics
    #     stats['summary'] = {
    #         'total_strategies': len(self.strategies),
    #         'active_strategies': sum(1 for s in stats.values() 
    #                             if isinstance(s, dict) and s.get('enabled', False)),
    #         'total_signals': sum(s.get('signals_generated', 0) 
    #                         for s in stats.values() 
    #                         if isinstance(s, dict)),
    #     }
        
        return stats
    
def _num0(x) -> float:
    """Return 0 if None/NaN, else numeric value."""
    if x is None:
        return 0
    if isinstance(x, float) and math.isnan(x):
        return 0
    return x

def _str0(x) -> str:
    """Return '' if None/NaN, else string."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)
