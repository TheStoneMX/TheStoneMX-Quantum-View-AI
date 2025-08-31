"""
Custom Exception Classes
========================
Specific exceptions for different failure modes in the trading system.
This enables precise error handling and debugging.

Author: Trading Systems
Version: 2.0
Date: August 2025
"""

from typing import Optional, Any


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# MARKET DATA EXCEPTIONS
# ============================================================================

class MarketDataError(TradingSystemError):
    """Base class for market data related errors."""
    pass


class NoMarketDataError(MarketDataError):
    """Raised when market data is unavailable."""
    pass


class StaleDataError(MarketDataError):
    """Raised when market data is too old."""
    
    def __init__(self, age_seconds: float):
        super().__init__(
            f"Market data is stale ({age_seconds:.1f} seconds old)",
            {"age_seconds": age_seconds}
        )


class InvalidPriceError(MarketDataError):
    """Raised when price data is invalid."""
    pass


# ============================================================================
# STRATEGY EXCEPTIONS
# ============================================================================

class StrategyError(TradingSystemError):
    """Base class for strategy related errors."""
    pass


class InsufficientCreditError(StrategyError):
    """Raised when option credit is below minimum."""
    
    def __init__(self, actual_credit: float, min_credit: float, strategy: str):
        super().__init__(
            f"{strategy}: Credit ${actual_credit:.2f} below minimum ${min_credit:.2f}",
            {
                "actual_credit": actual_credit,
                "min_credit": min_credit,
                "strategy": strategy
            }
        )


class StrikeTooCloseError(StrategyError):
    """Raised when strikes are too close to current price."""
    
    def __init__(self, strike: float, current_price: float, min_distance: float):
        distance = abs(strike - current_price)
        super().__init__(
            f"Strike {strike} only {distance:.0f} points from price {current_price:.0f}",
            {
                "strike": strike,
                "current_price": current_price,
                "distance": distance,
                "min_distance": min_distance
            }
        )


class NoViableStrategyError(StrategyError):
    """Raised when no suitable strategy can be found."""
    pass


class PositionLimitError(StrategyError):
    """Raised when position limits are reached."""
    
    def __init__(self, current_positions: int, max_positions: int):
        super().__init__(
            f"Position limit reached: {current_positions}/{max_positions}",
            {
                "current_positions": current_positions,
                "max_positions": max_positions
            }
        )


# ============================================================================
# EXECUTION EXCEPTIONS
# ============================================================================

class ExecutionError(TradingSystemError):
    """Base class for execution related errors."""
    pass


class OrderRejectedError(ExecutionError):
    """Raised when broker rejects an order."""
    
    def __init__(self, reason: str, order_details: Optional[dict] = None):
        super().__init__(f"Order rejected: {reason}", order_details)


class FillError(ExecutionError):
    """Raised when order doesn't fill as expected."""
    pass


class ConnectionError(ExecutionError):
    """Raised when broker connection fails."""
    pass


# ============================================================================
# RISK MANAGEMENT EXCEPTIONS
# ============================================================================

class RiskError(TradingSystemError):
    """Base class for risk management errors."""
    pass


class DailyLossLimitError(RiskError):
    """Raised when daily loss limit is reached."""
    
    def __init__(self, current_loss: float, max_loss: float):
        super().__init__(
            f"Daily loss limit reached: ${current_loss:.2f} / ${max_loss:.2f}",
            {
                "current_loss": current_loss,
                "max_loss": max_loss
            }
        )


class RiskLimitExceededError(RiskError):
    """Raised when position risk exceeds limits."""
    
    def __init__(self, risk_amount: float, max_risk: float):
        super().__init__(
            f"Risk ${risk_amount:.2f} exceeds limit ${max_risk:.2f}",
            {
                "risk_amount": risk_amount,
                "max_risk": max_risk
            }
        )


class MarginError(RiskError):
    """Raised when insufficient margin is available."""
    pass


# ============================================================================
# TIMING EXCEPTIONS
# ============================================================================

class TimingError(TradingSystemError):
    """Base class for timing related errors."""
    pass


class MarketClosedError(TimingError):
    """Raised when attempting to trade outside market hours."""
    
    def __init__(self, current_time: str, market_hours: str):
        super().__init__(
            f"Market closed. Current: {current_time}, Hours: {market_hours}",
            {
                "current_time": current_time,
                "market_hours": market_hours
            }
        )


class TooLateToTradeError(TimingError):
    """Raised when too close to market close for new trades."""
    
    def __init__(self, minutes_to_close: float):
        super().__init__(
            f"Too late to trade: {minutes_to_close:.0f} minutes to close",
            {"minutes_to_close": minutes_to_close}
        )


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(TradingSystemError):
    """Base class for configuration errors."""
    pass


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter value is invalid."""
    
    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(
            f"Invalid parameter {parameter}={value}: {reason}",
            {
                "parameter": parameter,
                "value": value,
                "reason": reason
            }
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass