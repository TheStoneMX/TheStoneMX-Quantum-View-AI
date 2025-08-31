"""
Trading System Configuration
============================
Central configuration file for all trading parameters and settings.
This is your single source of truth for all configuration values.

Author: Trading Systems
Version: 2.0
Date: August 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class BrokerConfig:
    """Interactive Brokers connection configuration."""
    
    host: str = "127.0.0.1"
    paper_port: int = 7497  # Paper trading port
    live_port: int = 7496   # Live trading port
    client_id: int = 1
    
    # Connection settings
    connection_timeout: int = 30  # seconds
    request_timeout: int = 10     # seconds
    max_retries: int = 3
    

@dataclass
class AccountConfig:
    """Account and risk management configuration."""
    
    # Account settings
    account_size: float = 15000  # Starting capital
    
    # Position limits
    max_daily_trades: int = 5
    max_concurrent_positions: int = 3
    max_contracts_per_trade: int = 3
    
    # Risk parameters (as percentage of account)
    max_risk_per_trade: float = 0.01  # 1% per trade
    max_daily_loss: float = 0.02      # 2% daily max
    max_position_risk: float = 0.015  # 1.5% per position
    
    # Emergency stops
    consecutive_loss_limit: int = 3
    emergency_stop_loss: float = 0.03  # 3% account drawdown


@dataclass
class MarketThresholds:
    """Market condition thresholds for decision making."""
    
    # VIX levels
    vix_low: float = 16.0
    vix_normal: float = 20.0
    vix_high: float = 25.0
    vix_extreme: float = 30.0
    
    # Trend strength thresholds
    trend_strong: float = 60.0
    trend_moderate: float = 30.0
    trend_neutral: float = 20.0
    
    # Movement thresholds (as percentage of price)
    min_strike_distance_pct: float = 0.004  # 0.4% minimum
    max_strike_distance_pct: float = 0.010  # 1.0% maximum
    
    # Time-based thresholds
    minutes_before_close_stop: int = 30  # Stop new trades
    minutes_before_close_exit: int = 20  # Exit all 0DTE
    

@dataclass
class StrategyParameters:
    """Base strategy parameters that adapt to market conditions."""
    
    # Wing widths by volatility regime (in points)
    wing_widths: Dict[str, int] = field(default_factory=lambda: {
        "low_vix": 75,      # VIX < 16: Wide wings for credit
        "normal_vix": 50,   # VIX 16-25: Standard wings
        "high_vix": 50,     # VIX > 25: Keep reasonable width
        "extreme_vix": 40   # VIX > 30: Tighter for protection
    })
    
    # Minimum credit requirements by VIX (base values)
    min_credit_base: Dict[str, float] = field(default_factory=lambda: {
        "iron_condor": 15.0,
        "put_spread": 8.0,
        "call_spread": 8.0
    })
    
    # Profit targets (as percentage of max profit)
    profit_targets: List[float] = field(default_factory=lambda: [
        0.25,  # First target: close half
        0.50,  # Second target: close remaining
        0.75   # Emergency target
    ])
    
    # Stop loss multipliers (times credit received)
    stop_loss_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "low_vix": 1.5,    # Tighter stops in low vol
        "normal_vix": 2.0,  # Standard 2x
        "high_vix": 2.5    # Wider stops in high vol
    })
    
    # Strike selection styles
    styles: Dict[str, Dict] = field(default_factory=lambda: {
        "safe": {
            "strike_distance_multiplier": 1.5,  # 1.5x ATR
            "size_multiplier": 1.0,
            "min_win_probability": 0.70
        },
        "balanced": {
            "strike_distance_multiplier": 1.0,  # 1.0x ATR
            "size_multiplier": 0.9,
            "min_win_probability": 0.60
        },
        "aggressive": {
            "strike_distance_multiplier": 0.7,  # 0.7x ATR
            "size_multiplier": 0.6,
            "min_win_probability": 0.55
        }
    })


@dataclass
class TradingHours:
    """Trading schedule configuration for Spain-based trading."""
    
    # Market hours in ET
    market_open_et: tuple = (9, 30)   # 9:30 AM ET
    market_close_et: tuple = (16, 0)  # 4:00 PM ET
    
    # Optimal trading windows from Spain (in Madrid time)
    spain_windows: List[Dict] = field(default_factory=lambda: [
        {"start": (15, 30), "end": (17, 0), "quality": "good"},       # Early US session
        {"start": (17, 0), "end": (19, 0), "quality": "optimal"},    # Prime time
        {"start": (19, 0), "end": (20, 30), "quality": "acceptable"},  # Late session
        {"start": (20, 30), "end": (21, 0), "quality": "closing_only"}  # Position management only
    ])
    
    # Days of week weights (0=Monday, 4=Friday)
    day_weights: Dict[int, float] = field(default_factory=lambda: {
        0: 1.2,  # Monday - highest profitability
        1: 0.9,  # Tuesday
        2: 1.1,  # Wednesday (FOMC days)
        3: 0.9,  # Thursday
        4: 0.8   # Friday - lowest (weekend risk)
    })


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log formatting
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format: str = "%H:%M:%S"
    
    # File settings
    log_directory: Path = Path("logs")
    max_log_size: int = 10_485_760  # 10MB
    backup_count: int = 5
    

# Main configuration instance
CONFIG = {
    "broker": BrokerConfig(),
    "account": AccountConfig(),
    "thresholds": MarketThresholds(),
    "strategy": StrategyParameters(),
    "hours": TradingHours(),
    "logging": LoggingConfig()
}


def get_config() -> Dict:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration sections
    """
    return CONFIG


def update_config(section: str, **kwargs) -> None:
    """
    Update configuration values at runtime.
    
    Args:
        section: Configuration section to update
        **kwargs: Key-value pairs to update
        
    Example:
        update_config("account", account_size=20000, max_daily_trades=10)
    """
    if section in CONFIG:
        for key, value in kwargs.items():
            if hasattr(CONFIG[section], key):
                setattr(CONFIG[section], key, value)
            else:
                raise ValueError(f"Invalid config key: {key} in section {section}")
    else:
        raise ValueError(f"Invalid config section: {section}")