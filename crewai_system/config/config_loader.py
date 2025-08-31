# crewai_system/config/config_loader.py
"""
Configuration Loader and Validator
===================================
Loads and validates the YAML configuration file.
Supports both OpenAI and Ollama LLM providers.

Author: CrewAI Trading System
Version: 2.0
Date: December 2024
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class TradingConfig:
    """
    Flexible configuration object that supports multiple LLM providers.
    
    Automatically detects whether OpenAI or Ollama is configured
    and provides appropriate properties.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Parsed YAML configuration
        """
        self.config = config_dict
        self.logger = logging.getLogger("ConfigLoader")
        
        # Validate configuration
        self._validate()
        
        # Detect LLM provider
        self.llm_provider = self._detect_llm_provider()
        self.logger.info(f"Using LLM provider: {self.llm_provider}")
    
    def _detect_llm_provider(self) -> str:
        """Detect which LLM provider is configured."""
        if 'openai' in self.config:
            return 'openai'
        elif 'ollama' in self.config:
            return 'ollama'
        else:
            raise ValueError("No LLM provider configured. Add 'openai' or 'ollama' section to config.")
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Risk validations
        max_risk = self.config['risk_limits']['max_risk_per_trade_pct']
        if not 0 < max_risk <= 10:
            raise ValueError(f"max_risk_per_trade_pct must be between 0 and 10, got {max_risk}")
        
        max_loss = self.config['risk_limits']['max_daily_loss_pct']
        if not 0 < max_loss <= 20:
            raise ValueError(f"max_daily_loss_pct must be between 0 and 20, got {max_loss}")
        
        # Path validations
        Path(self.logs_path).mkdir(parents=True, exist_ok=True)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
    # ==================== Test / Backtest Mode ====================

    @property
    def is_test_mode(self) -> bool:
        """Whether the system is running in local/backtest mode."""
        return bool(self.config.get('test_mode', False))        
    
    # ==================== Path Properties ====================
    
    @property
    def database_path(self) -> str:
        return self.config['paths']['database']
    
    @property
    def logs_path(self) -> str:
        return self.config['paths']['logs']
    
    @property
    def existing_modules_path(self) -> str:
        return self.config['paths']['existing_modules']
    
    # ==================== LLM Properties ====================
    
    @property
    def has_openai(self) -> bool:
        """Check if OpenAI is configured."""
        return 'openai' in self.config
    
    @property
    def has_ollama(self) -> bool:
        """Check if Ollama is configured."""
        return 'ollama' in self.config
    
    # OpenAI Properties
    @property
    def openai_model(self) -> str:
        return self.config.get('openai', {}).get('model', 'gpt-4o-mini')
    
    @property
    def openai_temperature(self) -> float:
        return self.config.get('openai', {}).get('temperature', 0.3)
    
    @property
    def openai_max_tokens(self) -> int:
        return self.config.get('openai', {}).get('max_tokens', 2000)
    
    @property
    def openai_timeout_seconds(self) -> int:
        return self.config.get('openai', {}).get('timeout_seconds', 30)
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or config."""
        # Priority: Environment variable > Config file
        return os.getenv('OPENAI_API_KEY') or self.config.get('openai', {}).get('api_key')
    
    # Ollama Properties (backward compatibility)
    @property
    def ollama_model(self) -> str:
        return self.config.get('ollama', {}).get('model', 'mixtral:8x7b-instruct-v0.1-q5_K_M')
    
    @property
    def ollama_base_url(self) -> str:
        return self.config.get('ollama', {}).get('base_url', 'http://localhost:11434')
    
    @property
    def ollama_temperature(self) -> float:
        return self.config.get('ollama', {}).get('temperature', 0.3)
    
    @property
    def ollama_timeout(self) -> int:
        return self.config.get('ollama', {}).get('timeout_seconds', 30)
    
    @property
    def ollama_url(self) -> str:
        """Alias for backward compatibility."""
        return self.ollama_base_url
    
    # ==================== Risk Properties ====================
    
    @property
    def max_daily_trades(self) -> int:
        return self.config['risk_limits']['max_daily_trades']
    
    @property
    def max_concurrent_positions(self) -> int:
        return self.config['risk_limits']['max_concurrent_positions']
    
    @property
    def max_contracts_per_trade(self) -> int:
        return self.config['risk_limits']['max_contracts_per_trade']
    
    @property
    def max_risk_per_trade_pct(self) -> float:
        return self.config['risk_limits']['max_risk_per_trade_pct']
    
    @property
    def max_daily_loss_pct(self) -> float:
        return self.config['risk_limits']['max_daily_loss_pct']
    
    @property
    def max_consecutive_losses(self) -> int:
        return self.config['risk_limits'].get('consecutive_loss_limit', 3)
    
    # ==================== Strategy Properties ====================
    
    @property
    def wing_width_min(self) -> int:
        return self.config['strategy_params']['wing_width_bounds']['min']
    
    @property
    def wing_width_max(self) -> int:
        return self.config['strategy_params']['wing_width_bounds']['max']
    
    @property
    def min_credit_ic(self) -> float:
        return self.config['strategy_params']['credit_requirements']['iron_condor_min']
    
    @property
    def min_credit_put(self) -> float:
        return self.config['strategy_params']['credit_requirements']['put_spread_min']
    
    @property
    def min_credit_call(self) -> float:
        return self.config['strategy_params']['credit_requirements']['call_spread_min']
    
    @property
    def strike_distance_min_pct(self) -> float:
        return self.config['strategy_params'].get('strike_distance_bounds', {}).get('min_pct', 0.3)
    
    @property
    def strike_distance_max_pct(self) -> float:
        return self.config['strategy_params'].get('strike_distance_bounds', {}).get('max_pct', 2.0)
    
    # ==================== Analysis Properties ====================
    
    @property
    def analysis_intervals(self) -> Dict[str, int]:
        return self.config['analysis_intervals']
    
    # ==================== Agent Properties ====================
    
    @property
    def memory_lookback(self) -> int:
        return self.config['agents']['memory_lookback_trades']
    
    @property
    def confidence_threshold(self) -> float:
        return self.config['agents']['confidence_threshold']
    
    @property
    def reasoning_detail(self) -> str:
        return self.config['agents']['reasoning_detail']
    
    @property
    def decision_timeouts(self) -> Dict[str, int]:
        return self.config['agents']['decision_timeouts']
    
    @property
    def market_analysis_timeout(self) -> int:
        return self.config['agents']['decision_timeouts'].get('market_analysis', 30)
    
    @property
    def strategy_selection_timeout(self) -> int:
        return self.config['agents']['decision_timeouts'].get('strategy_selection', 30)
    
    @property
    def risk_assessment_timeout(self) -> int:
        return self.config['agents']['decision_timeouts'].get('risk_assessment', 20)
    
    @property
    def chief_trading_officer_timeout(self) -> int:
        return self.config['agents']['decision_timeouts'].get('chief_trading_officer', 45)
    
    # ==================== CrewAI Properties ====================
    
    @property
    def crew_verbose(self) -> bool:
        return self.config['crew']['verbose']
    
    @property
    def crew_memory(self) -> bool:
        return self.config['crew']['memory']
    
    @property
    def max_iterations(self) -> int:
        return self.config['crew']['max_iterations']
    
    # ==================== Position Management Properties ====================
    
    @property
    def stop_loss_pct(self) -> float:
        return self.config.get('position_management', {}).get('stop_loss_pct', 50)
    
    @property
    def profit_target_pct(self) -> float:
        return self.config.get('position_management', {}).get('profit_target_pct', 25)
    
    @property
    def time_stop_minutes(self) -> int:
        return self.config.get('position_management', {}).get('time_stop_minutes', 30)
    
    # ==================== Helper Methods ====================
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get the appropriate LLM configuration based on provider.
        
        Returns:
            Dictionary with LLM configuration
        """
        if self.llm_provider == 'openai':
            return {
                'provider': 'openai',
                'model': self.openai_model,
                'temperature': self.openai_temperature,
                'max_tokens': self.openai_max_tokens,
                'timeout': self.openai_timeout_seconds,
                'api_key': self.openai_api_key
            }
        else:  # ollama
            return {
                'provider': 'ollama',
                'model': self.ollama_model,
                'base_url': self.ollama_base_url,
                'temperature': self.ollama_temperature,
                'timeout': self.ollama_timeout
            }
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on emergency overrides."""
        overrides = self.config.get('emergency_overrides', {})
        return not overrides.get('kill_switch', False)
    
    def is_paper_trading(self) -> bool:
        """Check if system is in paper trading mode."""
        overrides = self.config.get('emergency_overrides', {})
        return overrides.get('paper_only', False)


def load_config(config_path: str = "crewai_system/config/trading_config.yaml") -> TradingConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated TradingConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TradingConfig(config_dict)
    
    # Log configuration summary
    logger = logging.getLogger("ConfigLoader")
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"LLM Provider: {config.llm_provider}")
    
    if config.llm_provider == 'openai':
        logger.info(f"OpenAI Model: {config.openai_model}")
        if config.openai_api_key:
            logger.info(f"OpenAI API Key: ...{config.openai_api_key[-4:]}")
        else:
            logger.warning("OpenAI API Key not found! Set OPENAI_API_KEY environment variable")
    else:
        logger.info(f"Ollama Model: {config.ollama_model}")
        logger.info(f"Ollama URL: {config.ollama_base_url}")
    
    return config


# Global configuration instance
TRADING_CONFIG = load_config()