# crewai_system/config/llm_factory.py
"""
LLM Factory
===========
Creates the appropriate LLM instance based on configuration.
Supports both OpenAI and Ollama providers.

Author: CrewAI Trading System
Version: 2.0
Date: December 2024
"""

import os
import logging
from typing import Optional, Any
from crewai import LLM
from dotenv import load_dotenv

import litellm

class LLMFactory:
    """
    Factory class for creating LLM instances.
    
    Automatically detects which provider is configured
    and creates the appropriate LLM instance.
    """
    load_dotenv(override=True)
    
    @staticmethod
    def make(config: Any, timeout_key: Optional[str] = None) -> LLM:
        """
        Create an LLM instance based on configuration.
        
        Args:
            config: TradingConfig instance
            timeout_key: Optional timeout override key (e.g., 'market_analysis')
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If no LLM provider is configured
        """
        logger = logging.getLogger("LLMFactory")
        
        # Get timeout if specified
        timeout = 45  # Default timeout
        if timeout_key:
            timeout = config.decision_timeouts.get(timeout_key)
        
        # Create LLM based on provider
        if config.has_openai:
            # Use OpenAI
            api_key = config.openai_api_key
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            llm = LLM(
                model=config.openai_model,
                temperature=config.openai_temperature,
                max_tokens=config.openai_max_tokens,
                api_key=api_key,
                timeout=timeout or config.openai_timeout_seconds,
                response_format={"type": "json_object"} 
            )
            
            logger.debug(f"Created OpenAI LLM: {config.openai_model}")
            
        elif config.has_ollama:
            # Use Ollama
            llm = LLM(
                model=f"ollama/{config.ollama_model}",
                base_url=config.ollama_base_url,
                temperature=config.ollama_temperature,
                timeout=timeout or config.ollama_timeout
            )
            
            logger.debug(f"Created Ollama LLM: {config.ollama_model}")
            
        else:
            raise ValueError("No LLM provider configured. Add 'openai' or 'ollama' to config.")
        
        return llm
    
    @staticmethod
    def create_fallback_llm(config: Any) -> LLM:
        """
        Create a fallback LLM for error scenarios.
        
        Uses a simpler model with longer timeout.
        
        Args:
            config: TradingConfig instance
            
        Returns:
            Fallback LLM instance
        """
        if config.has_openai:
            # Use cheaper/faster model as fallback
            return LLM(
                model="gpt-3.5-turbo",
                temperature=0.1,  # Lower temp for consistency
                max_tokens=500,    # Shorter responses
                api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY"),
                timeout=60        # Longer timeout
            )
        else:
            # Use Ollama with simple model
            return LLM(
                model="ollama/phi3:mini",  # Tiny fast model
                base_url=config.ollama_base_url,
                temperature=0.1,
                timeout=60
            )
    
    @staticmethod
    def test_connection(config: Any) -> bool:
        """
        Test if LLM connection works.
        
        Args:
            config: TradingConfig instance
            
        Returns:
            True if connection works, False otherwise
        """
        logger = logging.getLogger("LLMFactory")
        
        try:
            llm = LLMFactory.make(config)
            response = llm.invoke("Say 'OK' and nothing else.")
            
            if response and ('OK' in str(response).upper()):
                logger.info("LLM connection test successful")
                return True
            else:
                logger.warning(f"LLM test returned unexpected response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False