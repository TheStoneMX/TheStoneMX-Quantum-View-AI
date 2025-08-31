# crewai_system/config/ollama_config.py
"""
Ollama Configuration for Local LLM Integration
==============================================
Configures qwen2.5-coder:32b for ultra-low latency trading decisions.
Using local LLM eliminates API rate limits and reduces decision latency.

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging


# crewai_system/config/ollama_config.py
"""
Ollama Configuration for Local LLM Integration
==============================================
Optimized for M1 Max with 64GB unified memory running Mixtral.
Parameters tuned specifically for 0DTE options trading decisions.

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging


@dataclass
class OllamaConfig:
    """
    Configuration for Ollama LLM integration.
    
    Optimized for:
    - M1 Max with 64GB unified memory
    - Mixtral 8x7b model (q5_K_M quantization)
    - 0DTE trading requiring fast, consistent decisions
    """
    
    # Model configuration
    model_name: str = "mixtral:8x7b-instruct-v0.1-q5_K_M"
    base_url: str = "http://localhost:11434"
    
    # Response configuration for trading decisions
    temperature: float = 0.3          # Low for consistency, not zero for adaptability
    top_p: float = 0.9               # Focus without being too narrow
    top_k: int = 40                  # Standard diversity
    repeat_penalty: float = 1.1      # Prevent repetitive analysis
    seed: int = 42                   # Fixed for reproducibility during paper trading
    
    # Token and context settings
    max_tokens: int = 2048           # num_predict in Ollama terms
    context_window: int = 8192       # Mixtral's sweet spot for speed/capability
    
    # M1 Max optimization parameters
    num_batch: int = 512             # Optimized for Apple Silicon
    num_thread: int = 8              # 8 of 10 cores (leave 2 for system/IB)
    num_gpu: int = 999               # Use all GPU layers on M1
    main_gpu: int = 0                # Single GPU on M1
    low_vram: bool = False           # Plenty of unified memory
    f16_kv: bool = True             # 16-bit KV cache for M1 efficiency
    
    # Response format
    response_format: str = "json"    # Enforce structured responses
    
    # Timeout settings critical for 0DTE
    timeout_seconds: int = 10        # Faster timeout for 0DTE
    retry_attempts: int = 2          # Quick retries
    
    # Trading-specific parameters
    history_to_include: int = 10     # Recent trades for context
    reasoning_detail: str = "significant"  # Balance detail vs speed
    include_confidence: bool = True   # Critical for risk assessment
    
    # Time-pressure adjustments
    use_adaptive_params: bool = True  # Adjust based on time to market close
    
    def to_langchain_config(self) -> Dict[str, Any]:
        """
        Convert to LangChain-compatible configuration.
        
        Returns:
            Dictionary with LangChain Ollama parameters
        """
        return {
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
            "num_predict": self.max_tokens,
            "num_ctx": self.context_window,
            "num_batch": self.num_batch,
            "num_thread": self.num_thread,
            "num_gpu": self.num_gpu,
            "main_gpu": self.main_gpu,
            "low_vram": self.low_vram,
            "f16_kv": self.f16_kv,
            "format": self.response_format,
            "timeout": self.timeout_seconds,
        }
    
    def get_adaptive_config(self, minutes_to_close: float) -> Dict[str, Any]:
        """
        Adjust parameters based on time pressure for 0DTE.
        
        Args:
            minutes_to_close: Minutes until market close
            
        Returns:
            Adapted configuration for current time pressure
        """
        if not self.use_adaptive_params:
            return self.to_langchain_config()
        
        config = self.to_langchain_config().copy()
        
        if minutes_to_close < 30:  # Final 30 minutes - maximum speed
            config.update({
                "temperature": 0.1,      # Very deterministic
                "num_predict": 1024,     # Shorter responses
                "timeout": 5,            # Fast timeout
                "top_p": 0.95,          # Slightly more focused
            })
            logging.info(f"Using FAST config: {minutes_to_close:.0f} min to close")
            
        elif minutes_to_close < 60:  # Final hour - balanced
            config.update({
                "temperature": 0.2,
                "num_predict": 1536,
                "timeout": 7,
            })
            logging.info(f"Using BALANCED config: {minutes_to_close:.0f} min to close")
            
        else:  # Normal trading - full analysis
            logging.info(f"Using NORMAL config: {minutes_to_close:.0f} min to close")
        
        return config
    
    def validate_connection(self) -> bool:
        """
        Verify Ollama is running and model is available.
        
        Returns:
            True if Ollama is accessible and model is loaded
            
        Raises:
            ConnectionError: If Ollama is not running
            ValueError: If model is not available
        """
        import requests
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {self.base_url}")
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if self.model_name not in model_names:
                # Try without the tag
                base_model = self.model_name.split(":")[0]
                if not any(base_model in name for name in model_names):
                    # Provide helpful error message
                    raise ValueError(
                        f"Model {self.model_name} not found.\n"
                        f"Please run: ollama pull {self.model_name}\n"
                        f"Available models: {model_names}"
                    )
            
            logging.info(f"✅ Ollama connected with {self.model_name}")
            return True
            
        except requests.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                f"Please ensure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )
    
    def estimate_response_time(self) -> Dict[str, float]:
        """
        Estimate response times on M1 Max.
        
        Returns:
            Dictionary with estimated timings
        """
        return {
            "first_token_ms": 2000,  # ~2 seconds
            "tokens_per_second": 18,  # Conservative estimate
            "full_response_seconds": self.max_tokens / 18 + 2,
            "timeout": self.timeout_seconds
        }


# Singleton instance for consistent configuration across agents
OLLAMA_CONFIG = OllamaConfig()

# Log configuration on import
logging.info(f"Ollama configured for {OLLAMA_CONFIG.model_name}")
logging.info(f"Expected response time: {OLLAMA_CONFIG.estimate_response_time()['full_response_seconds']:.1f}s")