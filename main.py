# main_crewai.py
"""
CrewAI Trading System - Main Entry Point
=========================================
Run this file to start your AI-powered trading system.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import signal
from dotenv import load_dotenv
# import yaml

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from crewai_system.crews.trading_crew import TradingCrew
from crewai_system.config.timezone_handler import TIMEZONE_HANDLER
from crewai_system.config.config_loader import TRADING_CONFIG
from crewai_system.memory.persistence import TRADING_MEMORY
from crewai_system.tools.base_tools import initialize_tool_infrastructure

class CrewAITradingSystem:
    """
    Main orchestrator for the CrewAI trading system.
    """
    
    def __init__(self):
        """Initialize the trading system."""
        self.setup_logging()
        self.logger = logging.getLogger("CrewAISystem")
        self.running = False
        self.crew = None
        
    def setup_logging(self):
        """Configure logging for the entire system."""
        log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"crewai_system/logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if successful
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("CREWAI TRADING SYSTEM INITIALIZATION")
            self.logger.info("=" * 80)
            
            # Step 1: Initialize tool infrastructure (your existing modules)
            self.logger.info("Initializing tool infrastructure...")
            if not initialize_tool_infrastructure():
                raise RuntimeError("Failed to initialize tools")
            
            # Step 2: Connect to Interactive Brokers
            self.logger.info("Connecting to Interactive Brokers...")
            from crewai_system.tools.market_tools import _execution_engine
            await _execution_engine.connect(paper=True)
            
            # Step 3: Initialize the trading crew
            self.logger.info("Assembling trading crew...")
            self.crew = TradingCrew()
            
            # Step 4: Verify Ollama is running
            # self.logger.info("Verifying Ollama connection...")
            # from crewai_system.config.ollama_config import OLLAMA_CONFIG
            # OLLAMA_CONFIG.validate_connection()
            
            self.logger.info("✅ System initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def run_trading_session(self):
        """
        Run the main trading loop.
        """
        self.running = True
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING TRADING SESSION")
        self.logger.info(TIMEZONE_HANDLER.format_times_for_logging())
        self.logger.info("=" * 80 + "\n")
        
        # Pre-market preparation phase
        minutes_to_open = TIMEZONE_HANDLER.get_minutes_to_market_open()
        if minutes_to_open > 0 and minutes_to_open <= 15:
            self.logger.info(f"📅 Pre-market preparation: {minutes_to_open} minutes until market open")
            self.logger.info("✅ System initialized and ready")
            self.logger.info("⏳ Waiting for market open...")
            print(f"\n📅 Pre-market: {minutes_to_open} min to open. System ready, waiting...")
        
        try:
            while self.running:
                # Check if market is open
                if not TIMEZONE_HANDLER.is_market_open():
                    from datetime import datetime
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    minutes_to_open = TIMEZONE_HANDLER.get_minutes_to_market_open()
                    
                    if minutes_to_open > 0 and minutes_to_open <= 15:
                        # Pre-market, check every 30 seconds
                        self.logger.info(f"Pre-market: {minutes_to_open} min to open. Checking in 30 seconds...")
                        await asyncio.sleep(30)
                    else:
                        # Market closed, check every 5 minutes
                        self.logger.info(f"Market is closed at {now_str}. Waiting 5 minutes...")
                        print(f"Market is closed at {now_str}. Waiting 5 minutes...")
                        await asyncio.sleep(300)
                    continue
                
                # Get current market phase and interval
                phase, interval = TIMEZONE_HANDLER.get_market_phase()
                
                self.logger.info(f"\n{'🔔' * 20}")
                self.logger.info(f"Market Phase: {phase} | Next analysis in {interval} seconds")
                self.logger.info(TIMEZONE_HANDLER.format_times_for_logging())
                
                # Run trading decision workflow
                result = self.crew.analyze_and_trade()
                
                self.logger.info(f"Decision: {result['decision']}")
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n⛔ Session interrupted by user")
        except Exception as e:
            self.logger.error(f"Session error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("\n🧹 Cleaning up...")
        
        # Disconnect from IB
        from crewai_system.tools.market_tools import _execution_engine
        if _execution_engine.connected:
            await _execution_engine.disconnect()
        
        self.running = False
        self.logger.info("✅ Cleanup complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("\nShutdown signal received")
        self.running = False


async def main():
    """Main entry point."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Quantum View AI 0DTE Options Trading System      ║
    ║         Spain → US Markets | Paper Trading               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    load_dotenv(override=True)
    # Create and initialize system
    system = CrewAITradingSystem()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    # Initialize
    if not await system.initialize():
        print("\n❌ Failed to initialize system")
        return
    
    # Run trading session
    await system.run_trading_session()


if __name__ == "__main__":
    asyncio.run(main())