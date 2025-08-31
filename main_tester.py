# test_crewai.py
"""
CrewAI Trading System - Historical Testing with Real Data
==========================================================
Tests the veteran agents using your actual QQQ options and VIX data.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import random

from crewai_system.crews.trading_crew import TradingCrew
from crewai_system.config.config_loader import TRADING_CONFIG
from crewai_system.memory.persistence import TRADING_MEMORY, TradeRecord


class RealDataTestSystem:
    """Test system using your actual historical data."""
    
    def __init__(self):
        """Initialize with your real data files."""
        self.setup_logging()
        self.logger = logging.getLogger("TestSystem")
        
        # Data directories
        self.chains_dir = Path("./data/chains")
        self.market_data_dir = Path("./data/marketdata")
        self.historical_dir = Path("./data/historical_data")
        
        # Load all available data
        self.available_dates = self._scan_available_dates()
        self.vix_data = self._load_vix_data()
        self.qqq_data = self._load_qqq_data()
        
        # Current test state
        self.current_date = None
        self.current_options = None
        self.test_results = []
        
        self.logger.info(f"Found {len(self.available_dates)} days with complete data")
        
    def setup_logging(self):
        """Setup test logging."""
        log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
    
    def _scan_available_dates(self) -> List[str]:
        """Scan for dates with complete data (options + market data)."""
        available = []
        
        # Get all QQQ option files
        option_files = list(self.chains_dir.glob("QQQ_*.json"))
        
        for opt_file in option_files:
            # Extract date from filename (QQQ_20230905.json -> 2023-09-05)
            date_str = opt_file.stem.split('_')[1]  # 20230905
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Check if we have market data for this date
            market_file = self.market_data_dir / f"QQQ_{formatted_date}.json"
            if market_file.exists():
                available.append(formatted_date)
                self.logger.debug(f"Complete data for {formatted_date}")
            else:
                self.logger.debug(f"Missing market data for {formatted_date}")
        
        return sorted(available)
    
    def _load_vix_data(self) -> pd.DataFrame:
        """Load VIX historical data."""
        vix_file = self.historical_dir / "vix_history.csv"
        if vix_file.exists():
            df = pd.read_csv(vix_file, index_col='date', parse_dates=True)
            self.logger.info(f"Loaded VIX data: {len(df)} days")
            return df
        else:
            self.logger.warning("No VIX data found - will use simulated values")
            return pd.DataFrame()
    
    def _load_qqq_data(self) -> pd.DataFrame:
        """Load QQQ historical data."""
        qqq_file = self.historical_dir / "qqq_history.csv"
        if qqq_file.exists():
            df = pd.read_csv(qqq_file, index_col='date', parse_dates=True)
            self.logger.info(f"Loaded QQQ data: {len(df)} days")
            return df
        else:
            self.logger.warning("No QQQ historical data found")
            return pd.DataFrame()
    
    def get_market_data_for_date(self, date: str) -> Dict[str, Any]:
        """Get complete market data for a specific date."""
        # Get VIX for this date
        vix_value = 16.0  # Default
        if not self.vix_data.empty and date in self.vix_data.index:
            vix_value = float(self.vix_data.loc[date, 'close'])
        
        # Get QQQ data for this date
        qqq_values = {}
        if not self.qqq_data.empty and date in self.qqq_data.index:
            row = self.qqq_data.loc[date]
            qqq_values = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            }
        else:
            # Try loading from individual market data file
            market_file = self.market_data_dir / f"QQQ_{date}.json"
            if market_file.exists():
                with open(market_file) as f:
                    data = json.load(f)
                    if 'underlying' in data:
                        qqq_values = data['underlying']
        
        # Determine VIX regime and market conditions
        vix_regime = self._classify_vix_regime(vix_value)
        trend = self._determine_trend(date)
        
        return {
            "success": True,
            "date": date,
            "underlying_price": qqq_values.get('close', 375.0),
            "open": qqq_values.get('open', 375.0),
            "high": qqq_values.get('high', 376.0),
            "low": qqq_values.get('low', 374.0),
            "volume": qqq_values.get('volume', 50000000),
            "vix": vix_value,
            "vix_regime": vix_regime,
            "trend": trend,
            "timestamp": f"{date}T14:30:00"
        }
    
    def get_options_for_date(self, date: str) -> List[Dict[str, Any]]:
        """Load option chain for specific date."""
        # Convert date format: 2023-09-05 -> 20230905
        date_compact = date.replace('-', '')
        
        options_file = self.chains_dir / f"QQQ_{date_compact}.json"
        if options_file.exists():
            with open(options_file) as f:
                return json.load(f)
        return []
    
    def _classify_vix_regime(self, vix: float) -> str:
        """Classify VIX into regime."""
        if vix < 15:
            return "very_low"
        elif vix < 20:
            return "low"
        elif vix < 25:
            return "normal"
        elif vix < 30:
            return "elevated"
        else:
            return "high"
    
    def _determine_trend(self, date: str) -> str:
        """Determine market trend around date."""
        if self.qqq_data.empty:
            return "neutral"
        
        try:
            # Get 20-day SMA
            current_idx = self.qqq_data.index.get_loc(date)
            if current_idx >= 20:
                sma20 = self.qqq_data['close'].iloc[current_idx-20:current_idx].mean()
                current_price = self.qqq_data.loc[date, 'close']
                
                if current_price > sma20 * 1.02:
                    return "bullish"
                elif current_price < sma20 * 0.98:
                    return "bearish"
        except:
            pass
        
        return "neutral"
    
    async def initialize_test_system(self):
        """Initialize the trading crew and mock connections."""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING TEST SYSTEM WITH REAL DATA")
        self.logger.info(f"Test dates available: {len(self.available_dates)}")
        self.logger.info(f"Date range: {self.available_dates[0]} to {self.available_dates[-1]}")
        self.logger.info("=" * 80)
        
        # Mock the market tools to use our data
        self._mock_market_tools()
        
        # Initialize trading crew
        self.crew = TradingCrew()
        
        self.logger.info("✅ Test system ready")
        return True
    
    def _mock_market_tools(self):
        """Mock market tools to use our historical data."""
        import crewai_system.tools.market_tools as market_tools
        
        # Create mock functions that return our data
        def mock_get_market_data():
            if self.current_date:
                return json.dumps(self.get_market_data_for_date(self.current_date))
            return json.dumps({"success": False, "error": "No date set"})
        
        def mock_get_option_chain():
            if self.current_date:
                options = self.get_options_for_date(self.current_date)
                market_data = self.get_market_data_for_date(self.current_date)
                return json.dumps({
                    "success": True,
                    "options": options,
                    "underlying_price": market_data["underlying_price"]
                })
            return json.dumps({"success": False, "error": "No date set"})
        
        def mock_execute_trade(trade_spec):
            # Simulate trade execution
            trade = json.loads(trade_spec)
            
            # 90% success rate
            if random.random() < 0.9:
                return json.dumps({
                    "success": True,
                    "order_id": f"TEST_{self.current_date}_{datetime.now().timestamp()}",
                    "filled": True,
                    "avg_fill_price": trade.get("limit_price", 1.0),
                    "commission": 2.60
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": random.choice(["Insufficient margin", "Invalid strikes"])
                })
        
        # Replace the functions
        market_tools.get_real_market_data = mock_get_market_data
        market_tools.get_real_option_chain = mock_get_option_chain
        market_tools.execute_real_trade = mock_execute_trade
        
        # Mock execution engine
        class MockEngine:
            connected = True
            async def connect(self, paper=True): return True
            async def disconnect(self): pass
        
        market_tools._execution_engine = MockEngine()
    
    async def run_historical_tests(self, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  sample_times: List[str] = None):
        """
        Run tests on historical data.
        
        Args:
            start_date: Start date (uses first available if None)
            end_date: End date (uses last available if None)
            sample_times: Times to test each day
        """
        if sample_times is None:
            sample_times = ["09:45", "10:30", "14:00", "15:30"]
        
        # Filter dates
        test_dates = self.available_dates
        if start_date:
            test_dates = [d for d in test_dates if d >= start_date]
        if end_date:
            test_dates = [d for d in test_dates if d <= end_date]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"RUNNING TESTS ON {len(test_dates)} DAYS")
        self.logger.info(f"{'='*80}\n")
        
        for date in test_dates:
            self.current_date = date
            market_data = self.get_market_data_for_date(date)
            
            self.logger.info(f"\n{'📅'*20}")
            self.logger.info(f"Testing Date: {date}")
            self.logger.info(f"VIX: {market_data['vix']:.2f} ({market_data['vix_regime']})")
            self.logger.info(f"QQQ: ${market_data['underlying_price']:.2f}")
            self.logger.info(f"Trend: {market_data['trend']}")
            self.logger.info(f"\n{'📅'*20}\n")
            
            for time in sample_times:
                self.logger.info(f"\n⏰ Testing at {date} {time}")
                
                try:
                    # Run trading decision
                    result = self.crew.analyze_and_trade()
                    
                    # Store result
                    self.test_results.append({
                        "date": date,
                        "time": time,
                        "vix": market_data['vix'],
                        "vix_regime": market_data['vix_regime'],
                        "underlying": market_data['underlying_price'],
                        "trend": market_data['trend'],
                        "decision": result["decision"],
                        "details": result.get("details", {})
                    })
                    
                    self.logger.info(f"Decision: {result['decision']}")
                    
                    # Simulate trade outcome if executed
                    if result["decision"] == "execute":
                        await self._simulate_trade_outcome(result["details"], market_data)
                    
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                await asyncio.sleep(0.5)  # Small delay
        
        # Print comprehensive summary
        self._print_test_summary()
    
    async def _simulate_trade_outcome(self, trade_details: Dict, market_data: Dict):
        """Simulate realistic trade outcome based on market conditions."""
        # Use VIX regime to influence outcomes
        vix = market_data['vix']
        vix_regime = market_data['vix_regime']
        
        # Realistic outcome probabilities based on VIX
        if vix_regime == "very_low":
            # Low vol = higher win rate for credit strategies
            outcomes = [
                (45, True, "Target reached"),
                (40, True, "Partial profit"),
                (35, True, "Early exit"),
                (-30, False, "Stop hit"),
            ]
            weights = [0.4, 0.3, 0.2, 0.1]
        elif vix_regime in ["low", "normal"]:
            outcomes = [
                (50, True, "Target reached"),
                (25, True, "Partial profit"),
                (-25, False, "Stop hit"),
                (-50, False, "Max loss"),
            ]
            weights = [0.35, 0.35, 0.2, 0.1]
        else:  # elevated/high
            outcomes = [
                (75, True, "Volatility profit"),
                (-50, False, "Whipsaw"),
                (-100, False, "Gap move"),
                (0, False, "Scratched"),
            ]
            weights = [0.3, 0.3, 0.2, 0.2]
        
        # Select outcome based on weights
        outcome_idx = np.random.choice(len(outcomes), p=weights)
        pnl, win, reason = outcomes[outcome_idx]
        
        # Store in memory for learning
        trade = TradeRecord(
            timestamp=f"{self.current_date}T{datetime.now().strftime('%H:%M:%S')}",
            strategy=trade_details.get("strategy", {}).get("strategy_type", "unknown"),
            vix_level=vix,
            vix_regime=vix_regime,
            market_regime=market_data['trend'],
            trend=market_data['trend'],
            contracts=trade_details.get("execution_params", {}).get("final_contracts", 1),
            credit_received=100,
            max_risk=500,
            realized_pnl=pnl,
            win=win,
            exit_reason=reason,
            pattern=f"{vix_regime}_{market_data['trend']}",
            confidence=trade_details.get("confidence", 50)
        )
        
        TRADING_MEMORY.store_trade(trade)
        self.logger.info(f"Outcome: {reason} - P&L: ${pnl}")
    
    def _print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total = len(self.test_results)
        if total == 0:
            print("No test results")
            return
        
        # Decision breakdown
        decisions = pd.DataFrame(self.test_results)
        decision_counts = decisions['decision'].value_counts()
        
        print(f"\nTotal Decisions: {total}")
        for decision, count in decision_counts.items():
            print(f"  {decision}: {count} ({count/total:.1%})")
        
        # VIX regime analysis
        print("\nDecisions by VIX Regime:")
        regime_decisions = decisions.groupby('vix_regime')['decision'].value_counts()
        for (regime, decision), count in regime_decisions.items():
            regime_total = len(decisions[decisions['vix_regime'] == regime])
            print(f"  {regime} - {decision}: {count} ({count/regime_total:.1%})")
        
        # Performance stats from memory
        stats = TRADING_MEMORY.get_performance_stats(days=1)
        print(f"\nSimulated Performance:")
        print(f"  Trades Executed: {stats.get('total_trades', 0)}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"  Total P&L: ${stats.get('total_pnl', 0):.2f}")
        print(f"  Avg P&L: ${stats.get('avg_pnl', 0):.2f}")
        
        print("\n" + "="*80)


async def main():
    """Run the test system."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     CrewAI Trading System - Real Data Testing            ║
    ║     Using Your QQQ Options and VIX Historical Data       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Create test system
    test_system = RealDataTestSystem()
    
    # Show available dates
    print(f"\n📊 Found {len(test_system.available_dates)} days with complete data")
    print(f"📅 Date range: {test_system.available_dates[0]} to {test_system.available_dates[-1]}")
    
    # Initialize
    if not await test_system.initialize_test_system():
        print("\n❌ Initialization failed")
        return
    
    # Run tests (you can specify date range)
    await test_system.run_historical_tests(
        start_date="2023-09-05",  # Or None for all
        end_date="2025-03-04",     # Or None for all
        sample_times=["09:45", "14:00", "15:30"]  # Key times
    )
    
    print("\n✅ Testing complete! Check the log file for details.")


if __name__ == "__main__":
    asyncio.run(main())