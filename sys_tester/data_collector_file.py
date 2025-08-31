#!/usr/bin/env python3
"""
Market Data Collector for 0DTE Trading System.

This module collects and saves market data from Interactive Brokers during market hours.
The saved data can be used for backtesting and strategy development when markets are closed.

Features:
    - Downloads historical price data for NQ futures
    - Captures complete option chains with Greeks
    - Saves VIX data and market statistics
    - Stores account information for position sizing
    - Organizes data by date and timestamp

Usage:
    Run during market hours to collect data:
    $ python data_collector.py --days 60 --interval 5

Author: Trading Systems
Version: 1.0.0
Date: December 2024
"""

import asyncio
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
import pytz
from ib_insync import FuturesOption, IB, Future, Index, Stock, util

# Use existing models from your system
from config import get_config
from models import Greeks, OptionContract


@dataclass
class MarketSnapshot:
    """
    Complete market snapshot at a point in time.

    Attributes:
        timestamp: When the snapshot was taken
        underlying_price: Current NQ futures price
        underlying_data: Complete futures contract data
        option_chain: List of all option contracts with Greeks
        vix_data: VIX index data
        market_stats: Additional market statistics
        account_data: Account snapshot for position sizing
    """

    timestamp: datetime
    underlying_price: float
    underlying_data: Dict[str, Any]
    option_chain: List[OptionContract]
    vix_data: Dict[str, float]
    market_stats: Dict[str, Any]
    account_data: Dict[str, float]


class DataCollector:
    """
    Collects and saves market data from Interactive Brokers.

    This class handles all data collection operations including:
    - Connecting to IB TWS/Gateway
    - Downloading historical data
    - Capturing real-time snapshots
    - Saving data in organized format
    """

    def __init__(
        self,
        data_dir: str = "market_data",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the data collector.

        Args:
            data_dir: Directory to save collected data
            logger: Optional logger instance, creates one if not provided
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.historical_dir = self.data_dir / "historical"
        self.historical_dir.mkdir(exist_ok=True)
        
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        self.options_dir = self.data_dir / "options"
        self.options_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logger or self._setup_logger()
        
        # Load configuration
        self.config = get_config()
        
        # IB connection
        self.ib: Optional[IB] = None
        self.connected = False
        
        # Timezone setup
        self.et_tz = pytz.timezone("US/Eastern")
        self.utc_tz = pytz.UTC
        
        # Track collection statistics
        self.stats = {
            "snapshots_collected": 0,
            "options_collected": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }
        
        self.logger.info(f"Data Collector initialized. Data directory: {self.data_dir}")

    def _setup_logger(self) -> logging.Logger:
        """
        Create and configure logger for data collection.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("DataCollector")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler with detailed formatting
        console = logging.StreamHandler()
        console_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console.setFormatter(console_format)
        logger.addHandler(console)
        
        # File handler for permanent record
        log_file = self.data_dir / f"collection_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        
        return logger

    async def connect(self, paper: bool = True) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway.

        Args:
            paper: Use paper trading connection (default True)

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ib = IB()
            
            # Select port based on paper/live trading
            port = (
                self.config["broker"].paper_port
                if paper
                else self.config["broker"].live_port
            )
            
            self.logger.info(
                f"Connecting to IB at {self.config['broker'].host}:{port} "
                f"({'Paper' if paper else 'LIVE'})"
            )
            
            # Connect synchronously (ib_insync handles async internally)
            self.ib.connect(
                self.config["broker"].host,
                port,
                clientId=self.config["broker"].client_id + 100,  # Different ID for collector
            )
            
            self.connected = True
            self.logger.info("✅ Connected to Interactive Brokers")
            
            # Request market data type (delayed is fine for collection)
            self.ib.reqMarketDataType(3)  # Delayed data
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib and self.connected:
            self.logger.info("Disconnecting from IB...")
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected")

    async def collect_historical_data(
        self,
        symbol: str = "NQ",
        days: int = 60,
    ) -> pd.DataFrame:
        """
        Collect historical price data for the underlying.

        Args:
            symbol: Futures symbol to collect (default NQ)
            days: Number of days of historical data to collect

        Returns:
            DataFrame with historical OHLCV data

        Raises:
            Exception: If data collection fails
        """
        self.logger.info(f"Collecting {days} days of historical data for {symbol}")
        
        try:
            # Create continuous future contract
            contract = Future(symbol, exchange="CME", includeExpired=False)
            
            # Qualify the contract
            self.ib.qualifyContracts(contract)
            
            # Calculate date range
            end_date = datetime.now()
            duration = f"{days} D"
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,  # Regular trading hours only
                formatDate=1,
            )
            
            if not bars:
                raise Exception("No historical data received")
            
            # Convert to DataFrame
            df = util.df(bars)
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Save to CSV
            filename = self.historical_dir / f"{symbol}_daily_{datetime.now():%Y%m%d}.csv"
            df.to_csv(filename, index=False)
            
            self.logger.info(
                f"✅ Saved {len(df)} days of historical data to {filename}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to collect historical data: {e}")
            self.stats["errors"] += 1
            raise

    async def collect_intraday_data(
        self,
        symbol: str = "NQ",
        interval: int = 5,
    ) -> pd.DataFrame:
        """
        Collect intraday bar data for the current trading day.

        Args:
            symbol: Futures symbol to collect
            interval: Bar size in minutes

        Returns:
            DataFrame with intraday bars
        """
        self.logger.info(f"Collecting intraday {interval}-minute bars for {symbol}")
        
        try:
            # Create future contract
            contract = Future(symbol, exchange="CME")
            self.ib.qualifyContracts(contract)
            
            # Request today's data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting=f"{interval} mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            
            if not bars:
                self.logger.warning("No intraday data received")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = util.df(bars)
            df["symbol"] = symbol
            
            # Save to CSV
            filename = (
                self.historical_dir / 
                f"{symbol}_intraday_{interval}min_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )
            df.to_csv(filename, index=False)
            
            self.logger.info(f"✅ Saved {len(df)} intraday bars to {filename}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to collect intraday data: {e}")
            self.stats["errors"] += 1
            return pd.DataFrame()

    async def collect_option_chain(
        self,
        symbol: str = "NQ",
        expiry: Optional[str] = None,
    ) -> List[OptionContract]:
        """
        Collect complete option chain with Greeks for given expiry.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD), None for 0DTE

        Returns:
            List of OptionContract objects with full Greeks
        """
        # Use today for 0DTE if not specified
        if not expiry:
            expiry = datetime.now().strftime("%Y%m%d")
        
        self.logger.info(f"Collecting option chain for {symbol} expiry {expiry}")
        
        option_contracts = []
        
        try:
            # Get current underlying price first
            future = Future(symbol, exchange="CME")
            self.ib.qualifyContracts(future)
            
            # Request market data for underlying
            ticker = self.ib.reqMktData(future, "", False, False)
            self.ib.sleep(2)  # Wait for data
            
            current_price = ticker.last or ticker.close
            
            if not current_price:
                self.logger.error("Could not get underlying price")
                return []
            
            self.logger.info(f"Current {symbol} price: ${current_price:,.2f}")
            
            # Cancel market data for underlying
            self.ib.cancelMktData(future)
            
            # Generate strike range (wider for complete data)
            strike_increment = 25  # NQ strikes are every 25 points
            min_strike = int((current_price - 1000) / strike_increment) * strike_increment
            max_strike = int((current_price + 1000) / strike_increment) * strike_increment
            strikes = list(range(min_strike, max_strike + strike_increment, strike_increment))
            
            self.logger.info(f"Collecting {len(strikes) * 2} option contracts...")
            
            # Create option contracts
            contracts = []
            for strike in strikes:
                for right in ["C", "P"]:
                    contract = FuturesOption(
                        symbol=symbol,
                        lastTradeDateOrContractMonth=expiry,
                        strike=strike,
                        right=right,
                        exchange="CME",
                    )
                    contracts.append(contract)
            
            # Qualify contracts in batches
            batch_size = 100
            qualified_contracts = []
            
            for i in range(0, len(contracts), batch_size):
                batch = contracts[i : i + batch_size]
                qualified_batch = self.ib.qualifyContracts(*batch)
                qualified_contracts.extend(
                    [c for c in qualified_batch if c.conId > 0]
                )
                self.logger.info(
                    f"Qualified batch {i // batch_size + 1}/{len(contracts) // batch_size + 1}"
                )
            
            # Request market data with Greeks for all contracts
            tickers = []
            for contract in qualified_contracts:
                ticker = self.ib.reqMktData(
                    contract,
                    genericTickList="106",  # Request Greeks
                    snapshot=False,
                    regulatorySnapshot=False,
                )
                tickers.append(ticker)
            
            # Wait for data to populate
            self.logger.info("Waiting for market data and Greeks...")
            self.ib.sleep(10)  # Give time for all data to arrive
            
            # Extract data from tickers
            valid_count = 0
            for ticker in tickers:
                # Only save options with valid bid/ask
                if ticker.bid and ticker.ask and ticker.bid > 0:
                    # Create Greeks object if available
                    greeks = None
                    if ticker.modelGreeks:
                        greeks = Greeks(
                            delta=ticker.modelGreeks.delta or 0,
                            gamma=ticker.modelGreeks.gamma or 0,
                            theta=ticker.modelGreeks.theta or 0,
                            vega=ticker.modelGreeks.vega or 0,
                            iv=ticker.modelGreeks.impliedVol or 0,
                        )
                    
                    # Create OptionContract
                    opt = OptionContract(
                        strike=ticker.contract.strike,
                        right=ticker.contract.right,
                        expiry=expiry,
                        bid=ticker.bid,
                        ask=ticker.ask,
                        mid=(ticker.bid + ticker.ask) / 2,
                        volume=ticker.volume or 0,
                        open_interest=0,  # Would need separate request
                        greeks=greeks,
                    )
                    option_contracts.append(opt)
                    valid_count += 1
            
            # Cancel all market data requests
            for ticker in tickers:
                self.ib.cancelMktData(ticker.contract)
            
            self.logger.info(
                f"✅ Collected {valid_count} valid option contracts out of {len(tickers)} total"
            )
            
            # Save option chain
            filename = self.options_dir / f"{symbol}_options_{expiry}_{datetime.now():%H%M%S}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(option_contracts, f)
            
            self.logger.info(f"✅ Saved option chain to {filename}")
            
            self.stats["options_collected"] += valid_count
            
            return option_contracts
            
        except Exception as e:
            self.logger.error(f"Failed to collect option chain: {e}")
            self.stats["errors"] += 1
            return []

    async def collect_vix_data(self) -> Dict[str, float]:
        """
        Collect current VIX index data.

        Returns:
            Dictionary with VIX data including price, high, low, close
        """
        self.logger.info("Collecting VIX data")
        
        try:
            # Create VIX index contract
            vix = Index("VIX", exchange="CBOE")
            self.ib.qualifyContracts(vix)
            
            # Request market data
            ticker = self.ib.reqMktData(vix, "", False, False)
            self.ib.sleep(3)
            
            vix_data = {
                "last": ticker.last or 0,
                "high": ticker.high or 0,
                "low": ticker.low or 0,
                "close": ticker.close or 0,
                "bid": ticker.bid or 0,
                "ask": ticker.ask or 0,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Cancel market data
            self.ib.cancelMktData(vix)
            
            self.logger.info(f"✅ VIX collected: {vix_data['last']:.2f}")
            
            return vix_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect VIX data: {e}")
            self.stats["errors"] += 1
            return {"last": 16.0, "high": 16.0, "low": 16.0, "close": 16.0}

    async def collect_account_snapshot(self) -> Dict[str, float]:
        """
        Collect account information for position sizing calculations.

        Returns:
            Dictionary with account metrics
        """
        self.logger.info("Collecting account snapshot")
        
        try:
            # Get account summary
            account_values = self.ib.accountSummary()
            
            snapshot = {}
            for av in account_values:
                if av.tag in [
                    "NetLiquidation",
                    "AvailableFunds",
                    "BuyingPower",
                    "UnrealizedPnL",
                    "RealizedPnL",
                    "GrossPositionValue",
                ]:
                    snapshot[av.tag] = float(av.value)
            
            # Add timestamp
            snapshot["timestamp"] = datetime.now().isoformat()
            
            self.logger.info(f"✅ Account snapshot collected: ${snapshot.get('NetLiquidation', 0):,.2f}")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to collect account data: {e}")
            self.stats["errors"] += 1
            # Return default values for testing
            return {
                "NetLiquidation": 15000.0,
                "AvailableFunds": 15000.0,
                "BuyingPower": 60000.0,
                "UnrealizedPnL": 0.0,
                "RealizedPnL": 0.0,
            }

    async def collect_market_snapshot(self) -> MarketSnapshot:
        """
        Collect complete market snapshot including all data.

        This is the main collection method that gathers everything
        needed for a complete market state.

        Returns:
            MarketSnapshot object with all current data
        """
        self.logger.info("=" * 60)
        self.logger.info("Collecting complete market snapshot")
        
        timestamp = datetime.now()
        
        # Get underlying data
        self.logger.info("1. Getting underlying futures data...")
        future = Future("NQ", exchange="CME")
        self.ib.qualifyContracts(future)
        
        ticker = self.ib.reqMktData(future, "", False, False)
        self.ib.sleep(3)
        
        underlying_price = ticker.last or ticker.close or 0
        underlying_data = {
            "price": underlying_price,
            "bid": ticker.bid or 0,
            "ask": ticker.ask or 0,
            "high": ticker.high or 0,
            "low": ticker.low or 0,
            "volume": ticker.volume or 0,
            "close": ticker.close or 0,
        }
        
        self.ib.cancelMktData(future)
        
        # Get option chain
        self.logger.info("2. Collecting option chain...")
        option_chain = await self.collect_option_chain()
        
        # Get VIX data
        self.logger.info("3. Collecting VIX data...")
        vix_data = await self.collect_vix_data()
        
        # Get account data
        self.logger.info("4. Collecting account data...")
        account_data = await self.collect_account_snapshot()
        
        # Calculate market statistics
        market_stats = {
            "options_collected": len(option_chain),
            "atm_iv": self._calculate_atm_iv(option_chain, underlying_price),
            "put_call_ratio": self._calculate_put_call_ratio(option_chain),
            "max_pain": self._estimate_max_pain(option_chain),
        }
        
        # Create snapshot
        snapshot = MarketSnapshot(
            timestamp=timestamp,
            underlying_price=underlying_price,
            underlying_data=underlying_data,
            option_chain=option_chain,
            vix_data=vix_data,
            market_stats=market_stats,
            account_data=account_data,
        )
        
        # Save snapshot
        filename = (
            self.snapshots_dir / 
            f"snapshot_{timestamp:%Y%m%d_%H%M%S}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(snapshot, f)
        
        self.logger.info(f"✅ Market snapshot saved to {filename}")
        self.logger.info("=" * 60)
        
        self.stats["snapshots_collected"] += 1
        
        return snapshot

    def _calculate_atm_iv(
        self,
        option_chain: List[OptionContract],
        underlying_price: float,
    ) -> float:
        """
        Calculate at-the-money implied volatility.

        Args:
            option_chain: List of option contracts
            underlying_price: Current underlying price

        Returns:
            Average ATM implied volatility
        """
        if not option_chain or not underlying_price:
            return 0.0
        
        # Find ATM options
        atm_options = sorted(
            option_chain,
            key=lambda x: abs(x.strike - underlying_price),
        )[:4]  # Get 2 closest calls and 2 closest puts
        
        # Average their IVs
        ivs = [
            opt.greeks.iv
            for opt in atm_options
            if opt.greeks and opt.greeks.iv > 0
        ]
        
        return sum(ivs) / len(ivs) if ivs else 0.0

    def _calculate_put_call_ratio(
        self,
        option_chain: List[OptionContract],
    ) -> float:
        """
        Calculate put/call volume ratio.

        Args:
            option_chain: List of option contracts

        Returns:
            Put/call ratio
        """
        put_volume = sum(opt.volume for opt in option_chain if opt.right == "P")
        call_volume = sum(opt.volume for opt in option_chain if opt.right == "C")
        
        return put_volume / call_volume if call_volume > 0 else 1.0

    def _estimate_max_pain(
        self,
        option_chain: List[OptionContract],
    ) -> float:
        """
        Estimate max pain strike price.

        This is a simplified calculation - full max pain
        would require open interest data.

        Args:
            option_chain: List of option contracts

        Returns:
            Estimated max pain strike
        """
        if not option_chain:
            return 0.0
        
        # Get unique strikes
        strikes = sorted(set(opt.strike for opt in option_chain))
        
        if not strikes:
            return 0.0
        
        # For now, return middle strike (would need OI for real calculation)
        return strikes[len(strikes) // 2]

    async def run_collection_session(
        self,
        interval_minutes: int = 5,
        duration_hours: float = 6.5,
    ) -> None:
        """
        Run a complete data collection session.

        Args:
            interval_minutes: Minutes between snapshots
            duration_hours: Total hours to run collection
        """
        self.stats["start_time"] = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING DATA COLLECTION SESSION")
        self.logger.info(f"Interval: {interval_minutes} minutes")
        self.logger.info(f"Duration: {duration_hours} hours")
        self.logger.info("=" * 80)
        
        # First, collect historical data
        try:
            await self.collect_historical_data(days=60)
            await self.collect_intraday_data(interval=5)
        except Exception as e:
            self.logger.error(f"Historical data collection failed: {e}")
        
        # Calculate end time
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        # Main collection loop
        while datetime.now() < end_time:
            try:
                # Collect snapshot
                await self.collect_market_snapshot()
                
                # Wait for next interval
                self.logger.info(
                    f"Waiting {interval_minutes} minutes until next snapshot..."
                )
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Collection interrupted by user")
                break
                
            except Exception as e:
                self.logger.error(f"Error during collection: {e}")
                self.stats["errors"] += 1
                # Continue after error
                await asyncio.sleep(60)
        
        self.stats["end_time"] = datetime.now()
        
        # Print session summary
        self._print_session_summary()

    def _print_session_summary(self) -> None:
        """Print summary of the collection session."""
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        summary = f"""
{"=" * 80}
DATA COLLECTION SESSION SUMMARY
{"=" * 80}
Duration: {duration}
Snapshots Collected: {self.stats['snapshots_collected']}
Options Collected: {self.stats['options_collected']}
Errors: {self.stats['errors']}

Data saved to: {self.data_dir}
{"=" * 80}
        """
        
        print(summary)
        self.logger.info("Session complete")

    async def cleanup(self) -> None:
        """Clean up resources and disconnect."""
        await self.disconnect()


# CLI interface
@click.command()
@click.option(
    "--days",
    default=60,
    help="Number of days of historical data to collect",
)
@click.option(
    "--interval",
    default=5,
    help="Minutes between snapshots",
)
@click.option(
    "--duration",
    default=6.5,
    help="Hours to run collection",
)
@click.option(
    "--paper/--live",
    default=True,
    help="Use paper trading connection",
)
@click.option(
    "--data-dir",
    default="market_data",
    help="Directory to save data",
)
async def collect_data(
    days: int,
    interval: int,
    duration: float,
    paper: bool,
    data_dir: str,
) -> None:
    """
    Run market data collection session.
    
    This will connect to IB and collect market data at regular intervals,
    saving everything needed for backtesting when markets are closed.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    collector = DataCollector(data_dir=data_dir)
    
    try:
        # Connect to IB
        connected = await collector.connect(paper=paper)
        
        if not connected:
            print("❌ Failed to connect to IB")
            return
        
        # Run collection session
        await collector.run_collection_session(
            interval_minutes=interval,
            duration_hours=duration,
        )
        
    finally:
        await collector.cleanup()


if __name__ == "__main__":
    print("\n🚀 Market Data Collector for 0DTE Trading System")
    print("=" * 50)
    print("Make sure TWS or IB Gateway is running!")
    print("=" * 50 + "\n")
    
    # Run the async command
    asyncio.run(collect_data())
