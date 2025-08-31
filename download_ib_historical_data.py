# download_historical_data.py
"""
Download 2 years of historical data from IB
============================================
Fetches VIX, underlying, and options data for training the trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, List
import pickle

from ib_insync import *
from nexus_core.execution_engine import ExecutionEngine


class HistoricalDataDownloader:
    """Download and store historical market data from IB."""

    async def download_qqq_history(self, years: int = 2) -> pd.DataFrame:
        """
        Download QQQ ETF historical data.
        
        Args:
            years: Number of years of history
            
        Returns:
            DataFrame with QQQ data
        """
        self.logger.info(f"Downloading {years} years of QQQ data...")
        
        # QQQ ETF contract
        qqq = Stock('QQQ', 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(qqq)
        
        # Request historical data
        bars = await self.ib.reqHistoricalDataAsync(
            qqq,
            endDateTime='',
            durationStr=f'{years} Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # Convert to DataFrame
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Save to CSV
        qqq_file = self.data_dir / 'qqq_history.csv'
        df.to_csv(qqq_file)
        self.logger.info(f"✅ Saved QQQ data to {qqq_file}")
        
        return df

    async def download_qqq_options_historical(self, 
                                            date: str,
                                            dte: int = 0) -> List[Dict]:
        """
        Download QQQ options chain for a specific date.
        
        Args:
            date: Date in YYYYMMDD format
            dte: Days to expiration (0 for 0DTE)
            
        Returns:
            List of option contracts with prices
        """
        self.logger.info(f"Downloading QQQ options for {date}...")
        
        qqq = Stock('QQQ', 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(qqq)
        
        # FIX: Correct date format for IB
        formatted_date = f"{date} 15:59:00 US/Eastern"  # End of trading day
        
        # Get the underlying price for that date
        bars = await self.ib.reqHistoricalDataAsync(
            qqq,
            endDateTime=formatted_date,
            durationStr='1 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        
        if bars:
            underlying_price = bars[-1].close
        else:
            return []
        
        # Get option chains
        chains = await self.ib.reqSecDefOptParamsAsync(
            qqq.symbol,
            '',
            qqq.secType,
            qqq.conId
        )
        
        options_data = []
        
        if chains:
            # Find the right expiration
            chain = chains[0]
            expiry = date  # For 0DTE
            
            # Get strikes around ATM
            strikes = [s for s in chain.strikes 
                    if underlying_price * 0.95 <= s <= underlying_price * 1.05]
            
            for strike in strikes:
                for right in ['C', 'P']:
                    opt = Option('QQQ', expiry, strike, right, 'SMART')
                    await self.ib.qualifyContractsAsync(opt)
                    
                    # Get historical data for this option
                    ticker = await self.ib.reqMktDataAsync(opt)
                    await asyncio.sleep(0.1)  # Let data populate
                    
                    options_data.append({
                        'strike': strike,
                        'right': right,
                        'expiry': expiry,
                        'bid': ticker.bid or 0,
                        'ask': ticker.ask or 0,
                        'mid': ticker.midpoint() or 0,
                        'volume': ticker.volume or 0
                    })
            
            # Cancel market data
            self.ib.cancelMktData(opt)
        
        return options_data
    
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the downloader."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.logger = logging.getLogger("DataDownloader")
        
        # IB connection
        self.ib = IB()
        self.connected = False
        
    async def connect(self):
        """Connect to IB Gateway/TWS."""
        try:
            await self.ib.connectAsync('127.0.0.1', 7497, clientId=2)
            self.connected = True
            self.logger.info("✅ Connected to IB")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def download_vix_history(self, years: int = 2) -> pd.DataFrame:
        """
        Download VIX index historical data.
        
        Args:
            years: Number of years of history
            
        Returns:
            DataFrame with VIX data
        """
        self.logger.info(f"Downloading {years} years of VIX data...")
        
        # VIX Index contract
        vix = Index('VIX', 'CBOE')
        
        # Request historical data
        bars = await self.ib.reqHistoricalDataAsync(
            vix,
            endDateTime='',  # Current time
            durationStr=f'{years} Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # Convert to DataFrame
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Save to CSV
        vix_file = self.data_dir / 'vix_history.csv'
        df.to_csv(vix_file)
        self.logger.info(f"✅ Saved VIX data to {vix_file}")
        
        return df
    
    async def download_nq_history(self, years: int = 2) -> pd.DataFrame:
        """
        Download NQ futures historical data.
        
        Args:
            years: Number of years of history
            
        Returns:
            DataFrame with NQ data
        """
        self.logger.info(f"Downloading {years} years of NQ data...")
        
        # NQ continuous future
        nq = ContFuture('NQ', 'CME')
        await self.ib.qualifyContractsAsync(nq)
        
        # Get the front month contract
        nq_contract = await self.ib.reqContractDetailsAsync(nq)
        if nq_contract:
            contract = nq_contract[0].contract
        else:
            # Fallback to specific contract
            contract = Future('NQ', '202412', 'CME')
        
        # Request historical data
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=f'{years} Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # Convert to DataFrame
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Save to CSV
        nq_file = self.data_dir / 'nq_history.csv'
        df.to_csv(nq_file)
        self.logger.info(f"✅ Saved NQ data to {nq_file}")
        
        return df
    
    async def download_intraday_data(self, 
                                    symbol: str,
                                    days_back: int = 30) -> pd.DataFrame:
        """
        Download intraday data for recent period.
        
        Args:
            symbol: Contract symbol (NQ or VIX)
            days_back: Number of days to download
            
        Returns:
            DataFrame with intraday data
        """
        self.logger.info(f"Downloading {days_back} days of intraday {symbol} data...")
        
        if symbol == 'VIX':
            contract = Index('VIX', 'CBOE')
        else:
            contract = ContFuture('NQ', 'CME')
            await self.ib.qualifyContractsAsync(contract)
        
        # Request 5-minute bars for intraday analysis
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=f'{days_back} D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Save to CSV
        intraday_file = self.data_dir / f'{symbol.lower()}_intraday.csv'
        df.to_csv(intraday_file)
        self.logger.info(f"✅ Saved intraday data to {intraday_file}")
        
        return df
    
    async def download_option_chains_sample(self, 
                                           dates: List[str] = None,
                                           underlying: str = 'NQ') -> Dict:
        """
        Download sample option chains for specific dates.
        
        Args:
            dates: List of dates (YYYY-MM-DD format)
            underlying: NQ or ES
            
        Returns:
            Dictionary of option chains by date
        """
        if dates is None:
            # Sample dates across different market conditions
            dates = [
                (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                for i in [1, 7, 14, 30, 60, 90]
            ]
        
        self.logger.info(f"Downloading option chains for {len(dates)} dates...")
        
        option_data = {}
        
        for date_str in dates:
            try:
                # Get the option chain for that date
                if underlying == 'NQ':
                    fut = Future('NQ', date_str[:6], 'CME')
                    await self.ib.qualifyContractsAsync(fut)
                    
                    # Get strikes around the money
                    chains = await self.ib.reqSecDefOptParamsAsync(
                        underlyingSymbol='NQ',
                        futFopExchange='CME',
                        underlyingSecType='FUT',
                        underlyingConId=fut.conId
                    )
                    
                    if chains:
                        chain = chains[0]
                        strikes = [s for s in chain.strikes 
                                 if 0.8 * fut.marketPrice() < s < 1.2 * fut.marketPrice()]
                        
                        # Get option data for these strikes
                        options = []
                        for strike in strikes[:20]:  # Limit to 20 strikes
                            for right in ['C', 'P']:
                                opt = FuturesOption(
                                    'NQ', date_str[:6], strike, right, 'CME'
                                )
                                await self.ib.qualifyContractsAsync(opt)
                                ticker = await self.ib.reqTickersAsync(opt)
                                
                                if ticker:
                                    options.append({
                                        'strike': strike,
                                        'right': right,
                                        'bid': ticker[0].bid,
                                        'ask': ticker[0].ask,
                                        'mid': ticker[0].midpoint(),
                                        'volume': ticker[0].volume
                                    })
                        
                        option_data[date_str] = options
                        self.logger.info(f"✅ Downloaded {len(options)} options for {date_str}")
                
            except Exception as e:
                self.logger.error(f"Error downloading options for {date_str}: {e}")
                continue
        
        # Save option data
        options_file = self.data_dir / 'option_chains_sample.json'
        with open(options_file, 'w') as f:
            json.dump(option_data, f, indent=2)
        
        self.logger.info(f"✅ Saved option chains to {options_file}")
        return option_data
    
    async def download_market_breadth(self, days: int = 30) -> pd.DataFrame:
        """
        Download market breadth indicators (TICK, ADD, etc).
        
        Args:
            days: Number of days to download
            
        Returns:
            DataFrame with market breadth data
        """
        self.logger.info(f"Downloading {days} days of market breadth...")
        
        breadth_data = {}
        
        # TICK Index
        tick = Index('TICK', 'NYSE')
        tick_bars = await self.ib.reqHistoricalDataAsync(
            tick,
            endDateTime='',
            durationStr=f'{days} D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        breadth_data['TICK'] = util.df(tick_bars)
        
        # ADD (Advance Decline)
        add = Index('ADD', 'NYSE')
        add_bars = await self.ib.reqHistoricalDataAsync(
            add,
            endDateTime='',
            durationStr=f'{days} D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        breadth_data['ADD'] = util.df(add_bars)
        
        # Save breadth data
        breadth_file = self.data_dir / 'market_breadth.pkl'
        with open(breadth_file, 'wb') as f:
            pickle.dump(breadth_data, f)
        
        self.logger.info(f"✅ Saved market breadth to {breadth_file}")
        return breadth_data
    
    async def create_training_dataset(self) -> pd.DataFrame:
        """
        Combine all downloaded data into a training dataset.
        
        Returns:
            Complete DataFrame for training
        """
        self.logger.info("Creating training dataset...")
        
        # Load saved data
        vix_df = pd.read_csv(self.data_dir / 'vix_history.csv', index_col='date', parse_dates=True)
        nq_df = pd.read_csv(self.data_dir / 'nq_history.csv', index_col='date', parse_dates=True)
        
        # Merge data
        training_df = pd.merge(
            nq_df[['open', 'high', 'low', 'close', 'volume']],
            vix_df[['close']].rename(columns={'close': 'vix'}),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Add features for pattern recognition
        training_df['vix_regime'] = pd.cut(
            training_df['vix'],
            bins=[0, 15, 20, 25, 30, 100],
            labels=['very_low', 'low', 'normal', 'elevated', 'high']
        )
        
        # Price movement features
        training_df['daily_range'] = training_df['high'] - training_df['low']
        training_df['daily_return'] = training_df['close'].pct_change()
        training_df['vix_change'] = training_df['vix'].pct_change()
        
        # Trend indicators
        training_df['sma_20'] = training_df['close'].rolling(20).mean()
        training_df['trend'] = training_df.apply(
            lambda x: 'bullish' if x['close'] > x['sma_20'] else 'bearish',
            axis=1
        )
        
        # Save complete dataset
        dataset_file = self.data_dir / 'training_dataset.csv'
        training_df.to_csv(dataset_file)
        self.logger.info(f"✅ Saved training dataset to {dataset_file}")
        
        return training_df
    
    async def run_full_download(self):
        """Execute complete data download process."""
        try:
            # Connect to IB
            if not await self.connect():
                return
            
            # Download all data
            print("\n" + "="*60)
            print("DOWNLOADING 2 YEARS OF HISTORICAL DATA")
            print("="*60 + "\n")
            
            # 1. VIX History (Most Important!)
            vix_df = await self.download_vix_history(years=2)
            print(f"VIX Data: {len(vix_df)} days")
            print(f"VIX Range: {vix_df['close'].min():.2f} - {vix_df['close'].max():.2f}")
            
            # 2. NQ History
            nq_df = await self.download_nq_history(years=2)
            print(f"NQ Data: {len(nq_df)} days")

            # 3. QQQ History (NEW!)
            qqq_df = await self.download_qqq_history(years=2)
            print(f"QQQ Data: {len(qqq_df)} days")
            print(f"QQQ Range: ${qqq_df['close'].min():.2f} - ${qqq_df['close'].max():.2f}")
            
            # 4. Download QQQ options for your specific dates
            for date in ['20230905', '20230906', '20230907']:
                options = await self.download_qqq_options_historical(date)
                if options:
                    filename = self.data_dir / f'QQQ_options_{date}.json'
                    with open(filename, 'w') as f:
                        json.dump(options, f, indent=2)
                    print(f"✅ Downloaded {len(options)} QQQ options for {date}")
  
            # 5. Recent Intraday Data
            await self.download_intraday_data('VIX', days_back=30)
            await self.download_intraday_data('NQ', days_back=30)
            
            # 4. Market Breadth
            await self.download_market_breadth(days=30)
            
            # 6. Sample Option Chains
            # Note: Full historical options would be massive
            await self.download_option_chains_sample()
            
            # 7. Create Training Dataset
            training_df = await self.create_training_dataset()
            
            print("\n" + "="*60)
            print("✅ DOWNLOAD COMPLETE!")
            print(f"Training dataset: {len(training_df)} days")
            print(f"Date range: {training_df.index.min()} to {training_df.index.max()}")
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.connected:
                self.ib.disconnect()
                self.logger.info("Disconnected from IB")


async def main():
    """Run the historical data download."""
    downloader = HistoricalDataDownloader()
    await downloader.run_full_download()


if __name__ == "__main__":
    print("\n🚀 Starting historical data download...")
    print("This will take several minutes. Make sure IB Gateway is running!\n")
    
    asyncio.run(main())