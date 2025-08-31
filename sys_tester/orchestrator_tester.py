#!/usr/bin/env python3
"""
Trading System Orchestrator for Backtesting.

This module provides a testing version of the orchestrator that uses saved
market data instead of live connections. It allows for comprehensive backtesting
of trading strategies when markets are closed.

Features:
    - Replay historical market data
    - Test strategies across different market conditions
    - Generate detailed performance reports
    - Support multiple testing modes (replay, scenario, optimization)
    - Visualize results with charts and metrics

Author: Trading Systems
Version: 1.0.0
Date: December 2024
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
import pytz
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Import from existing system
from config import get_config, update_config
from market_context import MarketContext
from mock_execution_engine import MockExecutionEngine
from models import (
    MarketData,
    Position,
    PositionState,
    SessionStats,
    StrategyType,
)
from orchestrator import TradingOrchestrator
from nexus_core.position_monitor import PositionMonitor
from strategy_engine import StrategyEngine


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest session.

    Attributes:
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting account balance
        max_trades_per_day: Maximum trades allowed per day
        fill_model: Fill simulation model
        style: Trading style (safe/balanced/aggressive)
        strategies: List of strategies to test
        data_dir: Directory containing market data
    """

    start_date: datetime
    end_date: datetime
    initial_capital: float = 15000.0
    max_trades_per_day: int = 5
    fill_model: str = "realistic"
    style: str = "balanced"
    strategies: List[StrategyType] = field(default_factory=list)
    data_dir: str = "market_data"


@dataclass
class BacktestResult:
    """
    Results from a backtest session.

    Attributes:
        config: Backtest configuration used
        trades: List of all trades executed
        positions: List of all positions
        performance: Performance metrics
        daily_stats: Daily statistics
        drawdown_series: Drawdown over time
    """

    config: BacktestConfig
    trades: List[Dict]
    positions: List[Position]
    performance: Dict[str, Any]
    daily_stats: pd.DataFrame
    drawdown_series: pd.Series


class OrchestratorTester(TradingOrchestrator):
    """
    Testing orchestrator for backtesting with saved data.

    This class extends the regular orchestrator to work with
    mock components and saved market data.
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize testing orchestrator.

        Args:
            backtest_config: Configuration for backtest
            logger: Optional logger instance
        """
        # Setup logger first
        self.logger = logger or self._setup_test_logger()
        self.config = get_config()
        
        # Store backtest configuration
        self.backtest_config = backtest_config
        
        # Update system config with backtest parameters
        update_config(
            "account",
            account_size=backtest_config.initial_capital,
            max_daily_trades=backtest_config.max_trades_per_day,
        )
        
        # Initialize components with mock versions
        self.logger.info("=" * 80)
        self.logger.info("BACKTEST ORCHESTRATOR v1.0")
        self.logger.info("Initializing components for backtesting...")
        self.logger.info("=" * 80)
        
        # Core components (using existing ones where possible)
        self.context = MarketContext(self.logger)
        
        # Use mock execution engine instead of real one
        self.executor = MockExecutionEngine(
            data_dir=backtest_config.data_dir,
            fill_model=backtest_config.fill_model,
            logger=self.logger,
        )
        
        # Strategy engine uses the context
        self.strategy_engine = StrategyEngine(self.context, self.logger)
        
        # Position monitor with mock executor
        self.monitor = PositionMonitor(
            self.context,
            self.executor,
            self.logger,
        )
        
        # Session management
        self.session_stats = SessionStats()
        self.is_running = False
        self.paper_trading = True  # Always paper in backtest
        
        # Backtest-specific tracking
        self.current_date = backtest_config.start_date
        self.backtest_trades = []
        self.backtest_positions = []
        self.daily_performance = []
        self.equity_curve = []
        
        # Rich console for pretty output
        self.console = Console()
        
        self.logger.info("✅ Backtest orchestrator initialized")

    def _setup_test_logger(self) -> logging.Logger:
        """Setup logger for backtesting."""
        # Create logs directory
        log_dir = Path("logs") / "backtests"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger("BacktestOrchestrator")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        # Console handler (less verbose for backtesting)
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console.setFormatter(console_format)
        logger.addHandler(console)
        
        # File handler (full detail)
        log_file = (
            log_dir / 
            f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger

    async def initialize(self, paper: bool = True) -> bool:
        """
        Initialize backtest environment.

        Args:
            paper: Ignored - always paper in backtest

        Returns:
            True if initialization successful
        """
        try:
            self.console.print(
                "[bold green]Initializing backtest environment...[/bold green]"
            )
            
            # Connect mock executor
            await self.executor.connect(paper=True)
            
            # Load first snapshot
            if not self.executor.available_snapshots:
                self.console.print(
                    "[bold red]No market snapshots found![/bold red]"
                )
                self.console.print(
                    "Run data_collector.py during market hours first."
                )
                return False
            
            # Find snapshots within date range
            self.available_snapshots = self._filter_snapshots_by_date()
            
            if not self.available_snapshots:
                self.console.print(
                    f"[bold red]No snapshots found between "
                    f"{self.backtest_config.start_date} and "
                    f"{self.backtest_config.end_date}[/bold red]"
                )
                return False
            
            self.console.print(
                f"[green]Found {len(self.available_snapshots)} snapshots "
                f"for backtesting[/green]"
            )
            
            # Load first snapshot
            await self.executor.load_snapshot(0)
            
            # Update market context with initial data
            await self._update_context_from_snapshot()
            
            # Perform initial analysis
            self.logger.info("Performing initial market analysis...")
            analysis = self.context.analyze_market()
            
            self.console.print(
                f"[cyan]Initial Market State:[/cyan]"
            )
            self.console.print(
                f"  Price: ${self.executor.current_snapshot.underlying_price:,.2f}"
            )
            self.console.print(
                f"  VIX: {self.executor.current_snapshot.vix_data['last']:.1f}"
            )
            self.console.print(
                f"  Regime: {analysis.market_regime.value}"
            )
            self.console.print(
                f"  Trend: {analysis.trend_state.value}"
            )
            
            self.console.print(
                "[bold green]✅ Backtest environment ready![/bold green]"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _filter_snapshots_by_date(self) -> List[int]:
        """
        Filter available snapshots by date range.

        Returns:
            List of snapshot indices within date range
        """
        valid_indices = []
        
        for i, snapshot_path in enumerate(self.executor.available_snapshots):
            # Parse date from filename
            filename = snapshot_path.name
            # Format: snapshot_YYYYMMDD_HHMMSS.pkl
            try:
                date_str = filename.split("_")[1]
                date = datetime.strptime(date_str, "%Y%m%d")
                
                if (
                    self.backtest_config.start_date.date() <= date.date() <=
                    self.backtest_config.end_date.date()
                ):
                    valid_indices.append(i)
                    
            except Exception:
                continue
        
        return valid_indices

    async def _update_context_from_snapshot(self) -> None:
        """Update market context with current snapshot data."""
        snapshot = self.executor.current_snapshot
        
        if not snapshot:
            return
        
        # Update market data
        self.context.update_market_data(
            price=snapshot.underlying_price,
            volume=snapshot.underlying_data.get("volume", 100000),
            vix=snapshot.vix_data["last"],
            high=snapshot.underlying_data.get("high", snapshot.underlying_price),
            low=snapshot.underlying_data.get("low", snapshot.underlying_price),
            open_price=snapshot.underlying_data.get("close", snapshot.underlying_price),
            previous_close=snapshot.underlying_data.get("close", snapshot.underlying_price),
        )
        
        # Update option chain
        self.context.update_option_chain(snapshot.option_chain)
        
        # Note: Historical data would need to be loaded separately
        # For now, we'll use a simplified approach
        if not self.context.historical_data or self.context.historical_data.empty:
            # Create synthetic historical data for analysis
            self._create_synthetic_historical_data()

    def _create_synthetic_historical_data(self) -> None:
        """Create synthetic historical data for market analysis."""
        # This is a simplified approach for backtesting
        # In production, you'd load actual historical data
        
        current_price = self.executor.current_snapshot.underlying_price
        
        # Generate 60 days of synthetic data
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        
        # Create realistic price movements
        returns = pd.Series(
            [random.gauss(0, 0.01) for _ in range(60)]
        )
        prices = current_price * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            "date": dates,
            "open": prices * 0.995,
            "high": prices * 1.005,
            "low": prices * 0.992,
            "close": prices,
            "volume": [100000] * 60,
            "vix_close": [16.0] * 60,
        })
        
        self.context.update_historical_data(data)

    async def run_backtest(self) -> BacktestResult:
        """
        Run complete backtest session.

        Returns:
            BacktestResult with all performance data
        """
        self.console.print("\n" + "=" * 80)
        self.console.print(
            "[bold cyan]STARTING BACKTEST SESSION[/bold cyan]"
        )
        self.console.print(f"Start: {self.backtest_config.start_date}")
        self.console.print(f"End: {self.backtest_config.end_date}")
        self.console.print(f"Initial Capital: ${self.backtest_config.initial_capital:,.2f}")
        self.console.print(f"Fill Model: {self.backtest_config.fill_model}")
        self.console.print(f"Style: {self.backtest_config.style}")
        self.console.print("=" * 80 + "\n")
        
        self.is_running = True
        self.session_stats.session_start = datetime.now()
        
        # Progress tracking
        total_snapshots = len(self.available_snapshots)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Processing snapshots...",
                total=total_snapshots,
            )
            
            # Process each snapshot
            for i in self.available_snapshots:
                # Load snapshot
                await self.executor.load_snapshot(i)
                
                # Update context
                await self._update_context_from_snapshot()
                
                # Get current timestamp
                current_time = self.executor.current_snapshot.timestamp
                
                # Check if we should trade
                if self._should_analyze(current_time):
                    # Run analysis cycle
                    trade_executed = await self._run_analysis_cycle()
                    
                    if trade_executed:
                        self.console.print(
                            f"[green]✓ Trade executed at {current_time}[/green]"
                        )
                
                # Monitor positions
                await self._monitor_positions()
                
                # Record daily stats if end of day
                if self._is_end_of_day(current_time):
                    self._record_daily_stats(current_time)
                
                # Update progress
                progress.update(task, advance=1)
        
        # Close any remaining positions
        await self._close_all_positions("Backtest end")
        
        # Generate final results
        result = self._generate_backtest_result()
        
        # Display results
        self._display_results(result)
        
        return result

    def _should_analyze(self, current_time: datetime) -> bool:
        """
        Determine if we should run analysis at current time.

        Args:
            current_time: Current snapshot time

        Returns:
            True if should analyze
        """
        # Only analyze during market hours
        hour = current_time.hour
        minute = current_time.minute
        
        # Market hours: 9:30 AM - 4:00 PM ET
        if hour < 9 or hour >= 16:
            return False
        
        if hour == 9 and minute < 30:
            return False
        
        # Analyze every 5 minutes
        return minute % 5 == 0

    def _is_end_of_day(self, current_time: datetime) -> bool:
        """Check if current time is end of trading day."""
        return current_time.hour == 15 and current_time.minute >= 55

    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        for position in self.monitor.active_positions.values():
            if position.state == PositionState.ACTIVE:
                await self.executor.close_position(position, reason)

    def _record_daily_stats(self, date: datetime) -> None:
        """Record daily performance statistics."""
        account_summary = asyncio.run(
            self.executor.get_account_summary()
        )
        
        daily_stat = {
            "date": date.date(),
            "ending_balance": account_summary["NetLiquidation"],
            "trades_executed": self.session_stats.trades_executed,
            "realized_pnl": account_summary["RealizedPnL"],
            "commission": account_summary["CommissionPaid"],
            "positions_open": len(self.monitor.active_positions),
        }
        
        self.daily_performance.append(daily_stat)
        self.equity_curve.append(account_summary["NetLiquidation"])

    def _generate_backtest_result(self) -> BacktestResult:
        """Generate comprehensive backtest results."""
        # Get final performance metrics
        performance = self.executor.get_performance_report()
        
        # Add additional metrics
        initial_capital = self.backtest_config.initial_capital
        final_capital = self.executor.virtual_account["cash"]
        
        performance.update({
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": (final_capital - initial_capital) / initial_capital * 100,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown(),
            "profit_factor": self._calculate_profit_factor(),
        })
        
        # Create daily stats DataFrame
        daily_df = pd.DataFrame(self.daily_performance) if self.daily_performance else pd.DataFrame()
        
        # Create drawdown series
        equity_series = pd.Series(self.equity_curve) if self.equity_curve else pd.Series()
        running_max = equity_series.expanding().max()
        drawdown_series = (equity_series - running_max) / running_max * 100
        
        return BacktestResult(
            config=self.backtest_config,
            trades=self.backtest_trades,
            positions=list(self.executor.mock_positions.values()),
            performance=performance,
            daily_stats=daily_df,
            drawdown_series=drawdown_series,
        )

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        if returns.empty:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days)
        return (
            returns.mean() / returns.std() * (252 ** 0.5)
            if returns.std() > 0 else 0.0
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if not self.equity_curve:
            return 0.0
        
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        return abs(drawdown.min())

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        positions = list(self.executor.mock_positions.values())
        
        gross_profit = sum(
            p.realized_pnl for p in positions
            if p.realized_pnl and p.realized_pnl > 0
        )
        
        gross_loss = abs(sum(
            p.realized_pnl for p in positions
            if p.realized_pnl and p.realized_pnl < 0
        ))
        
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def _display_results(self, result: BacktestResult) -> None:
        """Display backtest results in a formatted table."""
        self.console.print("\n" + "=" * 80)
        self.console.print(
            "[bold green]BACKTEST RESULTS[/bold green]"
        )
        self.console.print("=" * 80)
        
        # Performance table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf = result.performance
        
        perf_table.add_row(
            "Initial Capital",
            f"${perf['initial_capital']:,.2f}"
        )
        perf_table.add_row(
            "Final Capital",
            f"${perf['final_capital']:,.2f}"
        )
        perf_table.add_row(
            "Total Return",
            f"{perf['total_return']:.2f}%"
        )
        perf_table.add_row(
            "Total Trades",
            str(perf['total_trades'])
        )
        perf_table.add_row(
            "Win Rate",
            f"{perf['win_rate']:.1f}%"
        )
        perf_table.add_row(
            "Net P&L",
            f"${perf['net_pnl']:,.2f}"
        )
        perf_table.add_row(
            "Sharpe Ratio",
            f"{perf['sharpe_ratio']:.2f}"
        )
        perf_table.add_row(
            "Max Drawdown",
            f"{perf['max_drawdown']:.2f}%"
        )
        perf_table.add_row(
            "Profit Factor",
            f"{perf['profit_factor']:.2f}"
        )
        
        self.console.print(perf_table)
        
        # Save detailed report
        self._save_backtest_report(result)

    def _save_backtest_report(self, result: BacktestResult) -> None:
        """Save detailed backtest report to file."""
        report_dir = Path("backtest_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save result object
        with open(report_dir / f"backtest_result_{timestamp}.pkl", "wb") as f:
            pickle.dump(result, f)
        
        # Save JSON summary
        summary = {
            "config": {
                "start_date": str(result.config.start_date),
                "end_date": str(result.config.end_date),
                "initial_capital": result.config.initial_capital,
                "fill_model": result.config.fill_model,
                "style": result.config.style,
            },
            "performance": result.performance,
        }
        
        with open(report_dir / f"backtest_summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save daily stats to CSV
        if not result.daily_stats.empty:
            result.daily_stats.to_csv(
                report_dir / f"daily_stats_{timestamp}.csv",
                index=False,
            )
        
        self.console.print(
            f"\n[green]Reports saved to {report_dir}/[/green]"
        )


# CLI interface
@click.command()
@click.option(
    "--start-date",
    type=click.DateTime(),
    default=datetime.now() - timedelta(days=7),
    help="Start date for backtest",
)
@click.option(
    "--end-date",
    type=click.DateTime(),
    default=datetime.now(),
    help="End date for backtest",
)
@click.option(
    "--capital",
    type=float,
    default=15000.0,
    help="Initial capital for backtest",
)
@click.option(
    "--max-trades",
    type=int,
    default=5,
    help="Maximum trades per day",
)
@click.option(
    "--fill-model",
    type=click.Choice(["optimistic", "realistic", "pessimistic"]),
    default="realistic",
    help="Fill simulation model",
)
@click.option(
    "--style",
    type=click.Choice(["safe", "balanced", "aggressive"]),
    default="balanced",
    help="Trading style",
)
@click.option(
    "--data-dir",
    type=str,
    default="market_data",
    help="Directory containing market data",
)
async def run_backtest(
    start_date: datetime,
    end_date: datetime,
    capital: float,
    max_trades: int,
    fill_model: str,
    style: str,
    data_dir: str,
) -> None:
    """
    Run backtest with saved market data.
    
    This will replay market data and test your trading strategies
    as if they were running in real-time.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        max_trades_per_day=max_trades,
        fill_model=fill_model,
        style=style,
        data_dir=data_dir,
    )
    
    # Create tester
    tester = OrchestratorTester(config)
    
    # Initialize
    success = await tester.initialize()
    
    if not success:
        print("❌ Failed to initialize backtest")
        return
    
    # Run backtest
    result = await tester.run_backtest()
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    import random
    
    print("\n" + "=" * 60)
    print("0DTE OPTIONS BACKTESTING SYSTEM")
    print("=" * 60)
    print("\nMake sure you have collected market data first!")
    print("Run: python data_collector.py")
    print("=" * 60 + "\n")
    
    asyncio.run(run_backtest())
