#!/usr/bin/env python3
"""
Mock Execution Engine for Backtesting.

This module provides a mock implementation of the ExecutionEngine that uses
saved market data instead of live IB connections. It simulates order execution,
fills, and position management for backtesting purposes.

Features:
    - Reads from saved market snapshots
    - Simulates realistic order fills with slippage
    - Tracks virtual positions and P&L
    - Provides same interface as real ExecutionEngine
    - Supports multiple fill models (optimistic, realistic, pessimistic)

Author: Trading Systems
Version: 1.0.0
Date: December 2024
"""

import logging
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import from existing system
from config import get_config
from nexus_core.execution_engine import ExecutionEngine, ExecutionResult, OrderDetails
from models import (
    OptionContract,
    Position,
    PositionState,
    StrategyType,
    TradeSetup,
)


@dataclass
class SimulatedFill:
    """
    Represents a simulated order fill.

    Attributes:
        contract_type: Type of contract (put/call)
        strike: Strike price
        action: BUY or SELL
        quantity: Number of contracts
        fill_price: Simulated fill price
        commission: Simulated commission
        timestamp: When the fill occurred
        slippage: Amount of slippage applied
    """

    contract_type: str
    strike: float
    action: str
    quantity: int
    fill_price: float
    commission: float
    timestamp: datetime
    slippage: float


class MockExecutionEngine(ExecutionEngine):
    """
    Mock execution engine for backtesting with saved data.

    This class simulates all broker interactions using historical data,
    allowing for realistic backtesting when markets are closed.
    """

    def __init__(
        self,
        data_dir: str = "market_data",
        fill_model: str = "realistic",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize mock execution engine.

        Args:
            data_dir: Directory containing saved market data
            fill_model: Fill simulation model ('optimistic', 'realistic', 'pessimistic')
            logger: Optional logger instance
        """
        # Don't call parent __init__ to avoid IB connection setup
        self.logger = logger or self._setup_logger()
        self.config = get_config()
        
        # Data directory setup
        self.data_dir = Path(data_dir)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.options_dir = self.data_dir / "options"
        self.historical_dir = self.data_dir / "historical"
        
        # Load available snapshots
        self.available_snapshots = self._load_snapshot_list()
        self.current_snapshot_idx = 0
        self.current_snapshot = None
        
        # Simulation parameters
        self.fill_model = fill_model
        self.slippage_params = self._get_slippage_params(fill_model)
        
        # Virtual account tracking
        self.virtual_account = {
            "cash": self.config["account"].account_size,
            "buying_power": self.config["account"].account_size * 4,  # 4x leverage
            "positions": {},
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "commission_paid": 0.0,
            "total_trades": 0,
        }
        
        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self.simulated_fills: List[SimulatedFill] = []
        
        # Position tracking
        self.mock_positions: Dict[str, Position] = {}
        self.position_counter = 0
        
        # Override connection status
        self.connected = True  # Always "connected" in mock mode
        self.underlying_symbol = "NQ"
        
        self.logger.info(
            f"Mock Execution Engine initialized with {len(self.available_snapshots)} snapshots"
        )
        self.logger.info(f"Fill model: {fill_model}")

    def _setup_logger(self) -> logging.Logger:
        """Create logger for mock execution."""
        logger = logging.getLogger("MockExecutionEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _load_snapshot_list(self) -> List[Path]:
        """
        Load list of available market snapshots.

        Returns:
            Sorted list of snapshot file paths
        """
        if not self.snapshots_dir.exists():
            self.logger.warning(f"Snapshots directory not found: {self.snapshots_dir}")
            return []
        
        snapshots = sorted(self.snapshots_dir.glob("snapshot_*.pkl"))
        self.logger.info(f"Found {len(snapshots)} market snapshots")
        
        return snapshots

    def _get_slippage_params(self, model: str) -> Dict[str, float]:
        """
        Get slippage parameters based on fill model.

        Args:
            model: Fill model type

        Returns:
            Dictionary of slippage parameters
        """
        params = {
            "optimistic": {
                "spread_fraction": 0.25,  # Fill at 25% into spread
                "market_impact": 0.0,      # No market impact
                "random_factor": 0.1,      # 10% randomness
            },
            "realistic": {
                "spread_fraction": 0.5,    # Fill at mid
                "market_impact": 0.05,     # 5 cents per contract impact
                "random_factor": 0.2,      # 20% randomness
            },
            "pessimistic": {
                "spread_fraction": 0.75,   # Fill at 75% into spread
                "market_impact": 0.10,     # 10 cents per contract impact
                "random_factor": 0.3,      # 30% randomness
            },
        }
        
        return params.get(model, params["realistic"])

    async def connect(self, paper: bool = True) -> bool:
        """
        Mock connection - always succeeds.

        Args:
            paper: Ignored in mock mode

        Returns:
            Always True
        """
        self.logger.info("Mock connection established (no real broker connection)")
        
        # Load first snapshot if available
        if self.available_snapshots:
            await self.load_snapshot(0)
        
        return True

    async def disconnect(self) -> None:
        """Mock disconnect."""
        self.logger.info("Mock connection closed")
        self.connected = False

    async def load_snapshot(self, index: int) -> bool:
        """
        Load a specific market snapshot.

        Args:
            index: Index of snapshot to load

        Returns:
            True if loaded successfully
        """
        if index < 0 or index >= len(self.available_snapshots):
            self.logger.error(f"Invalid snapshot index: {index}")
            return False
        
        try:
            with open(self.available_snapshots[index], "rb") as f:
                self.current_snapshot = pickle.load(f)
            
            self.current_snapshot_idx = index
            
            self.logger.info(
                f"Loaded snapshot {index}: {self.current_snapshot.timestamp}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load snapshot: {e}")
            return False

    async def advance_time(self) -> bool:
        """
        Advance to next available snapshot.

        Returns:
            True if advanced, False if no more snapshots
        """
        next_idx = self.current_snapshot_idx + 1
        
        if next_idx >= len(self.available_snapshots):
            self.logger.info("No more snapshots available")
            return False
        
        return await self.load_snapshot(next_idx)

    async def get_underlying_price(self) -> float:
        """
        Get current underlying price from loaded snapshot.

        Returns:
            Current price of underlying

        Raises:
            ExecutionError: If no snapshot loaded
        """
        if not self.current_snapshot:
            raise Exception("No market snapshot loaded")
        
        return self.current_snapshot.underlying_price

    async def fetch_option_chain(self, expiry: str = None) -> List[OptionContract]:
        """
        Get option chain from current snapshot.

        Args:
            expiry: Ignored - uses snapshot data

        Returns:
            List of OptionContract objects
        """
        if not self.current_snapshot:
            self.logger.error("No snapshot loaded")
            return []
        
        return self.current_snapshot.option_chain

    async def execute_trade(self, trade_setup: TradeSetup) -> ExecutionResult:
        """
        Simulate trade execution with realistic fills.

        Args:
            trade_setup: Trade setup to execute

        Returns:
            Simulated execution result
        """
        self.logger.info("=" * 60)
        self.logger.info(f"MOCK EXECUTING {trade_setup.strategy.value.upper()}")
        self.logger.info(f"Contracts: {trade_setup.contracts}")
        self.logger.info("=" * 60)
        
        if not self.current_snapshot:
            return ExecutionResult(
                success=False,
                error_message="No market snapshot loaded",
            )
        
        try:
            # Route to appropriate execution method
            if trade_setup.strategy == StrategyType.IRON_CONDOR:
                result = await self._mock_execute_iron_condor(trade_setup)
            elif trade_setup.strategy == StrategyType.PUT_SPREAD:
                result = await self._mock_execute_put_spread(trade_setup)
            elif trade_setup.strategy == StrategyType.CALL_SPREAD:
                result = await self._mock_execute_call_spread(trade_setup)
            else:
                return ExecutionResult(
                    success=False,
                    error_message=f"Strategy {trade_setup.strategy} not implemented",
                )
            
            # Update virtual account
            if result.success:
                self._update_virtual_account(result, trade_setup)
            
            # Store in history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Mock execution failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
            )

    async def _mock_execute_iron_condor(
        self, trade_setup: TradeSetup
    ) -> ExecutionResult:
        """Simulate iron condor execution."""
        strikes = trade_setup.strikes
        contracts = trade_setup.contracts
        
        self.logger.info("Simulating Iron Condor execution...")
        
        # Find options in chain
        option_chain = self.current_snapshot.option_chain
        
        # Simulate fills for each leg
        fills = {}
        total_credit = 0
        commission = 0
        
        # Short put
        short_put = self._find_option(
            option_chain, strikes.short_put_strike, "P"
        )
        if short_put:
            fill_price = self._simulate_fill_price(short_put, "SELL")
            fills["short_put"] = fill_price
            total_credit += fill_price * contracts * 20  # NQ multiplier
            commission += self._calculate_commission(contracts)
            
            self.simulated_fills.append(
                SimulatedFill(
                    contract_type="P",
                    strike=strikes.short_put_strike,
                    action="SELL",
                    quantity=contracts,
                    fill_price=fill_price,
                    commission=commission,
                    timestamp=self.current_snapshot.timestamp,
                    slippage=fill_price - short_put.mid,
                )
            )
        
        # Long put
        long_put = self._find_option(
            option_chain, strikes.long_put_strike, "P"
        )
        if long_put:
            fill_price = self._simulate_fill_price(long_put, "BUY")
            fills["long_put"] = fill_price
            total_credit -= fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Short call
        short_call = self._find_option(
            option_chain, strikes.short_call_strike, "C"
        )
        if short_call:
            fill_price = self._simulate_fill_price(short_call, "SELL")
            fills["short_call"] = fill_price
            total_credit += fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Long call
        long_call = self._find_option(
            option_chain, strikes.long_call_strike, "C"
        )
        if long_call:
            fill_price = self._simulate_fill_price(long_call, "BUY")
            fills["long_call"] = fill_price
            total_credit -= fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Create position ID
        self.position_counter += 1
        position_id = f"MOCK_IC_{self.position_counter:04d}"
        
        # Create mock position
        position = Position(
            position_id=position_id,
            trade_setup=trade_setup,
            entry_time=self.current_snapshot.timestamp,
            entry_price=self.current_snapshot.underlying_price,
            fill_prices=fills,
            state=PositionState.ACTIVE,
        )
        
        self.mock_positions[position_id] = position
        
        self.logger.info(f"✅ Mock Iron Condor executed")
        self.logger.info(f"  Position ID: {position_id}")
        self.logger.info(f"  Net credit: ${total_credit:.2f}")
        self.logger.info(f"  Commission: ${commission:.2f}")
        
        return ExecutionResult(
            success=True,
            position_id=position_id,
            orders=[],  # Empty for mock
            fill_prices=fills,
            total_fill_price=total_credit,
            commission=commission,
        )

    async def _mock_execute_put_spread(
        self, trade_setup: TradeSetup
    ) -> ExecutionResult:
        """Simulate put spread execution."""
        strikes = trade_setup.strikes
        contracts = trade_setup.contracts
        
        self.logger.info("Simulating Put Spread execution...")
        
        option_chain = self.current_snapshot.option_chain
        
        fills = {}
        total_credit = 0
        commission = 0
        
        # Short put
        short_put = self._find_option(
            option_chain, strikes.short_put_strike, "P"
        )
        if short_put:
            fill_price = self._simulate_fill_price(short_put, "SELL")
            fills["short_put"] = fill_price
            total_credit += fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Long put
        long_put = self._find_option(
            option_chain, strikes.long_put_strike, "P"
        )
        if long_put:
            fill_price = self._simulate_fill_price(long_put, "BUY")
            fills["long_put"] = fill_price
            total_credit -= fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Create position
        self.position_counter += 1
        position_id = f"MOCK_PS_{self.position_counter:04d}"
        
        position = Position(
            position_id=position_id,
            trade_setup=trade_setup,
            entry_time=self.current_snapshot.timestamp,
            entry_price=self.current_snapshot.underlying_price,
            fill_prices=fills,
            state=PositionState.ACTIVE,
        )
        
        self.mock_positions[position_id] = position
        
        self.logger.info(f"✅ Mock Put Spread executed")
        self.logger.info(f"  Net credit: ${total_credit:.2f}")
        
        return ExecutionResult(
            success=True,
            position_id=position_id,
            orders=[],
            fill_prices=fills,
            total_fill_price=total_credit,
            commission=commission,
        )

    async def _mock_execute_call_spread(
        self, trade_setup: TradeSetup
    ) -> ExecutionResult:
        """Simulate call spread execution."""
        strikes = trade_setup.strikes
        contracts = trade_setup.contracts
        
        self.logger.info("Simulating Call Spread execution...")
        
        option_chain = self.current_snapshot.option_chain
        
        fills = {}
        total_credit = 0
        commission = 0
        
        # Short call
        short_call = self._find_option(
            option_chain, strikes.short_call_strike, "C"
        )
        if short_call:
            fill_price = self._simulate_fill_price(short_call, "SELL")
            fills["short_call"] = fill_price
            total_credit += fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Long call
        long_call = self._find_option(
            option_chain, strikes.long_call_strike, "C"
        )
        if long_call:
            fill_price = self._simulate_fill_price(long_call, "BUY")
            fills["long_call"] = fill_price
            total_credit -= fill_price * contracts * 20
            commission += self._calculate_commission(contracts)
        
        # Create position
        self.position_counter += 1
        position_id = f"MOCK_CS_{self.position_counter:04d}"
        
        position = Position(
            position_id=position_id,
            trade_setup=trade_setup,
            entry_time=self.current_snapshot.timestamp,
            entry_price=self.current_snapshot.underlying_price,
            fill_prices=fills,
            state=PositionState.ACTIVE,
        )
        
        self.mock_positions[position_id] = position
        
        self.logger.info(f"✅ Mock Call Spread executed")
        self.logger.info(f"  Net credit: ${total_credit:.2f}")
        
        return ExecutionResult(
            success=True,
            position_id=position_id,
            orders=[],
            fill_prices=fills,
            total_fill_price=total_credit,
            commission=commission,
        )

    async def close_position(
        self, position: Position, reason: str = "Manual close"
    ) -> ExecutionResult:
        """
        Simulate position closing.

        Args:
            position: Position to close
            reason: Reason for closing

        Returns:
            Simulated closing result
        """
        self.logger.info(f"Mock closing position {position.position_id}: {reason}")
        
        if not self.current_snapshot:
            return ExecutionResult(
                success=False,
                error_message="No snapshot loaded",
            )
        
        # Calculate closing cost based on current prices
        option_chain = self.current_snapshot.option_chain
        closing_cost = 0
        commission = 0
        
        strikes = position.trade_setup.strikes
        contracts = position.trade_setup.contracts
        
        # Simulate closing each leg
        if position.trade_setup.strategy == StrategyType.IRON_CONDOR:
            # Buy back short put
            if strikes.short_put_strike:
                opt = self._find_option(option_chain, strikes.short_put_strike, "P")
                if opt:
                    fill = self._simulate_fill_price(opt, "BUY")
                    closing_cost += fill * contracts * 20
                    commission += self._calculate_commission(contracts)
            
            # Sell long put
            if strikes.long_put_strike:
                opt = self._find_option(option_chain, strikes.long_put_strike, "P")
                if opt:
                    fill = self._simulate_fill_price(opt, "SELL")
                    closing_cost -= fill * contracts * 20
                    commission += self._calculate_commission(contracts)
            
            # Buy back short call
            if strikes.short_call_strike:
                opt = self._find_option(option_chain, strikes.short_call_strike, "C")
                if opt:
                    fill = self._simulate_fill_price(opt, "BUY")
                    closing_cost += fill * contracts * 20
                    commission += self._calculate_commission(contracts)
            
            # Sell long call
            if strikes.long_call_strike:
                opt = self._find_option(option_chain, strikes.long_call_strike, "C")
                if opt:
                    fill = self._simulate_fill_price(opt, "SELL")
                    closing_cost -= fill * contracts * 20
                    commission += self._calculate_commission(contracts)
        
        elif position.trade_setup.strategy == StrategyType.PUT_SPREAD:
            # Buy back short put
            opt = self._find_option(option_chain, strikes.short_put_strike, "P")
            if opt:
                fill = self._simulate_fill_price(opt, "BUY")
                closing_cost += fill * contracts * 20
                commission += self._calculate_commission(contracts)
            
            # Sell long put
            opt = self._find_option(option_chain, strikes.long_put_strike, "P")
            if opt:
                fill = self._simulate_fill_price(opt, "SELL")
                closing_cost -= fill * contracts * 20
                commission += self._calculate_commission(contracts)
        
        elif position.trade_setup.strategy == StrategyType.CALL_SPREAD:
            # Buy back short call
            opt = self._find_option(option_chain, strikes.short_call_strike, "C")
            if opt:
                fill = self._simulate_fill_price(opt, "BUY")
                closing_cost += fill * contracts * 20
                commission += self._calculate_commission(contracts)
            
            # Sell long call
            opt = self._find_option(option_chain, strikes.long_call_strike, "C")
            if opt:
                fill = self._simulate_fill_price(opt, "SELL")
                closing_cost -= fill * contracts * 20
                commission += self._calculate_commission(contracts)
        
        # Calculate P&L
        entry_credit = position.trade_setup.total_credit
        total_pnl = entry_credit - closing_cost - commission
        
        # Update position
        position.state = PositionState.CLOSED
        position.exit_time = self.current_snapshot.timestamp
        position.exit_reason = reason
        position.realized_pnl = total_pnl
        
        # Update virtual account
        self.virtual_account["realized_pnl"] += total_pnl
        self.virtual_account["commission_paid"] += commission
        
        self.logger.info(f"✅ Position closed")
        self.logger.info(f"  Closing cost: ${closing_cost:.2f}")
        self.logger.info(f"  Total P&L: ${total_pnl:.2f}")
        
        return ExecutionResult(
            success=True,
            total_fill_price=-closing_cost,  # Negative for cost
            commission=commission,
        )

    def _find_option(
        self,
        chain: List[OptionContract],
        strike: float,
        right: str,
    ) -> Optional[OptionContract]:
        """
        Find specific option in chain.

        Args:
            chain: Option chain to search
            strike: Strike price
            right: 'P' or 'C'

        Returns:
            Option contract or None
        """
        for opt in chain:
            if opt.strike == strike and opt.right == right:
                return opt
        return None

    def _simulate_fill_price(
        self,
        option: OptionContract,
        action: str,
    ) -> float:
        """
        Simulate realistic fill price with slippage.

        Args:
            option: Option contract
            action: 'BUY' or 'SELL'

        Returns:
            Simulated fill price
        """
        spread = option.ask - option.bid
        mid = option.mid
        
        # Base fill price
        if action == "BUY":
            # Buying - pay more than mid
            base_price = (
                option.bid + spread * self.slippage_params["spread_fraction"]
            )
        else:
            # Selling - receive less than mid
            base_price = (
                option.ask - spread * self.slippage_params["spread_fraction"]
            )
        
        # Add market impact
        impact = self.slippage_params["market_impact"]
        if action == "BUY":
            base_price += impact
        else:
            base_price -= impact
        
        # Add random factor
        random_factor = self.slippage_params["random_factor"]
        random_adjustment = random.uniform(-random_factor, random_factor) * 0.05
        
        final_price = base_price * (1 + random_adjustment)
        
        # Ensure price is within bid-ask spread
        final_price = max(option.bid, min(option.ask, final_price))
        
        return round(final_price, 2)

    def _calculate_commission(self, contracts: int) -> float:
        """
        Calculate commission for trade.

        Args:
            contracts: Number of contracts

        Returns:
            Total commission
        """
        # Typical futures options commission
        per_contract = 2.25  # Includes exchange fees
        return contracts * per_contract

    def _update_virtual_account(
        self,
        result: ExecutionResult,
        trade_setup: TradeSetup,
    ) -> None:
        """
        Update virtual account after trade.

        Args:
            result: Execution result
            trade_setup: Trade setup that was executed
        """
        # Update cash (credit received minus commission)
        self.virtual_account["cash"] += result.total_fill_price - result.commission
        self.virtual_account["commission_paid"] += result.commission
        self.virtual_account["total_trades"] += 1
        
        # Track position
        self.virtual_account["positions"][result.position_id] = {
            "strategy": trade_setup.strategy.value,
            "credit": result.total_fill_price,
            "max_risk": trade_setup.total_max_risk,
            "contracts": trade_setup.contracts,
        }

    async def get_positions(self) -> List:
        """Get current mock positions."""
        return list(self.mock_positions.values())

    async def get_account_summary(self) -> Dict:
        """Get virtual account summary."""
        return {
            "NetLiquidation": self.virtual_account["cash"],
            "AvailableFunds": self.virtual_account["cash"],
            "BuyingPower": self.virtual_account["buying_power"],
            "UnrealizedPnL": self.virtual_account["unrealized_pnl"],
            "RealizedPnL": self.virtual_account["realized_pnl"],
            "CommissionPaid": self.virtual_account["commission_paid"],
            "TotalTrades": self.virtual_account["total_trades"],
        }

    def get_performance_report(self) -> Dict:
        """
        Generate performance report for the session.

        Returns:
            Dictionary with performance metrics
        """
        total_trades = len(self.execution_history)
        successful_trades = sum(1 for r in self.execution_history if r.success)
        
        # Calculate P&L from closed positions
        closed_positions = [
            p for p in self.mock_positions.values()
            if p.state == PositionState.CLOSED
        ]
        
        total_pnl = sum(p.realized_pnl or 0 for p in closed_positions)
        winning_trades = sum(
            1 for p in closed_positions if (p.realized_pnl or 0) > 0
        )
        losing_trades = sum(
            1 for p in closed_positions if (p.realized_pnl or 0) < 0
        )
        
        # Calculate statistics
        win_rate = (
            winning_trades / len(closed_positions) * 100
            if closed_positions else 0
        )
        
        avg_win = (
            sum(p.realized_pnl for p in closed_positions if p.realized_pnl > 0)
            / winning_trades
            if winning_trades > 0 else 0
        )
        
        avg_loss = (
            sum(p.realized_pnl for p in closed_positions if p.realized_pnl < 0)
            / losing_trades
            if losing_trades > 0 else 0
        )
        
        return {
            "total_trades": total_trades,
            "successful_executions": successful_trades,
            "closed_positions": len(closed_positions),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "commission_paid": self.virtual_account["commission_paid"],
            "net_pnl": total_pnl - self.virtual_account["commission_paid"],
        }

    def is_connected(self) -> bool:
        """Always return True for mock engine."""
        return True


# Test function
async def test_mock_engine():
    """Test the mock execution engine."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("Testing Mock Execution Engine")
    print("=" * 60)
    
    # Create mock engine
    engine = MockExecutionEngine(fill_model="realistic")
    
    # Connect
    await engine.connect()
    
    # Get price
    price = await engine.get_underlying_price()
    print(f"\nUnderlying price: ${price:,.2f}")
    
    # Get options
    options = await engine.fetch_option_chain()
    print(f"Options in chain: {len(options)}")
    
    # Get account
    account = await engine.get_account_summary()
    print(f"\nVirtual Account:")
    for key, value in account.items():
        print(f"  {key}: ${value:,.2f}")
    
    print("\n✅ Mock engine test complete!")


if __name__ == "__main__":
    import asyncio
    
    asyncio.run(test_mock_engine())
