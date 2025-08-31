"""
Execution Engine - Fixed Version
=================================
Handles all broker interactions and order execution.
Fixed to properly handle asyncio and IB connections.

Author: Trading Systems
Version: 2.1
Date: August 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import asyncio
from ib_insync import *
import nest_asyncio

# CRITICAL FIX: Allow nested event loops - this is what makes it work!
nest_asyncio.apply()

from config import get_config
from models import (
    TradeSetup, Position, PositionState, OptionContract,
    StrikeSelection, StrategyType
)
from exceptions import (
    ExecutionError, OrderRejectedError, FillError,
    ConnectionError as TradingConnectionError
)


@dataclass
class OrderDetails:
    """Details of an order for tracking."""
    order_id: str
    contract: Contract
    order: Order
    trade: Trade
    strategy_leg: str  # 'short_put', 'long_put', etc.
    
    
@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    success: bool
    position_id: Optional[str] = None
    orders: List[OrderDetails] = field(default_factory=list)
    fill_prices: Dict[str, float] = field(default_factory=dict)
    total_fill_price: float = 0
    commission: float = 0
    error_message: Optional[str] = None
    execution_time: datetime = field(default_factory=datetime.now)


class ExecutionEngine:
    """
    Clean execution layer for Interactive Brokers.
    
    Fixed version that properly handles IB connections and async operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize execution engine.
        
        Args:
            logger: Optional logger for debugging
        """
        self.logger = logger or self._setup_logger()
        self.config = get_config()
        
        # IB connection
        self.ib: Optional[IB] = None
        self.connected = False
        
        # Underlying contract
        self.underlying_contract: Optional[Contract] = None
        self.underlying_symbol = "NQ"
        
        # Order tracking
        self.active_orders: Dict[str, OrderDetails] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Position tracking (just for IB reconciliation)
        self.ib_positions: List = []
        
        # Connection parameters
        self.host = self.config["broker"].host
        self.port = self.config["broker"].paper_port  # Default to paper
        self.client_id = self.config["broker"].client_id
        
        self.logger.info("Execution Engine initialized (Fixed Version)")
    
    def _setup_logger(self) -> logging.Logger:
        """Create logger if none provided."""
        logger = logging.getLogger("ExecutionEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # ========================================================================
    # CONNECTION MANAGEMENT - FIXED VERSION
    # ========================================================================
    
    async def connect(self, paper: bool = True) -> bool:
        """
        Connect to Interactive Brokers - Fixed to use synchronous connection.
        
        Args:
            paper: Use paper trading (True) or live (False)
            
        Returns:
            True if connection successful
            
        Raises:
            TradingConnectionError: If connection fails
        """
        try:
            # Create IB instance
            self.ib = IB()
            
            # Select port based on paper/live
            port = self.config["broker"].paper_port if paper else self.config["broker"].live_port
            
            self.logger.info(f"Connecting to IB at {self.host}:{port} "
                           f"({'Paper' if paper else 'LIVE'})")
            
            # Use SYNCHRONOUS connection (not async) - this is the key fix!
            self.ib.connect(self.host, port, clientId=self.client_id)
            
            self.connected = True
            self.logger.info("✅ Connected to Interactive Brokers")
            
            # Setup underlying contract
            await self._setup_underlying()
            
            # Subscribe to events
            self._setup_event_handlers()
            
            return True
            
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            self.logger.error(error_msg)
            raise TradingConnectionError(error_msg)
    
    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib and self.connected:
            self.logger.info("Disconnecting from IB...")
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected")
    
    def _setup_event_handlers(self) -> None:
        """Setup IB event handlers for order tracking."""
        if self.ib:
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.execDetailsEvent += self._on_execution
            self.ib.errorEvent += self._on_error
    
    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status updates."""
        self.logger.info(f"Order status: {trade.order.orderId} - {trade.orderStatus.status}")
    
    def _on_execution(self, trade: Trade, fill: Fill) -> None:
        """Handle execution reports."""
        self.logger.info(f"Execution: {fill.contract.localSymbol} "
                        f"{fill.execution.shares} @ {fill.execution.avgPrice}")
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract) -> None:
        """Handle IB errors."""
        if errorCode < 2000:  # Informational
            self.logger.info(f"IB Info {errorCode}: {errorString}")
        else:
            self.logger.error(f"IB Error {errorCode}: {errorString}")
    
    # ========================================================================
    # UNDERLYING SETUP - FIXED VERSION
    # ========================================================================
    
    async def _setup_underlying(self) -> None:
        """Setup underlying futures contract - Fixed version."""
        self.logger.info(f"Setting up {self.underlying_symbol} futures...")
        
        # Create future contract (not continuous for simplicity)
        self.underlying_contract = Future(
            symbol=self.underlying_symbol,
            exchange="CME"
        )
        
        # Get contract details
        contracts = self.ib.reqContractDetails(self.underlying_contract)
        
        if contracts:
            # Find front month
            sorted_contracts = sorted(contracts, 
                                    key=lambda x: x.contract.lastTradeDateOrContractMonth)
            
            today = datetime.now().strftime('%Y%m%d')
            for detail in sorted_contracts:
                if detail.contract.lastTradeDateOrContractMonth >= today:
                    self.underlying_contract = detail.contract
                    break
            
            self.logger.info(f"✅ Using {self.underlying_contract.localSymbol} "
                           f"(expires {self.underlying_contract.lastTradeDateOrContractMonth})")
        else:
            self.logger.warning("Could not find NQ contracts")
    
    async def get_underlying_price(self) -> float:
        """
        Get current underlying price - Fixed version.
        
        Returns:
            Current price of underlying
            
        Raises:
            ExecutionError: If price unavailable
        """
        if not self.connected or not self.underlying_contract:
            raise ExecutionError("Not connected or underlying not setup")
        
        # Request market data
        ticker = self.ib.reqMktData(
            self.underlying_contract,
            '',
            False,
            False
        )
        
        # Wait for price using ib.sleep (not asyncio.sleep)
        self.ib.sleep(2)
        
        if ticker.last:
            self.ib.cancelMktData(ticker.contract)
            return ticker.last
        elif ticker.close:
            self.ib.cancelMktData(ticker.contract)
            return ticker.close
        else:
            self.ib.cancelMktData(ticker.contract)
            raise ExecutionError("No price data available")
    
    # ========================================================================
    # OPTION CHAIN - FIXED VERSION
    # ========================================================================
    async def fetch_option_chain(self, expiry: str = None) -> List[OptionContract]:
        """
        Fetch 0DTE option chain for the current underlying.

        Enforces same-day expiry, logs both the underlying future and the option expiry,
        qualifies contracts, requests quotes, filters out junk quotes, and cleans up
        all market data subscriptions.

        Returns:
            List[OptionContract]
        """
        if not self.connected or not self.ib or not self.ib.isConnected():
            raise ExecutionError("Not connected to IB")

        if not self.underlying_contract:
            raise ExecutionError("Underlying contract is not set up")

        # ---- Strict 0DTE guard ----
        today = datetime.now().strftime("%Y%m%d")
        if expiry is None:
            expiry = today
        if expiry != today:
            raise ExecutionError(
                f"Non-0DTE expiry requested ({expiry}). This system is strict 0DTE ({today})."
            )

        # ---- Clear logging so it’s obvious we’re 0DTE on a front future ----
        self.logger.info(
            f"Fetching 0DTE option chain for {expiry} on underlying "
            f"{self.underlying_contract.localSymbol} "
            f"(future expires {self.underlying_contract.lastTradeDateOrContractMonth})..."
        )

        # ---- Price + strike grid ----
        current_price = await self.get_underlying_price()
        strike_increment = 25  # NQ standard
        span = 500             # ±500 points around spot

        min_strike = int((current_price - span) / strike_increment) * strike_increment
        max_strike = int((current_price + span) / strike_increment) * strike_increment
        strikes = list(range(min_strike, max_strike + strike_increment, strike_increment))

        # ---- Build contracts (puts + calls) ----
        raw_contracts: List[Contract] = []
        for strike in strikes:
            for right in ("C", "P"):
                raw_contracts.append(
                    FuturesOption(
                        symbol=self.underlying_symbol,
                        lastTradeDateOrContractMonth=expiry,  # 0DTE enforced above
                        strike=strike,
                        right=right,
                        exchange="CME",
                        tradingClass="NQ",   # NQ options trading class
                        multiplier="20",     # $20/point standard size
                        currency="USD",
                    )
                )

        # ---- Qualify with IB ----
        try:
            qualified = self.ib.qualifyContracts(*raw_contracts)
        except Exception as e:
            raise ExecutionError(f"Failed to qualify option contracts: {e}")

        valid_contracts = [c for c in qualified if getattr(c, "conId", 0) > 0]
        if not valid_contracts:
            self.logger.warning("No valid option contracts after qualification")
            return []

        # ---- Request market data ----
        tickers: List[Tuple[Ticker, Contract]] = []
        try:
            for c in valid_contracts:
                t = self.ib.reqMktData(c, "", False, False)
                tickers.append((t, c))

            # Give IB a moment to populate quotes
            self.ib.sleep(5)

            # ---- Convert to OptionContract domain objects ----
            options: List[OptionContract] = []
            for t, c in tickers:
                bid = getattr(t, "bid", None)
                ask = getattr(t, "ask", None)
                vol = getattr(t, "volume", 0) or 0

                # Basic quote sanity: both sides present & positive
                if bid is None or ask is None or bid <= 0 or ask <= 0:
                    continue

                # Reasonable spread filter: ignore crazy/wide quotes
                mid = (bid + ask) / 2.0
                if mid <= 0:
                    continue
                spread_pct = (ask - bid) / mid * 100.0
                if spread_pct > 40:  # keep it generous; strategy layer can tighten more
                    continue

                options.append(
                    OptionContract(
                        strike=float(c.strike),
                        right=str(c.right),
                        expiry=expiry,
                        bid=float(bid),
                        ask=float(ask),
                        mid=float(mid),
                        volume=int(vol),
                        open_interest=0,  # request separately if needed
                        # greeks can be attached later if you enable modelGreeks requests
                    )
                )

            self.logger.info(f"Fetched {len(options)} option contracts (0DTE)")
            return options

        finally:
            # ---- Always clean up subscriptions ----
            for t, c in tickers:
                try:
                    self.ib.cancelMktData(c)
                except Exception:
                    # best-effort cancel; don't raise on cleanup
                    pass
    
    # async def fetch_option_chain(self, expiry: str = None) -> List[OptionContract]:
    #     """
    #     Fetch option chain for underlying - Fixed version.
        
    #     Args:
    #         expiry: Expiration date (YYYYMMDD), defaults to today for 0DTE
            
    #     Returns:
    #         List of OptionContract objects with Greeks
    #     """
    #     if not self.connected:
    #         raise ExecutionError("Not connected to IB")
        
    #     # Use today for 0DTE if not specified
    #     if not expiry:
    #         expiry = datetime.now().strftime('%Y%m%d')
        
    #     self.logger.info(f"Fetching option chain for {expiry}...")
        
    #     # Generate strikes around current price
    #     current_price = await self.get_underlying_price()
    #     strike_increment = 25  # NQ strikes are every 25 points
        
    #     min_strike = int((current_price - 500) / strike_increment) * strike_increment
    #     max_strike = int((current_price + 500) / strike_increment) * strike_increment
    #     strikes = list(range(min_strike, max_strike + strike_increment, strike_increment))
        
    #     # Create option contracts
    #     option_contracts = []
        
    #     for strike in strikes:
    #         for right in ['C', 'P']:
    #             contract = FuturesOption(
    #                 symbol=self.underlying_symbol,
    #                 lastTradeDateOrContractMonth=expiry,
    #                 strike=strike,
    #                 right=right,
    #                 exchange='CME',
    #                 tradingClass='Q3D',  # Add this - specifies standard NQ options
    #                 multiplier='20'      # Add this - $20 per point (standard size)
    #             )
    #             option_contracts.append(contract)
        
    #     # Get valid contracts
    #     qualified = self.ib.qualifyContracts(*option_contracts)
    #     valid_contracts = [c for c in qualified if c.conId > 0]
        
    #     # Request market data for all
    #     tickers = []
    #     for contract in valid_contracts:
    #         ticker = self.ib.reqMktData(contract, '', False, False)
    #         tickers.append((ticker, contract))
        
    #     # Wait for data using ib.sleep
    #     self.ib.sleep(5)
        
    #     # Convert to OptionContract objects
    #     options = []
    #     for ticker, contract in tickers:
    #         if ticker.bid and ticker.ask and ticker.bid > 0:
    #             # Create OptionContract (simplified - would include Greeks)
    #             opt = OptionContract(
    #                 strike=contract.strike,
    #                 right=contract.right,
    #                 expiry=expiry,
    #                 bid=ticker.bid,
    #                 ask=ticker.ask,
    #                 mid=(ticker.bid + ticker.ask) / 2,
    #                 volume=ticker.volume or 0,
    #                 open_interest=0  # Would need to request separately
    #             )
    #             options.append(opt)
        
    #     # Cancel market data subscriptions
    #     for ticker, contract in tickers:
    #         self.ib.cancelMktData(contract)
        
    #     self.logger.info(f"Fetched {len(options)} option contracts")
    #     return options
    
    # ========================================================================
    # ORDER EXECUTION - Using sync methods where appropriate
    # ========================================================================
    
    async def execute_trade(self, trade_setup: TradeSetup) -> ExecutionResult:
        """
        Execute a complete trade setup - Fixed version.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"EXECUTING {trade_setup.strategy.value.upper()}")
        self.logger.info(f"Contracts: {trade_setup.contracts}")
        self.logger.info("=" * 60)
        
        if not self.connected:
            return ExecutionResult(
                success=False,
                error_message="Not connected to broker"
            )
        
        try:
            # Route to appropriate execution method
            if trade_setup.strategy == StrategyType.IRON_CONDOR:
                result = await self._execute_iron_condor(trade_setup)
            elif trade_setup.strategy == StrategyType.PUT_SPREAD:
                result = await self._execute_put_spread(trade_setup)
            elif trade_setup.strategy == StrategyType.CALL_SPREAD:
                result = await self._execute_call_spread(trade_setup)
            else:
                return ExecutionResult(
                    success=False,
                    error_message=f"Strategy {trade_setup.strategy} not implemented"
                )
            
            # Store in history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_nq_option_contract(self, strike: float, right: str, expiry: str) -> Contract:
        """
        Create a properly specified NQ futures option contract.
        
        This fixes the ambiguity issue by explicitly specifying all parameters
        for standard NQ options (not micro).
        
        Args:
            strike: Strike price
            right: 'P' for put, 'C' for call
            expiry: Expiry date in YYYYMMDD format
            
        Returns:
            Fully specified FuturesOption contract
        """
        contract = FuturesOption(
            symbol='NQ',
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange='CME',
            multiplier='20',  # CRITICAL: Specify standard NQ multiplier
            currency='USD',
            tradingClass='NQ'   # CRITICAL: Specify standard NQ trading class
        )
        return contract

    async def _execute_iron_condor(self, setup: TradeSetup) -> ExecutionResult:
        """
        Execute iron condor trade with proper contract specification and price rounding.
        FIXED: Price rounding and async issues
        """
        try:
            self.logger.info("Executing Iron Condor...")
            
            # Get expiry (0DTE)
            expiry = datetime.now().strftime("%Y%m%d")
            
            # Validate strike values are not None
            if (
                setup.strikes.long_put_strike is None or
                setup.strikes.short_put_strike is None or
                setup.strikes.short_call_strike is None or
                setup.strikes.long_call_strike is None
            ):
                raise ValueError("One or more strike values are None. All strikes must be specified.")

            # Create properly specified contracts
            contracts = [
                self._create_nq_option_contract(setup.strikes.long_put_strike, 'P', expiry),
                self._create_nq_option_contract(setup.strikes.short_put_strike, 'P', expiry),
                self._create_nq_option_contract(setup.strikes.short_call_strike, 'C', expiry),
                self._create_nq_option_contract(setup.strikes.long_call_strike, 'C', expiry)
            ]
            
            # Qualify contracts with IB
            qualified_contracts = []
            for contract in contracts:
                if not self.ib or not self.ib.isConnected():
                    self.logger.error("IB connection not established.")
                    return ExecutionResult(
                        success=False,
                        error_message="IB connection not established"
                    )
                qualified = self.ib.qualifyContracts(contract)
                if not qualified:
                    self.logger.error(f"Failed to qualify contract: {contract}")
                    return ExecutionResult(
                        success=False,
                        error_message=f"Failed to qualify contract: {contract}"
                    )
                qualified_contracts.append(qualified[0])
            
            # Build the combo order
            combo_contract = Contract(
                symbol='NQ',
                secType='BAG',
                exchange='CME',
                currency='USD'
            )
            
            combo_legs = [
                ComboLeg(conId=qualified_contracts[0].conId, ratio=1, action='BUY', exchange='CME'),   # Buy long put
                ComboLeg(conId=qualified_contracts[1].conId, ratio=1, action='SELL', exchange='CME'),  # Sell short put
                ComboLeg(conId=qualified_contracts[2].conId, ratio=1, action='SELL', exchange='CME'),  # Sell short call
                ComboLeg(conId=qualified_contracts[3].conId, ratio=1, action='BUY', exchange='CME')    # Buy long call
            ]
            
            combo_contract.comboLegs = combo_legs
            
            # FIX 1: Round the limit price to nearest 0.25 (valid tick size for NQ options)
            raw_price = setup.credit_per_contract
            rounded_price = self._round_to_tick_size(raw_price)  # Use the helper function
            
            self.logger.info(f"Credit target: ${raw_price:.3f} -> Rounded to: ${rounded_price:.2f}")
            
            # Create and place the order
            order = LimitOrder(
                action='BUY',  # BUY the combo (net credit)
                totalQuantity=setup.contracts,
                lmtPrice=rounded_price
            )
            order.tif = 'DAY'
            
            # Place the order (don't await the placeOrder call - it's not async)
            trade = self.ib.placeOrder(combo_contract, order)
            
            # FIX 2: Use ib.sleep instead of await - this is the proper way with ib_insync
            self.ib.sleep(5)  # Wait for potential fill
            
            # Check if filled or working
            if trade.orderStatus.status in ['Filled', 'PreSubmitted', 'Submitted']:
                if trade.orderStatus.status == 'Filled':
                    self.logger.info(f"✅ Iron Condor FILLED at ${trade.orderStatus.avgFillPrice:.2f}")
                    fill_price = trade.orderStatus.avgFillPrice
                    total_credit = fill_price * setup.contracts * 20  # x20 for NQ multiplier
                else:
                    self.logger.info(f"Iron Condor order status: {trade.orderStatus.status}")
                    # Order is working, we can track it
                    fill_price = rounded_price  # Use limit price as expected fill
                    total_credit = fill_price * setup.contracts * 20
                
                # Generate position ID
                position_id = f"IC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create fill prices dict
                fill_prices = {
                    'long_put': 0,  # Individual fills not tracked for combo
                    'short_put': 0,
                    'short_call': 0,
                    'long_call': 0,
                    'combo': fill_price if trade.orderStatus.status == 'Filled' else rounded_price
                }
                
                return ExecutionResult(
                    success=True,
                    position_id=position_id,
                    orders=[OrderDetails(
                        order_id=str(trade.order.orderId),
                        contract=combo_contract,
                        order=order,
                        trade=trade,
                        strategy_leg="iron_condor_combo"
                    )],
                    fill_prices=fill_prices,
                    total_fill_price=total_credit
                )
            else:
                # Order rejected or failed
                self.logger.error(f"Order failed with status: {trade.orderStatus.status}")
                # Cancel if still active
                if trade.orderStatus.status not in ['Filled', 'Cancelled', 'Inactive']:
                    self.ib.cancelOrder(order)
                
                return ExecutionResult(
                    success=False,
                    error_message=f"Order failed: {trade.orderStatus.status}"
                )
                
        except Exception as e:
            self.logger.error(f"Iron condor execution failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
        
    async def _execute_iron_condor_old(self, setup: TradeSetup) -> Optional[Position]:
        """
        Execute iron condor trade with proper contract specification.
        """
        try:
            self.logger.info("Executing Iron Condor...")
            
            # Get expiry (0DTE)
            expiry = datetime.now().strftime("%Y%m%d")
            
            # Validate strike values are not None
            if (
                setup.strikes.long_put_strike is None or
                setup.strikes.short_put_strike is None or
                setup.strikes.short_call_strike is None or
                setup.strikes.long_call_strike is None
            ):
                raise ValueError("One or more strike values are None. All strikes must be specified.")

            # Create properly specified contracts
            contracts = [
                self._create_nq_option_contract(setup.strikes.long_put_strike, 'P', expiry),
                self._create_nq_option_contract(setup.strikes.short_put_strike, 'P', expiry),
                self._create_nq_option_contract(setup.strikes.short_call_strike, 'C', expiry),
                self._create_nq_option_contract(setup.strikes.long_call_strike, 'C', expiry)
            ]
            
            # Qualify contracts with IB (this should now work without ambiguity)
            qualified_contracts = []
            for contract in contracts:
                if not self.ib or not self.ib.isConnected():
                    self.logger.error("IB connection not established.")
                    return None
                qualified = self.ib.qualifyContracts(contract)
                if not qualified:
                    self.logger.error(f"Failed to qualify contract: {contract}")
                    return None
                qualified_contracts.append(qualified[0])
            
            # Build the combo order
            combo_contract = Contract(
                symbol='NQ',
                secType='BAG',
                exchange='CME',
                currency='USD'
            )
            
            combo_legs = [
                ComboLeg(conId=qualified_contracts[0].conId, ratio=1, action='BUY', exchange='CME'),   # Buy long put
                ComboLeg(conId=qualified_contracts[1].conId, ratio=1, action='SELL', exchange='CME'),  # Sell short put
                ComboLeg(conId=qualified_contracts[2].conId, ratio=1, action='SELL', exchange='CME'),  # Sell short call
                ComboLeg(conId=qualified_contracts[3].conId, ratio=1, action='BUY', exchange='CME')    # Buy long call
            ]
            
            combo_contract.comboLegs = combo_legs
            
            # Create and place the order
            order = LimitOrder(
                action='BUY',
                totalQuantity=setup.contracts,
                lmtPrice=setup.credit_per_contract
            )
            order.tif = 'DAY'
            
            trade = self.ib.placeOrder(combo_contract, order)
            
            # Wait for fill
            self.ib.sleep(2)
            
            if trade.orderStatus.status == 'Filled':
                self.logger.info(f"✅ Iron Condor filled at ${trade.orderStatus.avgFillPrice:.2f}")
                # Create position object...
                return self._create_position_from_trade(trade, setup)
            else:
                self.logger.warning(f"Order status: {trade.orderStatus.status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Iron condor execution failed: {e}")
            return None
    
    async def _execute_put_spread(self, trade_setup: TradeSetup) -> ExecutionResult:
        """Execute put credit spread - Fixed version."""
        strikes = trade_setup.strikes
        contracts = trade_setup.contracts
        
        self.logger.info(f"Executing Put Spread:")
        self.logger.info(f"  Strikes: {strikes.long_put_strike}/{strikes.short_put_strike}")
        
        orders = []
        
        try:
            expiry = datetime.now().strftime('%Y%m%d')
            
            # Short put (SELL)
            short_put_contract = FuturesOption(
                symbol=self.underlying_symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strikes.short_put_strike,
                right='P',
                exchange='CME'
            )
            [short_put_contract] = self.ib.qualifyContracts(short_put_contract)
            
            short_put_order = MarketOrder('SELL', contracts)
            short_put_trade = self.ib.placeOrder(short_put_contract, short_put_order)
            
            orders.append(OrderDetails(
                order_id=f"PS_SP_{datetime.now().timestamp()}",
                contract=short_put_contract,
                order=short_put_order,
                trade=short_put_trade,
                strategy_leg="short_put"
            ))
            
            # Long put (BUY)
            long_put_contract = FuturesOption(
                symbol=self.underlying_symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strikes.long_put_strike,
                right='P',
                exchange='CME'
            )
            [long_put_contract] = self.ib.qualifyContracts(long_put_contract)
            
            long_put_order = MarketOrder('BUY', contracts)
            long_put_trade = self.ib.placeOrder(long_put_contract, long_put_order)
            
            orders.append(OrderDetails(
                order_id=f"PS_LP_{datetime.now().timestamp()}",
                contract=long_put_contract,
                order=long_put_order,
                trade=long_put_trade,
                strategy_leg="long_put"
            ))
            
            # Wait for fills
            self.ib.sleep(3)
            
            # Check fills
            all_filled = all(
                order.trade.orderStatus.status == 'Filled'
                for order in orders
            )
            
            if not all_filled:
                raise FillError("Put spread orders not filled")
            
            # Calculate net credit
            fill_prices = {}
            total_credit = 0
            
            for order in orders:
                fill_price = order.trade.orderStatus.avgFillPrice
                fill_prices[order.strategy_leg] = fill_price
                
                if order.order.action == 'SELL':
                    total_credit += fill_price * contracts * 20
                else:
                    total_credit -= fill_price * contracts * 20
            
            position_id = f"PS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info("✅ Put spread executed successfully!")
            
            return ExecutionResult(
                success=True,
                position_id=position_id,
                orders=orders,
                fill_prices=fill_prices,
                total_fill_price=total_credit
            )
            
        except Exception as e:
            for order in orders:
                if order.trade.orderStatus.status not in ['Filled', 'Cancelled']:
                    self.ib.cancelOrder(order.order)
            
            self.logger.error(f"Put spread execution failed: {e}")
            raise
    
    async def _execute_call_spread(self, trade_setup: TradeSetup) -> ExecutionResult:
        """Execute call credit spread - Fixed version."""
        strikes = trade_setup.strikes
        contracts = trade_setup.contracts
        
        self.logger.info(f"Executing Call Spread:")
        self.logger.info(f"  Strikes: {strikes.short_call_strike}/{strikes.long_call_strike}")
        
        orders = []
        
        try:
            expiry = datetime.now().strftime('%Y%m%d')
            
            # Short call (SELL)
            short_call_contract = FuturesOption(
                symbol=self.underlying_symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strikes.short_call_strike,
                right='C',
                exchange='CME'
            )
            [short_call_contract] = self.ib.qualifyContracts(short_call_contract)
            
            short_call_order = MarketOrder('SELL', contracts)
            short_call_trade = self.ib.placeOrder(short_call_contract, short_call_order)
            
            orders.append(OrderDetails(
                order_id=f"CS_SC_{datetime.now().timestamp()}",
                contract=short_call_contract,
                order=short_call_order,
                trade=short_call_trade,
                strategy_leg="short_call"
            ))
            
            # Long call (BUY)
            long_call_contract = FuturesOption(
                symbol=self.underlying_symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strikes.long_call_strike,
                right='C',
                exchange='CME'
            )
            [long_call_contract] = self.ib.qualifyContracts(long_call_contract)
            
            long_call_order = MarketOrder('BUY', contracts)
            long_call_trade = self.ib.placeOrder(long_call_contract, long_call_order)
            
            orders.append(OrderDetails(
                order_id=f"CS_LC_{datetime.now().timestamp()}",
                contract=long_call_contract,
                order=long_call_order,
                trade=long_call_trade,
                strategy_leg="long_call"
            ))
            
            # Wait for fills
            self.ib.sleep(3)
            
            # Check fills
            all_filled = all(
                order.trade.orderStatus.status == 'Filled'
                for order in orders
            )
            
            if not all_filled:
                raise FillError("Call spread orders not filled")
            
            # Calculate net credit
            fill_prices = {}
            total_credit = 0
            
            for order in orders:
                fill_price = order.trade.orderStatus.avgFillPrice
                fill_prices[order.strategy_leg] = fill_price
                
                if order.order.action == 'SELL':
                    total_credit += fill_price * contracts * 20
                else:
                    total_credit -= fill_price * contracts * 20
            
            position_id = f"CS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info("✅ Call spread executed successfully!")
            
            return ExecutionResult(
                success=True,
                position_id=position_id,
                orders=orders,
                fill_prices=fill_prices,
                total_fill_price=total_credit
            )
            
        except Exception as e:
            for order in orders:
                if order.trade.orderStatus.status not in ['Filled', 'Cancelled']:
                    self.ib.cancelOrder(order.order)
            
            self.logger.error(f"Call spread execution failed: {e}")
            raise
    
    # ========================================================================
    # POSITION MANAGEMENT - Using synchronous methods
    # ========================================================================
    
    async def close_position(
        self,
        position: Position,
        reason: str = "Manual close"
    ) -> ExecutionResult:
        """
        Close an existing position - Fixed version.
        """
        self.logger.info(f"Closing position {position.position_id}: {reason}")
        
        try:
            # Determine what needs to be closed based on strategy
            if position.trade_setup.strategy == StrategyType.IRON_CONDOR:
                result = await self._close_iron_condor(position)
            elif position.trade_setup.strategy == StrategyType.PUT_SPREAD:
                result = await self._close_put_spread(position)
            elif position.trade_setup.strategy == StrategyType.CALL_SPREAD:
                result = await self._close_call_spread(position)
            else:
                return ExecutionResult(
                    success=False,
                    error_message=f"Unknown strategy: {position.trade_setup.strategy}"
                )
            
            self.logger.info(f"✅ Position closed: {position.position_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _close_iron_condor(self, position: Position) -> ExecutionResult:
        """
        Close iron condor position - FIXED version with proper contract specification.
        """
        strikes = position.trade_setup.strikes
        contracts = position.trade_setup.contracts
        
        try:
            expiry = datetime.now().strftime('%Y%m%d')
            
            self.logger.info(f"Closing Iron Condor with strikes: {strikes.short_put_strike}/{strikes.short_call_strike}")
            
            # Create PROPERLY SPECIFIED contracts using the helper
            contracts_to_qualify = [
                self._create_nq_option_contract(strikes.long_put_strike, 'P', expiry),
                self._create_nq_option_contract(strikes.short_put_strike, 'P', expiry),
                self._create_nq_option_contract(strikes.short_call_strike, 'C', expiry),
                self._create_nq_option_contract(strikes.long_call_strike, 'C', expiry)
            ]
            
            # Qualify each contract individually with proper error handling
            qualified = []
            for i, contract in enumerate(contracts_to_qualify):
                result = self.ib.qualifyContracts(contract)
                if not result:
                    self.logger.error(f"Failed to qualify contract {i}: {contract}")
                    # Try to create manually with all details
                    contract.multiplier = '20'
                    contract.currency = 'USD'
                    contract.tradingClass = 'NQ'
                    result = self.ib.qualifyContracts(contract)
                    if not result:
                        raise ExecutionError(f"Cannot qualify contract: {contract}")
                qualified.append(result[0])
            
            # Option 1: Try combo order first (most efficient)
            combo_contract = Contract(
                symbol='NQ',
                secType='BAG',
                exchange='CME',
                currency='USD'
            )
            
            # Create REVERSE combo legs (to close the position)
            combo_legs = [
                ComboLeg(conId=qualified[0].conId, ratio=1, action='SELL', exchange='CME'),  # Sell long put
                ComboLeg(conId=qualified[1].conId, ratio=1, action='BUY', exchange='CME'),   # Buy short put
                ComboLeg(conId=qualified[2].conId, ratio=1, action='BUY', exchange='CME'),   # Buy short call
                ComboLeg(conId=qualified[3].conId, ratio=1, action='SELL', exchange='CME')   # Sell long call
            ]
            
            combo_contract.comboLegs = combo_legs
            
            # Use market order for immediate close (especially if position is threatened)
            order = MarketOrder('SELL', contracts)  # SELL the combo to close
            
            self.logger.info("Placing market order to close Iron Condor...")
            trade = self.ib.placeOrder(combo_contract, order)
            
            # Wait for fill
            self.ib.sleep(5)
            
            if trade.orderStatus.status == 'Filled':
                closing_cost = trade.orderStatus.avgFillPrice * contracts * 20
                self.logger.info(f"✅ Iron Condor closed for ${abs(closing_cost):.2f} {'debit' if closing_cost > 0 else 'credit'}")
                
                return ExecutionResult(
                    success=True,
                    total_fill_price=-abs(closing_cost)  # Negative because it's usually a cost to close
                )
            
            # Option 2: If combo fails, close legs individually
            self.logger.warning("Combo close failed, trying individual legs...")
            
            # Cancel combo order
            if trade.orderStatus.status not in ['Filled', 'Cancelled']:
                self.ib.cancelOrder(order)
            
            return await self._close_iron_condor_legs_individually(qualified, contracts)
            
        except Exception as e:
            self.logger.error(f"Failed to close Iron Condor: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )

    async def _close_iron_condor_legs_individually(self, qualified_contracts: list, num_contracts: int) -> ExecutionResult:
        """
        Fallback method to close iron condor legs individually.
        """
        total_cost = 0
        orders = []
        
        try:
            # Close each leg with market orders
            actions = ['SELL', 'BUY', 'BUY', 'SELL']  # Actions to close
            leg_names = ['long_put', 'short_put', 'short_call', 'long_call']
            
            for contract, action, name in zip(qualified_contracts, actions, leg_names):
                order = MarketOrder(action, num_contracts)
                trade = self.ib.placeOrder(contract, order)
                orders.append((trade, action, name))
                
            # Wait for all fills
            self.ib.sleep(5)
            
            # Calculate total cost
            all_filled = True
            for trade, action, name in orders:
                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    if action == 'BUY':
                        total_cost += fill_price * num_contracts * 20
                    else:
                        total_cost -= fill_price * num_contracts * 20
                    self.logger.info(f"  {name}: {action} @ ${fill_price:.2f}")
                else:
                    all_filled = False
                    self.logger.warning(f"  {name}: NOT FILLED")
            
            if all_filled:
                self.logger.info(f"✅ All legs closed. Net cost: ${abs(total_cost):.2f}")
                return ExecutionResult(
                    success=True,
                    total_fill_price=-abs(total_cost)
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message="Not all legs filled"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to close legs individually: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
        
    async def _close_put_spread(self, position: Position) -> ExecutionResult:
        """
        Close put spread position - FIXED version.
        """
        strikes = position.trade_setup.strikes
        contracts = position.trade_setup.contracts
        
        try:
            expiry = datetime.now().strftime('%Y%m%d')
            
            self.logger.info(f"Closing Put Spread: {strikes.short_put_strike}P / {strikes.long_put_strike}P")
            
            # Create PROPERLY SPECIFIED contracts
            short_put = self._create_nq_option_contract(strikes.short_put_strike, 'P', expiry)
            long_put = self._create_nq_option_contract(strikes.long_put_strike, 'P', expiry)
            
            # Qualify with error handling
            short_put_qualified = self.ib.qualifyContracts(short_put)
            long_put_qualified = self.ib.qualifyContracts(long_put)
            
            if not short_put_qualified or not long_put_qualified:
                raise ExecutionError("Failed to qualify put spread contracts")
            
            short_put = short_put_qualified[0]
            long_put = long_put_qualified[0]
            
            # Try combo order first
            combo_contract = Contract(
                symbol='NQ',
                secType='BAG',
                exchange='CME',
                currency='USD'
            )
            
            combo_legs = [
                ComboLeg(conId=short_put.conId, ratio=1, action='BUY', exchange='CME'),   # Buy back short put
                ComboLeg(conId=long_put.conId, ratio=1, action='SELL', exchange='CME')    # Sell long put
            ]
            
            combo_contract.comboLegs = combo_legs
            
            # Market order for immediate close
            order = MarketOrder('BUY', contracts)  # BUY the combo to close
            
            trade = self.ib.placeOrder(combo_contract, order)
            self.ib.sleep(5)
            
            if trade.orderStatus.status == 'Filled':
                closing_cost = trade.orderStatus.avgFillPrice * contracts * 20
                self.logger.info(f"✅ Put spread closed for ${abs(closing_cost):.2f}")
                return ExecutionResult(
                    success=True,
                    total_fill_price=-abs(closing_cost)
                )
            
            # Fallback to individual orders
            self.logger.warning("Combo failed, closing legs individually...")
            
            if trade.orderStatus.status not in ['Filled', 'Cancelled']:
                self.ib.cancelOrder(order)
            
            # Close individually
            orders = []
            
            # Buy back short put
            order1 = MarketOrder('BUY', contracts)
            trade1 = self.ib.placeOrder(short_put, order1)
            orders.append((trade1, 'BUY'))
            
            # Sell long put
            order2 = MarketOrder('SELL', contracts)
            trade2 = self.ib.placeOrder(long_put, order2)
            orders.append((trade2, 'SELL'))
            
            self.ib.sleep(5)
            
            total_cost = 0
            for trade, action in orders:
                if trade.orderStatus.status == 'Filled':
                    fill = trade.orderStatus.avgFillPrice * contracts * 20
                    total_cost += fill if action == 'BUY' else -fill
            
            return ExecutionResult(
                success=True,
                total_fill_price=-abs(total_cost)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to close put spread: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _close_call_spread(self, position: Position) -> ExecutionResult:
        """
        Close call spread position - FIXED version.
        """
        strikes = position.trade_setup.strikes
        contracts = position.trade_setup.contracts
        
        try:
            expiry = datetime.now().strftime('%Y%m%d')
            
            self.logger.info(f"Closing Call Spread: {strikes.short_call_strike}C / {strikes.long_call_strike}C")
            
            # Create PROPERLY SPECIFIED contracts
            short_call = self._create_nq_option_contract(strikes.short_call_strike, 'C', expiry)
            long_call = self._create_nq_option_contract(strikes.long_call_strike, 'C', expiry)
            
            # Qualify with error handling
            short_call_qualified = self.ib.qualifyContracts(short_call)
            long_call_qualified = self.ib.qualifyContracts(long_call)
            
            if not short_call_qualified or not long_call_qualified:
                raise ExecutionError("Failed to qualify call spread contracts")
            
            short_call = short_call_qualified[0]
            long_call = long_call_qualified[0]
            
            # Try combo order first
            combo_contract = Contract(
                symbol='NQ',
                secType='BAG',
                exchange='CME',
                currency='USD'
            )
            
            combo_legs = [
                ComboLeg(conId=short_call.conId, ratio=1, action='BUY', exchange='CME'),   # Buy back short call
                ComboLeg(conId=long_call.conId, ratio=1, action='SELL', exchange='CME')    # Sell long call
            ]
            
            combo_contract.comboLegs = combo_legs
            
            # Market order for immediate close
            order = MarketOrder('BUY', contracts)  # BUY the combo to close
            
            trade = self.ib.placeOrder(combo_contract, order)
            self.ib.sleep(5)
            
            if trade.orderStatus.status == 'Filled':
                closing_cost = trade.orderStatus.avgFillPrice * contracts * 20
                self.logger.info(f"✅ Call spread closed for ${abs(closing_cost):.2f}")
                return ExecutionResult(
                    success=True,
                    total_fill_price=-abs(closing_cost)
                )
            
            # Fallback to individual orders
            self.logger.warning("Combo failed, closing legs individually...")
            
            if trade.orderStatus.status not in ['Filled', 'Cancelled']:
                self.ib.cancelOrder(order)
            
            # Close individually
            orders = []
            
            # Buy back short call
            order1 = MarketOrder('BUY', contracts)
            trade1 = self.ib.placeOrder(short_call, order1)
            orders.append((trade1, 'BUY'))
            
            # Sell long call
            order2 = MarketOrder('SELL', contracts)
            trade2 = self.ib.placeOrder(long_call, order2)
            orders.append((trade2, 'SELL'))
            
            self.ib.sleep(5)
            
            total_cost = 0
            for trade, action in orders:
                if trade.orderStatus.status == 'Filled':
                    fill = trade.orderStatus.avgFillPrice * contracts * 20
                    total_cost += fill if action == 'BUY' else -fill
            
            return ExecutionResult(
                success=True,
                total_fill_price=-abs(total_cost)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to close call spread: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    # ========================================================================
    # UTILITY METHODS - Using synchronous IB methods
    # ========================================================================
    
    async def get_positions(self) -> List:
        """
        Get current NQ option positions from IB - WORKING VERSION.
        Based on actual IB position structure.
        """
        if not self.connected:
            return []
        
        # Get all positions
        self.ib_positions = self.ib.positions()
        
        self.logger.info(f"Total positions in account: {len(self.ib_positions)}")
        
        # WORKING FILTER based on your debug output
        nq_positions = []
        
        for pos in self.ib_positions:
            # Method 1: Check if it's a FuturesOption with NQ symbol
            if (hasattr(pos.contract, 'symbol') and 
                pos.contract.symbol == 'NQ'):
                # It's an NQ contract - could be option or future
                
                # Check if it's an option by looking for strike/right
                if (hasattr(pos.contract, 'strike') and 
                    hasattr(pos.contract, 'right') and
                    pos.contract.strike > 0):
                    # It's definitely an NQ option
                    nq_positions.append(pos)
                    self.logger.info(f"Found NQ option: Strike={pos.contract.strike}, "
                                f"Right={pos.contract.right}, "
                                f"Qty={pos.position}, "
                                f"AvgCost=${pos.avgCost:.2f}")
            
            # Method 2: Alternative - check contract type name
            elif 'FuturesOption' in str(type(pos.contract)):
                if hasattr(pos.contract, 'symbol') and pos.contract.symbol == 'NQ':
                    nq_positions.append(pos)
                    self.logger.info(f"Found NQ FuturesOption: {pos.contract}")
        
        self.logger.info(f"✅ Filtered to {len(nq_positions)} NQ option positions")
        
        # Debug: Print what we found
        if nq_positions:
            self.logger.info("\n=== NQ POSITIONS DETAIL ===")
            for i, pos in enumerate(nq_positions):
                self.logger.info(f"Position {i+1}:")
                self.logger.info(f"  ConId: {pos.contract.conId}")
                self.logger.info(f"  Strike: {pos.contract.strike if hasattr(pos.contract, 'strike') else 'N/A'}")
                self.logger.info(f"  Right: {pos.contract.right if hasattr(pos.contract, 'right') else 'N/A'}")
                self.logger.info(f"  Expiry: {pos.contract.lastTradeDateOrContractMonth if hasattr(pos.contract, 'lastTradeDateOrContractMonth') else 'N/A'}")
                self.logger.info(f"  Position: {pos.position}")
                self.logger.info(f"  AvgCost: ${pos.avgCost:.2f}")
        
        return nq_positions
    
    async def get_account_summary(self) -> Dict:
        """
        Get account summary from IB - Fixed version.
        """
        if not self.connected:
            return {}
        
        account_values = self.ib.accountSummary()
        
        summary = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower',
                         'UnrealizedPnL', 'RealizedPnL']:
                summary[av.tag] = float(av.value)
        
        return summary
    
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self.connected and self.ib and self.ib.isConnected()
    
    # Add these helper methods to the ExecutionEngine class

    def _round_to_tick_size(self, price: float, tick_size: float = 0.25) -> float:
        """
        Round price to valid tick size for NQ options.
        
        Args:
            price: Raw price to round
            tick_size: Minimum price increment (0.25 for NQ options)
        
        Returns:
            Rounded price
        """
        return round(price / tick_size) * tick_size

    def _validate_and_round_credit(self, credit: float, min_credit: float = 0.25) -> float:
        """
        Validate and round credit to valid tick size.
        
        Args:
            credit: Target credit amount
            min_credit: Minimum acceptable credit
        
        Returns:
            Valid rounded credit
        """
        # Round to tick size
        rounded = self._round_to_tick_size(credit)
        
        # Ensure minimum credit
        if rounded < min_credit:
            self.logger.warning(f"Credit ${rounded:.2f} below minimum ${min_credit:.2f}")
            rounded = min_credit
        
        return rounded

    def _adjust_limit_price_for_fill(self, limit_price: float, is_credit: bool = True) -> float:
        """
        Adjust limit price slightly to improve fill probability.
        
        For credits: reduce slightly
        For debits: increase slightly
        
        Args:
            limit_price: Original limit price
            is_credit: True if collecting credit, False if paying debit
        
        Returns:
            Adjusted limit price
        """
        tick_size = 0.25
        
        if is_credit:
            # For credit trades, reduce by 1 tick to improve fill
            adjusted = limit_price - tick_size
        else:
            # For debit trades, increase by 1 tick
            adjusted = limit_price + tick_size
        
        # Ensure it's still valid
        return self._round_to_tick_size(adjusted)
