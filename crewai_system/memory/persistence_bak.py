# crewai_system/memory/persistence.py
"""
Trade Memory Persistence Layer
==============================
SQLite-based storage for executed trades and outcomes.
Designed for rapid querying during 0DTE decision-making.

Why SQLite?
- Zero configuration needed
- Lightning-fast for our data volume
- Easy to backup and analyze offline
- Perfect for single-user trading system

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import pytz
from dataclasses import field

@dataclass
class TradeRecord:
    """
    Comprehensive trade record for learning and analysis.
    
    Each field captures critical information for agents to learn from.
    """
    # Trade identification
    trade_id: str  # Unique identifier
    position_id: str  # From your Position class
    timestamp: datetime  # When executed (Madrid time)
    timestamp_et: datetime  # When executed (ET for market alignment)
    
    # Trade details
    strategy: str  # 'iron_condor', 'put_spread', 'call_spread'
    strikes: Dict[str, float]  # Strike prices used
    contracts: int  # Number of contracts
    
    # Market conditions at entry
    underlying_price: float  # NQ price at entry
    vix_level: float  # Market volatility
    minutes_to_close: float  # Time pressure factor
    market_regime: str  # From MarketContext analysis
    trend_strength: float  # -100 to +100
    
    # Execution details
    credit_received: float  # Premium collected
    max_risk: float  # Maximum potential loss
    fill_prices: Dict[str, float]  # Actual fill prices
    
    # Agent decisions
    confidence_score: float  # Agent's confidence (0-100)
    reasoning: str  # Why this trade was chosen
    rejected_alternatives: List[str]  # What else was considered
    
    # Outcome (updated when closed)
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # 'profit_target', 'stop_loss', 'time_exit'
    realized_pnl: Optional[float] = None
    win: Optional[bool] = None  # True if profitable
    
    # Learning metadata
    ib_rejections: int = 0  # How many times IB rejected
    adjustments_made: List[str] = field(default_factory=list)  # Mid-trade adjustments
    lessons_learned: Optional[str] = None  # Post-trade analysis


class TradingMemory:
    """
    Persistent memory system for the trading crew.
    
    This class manages all historical trade data, providing agents
    with context for better decision-making.
    """
    
    def __init__(self, db_path: str = "crewai_system/memory/trades.db"):
        """
        Initialize the trading memory system.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.madrid_tz = pytz.timezone("Europe/Madrid")
        self.et_tz = pytz.timezone("US/Eastern")
        
        self._init_database()
    
    def _init_database(self) -> None:
        """
        Create database schema if it doesn't exist.
        
        Schema design optimized for:
        - Fast pattern matching queries
        - Performance analysis by strategy
        - Learning from similar market conditions
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    position_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,  -- ISO format Madrid time
                    timestamp_et TEXT NOT NULL,  -- ISO format ET
                    
                    -- Trade specifications
                    strategy TEXT NOT NULL,
                    strikes TEXT NOT NULL,  -- JSON
                    contracts INTEGER NOT NULL,
                    
                    -- Market snapshot
                    underlying_price REAL NOT NULL,
                    vix_level REAL NOT NULL,
                    minutes_to_close REAL NOT NULL,
                    market_regime TEXT NOT NULL,
                    trend_strength REAL NOT NULL,
                    
                    -- Execution
                    credit_received REAL NOT NULL,
                    max_risk REAL NOT NULL,
                    fill_prices TEXT NOT NULL,  -- JSON
                    
                    -- Agent reasoning
                    confidence_score REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    rejected_alternatives TEXT,  -- JSON
                    
                    -- Outcome (NULL until closed)
                    exit_time TEXT,
                    exit_reason TEXT,
                    realized_pnl REAL,
                    win INTEGER,  -- 1 for win, 0 for loss, NULL if open
                    
                    -- Learning
                    ib_rejections INTEGER DEFAULT 0,
                    adjustments_made TEXT,  -- JSON
                    lessons_learned TEXT,
                    
                    -- Indexes for common queries
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategy 
                ON trades(strategy)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON trades(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_regime 
                ON trades(market_regime, vix_level)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome 
                ON trades(win, realized_pnl)
            """)
            
            conn.commit()
            self.logger.info("Trade database initialized")
    
    def record_trade(self, trade: TradeRecord) -> bool:
        """
        Store a new trade in memory.
        
        Args:
            trade: TradeRecord with execution details
            
        Returns:
            True if successfully stored
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (
                        trade_id, position_id, timestamp, timestamp_et,
                        strategy, strikes, contracts,
                        underlying_price, vix_level, minutes_to_close,
                        market_regime, trend_strength,
                        credit_received, max_risk, fill_prices,
                        confidence_score, reasoning, rejected_alternatives,
                        ib_rejections, adjustments_made
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id,
                    trade.position_id,
                    trade.timestamp.isoformat(),
                    trade.timestamp_et.isoformat(),
                    trade.strategy,
                    json.dumps(trade.strikes),
                    trade.contracts,
                    trade.underlying_price,
                    trade.vix_level,
                    trade.minutes_to_close,
                    trade.market_regime,
                    trade.trend_strength,
                    trade.credit_received,
                    trade.max_risk,
                    json.dumps(trade.fill_prices),
                    trade.confidence_score,
                    trade.reasoning,
                    json.dumps(trade.rejected_alternatives),
                    trade.ib_rejections,
                    json.dumps(trade.adjustments_made)
                ))
                conn.commit()
                
                self.logger.info(f"Recorded trade {trade.trade_id}: "
                               f"{trade.strategy} for ${trade.credit_received:.2f}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to record trade: {e}")
            return False
    
    def update_trade_outcome(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_reason: str,
        realized_pnl: float,
        lessons_learned: Optional[str] = None
    ) -> bool:
        """
        Update trade with closing results.
        
        Args:
            trade_id: Unique trade identifier
            exit_time: When position was closed
            exit_reason: Why it was closed
            realized_pnl: Actual profit/loss
            lessons_learned: Agent's analysis of the trade
            
        Returns:
            True if successfully updated
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE trades
                    SET exit_time = ?,
                        exit_reason = ?,
                        realized_pnl = ?,
                        win = ?,
                        lessons_learned = ?
                    WHERE trade_id = ?
                """, (
                    exit_time.isoformat(),
                    exit_reason,
                    realized_pnl,
                    1 if realized_pnl > 0 else 0,
                    lessons_learned,
                    trade_id
                ))
                conn.commit()
                
                self.logger.info(f"Updated trade {trade_id}: "
                               f"P&L=${realized_pnl:.2f}, Reason={exit_reason}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update trade outcome: {e}")
            return False
    
    def get_similar_trades(
        self,
        market_regime: str,
        vix_level: float,
        strategy: Optional[str] = None,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Find similar historical trades for pattern recognition.
        
        Agents use this to learn from past similar situations.
        
        Args:
            market_regime: Current market classification
            vix_level: Current VIX (will search ±3 points)
            strategy: Specific strategy to filter
            lookback_days: How far back to search
            
        Returns:
            DataFrame with similar trades and outcomes
        """
        query = """
            SELECT * FROM trades
            WHERE market_regime = ?
            AND vix_level BETWEEN ? AND ?
            AND timestamp > datetime('now', '-' || ? || ' days')
        """
        
        params = [
            market_regime,
            vix_level - 3,
            vix_level + 3,
            lookback_days
        ]
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
            # Parse JSON columns
            if not df.empty:
                df['strikes'] = df['strikes'].apply(json.loads)
                df['fill_prices'] = df['fill_prices'].apply(json.loads)
                df['rejected_alternatives'] = df['rejected_alternatives'].apply(
                    lambda x: json.loads(x) if x else []
                )
                df['adjustments_made'] = df['adjustments_made'].apply(
                    lambda x: json.loads(x) if x else []
                )
            
            return df
    
    def get_performance_stats(
        self,
        strategy: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate performance statistics for agent learning.
        
        Args:
            strategy: Filter by specific strategy
            days: Lookback period
            
        Returns:
            Dictionary with win rate, avg P&L, etc.
        """
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(realized_pnl) as avg_pnl,
                MAX(realized_pnl) as best_trade,
                MIN(realized_pnl) as worst_trade,
                SUM(realized_pnl) as total_pnl
            FROM trades
            WHERE exit_time IS NOT NULL
            AND timestamp > datetime('now', '-' || ? || ' days')
        """
        
        params = [days]
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(query, params).fetchone()
            
            if result and result[0] > 0:  # Has trades
                return {
                    "total_trades": result[0],
                    "wins": result[1],
                    "win_rate": (result[1] / result[0] * 100) if result[0] > 0 else 0,
                    "avg_pnl": result[2] or 0,
                    "best_trade": result[3] or 0,
                    "worst_trade": result[4] or 0,
                    "total_pnl": result[5] or 0
                }
            else:
                return {
                    "total_trades": 0,
                    "wins": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "best_trade": 0,
                    "worst_trade": 0,
                    "total_pnl": 0
                }
    
    def get_recent_trades(self, limit: int = 10) -> pd.DataFrame:
        """
        Get most recent trades for context.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame with recent trade history
        """
        query = """
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[limit])


# Initialize global memory instance
TRADING_MEMORY = TradingMemory()