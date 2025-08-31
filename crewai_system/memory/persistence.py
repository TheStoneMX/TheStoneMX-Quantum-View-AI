# crewai_system/memory/persistence.py
"""
Trading Memory and Persistence Layer
=====================================
Provides pattern recognition, historical analysis, and learning capabilities
for the veteran trading agents. Uses SQLite for persistence and pandas for
efficient pattern analysis.

Key Features:
- Pattern recognition across VIX regimes
- Strategy performance tracking
- Risk pattern identification
- Aggressive learning with recency weighting
- CTO decision tracking

Author: Quantum View AI Trading System
Version: 2.0
Date: December 2024
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class TradeRecord:
    """Structure for storing trade records."""
    timestamp: str
    strategy: str
    vix_level: float
    vix_regime: str
    market_regime: str
    trend: str
    contracts: int
    credit_received: float
    max_risk: float
    realized_pnl: float
    win: bool
    exit_reason: str
    pattern: str
    confidence: int
    max_adverse_excursion: float = 0.0
    regime_changed: bool = False
    stop_hit: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)


class TradingMemory:
    """
    Comprehensive memory system for pattern-based trading.
    
    Stores, analyzes, and learns from all trading activity to improve
    decision making over time.
    """
    
    def __init__(self, db_path: str = "crewai_system/data/trading_memory.db"):
        """
        Initialize the trading memory system.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.logger = logging.getLogger("TradingMemory")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Cache for frequently accessed patterns
        self._pattern_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        
        self.logger.info(f"Trading memory initialized at {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Main trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    vix_level REAL NOT NULL,
                    vix_regime TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    trend TEXT NOT NULL,
                    contracts INTEGER NOT NULL,
                    credit_received REAL NOT NULL,
                    max_risk REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    win INTEGER NOT NULL,
                    exit_reason TEXT,
                    pattern TEXT,
                    confidence INTEGER,
                    max_adverse_excursion REAL DEFAULT 0,
                    regime_changed INTEGER DEFAULT 0,
                    stop_hit INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    vix REAL NOT NULL,
                    underlying_price REAL NOT NULL,
                    volume INTEGER,
                    high REAL,
                    low REAL,
                    open REAL,
                    close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT UNIQUE NOT NULL,
                    pattern_type TEXT NOT NULL,
                    occurrences INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_vix_change REAL DEFAULT 0,
                    avg_price_change REAL DEFAULT 0,
                    best_strategy TEXT,
                    worst_strategy TEXT,
                    metadata TEXT,
                    last_seen TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # CTO decisions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cto_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence INTEGER,
                    vix_level REAL,
                    vix_regime TEXT,
                    pattern TEXT,
                    team_alignment TEXT,
                    risk_override INTEGER DEFAULT 0,
                    outcome REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Rejections table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    rejection_reason TEXT NOT NULL,
                    vix_level REAL,
                    retry_successful INTEGER DEFAULT 0,
                    alternative_strategy TEXT,
                    key_change TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_vix ON trades(vix_level, vix_regime)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy, win)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(pattern_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cto_timestamp ON cto_decisions(timestamp)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ==================== Pattern Recognition Methods ====================
    
    def get_aggregated_patterns(self) -> Dict[str, Any]:
        """
        Get aggregated market patterns for pattern recognition.
        
        Returns:
            Dictionary of patterns with performance statistics
        """
        # Check cache first
        if self._is_cache_valid("aggregated_patterns"):
            return self._pattern_cache["aggregated_patterns"]
        
        with self._get_connection() as conn:
            # Get VIX regime patterns
            vix_patterns = pd.read_sql_query("""
                SELECT 
                    vix_regime,
                    strategy,
                    COUNT(*) as trades,
                    SUM(win) as wins,
                    AVG(realized_pnl) as avg_pnl,
                    AVG(credit_received) as avg_credit,
                    MAX(realized_pnl) as max_win,
                    MIN(realized_pnl) as max_loss
                FROM trades
                WHERE timestamp > datetime('now', '-180 days')
                GROUP BY vix_regime, strategy
            """, conn)
            
            # Get pattern-specific performance
            pattern_perf = pd.read_sql_query("""
                SELECT 
                    pattern_name,
                    pattern_type,
                    occurrences,
                    wins,
                    total_pnl / NULLIF(occurrences, 0) as avg_pnl,
                    best_strategy,
                    worst_strategy
                FROM patterns
                WHERE occurrences > 5
                ORDER BY avg_pnl DESC
            """, conn)
        
        patterns = {
            "vix_regime_patterns": vix_patterns.to_dict('records') if not vix_patterns.empty else [],
            "named_patterns": pattern_perf.to_dict('records') if not pattern_perf.empty else [],
            "pattern_count": len(pattern_perf),
            "last_update": datetime.now().isoformat()
        }
        
        # Cache the results
        self._pattern_cache["aggregated_patterns"] = patterns
        self._cache_timestamp = datetime.now()
        
        return patterns
    
    def find_similar_setups(self, 
                           vix_range: Tuple[float, float],
                           trend_range: Tuple[float, float],
                           min_similarity: float = 0.7,
                           lookback_days: int = 90) -> List[Dict[str, Any]]:
        """
        Find historically similar market setups.
        
        Args:
            vix_range: (min_vix, max_vix) to search
            trend_range: (min_trend, max_trend) to search  
            min_similarity: Minimum similarity score (0-1)
            lookback_days: Days of history to search
            
        Returns:
            List of similar historical setups with outcomes
        """
        with self._get_connection() as conn:
            query = """
                SELECT 
                    timestamp as date,
                    strategy,
                    vix_level,
                    vix_regime,
                    market_regime,
                    trend,
                    realized_pnl as outcome,
                    win,
                    pattern,
                    confidence
                FROM trades
                WHERE vix_level BETWEEN ? AND ?
                    AND timestamp > datetime('now', ? || ' days')
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(
                query, 
                conn,
                params=(vix_range[0], vix_range[1], f'-{lookback_days}')
            )
        
        if df.empty:
            return []
        
        # Calculate similarity scores
        similar_setups = []
        vix_center = (vix_range[0] + vix_range[1]) / 2
        vix_width = vix_range[1] - vix_range[0]
        
        for _, row in df.iterrows():
            # Calculate VIX similarity (0-1)
            vix_distance = abs(row['vix_level'] - vix_center)
            vix_similarity = max(0, 1 - (vix_distance / vix_width))
            
            # Weight recent trades higher
            days_ago = (datetime.now() - pd.to_datetime(row['date'])).days
            recency_weight = 1.0 if days_ago < 30 else 0.7 if days_ago < 60 else 0.5
            
            # Combined similarity
            similarity = vix_similarity * recency_weight
            
            if similarity >= min_similarity:
                similar_setups.append({
                    "date": row['date'],
                    "strategy": row['strategy'],
                    "vix_level": row['vix_level'],
                    "vix_regime": row['vix_regime'],
                    "outcome": row['outcome'],
                    "win": bool(row['win']),
                    "pattern": row['pattern'],
                    "similarity": similarity
                })
        
        return sorted(similar_setups, key=lambda x: x['similarity'], reverse=True)
    
    def get_pattern_performance(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific pattern.
        
        Args:
            pattern_name: Name of the pattern to analyze
            
        Returns:
            Performance statistics for the pattern
        """
        with self._get_connection() as conn:
            # Get pattern stats
            cursor = conn.execute("""
                SELECT * FROM patterns WHERE pattern_name = ?
            """, (pattern_name,))
            
            pattern = cursor.fetchone()
            
            if not pattern:
                # Create new pattern entry
                conn.execute("""
                    INSERT INTO patterns (pattern_name, pattern_type, occurrences)
                    VALUES (?, 'unknown', 0)
                """, (pattern_name,))
                conn.commit()
                return {"status": "new_pattern", "count": 0}
            
            # Get recent trades with this pattern
            recent_trades = pd.read_sql_query("""
                SELECT 
                    strategy,
                    vix_level,
                    realized_pnl,
                    win
                FROM trades
                WHERE pattern = ?
                    AND timestamp > datetime('now', '-30 days')
                ORDER BY timestamp DESC
                LIMIT 10
            """, conn, params=(pattern_name,))
        
        return {
            "count": pattern['occurrences'],
            "win_rate": pattern['wins'] / max(pattern['occurrences'], 1),
            "avg_pnl": pattern['total_pnl'] / max(pattern['occurrences'], 1),
            "avg_vix_change": pattern['avg_vix_change'],
            "avg_price_change": pattern['avg_price_change'],
            "best_strategy": pattern['best_strategy'],
            "worst_strategy": pattern['worst_strategy'],
            "recent_trades": recent_trades.to_dict('records') if not recent_trades.empty else []
        }
    
    # ==================== Strategy Performance Methods ====================
    
    def get_strategy_patterns(self) -> Dict[str, Any]:
        """
        Get strategy performance patterns across regimes.
        
        Returns:
            Strategy success rates and patterns by regime
        """
        with self._get_connection() as conn:
            # Strategy performance by VIX regime
            df = pd.read_sql_query("""
                SELECT 
                    strategy,
                    vix_regime,
                    COUNT(*) as trades,
                    SUM(win) as wins,
                    AVG(realized_pnl) as avg_pnl,
                    AVG(credit_received) as avg_credit,
                    AVG(max_adverse_excursion) as avg_mae
                FROM trades
                WHERE timestamp > datetime('now', '-90 days')
                GROUP BY strategy, vix_regime
                HAVING trades >= 5
            """, conn)
        
        if df.empty:
            return {}

        # Then calculate std in pandas after the query:
        # Group by strategy and vix_regime to calculate std
        for idx, row in df.iterrows():
            std_query = """
                SELECT realized_pnl FROM trades 
                WHERE strategy = ? AND vix_regime = ?
                AND timestamp > datetime('now', '-90 days')
            """
            pnl_data = pd.read_sql_query(std_query, conn, params=(row['strategy'], row['vix_regime']))
            df.at[idx, 'pnl_std'] = pnl_data['realized_pnl'].std() if not pnl_data.empty else 0
                    
            # Calculate Sharpe-like metric for each strategy/regime combo
            df['risk_adjusted_return'] = df.apply(
                lambda x: x['avg_pnl'] / max(x['pnl_std'], 1) if x['pnl_std'] > 0 else 0,
                axis=1
            )
        
        # Organize by strategy
        patterns = {}
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            patterns[strategy] = {
                "regimes": strategy_data.to_dict('records'),
                "best_regime": strategy_data.loc[strategy_data['avg_pnl'].idxmax()]['vix_regime'] if not strategy_data.empty else None,
                "overall_win_rate": strategy_data['wins'].sum() / max(strategy_data['trades'].sum(), 1),
                "overall_avg_pnl": strategy_data['avg_pnl'].mean()
            }
        
        return patterns
    
    def get_regime_performance(self,
                              vix_range: Tuple[float, float],
                              trend: str,
                              lookback_days: int = 180) -> Dict[str, Any]:
        """
        Get historical performance for strategies in specific regime.
        
        Args:
            vix_range: (min_vix, max_vix) for the regime
            trend: Market trend (bullish/bearish/neutral)
            lookback_days: Days of history to analyze
            
        Returns:
            Performance statistics by strategy in this regime
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    strategy,
                    COUNT(*) as count,
                    SUM(win) as wins,
                    AVG(realized_pnl) as avg_return,
                    AVG(credit_received) as avg_credit,
                    MAX(realized_pnl) as best_trade,
                    MIN(realized_pnl) as worst_trade,
                    AVG(max_adverse_excursion) as avg_mae,
                    AVG(contracts) as avg_size
                FROM trades
                WHERE vix_level BETWEEN ? AND ?
                    AND trend = ?
                    AND timestamp > datetime('now', ? || ' days')
                GROUP BY strategy
                HAVING count >= 3
            """, conn, params=(vix_range[0], vix_range[1], trend, f'-{lookback_days}'))
        
        if df.empty:
            return {}
        
        # Process results
        regime_data = {}
        for _, row in df.iterrows():
            strategy = row['strategy']
            regime_data[strategy] = {
                "count": int(row['count']),
                "win_rate": row['wins'] / row['count'],
                "avg_return": float(row['avg_return']),
                "avg_credit": float(row['avg_credit']),
                "sharpe": row['avg_return'] / max(abs(row['worst_trade']), 1),
                "max_dd": float(row['worst_trade']),
                "avg_mae": float(row['avg_mae']),
                "avg_size": float(row['avg_size']),
                "best_variant": self._get_best_variant(strategy, vix_range, trend)
            }
        
        return regime_data
    
    def get_recent_strategy_performance(self, days: int = 30) -> pd.DataFrame:
        """
        Get recent strategy performance for aggressive learning.
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            DataFrame with recent performance metrics
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    timestamp,
                    strategy,
                    vix_level,
                    vix_regime,
                    realized_pnl as return,
                    win,
                    confidence
                FROM trades
                WHERE timestamp > datetime('now', ? || ' days')
                ORDER BY timestamp DESC
            """, conn, params=(f'-{days}',))
        
        return df
    
    def get_recent_trades(self, days: int = 30) -> pd.DataFrame:
        """
        Get recent trades for analysis.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame of recent trades
        """
        return self.get_recent_strategy_performance(days)
    
    # ==================== Risk Pattern Methods ====================
    
    def get_risk_patterns(self) -> Dict[str, Any]:
        """
        Get historical risk patterns for pattern recognition.
        
        Returns:
            Dictionary of risk patterns and their characteristics
        """
        with self._get_connection() as conn:
            # Get high-risk scenarios
            high_risk = pd.read_sql_query("""
                SELECT 
                    vix_regime,
                    market_regime,
                    COUNT(*) as occurrences,
                    AVG(max_adverse_excursion) as avg_mae,
                    MAX(max_adverse_excursion) as max_mae,
                    SUM(CASE WHEN stop_hit = 1 THEN 1 ELSE 0 END) as stops_hit,
                    SUM(CASE WHEN regime_changed = 1 THEN 1 ELSE 0 END) as regime_changes
                FROM trades
                WHERE max_adverse_excursion > 200
                GROUP BY vix_regime, market_regime
            """, conn)
            
            # Get regime transition patterns
            transitions = pd.read_sql_query("""
                SELECT 
                    vix_regime,
                    COUNT(*) as count,
                    AVG(realized_pnl) as avg_pnl_during_transition
                FROM trades
                WHERE regime_changed = 1
                GROUP BY vix_regime
            """, conn)
        
        patterns = {
            "high_risk_scenarios": high_risk.to_dict('records') if not high_risk.empty else [],
            "regime_transitions": transitions.to_dict('records') if not transitions.empty else [],
            "total_stops_hit": int(high_risk['stops_hit'].sum()) if not high_risk.empty else 0,
            "total_regime_changes": int(high_risk['regime_changes'].sum()) if not high_risk.empty else 0
        }
        
        return patterns
    
    def get_blow_up_patterns(self) -> List[Dict[str, Any]]:
        """
        Get historical blow-up patterns (large losses).
        
        Returns:
            List of patterns that led to significant losses
        """
        with self._get_connection() as conn:
            # Define blow-up as losses > $500 or > 3x average loss
            avg_loss_query = """
                SELECT AVG(ABS(realized_pnl)) as avg_loss
                FROM trades
                WHERE win = 0
            """
            avg_loss = conn.execute(avg_loss_query).fetchone()['avg_loss'] or 100
            
            blow_ups = pd.read_sql_query("""
                SELECT 
                    timestamp,
                    vix_level,
                    vix_regime,
                    market_regime,
                    strategy,
                    realized_pnl as loss,
                    pattern,
                    exit_reason
                FROM trades
                WHERE realized_pnl < -500 
                   OR realized_pnl < ?
                ORDER BY realized_pnl ASC
                LIMIT 20
            """, conn, params=(-3 * avg_loss,))
        
        if blow_ups.empty:
            return []
        
        # Analyze patterns
        patterns = []
        for vix_regime in blow_ups['vix_regime'].unique():
            regime_blowups = blow_ups[blow_ups['vix_regime'] == vix_regime]
            if len(regime_blowups) >= 2:  # Need multiple instances to be a pattern
                patterns.append({
                    "name": f"blow_up_{vix_regime}",
                    "vix_regime": vix_regime,
                    "vix_range": (
                        float(regime_blowups['vix_level'].min()),
                        float(regime_blowups['vix_level'].max())
                    ),
                    "avg_loss": float(regime_blowups['loss'].mean()),
                    "worst_loss": float(regime_blowups['loss'].min()),
                    "occurrences": len(regime_blowups),
                    "common_strategy": regime_blowups['strategy'].mode()[0] if not regime_blowups['strategy'].mode().empty else None,
                    "example": f"{regime_blowups.iloc[0]['timestamp']}: {regime_blowups.iloc[0]['exit_reason']}"
                })
        
        return patterns
    
    def get_similar_risk_scenarios(self,
                                  vix_level: float,
                                  strategy: str,
                                  regime: str,
                                  lookback_days: int = 365) -> List[Dict[str, Any]]:
        """
        Find similar historical risk scenarios.
        
        Args:
            vix_level: Current VIX level
            strategy: Strategy type
            regime: Market regime
            lookback_days: Days to look back
            
        Returns:
            List of similar risk scenarios with outcomes
        """
        with self._get_connection() as conn:
            scenarios = pd.read_sql_query("""
                SELECT 
                    timestamp as date,
                    vix_level,
                    strategy,
                    market_regime,
                    realized_pnl,
                    max_adverse_excursion,
                    stop_hit,
                    regime_changed,
                    exit_reason
                FROM trades
                WHERE ABS(vix_level - ?) < 5
                    AND strategy = ?
                    AND timestamp > datetime('now', ? || ' days')
                ORDER BY max_adverse_excursion DESC
            """, conn, params=(vix_level, strategy, f'-{lookback_days}'))
        
        if scenarios.empty:
            return []
        
        # Process scenarios
        risk_scenarios = []
        for _, row in scenarios.iterrows():
            risk_scenarios.append({
                "date": row['date'],
                "vix_level": float(row['vix_level']),
                "max_loss": float(row['realized_pnl']) if row['realized_pnl'] < 0 else 0,
                "max_adverse_excursion": float(row['max_adverse_excursion']),
                "stop_hit": bool(row['stop_hit']),
                "regime_changed": bool(row['regime_changed']),
                "description": row['exit_reason'] or "Normal exit"
            })
        
        return risk_scenarios
    
    def get_rejection_patterns(self, rejection_reason: str) -> List[Dict[str, Any]]:
        """
        Get patterns of similar rejections and successful adaptations.
        
        Args:
            rejection_reason: The rejection message
            
        Returns:
            List of similar rejections with outcomes
        """
        # Simple keyword matching for rejection similarity
        keywords = rejection_reason.lower().split()
        key_terms = [k for k in keywords if k in ['margin', 'strike', 'width', 'price', 'credit']]
        
        with self._get_connection() as conn:
            if key_terms:
                # Find similar rejections
                query = "SELECT * FROM rejections WHERE "
                conditions = [f"LOWER(rejection_reason) LIKE '%{term}%'" for term in key_terms]
                query += " OR ".join(conditions)
                query += " ORDER BY timestamp DESC LIMIT 10"
                
                rejections = pd.read_sql_query(query, conn)
            else:
                rejections = pd.DataFrame()
        
        if rejections.empty:
            return []
        
        # Process rejections
        patterns = []
        for _, row in rejections.iterrows():
            patterns.append({
                "original_strategy": row['strategy'],
                "retry_successful": bool(row['retry_successful']),
                "successful_strategy": row['alternative_strategy'],
                "key_change": row['key_change']
            })
        
        return patterns
    
    # ==================== CTO Decision Methods ====================
    
    def get_cto_decision_patterns(self) -> Dict[str, Any]:
        """
        Get historical CTO decision patterns.
        
        Returns:
            CTO decision patterns and outcomes
        """
        with self._get_connection() as conn:
            # Get decision patterns
            decisions = pd.read_sql_query("""
                SELECT 
                    decision,
                    vix_regime,
                    team_alignment,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    AVG(outcome) as avg_outcome,
                    SUM(risk_override) as overrides
                FROM cto_decisions
                WHERE timestamp > datetime('now', '-90 days')
                GROUP BY decision, vix_regime, team_alignment
            """, conn)
        
        if decisions.empty:
            return {}
        
        return {
            "decision_patterns": decisions.to_dict('records'),
            "total_decisions": int(decisions['count'].sum()),
            "total_overrides": int(decisions['overrides'].sum()),
            "avg_confidence": float(decisions['avg_confidence'].mean())
        }
    
    def get_override_patterns(self) -> List[Dict[str, Any]]:
        """
        Get patterns of when CTO overrode team recommendations.
        
        Returns:
            List of override patterns with outcomes
        """
        with self._get_connection() as conn:
            overrides = pd.read_sql_query("""
                SELECT 
                    timestamp,
                    vix_level,
                    vix_regime,
                    pattern,
                    team_alignment,
                    outcome
                FROM cto_decisions
                WHERE risk_override = 1
                ORDER BY timestamp DESC
                LIMIT 50
            """, conn)
        
        if overrides.empty:
            return []
        
        # Analyze override success
        patterns = []
        for regime in overrides['vix_regime'].unique():
            regime_overrides = overrides[overrides['vix_regime'] == regime]
            if len(regime_overrides) >= 2:
                patterns.append({
                    "vix_regime": regime,
                    "count": len(regime_overrides),
                    "success_rate": (regime_overrides['outcome'] > 0).mean() if 'outcome' in regime_overrides else 0,
                    "avg_outcome": float(regime_overrides['outcome'].mean()) if 'outcome' in regime_overrides else 0
                })
        
        return patterns
    
    def get_similar_cto_decisions(self,
                                 vix_level: float,
                                 strategy_type: str,
                                 risk_score: float,
                                 lookback_days: int = 90) -> List[Dict[str, Any]]:
        """
        Find similar historical CTO decisions.
        
        Args:
            vix_level: Current VIX level
            strategy_type: Proposed strategy
            risk_score: Risk assessment score
            lookback_days: Days to look back
            
        Returns:
            List of similar decisions with outcomes
        """
        with self._get_connection() as conn:
            # Get similar decisions
            decisions = pd.read_sql_query("""
                SELECT 
                    d.timestamp,
                    d.decision,
                    d.confidence,
                    d.risk_override,
                    d.outcome,
                    t.strategy,
                    t.realized_pnl
                FROM cto_decisions d
                LEFT JOIN trades t ON DATE(d.timestamp) = DATE(t.timestamp)
                WHERE ABS(d.vix_level - ?) < 5
                    AND d.timestamp > datetime('now', ? || ' days')
                ORDER BY d.timestamp DESC
            """, conn, params=(vix_level, f'-{lookback_days}'))
        
        if decisions.empty:
            return []
        
        # Process decisions
        similar = []
        for _, row in decisions.iterrows():
            similar.append({
                "timestamp": row['timestamp'],
                "executed": row['decision'] == 'execute',
                "risk_overridden": bool(row['risk_override']),
                "profitable": row['realized_pnl'] > 0 if row['realized_pnl'] else False,
                "pnl": float(row['realized_pnl']) if row['realized_pnl'] else 0
            })
        
        return similar
    
    # ==================== Storage Methods ====================
    
    def store_trade(self, trade: TradeRecord) -> None:
        """
        Store a completed trade record.
        
        Args:
            trade: TradeRecord object with trade details
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO trades (
                    timestamp, strategy, vix_level, vix_regime, market_regime,
                    trend, contracts, credit_received, max_risk, realized_pnl,
                    win, exit_reason, pattern, confidence, max_adverse_excursion,
                    regime_changed, stop_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.timestamp, trade.strategy, trade.vix_level, trade.vix_regime,
                trade.market_regime, trade.trend, trade.contracts, trade.credit_received,
                trade.max_risk, trade.realized_pnl, int(trade.win), trade.exit_reason,
                trade.pattern, trade.confidence, trade.max_adverse_excursion,
                int(trade.regime_changed), int(trade.stop_hit)
            ))
            
            # Update pattern statistics
            self._update_pattern_stats(trade)
            conn.commit()
        
        # Invalidate cache
        self._invalidate_cache()
        
        self.logger.info(f"Stored trade: {trade.strategy} - P&L: ${trade.realized_pnl:.2f}")
    
    def store_market_data(self, data: Dict[str, Any]) -> None:
        """
        Store market data snapshot.
        
        Args:
            data: Market data dictionary
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO market_data (
                    timestamp, vix, underlying_price, volume, high, low, open, close
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['timestamp'], data['vix'], data['underlying_price'],
                data.get('volume'), data.get('high'), data.get('low'),
                data.get('open'), data.get('close')
            ))
            conn.commit()
    
    def store_cto_decision(self, pattern: Dict[str, Any]) -> None:
        """
        Store CTO decision for learning.
        
        Args:
            pattern: Decision pattern dictionary
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO cto_decisions (
                    timestamp, decision, confidence, vix_level, vix_regime,
                    pattern, team_alignment, risk_override, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern['timestamp'], pattern['decision'], pattern['confidence'],
                pattern.get('vix_level'), pattern.get('vix_regime'),
                pattern.get('pattern'), pattern.get('team_alignment'),
                int(pattern.get('risk_override', False)),
                json.dumps(pattern) if pattern else None
            ))
            conn.commit()
    
    def store_rejection_pattern(self, strategy: Dict[str, Any], reason: str) -> None:
        """
        Store rejection pattern for learning.
        
        Args:
            strategy: Rejected strategy
            reason: Rejection reason
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO rejections (
                    timestamp, strategy, rejection_reason, vix_level
                ) VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                strategy.get('strategy_type', 'unknown'),
                reason,
                strategy.get('market_context', {}).get('vix_level')
            ))
            conn.commit()
    
    def store_retry_pattern(self, 
                          original: Dict[str, Any],
                          alternative: Dict[str, Any],
                          decision: Dict[str, Any]) -> None:
        """
        Store retry pattern after rejection.
        
        Args:
            original: Original strategy
            alternative: Alternative strategy
            decision: Retry decision
        """
        with self._get_connection() as conn:
            # Update the rejection record
            conn.execute("""
                UPDATE rejections
                SET retry_successful = ?,
                    alternative_strategy = ?,
                    key_change = ?
                WHERE timestamp = (
                    SELECT timestamp FROM rejections
                    ORDER BY timestamp DESC LIMIT 1
                )
            """, (
                int(decision.get('approve_retry', False)),
                alternative.get('strategy_type'),
                decision.get('modification', 'unknown')
            ))
            conn.commit()
    
    def store_trade_pattern(self, result: Dict[str, Any], regime: str) -> None:
        """
        Store trade outcome pattern for risk learning.
        
        Args:
            result: Trade result
            regime: Current market regime
        """
        # This is handled by store_trade, but we can add additional pattern tracking
        pattern_name = f"{regime}_{result.get('strategy', 'unknown')}"
        
        with self._get_connection() as conn:
            # Update or create pattern
            conn.execute("""
                INSERT INTO patterns (pattern_name, pattern_type, occurrences, wins, total_pnl)
                VALUES (?, 'trade', 1, ?, ?)
                ON CONFLICT(pattern_name) DO UPDATE SET
                    occurrences = occurrences + 1,
                    wins = wins + ?,
                    total_pnl = total_pnl + ?,
                    last_seen = ?
            """, (
                pattern_name,
                1 if result.get('pnl', 0) > 0 else 0,
                result.get('pnl', 0),
                1 if result.get('pnl', 0) > 0 else 0,
                result.get('pnl', 0),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    # ==================== Query Methods ====================
    
    def get_performance_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance statistics for recent period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance statistics dictionary
        """
        with self._get_connection() as conn:
            stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(win) as wins,
                    SUM(realized_pnl) as total_pnl,
                    AVG(realized_pnl) as avg_pnl,
                    MAX(realized_pnl) as best_trade,
                    MIN(realized_pnl) as worst_trade,
                    AVG(confidence) as avg_confidence
                FROM trades
                WHERE timestamp > datetime('now', ? || ' days')
            """, conn, params=(f'-{days}',))
        
        if stats.empty or stats.iloc[0]['total_trades'] == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0
            }
        
        row = stats.iloc[0]
        return {
            "total_trades": int(row['total_trades']),
            "win_rate": row['wins'] / row['total_trades'] if row['total_trades'] > 0 else 0,
            "total_pnl": float(row['total_pnl']),
            "avg_pnl": float(row['avg_pnl']),
            "best_trade": float(row['best_trade']),
            "worst_trade": float(row['worst_trade']),
            "avg_confidence": float(row['avg_confidence']) if row['avg_confidence'] else 0
        }
    
    def get_similar_trades(self,
                          market_regime: str,
                          vix_level: float,
                          lookback_days: int = 30) -> pd.DataFrame:
        """
        Get similar historical trades.
        
        Args:
            market_regime: Current market regime
            vix_level: Current VIX level
            lookback_days: Days to look back
            
        Returns:
            DataFrame of similar trades
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT * FROM trades
                WHERE market_regime = ?
                    AND ABS(vix_level - ?) < 5
                    AND timestamp > datetime('now', ? || ' days')
                ORDER BY timestamp DESC
            """, conn, params=(market_regime, vix_level, f'-{lookback_days}'))
        
        return df
    
    def get_market_data(self, days: int = 10) -> pd.DataFrame:
        """
        Get recent market data.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame of market data
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT * FROM market_data
                WHERE timestamp > datetime('now', ? || ' days')
                ORDER BY timestamp DESC
            """, conn, params=(f'-{days}',))
        
        return df
    
    # ==================== Helper Methods ====================
    
    def _update_pattern_stats(self, trade: TradeRecord) -> None:
        """Update pattern statistics after a trade."""
        if not trade.pattern:
            return
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO patterns (
                    pattern_name, pattern_type, occurrences, wins, total_pnl
                ) VALUES (?, 'trade', 1, ?, ?)
                ON CONFLICT(pattern_name) DO UPDATE SET
                    occurrences = occurrences + 1,
                    wins = wins + ?,
                    total_pnl = total_pnl + ?,
                    last_seen = ?
            """, (
                trade.pattern,
                1 if trade.win else 0,
                trade.realized_pnl,
                1 if trade.win else 0,
                trade.realized_pnl,
                trade.timestamp
            ))
    
    def _get_best_variant(self, strategy: str, vix_range: Tuple, trend: str) -> str:
        """Get best performing variant of a strategy in regime."""
        # This would need more detailed tracking of strategy variants
        # For now, return standard variants
        variants = {
            "iron_condor": "balanced",
            "put_spread": "aggressive",
            "call_spread": "aggressive",
            "butterfly": "skip_strike"
        }
        return variants.get(strategy, "standard")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._pattern_cache:
            return False
        
        if self._cache_timestamp is None:
            return False
        
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl
    
    def _invalidate_cache(self) -> None:
        """Invalidate the pattern cache."""
        self._pattern_cache = {}
        self._cache_timestamp = None


# Create singleton instance
TRADING_MEMORY = TradingMemory()