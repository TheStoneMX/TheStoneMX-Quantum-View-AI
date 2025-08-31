# crewai_system/config/timezone_handler.py
"""
Spain-US Market Timezone Coordination
=====================================
Handles timezone conversion and trading window management
for a Spain-based trader operating in US markets.

Critical for 0DTE: Ensures all decisions account for the 
6-7 hour time difference (depending on DST).

Author: CrewAI Trading System
Version: 1.0
Date: December 2024
"""

import pytz
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
import logging


class TimezoneHandler:
    """
    Manages timezone conversions and trading windows.
    
    Essential for coordinating Spain-based operations with US markets.
    """
    
    def __init__(self):
        """Initialize timezone handler with Spain and US Eastern timezones."""
        self.madrid_tz = pytz.timezone("Europe/Madrid")
        self.et_tz = pytz.timezone("US/Eastern")
        self.logger = logging.getLogger(__name__)
        
        # US Market hours in ET
        self.market_open_et = time(9, 30)  # 9:30 AM ET
        self.market_close_et = time(16, 0)  # 4:00 PM ET
        
        # Analysis intervals based on market time (in seconds)
        self.analysis_intervals = {
            "open": 60,       # First 30 min: every 60 seconds
            "morning": 120,   # 10:00-11:30: every 2 minutes
            "midday": 180,    # 11:30-14:30: every 3 minutes
            "power": 90,      # 14:30-15:30: every 90 seconds
            "closing": 30     # Last 30 min: every 30 seconds
        }
        
        # Position monitoring interval (constant)
        self.monitor_interval = 15  # Every 15 seconds with open positions
    
    def get_current_times(self) -> Dict[str, datetime]:
        """
        Get current time in both timezones.
        
        Returns:
            Dictionary with 'madrid' and 'et' datetime objects
        """
        now_utc = datetime.now(pytz.UTC)
        return {
            "madrid": now_utc.astimezone(self.madrid_tz),
            "et": now_utc.astimezone(self.et_tz),
            "utc": now_utc
        }
    
    def is_market_open(self) -> bool:
        """
        Check if US market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """        
        times = self.get_current_times()
        et_now = times["et"]
        
        # Check if it's a weekday (0-4 are Monday-Friday)
        if et_now.weekday() > 4:
            return False
        
        # Check if within market hours
        current_time = et_now.time()
        return self.market_open_et <= current_time <= self.market_close_et
    
    def get_market_phase(self) -> Tuple[str, int]:
        """
        Determine current market phase and analysis interval.
        
        Returns:
            Tuple of (phase_name, interval_seconds)
            
        Phases:
            - 'pre_market': Before 9:30 ET
            - 'open': 9:30-10:00 ET (Market open)
            - 'morning': 10:00-11:30 ET (Morning session)
            - 'midday': 11:30-14:30 ET (Lunch/quiet period)
            - 'power': 14:30-15:30 ET (Power hour)
            - 'closing': 15:30-16:00 ET (Final 30 minutes)
            - 'after_hours': After 16:00 ET
            - 'closed': Weekend
        """
        
        
        times = self.get_current_times()
        et_now = times["et"]
        
        # Check if weekend
        if et_now.weekday() > 4:
            return "closed", 300  # Check every 5 minutes on weekends
        
        current_time = et_now.time()
        
        # Determine phase
        if current_time < self.market_open_et:
            return "pre_market", 300
        elif current_time < time(10, 0):
            return "open", self.analysis_intervals["open"]
        elif current_time < time(11, 30):
            return "morning", self.analysis_intervals["morning"]
        elif current_time < time(14, 30):
            return "midday", self.analysis_intervals["midday"]
        elif current_time < time(15, 30):
            return "power", self.analysis_intervals["power"]
        elif current_time < self.market_close_et:
            return "closing", self.analysis_intervals["closing"]
        else:
            return "after_hours", 300
    
    def minutes_to_close(self) -> float:
        """
        Calculate minutes remaining until market close.
        
        Critical for 0DTE position management.
        
        Returns:
            Minutes to close (0 if market is closed)
        """
        
        if not self.is_market_open():
            return 0
        
        times = self.get_current_times()
        et_now = times["et"]
        
        # Create close time for today
        close_datetime = et_now.replace(
            hour=self.market_close_et.hour,
            minute=self.market_close_et.minute,
            second=0,
            microsecond=0
        )
        
        # Calculate difference
        time_diff = close_datetime - et_now
        return max(0, time_diff.total_seconds() / 60)
    
    def get_minutes_to_market_open(self) -> int:
        """
        Calculate minutes until market opens.
        
        Returns:
            Minutes until market open (negative if market is already open, 0 if weekend)
        """
        times = self.get_current_times()
        et_now = times["et"]
        
        # Check if weekend
        if et_now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return 0
        
        # Create market open time for today
        open_datetime = et_now.replace(
            hour=self.market_open_et.hour,
            minute=self.market_open_et.minute,
            second=0,
            microsecond=0
        )
        
        # If market hasn't opened yet today
        if et_now < open_datetime:
            time_diff = open_datetime - et_now
            return int(time_diff.total_seconds() / 60)
        
        # Market is already open or closed for the day
        return -1
    
    def get_minutes_to_close(self) -> int:
        """
        Calculate minutes until market closes.
        
        Returns:
            Minutes until close (0 if market is closed)
        """
        if not self.is_market_open():
            return 0
        
        times = self.get_current_times()
        et_now = times["et"]
        
        # Create close time for today
        close_datetime = et_now.replace(
            hour=self.market_close_et.hour,
            minute=self.market_close_et.minute,
            second=0,
            microsecond=0
        )
        
        # Calculate difference
        time_diff = close_datetime - et_now
        return max(0, int(time_diff.total_seconds() / 60))
    
    def minutes_since_open(self) -> float:
        """
        Calculate minutes since market opened.
        
        Useful for volatility assessment.
        
        Returns:
            Minutes since open (0 if market is closed)
        """
       
        if not self.is_market_open():
            return 0
        
        times = self.get_current_times()
        et_now = times["et"]
        
        # Create open time for today
        open_datetime = et_now.replace(
            hour=self.market_open_et.hour,
            minute=self.market_open_et.minute,
            second=0,
            microsecond=0
        )
        
        # Calculate difference
        time_diff = et_now - open_datetime
        return max(0, time_diff.total_seconds() / 60)
    
    def get_spain_trading_window(self) -> str:
        """
        Determine quality of current trading window for Spain-based trader.
        
        Returns:
            Window quality: 'optimal', 'good', 'acceptable', 'poor', 'closed'
            
        Considers:
        - Spain local time (fatigue factor)
        - Market phase (opportunity factor)
        - Time to close (0DTE criticality)
        """
        
        times = self.get_current_times()
        madrid_hour = times["madrid"].hour
        market_phase, _ = self.get_market_phase()
        
        # Spain time windows (in Madrid local time)
        # 15:30-17:00: Market open, trader fresh (Good)
        # 17:00-19:00: Mid-day US, evening Spain (Optimal)
        # 19:00-20:30: Power hour US, dinner time Spain (Good)
        # 20:30-21:30: Market close, late evening (Acceptable)
        # 21:30-22:00: Final 30 min, requires focus (Poor but critical)
        
        if market_phase == "closed" or market_phase == "pre_market":
            return "closed"
        
        if 15 <= madrid_hour < 17:
            return "good"  # Fresh trader, market opening volatility
        elif 17 <= madrid_hour < 19:
            return "optimal"  # Best combination of trader alertness and market stability
        elif 19 <= madrid_hour < 21:
            if market_phase == "power":
                return "good"  # Power hour opportunities
            return "acceptable"
        elif 21 <= madrid_hour < 22:
            if market_phase == "closing":
                return "acceptable"  # Critical for 0DTE management
            return "poor"
        else:
            return "closed"
    
    def format_times_for_logging(self) -> str:
        """
        Create formatted string with both times for logging.
        
        Returns:
            String like "Madrid: 17:45 | ET: 11:45 | Market: OPEN (255 min to close)"
        """
        times = self.get_current_times()
        phase, _ = self.get_market_phase()
        minutes_left = self.minutes_to_close()
        
        status = "OPEN" if self.is_market_open() else "CLOSED"
        
        return (
            f"Madrid: {times['madrid'].strftime('%H:%M')} | "
            f"ET: {times['et'].strftime('%H:%M')} | "
            f"Market: {status} ({phase}) | "
            f"{minutes_left:.0f} min to close"
        )
    
    def get_next_analysis_time(self) -> datetime:
        """
        Calculate when next market analysis should occur.
        
        Returns:
            Datetime for next scheduled analysis
        """
        _, interval = self.get_market_phase()
        now = datetime.now(self.madrid_tz)
        return now + timedelta(seconds=interval)


# Global timezone handler instance
TIMEZONE_HANDLER = TimezoneHandler()