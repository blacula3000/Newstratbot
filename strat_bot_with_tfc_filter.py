import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timezone, timedelta
import pytz
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# -------- Logging (rotating) --------
logger = logging.getLogger("trading_bot")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("trading_bot.log", maxBytes=1_000_000, backupCount=3)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.handlers = [handler, console_handler]

# -------- Timeframe mapping --------
VALID_INTERVALS = {"1m":"1m", "5m":"5m", "15m":"15m", "1h":"60m"}
def map_interval(s: str) -> str:
    s = s.strip().lower()
    if s not in VALID_INTERVALS:
        raise ValueError(f"Unsupported timeframe '{s}'. Use one of: {list(VALID_INTERVALS.keys())}")
    return VALID_INTERVALS[s]

# -------- Market Hours Checker --------
class MarketHours:
    """Simple market hours checker for US markets"""
    def __init__(self, timezone_str='America/New_York'):
        self.tz = pytz.timezone(timezone_str)
        self.market_open_hour = 9
        self.market_open_minute = 30
        self.market_close_hour = 16
        self.market_close_minute = 0
    
    def is_market_open(self) -> tuple[bool, str]:
        """Check if US stock market is open"""
        now = datetime.now(self.tz)
        
        if now.weekday() >= 5:  # Weekend
            next_monday = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
            days_until_monday = 7 - now.weekday()
            next_monday += pd.Timedelta(days=days_until_monday)
            return False, f"Weekend. Market opens Monday at {next_monday.strftime('%I:%M %p %Z')}"
        
        current_time = now.time()
        market_open = datetime.combine(now.date(), datetime.min.time().replace(
            hour=self.market_open_hour, minute=self.market_open_minute)).time()
        market_close = datetime.combine(now.date(), datetime.min.time().replace(
            hour=self.market_close_hour, minute=self.market_close_minute)).time()
        
        if market_open <= current_time <= market_close:
            return True, f"Market is OPEN (closes at {market_close.strftime('%I:%M %p')})"
        elif current_time < market_open:
            return False, f"Market opens at {market_open.strftime('%I:%M %p')} ({(datetime.combine(now.date(), market_open) - now).seconds // 60} mins)"
        else:
            tomorrow_open = (now + pd.Timedelta(days=1)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
            return False, f"Market closed. Opens tomorrow at {tomorrow_open.strftime('%I:%M %p %Z')}"
    
    def time_until_open(self) -> int:
        """Returns seconds until market opens"""
        now = datetime.now(self.tz)
        
        if now.weekday() >= 5:  # Weekend
            days_until_monday = 7 - now.weekday()
            next_open = (now + pd.Timedelta(days=days_until_monday)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        elif now.time() >= datetime.min.time().replace(hour=self.market_close_hour):  # After close
            next_open = (now + pd.Timedelta(days=1)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        else:  # Before open same day
            next_open = now.replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        
        return int((next_open - now).total_seconds())

# -------- Timeframe Continuity Filter --------
@dataclass
class TimeframeContinuity:
    """Timeframe continuity status"""
    tfc_up: bool
    tfc_down: bool
    current_price: float
    day_open: float
    week_open: float
    day_direction: str  # "up", "down", "flat"
    week_direction: str  # "up", "down", "flat"
    message: str

class TFCFilter:
    """Timeframe Continuity Filter implementation"""
    
    def __init__(self, symbol: str, timezone_str='America/New_York'):
        self.symbol = symbol
        self.tz = pytz.timezone(timezone_str)
        self._day_open_cache = None
        self._week_open_cache = None
        self._cache_date = None
    
    def get_day_open(self) -> Optional[float]:
        """Get the opening price of the current trading day"""
        try:
            now = datetime.now(self.tz)
            today = now.date()
            
            # Check cache first
            if self._cache_date == today and self._day_open_cache is not None:
                return self._day_open_cache
            
            # For after hours or pre-market, we want the actual trading day open
            # If it's weekend, get Friday's open
            if now.weekday() >= 5:  # Weekend
                days_back = now.weekday() - 4  # Go back to Friday
                target_date = now - timedelta(days=days_back)
                start_date = target_date.strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # Regular weekday - get today's data
                start_date = today.strftime('%Y-%m-%d')
                end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Fetch daily data
            ticker = yf.Ticker(self.symbol)
            daily_data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if not daily_data.empty:
                day_open = float(daily_data['Open'].iloc[0])
                
                # Cache the result
                self._day_open_cache = day_open
                self._cache_date = today
                
                return day_open
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting day open for {self.symbol}: {e}")
            return None
    
    def get_week_open(self) -> Optional[float]:
        """Get the opening price of the current trading week (Monday)"""
        try:
            now = datetime.now(self.tz)
            
            # Find the Monday of current week
            days_since_monday = now.weekday()  # 0=Monday, 6=Sunday
            
            if days_since_monday == 0 and now.hour < 9:
                # It's Monday morning before market open - get last week's Monday
                monday_date = now - timedelta(days=7)
            else:
                # Get this week's Monday
                monday_date = now - timedelta(days=days_since_monday)
            
            # Check cache
            monday_date_str = monday_date.date()
            if hasattr(self, '_week_cache_date') and self._week_cache_date == monday_date_str and self._week_open_cache is not None:
                return self._week_open_cache
            
            # Fetch data starting from Monday
            start_date = monday_date.strftime('%Y-%m-%d')
            end_date = (monday_date + timedelta(days=7)).strftime('%Y-%m-%d')
            
            ticker = yf.Ticker(self.symbol)
            daily_data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if not daily_data.empty:
                week_open = float(daily_data['Open'].iloc[0])  # First trading day of the week
                
                # Cache the result
                self._week_open_cache = week_open
                self._week_cache_date = monday_date_str
                
                return week_open
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting week open for {self.symbol}: {e}")
            return None
    
    def check_continuity(self, current_price: float) -> TimeframeContinuity:
        """
        Check timeframe continuity based on current price vs day/week opens
        
        TFC Logic:
        TFC_UP = (current_close > open_of_day) AND (current_close > open_of_week)
        TFC_DOWN = (current_close < open_of_day) AND (current_close < open_of_week)
        """
        day_open = self.get_day_open()
        week_open = self.get_week_open()
        
        if day_open is None or week_open is None:
            return TimeframeContinuity(
                tfc_up=False,
                tfc_down=False,
                current_price=current_price,
                day_open=day_open or 0,
                week_open=week_open or 0,
                day_direction="unknown",
                week_direction="unknown",
                message="⚠️ TFC: Unable to fetch open prices"
            )
        
        # Calculate TFC conditions
        above_day_open = current_price > day_open
        above_week_open = current_price > week_open
        below_day_open = current_price < day_open
        below_week_open = current_price < week_open
        
        # TFC UP: Both day and week are bullish
        tfc_up = above_day_open and above_week_open
        
        # TFC DOWN: Both day and week are bearish
        tfc_down = below_day_open and below_week_open
        
        # Direction strings
        day_direction = "up" if above_day_open else "down" if below_day_open else "flat"
        week_direction = "up" if above_week_open else "down" if below_week_open else "flat"
        
        # Generate status message
        if tfc_up:
            message = "✅ TFC UP: Day & Week bullish"
        elif tfc_down:
            message = "🔻 TFC DOWN: Day & Week bearish"
        else:
            message = f"⚠️ TFC MIXED: Day {day_direction}, Week {week_direction}"
        
        return TimeframeContinuity(
            tfc_up=tfc_up,
            tfc_down=tfc_down,
            current_price=current_price,
            day_open=day_open,
            week_open=week_open,
            day_direction=day_direction,
            week_direction=week_direction,
            message=message
        )

# -------- STRAT Candle Classification --------
class StratCandle:
    """STRAT methodology candle classification"""
    
    @staticmethod
    def classify_candle(current: pd.Series, previous: pd.Series) -> str:
        """
        Classify candle according to STRAT methodology
        Returns: '1', '2U', '2D', '3', or 'Unknown'
        """
        high = current['High']
        low = current['Low']
        prev_high = previous['High']
        prev_low = previous['Low']
        
        # Type 1 (Inside Bar): High < prev High AND Low > prev Low
        if (high < prev_high) and (low > prev_low):
            return '1'
        
        # Type 3 (Outside Bar): High > prev High AND Low < prev Low  
        elif (high > prev_high) and (low < prev_low):
            return '3'
        
        # Type 2U (Directional Up): Break above prev High, did not break below prev Low
        elif (high > prev_high) and (low >= prev_low):
            return '2U'
        
        # Type 2D (Directional Down): Break below prev Low, did not break above prev High
        elif (low < prev_low) and (high <= prev_high):
            return '2D'
        
        else:
            return 'Unknown'
    
    @staticmethod
    def get_candle_sequence(candles: pd.DataFrame, length: int = 3) -> List[str]:
        """Get sequence of candle classifications"""
        if len(candles) < length + 1:
            return []
        
        sequence = []
        for i in range(len(candles) - length, len(candles)):
            if i == 0:
                sequence.append('Start')  # First candle has no previous
            else:
                candle_type = StratCandle.classify_candle(candles.iloc[i], candles.iloc[i-1])
                sequence.append(candle_type)
        
        return sequence

# -------- Actionable STRAT Entry & Risk Management --------
@dataclass
class StratEntry:
    """Actionable STRAT entry signal with precise conditions"""
    pattern_name: str
    entry_type: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float
    entry_condition_met: bool
    breakout_level: float  # The level that needs to be broken
    pattern_candles: List[pd.Series]  # The candles forming the pattern
    timestamp: pd.Timestamp
    confidence: float
    tfc_approved: bool = False  # NEW: TFC filter approval

class ActionableStratDetector:
    """Detects actionable STRAT patterns with precise entry conditions and TFC filter"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tfc_filter = TFCFilter(symbol)
    
    def check_212_reversal_up(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """2-1-2 Reversal (Up): Down → Inside → Break up"""
        if len(candles) < 3:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(3), 3)
        if len(sequence) < 3:
            return None
        
        # Pattern: 2D -> 1 -> Need breakout above inside bar high
        if sequence[-3] == '2D' and sequence[-2] == '1':
            inside_bar = candles.iloc[-2]  # The inside bar (1)
            inside_high = inside_bar['High']
            inside_low = inside_bar['Low']
            
            # Check if current price breaks above inside bar high
            entry_condition_met = current_price > inside_high
            
            # TFC Filter: Only allow bullish entries when TFC is UP
            tfc_approved = tfc.tfc_up
            
            return StratEntry(
                pattern_name="2-1-2_Reversal_Up",
                entry_type="buy",
                entry_price=inside_high + 0.01,
                stop_loss=inside_low,
                entry_condition_met=entry_condition_met,
                breakout_level=inside_high,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.85,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_212_reversal_down(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """2-1-2 Reversal (Down): Up → Inside → Break down"""
        if len(candles) < 3:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(3), 3)
        if len(sequence) < 3:
            return None
        
        # Pattern: 2U -> 1 -> Need breakout below inside bar low
        if sequence[-3] == '2U' and sequence[-2] == '1':
            inside_bar = candles.iloc[-2]  # The inside bar (1)
            inside_high = inside_bar['High']
            inside_low = inside_bar['Low']
            
            # Check if current price breaks below inside bar low
            entry_condition_met = current_price < inside_low
            
            # TFC Filter: Only allow bearish entries when TFC is DOWN
            tfc_approved = tfc.tfc_down
            
            return StratEntry(
                pattern_name="2-1-2_Reversal_Down",
                entry_type="sell",
                entry_price=inside_low - 0.01,
                stop_loss=inside_high,
                entry_condition_met=entry_condition_met,
                breakout_level=inside_low,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.85,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_312_continuation_up(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """3-1-2 Continuation (Up): Outside → Inside → Break up"""
        if len(candles) < 3:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(3), 3)
        if len(sequence) < 3:
            return None
        
        # Pattern: 3 -> 1 -> Need breakout above inside bar high
        if sequence[-3] == '3' and sequence[-2] == '1':
            inside_bar = candles.iloc[-2]
            inside_high = inside_bar['High']
            inside_low = inside_bar['Low']
            
            entry_condition_met = current_price > inside_high
            tfc_approved = tfc.tfc_up  # TFC Filter
            
            return StratEntry(
                pattern_name="3-1-2_Continuation_Up",
                entry_type="buy",
                entry_price=inside_high + 0.01,
                stop_loss=inside_low,
                entry_condition_met=entry_condition_met,
                breakout_level=inside_high,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.80,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_312_continuation_down(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """3-1-2 Continuation (Down): Outside → Inside → Break down"""
        if len(candles) < 3:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(3), 3)
        if len(sequence) < 3:
            return None
        
        # Pattern: 3 -> 1 -> Need breakout below inside bar low
        if sequence[-3] == '3' and sequence[-2] == '1':
            inside_bar = candles.iloc[-2]
            inside_high = inside_bar['High']
            inside_low = inside_bar['Low']
            
            entry_condition_met = current_price < inside_low
            tfc_approved = tfc.tfc_down  # TFC Filter
            
            return StratEntry(
                pattern_name="3-1-2_Continuation_Down",
                entry_type="sell",
                entry_price=inside_low - 0.01,
                stop_loss=inside_high,
                entry_condition_met=entry_condition_met,
                breakout_level=inside_low,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.80,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_22_reversal_up(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """2-2 Reversal (Up): Down → Break up"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        if sequence[-2] == '2D':
            previous_candle = candles.iloc[-2]
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            entry_condition_met = current_price > prev_high
            tfc_approved = tfc.tfc_up  # TFC Filter
            
            return StratEntry(
                pattern_name="2-2_Reversal_Up",
                entry_type="buy",
                entry_price=prev_high + 0.01,
                stop_loss=prev_low,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_high,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.75,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_22_reversal_down(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """2-2 Reversal (Down): Up → Break down"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        if sequence[-2] == '2U':
            previous_candle = candles.iloc[-2]
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            entry_condition_met = current_price < prev_low
            tfc_approved = tfc.tfc_down  # TFC Filter
            
            return StratEntry(
                pattern_name="2-2_Reversal_Down",
                entry_type="sell",
                entry_price=prev_low - 0.01,
                stop_loss=prev_high,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_low,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.75,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_32_reversal_up(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """3-2 Reversal (Up): Outside → Break up"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        if sequence[-2] == '3':
            previous_candle = candles.iloc[-2]
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            entry_condition_met = current_price > prev_high
            tfc_approved = tfc.tfc_up  # TFC Filter
            
            return StratEntry(
                pattern_name="3-2_Reversal_Up",
                entry_type="buy",
                entry_price=prev_high + 0.01,
                stop_loss=prev_low,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_high,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.70,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def check_32_reversal_down(self, candles: pd.DataFrame, current_price: float, tfc: TimeframeContinuity) -> Optional[StratEntry]:
        """3-2 Reversal (Down): Outside → Break down"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        if sequence[-2] == '3':
            previous_candle = candles.iloc[-2]
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            entry_condition_met = current_price < prev_low
            tfc_approved = tfc.tfc_down  # TFC Filter
            
            return StratEntry(
                pattern_name="3-2_Reversal_Down",
                entry_type="sell",
                entry_price=prev_low - 0.01,
                stop_loss=prev_high,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_low,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.70,
                tfc_approved=tfc_approved
            )
        
        return None
    
    def scan_all_patterns(self, candles: pd.DataFrame, current_price: float) -> List[StratEntry]:
        """Scan for all actionable STRAT patterns with TFC filter"""
        patterns = []
        
        # Get TFC status first
        tfc = self.tfc_filter.check_continuity(current_price)
        
        # Check all pattern types
        methods = [
            self.check_212_reversal_up,
            self.check_212_reversal_down,
            self.check_312_continuation_up, 
            self.check_312_continuation_down,
            self.check_22_reversal_up,
            self.check_22_reversal_down,
            self.check_32_reversal_up,
            self.check_32_reversal_down
        ]
        
        for method in methods:
            try:
                pattern = method(candles, current_price, tfc)
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                logger.error(f"Error in pattern detection {method.__name__}: {e}")
        
        return patterns, tfc

@dataclass
class TradeSignal:
    ts: pd.Timestamp
    symbol: str
    price: float
    pattern: str
    entry_type: str  # buy/sell
    stop_loss: float
    entry_triggered: bool
    confidence: float
    breakout_level: float
    tfc_approved: bool

class TFCStratTradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m', prepost: bool=False, sleep_secs: int=30):
        self.symbol = symbol.upper()
        self.interval = map_interval(timeframe)
        self.prepost = prepost
        self.sleep_secs = sleep_secs
        self.detector = ActionableStratDetector(symbol)
        self._last_bar_ts = None
        self.market_hours = MarketHours()
        self._signal_history = []
        self._active_setups = []  # Track pending setups waiting for breakout
        logger.info(f"TFC STRAT Bot initialized for {self.symbol} @ {self.interval}")

    def fetch_data(self, period="1d", max_retries=3) -> pd.DataFrame:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(self.symbol).history(period=period, interval=self.interval, prepost=self.prepost)
                cols = {c: c.capitalize() for c in data.columns}
                data = data.rename(columns=cols)
                data = data[['Open','High','Low','Close','Volume']].dropna()
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching data after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}. Retrying in {(attempt + 1) * 2} seconds...")
                time.sleep((attempt + 1) * 2)
        return pd.DataFrame()

    def _log_signal_csv(self, sig: TradeSignal, path="tfc_strat_signals.csv"):
        try:
            df = pd.DataFrame([{
                "ts": sig.ts.isoformat(),
                "symbol": sig.symbol,
                "price": sig.price,
                "pattern": sig.pattern,
                "entry_type": sig.entry_type,
                "stop_loss": sig.stop_loss,
                "entry_triggered": sig.entry_triggered,
                "confidence": sig.confidence,
                "breakout_level": sig.breakout_level,
                "tfc_approved": sig.tfc_approved
            }])
            header = not pd.io.common.file_exists(path)
            df.to_csv(path, mode="a", index=False, header=header)
        except Exception as e:
            logger.error(f"Failed to write signal CSV: {e}")

    def run(self):
        print(f"🚀 Starting TFC STRAT bot for {self.symbol} ({self.interval}) — Ctrl+C to stop.")
        logger.info(f"TFC STRAT Bot started for {self.symbol}")

        while True:
            try:
                # Check market hours
                is_open, message = self.market_hours.is_market_open()
                
                if not is_open:
                    print(f"\n⏰ {message}")
                    logger.info(f"Market closed: {message}")
                    sleep_time = min(self.market_hours.time_until_open(), 1800)
                    if sleep_time > 60:
                        print(f"💤 Sleeping for {sleep_time // 60} minutes until market check...")
                    time.sleep(sleep_time)
                    continue

                print(f"✅ {message}")
                
                data = self.fetch_data(period="1d")
                if data.empty:
                    print("No data (feed delayed or symbol unavailable).")
                    time.sleep(self.sleep_secs)
                    continue

                last_ts = data.index[-1]
                if self._last_bar_ts is not None and last_ts == self._last_bar_ts:
                    # Same bar - check for intrabar breakouts
                    current_price = float(data['Close'].iloc[-1])
                    self._check_active_setups(current_price, last_ts)
                    time.sleep(self.sleep_secs)
                    continue
                
                self._last_bar_ts = last_ts
                current_price = float(data['Close'].iloc[-1])
                
                # Get candle sequence for context
                if len(data) >= 4:
                    sequence = StratCandle.get_candle_sequence(data.tail(4), 3)
                else:
                    sequence = []

                print(f"\n📊 Bar close: {last_ts.strftime('%H:%M:%S')}  {self.symbol} ${current_price:.2f}")
                print(f"📈 Sequence: {' -> '.join(sequence[-3:]) if len(sequence) >= 3 else 'N/A'}")
                
                # Scan for actionable patterns (returns patterns and TFC status)
                patterns, tfc = self.detector.scan_all_patterns(data, current_price)
                
                # Display TFC status
                print(f"📊 {tfc.message} | Day: ${tfc.day_open:.2f} Week: ${tfc.week_open:.2f}")
                
                if patterns:
                    for pattern in patterns:
                        if pattern.entry_condition_met and pattern.tfc_approved:
                            # TFC approved entry signal
                            emoji = "🟢" if pattern.entry_type == "buy" else "🔴"
                            print(f"{emoji} TFC APPROVED ENTRY: {pattern.pattern_name}")
                            print(f"   📍 Entry: ${pattern.entry_price:.2f} | 🛑 Stop: ${pattern.stop_loss:.2f}")
                            print(f"   📊 Confidence: {pattern.confidence:.1%} | ✅ TFC: {pattern.tfc_approved}")
                            
                            # Log the signal
                            trade_signal = TradeSignal(
                                ts=last_ts,
                                symbol=self.symbol,
                                price=current_price,
                                pattern=pattern.pattern_name,
                                entry_type=pattern.entry_type,
                                stop_loss=pattern.stop_loss,
                                entry_triggered=True,
                                confidence=pattern.confidence,
                                breakout_level=pattern.breakout_level,
                                tfc_approved=pattern.tfc_approved
                            )
                            self._log_signal_csv(trade_signal)
                            self._signal_history.append(trade_signal)
                            
                            logger.info(f"TFC APPROVED ENTRY: {pattern.pattern_name} {pattern.entry_type.upper()} @ ${current_price:.2f}")
                        
                        elif pattern.entry_condition_met and not pattern.tfc_approved:
                            # Pattern triggered but TFC rejected
                            emoji = "❌"
                            print(f"{emoji} TFC REJECTED: {pattern.pattern_name} | Pattern valid but TFC filter blocks")
                            logger.info(f"TFC REJECTED: {pattern.pattern_name} - Pattern valid but TFC not aligned")
                        
                        elif not pattern.entry_condition_met and pattern.tfc_approved:
                            # Setup detected with TFC approval, waiting for breakout
                            emoji = "🟡"
                            direction = "↗" if pattern.entry_type == "buy" else "↘"
                            print(f"{emoji} TFC APPROVED SETUP: {pattern.pattern_name} {direction}")
                            print(f"   ⏳ Waiting for breakout {'above' if pattern.entry_type == 'buy' else 'below'} ${pattern.breakout_level:.2f}")
                            print(f"   📍 Entry: ${pattern.entry_price:.2f} | 🛑 Stop: ${pattern.stop_loss:.2f} | ✅ TFC Ready")
                            
                            # Add to active setups to monitor
                            self._active_setups.append(pattern)
                            
                            logger.info(f"TFC APPROVED SETUP: {pattern.pattern_name} waiting for ${pattern.breakout_level:.2f} break")
                        
                        else:
                            # Setup detected but TFC not approved
                            emoji = "⚠️"
                            print(f"{emoji} SETUP (TFC Pending): {pattern.pattern_name} | Waiting for TFC alignment")
                else:
                    print("📊 No actionable STRAT patterns detected.")
                
                # Clean old setups
                self._active_setups = [setup for setup in self._active_setups 
                                     if (last_ts - setup.timestamp).seconds < 300 * 5]

                time.sleep(self.sleep_secs)
                
            except KeyboardInterrupt:
                print("\n🛑 TFC STRAT Bot stopped by user.")
                logger.info("TFC STRAT Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Main loop error: {e}")
                time.sleep(self.sleep_secs)
    
    def _check_active_setups(self, current_price: float, timestamp: pd.Timestamp):
        """Check if any TFC-approved active setups have triggered"""
        triggered_setups = []
        
        for setup in self._active_setups:
            if not setup.tfc_approved:
                continue  # Skip setups that don't have TFC approval
                
            if setup.entry_type == "buy" and current_price > setup.breakout_level:
                # Bullish breakout triggered with TFC approval
                triggered_setups.append(setup)
                emoji = "🚀"
                print(f"{emoji} TFC BREAKOUT! {setup.pattern_name} triggered @ ${current_price:.2f}")
                
            elif setup.entry_type == "sell" and current_price < setup.breakout_level:
                # Bearish breakout triggered with TFC approval
                triggered_setups.append(setup)
                emoji = "💥"
                print(f"{emoji} TFC BREAKDOWN! {setup.pattern_name} triggered @ ${current_price:.2f}")
        
        # Remove triggered setups and log them
        for setup in triggered_setups:
            self._active_setups.remove(setup)
            
            # Log the triggered signal
            trade_signal = TradeSignal(
                ts=timestamp,
                symbol=self.symbol,
                price=current_price,
                pattern=setup.pattern_name,
                entry_type=setup.entry_type,
                stop_loss=setup.stop_loss,
                entry_triggered=True,
                confidence=setup.confidence,
                breakout_level=setup.breakout_level,
                tfc_approved=setup.tfc_approved
            )
            self._log_signal_csv(trade_signal)
            self._signal_history.append(trade_signal)
            
            logger.info(f"TFC BREAKOUT TRIGGERED: {setup.pattern_name} {setup.entry_type.upper()} @ ${current_price:.2f}")

    def backtest(self, days=10):
        """Enhanced backtest with TFC filter analysis"""
        period = f"{max(days,1)}d"
        data = self.fetch_data(period=period)
        if data.empty or len(data) < 30:
            print("Not enough data to backtest.")
            return

        wins = 0; losses = 0; total_trades = 0
        tfc_approved_trades = 0; tfc_rejected_trades = 0
        pattern_stats = {}
        
        print(f"\n📈 Running TFC STRAT backtest for {self.symbol} over {days} days...")

        for i in range(5, len(data)-1):  # Need more bars for TFC calculation
            window = data.iloc[:i+1]
            current_price = data['Close'].iloc[i]
            
            # Scan for patterns
            patterns, tfc = self.detector.scan_all_patterns(window, current_price)
            
            for pattern in patterns:
                if pattern.entry_condition_met:  # Pattern triggered
                    total_trades += 1
                    
                    pattern_key = f"{pattern.pattern_name}_{'TFC' if pattern.tfc_approved else 'NoTFC'}"
                    
                    if pattern_key not in pattern_stats:
                        pattern_stats[pattern_key] = {"trades": 0, "wins": 0, "total_pnl": 0}
                    
                    pattern_stats[pattern_key]["trades"] += 1
                    
                    if pattern.tfc_approved:
                        tfc_approved_trades += 1
                    else:
                        tfc_rejected_trades += 1
                        continue  # Skip execution of TFC rejected trades
                    
                    # Simulate the trade (only for TFC approved)
                    entry_price = pattern.entry_price
                    stop_loss = pattern.stop_loss
                    
                    # Look ahead for exit
                    exit_price = None
                    
                    for j in range(i+1, min(i+11, len(data))):
                        next_high = data['High'].iloc[j]
                        next_low = data['Low'].iloc[j]
                        
                        if pattern.entry_type == "buy":
                            if next_low <= stop_loss:
                                exit_price = stop_loss
                                break
                            target = entry_price + (2 * abs(entry_price - stop_loss))
                            if next_high >= target:
                                exit_price = target
                                break
                        else:  # sell
                            if next_high >= stop_loss:
                                exit_price = stop_loss
                                break
                            target = entry_price - (2 * abs(stop_loss - entry_price))
                            if next_low <= target:
                                exit_price = target
                                break
                    
                    if exit_price is None:
                        exit_price = data['Close'].iloc[min(i+10, len(data)-1)]
                    
                    # Calculate P&L
                    if pattern.entry_type == "buy":
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    
                    pattern_stats[pattern_key]["total_pnl"] += pnl
                    
                    if pnl > 0:
                        wins += 1
                        pattern_stats[pattern_key]["wins"] += 1
                    else:
                        losses += 1

        executed_trades = tfc_approved_trades
        hit_rate = (wins / max(executed_trades, 1)) * 100
        
        print(f"\n📊 TFC STRAT Backtest Results ({period}) — {self.symbol}")
        print(f"🎯 Total Pattern Signals: {total_trades}")
        print(f"✅ TFC Approved & Executed: {tfc_approved_trades} | ❌ TFC Rejected: {tfc_rejected_trades}")
        print(f"📈 Executed Trades Results: {executed_trades} trades | {wins} wins | {losses} losses | Hit Rate: {hit_rate:.1f}%")
        print(f"🛡️ TFC Filter Effectiveness: {(tfc_rejected_trades/max(total_trades,1)*100):.1f}% of signals filtered out")
        
        print(f"\n📋 Pattern Performance Comparison:")
        for pattern, stats in pattern_stats.items():
            if stats["trades"] > 0:
                win_rate = (stats["wins"] / stats["trades"]) * 100
                avg_pnl = stats["total_pnl"] / stats["trades"]
                print(f"  {pattern}:")
                print(f"    Trades: {stats['trades']} | Win Rate: {win_rate:.1f}% | Avg P&L: ${avg_pnl:.2f}")

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., SPY): ").upper().strip()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ").strip()

    bot = TFCStratTradingBot(symbol=symbol, timeframe=timeframe, prepost=False, sleep_secs=30)
    mode = input("Run mode — 'live' or 'backtest': ").strip().lower()
    if mode.startswith("b"):
        days = input("How many days (e.g., 10): ").strip()
        try:
            days = int(days)
        except:
            days = 10
        bot.backtest(days=days)
    else:
        bot.run()