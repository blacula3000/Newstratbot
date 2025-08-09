import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timezone
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

class ActionableStratDetector:
    """Detects actionable STRAT patterns with precise entry conditions and stop losses"""
    
    def __init__(self):
        pass
    
    def check_212_reversal_up(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
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
            
            return StratEntry(
                pattern_name="2-1-2_Reversal_Up",
                entry_type="buy",
                entry_price=inside_high + 0.01,  # Entry just above inside bar high
                stop_loss=inside_low,  # Stop at inside bar low
                entry_condition_met=entry_condition_met,
                breakout_level=inside_high,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.85
            )
        
        return None
    
    def check_212_reversal_down(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
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
            
            return StratEntry(
                pattern_name="2-1-2_Reversal_Down",
                entry_type="sell",
                entry_price=inside_low - 0.01,  # Entry just below inside bar low
                stop_loss=inside_high,  # Stop at inside bar high
                entry_condition_met=entry_condition_met,
                breakout_level=inside_low,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.85
            )
        
        return None
    
    def check_312_continuation_up(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
        """3-1-2 Continuation (Up): Outside → Inside → Break up"""
        if len(candles) < 3:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(3), 3)
        if len(sequence) < 3:
            return None
        
        # Pattern: 3 -> 1 -> Need breakout above inside bar high
        if sequence[-3] == '3' and sequence[-2] == '1':
            inside_bar = candles.iloc[-2]  # The inside bar (1)
            inside_high = inside_bar['High']
            inside_low = inside_bar['Low']
            
            # Check if current price breaks above inside bar high
            entry_condition_met = current_price > inside_high
            
            return StratEntry(
                pattern_name="3-1-2_Continuation_Up",
                entry_type="buy",
                entry_price=inside_high + 0.01,
                stop_loss=inside_low,
                entry_condition_met=entry_condition_met,
                breakout_level=inside_high,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.80
            )
        
        return None
    
    def check_312_continuation_down(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
        """3-1-2 Continuation (Down): Outside → Inside → Break down"""
        if len(candles) < 3:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(3), 3)
        if len(sequence) < 3:
            return None
        
        # Pattern: 3 -> 1 -> Need breakout below inside bar low
        if sequence[-3] == '3' and sequence[-2] == '1':
            inside_bar = candles.iloc[-2]  # The inside bar (1)
            inside_high = inside_bar['High']
            inside_low = inside_bar['Low']
            
            # Check if current price breaks below inside bar low
            entry_condition_met = current_price < inside_low
            
            return StratEntry(
                pattern_name="3-1-2_Continuation_Down",
                entry_type="sell",
                entry_price=inside_low - 0.01,
                stop_loss=inside_high,
                entry_condition_met=entry_condition_met,
                breakout_level=inside_low,
                pattern_candles=[candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.80
            )
        
        return None
    
    def check_22_reversal_up(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
        """2-2 Reversal (Up): Down → Break up"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        # Pattern: 2D -> Need breakout above previous candle high
        if sequence[-2] == '2D':
            previous_candle = candles.iloc[-2]  # The 2D candle
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            # Check if current price breaks above previous candle high
            entry_condition_met = current_price > prev_high
            
            return StratEntry(
                pattern_name="2-2_Reversal_Up",
                entry_type="buy",
                entry_price=prev_high + 0.01,
                stop_loss=prev_low,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_high,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.75
            )
        
        return None
    
    def check_22_reversal_down(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
        """2-2 Reversal (Down): Up → Break down"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        # Pattern: 2U -> Need breakout below previous candle low
        if sequence[-2] == '2U':
            previous_candle = candles.iloc[-2]  # The 2U candle
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            # Check if current price breaks below previous candle low
            entry_condition_met = current_price < prev_low
            
            return StratEntry(
                pattern_name="2-2_Reversal_Down",
                entry_type="sell",
                entry_price=prev_low - 0.01,
                stop_loss=prev_high,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_low,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.75
            )
        
        return None
    
    def check_32_reversal_up(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
        """3-2 Reversal (Up): Outside → Break up"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        # Pattern: 3 -> Need breakout above previous candle high
        if sequence[-2] == '3':
            previous_candle = candles.iloc[-2]  # The 3 candle
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            # Check if current price breaks above previous candle high
            entry_condition_met = current_price > prev_high
            
            return StratEntry(
                pattern_name="3-2_Reversal_Up",
                entry_type="buy",
                entry_price=prev_high + 0.01,
                stop_loss=prev_low,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_high,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.70
            )
        
        return None
    
    def check_32_reversal_down(self, candles: pd.DataFrame, current_price: float) -> Optional[StratEntry]:
        """3-2 Reversal (Down): Outside → Break down"""
        if len(candles) < 2:
            return None
        
        sequence = StratCandle.get_candle_sequence(candles.tail(2), 2)
        if len(sequence) < 2:
            return None
        
        # Pattern: 3 -> Need breakout below previous candle low
        if sequence[-2] == '3':
            previous_candle = candles.iloc[-2]  # The 3 candle
            prev_high = previous_candle['High']
            prev_low = previous_candle['Low']
            
            # Check if current price breaks below previous candle low
            entry_condition_met = current_price < prev_low
            
            return StratEntry(
                pattern_name="3-2_Reversal_Down",
                entry_type="sell",
                entry_price=prev_low - 0.01,
                stop_loss=prev_high,
                entry_condition_met=entry_condition_met,
                breakout_level=prev_low,
                pattern_candles=[candles.iloc[-2], candles.iloc[-1]],
                timestamp=candles.index[-1],
                confidence=0.70
            )
        
        return None
    
    def scan_all_patterns(self, candles: pd.DataFrame, current_price: float) -> List[StratEntry]:
        """Scan for all actionable STRAT patterns"""
        patterns = []
        
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
                pattern = method(candles, current_price)
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                logger.error(f"Error in pattern detection {method.__name__}: {e}")
        
        return patterns

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

class ActionableStratTradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m', prepost: bool=False, sleep_secs: int=30):
        self.symbol = symbol.upper()
        self.interval = map_interval(timeframe)
        self.prepost = prepost
        self.sleep_secs = sleep_secs
        self.detector = ActionableStratDetector()
        self._last_bar_ts = None
        self.market_hours = MarketHours()
        self._signal_history = []
        self._active_setups = []  # Track pending setups waiting for breakout
        logger.info(f"Actionable STRAT Bot initialized for {self.symbol} @ {self.interval}")

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

    def _log_signal_csv(self, sig: TradeSignal, path="actionable_strat_signals.csv"):
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
                "breakout_level": sig.breakout_level
            }])
            header = not pd.io.common.file_exists(path)
            df.to_csv(path, mode="a", index=False, header=header)
        except Exception as e:
            logger.error(f"Failed to write signal CSV: {e}")

    def run(self):
        print(f"🚀 Starting Actionable STRAT bot for {self.symbol} ({self.interval}) — Ctrl+C to stop.")
        logger.info(f"Actionable STRAT Bot started for {self.symbol}")

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
                    # Same bar - check for intrabar breakouts on live data
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
                
                # Scan for actionable patterns
                patterns = self.detector.scan_all_patterns(data, current_price)
                
                if patterns:
                    for pattern in patterns:
                        if pattern.entry_condition_met:
                            # Immediate entry signal
                            emoji = "🟢" if pattern.entry_type == "buy" else "🔴"
                            print(f"{emoji} ENTRY TRIGGERED: {pattern.pattern_name}")
                            print(f"   📍 Entry: ${pattern.entry_price:.2f} | 🛑 Stop: ${pattern.stop_loss:.2f}")
                            print(f"   📊 Confidence: {pattern.confidence:.1%} | R/R: {abs(current_price - pattern.stop_loss) / 0.01:.1f}:1")
                            
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
                                breakout_level=pattern.breakout_level
                            )
                            self._log_signal_csv(trade_signal)
                            self._signal_history.append(trade_signal)
                            
                            logger.info(f"ACTIONABLE ENTRY: {pattern.pattern_name} {pattern.entry_type.upper()} @ ${current_price:.2f}")
                        
                        else:
                            # Setup detected, waiting for breakout
                            emoji = "🟡"
                            direction = "↗" if pattern.entry_type == "buy" else "↘"
                            print(f"{emoji} SETUP DETECTED: {pattern.pattern_name} {direction}")
                            print(f"   ⏳ Waiting for breakout {'above' if pattern.entry_type == 'buy' else 'below'} ${pattern.breakout_level:.2f}")
                            print(f"   📍 Entry: ${pattern.entry_price:.2f} | 🛑 Stop: ${pattern.stop_loss:.2f}")
                            
                            # Add to active setups to monitor
                            self._active_setups.append(pattern)
                            
                            logger.info(f"SETUP PENDING: {pattern.pattern_name} waiting for ${pattern.breakout_level:.2f} break")
                else:
                    print("📊 No actionable STRAT patterns detected.")
                
                # Clean old setups (remove setups older than 5 bars)
                self._active_setups = [setup for setup in self._active_setups 
                                     if (last_ts - setup.timestamp).seconds < 300 * 5]  # 5 bars * 5 minutes

                time.sleep(self.sleep_secs)
                
            except KeyboardInterrupt:
                print("\n🛑 Actionable STRAT Bot stopped by user.")
                logger.info("Actionable STRAT Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Main loop error: {e}")
                time.sleep(self.sleep_secs)
    
    def _check_active_setups(self, current_price: float, timestamp: pd.Timestamp):
        """Check if any active setups have triggered"""
        triggered_setups = []
        
        for setup in self._active_setups:
            if setup.entry_type == "buy" and current_price > setup.breakout_level:
                # Bullish breakout triggered
                triggered_setups.append(setup)
                emoji = "🚀"
                print(f"{emoji} BREAKOUT! {setup.pattern_name} triggered @ ${current_price:.2f}")
                
            elif setup.entry_type == "sell" and current_price < setup.breakout_level:
                # Bearish breakout triggered  
                triggered_setups.append(setup)
                emoji = "💥"
                print(f"{emoji} BREAKDOWN! {setup.pattern_name} triggered @ ${current_price:.2f}")
        
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
                breakout_level=setup.breakout_level
            )
            self._log_signal_csv(trade_signal)
            self._signal_history.append(trade_signal)
            
            logger.info(f"BREAKOUT TRIGGERED: {setup.pattern_name} {setup.entry_type.upper()} @ ${current_price:.2f}")

    def backtest(self, days=5):
        """Enhanced backtest with actionable STRAT patterns"""
        period = f"{max(days,1)}d"
        data = self.fetch_data(period=period)
        if data.empty or len(data) < 20:
            print("Not enough data to backtest.")
            return

        wins = 0; losses = 0; total_trades = 0
        pattern_stats = {}
        
        print(f"\n📈 Running Actionable STRAT backtest for {self.symbol} over {days} days...")

        for i in range(3, len(data)-1):  # Need at least 3 bars for patterns
            window = data.iloc[:i+1]
            current_price = data['Close'].iloc[i]
            
            # Scan for patterns
            patterns = self.detector.scan_all_patterns(window, current_price)
            
            for pattern in patterns:
                if pattern.entry_condition_met:  # Only trade triggered entries
                    total_trades += 1
                    
                    if pattern.pattern_name not in pattern_stats:
                        pattern_stats[pattern.pattern_name] = {"trades": 0, "wins": 0, "total_pnl": 0}
                    
                    pattern_stats[pattern.pattern_name]["trades"] += 1
                    
                    # Simulate the trade
                    entry_price = pattern.entry_price
                    stop_loss = pattern.stop_loss
                    
                    # Look ahead to see trade outcome
                    exit_price = None
                    exit_reason = ""
                    
                    # Check next 10 bars for exit
                    for j in range(i+1, min(i+11, len(data))):
                        next_high = data['High'].iloc[j]
                        next_low = data['Low'].iloc[j]
                        
                        if pattern.entry_type == "buy":
                            # Check stop loss first
                            if next_low <= stop_loss:
                                exit_price = stop_loss
                                exit_reason = "Stop Loss"
                                break
                            # Check for 2:1 reward (simple target)
                            target = entry_price + (2 * abs(entry_price - stop_loss))
                            if next_high >= target:
                                exit_price = target
                                exit_reason = "Target Hit"
                                break
                        else:  # sell
                            # Check stop loss first
                            if next_high >= stop_loss:
                                exit_price = stop_loss
                                exit_reason = "Stop Loss"
                                break
                            # Check for 2:1 reward
                            target = entry_price - (2 * abs(stop_loss - entry_price))
                            if next_low <= target:
                                exit_price = target
                                exit_reason = "Target Hit"
                                break
                    
                    if exit_price is None:
                        # No exit found, close at last available price
                        exit_price = data['Close'].iloc[min(i+10, len(data)-1)]
                        exit_reason = "Time Exit"
                    
                    # Calculate P&L
                    if pattern.entry_type == "buy":
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    
                    pattern_stats[pattern.pattern_name]["total_pnl"] += pnl
                    
                    if pnl > 0:
                        wins += 1
                        pattern_stats[pattern.pattern_name]["wins"] += 1
                    else:
                        losses += 1

        hit_rate = (wins / max(total_trades, 1)) * 100
        
        print(f"\n📊 Actionable STRAT Backtest Results ({period}) — {self.symbol}")
        print(f"🎯 Total Trades: {total_trades} | ✅ Wins: {wins} | ❌ Losses: {losses} | 📈 Hit Rate: {hit_rate:.1f}%")
        print(f"\n📋 Pattern Performance:")
        
        for pattern, stats in pattern_stats.items():
            if stats["trades"] > 0:
                win_rate = (stats["wins"] / stats["trades"]) * 100
                avg_pnl = stats["total_pnl"] / stats["trades"]
                print(f"  {pattern}:")
                print(f"    Trades: {stats['trades']} | Win Rate: {win_rate:.1f}% | Avg P&L: ${avg_pnl:.2f}")

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., SPY): ").upper().strip()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ").strip()

    bot = ActionableStratTradingBot(symbol=symbol, timeframe=timeframe, prepost=False, sleep_secs=30)
    mode = input("Run mode — 'live' or 'backtest': ").strip().lower()
    if mode.startswith("b"):
        days = input("How many days (e.g., 5): ").strip()
        try:
            days = int(days)
        except:
            days = 5
        bot.backtest(days=days)
    else:
        bot.run()