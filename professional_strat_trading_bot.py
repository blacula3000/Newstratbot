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
logger = logging.getLogger("professional_strat_bot")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("professional_strat_bot.log", maxBytes=1_000_000, backupCount=3)
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

# -------- Professional STRAT System --------
@dataclass
class TimeframeContinuity:
    """TFC data for timeframe alignment"""
    tfc_up: bool
    tfc_down: bool  
    day_open: float
    week_open: float
    current_close: float

@dataclass 
class StratSignal:
    """Professional STRAT signal with all required data"""
    ts: pd.Timestamp
    symbol: str
    timeframe: str
    pattern: str
    direction: str
    entry: float
    stop: float
    target1: float
    target2: Optional[float]
    tfc_pass: bool
    notes: str = ""

class StratDetector:
    """Professional STRAT detector with refined candle labeling and TFC"""
    
    def __init__(self):
        self.pattern_names = {
            '212_bull': '2-1-2 Bullish Reversal',
            '212_bear': '2-1-2 Bearish Reversal', 
            '312_bull': '3-1-2 Bullish Continuation',
            '312_bear': '3-1-2 Bearish Continuation',
            '22_bull': '2-2 Bullish Reversal',
            '22_bear': '2-2 Bearish Reversal',
            '32_bull': '3-2 Bullish Reversal', 
            '32_bear': '3-2 Bearish Reversal'
        }
    
    def label_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label candles with refined STRAT types"""
        df = df.copy()
        df['candle_type'] = 'Unknown'
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Type 1: Inside Bar
            if curr['High'] < prev['High'] and curr['Low'] > prev['Low']:
                df.at[df.index[i], 'candle_type'] = '1'
            
            # Type 3: Outside Bar  
            elif curr['High'] > prev['High'] and curr['Low'] < prev['Low']:
                df.at[df.index[i], 'candle_type'] = '3'
            
            # Type 2U: Directional Up
            elif curr['High'] > prev['High'] and curr['Low'] >= prev['Low']:
                df.at[df.index[i], 'candle_type'] = '2U'
            
            # Type 2D: Directional Down
            elif curr['Low'] < prev['Low'] and curr['High'] <= prev['High']:
                df.at[df.index[i], 'candle_type'] = '2D'
        
        return df
    
    def get_resampled_opens(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get daily and weekly opens for TFC calculation"""
        try:
            # Resample to daily and get open
            daily = df.resample('D').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last'
            }).dropna()
            
            # Resample to weekly and get open
            weekly = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min', 
                'Close': 'last'
            }).dropna()
            
            day_open = daily['Open'].iloc[-1] if len(daily) > 0 else df['Open'].iloc[0]
            week_open = weekly['Open'].iloc[-1] if len(weekly) > 0 else df['Open'].iloc[0]
            
            return float(day_open), float(week_open)
        except Exception as e:
            logger.warning(f"Error getting resampled opens: {e}")
            return df['Open'].iloc[0], df['Open'].iloc[0]
    
    def calculate_tfc(self, df: pd.DataFrame) -> TimeframeContinuity:
        """Calculate Timeframe Continuity Filter"""
        current_close = df['Close'].iloc[-1]
        day_open, week_open = self.get_resampled_opens(df)
        
        tfc_up = (current_close > day_open) and (current_close > week_open)
        tfc_down = (current_close < day_open) and (current_close < week_open)
        
        return TimeframeContinuity(
            tfc_up=tfc_up,
            tfc_down=tfc_down,
            day_open=day_open,
            week_open=week_open,
            current_close=current_close
        )
    
    def find_pivot_levels(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[float]]:
        """Find pivot highs and lows for target calculation"""
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == highs.iloc[i]:
                pivot_highs.append(df['High'].iloc[i])
            if df['Low'].iloc[i] == lows.iloc[i]:
                pivot_lows.append(df['Low'].iloc[i])
        
        return {
            'pivot_highs': sorted(pivot_highs, reverse=True)[:3],
            'pivot_lows': sorted(pivot_lows)[:3]
        }
    
    def detect_212_reversal(self, df: pd.DataFrame, tfc: TimeframeContinuity) -> List[StratSignal]:
        """Detect 2-1-2 reversal patterns"""
        if len(df) < 3:
            return []
        
        df_labeled = self.label_candles(df)
        signals = []
        
        # Get last 3 candles
        sequence = df_labeled['candle_type'].tail(3).tolist()
        
        if len(sequence) >= 3:
            c1, c2, c3 = sequence[-3], sequence[-2], sequence[-1]
            
            # 2D-1-2U (Bullish reversal)
            if c1 == '2D' and c2 == '1' and c3 == '2U':
                inside_bar = df.iloc[-2]
                entry = inside_bar['High'] + 0.01
                stop = inside_bar['Low']
                target1 = entry + (2 * (entry - stop))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='2-1-2 Bullish Reversal',
                    direction='buy',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=target1 * 1.5,
                    tfc_pass=tfc.tfc_up,
                    notes=f"Inside bar breakout, TFC: {'Pass' if tfc.tfc_up else 'Fail'}"
                ))
            
            # 2U-1-2D (Bearish reversal) 
            elif c1 == '2U' and c2 == '1' and c3 == '2D':
                inside_bar = df.iloc[-2]
                entry = inside_bar['Low'] - 0.01
                stop = inside_bar['High']
                target1 = entry - (2 * (stop - entry))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='2-1-2 Bearish Reversal',
                    direction='sell',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=target1 * 0.5,
                    tfc_pass=tfc.tfc_down,
                    notes=f"Inside bar breakdown, TFC: {'Pass' if tfc.tfc_down else 'Fail'}"
                ))
        
        return signals
    
    def detect_312_continuation(self, df: pd.DataFrame, tfc: TimeframeContinuity) -> List[StratSignal]:
        """Detect 3-1-2 continuation patterns"""
        if len(df) < 3:
            return []
        
        df_labeled = self.label_candles(df)
        signals = []
        
        sequence = df_labeled['candle_type'].tail(3).tolist()
        
        if len(sequence) >= 3:
            c1, c2, c3 = sequence[-3], sequence[-2], sequence[-1]
            
            # 3-1-2U (Bullish continuation)
            if c1 == '3' and c2 == '1' and c3 == '2U':
                inside_bar = df.iloc[-2]
                entry = inside_bar['High'] + 0.01
                stop = inside_bar['Low']
                target1 = entry + (1.5 * (entry - stop))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='3-1-2 Bullish Continuation',
                    direction='buy',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=target1 * 1.3,
                    tfc_pass=tfc.tfc_up,
                    notes=f"Outside bar continuation, TFC: {'Pass' if tfc.tfc_up else 'Fail'}"
                ))
            
            # 3-1-2D (Bearish continuation)
            elif c1 == '3' and c2 == '1' and c3 == '2D':
                inside_bar = df.iloc[-2]
                entry = inside_bar['Low'] - 0.01
                stop = inside_bar['High']
                target1 = entry - (1.5 * (stop - entry))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='3-1-2 Bearish Continuation',
                    direction='sell',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=target1 * 0.7,
                    tfc_pass=tfc.tfc_down,
                    notes=f"Outside bar continuation, TFC: {'Pass' if tfc.tfc_down else 'Fail'}"
                ))
        
        return signals
    
    def detect_22_reversal(self, df: pd.DataFrame, tfc: TimeframeContinuity) -> List[StratSignal]:
        """Detect 2-2 reversal patterns"""
        if len(df) < 2:
            return []
        
        df_labeled = self.label_candles(df)
        signals = []
        
        sequence = df_labeled['candle_type'].tail(2).tolist()
        
        if len(sequence) >= 2:
            c1, c2 = sequence[-2], sequence[-1]
            
            # 2D-2U (Bullish reversal)
            if c1 == '2D' and c2 == '2U':
                prev_candle = df.iloc[-2]
                entry = prev_candle['High'] + 0.01
                stop = prev_candle['Low']
                target1 = entry + (1.5 * (entry - stop))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='2-2 Bullish Reversal',
                    direction='buy',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=None,
                    tfc_pass=tfc.tfc_up,
                    notes=f"Directional reversal, TFC: {'Pass' if tfc.tfc_up else 'Fail'}"
                ))
            
            # 2U-2D (Bearish reversal)
            elif c1 == '2U' and c2 == '2D':
                prev_candle = df.iloc[-2]
                entry = prev_candle['Low'] - 0.01
                stop = prev_candle['High']
                target1 = entry - (1.5 * (stop - entry))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='2-2 Bearish Reversal',
                    direction='sell',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=None,
                    tfc_pass=tfc.tfc_down,
                    notes=f"Directional reversal, TFC: {'Pass' if tfc.tfc_down else 'Fail'}"
                ))
        
        return signals
    
    def detect_32_reversal(self, df: pd.DataFrame, tfc: TimeframeContinuity) -> List[StratSignal]:
        """Detect 3-2 reversal patterns"""
        if len(df) < 2:
            return []
        
        df_labeled = self.label_candles(df)
        signals = []
        
        sequence = df_labeled['candle_type'].tail(2).tolist()
        
        if len(sequence) >= 2:
            c1, c2 = sequence[-2], sequence[-1]
            
            # 3-2U (Bullish reversal)
            if c1 == '3' and c2 == '2U':
                outside_bar = df.iloc[-2]
                entry = outside_bar['High'] + 0.01
                stop = outside_bar['Low']
                target1 = entry + (1.2 * (entry - stop))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='3-2 Bullish Reversal',
                    direction='buy',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=None,
                    tfc_pass=tfc.tfc_up,
                    notes=f"Outside bar reversal, TFC: {'Pass' if tfc.tfc_up else 'Fail'}"
                ))
            
            # 3-2D (Bearish reversal)
            elif c1 == '3' and c2 == '2D':
                outside_bar = df.iloc[-2]
                entry = outside_bar['Low'] - 0.01
                stop = outside_bar['High']
                target1 = entry - (1.2 * (stop - entry))
                
                signals.append(StratSignal(
                    ts=df.index[-1],
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    timeframe=df.attrs.get('timeframe', '5m'),
                    pattern='3-2 Bearish Reversal',
                    direction='sell',
                    entry=entry,
                    stop=stop,
                    target1=target1,
                    target2=None,
                    tfc_pass=tfc.tfc_down,
                    notes=f"Outside bar reversal, TFC: {'Pass' if tfc.tfc_down else 'Fail'}"
                ))
        
        return signals
    
    def scan_all_patterns(self, df: pd.DataFrame) -> List[StratSignal]:
        """Scan for all STRAT patterns with TFC filter"""
        df.attrs['symbol'] = df.attrs.get('symbol', 'UNKNOWN')
        df.attrs['timeframe'] = df.attrs.get('timeframe', '5m')
        
        # Calculate TFC
        tfc = self.calculate_tfc(df)
        
        all_signals = []
        
        # Detect all pattern types
        all_signals.extend(self.detect_212_reversal(df, tfc))
        all_signals.extend(self.detect_312_continuation(df, tfc))
        all_signals.extend(self.detect_22_reversal(df, tfc))
        all_signals.extend(self.detect_32_reversal(df, tfc))
        
        return all_signals

@dataclass
class TradeSignal:
    ts: pd.Timestamp
    symbol: str
    price: float
    pattern: str
    direction: str
    entry: float
    stop: float
    target1: float
    target2: Optional[float]
    tfc_pass: bool
    notes: str = ""

class ProfessionalStratTradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m', prepost: bool=False, sleep_secs: int=30):
        self.symbol = symbol.upper()
        self.interval = map_interval(timeframe)
        self.prepost = prepost
        self.sleep_secs = sleep_secs
        self.detector = StratDetector()
        self._last_bar_ts = None
        self.market_hours = MarketHours()
        self._signal_history = []
        self._active_setups = []
        logger.info(f"Professional STRAT Bot initialized for {self.symbol} @ {self.interval}")

    def fetch_data(self, period="1d", max_retries=3) -> pd.DataFrame:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(self.symbol).history(period=period, interval=self.interval, prepost=self.prepost)
                cols = {c: c.capitalize() for c in data.columns}
                data = data.rename(columns=cols)
                data = data[['Open','High','Low','Close','Volume']].dropna()
                data.attrs['symbol'] = self.symbol
                data.attrs['timeframe'] = self.interval
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching data after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}. Retrying in {(attempt + 1) * 2} seconds...")
                time.sleep((attempt + 1) * 2)
        return pd.DataFrame()

    def _log_signal_csv(self, sig: TradeSignal, path="professional_strat_signals.csv"):
        try:
            df = pd.DataFrame([{
                "ts": sig.ts.isoformat(),
                "symbol": sig.symbol,
                "price": sig.price,
                "pattern": sig.pattern,
                "direction": sig.direction,
                "entry": sig.entry,
                "stop": sig.stop,
                "target1": sig.target1,
                "target2": sig.target2,
                "tfc_pass": sig.tfc_pass,
                "notes": sig.notes
            }])
            header = not pd.io.common.file_exists(path)
            df.to_csv(path, mode="a", index=False, header=header)
        except Exception as e:
            logger.error(f"Failed to write signal CSV: {e}")

    def run(self):
        print(f"🚀 Starting Professional STRAT bot for {self.symbol} ({self.interval}) — Ctrl+C to stop.")
        logger.info(f"Professional STRAT Bot started for {self.symbol}")

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
                
                data = self.fetch_data(period="2d")  # Get more data for TFC calculation
                if data.empty:
                    print("No data (feed delayed or symbol unavailable).")
                    time.sleep(self.sleep_secs)
                    continue

                last_ts = data.index[-1]
                if self._last_bar_ts is not None and last_ts == self._last_bar_ts:
                    time.sleep(self.sleep_secs)
                    continue
                
                self._last_bar_ts = last_ts
                current_price = float(data['Close'].iloc[-1])
                
                # Get candle sequence for context
                df_labeled = self.detector.label_candles(data.tail(5))
                sequence = df_labeled['candle_type'].tail(3).tolist()
                
                print(f"\n📊 Bar close: {last_ts.strftime('%H:%M:%S')}  {self.symbol} ${current_price:.2f}")
                print(f"📈 Sequence: {' -> '.join(sequence) if sequence else 'N/A'}")
                
                # Scan for STRAT patterns
                signals = self.detector.scan_all_patterns(data)
                
                if signals:
                    for signal in signals:
                        # Only show TFC-passing signals prominently
                        if signal.tfc_pass:
                            emoji = "🟢✅" if signal.direction == "buy" else "🔴✅"
                            print(f"{emoji} TFC PASS: {signal.pattern}")
                        else:
                            emoji = "🟡" if signal.direction == "buy" else "🟠"
                            print(f"{emoji} TFC FAIL: {signal.pattern}")
                        
                        print(f"   📍 Entry: ${signal.entry:.2f} | 🛑 Stop: ${signal.stop:.2f} | 🎯 Target: ${signal.target1:.2f}")
                        if signal.target2:
                            print(f"   🎯 Target2: ${signal.target2:.2f}")
                        print(f"   📊 TFC Status: {'PASS' if signal.tfc_pass else 'FAIL'} | Notes: {signal.notes}")
                        
                        # Convert to TradeSignal and log
                        trade_signal = TradeSignal(
                            ts=last_ts,
                            symbol=self.symbol,
                            price=current_price,
                            pattern=signal.pattern,
                            direction=signal.direction,
                            entry=signal.entry,
                            stop=signal.stop,
                            target1=signal.target1,
                            target2=signal.target2,
                            tfc_pass=signal.tfc_pass,
                            notes=signal.notes
                        )
                        self._log_signal_csv(trade_signal)
                        self._signal_history.append(trade_signal)
                        
                        logger.info(f"PROFESSIONAL STRAT: {signal.pattern} {signal.direction.upper()} @ ${current_price:.2f} (TFC: {'PASS' if signal.tfc_pass else 'FAIL'})")
                else:
                    print("📊 No STRAT patterns detected.")

                time.sleep(self.sleep_secs)
                
            except KeyboardInterrupt:
                print("\n🛑 Professional STRAT Bot stopped by user.")
                logger.info("Professional STRAT Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Main loop error: {e}")
                time.sleep(self.sleep_secs)

    def backtest(self, days=5):
        """Enhanced backtest with professional STRAT patterns and TFC analysis"""
        period = f"{max(days,1)}d"
        data = self.fetch_data(period=period)
        if data.empty or len(data) < 20:
            print("Not enough data to backtest.")
            return

        wins = 0; losses = 0; total_trades = 0
        tfc_wins = 0; tfc_losses = 0; tfc_trades = 0
        pattern_stats = {}
        
        print(f"\n📈 Running Professional STRAT backtest for {self.symbol} over {days} days...")

        for i in range(10, len(data)-1):  # Need sufficient history for TFC
            window = data.iloc[:i+1]
            
            # Scan for patterns
            signals = self.detector.scan_all_patterns(window)
            
            for signal in signals:
                total_trades += 1
                
                if signal.pattern not in pattern_stats:
                    pattern_stats[signal.pattern] = {"trades": 0, "wins": 0, "tfc_trades": 0, "tfc_wins": 0}
                
                pattern_stats[signal.pattern]["trades"] += 1
                
                if signal.tfc_pass:
                    tfc_trades += 1
                    pattern_stats[signal.pattern]["tfc_trades"] += 1
                
                # Simulate the trade
                entry_price = signal.entry
                stop_loss = signal.stop
                target = signal.target1
                
                # Look ahead to see trade outcome
                exit_price = None
                exit_reason = ""
                
                # Check next 15 bars for exit
                for j in range(i+1, min(i+16, len(data))):
                    next_high = data['High'].iloc[j]
                    next_low = data['Low'].iloc[j]
                    
                    if signal.direction == "buy":
                        # Check stop loss first
                        if next_low <= stop_loss:
                            exit_price = stop_loss
                            exit_reason = "Stop Loss"
                            break
                        # Check target
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
                        # Check target
                        if next_low <= target:
                            exit_price = target
                            exit_reason = "Target Hit"
                            break
                
                if exit_price is None:
                    # No exit found, close at last available price
                    exit_price = data['Close'].iloc[min(i+15, len(data)-1)]
                    exit_reason = "Time Exit"
                
                # Calculate P&L
                if signal.direction == "buy":
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price
                
                if pnl > 0:
                    wins += 1
                    pattern_stats[signal.pattern]["wins"] += 1
                    if signal.tfc_pass:
                        tfc_wins += 1
                        pattern_stats[signal.pattern]["tfc_wins"] += 1
                else:
                    losses += 1
                    if signal.tfc_pass:
                        tfc_losses += 1

        hit_rate = (wins / max(total_trades, 1)) * 100
        tfc_hit_rate = (tfc_wins / max(tfc_trades, 1)) * 100
        
        print(f"\n📊 Professional STRAT Backtest Results ({period}) — {self.symbol}")
        print(f"🎯 Total Trades: {total_trades} | ✅ Wins: {wins} | ❌ Losses: {losses} | 📈 Hit Rate: {hit_rate:.1f}%")
        print(f"✅ TFC Trades: {tfc_trades} | ✅ TFC Wins: {tfc_wins} | ❌ TFC Losses: {tfc_losses} | 📈 TFC Hit Rate: {tfc_hit_rate:.1f}%")
        print(f"🔥 TFC Improvement: {tfc_hit_rate - hit_rate:+.1f}% over all signals")
        print(f"\n📋 Pattern Performance:")
        
        for pattern, stats in pattern_stats.items():
            if stats["trades"] > 0:
                win_rate = (stats["wins"] / stats["trades"]) * 100
                tfc_win_rate = (stats["tfc_wins"] / max(stats["tfc_trades"], 1)) * 100
                print(f"  {pattern}:")
                print(f"    All: {stats['trades']} trades, {stats['wins']} wins ({win_rate:.1f}%)")
                print(f"    TFC: {stats['tfc_trades']} trades, {stats['tfc_wins']} wins ({tfc_win_rate:.1f}%)")

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., SPY): ").upper().strip()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ").strip()

    bot = ProfessionalStratTradingBot(symbol=symbol, timeframe=timeframe, prepost=False, sleep_secs=30)
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