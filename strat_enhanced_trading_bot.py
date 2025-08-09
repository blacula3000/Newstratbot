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

# -------- STRAT Pattern Detection --------
class StratPatternDetector:
    """STRAT methodology pattern detection and signal generation"""
    
    def __init__(self):
        self.strat_candle = StratCandle()
    
    def detect_2_1_2_reversal(self, candles: pd.DataFrame) -> Dict[str, any]:
        """
        Detect 2-1-2 reversal pattern
        Pattern: Directional -> Inside -> Opposite Directional
        """
        if len(candles) < 4:
            return {"detected": False}
        
        # Get last 3 candles classification
        sequence = self.strat_candle.get_candle_sequence(candles, 3)
        
        if len(sequence) < 3:
            return {"detected": False}
        
        c1, c2, c3 = sequence[-3], sequence[-2], sequence[-1]
        
        # 2U-1-2D (Bearish reversal)
        if c1 == '2U' and c2 == '1' and c3 == '2D':
            return {
                "detected": True,
                "pattern": "2U-1-2D_Bearish_Reversal",
                "signal": "sell",
                "confidence": 0.8,
                "sequence": sequence
            }
        
        # 2D-1-2U (Bullish reversal) 
        if c1 == '2D' and c2 == '1' and c3 == '2U':
            return {
                "detected": True,
                "pattern": "2D-1-2U_Bullish_Reversal", 
                "signal": "buy",
                "confidence": 0.8,
                "sequence": sequence
            }
        
        return {"detected": False, "sequence": sequence}
    
    def detect_3_1_2_setup(self, candles: pd.DataFrame) -> Dict[str, any]:
        """
        Detect 3-1-2 setup pattern
        Pattern: Outside -> Inside -> Directional (breakout)
        """
        if len(candles) < 4:
            return {"detected": False}
        
        sequence = self.strat_candle.get_candle_sequence(candles, 3)
        
        if len(sequence) < 3:
            return {"detected": False}
        
        c1, c2, c3 = sequence[-3], sequence[-2], sequence[-1]
        
        # 3-1-2U (Bullish breakout)
        if c1 == '3' and c2 == '1' and c3 == '2U':
            return {
                "detected": True,
                "pattern": "3-1-2U_Bullish_Breakout",
                "signal": "buy", 
                "confidence": 0.75,
                "sequence": sequence
            }
        
        # 3-1-2D (Bearish breakdown)
        if c1 == '3' and c2 == '1' and c3 == '2D':
            return {
                "detected": True,
                "pattern": "3-1-2D_Bearish_Breakdown",
                "signal": "sell",
                "confidence": 0.75, 
                "sequence": sequence
            }
        
        return {"detected": False, "sequence": sequence}
    
    def detect_2_2_continuation(self, candles: pd.DataFrame) -> Dict[str, any]:
        """
        Detect 2-2 continuation pattern
        Pattern: Directional -> Same Direction Directional
        """
        if len(candles) < 3:
            return {"detected": False}
        
        sequence = self.strat_candle.get_candle_sequence(candles, 2)
        
        if len(sequence) < 2:
            return {"detected": False}
        
        c1, c2 = sequence[-2], sequence[-1]
        
        # 2U-2U (Bullish continuation)
        if c1 == '2U' and c2 == '2U':
            return {
                "detected": True,
                "pattern": "2U-2U_Bullish_Continuation",
                "signal": "buy",
                "confidence": 0.7,
                "sequence": sequence
            }
        
        # 2D-2D (Bearish continuation)
        if c1 == '2D' and c2 == '2D':
            return {
                "detected": True,
                "pattern": "2D-2D_Bearish_Continuation", 
                "signal": "sell",
                "confidence": 0.7,
                "sequence": sequence
            }
        
        return {"detected": False, "sequence": sequence}
    
    def detect_inside_bar_setup(self, candles: pd.DataFrame) -> Dict[str, any]:
        """
        Detect inside bar setups (consolidation before breakout)
        """
        if len(candles) < 2:
            return {"detected": False}
        
        sequence = self.strat_candle.get_candle_sequence(candles, 1)
        
        if len(sequence) < 1:
            return {"detected": False}
        
        if sequence[-1] == '1':
            # Inside bar detected - potential breakout setup
            return {
                "detected": True,
                "pattern": "Inside_Bar_Setup",
                "signal": "watch", # Wait for breakout direction
                "confidence": 0.6,
                "sequence": sequence
            }
        
        return {"detected": False, "sequence": sequence}

# -------- Enhanced Candlestick Pattern Class --------
class CandleStickPattern:
    """Enhanced pattern detection with both classic and STRAT patterns"""
    
    def __init__(self):
        self.strat_detector = StratPatternDetector()
    
    @staticmethod
    def _real_body(row): 
        return abs(row['Close'] - row['Open'])

    @staticmethod
    def _range(row): 
        return row['High'] - row['Low']

    def bullish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Prev red, current green, current body engulfs prev body, body is meaningful."""
        if len(candles) < 2: return False
        prev = candles.iloc[-2]; cur = candles.iloc[-1]
        if prev['Close'] >= prev['Open']: return False
        if cur['Close'] <= cur['Open']: return False

        prev_body_low  = min(prev['Open'], prev['Close'])
        prev_body_high = max(prev['Open'], prev['Close'])
        cur_body_low   = min(cur['Open'], cur['Close'])
        cur_body_high  = max(cur['Open'], cur['Close'])

        body_ok = self._real_body(cur) >= 0.5 * self._range(cur)
        engulf  = (cur_body_low <= prev_body_low) and (cur_body_high >= prev_body_high)
        return body_ok and engulf

    def bearish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Prev green, current red, current body engulfs prev body, body is meaningful."""
        if len(candles) < 2: return False
        prev = candles.iloc[-2]; cur = candles.iloc[-1]
        if prev['Close'] <= prev['Open']: return False
        if cur['Close'] >= cur['Open']: return False

        prev_body_low  = min(prev['Open'], prev['Close'])
        prev_body_high = max(prev['Open'], prev['Close'])
        cur_body_low   = min(cur['Open'], cur['Close'])
        cur_body_high  = max(cur['Open'], cur['Close'])

        body_ok = self._real_body(cur) >= 0.5 * self._range(cur)
        engulf  = (cur_body_low <= prev_body_low) and (cur_body_high >= prev_body_high)
        return body_ok and engulf

    def simple_bullish_reversal(self, candles: pd.DataFrame) -> bool:
        if len(candles) < 2: return False
        prev, cur = candles.iloc[-2], candles.iloc[-1]
        return (prev['Close'] < prev['Open'] and
                cur['Close'] > cur['Open'] and
                cur['Low'] <= prev['Low'])

    def simple_bearish_reversal(self, candles: pd.DataFrame) -> bool:
        if len(candles) < 2: return False
        prev, cur = candles.iloc[-2], candles.iloc[-1]
        return (prev['Close'] > prev['Open'] and
                cur['Close'] < cur['Open'] and
                cur['High'] >= prev['High'])

    def evaluate(self, candles: pd.DataFrame) -> dict:
        """Evaluate both classic candlestick and STRAT patterns"""
        if candles.empty or len(candles) < 2:
            return {}
        
        results = {}
        
        # Classic candlestick patterns
        if len(candles) >= 2:
            results.update({
                "bullish_engulfing": self.bullish_engulfing(candles),
                "bearish_engulfing": self.bearish_engulfing(candles),
                "bullish_reversal": self.simple_bullish_reversal(candles),
                "bearish_reversal": self.simple_bearish_reversal(candles),
            })
        
        # STRAT patterns
        strat_patterns = {}
        
        # 2-1-2 Reversals
        reversal_212 = self.strat_detector.detect_2_1_2_reversal(candles)
        if reversal_212["detected"]:
            strat_patterns[reversal_212["pattern"]] = {
                "detected": True,
                "signal": reversal_212["signal"],
                "confidence": reversal_212["confidence"],
                "sequence": reversal_212["sequence"]
            }
        
        # 3-1-2 Setups  
        setup_312 = self.strat_detector.detect_3_1_2_setup(candles)
        if setup_312["detected"]:
            strat_patterns[setup_312["pattern"]] = {
                "detected": True,
                "signal": setup_312["signal"], 
                "confidence": setup_312["confidence"],
                "sequence": setup_312["sequence"]
            }
        
        # 2-2 Continuations
        continuation_22 = self.strat_detector.detect_2_2_continuation(candles)
        if continuation_22["detected"]:
            strat_patterns[continuation_22["pattern"]] = {
                "detected": True,
                "signal": continuation_22["signal"],
                "confidence": continuation_22["confidence"], 
                "sequence": continuation_22["sequence"]
            }
        
        # Inside Bar Setups
        inside_setup = self.strat_detector.detect_inside_bar_setup(candles)
        if inside_setup["detected"]:
            strat_patterns[inside_setup["pattern"]] = {
                "detected": True,
                "signal": inside_setup["signal"],
                "confidence": inside_setup["confidence"],
                "sequence": inside_setup["sequence"]
            }
        
        results["strat_patterns"] = strat_patterns
        
        return results

@dataclass
class TradeSignal:
    ts: pd.Timestamp
    symbol: str
    price: float
    kind: str
    signal: str = None  # buy/sell/watch
    confidence: float = 0.0
    sequence: List[str] = None

class StratTradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m', prepost: bool=False, sleep_secs: int=30):
        self.symbol = symbol.upper()
        self.interval = map_interval(timeframe)
        self.prepost = prepost
        self.sleep_secs = sleep_secs
        self.patterns = CandleStickPattern()
        self._last_bar_ts = None
        self.market_hours = MarketHours()
        self._signal_history = []
        logger.info(f"STRAT Bot initialized for {self.symbol} @ {self.interval} (prepost={self.prepost})")

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

    def analyze(self, data: pd.DataFrame) -> dict:
        """Enhanced analysis with STRAT patterns"""
        return self.patterns.evaluate(data)

    def _get_signal_strength(self, results: dict) -> Tuple[str, float]:
        """Determine overall signal strength from all patterns"""
        buy_signals = []
        sell_signals = []
        watch_signals = []
        
        # Classic patterns
        if results.get("bullish_engulfing") or results.get("bullish_reversal"):
            buy_signals.append(("classic_bullish", 0.7))
        if results.get("bearish_engulfing") or results.get("bearish_reversal"):
            sell_signals.append(("classic_bearish", 0.7))
        
        # STRAT patterns
        strat_patterns = results.get("strat_patterns", {})
        for pattern_name, pattern_data in strat_patterns.items():
            signal = pattern_data["signal"]
            confidence = pattern_data["confidence"]
            
            if signal == "buy":
                buy_signals.append((pattern_name, confidence))
            elif signal == "sell":
                sell_signals.append((pattern_name, confidence))
            elif signal == "watch":
                watch_signals.append((pattern_name, confidence))
        
        # Determine strongest signal
        if buy_signals:
            max_confidence = max([conf for _, conf in buy_signals])
            return "buy", max_confidence
        elif sell_signals:
            max_confidence = max([conf for _, conf in sell_signals])
            return "sell", max_confidence
        elif watch_signals:
            max_confidence = max([conf for _, conf in watch_signals])
            return "watch", max_confidence
        
        return "none", 0.0

    def _log_signal_csv(self, sig: TradeSignal, path="strat_signals.csv"):
        try:
            df = pd.DataFrame([{
                "ts": sig.ts.isoformat(),
                "symbol": sig.symbol,
                "price": sig.price,
                "pattern": sig.kind,
                "signal": sig.signal,
                "confidence": sig.confidence,
                "sequence": str(sig.sequence) if sig.sequence else ""
            }])
            header = not pd.io.common.file_exists(path)
            df.to_csv(path, mode="a", index=False, header=header)
        except Exception as e:
            logger.error(f"Failed to write signal CSV: {e}")

    def run(self):
        print(f"🚀 Starting STRAT bot for {self.symbol} ({self.interval}) — Ctrl+C to stop.")
        logger.info(f"STRAT Bot started for {self.symbol}")

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
                    time.sleep(self.sleep_secs)
                    continue
                self._last_bar_ts = last_ts

                price = float(data['Close'].iloc[-1])
                results = self.analyze(data)
                
                # Get candle sequence for context
                if len(data) >= 4:
                    sequence = StratCandle.get_candle_sequence(data.tail(4), 3)
                else:
                    sequence = []

                print(f"\n📊 Bar close: {last_ts.strftime('%H:%M:%S')}  {self.symbol} ${price:.2f}")
                print(f"📈 Sequence: {' -> '.join(sequence[-3:]) if len(sequence) >= 3 else 'N/A'}")
                
                signal_fired = False
                
                # Classic patterns
                for name, detected in results.items():
                    if name != "strat_patterns" and detected:
                        signal_fired = True
                        print(f"{'🟢' if 'bullish' in name else '🔴'} Classic: {name.replace('_',' ').title()}")
                        logger.info(f"Classic pattern {name} detected for {self.symbol} @ {price:.2f}")
                
                # STRAT patterns
                strat_patterns = results.get("strat_patterns", {})
                for pattern_name, pattern_data in strat_patterns.items():
                    signal_fired = True
                    signal = pattern_data["signal"] 
                    confidence = pattern_data["confidence"]
                    seq = pattern_data.get("sequence", [])
                    
                    emoji = "🟢" if signal == "buy" else "🔴" if signal == "sell" else "🟡"
                    print(f"{emoji} STRAT: {pattern_name} | Signal: {signal.upper()} | Confidence: {confidence:.1%}")
                    
                    logger.info(f"STRAT pattern {pattern_name} detected: {signal} @ {price:.2f} (conf: {confidence:.1%})")
                    
                    # Log to CSV
                    trade_signal = TradeSignal(
                        ts=last_ts,
                        symbol=self.symbol,
                        price=price,
                        kind=pattern_name,
                        signal=signal,
                        confidence=confidence,
                        sequence=seq
                    )
                    self._log_signal_csv(trade_signal)
                    self._signal_history.append(trade_signal)

                if not signal_fired:
                    print("📊 No patterns detected.")

                # Show overall signal strength
                overall_signal, overall_confidence = self._get_signal_strength(results)
                if overall_signal != "none":
                    print(f"🎯 Overall Signal: {overall_signal.upper()} ({overall_confidence:.1%})")

                time.sleep(self.sleep_secs)
                
            except KeyboardInterrupt:
                print("\n🛑 STRAT Bot stopped by user.")
                logger.info("STRAT Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Main loop error: {e}")
                time.sleep(self.sleep_secs)

    def backtest(self, days=5):
        """Enhanced backtest with STRAT pattern analysis"""
        period = f"{max(days,1)}d"
        data = self.fetch_data(period=period)
        if data.empty or len(data) < 20:
            print("Not enough data to backtest.")
            return

        df = data.copy()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

        wins = 0; losses = 0; total_signals = 0
        pattern_stats = {}
        
        print(f"\n📈 Running STRAT backtest for {self.symbol} over {days} days...")

        for i in range(20, len(df)-1):
            window = df.iloc[:i+1]
            results = self.analyze(window)
            
            # Process classic patterns
            for name, detected in results.items():
                if name != "strat_patterns" and detected:
                    total_signals += 1
                    if name not in pattern_stats:
                        pattern_stats[name] = {"signals": 0, "wins": 0}
                    pattern_stats[name]["signals"] += 1
                    
                    # Simple follow-through test
                    entry = df['Close'].iloc[i]
                    next_high = df['High'].iloc[i+1]
                    next_low = df['Low'].iloc[i+1]
                    atr = df['ATR'].iloc[i]
                    
                    if not np.isnan(atr) and atr > 0:
                        target = 0.5 * atr  # 50% of ATR target
                        
                        if "bullish" in name:
                            if next_high - entry >= target:
                                wins += 1
                                pattern_stats[name]["wins"] += 1
                            elif entry - next_low >= target:
                                losses += 1
                        else:
                            if entry - next_low >= target:
                                wins += 1
                                pattern_stats[name]["wins"] += 1
                            elif next_high - entry >= target:
                                losses += 1
            
            # Process STRAT patterns
            strat_patterns = results.get("strat_patterns", {})
            for pattern_name, pattern_data in strat_patterns.items():
                total_signals += 1
                if pattern_name not in pattern_stats:
                    pattern_stats[pattern_name] = {"signals": 0, "wins": 0}
                pattern_stats[pattern_name]["signals"] += 1
                
                signal = pattern_data["signal"]
                entry = df['Close'].iloc[i]
                next_high = df['High'].iloc[i+1]
                next_low = df['Low'].iloc[i+1]
                atr = df['ATR'].iloc[i]
                
                if not np.isnan(atr) and atr > 0:
                    target = 0.5 * atr
                    
                    if signal == "buy":
                        if next_high - entry >= target:
                            wins += 1
                            pattern_stats[pattern_name]["wins"] += 1
                        elif entry - next_low >= target:
                            losses += 1
                    elif signal == "sell":
                        if entry - next_low >= target:
                            wins += 1
                            pattern_stats[pattern_name]["wins"] += 1
                        elif next_high - entry >= target:
                            losses += 1

        hit_rate = (wins / max(wins + losses, 1)) * 100
        
        print(f"\n📊 STRAT Backtest Results ({period}) — {self.symbol}")
        print(f"🎯 Total Signals: {total_signals} | ✅ Wins: {wins} | ❌ Losses: {losses} | 📈 Hit Rate: {hit_rate:.1f}%")
        print(f"\n📋 Pattern Breakdown:")
        
        for pattern, stats in pattern_stats.items():
            if stats["signals"] > 0:
                pattern_hit_rate = (stats["wins"] / stats["signals"]) * 100
                print(f"  {pattern}: {stats['signals']} signals, {stats['wins']} wins ({pattern_hit_rate:.1f}%)")

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., SPY): ").upper().strip()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ").strip()

    bot = StratTradingBot(symbol=symbol, timeframe=timeframe, prepost=False, sleep_secs=30)
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