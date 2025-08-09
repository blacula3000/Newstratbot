import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timezone
import pytz
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass

# -------- Logging (rotating) --------
logger = logging.getLogger("trading_bot")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("trading_bot.log", maxBytes=1_000_000, backupCount=3)
console_handler = logging.StreamHandler()  # Add console output
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
        # Standard NYSE hours: 9:30 AM - 4:00 PM ET
        self.market_open_hour = 9
        self.market_open_minute = 30
        self.market_close_hour = 16
        self.market_close_minute = 0
    
    def is_market_open(self) -> tuple[bool, str]:
        """
        Check if US stock market is open
        Returns: (is_open: bool, message: str)
        """
        now = datetime.now(self.tz)
        
        # Check if it's weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_monday = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
            days_until_monday = 7 - now.weekday()
            next_monday += pd.Timedelta(days=days_until_monday)
            return False, f"Weekend. Market opens Monday at {next_monday.strftime('%I:%M %p %Z')}"
        
        # Check if it's during market hours
        current_time = now.time()
        market_open = datetime.combine(now.date(), datetime.min.time().replace(
            hour=self.market_open_hour, minute=self.market_open_minute
        )).time()
        market_close = datetime.combine(now.date(), datetime.min.time().replace(
            hour=self.market_close_hour, minute=self.market_close_minute
        )).time()
        
        if market_open <= current_time <= market_close:
            return True, f"Market is OPEN (closes at {market_close.strftime('%I:%M %p')})"
        elif current_time < market_open:
            return False, f"Market opens at {market_open.strftime('%I:%M %p')} ({(datetime.combine(now.date(), market_open) - now).seconds // 60} mins)"
        else:
            tomorrow_open = (now + pd.Timedelta(days=1)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0
            )
            return False, f"Market closed. Opens tomorrow at {tomorrow_open.strftime('%I:%M %p %Z')}"
    
    def time_until_open(self) -> int:
        """Returns seconds until market opens"""
        now = datetime.now(self.tz)
        
        if now.weekday() >= 5:  # Weekend
            days_until_monday = 7 - now.weekday()
            next_open = (now + pd.Timedelta(days=days_until_monday)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0
            )
        elif now.time() >= datetime.min.time().replace(hour=self.market_close_hour):  # After close
            next_open = (now + pd.Timedelta(days=1)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0
            )
        else:  # Before open same day
            next_open = now.replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0
            )
        
        return int((next_open - now).total_seconds())

# -------- Pattern detector --------
class CandleStickPattern:
    """A handful of saner intraday patterns."""
    @staticmethod
    def _real_body(row): 
        return abs(row['Close'] - row['Open'])

    @staticmethod
    def _range(row): 
        return row['High'] - row['Low']

    def bullish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Prev red, current green, current body engulfs prev body, body is meaningful."""
        if len(candles) < 3: return False
        prev = candles.iloc[-2]; cur = candles.iloc[-1]
        if prev['Close'] >= prev['Open']: return False
        if cur['Close'] <= cur['Open']: return False

        prev_body_low  = min(prev['Open'], prev['Close'])
        prev_body_high = max(prev['Open'], prev['Close'])
        cur_body_low   = min(cur['Open'], cur['Close'])
        cur_body_high  = max(cur['Open'], cur['Close'])

        body_ok = self._real_body(cur) >= 0.5 * self._range(cur)  # conviction
        engulf  = (cur_body_low <= prev_body_low) and (cur_body_high >= prev_body_high)
        return body_ok and engulf

    def bearish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Prev green, current red, current body engulfs prev body, body is meaningful."""
        if len(candles) < 3: return False
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

    # keep your original "reversal" as optional
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
        if candles.empty or len(candles) < 3:
            return {k: False for k in ["bullish_engulfing","bearish_engulfing","bullish_reversal","bearish_reversal"]}
        return {
            "bullish_engulfing": self.bullish_engulfing(candles),
            "bearish_engulfing": self.bearish_engulfing(candles),
            "bullish_reversal": self.simple_bullish_reversal(candles),
            "bearish_reversal": self.simple_bearish_reversal(candles),
        }

@dataclass
class TradeSignal:
    ts: pd.Timestamp
    symbol: str
    price: float
    kind: str  # "bullish_engulfing", etc.

class TradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m', prepost: bool=False, sleep_secs: int=30):
        self.symbol = symbol.upper()
        self.interval = map_interval(timeframe)
        self.prepost = prepost
        self.sleep_secs = sleep_secs
        self.patterns = CandleStickPattern()
        self._last_bar_ts = None
        self.market_hours = MarketHours()  # Add market hours checker
        logger.info(f"Bot initialized for {self.symbol} @ {self.interval} (prepost={self.prepost})")

    def fetch_data(self, period="1d", max_retries=3) -> pd.DataFrame:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(self.symbol).history(period=period, interval=self.interval, prepost=self.prepost)
                # Normalize columns (yfinance sometimes uses lowercase)
                cols = {c: c.capitalize() for c in data.columns}
                data = data.rename(columns=cols)
                # Drop weird rows
                data = data[['Open','High','Low','Close','Volume']].dropna()
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching data after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}. Retrying in {(attempt + 1) * 2} seconds...")
                time.sleep((attempt + 1) * 2)  # Exponential backoff
        return pd.DataFrame()

    def analyze(self, data: pd.DataFrame) -> dict:
        return self.patterns.evaluate(data)

    def _log_signal_csv(self, sig: TradeSignal, path="signals.csv"):
        try:
            df = pd.DataFrame([{
                "ts": sig.ts.isoformat(),
                "symbol": sig.symbol,
                "price": sig.price,
                "signal": sig.kind
            }])
            header = not pd.io.common.file_exists(path)
            df.to_csv(path, mode="a", index=False, header=header)
        except Exception as e:
            logger.error(f"Failed to write signal CSV: {e}")

    def run(self):
        print(f"Starting bot for {self.symbol} ({self.interval}) — Ctrl+C to stop.")
        logger.info(f"Bot started for {self.symbol}")

        while True:
            try:
                # Check market hours first
                is_open, message = self.market_hours.is_market_open()
                
                if not is_open:
                    print(f"\n⏰ {message}")
                    logger.info(f"Market closed: {message}")
                    
                    # Calculate sleep time - don't sleep more than 30 minutes
                    sleep_time = min(self.market_hours.time_until_open(), 1800)  # Max 30 mins
                    if sleep_time > 60:
                        print(f"💤 Sleeping for {sleep_time // 60} minutes until market check...")
                    time.sleep(sleep_time)
                    continue

                # Market is open, proceed with trading logic
                print(f"✅ {message}")
                
                data = self.fetch_data(period="1d")
                if data.empty:
                    print("No data (feed delayed or symbol unavailable).")
                    time.sleep(self.sleep_secs)
                    continue

                last_ts = data.index[-1]
                # Only act on NEWLY CLOSED bars
                if self._last_bar_ts is not None and last_ts == self._last_bar_ts:
                    time.sleep(self.sleep_secs)
                    continue
                self._last_bar_ts = last_ts

                price = float(data['Close'].iloc[-1])
                sigs = self.analyze(data)

                print(f"\n📊 Bar close: {last_ts.strftime('%H:%M:%S')}  {self.symbol} ${price:.2f}")
                fired = False
                for name, flag in sigs.items():
                    if flag:
                        fired = True
                        print(("🟢" if "bullish" in name else "🔴"), name.replace("_"," ").title(), "detected!")
                        logger.info(f"{name} detected for {self.symbol} @ {price:.2f}")
                        self._log_signal_csv(TradeSignal(last_ts, self.symbol, price, name))
                if not fired:
                    print("📈 No pattern detected.")

                time.sleep(self.sleep_secs)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user.")
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Main loop error: {e}")
                time.sleep(self.sleep_secs)

    # ------- quick & dirty backtest over N days --------
    def backtest(self, days=5):
        """Counts signals over past N days of {interval} data and reports hit rate proxy using simple follow-through.
        NOTE: Yahoo intraday is often delayed and may have gaps; this is a *rough* sanity check."""
        period = f"{max(days,1)}d"
        data = self.fetch_data(period=period)
        if data.empty or len(data) < 20:
            print("Not enough data to backtest.")
            return

        # Iterate bar by bar, checking if the signal bar is followed by favorable move = 0.25 * ATR(14) in signal direction
        df = data.copy()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

        wins = 0; losses = 0; total = 0
        sig_counts = {"bullish_engulfing":0,"bearish_engulfing":0,"bullish_reversal":0,"bearish_reversal":0}

        print(f"\n📈 Running backtest for {self.symbol} over {days} days...")

        for i in range(20, len(df)-1):
            window = df.iloc[:i+1]  # up to current closed bar
            sigs = self.analyze(window)
            for name, on in sigs.items():
                if not on: 
                    continue
                sig_counts[name] += 1
                total += 1
                atr = df['ATR'].iloc[i]
                if np.isnan(atr) or atr == 0:
                    continue
                entry = df['Close'].iloc[i]
                next_high = df['High'].iloc[i+1]
                next_low  = df['Low'].iloc[i+1]
                target = 0.25 * atr

                if "bullish" in name:
                    if next_high - entry >= target: wins += 1
                    elif entry - next_low >= target: losses += 1
                else:
                    if entry - next_low >= target: wins += 1
                    elif next_high - entry >= target: losses += 1

        hit = (wins / max(wins+losses,1)) * 100
        print(f"\n📊 Backtest Results ({period}, {self.interval}) — {self.symbol}")
        print(f"🎯 Total Signals: {total} | ✅ Wins: {wins} | ❌ Losses: {losses} | 📈 Hit Rate: {hit:.1f}%")
        print(f"📋 Signal Breakdown: {dict(sig_counts)}")

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., SPY): ").upper().strip()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ").strip()

    bot = TradingBot(symbol=symbol, timeframe=timeframe, prepost=False, sleep_secs=30)
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