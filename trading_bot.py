import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class CandleStickPattern:
    def __init__(self):
        self.patterns = {
            'bullish_reversal': self._check_bullish_reversal,
            'bearish_reversal': self._check_bearish_reversal
        }
    
    def _check_bullish_reversal(self, candles: pd.DataFrame) -> bool:
        if len(candles) < 2:
            return False
        
        current = candles.iloc[-1]
        previous = candles.iloc[-2]
        
        return (previous['Close'] < previous['Open'] and
                current['Close'] > current['Open'] and
                current['Low'] < previous['Low'])
    
    def _check_bearish_reversal(self, candles: pd.DataFrame) -> bool:
        if len(candles) < 2:
            return False
            
        current = candles.iloc[-1]
        previous = candles.iloc[-2]
        
        return (previous['Close'] > previous['Open'] and
                current['Close'] < current['Open'] and
                current['High'] > previous['High'])

class TradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.pattern_detector = CandleStickPattern()
        logging.info(f"Bot initialized for {symbol} with {timeframe} timeframe")
    
    def fetch_data(self) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period='1d', interval=self.timeframe)
            return data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def analyze_pattern(self, data: pd.DataFrame) -> dict:
        if data.empty:
            return {'bullish_reversal': False, 'bearish_reversal': False}
        
        signals = {
            'bullish_reversal': self.pattern_detector._check_bullish_reversal(data),
            'bearish_reversal': self.pattern_detector._check_bearish_reversal(data)
        }
        return signals

    def run(self):
        print(f"Starting bot for {self.symbol}...")
        logging.info(f"Bot started for {self.symbol}")
        
        while True:
            try:
                # Get current time
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Fetch current market data
                data = self.fetch_data()
                
                if not data.empty:
                    # Analyze patterns
                    signals = self.analyze_pattern(data)
                    
                    # Current price
                    current_price = data['Close'].iloc[-1]
                    
                    # Print status
                    print(f"\nTime: {current_time}")
                    print(f"Symbol: {self.symbol}")
                    print(f"Current Price: ${current_price:.2f}")
                    
                    if signals['bullish_reversal']:
                        print("🟢 Bullish reversal detected!")
                        logging.info(f"Bullish reversal detected for {self.symbol} at ${current_price:.2f}")
                    
                    elif signals['bearish_reversal']:
                        print("🔴 Bearish reversal detected!")
                        logging.info(f"Bearish reversal detected for {self.symbol} at ${current_price:.2f}")
                    
                    else:
                        print("No pattern detected")
                
                # Wait before next check
                time.sleep(60)  # 60 seconds delay
                
            except KeyboardInterrupt:
                print("\nBot stopped by user")
                logging.info("Bot stopped by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    # Create a bot instance (example with SPY)
    symbol = input("Enter stock symbol (e.g., SPY): ").upper()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ")
    
    bot = TradingBot(symbol=symbol, timeframe=timeframe)
    bot.run()