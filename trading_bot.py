from datetime import datetime
import talib
import threading
import time
import yfinance as yf
import pandas as pd
import logging
from agent_integration import QuantPatternAnalystAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, market_type="futures", timeframe="1h", config=None):
        self.market_type = market_type
        self.timeframe = timeframe
        self.config = config or {}
        self.stop_losses = {}  # Track stop losses for positions
        self.take_profits = {}  # Track take profits for positions
        self.max_positions = 3  # Maximum concurrent positions
        self.position_timeout = 24  # Hours until position is closed
        self.stop_trading = False
        self.trading_lock = threading.Lock()
        self.pattern_history = []  # Store detected patterns
        self._historical_data = {}
        self.active_trades = []
        
        # Initialize Quant Pattern Analyst Agent
        self.pattern_agent = QuantPatternAnalystAgent({
            'min_confidence': 0.75,  # Minimum confidence for pattern signals
            'max_patterns': 10  # Maximum patterns to track per symbol
        })
        self.use_agent_patterns = self.config.get('use_agent_patterns', True)

    def get_current_price(self, symbol):
        """Get current price for a symbol."""
        try:
            if self.market_type == "crypto" and self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker['last']
            elif self.market_type in ["stocks", "options", "futures"] and self.api:
                bars = self.api.get_latest_bar(symbol)
                return float(bars.close)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def manage_open_positions(self):
        """Monitor and manage open positions."""
        try:
            current_time = datetime.now()
            
            if self.market_type == "crypto" and self.exchange:
                positions = self.exchange.fetch_positions()
            else:
                positions = self.api.list_positions()

            for position in positions:
                symbol = position['symbol']
                entry_price = float(position['avg_entry_price'])
                current_price = self.get_current_price(symbol)
                
                if not current_price:
                    continue

                # Check stop loss
                if symbol in self.stop_losses:
                    stop_price = self.stop_losses[symbol]
                    if current_price <= stop_price:
                        self.execute_trade(symbol, "sell", position['qty'])
                        logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                        continue

                # Check take profit
                if symbol in self.take_profits:
                    take_profit = self.take_profits[symbol]
                    if current_price >= take_profit:
                        self.execute_trade(symbol, "sell", position['qty'])
                        logger.info(f"Take profit triggered for {symbol} at {current_price}")
                        continue

                # Check position timeout
                entry_time = datetime.fromtimestamp(position['timestamp'] / 1000)
                if (current_time - entry_time).total_seconds() / 3600 >= self.position_timeout:
                    self.execute_trade(symbol, "sell", position['qty'])
                    logger.info(f"Position timeout reached for {symbol}")

        except Exception as e:
            logger.error(f"Error managing positions: {e}") 

    def analyze_technical_indicators(self, df):
        """Analyze multiple technical indicators for trading signals."""
        try:
            # Calculate indicators
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Get latest values
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Generate signals
            signals = {
                'ema_cross': current['ema_9'] > current['ema_20'] and previous['ema_9'] <= previous['ema_20'],
                'rsi_oversold': current['rsi'] < 30,
                'rsi_overbought': current['rsi'] > 70,
                'macd_cross': current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal'],
                'trend': 'bullish' if current['ema_20'] > current['ema_50'] else 'bearish'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return {} 

    def candle_type(self, candle, prev_candle):
        """Identify candle type (1-Inside, 2-Directional, 3-Outside)"""
        if candle['High'] < prev_candle['High'] and candle['Low'] > prev_candle['Low']:
            return '1'  # Inside
        elif candle['High'] > prev_candle['High'] and candle['Low'] < prev_candle['Low']:
            return '3'  # Outside
        elif candle['High'] > prev_candle['High']:
            return '2u'  # Directional up
        elif candle['Low'] < prev_candle['Low']:
            return '2d'  # Directional down
        return 'Unknown'

    def detect_patterns(self, df):
        """Enhanced pattern detection with multiple pattern types and agent integration"""
        patterns = []
        
        # Use agent for advanced pattern detection if enabled
        if self.use_agent_patterns and hasattr(self, 'pattern_agent'):
            try:
                # Get agent-detected patterns
                agent_analysis = self.pattern_agent.analyze_patterns(
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    df=df,
                    timeframe=self.timeframe
                )
                
                # Add agent patterns to the list
                for agent_pattern in agent_analysis.get('patterns', []):
                    patterns.append({
                        'time': datetime.now(),
                        'type': agent_pattern.get('type', 'Unknown'),
                        'confidence': agent_pattern.get('confidence', 0.5),
                        'action': self._determine_action_from_pattern(agent_pattern),
                        'source': 'agent',
                        'details': agent_pattern
                    })
                    
                logger.info(f"Agent detected {len(agent_analysis.get('patterns', []))} patterns")
            except Exception as e:
                logger.warning(f"Agent pattern detection failed, falling back to basic: {e}")
        
        # Original pattern detection logic
        for i in range(2, len(df)):
            candle1, candle2, candle3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            
            c1_type = self.candle_type(candle1, df.iloc[i-3]) if i >= 3 else 'N/A'
            c2_type = self.candle_type(candle2, candle1)
            c3_type = self.candle_type(candle3, candle2)
            
            current_time = df.index[i]
            
            # Pattern detection with confidence levels
            if c2_type == '2d' and c3_type == '2u':
                patterns.append({
                    'time': current_time,
                    'type': '2-2 Bullish Reversal',
                    'confidence': 0.8,
                    'action': 'buy',
                    'source': 'basic'
                })

            if c2_type == '2u' and c3_type == '2d':
                patterns.append({
                    'time': current_time,
                    'type': '2-2 Bearish Reversal',
                    'confidence': 0.8,
                    'action': 'sell',
                    'source': 'basic'
                })

            if c3_type == '1':
                patterns.append({
                    'time': current_time,
                    'type': 'Inside Bar',
                    'confidence': 0.6,
                    'action': 'hold',
                    'source': 'basic'
                })

            if c3_type == '3':
                patterns.append({
                    'time': current_time,
                    'type': 'Outside Bar',
                    'confidence': 0.7,
                    'action': 'hold',
                    'source': 'basic'
                })

        return patterns
    
    def _determine_action_from_pattern(self, pattern):
        """Determine trading action based on pattern type"""
        pattern_type = pattern.get('type', '').lower()
        if any(word in pattern_type for word in ['bullish', 'cup', 'ascending', 'bottom']):
            return 'buy'
        elif any(word in pattern_type for word in ['bearish', 'head and shoulders', 'descending', 'top']):
            return 'sell'
        else:
            return 'hold'

    def get_historical_data(self, symbol, lookback_periods=100):
        """Enhanced historical data fetching with multiple sources"""
        if symbol in self._historical_data:
            return self._historical_data[symbol]

        try:
            if self.market_type == "crypto" and self.exchange:
                # ... existing crypto code ...
                pass
            
            elif self.market_type in ["stocks", "options", "futures"]:
                # Try yfinance first
                try:
                    df = yf.download(symbol, period=f"{lookback_periods}d", interval=self.timeframe)
                    if not df.empty:
                        self._historical_data[symbol] = df
                        return df
                except Exception:
                    # Fallback to Alpaca if yfinance fails
                    if self.api:
                        # ... existing Alpaca code ...
                        pass

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
        
        return pd.DataFrame()

    def process_symbol(self, symbol):
        """Enhanced symbol processing with pattern detection"""
        try:
            if len(self.active_trades) >= self.max_positions:
                return

            df = self.get_historical_data(symbol)
            if df.empty:
                return

            # Get both technical signals and patterns
            signals = self.analyze_technical_indicators(df)
            patterns = self.detect_patterns(df)
            
            # Store recent patterns
            if patterns:
                self.pattern_history.extend(patterns)
                # Keep only last 100 patterns
                self.pattern_history = self.pattern_history[-100:]

            current_price = self.get_current_price(symbol)
            if not current_price:
                return

            # Enhanced entry conditions using both signals and patterns
            latest_pattern = patterns[-1] if patterns else None
            
            bullish_entry = (
                signals['ema_cross'] and 
                signals['rsi_oversold'] and 
                signals['macd_cross'] and 
                signals['trend'] == 'bullish' and
                (latest_pattern and latest_pattern['action'] == 'buy')
            )
            
            bearish_entry = (
                signals['rsi_overbought'] and 
                signals['trend'] == 'bearish' and
                (latest_pattern and latest_pattern['action'] == 'sell')
            )

            if bullish_entry:
                position_size = self.calculate_position_size(symbol)
                if position_size > 0:
                    order = self.execute_trade(symbol, "buy", position_size)
                    if order:
                        # Set stop loss and take profit
                        stop_loss = current_price * 0.98  # 2% stop loss
                        take_profit = current_price * 1.04  # 4% take profit
                        self.stop_losses[symbol] = stop_loss
                        self.take_profits[symbol] = take_profit
                        self.active_trades.append({
                            'symbol': symbol,
                            'entry_price': current_price,
                            'position_size': position_size,
                            'entry_time': datetime.now()
                        })
                        logger.info(f"Opened bullish position in {symbol} at {current_price}")

            # Manage existing positions
            self.manage_open_positions()

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}") 

    def run(self, symbols, interval=60):
        """Main bot loop to continuously monitor markets and execute trades."""
        logger.info(f"Starting trading bot for {self.market_type} in {symbols}")
        
        self.stop_trading = False
        
        while not self.stop_trading:
            threads = []
            for symbol in symbols:
                if self.stop_trading:
                    break
                thread = threading.Thread(target=self.process_symbol, args=(symbol,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
                
            if not self.stop_trading:
                time.sleep(interval)

if __name__ == "__main__":
    config = {
        "api_key": "YOUR_API_KEY",
        "api_secret": "YOUR_API_SECRET",
        "exchange": "binance",
        "base_url": "https://paper-api.alpaca.markets",
    }

    # Initialize bot with risk parameters
    bot = TradingBot(
        market_type="futures",
        timeframe="1h",
        config=config
    )

    # Test with paper trading first
    symbols = ["BTC/USDT", "ETH/USDT"]
    bot.run(symbols, interval=300)
