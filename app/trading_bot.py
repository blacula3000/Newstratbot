from datetime import datetime
import talib
import threading
import time
import pandas as pd
from pybit import usdt_perpetual
import logging
import numpy as np

# Configure logging
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
        self.stop_losses = {}
        self.take_profits = {}
        self.max_positions = 3
        self.position_timeout = 24
        self.stop_trading = False
        self.trading_lock = threading.Lock()
        self.pattern_history = []
        self._historical_data = {}
        self.active_trades = []
        self.trade_history = []
        
        # Initialize Bybit client
        self.session = usdt_perpetual.HTTP(
            endpoint=self.config.get('base_url', 'https://api-testnet.bybit.com'),
            api_key=self.config.get('api_key'),
            api_secret=self.config.get('api_secret')
        )
        
        # Initialize WebSocket for real-time data
        self.ws = usdt_perpetual.WebSocket(
            test=True,  # Set to False for production
            api_key=self.config.get('api_key'),
            api_secret=self.config.get('api_secret')
        )

    def get_current_price(self, symbol):
        """Get current price for a symbol."""
        try:
            ticker = self.session.latest_information_for_symbol(symbol=symbol)
            if ticker['ret_code'] == 0 and ticker['result']:
                return float(ticker['result'][0]['last_price'])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, lookback_periods=100):
        """Get historical data from Bybit."""
        try:
            # Convert timeframe to minutes for Bybit API
            interval_map = {
                '1m': 1, '5m': 5, '15m': 15,
                '1h': 60, '4h': 240, '1d': 'D'
            }
            interval = interval_map.get(self.timeframe, 60)
            
            klines = self.session.query_kline(
                symbol=symbol,
                interval=interval,
                limit=lookback_periods
            )
            
            if klines['ret_code'] == 0 and klines['result']:
                df = pd.DataFrame(klines['result'])
                df['time'] = pd.to_datetime(df['open_time'], unit='s')
                df.set_index('time', inplace=True)
                df = df.astype(float)
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def execute_trade(self, symbol, trade_type, quantity):
        """Execute a trade on Bybit."""
        try:
            side = "Buy" if trade_type == "buy" else "Sell"
            order = self.session.place_active_order(
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=quantity,
                time_in_force="GoodTillCancel",
                reduce_only=False,
                close_on_trigger=False
            )
            
            if order['ret_code'] == 0:
                logger.info(f"Executed {side} order for {quantity} {symbol}")
                return order['result']
            else:
                logger.error(f"Order error: {order['ret_msg']}")
                return None
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    def calculate_position_size(self, symbol, risk_percentage=1.0, stop_loss_pct=2.0):
        """Calculate position size based on account balance and risk."""
        try:
            # Get wallet balance
            wallet = self.session.get_wallet_balance()
            if wallet['ret_code'] == 0:
                balance = float(wallet['result']['USDT']['available_balance'])
                risk_amount = balance * (risk_percentage / 100)
                current_price = self.get_current_price(symbol)
                
                if current_price:
                    position_size = risk_amount / (current_price * (stop_loss_pct / 100))
                    # Round to appropriate decimal places
                    return round(position_size, 3)
            return 0
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def analyze_technical_indicators(self, df):
        """Analyze multiple technical indicators for trading signals."""
        try:
            # Calculate indicators
            df['ema_9'] = talib.EMA(df['Close'], timeperiod=9)
            df['ema_20'] = talib.EMA(df['Close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['Close'], timeperiod=50)
            df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
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

    def manage_open_positions(self):
        """Monitor and manage open positions."""
        try:
            positions = self.session.my_position()
            if positions['ret_code'] == 0:
                for position in positions['result']:
                    if float(position['size']) > 0:  # Active position
                        symbol = position['symbol']
                        entry_price = float(position['entry_price'])
                        current_price = self.get_current_price(symbol)
                        
                        if not current_price:
                            continue

                        # Check stop loss
                        if symbol in self.stop_losses:
                            stop_price = self.stop_losses[symbol]
                            if current_price <= stop_price:
                                self.execute_trade(symbol, "sell", position['size'])
                                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                                continue

                        # Check take profit
                        if symbol in self.take_profits:
                            take_profit = self.take_profits[symbol]
                            if current_price >= take_profit:
                                self.execute_trade(symbol, "sell", position['size'])
                                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                                continue

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    def process_symbol(self, symbol):
        """Process a single symbol for trading."""
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

            # Enhanced entry conditions
            latest_pattern = patterns[-1] if patterns else None
            
            bullish_entry = (
                signals['ema_cross'] and 
                signals['rsi_oversold'] and 
                signals['macd_cross'] and 
                signals['trend'] == 'bullish' and
                (latest_pattern and latest_pattern['action'] == 'buy')
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
                            'entry_time': datetime.now().isoformat(),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                        logger.info(f"Opened bullish position in {symbol} at {current_price}")

            # Manage existing positions
            self.manage_open_positions()

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")

    def run(self, symbols, interval=60):
        """Main bot loop to continuously monitor markets and execute trades."""
        logger.info(f"Starting trading bot for symbols: {symbols}")
        
        self.stop_trading = False
        
        # Subscribe to WebSocket for real-time data
        for symbol in symbols:
            self.ws.kline_stream(
                symbol=symbol,
                interval=self.timeframe,
                callback=self._handle_kline_data
            )
        
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

        # Clean up WebSocket connection
        self.ws.exit()

    def _handle_kline_data(self, message):
        """Handle real-time kline data from WebSocket."""
        try:
            data = message['data']
            symbol = data['symbol']
            
            # Update historical data cache
            if symbol in self._historical_data:
                latest_candle = pd.DataFrame([{
                    'time': pd.to_datetime(data['start'], unit='s'),
                    'Open': float(data['open']),
                    'High': float(data['high']),
                    'Low': float(data['low']),
                    'Close': float(data['close']),
                    'Volume': float(data['volume'])
                }]).set_index('time')
                
                self._historical_data[symbol] = pd.concat([
                    self._historical_data[symbol][:-1],
                    latest_candle
                ])
            
        except Exception as e:
            logger.error(f"Error handling WebSocket data: {e}")

    def close_position(self, symbol, position_size, exit_price, entry_time):
        """Record closed position in trade history."""
        trade = {
            'symbol': symbol,
            'type': 'buy',  # You might want to track this when opening positions
            'entry_price': float(self.active_trades[-1]['entry_price']),
            'exit_price': float(exit_price),
            'position_size': float(position_size),
            'entry_time': entry_time,
            'exit_time': datetime.now().isoformat(),
            'pnl': ((exit_price - self.active_trades[-1]['entry_price']) / 
                    self.active_trades[-1]['entry_price']) * 100
        }
        self.trade_history.append(trade)
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def get_trade_history(self):
        """Get completed trade history."""
        return self.trade_history

    def get_performance_metrics(self):
        """Calculate performance metrics."""
        try:
            metrics = {
                'total_trades': len(self.trade_history),
                'win_rate': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'current_drawdown': 0,
                'risk_per_trade': 0
            }
            
            if not self.trade_history:
                return metrics
                
            # Calculate win rate and averages
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            metrics['win_rate'] = (len(winning_trades) / len(self.trade_history)) * 100
            
            # Calculate average profit
            profits = [t['pnl'] for t in self.trade_history]
            metrics['avg_profit'] = sum(profits) / len(profits)
            
            # Calculate max drawdown
            cumulative = 0
            peak = 0
            drawdown = 0
            
            for trade in self.trade_history:
                cumulative += trade['pnl']
                peak = max(peak, cumulative)
                drawdown = min(drawdown, cumulative - peak)
            
            metrics['max_drawdown'] = abs(drawdown)
            
            # Calculate profit factor
            gains = sum(p for p in profits if p > 0)
            losses = abs(sum(p for p in profits if p < 0))
            metrics['profit_factor'] = gains / losses if losses != 0 else gains
            
            # Calculate Sharpe ratio (simplified)
            returns = np.array(profits)
            if len(returns) > 1:
                metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
            
            # Current drawdown from peak
            if self.active_trades:
                current_value = sum(t['pnl'] for t in self.active_trades)
                metrics['current_drawdown'] = max(0, peak - current_value)
            
            # Average risk per trade
            metrics['risk_per_trade'] = 2.0  # Default 2% risk per trade
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def get_risk_metrics(self):
        """Get current risk exposure and limits."""
        try:
            metrics = {
                'total_exposure': 0,
                'available_margin': 0,
                'margin_ratio': 0,
                'position_limits': self.max_positions,
                'active_positions': len(self.active_trades),
                'risk_per_trade': 2.0,
                'max_drawdown_limit': 15.0,  # 15% max drawdown limit
                'daily_loss_limit': 5.0,     # 5% daily loss limit
                'position_sizes': {}
            }
            
            # Get account info
            wallet = self.session.get_wallet_balance()
            if wallet['ret_code'] == 0:
                balance = float(wallet['result']['USDT']['available_balance'])
                metrics['available_margin'] = balance
                
                # Calculate total exposure
                for trade in self.active_trades:
                    symbol = trade['symbol']
                    size = float(trade['position_size'])
                    price = self.get_current_price(symbol)
                    if price:
                        exposure = size * price
                        metrics['total_exposure'] += exposure
                        metrics['position_sizes'][symbol] = {
                            'size': size,
                            'exposure': exposure,
                            'pct_account': (exposure / balance) * 100
                        }
                
                # Calculate margin ratio
                metrics['margin_ratio'] = (metrics['total_exposure'] / balance) * 100 if balance > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    # ... (keep other existing methods like analyze_technical_indicators, 
    #      detect_patterns, process_symbol, and run) ... 