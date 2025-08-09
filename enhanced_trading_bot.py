"""
Enhanced Trading Bot with Production-Ready Features
Implements all suggested improvements for robust trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, time
import pytz
import threading
import time as time_module
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os

# Setup rotating log handler
def setup_logging():
    """Setup logging with rotation to prevent infinite growth"""
    logger = logging.getLogger('enhanced_trading_bot')
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Rotating file handler - 10MB max, keep 5 backups
    file_handler = RotatingFileHandler(
        'logs/enhanced_trading_bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: int
    stop_loss: float
    take_profit: float
    position_type: str  # 'long' or 'short'
    status: str = 'open'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current P&L"""
        if self.position_type == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

@dataclass
class TradingSignal:
    """Enhanced trading signal with risk parameters"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    pattern: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timeframe: str
    indicators: Dict[str, float] = field(default_factory=dict)

class MarketHours:
    """Handle market hours and trading sessions"""
    
    def __init__(self, timezone='America/New_York'):
        self.tz = pytz.timezone(timezone)
        self.regular_open = time(9, 30)
        self.regular_close = time(16, 0)
        self.premarket_open = time(4, 0)
        self.afterhours_close = time(20, 0)
    
    def is_market_open(self, include_extended=False) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.tz)
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        current_time = now.time()
        
        if include_extended:
            return self.premarket_open <= current_time <= self.afterhours_close
        else:
            return self.regular_open <= current_time <= self.regular_close
    
    def time_to_next_open(self) -> timedelta:
        """Calculate time until next market open"""
        now = datetime.now(self.tz)
        
        # If it's during the week but before market open
        if now.weekday() < 5 and now.time() < self.regular_open:
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        # If it's Friday after close or weekend
        elif now.weekday() >= 4 and now.time() > self.regular_close:
            days_ahead = 7 - now.weekday() if now.weekday() == 6 else 1
            next_open = (now + timedelta(days=days_ahead)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
        else:
            next_open = (now + timedelta(days=1)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
        
        return next_open - now

class EnhancedPatternDetector:
    """Advanced pattern detection with multiple pattern types"""
    
    def __init__(self, min_body_ratio=0.5, min_volume_ratio=1.2):
        self.min_body_ratio = min_body_ratio
        self.min_volume_ratio = min_volume_ratio
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect multiple candlestick patterns"""
        if len(df) < 3:
            return []
        
        patterns = []
        
        # Add pattern detection using TA-Lib
        if len(df) >= 5:
            # Hammer
            hammer = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
            if hammer.iloc[-1] != 0:
                patterns.append({
                    'pattern': 'Hammer',
                    'signal': 'bullish',
                    'strength': abs(hammer.iloc[-1]) / 100
                })
            
            # Shooting Star
            shooting_star = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
            if shooting_star.iloc[-1] != 0:
                patterns.append({
                    'pattern': 'Shooting Star',
                    'signal': 'bearish',
                    'strength': abs(shooting_star.iloc[-1]) / 100
                })
            
            # Engulfing
            engulfing = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
            if engulfing.iloc[-1] > 0:
                patterns.append({
                    'pattern': 'Bullish Engulfing',
                    'signal': 'bullish',
                    'strength': 0.8
                })
            elif engulfing.iloc[-1] < 0:
                patterns.append({
                    'pattern': 'Bearish Engulfing',
                    'signal': 'bearish',
                    'strength': 0.8
                })
            
            # Doji
            doji = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
            if doji.iloc[-1] != 0:
                patterns.append({
                    'pattern': 'Doji',
                    'signal': 'neutral',
                    'strength': 0.5
                })
        
        # Custom STRAT patterns
        strat_patterns = self.detect_strat_patterns(df)
        patterns.extend(strat_patterns)
        
        return patterns
    
    def detect_strat_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect STRAT methodology patterns"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        # Get last 3 candles
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        
        # Classify candles (1=inside, 2=directional, 3=outside)
        def classify_candle(current, previous):
            if current['High'] <= previous['High'] and current['Low'] >= previous['Low']:
                return 1  # Inside bar
            elif current['High'] > previous['High'] and current['Low'] < previous['Low']:
                return 3  # Outside bar
            elif current['High'] > previous['High'] or current['Low'] < previous['Low']:
                return 2  # Directional
            return 0
        
        type2 = classify_candle(c2, c1)
        type3 = classify_candle(c3, c2)
        
        # 2-1-2 Reversal (Bullish)
        if type2 == 2 and type3 == 1 and c2['Close'] < c2['Open']:
            patterns.append({
                'pattern': 'STRAT 2-1-2 Bullish',
                'signal': 'bullish',
                'strength': 0.75
            })
        
        # 2-1-2 Reversal (Bearish)
        if type2 == 2 and type3 == 1 and c2['Close'] > c2['Open']:
            patterns.append({
                'pattern': 'STRAT 2-1-2 Bearish',
                'signal': 'bearish',
                'strength': 0.75
            })
        
        # 3-1-2 Breakout
        if type2 == 3 and type3 == 1:
            patterns.append({
                'pattern': 'STRAT 3-1-2 Consolidation',
                'signal': 'neutral',
                'strength': 0.6
            })
        
        return patterns

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, account_balance: float, max_risk_per_trade: float = 0.02,
                 max_positions: int = 5, max_daily_loss: float = 0.06):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        risk_amount = self.account_balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = int(risk_amount / price_risk)
        
        # Check if we can afford it
        max_affordable = int(self.account_balance * 0.95 / entry_price)
        
        return min(position_size, max_affordable)
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        if len(self.open_positions) >= self.max_positions:
            logger.warning("Maximum positions reached")
            return False
        
        if abs(self.daily_pnl) >= self.account_balance * self.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False
        
        return True
    
    def calculate_stop_loss(self, entry_price: float, signal_type: str, 
                          atr: float = None) -> float:
        """Calculate stop loss price"""
        if atr:
            # ATR-based stop loss
            multiplier = 2.0
            if signal_type == 'long':
                return entry_price - (atr * multiplier)
            else:
                return entry_price + (atr * multiplier)
        else:
            # Percentage-based stop loss
            stop_percentage = 0.02  # 2%
            if signal_type == 'long':
                return entry_price * (1 - stop_percentage)
            else:
                return entry_price * (1 + stop_percentage)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                            risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit based on risk/reward ratio"""
        risk = abs(entry_price - stop_loss)
        
        if entry_price > stop_loss:  # Long position
            return entry_price + (risk * risk_reward_ratio)
        else:  # Short position
            return entry_price - (risk * risk_reward_ratio)

class EnhancedTradingBot:
    """Enhanced trading bot with all improvements"""
    
    # Timeframe mapping for yfinance
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '60m',  # Critical fix: yfinance uses 60m not 1h
        '60m': '60m',
        '1d': '1d',
        '1w': '1wk'
    }
    
    def __init__(self, symbols: List[str], timeframe: str = '15m',
                 account_balance: float = 10000, include_prepost: bool = False):
        
        # Map timeframe
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        self.symbols = symbols
        self.input_timeframe = timeframe
        self.yf_interval = self.TIMEFRAME_MAP[timeframe]
        self.include_prepost = include_prepost
        
        # Components
        self.pattern_detector = EnhancedPatternDetector()
        self.risk_manager = RiskManager(account_balance)
        self.market_hours = MarketHours()
        
        # State tracking
        self.last_candle_index = {}  # Track last processed candle per symbol
        self.data_cache = {}  # Cache downloaded data
        self.no_new_candle_count = {}  # Track stale data
        self.max_stale_candles = 10  # Exit after 10 loops with no new data
        
        # Threading
        self.stop_trading = False
        self.trading_lock = threading.Lock()
        
        logger.info(f"Enhanced Trading Bot initialized - Symbols: {symbols}, "
                   f"Timeframe: {timeframe} (mapped to {self.yf_interval})")
    
    def get_historical_data(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch historical data with caching"""
        try:
            # Check cache first (unless forced refresh)
            if not force_refresh and symbol in self.data_cache:
                cached_time, cached_df = self.data_cache[symbol]
                if (datetime.now() - cached_time).seconds < 60:
                    return cached_df
            
            # Determine period based on timeframe
            if self.input_timeframe in ['1m', '5m', '15m', '30m']:
                period = '1d'  # Intraday needs 1d max
            elif self.input_timeframe == '1h':
                period = '5d'
            else:
                period = '30d'
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period=period,
                interval=self.yf_interval,
                prepost=self.include_prepost
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Cache the data
            self.data_cache[symbol] = (datetime.now(), df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        if len(df) < 20:
            return df
        
        # Price-based indicators
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['EMA_9'] = talib.EMA(df['Close'], timeperiod=9)
        df['EMA_21'] = talib.EMA(df['Close'], timeperiod=21)
        
        # Volatility
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Momentum
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # MACD
        if len(df) >= 26:
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
        
        # Volume indicators
        df['Volume_SMA'] = talib.SMA(df['Volume'], timeperiod=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def has_new_candle(self, symbol: str, df: pd.DataFrame) -> bool:
        """Check if we have a new candle to process"""
        if df.empty:
            return False
        
        current_index = df.index[-1]
        
        # First time checking this symbol
        if symbol not in self.last_candle_index:
            self.last_candle_index[symbol] = current_index
            self.no_new_candle_count[symbol] = 0
            return True
        
        # Check if index changed
        if current_index != self.last_candle_index[symbol]:
            self.last_candle_index[symbol] = current_index
            self.no_new_candle_count[symbol] = 0
            return True
        
        # No new candle
        self.no_new_candle_count[symbol] = self.no_new_candle_count.get(symbol, 0) + 1
        
        if self.no_new_candle_count[symbol] >= self.max_stale_candles:
            logger.warning(f"No new candles for {symbol} after {self.max_stale_candles} checks")
        
        return False
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal from data"""
        if len(df) < 20:
            return None
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(df)
        
        if not patterns:
            return None
        
        # Get the strongest pattern
        bullish_patterns = [p for p in patterns if p['signal'] == 'bullish']
        bearish_patterns = [p for p in patterns if p['signal'] == 'bearish']
        
        current_price = df['Close'].iloc[-1]
        current_time = df.index[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df else None
        
        # Generate signal based on patterns and indicators
        signal = None
        
        if bullish_patterns:
            # Additional confirmation from indicators
            rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
            price_above_sma = current_price > df['SMA_20'].iloc[-1] if 'SMA_20' in df else False
            
            confidence = max([p['strength'] for p in bullish_patterns])
            
            # Adjust confidence based on indicators
            if rsi < 30:
                confidence += 0.1  # Oversold
            if price_above_sma:
                confidence += 0.1
            
            if confidence >= 0.6:  # Minimum confidence threshold
                stop_loss = self.risk_manager.calculate_stop_loss(
                    current_price, 'long', atr
                )
                take_profit = self.risk_manager.calculate_take_profit(
                    current_price, stop_loss, 2.5
                )
                
                signal = TradingSignal(
                    timestamp=current_time,
                    symbol=symbol,
                    signal_type=SignalType.BUY if confidence < 0.8 else SignalType.STRONG_BUY,
                    pattern=bullish_patterns[0]['pattern'],
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=2.5,
                    timeframe=self.input_timeframe,
                    indicators={'RSI': rsi, 'ATR': atr}
                )
        
        elif bearish_patterns:
            # Similar logic for bearish signals
            rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
            price_below_sma = current_price < df['SMA_20'].iloc[-1] if 'SMA_20' in df else False
            
            confidence = max([p['strength'] for p in bearish_patterns])
            
            if rsi > 70:
                confidence += 0.1  # Overbought
            if price_below_sma:
                confidence += 0.1
            
            if confidence >= 0.6:
                stop_loss = self.risk_manager.calculate_stop_loss(
                    current_price, 'short', atr
                )
                take_profit = self.risk_manager.calculate_take_profit(
                    current_price, stop_loss, 2.5
                )
                
                signal = TradingSignal(
                    timestamp=current_time,
                    symbol=symbol,
                    signal_type=SignalType.SELL if confidence < 0.8 else SignalType.STRONG_SELL,
                    pattern=bearish_patterns[0]['pattern'],
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=2.5,
                    timeframe=self.input_timeframe,
                    indicators={'RSI': rsi, 'ATR': atr}
                )
        
        return signal
    
    def execute_trade(self, signal: TradingSignal) -> Optional[Position]:
        """Execute trade based on signal"""
        if not self.risk_manager.can_open_position():
            logger.warning(f"Cannot open position for {signal.symbol}")
            return None
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            signal.entry_price, signal.stop_loss
        )
        
        if position_size == 0:
            logger.warning(f"Position size is 0 for {signal.symbol}")
            return None
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            entry_time=signal.timestamp,
            quantity=position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_type='long' if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 'short'
        )
        
        # Add to open positions
        self.risk_manager.open_positions.append(position)
        
        logger.info(f"TRADE EXECUTED: {position.position_type.upper()} {position.quantity} "
                   f"{position.symbol} @ {position.entry_price:.2f} "
                   f"SL: {position.stop_loss:.2f} TP: {position.take_profit:.2f}")
        
        return position
    
    def manage_positions(self):
        """Check and manage open positions"""
        for position in self.risk_manager.open_positions[:]:
            # Get current price
            df = self.get_historical_data(position.symbol)
            if df is None or df.empty:
                continue
            
            current_price = df['Close'].iloc[-1]
            
            # Check stop loss
            if position.position_type == 'long':
                if current_price <= position.stop_loss:
                    self.close_position(position, current_price, "Stop Loss Hit")
                elif current_price >= position.take_profit:
                    self.close_position(position, current_price, "Take Profit Hit")
            else:  # Short position
                if current_price >= position.stop_loss:
                    self.close_position(position, current_price, "Stop Loss Hit")
                elif current_price <= position.take_profit:
                    self.close_position(position, current_price, "Take Profit Hit")
    
    def close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position"""
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.status = 'closed'
        position.pnl = position.calculate_pnl(exit_price)
        
        # Update daily PnL
        self.risk_manager.daily_pnl += position.pnl
        
        # Move to closed positions
        self.risk_manager.open_positions.remove(position)
        self.risk_manager.closed_positions.append(position)
        
        logger.info(f"POSITION CLOSED: {position.symbol} - Reason: {reason} - "
                   f"PnL: ${position.pnl:.2f}")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting enhanced trading bot...")
        
        while not self.stop_trading:
            try:
                # Check market hours
                if not self.market_hours.is_market_open(self.include_prepost):
                    time_to_open = self.market_hours.time_to_next_open()
                    logger.info(f"Market closed. Opens in {time_to_open}")
                    
                    # Sleep for a while
                    time_module.sleep(min(3600, time_to_open.total_seconds()))
                    continue
                
                # Process each symbol
                for symbol in self.symbols:
                    if self.stop_trading:
                        break
                    
                    # Get data
                    df = self.get_historical_data(symbol)
                    if df is None or df.empty:
                        continue
                    
                    # Check for new candle
                    if not self.has_new_candle(symbol, df):
                        logger.debug(f"No new candle for {symbol}")
                        continue
                    
                    # Generate signal
                    signal = self.generate_signal(symbol, df)
                    
                    if signal:
                        logger.info(f"SIGNAL: {signal.signal_type.value} for {symbol} - "
                                  f"Pattern: {signal.pattern} - Confidence: {signal.confidence:.2f}")
                        
                        # Execute trade if conditions are met
                        if signal.confidence >= 0.7:  # Higher threshold for execution
                            self.execute_trade(signal)
                
                # Manage existing positions
                self.manage_positions()
                
                # Sleep based on timeframe
                sleep_seconds = {
                    '1m': 60,
                    '5m': 300,
                    '15m': 900,
                    '30m': 1800,
                    '1h': 3600,
                    '60m': 3600,
                    '1d': 86400
                }.get(self.input_timeframe, 60)
                
                logger.debug(f"Sleeping for {sleep_seconds} seconds...")
                time_module.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time_module.sleep(60)
        
        self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down trading bot...")
        
        # Close all open positions
        for position in self.risk_manager.open_positions[:]:
            df = self.get_historical_data(position.symbol)
            if df is not None and not df.empty:
                current_price = df['Close'].iloc[-1]
                self.close_position(position, current_price, "Shutdown")
        
        # Print summary
        total_pnl = sum([p.pnl for p in self.risk_manager.closed_positions if p.pnl])
        logger.info(f"Trading Summary: Total PnL: ${total_pnl:.2f}")
        logger.info(f"Total Trades: {len(self.risk_manager.closed_positions)}")
        
        if self.risk_manager.closed_positions:
            winning_trades = [p for p in self.risk_manager.closed_positions if p.pnl and p.pnl > 0]
            win_rate = len(winning_trades) / len(self.risk_manager.closed_positions) * 100
            logger.info(f"Win Rate: {win_rate:.1f}%")

def main():
    """Main entry point"""
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    timeframe = '15m'  # Will be properly mapped to yfinance format
    account_balance = 100000
    
    # Create and run bot
    bot = EnhancedTradingBot(
        symbols=symbols,
        timeframe=timeframe,
        account_balance=account_balance,
        include_prepost=False  # Set to True for extended hours
    )
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        bot.shutdown()

if __name__ == "__main__":
    main()