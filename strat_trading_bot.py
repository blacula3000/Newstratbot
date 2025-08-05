import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='strat_trading_bot.log'
)

class BarType(Enum):
    """The Strat bar classifications"""
    INSIDE = 1  # Inside bar - lower high, higher low
    DIRECTIONAL_UP = 2  # Up bar - higher high, same or lower low
    DIRECTIONAL_DOWN = 2  # Down bar - lower low, same or higher high
    OUTSIDE = 3  # Outside bar - higher high AND lower low

class StratScenario:
    """Identifies Strat scenarios and actionable signals"""
    
    @staticmethod
    def classify_bar(current: pd.Series, previous: pd.Series) -> BarType:
        """Classify a bar according to The Strat methodology"""
        higher_high = current['High'] > previous['High']
        lower_high = current['High'] < previous['High']
        higher_low = current['Low'] > previous['Low']
        lower_low = current['Low'] < previous['Low']
        
        # Scenario 1: Inside bar
        if lower_high and higher_low:
            return BarType.INSIDE
        
        # Scenario 3: Outside bar
        elif higher_high and lower_low:
            return BarType.OUTSIDE
        
        # Scenario 2: Directional bars
        elif higher_high:
            return BarType.DIRECTIONAL_UP
        elif lower_low:
            return BarType.DIRECTIONAL_DOWN
        
        # Equal highs/lows treated as inside
        return BarType.INSIDE
    
    @staticmethod
    def get_bar_sequence(bars: pd.DataFrame, lookback: int = 5) -> List[BarType]:
        """Get sequence of bar types for pattern recognition"""
        if len(bars) < lookback + 1:
            return []
        
        sequence = []
        for i in range(len(bars) - lookback, len(bars)):
            if i > 0:
                bar_type = StratScenario.classify_bar(bars.iloc[i], bars.iloc[i-1])
                sequence.append(bar_type)
        
        return sequence

class StratPatterns:
    """Identifies actionable Strat patterns"""
    
    @staticmethod
    def is_2_2_reversal(sequence: List[BarType], bars: pd.DataFrame) -> Optional[str]:
        """Identify 2-2 reversal pattern (two directional bars in opposite directions)"""
        if len(sequence) < 2:
            return None
        
        # Check last two bars
        if len(bars) < 3:
            return None
            
        prev_bar = bars.iloc[-2]
        curr_bar = bars.iloc[-1]
        
        # 2 up followed by 2 down
        if (sequence[-2] == BarType.DIRECTIONAL_UP and 
            sequence[-1] == BarType.DIRECTIONAL_DOWN and
            curr_bar['Close'] < prev_bar['Low']):
            return "BEARISH_2_2_REVERSAL"
        
        # 2 down followed by 2 up
        elif (sequence[-2] == BarType.DIRECTIONAL_DOWN and 
              sequence[-1] == BarType.DIRECTIONAL_UP and
              curr_bar['Close'] > prev_bar['High']):
            return "BULLISH_2_2_REVERSAL"
        
        return None
    
    @staticmethod
    def is_3_1_2_combo(sequence: List[BarType], bars: pd.DataFrame) -> Optional[str]:
        """Identify 3-1-2 combo (outside bar, inside bar, directional break)"""
        if len(sequence) < 3:
            return None
            
        if (sequence[-3] == BarType.OUTSIDE and 
            sequence[-2] == BarType.INSIDE):
            
            # Check direction of breakout
            if sequence[-1] == BarType.DIRECTIONAL_UP:
                if bars.iloc[-1]['Close'] > bars.iloc[-3]['High']:
                    return "BULLISH_3_1_2_COMBO"
            elif sequence[-1] == BarType.DIRECTIONAL_DOWN:
                if bars.iloc[-1]['Close'] < bars.iloc[-3]['Low']:
                    return "BEARISH_3_1_2_COMBO"
        
        return None
    
    @staticmethod
    def is_1_2_2_reversal(sequence: List[BarType], bars: pd.DataFrame) -> Optional[str]:
        """Identify 1-2-2 reversal (inside bar followed by two directional bars)"""
        if len(sequence) < 3:
            return None
            
        if sequence[-3] == BarType.INSIDE:
            # Bullish: inside bar, up bar, up bar with higher close
            if (sequence[-2] == BarType.DIRECTIONAL_UP and 
                sequence[-1] == BarType.DIRECTIONAL_UP and
                bars.iloc[-1]['Close'] > bars.iloc[-2]['High']):
                return "BULLISH_1_2_2_CONTINUATION"
            
            # Bearish: inside bar, down bar, down bar with lower close
            elif (sequence[-2] == BarType.DIRECTIONAL_DOWN and 
                  sequence[-1] == BarType.DIRECTIONAL_DOWN and
                  bars.iloc[-1]['Close'] < bars.iloc[-2]['Low']):
                return "BEARISH_1_2_2_CONTINUATION"
        
        return None

class TimeframeContinuity:
    """Analyzes multiple timeframes for continuity"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.timeframes = ['1d', '1h', '15m', '5m']
        
    def get_trend_direction(self, bars: pd.DataFrame) -> str:
        """Determine trend direction based on price action"""
        if len(bars) < 10:
            return "NEUTRAL"
            
        # Simple trend: compare current close to 10-bar average
        current_close = bars['Close'].iloc[-1]
        ma10 = bars['Close'].rolling(10).mean().iloc[-1]
        
        # Also check higher highs/higher lows or lower highs/lower lows
        recent_bars = bars.tail(5)
        highs_ascending = recent_bars['High'].is_monotonic_increasing
        lows_ascending = recent_bars['Low'].is_monotonic_increasing
        
        if current_close > ma10 and (highs_ascending or lows_ascending):
            return "BULLISH"
        elif current_close < ma10 and not (highs_ascending or lows_ascending):
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def check_continuity(self) -> Dict[str, str]:
        """Check trend continuity across timeframes"""
        continuity = {}
        
        for tf in self.timeframes:
            try:
                ticker = yf.Ticker(self.symbol)
                
                # Adjust period based on timeframe
                if tf == '1d':
                    data = ticker.history(period='3mo', interval=tf)
                elif tf == '1h':
                    data = ticker.history(period='1mo', interval=tf)
                else:
                    data = ticker.history(period='5d', interval=tf)
                
                if not data.empty:
                    continuity[tf] = self.get_trend_direction(data)
                else:
                    continuity[tf] = "NO_DATA"
                    
            except Exception as e:
                logging.error(f"Error checking {tf} continuity: {e}")
                continuity[tf] = "ERROR"
        
        return continuity
    
    def has_full_continuity(self, continuity: Dict[str, str]) -> Tuple[bool, str]:
        """Check if all timeframes align"""
        valid_trends = [v for v in continuity.values() if v not in ["NO_DATA", "ERROR", "NEUTRAL"]]
        
        if not valid_trends:
            return False, "NEUTRAL"
        
        # Check if all valid trends are the same
        if all(trend == "BULLISH" for trend in valid_trends):
            return True, "BULLISH"
        elif all(trend == "BEARISH" for trend in valid_trends):
            return True, "BEARISH"
        else:
            return False, "MIXED"

class StratTradingBot:
    """The Strat methodology trading bot"""
    
    def __init__(self, symbol: str, primary_timeframe: str = '5m'):
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.timeframe_continuity = TimeframeContinuity(symbol)
        self.last_signal_time = None
        self.signal_cooldown = 300  # 5 minutes between signals
        
        logging.info(f"Strat Bot initialized for {symbol} on {primary_timeframe}")
    
    def fetch_data(self, timeframe: str = None) -> pd.DataFrame:
        """Fetch market data"""
        try:
            ticker = yf.Ticker(self.symbol)
            tf = timeframe or self.primary_timeframe
            
            # Adjust period based on timeframe
            if tf in ['1m', '5m']:
                period = '1d'
            elif tf == '15m':
                period = '5d'
            elif tf == '1h':
                period = '1mo'
            else:
                period = '3mo'
                
            data = ticker.history(period=period, interval=tf)
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def analyze_strat_patterns(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze for Strat patterns"""
        signals = {
            'bar_type': None,
            'pattern': None,
            'actionable': False,
            'direction': None,
            'strength': 0,
            'sequence': []
        }
        
        if len(data) < 5:
            return signals
        
        # Get bar sequence
        sequence = StratScenario.get_bar_sequence(data, lookback=5)
        if not sequence:
            return signals
        
        signals['sequence'] = [bar.name for bar in sequence[-3:]]  # Last 3 bars
        
        # Current bar type
        if len(data) >= 2:
            current_bar_type = StratScenario.classify_bar(data.iloc[-1], data.iloc[-2])
            signals['bar_type'] = current_bar_type.name
        
        # Check for patterns
        pattern_checks = [
            StratPatterns.is_2_2_reversal(sequence, data),
            StratPatterns.is_3_1_2_combo(sequence, data),
            StratPatterns.is_1_2_2_reversal(sequence, data)
        ]
        
        for pattern in pattern_checks:
            if pattern:
                signals['pattern'] = pattern
                signals['actionable'] = True
                signals['direction'] = 'BULLISH' if 'BULLISH' in pattern else 'BEARISH'
                
                # Assign strength based on pattern type
                if '3_1_2' in pattern:
                    signals['strength'] = 3  # Strongest
                elif '2_2' in pattern:
                    signals['strength'] = 2
                else:
                    signals['strength'] = 1
                break
        
        return signals
    
    def generate_trade_signal(self, strat_signals: Dict, continuity: Dict[str, str]) -> Dict:
        """Generate actionable trade signals with risk management"""
        trade_signal = {
            'action': None,
            'entry': None,
            'stop_loss': None,
            'target': None,
            'timeframe_alignment': False,
            'confidence': 0
        }
        
        if not strat_signals['actionable']:
            return trade_signal
        
        # Check timeframe continuity
        has_continuity, trend = self.timeframe_continuity.has_full_continuity(continuity)
        trade_signal['timeframe_alignment'] = has_continuity
        
        # Only trade if pattern aligns with timeframe continuity
        if has_continuity and trend == strat_signals['direction']:
            trade_signal['confidence'] = 80 + (strat_signals['strength'] * 5)
        else:
            trade_signal['confidence'] = 50 + (strat_signals['strength'] * 5)
        
        # Set trade parameters if confidence is high enough
        if trade_signal['confidence'] >= 70:
            current_price = self.current_data['Close'].iloc[-1]
            trade_signal['entry'] = current_price
            
            # Calculate stop loss and targets based on ATR or recent range
            recent_range = self.current_data['High'].tail(10).max() - self.current_data['Low'].tail(10).min()
            risk_amount = recent_range * 0.5  # 50% of recent range
            
            if strat_signals['direction'] == 'BULLISH':
                trade_signal['action'] = 'BUY'
                trade_signal['stop_loss'] = current_price - risk_amount
                trade_signal['target'] = current_price + (risk_amount * 2)  # 2:1 risk/reward
            else:
                trade_signal['action'] = 'SELL'
                trade_signal['stop_loss'] = current_price + risk_amount
                trade_signal['target'] = current_price - (risk_amount * 2)
        
        return trade_signal
    
    def should_send_signal(self) -> bool:
        """Check if enough time has passed since last signal"""
        if self.last_signal_time is None:
            return True
        
        time_elapsed = time.time() - self.last_signal_time
        return time_elapsed >= self.signal_cooldown
    
    def run(self):
        """Main bot loop"""
        print(f"Starting Strat Trading Bot for {self.symbol}...")
        print("="*50)
        print("The Strat Methodology Active")
        print("Monitoring for: 2-2 Reversals, 3-1-2 Combos, 1-2-2 Continuations")
        print("="*50)
        
        logging.info(f"Strat Bot started for {self.symbol}")
        
        while True:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Fetch current data
                self.current_data = self.fetch_data()
                
                if not self.current_data.empty:
                    # Get Strat patterns
                    strat_signals = self.analyze_strat_patterns(self.current_data)
                    
                    # Check timeframe continuity
                    continuity = self.timeframe_continuity.check_continuity()
                    
                    # Generate trade signal
                    trade_signal = self.generate_trade_signal(strat_signals, continuity)
                    
                    # Current price
                    current_price = self.current_data['Close'].iloc[-1]
                    
                    # Display status
                    print(f"\n[{current_time}] {self.symbol} @ ${current_price:.2f}")
                    print(f"Bar Type: {strat_signals['bar_type']}")
                    print(f"Sequence: {' -> '.join(strat_signals['sequence'][-3:])}")
                    
                    # Show timeframe continuity
                    continuity_str = " | ".join([f"{tf}:{trend[:1]}" for tf, trend in continuity.items()])
                    print(f"Timeframes: {continuity_str}")
                    
                    # Show signals if actionable
                    if strat_signals['actionable'] and self.should_send_signal():
                        print("\n" + "🚨 STRAT SIGNAL DETECTED! 🚨")
                        print(f"Pattern: {strat_signals['pattern']}")
                        print(f"Direction: {strat_signals['direction']}")
                        print(f"Strength: {'⭐' * strat_signals['strength']}")
                        
                        if trade_signal['action']:
                            print(f"\n📊 TRADE SETUP:")
                            print(f"Action: {trade_signal['action']}")
                            print(f"Entry: ${trade_signal['entry']:.2f}")
                            print(f"Stop Loss: ${trade_signal['stop_loss']:.2f}")
                            print(f"Target: ${trade_signal['target']:.2f}")
                            print(f"Risk/Reward: 1:2")
                            print(f"Confidence: {trade_signal['confidence']}%")
                            
                            self.last_signal_time = time.time()
                            
                            logging.info(f"Strat Signal: {strat_signals['pattern']} for {self.symbol} @ ${current_price:.2f}")
                    
                    print("-" * 50)
                
                # Wait before next check
                time.sleep(30)  # 30 seconds for more responsive monitoring
                
            except KeyboardInterrupt:
                print("\nStrat Bot stopped by user")
                logging.info("Strat Bot stopped by user")
                break
                
            except Exception as e:
                print(f"Error: {e}")
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    # Example usage
    symbol = input("Enter stock symbol (e.g., SPY, QQQ, AAPL): ").upper()
    timeframe = input("Enter primary timeframe (1m, 5m, 15m, 1h): ") or '5m'
    
    bot = StratTradingBot(symbol=symbol, primary_timeframe=timeframe)
    bot.run()