"""
STRAT Signal Engine - Proper Implementation of The STRAT Methodology
Implements actionable signals with trigger levels and Full Time Frame Continuity (FTFC)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf


class StratCandle:
    """Represents a single STRAT candlestick with classification"""
    
    def __init__(self, open_price: float, high: float, low: float, close: float, 
                 timestamp: datetime, volume: int = 0):
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.timestamp = timestamp
        self.volume = volume
        self.strat_type = None
        self.trigger_high = None
        self.trigger_low = None
        
    def __str__(self):
        return f"STRAT {self.strat_type} - O:{self.open:.2f} H:{self.high:.2f} L:{self.low:.2f} C:{self.close:.2f}"


class StratSignalEngine:
    """Main engine for detecting actionable STRAT signals"""
    
    def __init__(self):
        self.timeframes = ['1d', '4h', '1h', '30m', '15m', '5m']
        self.signals = []
        
    def classify_candle_strat_type(self, current_candle: StratCandle, 
                                 previous_candle: StratCandle) -> str:
        """
        Classify candle as Type 1 (Inside), 2U, 2D (Directional), or 3 (Outside)
        
        Args:
            current_candle: Current candle to classify
            previous_candle: Previous candle for reference
            
        Returns:
            str: '1', '2U', '2D', or '3'
        """
        curr_high = current_candle.high
        curr_low = current_candle.low
        prev_high = previous_candle.high
        prev_low = previous_candle.low
        
        # Check if current candle breaks previous high
        breaks_high = curr_high > prev_high
        # Check if current candle breaks previous low  
        breaks_low = curr_low < prev_low
        
        if breaks_high and breaks_low:
            # Breaks both high and low = Outside bar (Type 3)
            current_candle.strat_type = '3'
            # Set trigger levels
            current_candle.trigger_high = curr_high
            current_candle.trigger_low = curr_low
            
        elif breaks_high and not breaks_low:
            # Breaks only high = Directional Up (Type 2U)
            current_candle.strat_type = '2U'
            current_candle.trigger_high = curr_high
            current_candle.trigger_low = prev_low
            
        elif breaks_low and not breaks_high:
            # Breaks only low = Directional Down (Type 2D)
            current_candle.strat_type = '2D'
            current_candle.trigger_high = prev_high
            current_candle.trigger_low = curr_low
            
        else:
            # Stays within previous bar = Inside bar (Type 1)
            current_candle.strat_type = '1'
            current_candle.trigger_high = prev_high
            current_candle.trigger_low = prev_low
            
        return current_candle.strat_type
    
    def detect_trigger_break(self, candles: List[StratCandle], 
                           trigger_candle_index: int) -> Dict:
        """
        Detect when price breaks above/below trigger levels
        
        Args:
            candles: List of candles
            trigger_candle_index: Index of the candle with trigger levels
            
        Returns:
            Dict with trigger break information
        """
        if trigger_candle_index >= len(candles) - 1:
            return {'trigger_broken': False}
            
        trigger_candle = candles[trigger_candle_index]
        subsequent_candles = candles[trigger_candle_index + 1:]
        
        results = {
            'trigger_broken': False,
            'direction': None,
            'break_candle': None,
            'trigger_level': None,
            'break_price': None
        }
        
        for candle in subsequent_candles:
            # Check for upside trigger break
            if candle.high > trigger_candle.trigger_high:
                results.update({
                    'trigger_broken': True,
                    'direction': 'LONG',
                    'break_candle': candle,
                    'trigger_level': trigger_candle.trigger_high,
                    'break_price': candle.high
                })
                break
                
            # Check for downside trigger break
            elif candle.low < trigger_candle.trigger_low:
                results.update({
                    'trigger_broken': True,
                    'direction': 'SHORT',
                    'break_candle': candle,
                    'trigger_level': trigger_candle.trigger_low,
                    'break_price': candle.low
                })
                break
                
        return results
    
    def check_full_timeframe_continuity(self, symbol: str, 
                                      direction: str) -> Dict:
        """
        Check Full Time Frame Continuity (FTFC) across multiple timeframes
        
        Args:
            symbol: Stock symbol
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Dict with FTFC analysis results
        """
        ftfc_results = {
            'has_continuity': False,
            'continuity_score': 0,
            'timeframe_analysis': {},
            'current_price': None
        }
        
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period='1d', interval='1m')
            if current_data.empty:
                return ftfc_results
                
            current_price = current_data['Close'].iloc[-1]
            ftfc_results['current_price'] = current_price
            
            continuity_count = 0
            
            for timeframe in self.timeframes:
                try:
                    # Get appropriate period for each timeframe
                    period_map = {
                        '1d': '3mo', '4h': '1mo', '1h': '5d', 
                        '30m': '5d', '15m': '2d', '5m': '1d'
                    }
                    
                    period = period_map.get(timeframe, '1mo')
                    tf_data = ticker.history(period=period, interval=timeframe)
                    
                    if not tf_data.empty:
                        tf_open = tf_data['Open'].iloc[-1]  # Current timeframe open
                        
                        # For LONG: all higher TF opens should be below current price
                        # For SHORT: all higher TF opens should be above current price
                        if direction == 'LONG':
                            has_continuity = tf_open < current_price
                        else:  # SHORT
                            has_continuity = tf_open > current_price
                            
                        ftfc_results['timeframe_analysis'][timeframe] = {
                            'open': tf_open,
                            'current_price': current_price,
                            'has_continuity': has_continuity,
                            'difference': current_price - tf_open
                        }
                        
                        if has_continuity:
                            continuity_count += 1
                            
                except Exception as e:
                    print(f"Error checking {timeframe}: {e}")
                    
            # Calculate continuity score
            ftfc_results['continuity_score'] = (continuity_count / len(self.timeframes)) * 100
            ftfc_results['has_continuity'] = ftfc_results['continuity_score'] >= 70
            
        except Exception as e:
            print(f"Error in FTFC analysis: {e}")
            
        return ftfc_results
    
    def identify_actionable_signal(self, symbol: str, 
                                 timeframe: str = '15m') -> Dict:
        """
        Identify complete actionable STRAT signals
        
        Args:
            symbol: Stock symbol
            timeframe: Primary timeframe for analysis
            
        Returns:
            Dict with complete signal analysis
        """
        signal_result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'has_signal': False,
            'signal_type': None,
            'direction': None,
            'entry_price': None,
            'stop_loss': None,
            'target': None,
            'confidence_score': 0,
            'pattern_sequence': [],
            'trigger_info': {},
            'ftfc_analysis': {},
            'timestamp': datetime.now()
        }
        
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            period_map = {
                '1d': '3mo', '4h': '1mo', '1h': '5d',
                '30m': '5d', '15m': '2d', '5m': '1d'
            }
            period = period_map.get(timeframe, '2d')
            
            hist_data = ticker.history(period=period, interval=timeframe)
            if len(hist_data) < 5:
                return signal_result
            
            # Convert to StratCandle objects
            candles = []
            for idx, (timestamp, row) in enumerate(hist_data.iterrows()):
                candle = StratCandle(
                    open_price=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    timestamp=timestamp,
                    volume=int(row['Volume'])
                )
                candles.append(candle)
            
            # Classify each candle (need previous candle for classification)
            for i in range(1, len(candles)):
                strat_type = self.classify_candle_strat_type(candles[i], candles[i-1])
                
            # Look for actionable patterns in recent candles
            recent_candles = candles[-5:]  # Last 5 candles
            pattern_sequence = [c.strat_type for c in recent_candles]
            signal_result['pattern_sequence'] = pattern_sequence
            
            # Check for high-probability patterns
            actionable_patterns = self._identify_actionable_patterns(recent_candles)
            
            if actionable_patterns:
                best_pattern = actionable_patterns[0]  # Take the best pattern
                
                # Check for trigger breaks
                trigger_info = self.detect_trigger_break(
                    candles, 
                    len(candles) - len(recent_candles) + best_pattern['candle_index']
                )
                signal_result['trigger_info'] = trigger_info
                
                if trigger_info['trigger_broken']:
                    direction = trigger_info['direction']
                    
                    # Check Full Time Frame Continuity
                    ftfc_analysis = self.check_full_timeframe_continuity(symbol, direction)
                    signal_result['ftfc_analysis'] = ftfc_analysis
                    
                    # Calculate confidence score
                    confidence = self._calculate_confidence_score(
                        best_pattern, trigger_info, ftfc_analysis
                    )
                    
                    if confidence >= 70:  # Only high-confidence signals
                        signal_result.update({
                            'has_signal': True,
                            'signal_type': best_pattern['pattern_name'],
                            'direction': direction,
                            'entry_price': trigger_info['break_price'],
                            'confidence_score': confidence,
                            'stop_loss': self._calculate_stop_loss(
                                recent_candles, direction, trigger_info
                            ),
                            'target': self._calculate_target(
                                recent_candles, direction, trigger_info
                            )
                        })
                        
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            
        return signal_result
    
    def _identify_actionable_patterns(self, candles: List[StratCandle]) -> List[Dict]:
        """Identify high-probability actionable patterns"""
        patterns = []
        
        if len(candles) < 3:
            return patterns
            
        # Look for specific STRAT combinations
        for i in range(len(candles) - 2):
            three_candle_seq = candles[i:i+3]
            types = [c.strat_type for c in three_candle_seq]
            
            # 2-1-2 Reversal (high probability)
            if types == ['2U', '1', '2U'] or types == ['2D', '1', '2D']:
                patterns.append({
                    'pattern_name': '2-1-2 Reversal',
                    'candle_index': i+2,  # Signal candle
                    'confidence_base': 85,
                    'pattern_type': 'reversal'
                })
                
            # 3-1-2 Combo (volatility expansion)
            elif types == ['3', '1', '2U'] or types == ['3', '1', '2D']:
                patterns.append({
                    'pattern_name': '3-1-2 Combo',
                    'candle_index': i+2,
                    'confidence_base': 80,
                    'pattern_type': 'continuation'
                })
                
            # 1-2-2 Broadening (breakout pattern)
            elif types == ['1', '2U', '2U'] or types == ['1', '2D', '2D']:
                patterns.append({
                    'pattern_name': '1-2-2 Breakout',
                    'candle_index': i+2,
                    'confidence_base': 75,
                    'pattern_type': 'breakout'
                })
                
        # Sort by confidence (highest first)
        patterns.sort(key=lambda x: x['confidence_base'], reverse=True)
        return patterns
    
    def _calculate_confidence_score(self, pattern: Dict, trigger_info: Dict, 
                                  ftfc_analysis: Dict) -> int:
        """Calculate overall confidence score for the signal"""
        base_confidence = pattern['confidence_base']
        
        # Add FTFC bonus
        ftfc_bonus = ftfc_analysis.get('continuity_score', 0) * 0.2
        
        # Add trigger strength bonus
        if trigger_info.get('trigger_broken', False):
            trigger_bonus = 10
        else:
            trigger_bonus = 0
            
        total_confidence = min(100, base_confidence + ftfc_bonus + trigger_bonus)
        return int(total_confidence)
    
    def _calculate_stop_loss(self, candles: List[StratCandle], direction: str, 
                           trigger_info: Dict) -> float:
        """Calculate stop loss level"""
        if direction == 'LONG':
            # Stop below the lowest low of recent candles
            recent_lows = [c.low for c in candles[-3:]]
            return min(recent_lows) - 0.01
        else:  # SHORT
            # Stop above the highest high of recent candles
            recent_highs = [c.high for c in candles[-3:]]
            return max(recent_highs) + 0.01
    
    def _calculate_target(self, candles: List[StratCandle], direction: str, 
                        trigger_info: Dict) -> float:
        """Calculate profit target (2:1 risk-reward minimum)"""
        entry_price = trigger_info['break_price']
        
        if direction == 'LONG':
            stop_loss = self._calculate_stop_loss(candles, direction, trigger_info)
            risk = entry_price - stop_loss
            return entry_price + (risk * 2)  # 2:1 reward-to-risk
        else:  # SHORT
            stop_loss = self._calculate_stop_loss(candles, direction, trigger_info)
            risk = stop_loss - entry_price
            return entry_price - (risk * 2)  # 2:1 reward-to-risk
    
    def scan_multiple_symbols(self, symbols: List[str], 
                            timeframe: str = '15m') -> List[Dict]:
        """Scan multiple symbols for actionable signals"""
        all_signals = []
        
        for symbol in symbols:
            print(f"Scanning {symbol}...")
            signal = self.identify_actionable_signal(symbol, timeframe)
            
            if signal['has_signal']:
                all_signals.append(signal)
                print(f"✅ SIGNAL FOUND: {symbol} - {signal['signal_type']} "
                      f"{signal['direction']} - Confidence: {signal['confidence_score']}%")
            else:
                print(f"❌ No signal: {symbol}")
                
        # Sort by confidence score
        all_signals.sort(key=lambda x: x['confidence_score'], reverse=True)
        return all_signals


# Example usage and testing
if __name__ == "__main__":
    # Initialize the engine
    engine = StratSignalEngine()
    
    # Test symbols
    test_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    
    print("🔍 STRAT Signal Scanner Starting...")
    print("=" * 50)
    
    # Scan for signals
    signals = engine.scan_multiple_symbols(test_symbols, '15m')
    
    print(f"\n📊 SCAN COMPLETE: Found {len(signals)} actionable signals")
    print("=" * 50)
    
    # Display results
    for i, signal in enumerate(signals, 1):
        print(f"\n🎯 SIGNAL #{i}")
        print(f"Symbol: {signal['symbol']}")
        print(f"Pattern: {signal['signal_type']}")
        print(f"Direction: {signal['direction']}")
        print(f"Entry: ${signal['entry_price']:.2f}")
        print(f"Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"Target: ${signal['target']:.2f}")
        print(f"Confidence: {signal['confidence_score']}%")
        print(f"FTFC Score: {signal['ftfc_analysis'].get('continuity_score', 0):.1f}%")
        print(f"Pattern Sequence: {' -> '.join(signal['pattern_sequence'])}")
        
    if not signals:
        print("\n📭 No actionable signals found at this time.")
        print("Waiting for proper STRAT setups with trigger breaks and FTFC...")