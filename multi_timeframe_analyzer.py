"""
Multi-Timeframe Analysis Module
Analyzes assets across multiple timeframes and provides trend/signal status
"""

import yfinance as yf
import pandas as pd
import talib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """Analyzes assets across multiple timeframes"""
    
    def __init__(self):
        self.timeframe_mappings = {
            '1m': ('1m', 60, '7d'),      # interval, periods, yf_period
            '5m': ('5m', 100, '5d'),
            '15m': ('15m', 100, '7d'),
            '30m': ('30m', 100, '30d'),
            '1h': ('1h', 100, '30d'),
            '2h': ('2h', 100, '60d'),
            '4h': ('4h', 100, '60d'),
            '1d': ('1d', 100, '100d'),
            '1w': ('1wk', 52, '1y')
        }
        
        self.default_timeframes = ['15m', '30m', '1h', '1d']
        
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Analyze a single timeframe for trend and signals
        
        Returns:
            Dictionary with trend, signal, indicators, and confidence
        """
        try:
            if df.empty or len(df) < 20:
                return {
                    'timeframe': timeframe,
                    'trend': 'neutral',
                    'signal': 'hold',
                    'arrow': '→',
                    'color': 'gray',
                    'confidence': 0,
                    'indicators': {}
                }
            
            # Calculate indicators
            df['ema_9'] = talib.EMA(df['Close'], timeperiod=9)
            df['ema_20'] = talib.EMA(df['Close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['Close'], timeperiod=50) if len(df) >= 50 else df['ema_20']
            df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
            
            # MACD
            if len(df) >= 26:
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                    df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
            else:
                df['macd'] = df['macd_signal'] = df['macd_hist'] = 0
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Determine trend
            trend_score = 0
            if latest['Close'] > latest['ema_9']:
                trend_score += 1
            if latest['Close'] > latest['ema_20']:
                trend_score += 1
            if latest['ema_9'] > latest['ema_20']:
                trend_score += 1
            if latest['ema_20'] > latest['ema_50']:
                trend_score += 1
            
            # Determine signal strength
            signal_score = 0
            
            # EMA crossovers
            if latest['ema_9'] > latest['ema_20'] and prev['ema_9'] <= prev['ema_20']:
                signal_score += 2  # Bullish crossover
            elif latest['ema_9'] < latest['ema_20'] and prev['ema_9'] >= prev['ema_20']:
                signal_score -= 2  # Bearish crossover
            
            # RSI signals
            if latest['rsi'] < 30:
                signal_score += 1  # Oversold
            elif latest['rsi'] > 70:
                signal_score -= 1  # Overbought
            
            # MACD signals
            if latest['macd'] > latest['macd_signal']:
                signal_score += 1
            else:
                signal_score -= 1
            
            # Price momentum
            price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
            if price_change > 1:
                signal_score += 1
            elif price_change < -1:
                signal_score -= 1
            
            # Determine overall status
            if trend_score >= 3:
                trend = 'bullish'
            elif trend_score <= 1:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Determine signal and arrow
            if signal_score >= 2:
                signal = 'strong_buy'
                arrow = '⬆'
                color = 'green'
            elif signal_score >= 1:
                signal = 'buy'
                arrow = '↗'
                color = 'lightgreen'
            elif signal_score <= -2:
                signal = 'strong_sell'
                arrow = '⬇'
                color = 'red'
            elif signal_score <= -1:
                signal = 'sell'
                arrow = '↘'
                color = 'lightcoral'
            else:
                signal = 'hold'
                arrow = '→'
                color = 'gray'
            
            # Calculate confidence (0-100)
            confidence = min(100, abs(signal_score) * 20)
            
            return {
                'timeframe': timeframe,
                'trend': trend,
                'signal': signal,
                'arrow': arrow,
                'color': color,
                'confidence': confidence,
                'indicators': {
                    'price': float(latest['Close']),
                    'ema_9': float(latest['ema_9']),
                    'ema_20': float(latest['ema_20']),
                    'ema_50': float(latest['ema_50']),
                    'rsi': float(latest['rsi']) if not pd.isna(latest['rsi']) else 50,
                    'macd': float(latest['macd']),
                    'volume': float(latest['Volume']),
                    'price_change': float(price_change)
                },
                'trend_score': trend_score,
                'signal_score': signal_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return {
                'timeframe': timeframe,
                'trend': 'error',
                'signal': 'error',
                'arrow': '?',
                'color': 'gray',
                'confidence': 0,
                'indicators': {},
                'error': str(e)
            }
    
    def get_multi_timeframe_analysis(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """
        Get analysis across multiple timeframes for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            timeframes: List of timeframes to analyze (default: ['15m', '30m', '1h', '1d'])
        
        Returns:
            Dictionary with analysis for each timeframe
        """
        if timeframes is None:
            timeframes = self.default_timeframes
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {},
            'overall_signal': 'neutral',
            'overall_confidence': 0
        }
        
        signal_scores = []
        confidences = []
        
        for tf in timeframes:
            try:
                # Get data for this timeframe
                df = self.fetch_data(symbol, tf)
                
                # Analyze the timeframe
                analysis = self.analyze_timeframe(df, tf)
                results['timeframes'][tf] = analysis
                
                # Collect scores for overall analysis
                if 'signal_score' in analysis:
                    signal_scores.append(analysis['signal_score'])
                    confidences.append(analysis['confidence'])
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} {tf}: {e}")
                results['timeframes'][tf] = {
                    'timeframe': tf,
                    'trend': 'error',
                    'signal': 'error',
                    'arrow': '?',
                    'color': 'gray',
                    'confidence': 0,
                    'error': str(e)
                }
        
        # Calculate overall signal
        if signal_scores:
            avg_signal = np.mean(signal_scores)
            if avg_signal >= 1.5:
                results['overall_signal'] = 'strong_buy'
                results['overall_arrow'] = '⬆⬆'
            elif avg_signal >= 0.5:
                results['overall_signal'] = 'buy'
                results['overall_arrow'] = '⬆'
            elif avg_signal <= -1.5:
                results['overall_signal'] = 'strong_sell'
                results['overall_arrow'] = '⬇⬇'
            elif avg_signal <= -0.5:
                results['overall_signal'] = 'sell'
                results['overall_arrow'] = '⬇'
            else:
                results['overall_signal'] = 'neutral'
                results['overall_arrow'] = '→'
            
            results['overall_confidence'] = int(np.mean(confidences))
        
        return results
    
    def fetch_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch historical data for a symbol and timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '15m', '1h', '1d')
        
        Returns:
            DataFrame with OHLCV data
        """
        if timeframe not in self.timeframe_mappings:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        interval, periods, period = self.timeframe_mappings[timeframe]
        
        try:
            # Handle crypto symbols
            if '/' in symbol:
                symbol = symbol.replace('/', '-')
            elif 'USDT' in symbol:
                symbol = symbol.replace('USDT', '-USD')
            
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def get_timeframe_confluence(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """
        Check for confluence across timeframes
        
        Returns:
            Dictionary with confluence analysis
        """
        analysis = self.get_multi_timeframe_analysis(symbol, timeframes)
        
        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for tf_data in analysis['timeframes'].values():
            signal = tf_data.get('signal', 'hold')
            if signal in ['strong_buy', 'buy']:
                bullish_count += 1
            elif signal in ['strong_sell', 'sell']:
                bearish_count += 1
            else:
                neutral_count += 1
        
        total = len(analysis['timeframes'])
        
        # Determine confluence
        confluence = {
            'bullish_percentage': (bullish_count / total * 100) if total > 0 else 0,
            'bearish_percentage': (bearish_count / total * 100) if total > 0 else 0,
            'neutral_percentage': (neutral_count / total * 100) if total > 0 else 0,
            'alignment': 'none'
        }
        
        if bullish_count >= total * 0.75:
            confluence['alignment'] = 'strong_bullish'
            confluence['message'] = f"Strong bullish alignment ({bullish_count}/{total} timeframes)"
        elif bullish_count >= total * 0.5:
            confluence['alignment'] = 'bullish'
            confluence['message'] = f"Bullish alignment ({bullish_count}/{total} timeframes)"
        elif bearish_count >= total * 0.75:
            confluence['alignment'] = 'strong_bearish'
            confluence['message'] = f"Strong bearish alignment ({bearish_count}/{total} timeframes)"
        elif bearish_count >= total * 0.5:
            confluence['alignment'] = 'bearish'
            confluence['message'] = f"Bearish alignment ({bearish_count}/{total} timeframes)"
        else:
            confluence['alignment'] = 'mixed'
            confluence['message'] = f"Mixed signals across timeframes"
        
        analysis['confluence'] = confluence
        return analysis