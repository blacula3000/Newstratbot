"""
Market Regime Classifier Agent - Identifies market conditions and regimes
Tags sessions as: trend, expansion, compression, reversal watch
Uses FTFC (Full Time Frame Continuity) and volatility analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats

class MarketRegime(Enum):
    STRONG_TREND_UP = "strong_trend_up"
    TREND_UP = "trend_up"
    WEAK_TREND_UP = "weak_trend_up"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    WEAK_TREND_DOWN = "weak_trend_down"
    TREND_DOWN = "trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    REVERSAL_WATCH = "reversal_watch"
    CHOPPY = "choppy"
    BREAKOUT_IMMINENT = "breakout_imminent"

@dataclass
class RegimeAnalysis:
    timestamp: datetime
    symbol: str
    timeframe: str
    current_regime: MarketRegime
    regime_strength: float  # 0-100
    volatility_state: str  # 'expanding', 'compressing', 'stable'
    ftfc_alignment: Dict[str, bool]  # Timeframe continuity
    inside_bar_count: int
    trend_quality: float  # 0-100
    reversal_probability: float  # 0-100
    key_levels: Dict[str, float]
    regime_duration: int  # Number of periods in current regime
    recommended_strategy: str
    agent_adjustments: Dict[str, any]  # Threshold adjustments for other agents

class MarketRegimeAgent:
    """
    Classifies market regimes and provides adaptive parameters for other agents
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.regime_history = {}
        self.timeframe_data = {}
        
    def _default_config(self) -> Dict:
        return {
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'lookback_periods': {
                '1m': 100,
                '5m': 60,
                '15m': 48,
                '1h': 24,
                '4h': 20,
                '1d': 20
            },
            'compression_threshold': 0.3,  # 30% of ATR
            'expansion_threshold': 1.5,  # 150% of ATR
            'trend_strength_threshold': 0.6,  # ADX > 25 equivalent
            'reversal_indicators': ['rsi', 'stochastic', 'volume'],
            'inside_bar_threshold': 3,  # Consecutive inside bars for compression
            'volatility_window': 20,
            'trend_window': 50,
            'regime_change_confirmation': 3  # Periods to confirm regime change
        }
    
    def analyze_regime(self, data: Dict[str, pd.DataFrame], symbol: str, primary_timeframe: str = '15m') -> RegimeAnalysis:
        """
        Comprehensive regime analysis across multiple timeframes
        """
        # Store timeframe data
        self.timeframe_data = data
        
        # Analyze each timeframe
        timeframe_regimes = {}
        ftfc_alignment = {}
        
        for tf in self.config['timeframes']:
            if tf in data and not data[tf].empty:
                tf_regime = self._analyze_single_timeframe(data[tf], tf)
                timeframe_regimes[tf] = tf_regime
                ftfc_alignment[tf] = tf_regime['trend_direction']
        
        # Primary timeframe analysis
        primary_data = data.get(primary_timeframe)
        if primary_data is None or primary_data.empty:
            raise ValueError(f"No data for primary timeframe {primary_timeframe}")
        
        # Core regime classification
        current_regime = self._classify_regime(primary_data, timeframe_regimes)
        
        # Volatility analysis
        volatility_state = self._analyze_volatility(primary_data)
        
        # Inside bar detection
        inside_bar_count = self._count_inside_bars(primary_data)
        
        # Trend quality assessment
        trend_quality = self._assess_trend_quality(primary_data)
        
        # Reversal probability
        reversal_prob = self._calculate_reversal_probability(primary_data, timeframe_regimes)
        
        # Key levels identification
        key_levels = self._identify_key_levels(primary_data)
        
        # Regime strength
        regime_strength = self._calculate_regime_strength(current_regime, timeframe_regimes)
        
        # Regime duration
        regime_duration = self._get_regime_duration(symbol, current_regime)
        
        # Strategy recommendation
        recommended_strategy = self._get_recommended_strategy(current_regime, volatility_state, trend_quality)
        
        # Agent adjustments
        agent_adjustments = self._calculate_agent_adjustments(
            current_regime, volatility_state, regime_strength, reversal_prob
        )
        
        analysis = RegimeAnalysis(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=primary_timeframe,
            current_regime=current_regime,
            regime_strength=regime_strength,
            volatility_state=volatility_state,
            ftfc_alignment=ftfc_alignment,
            inside_bar_count=inside_bar_count,
            trend_quality=trend_quality,
            reversal_probability=reversal_prob,
            key_levels=key_levels,
            regime_duration=regime_duration,
            recommended_strategy=recommended_strategy,
            agent_adjustments=agent_adjustments
        )
        
        # Update history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append(analysis)
        
        return analysis
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Analyze regime for a single timeframe
        """
        if len(df) < 20:
            return {'trend_direction': None, 'strength': 0}
        
        # Calculate indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else df['sma_20']
        df['atr'] = self._calculate_atr(df)
        
        # Determine trend direction
        latest = df.iloc[-1]
        if latest['close'] > latest['sma_20'] > latest['sma_50']:
            trend_direction = 'up'
        elif latest['close'] < latest['sma_20'] < latest['sma_50']:
            trend_direction = 'down'
        else:
            trend_direction = 'neutral'
        
        # Calculate trend strength (0-100)
        price_distance = abs(latest['close'] - latest['sma_20']) / latest['atr'] if latest['atr'] > 0 else 0
        trend_strength = min(100, price_distance * 20)
        
        return {
            'trend_direction': trend_direction,
            'strength': trend_strength,
            'atr': latest['atr'],
            'volatility': df['close'].pct_change().std() * 100
        }
    
    def _classify_regime(self, df: pd.DataFrame, timeframe_regimes: Dict) -> MarketRegime:
        """
        Classify the overall market regime
        """
        # Get primary metrics
        atr = self._calculate_atr(df).iloc[-1]
        recent_atr = self._calculate_atr(df).tail(20).mean()
        
        # Volatility compression/expansion
        volatility_ratio = atr / recent_atr if recent_atr > 0 else 1
        
        # Trend alignment across timeframes
        trend_votes = {'up': 0, 'down': 0, 'neutral': 0}
        total_strength = 0
        
        for tf, regime in timeframe_regimes.items():
            if regime['trend_direction']:
                trend_votes[regime['trend_direction']] += 1
                total_strength += regime['strength']
        
        avg_strength = total_strength / len(timeframe_regimes) if timeframe_regimes else 0
        
        # Inside bar check
        inside_bars = self._count_inside_bars(df)
        
        # Classification logic
        if inside_bars >= self.config['inside_bar_threshold']:
            if volatility_ratio < self.config['compression_threshold']:
                return MarketRegime.BREAKOUT_IMMINENT
            return MarketRegime.COMPRESSION
        
        if volatility_ratio > self.config['expansion_threshold']:
            return MarketRegime.EXPANSION
        
        # Trend classification
        if trend_votes['up'] > trend_votes['down']:
            if avg_strength > 70:
                return MarketRegime.STRONG_TREND_UP
            elif avg_strength > 40:
                return MarketRegime.TREND_UP
            else:
                return MarketRegime.WEAK_TREND_UP
        elif trend_votes['down'] > trend_votes['up']:
            if avg_strength > 70:
                return MarketRegime.STRONG_TREND_DOWN
            elif avg_strength > 40:
                return MarketRegime.TREND_DOWN
            else:
                return MarketRegime.WEAK_TREND_DOWN
        else:
            # Check for reversal conditions
            if self._check_reversal_conditions(df):
                return MarketRegime.REVERSAL_WATCH
            return MarketRegime.CHOPPY
    
    def _analyze_volatility(self, df: pd.DataFrame) -> str:
        """
        Analyze volatility state
        """
        if len(df) < 40:
            return 'stable'
        
        # Calculate recent vs historical volatility
        recent_vol = df['close'].pct_change().tail(20).std()
        historical_vol = df['close'].pct_change().tail(40).std()
        
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        
        if vol_ratio > 1.3:
            return 'expanding'
        elif vol_ratio < 0.7:
            return 'compressing'
        else:
            return 'stable'
    
    def _count_inside_bars(self, df: pd.DataFrame) -> int:
        """
        Count consecutive inside bars (Type 1 in STRAT)
        """
        if len(df) < 2:
            return 0
        
        count = 0
        for i in range(len(df) - 1, 0, -1):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            
            # Inside bar: high <= prev high and low >= prev low
            if curr['high'] <= prev['high'] and curr['low'] >= prev['low']:
                count += 1
            else:
                break
        
        return count
    
    def _assess_trend_quality(self, df: pd.DataFrame) -> float:
        """
        Assess trend quality (0-100)
        """
        if len(df) < 20:
            return 0
        
        # Calculate trend metrics
        closes = df['close'].values[-20:]
        
        # Linear regression for trend consistency
        x = np.arange(len(closes))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes)
        
        # R-squared indicates trend quality
        trend_quality = abs(r_value) * 100
        
        # Adjust for volatility
        volatility = df['close'].pct_change().tail(20).std()
        if volatility > 0.03:  # High volatility reduces quality
            trend_quality *= 0.8
        
        return min(100, trend_quality)
    
    def _calculate_reversal_probability(self, df: pd.DataFrame, timeframe_regimes: Dict) -> float:
        """
        Calculate probability of trend reversal
        """
        reversal_score = 0
        
        # RSI divergence
        if 'rsi' in df.columns or len(df) >= 14:
            rsi = self._calculate_rsi(df)
            if rsi > 70:
                reversal_score += 30
            elif rsi < 30:
                reversal_score += 30
        
        # Volume analysis
        recent_volume = df['volume'].tail(5).mean()
        avg_volume = df['volume'].tail(20).mean()
        if recent_volume > avg_volume * 1.5:
            reversal_score += 20
        
        # Timeframe divergence
        tf_divergence = self._check_timeframe_divergence(timeframe_regimes)
        if tf_divergence:
            reversal_score += 25
        
        # Price at key levels
        key_levels = self._identify_key_levels(df)
        current_price = df['close'].iloc[-1]
        for level_name, level_price in key_levels.items():
            if abs(current_price - level_price) / current_price < 0.005:  # Within 0.5%
                reversal_score += 15
                break
        
        return min(100, reversal_score)
    
    def _identify_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Identify key support/resistance levels
        """
        levels = {}
        
        if len(df) < 20:
            return levels
        
        # Recent high/low
        levels['recent_high'] = df['high'].tail(20).max()
        levels['recent_low'] = df['low'].tail(20).min()
        
        # Pivot points
        last_candle = df.iloc[-1]
        pivot = (last_candle['high'] + last_candle['low'] + last_candle['close']) / 3
        levels['pivot'] = pivot
        levels['r1'] = 2 * pivot - last_candle['low']
        levels['s1'] = 2 * pivot - last_candle['high']
        
        # Volume-weighted levels
        if 'volume' in df.columns:
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            levels['vwap'] = vwap
        
        return levels
    
    def _calculate_regime_strength(self, regime: MarketRegime, timeframe_regimes: Dict) -> float:
        """
        Calculate strength of current regime (0-100)
        """
        base_strength = 50
        
        # Trend regimes get strength from alignment
        if 'TREND' in regime.name:
            aligned = sum(1 for r in timeframe_regimes.values() 
                         if r['trend_direction'] == ('up' if 'UP' in regime.name else 'down'))
            base_strength = (aligned / len(timeframe_regimes)) * 100 if timeframe_regimes else 50
        
        # Compression/expansion regimes
        elif regime in [MarketRegime.COMPRESSION, MarketRegime.BREAKOUT_IMMINENT]:
            # Stronger with more inside bars
            inside_bars = self._count_inside_bars(list(self.timeframe_data.values())[0])
            base_strength = min(100, 40 + inside_bars * 15)
        
        elif regime == MarketRegime.EXPANSION:
            base_strength = 70  # Expansion is typically strong
        
        return base_strength
    
    def _get_regime_duration(self, symbol: str, current_regime: MarketRegime) -> int:
        """
        Get duration of current regime in periods
        """
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return 1
        
        duration = 1
        for analysis in reversed(self.regime_history[symbol][:-1]):
            if analysis.current_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _get_recommended_strategy(self, regime: MarketRegime, volatility: str, trend_quality: float) -> str:
        """
        Recommend trading strategy based on regime
        """
        strategies = {
            MarketRegime.STRONG_TREND_UP: "Trend following with pullback entries, trail stops aggressively",
            MarketRegime.TREND_UP: "Trend following with careful entries, standard trailing stops",
            MarketRegime.WEAK_TREND_UP: "Scalping longs on support, tight stops",
            MarketRegime.COMPRESSION: "Avoid trading or prepare for breakout with pending orders",
            MarketRegime.EXPANSION: "Momentum trading with wide stops, reduce position size",
            MarketRegime.WEAK_TREND_DOWN: "Scalping shorts on resistance, tight stops",
            MarketRegime.TREND_DOWN: "Trend following shorts, standard trailing stops",
            MarketRegime.STRONG_TREND_DOWN: "Aggressive shorting on rallies, trail stops",
            MarketRegime.REVERSAL_WATCH: "Reduce position size, tighten stops, watch for confirmation",
            MarketRegime.CHOPPY: "Range trading between support/resistance, avoid trends",
            MarketRegime.BREAKOUT_IMMINENT: "Set breakout orders above/below range, prepare for momentum"
        }
        
        base_strategy = strategies.get(regime, "Monitor only")
        
        # Adjust for volatility
        if volatility == 'expanding':
            base_strategy += " | CAUTION: Widening stops required"
        elif volatility == 'compressing':
            base_strategy += " | OPPORTUNITY: Breakout imminent"
        
        return base_strategy
    
    def _calculate_agent_adjustments(self, regime: MarketRegime, volatility: str, 
                                    strength: float, reversal_prob: float) -> Dict:
        """
        Calculate parameter adjustments for other agents
        """
        adjustments = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'entry_threshold_adjustment': 0,
            'max_positions': 3,
            'use_limit_orders': False,
            'require_confirmation': False,
            'timeframe_priority': 'current'
        }
        
        # Regime-specific adjustments
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            adjustments['position_size_multiplier'] = 1.2
            adjustments['stop_loss_multiplier'] = 1.5  # Wider stops in strong trends
            adjustments['take_profit_multiplier'] = 2.0  # Let winners run
            adjustments['max_positions'] = 5
            
        elif regime in [MarketRegime.COMPRESSION, MarketRegime.BREAKOUT_IMMINENT]:
            adjustments['position_size_multiplier'] = 0.5
            adjustments['use_limit_orders'] = True
            adjustments['require_confirmation'] = True
            adjustments['max_positions'] = 1
            
        elif regime == MarketRegime.EXPANSION:
            adjustments['stop_loss_multiplier'] = 2.0  # Much wider stops
            adjustments['position_size_multiplier'] = 0.7  # Reduce size
            
        elif regime in [MarketRegime.CHOPPY]:
            adjustments['entry_threshold_adjustment'] = 10  # More selective
            adjustments['use_limit_orders'] = True
            adjustments['timeframe_priority'] = 'lower'  # Focus on smaller timeframes
            
        elif regime == MarketRegime.REVERSAL_WATCH:
            adjustments['position_size_multiplier'] = 0.3
            adjustments['stop_loss_multiplier'] = 0.7  # Tighter stops
            adjustments['require_confirmation'] = True
            adjustments['max_positions'] = 1
        
        # Volatility adjustments
        if volatility == 'expanding':
            adjustments['stop_loss_multiplier'] *= 1.3
            adjustments['position_size_multiplier'] *= 0.8
        elif volatility == 'compressing':
            adjustments['stop_loss_multiplier'] *= 0.8
            
        # High reversal probability adjustments
        if reversal_prob > 70:
            adjustments['position_size_multiplier'] *= 0.5
            adjustments['require_confirmation'] = True
        
        return adjustments
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate RSI
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _check_timeframe_divergence(self, timeframe_regimes: Dict) -> bool:
        """
        Check if timeframes are diverging (potential reversal)
        """
        if len(timeframe_regimes) < 3:
            return False
        
        directions = [r['trend_direction'] for r in timeframe_regimes.values() if r['trend_direction']]
        
        # Check for mixed signals
        up_count = directions.count('up')
        down_count = directions.count('down')
        
        # Divergence if roughly equal up/down signals
        total = up_count + down_count
        if total > 0:
            ratio = min(up_count, down_count) / total
            return ratio > 0.3  # At least 30% disagreement
        
        return False
    
    def _check_reversal_conditions(self, df: pd.DataFrame) -> bool:
        """
        Check for reversal conditions
        """
        if len(df) < 50:
            return False
        
        # Multiple reversal indicators
        rsi = self._calculate_rsi(df)
        
        # Extreme RSI
        if rsi > 80 or rsi < 20:
            return True
        
        # Price rejection from recent high/low
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current = df['close'].iloc[-1]
        
        # Check for rejection candles
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        upper_wick = last_candle['high'] - max(last_candle['close'], last_candle['open'])
        lower_wick = min(last_candle['close'], last_candle['open']) - last_candle['low']
        
        # Shooting star or hammer
        if upper_wick > body * 2 or lower_wick > body * 2:
            if abs(current - recent_high) / current < 0.01 or abs(current - recent_low) / current < 0.01:
                return True
        
        return False
    
    def get_regime_report(self, symbol: str) -> Dict:
        """
        Get comprehensive regime report for symbol
        """
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return {"error": "No regime history for symbol"}
        
        latest = self.regime_history[symbol][-1]
        
        return {
            'current_regime': latest.current_regime.value,
            'strength': latest.regime_strength,
            'volatility': latest.volatility_state,
            'inside_bars': latest.inside_bar_count,
            'trend_quality': latest.trend_quality,
            'reversal_risk': latest.reversal_probability,
            'duration': latest.regime_duration,
            'strategy': latest.recommended_strategy,
            'key_levels': latest.key_levels,
            'ftfc_alignment': latest.ftfc_alignment,
            'trading_adjustments': latest.agent_adjustments
        }