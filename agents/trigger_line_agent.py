"""
Trigger Line Agent - Advanced STRAT 2u/2d/3 Breakout Detection
Monitors trigger line breaks, angles, and momentum across timeframes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

class TriggerLineType(Enum):
    PRIOR_HIGH = "prior_high"
    PRIOR_LOW = "prior_low"
    WEEKLY_HIGH = "weekly_high"
    WEEKLY_LOW = "weekly_low"
    MONTHLY_HIGH = "monthly_high"
    MONTHLY_LOW = "monthly_low"
    PIVOT_HIGH = "pivot_high"
    PIVOT_LOW = "pivot_low"

class BreakType(Enum):
    CLEAN_BREAK = "clean_break"
    FALSE_BREAK = "false_break"
    RETEST_BREAK = "retest_break"
    MOMENTUM_BREAK = "momentum_break"

class TriggerLineDirection(Enum):
    BULLISH_BREAK = "bullish_break"
    BEARISH_BREAK = "bearish_break"
    NO_BREAK = "no_break"

@dataclass
class TriggerLine:
    level: float
    line_type: TriggerLineType
    timestamp: datetime
    timeframe: str
    strength: float  # 1-10 strength rating
    angle: float  # Angle of approach
    touches: int  # Number of times tested
    last_test: datetime
    metadata: Dict = None

@dataclass
class TriggerBreak:
    trigger_line: TriggerLine
    break_price: float
    break_time: datetime
    break_type: BreakType
    direction: TriggerLineDirection
    momentum: float  # 1-10 momentum rating
    volume_confirmation: bool
    price_confirmation: bool
    time_confirmation: bool
    confidence: float  # Overall confidence 0-100
    follow_through: Optional[float] = None  # Price follow-through
    metadata: Dict = None

@dataclass
class TriggerLineAnalysis:
    symbol: str
    timeframe: str
    active_triggers: List[TriggerLine]
    recent_breaks: List[TriggerBreak]
    clustering_score: float  # Multiple trigger lines clustered
    momentum_score: float  # Overall momentum assessment
    directional_bias: TriggerLineDirection
    confidence_score: float
    actionable_signals: List[Dict]
    timestamp: datetime

class TriggerLineAgent:
    """
    Advanced Trigger Line Agent for STRAT methodology
    
    Monitors:
    - 2u/2d/3 candle type trigger line breaks
    - Prior highs/lows from multiple timeframes
    - Trigger line angles and momentum
    - False break vs confirmed break detection
    - Clustering of multiple trigger lines
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.active_trigger_lines = {}  # {symbol: {timeframe: [TriggerLine]}}
        self.break_history = {}  # {symbol: deque[TriggerBreak]}
        self.momentum_tracker = {}  # {symbol: momentum_data}
        
        # Performance tracking
        self.break_accuracy = deque(maxlen=100)
        self.false_break_rate = deque(maxlen=100)
        self.momentum_success = deque(maxlen=100)
        
        self.logger.info("🎯 Trigger Line Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            'min_trigger_strength': 6,
            'min_break_momentum': 5,
            'false_break_threshold_pct': 0.002,  # 0.2% false break threshold
            'min_volume_confirmation': 1.2,  # 1.2x average volume
            'clustering_distance_pct': 0.01,  # 1% price distance for clustering
            'momentum_lookback': 5,  # Candles to look back for momentum
            'min_touches_for_strength': 2,
            'max_trigger_age_hours': 168,  # 1 week max age
            'timeframes': ['5m', '15m', '1h', '4h', '1d'],
            'confidence_threshold': 70,
            'angle_sensitivity': 0.5,  # Degrees for angle calculation
            'retest_window_candles': 10
        }
    
    def analyze_trigger_lines(self, symbol: str, timeframe: str, data: pd.DataFrame) -> TriggerLineAnalysis:
        """
        Comprehensive trigger line analysis for a symbol/timeframe
        """
        try:
            # Initialize symbol tracking if needed
            if symbol not in self.active_trigger_lines:
                self.active_trigger_lines[symbol] = {}
                self.break_history[symbol] = deque(maxlen=50)
                self.momentum_tracker[symbol] = {}
            
            if timeframe not in self.active_trigger_lines[symbol]:
                self.active_trigger_lines[symbol][timeframe] = []
            
            # Step 1: Identify and update trigger lines
            trigger_lines = self._identify_trigger_lines(symbol, timeframe, data)
            
            # Step 2: Classify current candle STRAT type
            current_strat_type = self._classify_current_strat_candle(data)
            
            # Step 3: Check for trigger line breaks
            recent_breaks = self._check_trigger_breaks(symbol, timeframe, data, current_strat_type)
            
            # Step 4: Calculate clustering and momentum
            clustering_score = self._calculate_clustering_score(trigger_lines)
            momentum_score = self._calculate_momentum_score(symbol, timeframe, data)
            
            # Step 5: Determine directional bias
            directional_bias = self._determine_directional_bias(trigger_lines, recent_breaks, momentum_score)
            
            # Step 6: Generate actionable signals
            actionable_signals = self._generate_actionable_signals(
                symbol, timeframe, trigger_lines, recent_breaks, current_strat_type
            )
            
            # Step 7: Calculate overall confidence
            confidence_score = self._calculate_confidence_score(
                trigger_lines, recent_breaks, clustering_score, momentum_score
            )
            
            analysis = TriggerLineAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                active_triggers=trigger_lines,
                recent_breaks=recent_breaks,
                clustering_score=clustering_score,
                momentum_score=momentum_score,
                directional_bias=directional_bias,
                confidence_score=confidence_score,
                actionable_signals=actionable_signals,
                timestamp=datetime.now()
            )
            
            # Update tracking
            self.active_trigger_lines[symbol][timeframe] = trigger_lines
            self.break_history[symbol].extend(recent_breaks)
            
            self.logger.debug(f"🎯 Trigger analysis complete: {symbol} {timeframe} - "
                            f"{len(trigger_lines)} triggers, {len(recent_breaks)} breaks")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trigger lines for {symbol} {timeframe}: {e}")
            return self._empty_analysis(symbol, timeframe)
    
    def _identify_trigger_lines(self, symbol: str, timeframe: str, data: pd.DataFrame) -> List[TriggerLine]:
        """Identify significant trigger lines from price action"""
        trigger_lines = []
        
        try:
            if len(data) < 20:
                return trigger_lines
            
            # Get existing trigger lines and clean up old ones
            existing_triggers = self.active_trigger_lines[symbol].get(timeframe, [])
            current_time = datetime.now()
            
            # Remove expired trigger lines
            valid_triggers = [
                tl for tl in existing_triggers 
                if (current_time - tl.timestamp).total_seconds() < self.config['max_trigger_age_hours'] * 3600
            ]
            
            # Identify new trigger lines from recent price action
            
            # 1. Prior highs and lows (most recent significant levels)
            highs = data['high'].rolling(window=5, center=True).max()
            lows = data['low'].rolling(window=5, center=True).min()
            
            # Find local highs/lows
            for i in range(10, len(data) - 2):
                current_high = data['high'].iloc[i]
                current_low = data['low'].iloc[i]
                
                # Check for local high
                if (current_high == highs.iloc[i] and 
                    current_high > data['high'].iloc[i-2:i].max() and
                    current_high > data['high'].iloc[i+1:i+3].max()):
                    
                    strength = self._calculate_level_strength(data, i, current_high, 'high')
                    if strength >= self.config['min_trigger_strength']:
                        trigger_line = TriggerLine(
                            level=current_high,
                            line_type=TriggerLineType.PRIOR_HIGH,
                            timestamp=data.index[i] if hasattr(data.index[i], 'to_pydatetime') else current_time,
                            timeframe=timeframe,
                            strength=strength,
                            angle=self._calculate_approach_angle(data, i),
                            touches=1,
                            last_test=data.index[i] if hasattr(data.index[i], 'to_pydatetime') else current_time
                        )
                        trigger_lines.append(trigger_line)
                
                # Check for local low
                if (current_low == lows.iloc[i] and 
                    current_low < data['low'].iloc[i-2:i].min() and
                    current_low < data['low'].iloc[i+1:i+3].min()):
                    
                    strength = self._calculate_level_strength(data, i, current_low, 'low')
                    if strength >= self.config['min_trigger_strength']:
                        trigger_line = TriggerLine(
                            level=current_low,
                            line_type=TriggerLineType.PRIOR_LOW,
                            timestamp=data.index[i] if hasattr(data.index[i], 'to_pydatetime') else current_time,
                            timeframe=timeframe,
                            strength=strength,
                            angle=self._calculate_approach_angle(data, i),
                            touches=1,
                            last_test=data.index[i] if hasattr(data.index[i], 'to_pydatetime') else current_time
                        )
                        trigger_lines.append(trigger_line)
            
            # 2. Weekly/Monthly levels (if daily or higher timeframe)
            if timeframe in ['1d', '1w']:
                weekly_levels = self._identify_weekly_levels(data)
                monthly_levels = self._identify_monthly_levels(data)
                trigger_lines.extend(weekly_levels + monthly_levels)
            
            # 3. Merge with existing valid triggers and update touches
            merged_triggers = self._merge_trigger_lines(valid_triggers, trigger_lines)
            
            # 4. Sort by strength and limit count
            merged_triggers.sort(key=lambda x: x.strength, reverse=True)
            
            return merged_triggers[:20]  # Keep top 20 most significant
            
        except Exception as e:
            self.logger.error(f"Error identifying trigger lines: {e}")
            return []
    
    def _calculate_level_strength(self, data: pd.DataFrame, index: int, level: float, level_type: str) -> float:
        """Calculate strength of a price level based on multiple factors"""
        try:
            strength = 0
            
            # Factor 1: Volume at the level
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[index]
                level_volume = data['volume'].iloc[index]
                if level_volume > avg_volume:
                    strength += min(level_volume / avg_volume, 3) * 2
            
            # Factor 2: Number of times tested
            price_tolerance = level * 0.002  # 0.2% tolerance
            if level_type == 'high':
                tests = ((data['high'] >= (level - price_tolerance)) & 
                        (data['high'] <= (level + price_tolerance))).sum()
            else:
                tests = ((data['low'] >= (level - price_tolerance)) & 
                        (data['low'] <= (level + price_tolerance))).sum()
            
            strength += min(tests, 5)
            
            # Factor 3: Time at level (consolidation)
            consolidation_score = 0
            window = min(10, len(data) - index - 1)
            if window > 0:
                for i in range(1, window + 1):
                    if index + i < len(data):
                        if level_type == 'high':
                            if abs(data['high'].iloc[index + i] - level) / level < 0.01:
                                consolidation_score += 0.5
                        else:
                            if abs(data['low'].iloc[index + i] - level) / level < 0.01:
                                consolidation_score += 0.5
            
            strength += consolidation_score
            
            # Factor 4: Significance in trend structure
            if index > 20:
                trend_significance = self._assess_trend_significance(data, index, level, level_type)
                strength += trend_significance
            
            return min(strength, 10)  # Cap at 10
            
        except Exception:
            return 5  # Default moderate strength
    
    def _calculate_approach_angle(self, data: pd.DataFrame, index: int) -> float:
        """Calculate the angle of price approach to the level"""
        try:
            if index < 3:
                return 0
            
            # Calculate slope of price movement approaching the level
            lookback = min(3, index)
            prices = data['close'].iloc[index-lookback:index+1].values
            
            if len(prices) < 2:
                return 0
            
            # Linear regression for angle
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Convert slope to angle in degrees
            angle = np.arctan(slope) * 180 / np.pi
            
            return angle
            
        except Exception:
            return 0
    
    def _classify_current_strat_candle(self, data: pd.DataFrame) -> str:
        """Classify the most recent candle as STRAT type"""
        try:
            if len(data) < 2:
                return "1"
            
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            # STRAT classification
            if current['high'] > previous['high'] and current['low'] < previous['low']:
                return "3"  # Outside bar
            elif current['high'] > previous['high']:
                return "2u"  # Directional up
            elif current['low'] < previous['low']:
                return "2d"  # Directional down
            else:
                return "1"  # Inside bar
                
        except Exception:
            return "1"
    
    def _check_trigger_breaks(self, symbol: str, timeframe: str, data: pd.DataFrame, strat_type: str) -> List[TriggerBreak]:
        """Check for trigger line breaks in current price action"""
        breaks = []
        
        try:
            if len(data) < 2:
                return breaks
            
            current_candle = data.iloc[-1]
            previous_candle = data.iloc[-2]
            
            # Get active trigger lines
            trigger_lines = self.active_trigger_lines[symbol].get(timeframe, [])
            
            for trigger in trigger_lines:
                # Check for breaks based on STRAT type and trigger type
                break_detected = False
                break_direction = TriggerLineDirection.NO_BREAK
                break_type = BreakType.CLEAN_BREAK
                
                # Bullish breaks (2u or 3 candles breaking above resistance)
                if (strat_type in ["2u", "3"] and 
                    trigger.line_type in [TriggerLineType.PRIOR_HIGH, TriggerLineType.WEEKLY_HIGH, TriggerLineType.MONTHLY_HIGH]):
                    
                    if (previous_candle['high'] <= trigger.level and 
                        current_candle['high'] > trigger.level):
                        break_detected = True
                        break_direction = TriggerLineDirection.BULLISH_BREAK
                
                # Bearish breaks (2d or 3 candles breaking below support)
                elif (strat_type in ["2d", "3"] and 
                      trigger.line_type in [TriggerLineType.PRIOR_LOW, TriggerLineType.WEEKLY_LOW, TriggerLineType.MONTHLY_LOW]):
                    
                    if (previous_candle['low'] >= trigger.level and 
                        current_candle['low'] < trigger.level):
                        break_detected = True
                        break_direction = TriggerLineDirection.BEARISH_BREAK
                
                if break_detected:
                    # Determine break type and quality
                    break_price = current_candle['high'] if break_direction == TriggerLineDirection.BULLISH_BREAK else current_candle['low']
                    
                    # Check for false break
                    price_penetration = abs(break_price - trigger.level) / trigger.level
                    if price_penetration < self.config['false_break_threshold_pct']:
                        break_type = BreakType.FALSE_BREAK
                    
                    # Check for momentum break
                    momentum = self._calculate_break_momentum(data, len(data) - 1)
                    if momentum >= self.config['min_break_momentum']:
                        break_type = BreakType.MOMENTUM_BREAK
                    
                    # Volume confirmation
                    volume_confirmation = self._check_volume_confirmation(data)
                    
                    # Price confirmation (sustained break)
                    price_confirmation = self._check_price_confirmation(current_candle, trigger, break_direction)
                    
                    # Time confirmation (not end of session weakness)
                    time_confirmation = True  # Simplified for now
                    
                    # Calculate overall confidence
                    confidence = self._calculate_break_confidence(
                        trigger, break_type, momentum, volume_confirmation, 
                        price_confirmation, time_confirmation
                    )
                    
                    trigger_break = TriggerBreak(
                        trigger_line=trigger,
                        break_price=break_price,
                        break_time=datetime.now(),
                        break_type=break_type,
                        direction=break_direction,
                        momentum=momentum,
                        volume_confirmation=volume_confirmation,
                        price_confirmation=price_confirmation,
                        time_confirmation=time_confirmation,
                        confidence=confidence,
                        metadata={
                            'strat_type': strat_type,
                            'penetration_pct': price_penetration * 100
                        }
                    )
                    
                    breaks.append(trigger_break)
            
            return breaks
            
        except Exception as e:
            self.logger.error(f"Error checking trigger breaks: {e}")
            return []
    
    def _calculate_break_momentum(self, data: pd.DataFrame, break_index: int) -> float:
        """Calculate momentum score for a trigger break"""
        try:
            if break_index < self.config['momentum_lookback']:
                return 5  # Default moderate momentum
            
            lookback = self.config['momentum_lookback']
            recent_data = data.iloc[break_index-lookback+1:break_index+1]
            
            # Calculate price momentum
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Calculate volume momentum
            volume_ratio = 1
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[break_index]
                current_volume = data['volume'].iloc[break_index]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Combine factors
            momentum = abs(price_change) * 50 + min(volume_ratio, 3) * 2
            
            return min(momentum, 10)
            
        except Exception:
            return 5
    
    def _check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Check if volume confirms the break"""
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return True  # Assume confirmation if no volume data
            
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            
            return current_volume >= (avg_volume * self.config['min_volume_confirmation'])
            
        except Exception:
            return True
    
    def _check_price_confirmation(self, current_candle, trigger: TriggerLine, direction: TriggerLineDirection) -> bool:
        """Check if price action confirms the break"""
        try:
            if direction == TriggerLineDirection.BULLISH_BREAK:
                # For bullish breaks, closing above the trigger level is confirmation
                return current_candle['close'] > trigger.level
            else:
                # For bearish breaks, closing below the trigger level is confirmation
                return current_candle['close'] < trigger.level
                
        except Exception:
            return True
    
    def _calculate_break_confidence(self, trigger: TriggerLine, break_type: BreakType, 
                                   momentum: float, volume_conf: bool, 
                                   price_conf: bool, time_conf: bool) -> float:
        """Calculate overall confidence in the trigger break"""
        try:
            confidence = 0
            
            # Base confidence from trigger strength
            confidence += trigger.strength * 5
            
            # Break type factor
            break_type_scores = {
                BreakType.MOMENTUM_BREAK: 20,
                BreakType.CLEAN_BREAK: 15,
                BreakType.RETEST_BREAK: 10,
                BreakType.FALSE_BREAK: -10
            }
            confidence += break_type_scores.get(break_type, 0)
            
            # Momentum factor
            confidence += momentum * 2
            
            # Confirmation factors
            if volume_conf:
                confidence += 10
            if price_conf:
                confidence += 10
            if time_conf:
                confidence += 5
            
            return min(max(confidence, 0), 100)
            
        except Exception:
            return 50
    
    def _calculate_clustering_score(self, trigger_lines: List[TriggerLine]) -> float:
        """Calculate clustering score when multiple triggers are near each other"""
        try:
            if len(trigger_lines) < 2:
                return 0
            
            clustering_score = 0
            cluster_distance = self.config['clustering_distance_pct']
            
            # Group triggers by price proximity
            clusters = []
            for trigger in trigger_lines:
                added_to_cluster = False
                for cluster in clusters:
                    cluster_center = np.mean([t.level for t in cluster])
                    if abs(trigger.level - cluster_center) / cluster_center <= cluster_distance:
                        cluster.append(trigger)
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    clusters.append([trigger])
            
            # Score clusters with multiple triggers
            for cluster in clusters:
                if len(cluster) >= 2:
                    cluster_strength = sum(t.strength for t in cluster) / len(cluster)
                    clustering_score += len(cluster) * cluster_strength
            
            return min(clustering_score, 100)
            
        except Exception:
            return 0
    
    def _calculate_momentum_score(self, symbol: str, timeframe: str, data: pd.DataFrame) -> float:
        """Calculate overall momentum score"""
        try:
            if len(data) < 10:
                return 50
            
            # Price momentum
            short_ma = data['close'].rolling(window=5).mean().iloc[-1]
            long_ma = data['close'].rolling(window=10).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            price_momentum = 0
            if current_price > short_ma > long_ma:
                price_momentum = 75  # Strong bullish momentum
            elif current_price > short_ma:
                price_momentum = 60  # Moderate bullish momentum
            elif current_price < short_ma < long_ma:
                price_momentum = 25  # Strong bearish momentum
            elif current_price < short_ma:
                price_momentum = 40  # Moderate bearish momentum
            else:
                price_momentum = 50  # Neutral
            
            # Volume momentum
            volume_momentum = 50
            if 'volume' in data.columns:
                recent_volume = data['volume'].rolling(window=3).mean().iloc[-1]
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                if recent_volume > avg_volume:
                    volume_momentum = 75
                else:
                    volume_momentum = 25
            
            # Combined momentum
            momentum_score = (price_momentum * 0.7) + (volume_momentum * 0.3)
            
            return momentum_score
            
        except Exception:
            return 50
    
    def _determine_directional_bias(self, trigger_lines: List[TriggerLine], 
                                  recent_breaks: List[TriggerBreak], 
                                  momentum_score: float) -> TriggerLineDirection:
        """Determine overall directional bias"""
        try:
            # Weight recent breaks heavily
            bullish_weight = 0
            bearish_weight = 0
            
            for break_event in recent_breaks[-5:]:  # Last 5 breaks
                if break_event.direction == TriggerLineDirection.BULLISH_BREAK:
                    bullish_weight += break_event.confidence / 100
                elif break_event.direction == TriggerLineDirection.BEARISH_BREAK:
                    bearish_weight += break_event.confidence / 100
            
            # Add momentum bias
            if momentum_score > 60:
                bullish_weight += 1
            elif momentum_score < 40:
                bearish_weight += 1
            
            # Determine bias
            if bullish_weight > bearish_weight + 0.5:
                return TriggerLineDirection.BULLISH_BREAK
            elif bearish_weight > bullish_weight + 0.5:
                return TriggerLineDirection.BEARISH_BREAK
            else:
                return TriggerLineDirection.NO_BREAK
                
        except Exception:
            return TriggerLineDirection.NO_BREAK
    
    def _generate_actionable_signals(self, symbol: str, timeframe: str, 
                                   trigger_lines: List[TriggerLine],
                                   recent_breaks: List[TriggerBreak],
                                   strat_type: str) -> List[Dict]:
        """Generate actionable trading signals"""
        signals = []
        
        try:
            # Only generate signals for high-confidence breaks
            high_confidence_breaks = [
                b for b in recent_breaks 
                if b.confidence >= self.config['confidence_threshold']
            ]
            
            for break_event in high_confidence_breaks:
                if break_event.break_type != BreakType.FALSE_BREAK:
                    
                    signal = {
                        'signal_id': f"trigger_{symbol}_{timeframe}_{int(break_event.break_time.timestamp())}",
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'signal_type': 'TRIGGER_BREAK',
                        'direction': 'LONG' if break_event.direction == TriggerLineDirection.BULLISH_BREAK else 'SHORT',
                        'entry_price': break_event.break_price,
                        'trigger_level': break_event.trigger_line.level,
                        'confidence_score': break_event.confidence,
                        'momentum_score': break_event.momentum,
                        'strat_type': strat_type,
                        'break_type': break_event.break_type.value,
                        'volume_confirmed': break_event.volume_confirmation,
                        'price_confirmed': break_event.price_confirmation,
                        'trigger_strength': break_event.trigger_line.strength,
                        'actionable': True,
                        'timestamp': break_event.break_time,
                        'metadata': {
                            'trigger_line_type': break_event.trigger_line.line_type.value,
                            'trigger_touches': break_event.trigger_line.touches,
                            'break_momentum': break_event.momentum,
                            'clustering_nearby': len([tl for tl in trigger_lines 
                                                    if abs(tl.level - break_event.trigger_line.level) / break_event.trigger_line.level < 0.02])
                        }
                    }
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating actionable signals: {e}")
            return []
    
    def _calculate_confidence_score(self, trigger_lines: List[TriggerLine], 
                                  recent_breaks: List[TriggerBreak],
                                  clustering_score: float, 
                                  momentum_score: float) -> float:
        """Calculate overall analysis confidence score"""
        try:
            confidence = 0
            
            # Base confidence from trigger line quality
            if trigger_lines:
                avg_trigger_strength = np.mean([t.strength for t in trigger_lines])
                confidence += avg_trigger_strength * 5
            
            # Recent break quality
            if recent_breaks:
                avg_break_confidence = np.mean([b.confidence for b in recent_breaks])
                confidence += avg_break_confidence * 0.3
            
            # Clustering bonus
            confidence += clustering_score * 0.1
            
            # Momentum factor
            momentum_factor = abs(momentum_score - 50) / 50  # Distance from neutral
            confidence += momentum_factor * 20
            
            return min(confidence, 100)
            
        except Exception:
            return 50
    
    def _identify_weekly_levels(self, data: pd.DataFrame) -> List[TriggerLine]:
        """Identify weekly high/low levels"""
        # Simplified implementation - would need actual weekly aggregation
        try:
            weekly_high = data['high'].max()
            weekly_low = data['low'].min()
            current_time = datetime.now()
            
            return [
                TriggerLine(
                    level=weekly_high,
                    line_type=TriggerLineType.WEEKLY_HIGH,
                    timestamp=current_time,
                    timeframe='1w',
                    strength=8,
                    angle=0,
                    touches=1,
                    last_test=current_time
                ),
                TriggerLine(
                    level=weekly_low,
                    line_type=TriggerLineType.WEEKLY_LOW,
                    timestamp=current_time,
                    timeframe='1w',
                    strength=8,
                    angle=0,
                    touches=1,
                    last_test=current_time
                )
            ]
        except Exception:
            return []
    
    def _identify_monthly_levels(self, data: pd.DataFrame) -> List[TriggerLine]:
        """Identify monthly high/low levels"""
        # Simplified implementation
        try:
            monthly_high = data['high'].max()
            monthly_low = data['low'].min()
            current_time = datetime.now()
            
            return [
                TriggerLine(
                    level=monthly_high,
                    line_type=TriggerLineType.MONTHLY_HIGH,
                    timestamp=current_time,
                    timeframe='1M',
                    strength=9,
                    angle=0,
                    touches=1,
                    last_test=current_time
                )
            ]
        except Exception:
            return []
    
    def _merge_trigger_lines(self, existing: List[TriggerLine], new: List[TriggerLine]) -> List[TriggerLine]:
        """Merge existing and new trigger lines, updating touches"""
        merged = existing.copy()
        merge_tolerance = 0.005  # 0.5% price tolerance
        
        for new_trigger in new:
            merged_with_existing = False
            
            for existing_trigger in merged:
                price_diff = abs(new_trigger.level - existing_trigger.level) / existing_trigger.level
                
                if (price_diff < merge_tolerance and 
                    new_trigger.line_type == existing_trigger.line_type):
                    
                    # Update existing trigger
                    existing_trigger.touches += 1
                    existing_trigger.last_test = new_trigger.timestamp
                    existing_trigger.strength = max(existing_trigger.strength, new_trigger.strength)
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(new_trigger)
        
        return merged
    
    def _assess_trend_significance(self, data: pd.DataFrame, index: int, level: float, level_type: str) -> float:
        """Assess the significance of a level in the context of the trend"""
        try:
            # Look at trend context around the level
            window = min(20, index)
            trend_data = data.iloc[max(0, index-window):index+1]
            
            # Calculate if level represents a significant trend point
            if level_type == 'high':
                # Check if this is a significant high in uptrend
                if trend_data['close'].iloc[-1] > trend_data['close'].iloc[0]:
                    return 2  # Higher significance in uptrend
            else:
                # Check if this is a significant low in downtrend
                if trend_data['close'].iloc[-1] < trend_data['close'].iloc[0]:
                    return 2  # Higher significance in downtrend
            
            return 1  # Default significance
            
        except Exception:
            return 1
    
    def _empty_analysis(self, symbol: str, timeframe: str) -> TriggerLineAnalysis:
        """Return empty analysis in case of errors"""
        return TriggerLineAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            active_triggers=[],
            recent_breaks=[],
            clustering_score=0,
            momentum_score=50,
            directional_bias=TriggerLineDirection.NO_BREAK,
            confidence_score=0,
            actionable_signals=[],
            timestamp=datetime.now()
        )
    
    def get_agent_status(self) -> Dict:
        """Get current agent status and performance metrics"""
        try:
            total_symbols = len(self.active_trigger_lines)
            total_triggers = sum(
                len(timeframes.get(tf, [])) 
                for timeframes in self.active_trigger_lines.values()
                for tf in timeframes
            )
            total_breaks = sum(len(breaks) for breaks in self.break_history.values())
            
            avg_accuracy = np.mean(self.break_accuracy) if self.break_accuracy else 0
            false_break_pct = np.mean(self.false_break_rate) if self.false_break_rate else 0
            
            return {
                'agent_name': 'Trigger Line Agent',
                'status': 'active',
                'symbols_tracked': total_symbols,
                'active_triggers': total_triggers,
                'total_breaks_detected': total_breaks,
                'break_accuracy': f"{avg_accuracy:.1%}",
                'false_break_rate': f"{false_break_pct:.1%}",
                'config_confidence_threshold': self.config['confidence_threshold'],
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Trigger Line Agent', 'status': 'error'}
    
    def reset_symbol_tracking(self, symbol: str):
        """Reset tracking for a specific symbol"""
        if symbol in self.active_trigger_lines:
            del self.active_trigger_lines[symbol]
        if symbol in self.break_history:
            del self.break_history[symbol]
        if symbol in self.momentum_tracker:
            del self.momentum_tracker[symbol]
        
        self.logger.info(f"🔄 Reset trigger line tracking for {symbol}")
    
    def update_break_outcome(self, signal_id: str, success: bool, pnl: float):
        """Update break outcome for performance tracking"""
        try:
            self.break_accuracy.append(1.0 if success else 0.0)
            
            # Track false breaks
            if not success and pnl < 0:
                self.false_break_rate.append(1.0)
            else:
                self.false_break_rate.append(0.0)
            
            # Track momentum success
            if success and pnl > 0:
                self.momentum_success.append(1.0)
            else:
                self.momentum_success.append(0.0)
                
        except Exception as e:
            self.logger.error(f"Error updating break outcome: {e}")