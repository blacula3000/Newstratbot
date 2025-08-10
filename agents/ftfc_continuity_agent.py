"""
FTFC Continuity Agent - Full Timeframe Continuity Analysis
Advanced timeframe alignment validation for STRAT methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

class TimeframeContinuity(Enum):
    PERFECT_ALIGNMENT = "perfect_alignment"  # All timeframes aligned
    STRONG_ALIGNMENT = "strong_alignment"    # 80%+ timeframes aligned
    MODERATE_ALIGNMENT = "moderate_alignment"  # 60-80% aligned
    WEAK_ALIGNMENT = "weak_alignment"        # 40-60% aligned
    NO_ALIGNMENT = "no_alignment"            # <40% aligned

class ContinuityDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class AlignmentStrength(Enum):
    VERY_STRONG = "very_strong"  # 90-100%
    STRONG = "strong"            # 75-90%
    MODERATE = "moderate"        # 50-75%
    WEAK = "weak"               # 25-50%
    VERY_WEAK = "very_weak"     # 0-25%

@dataclass
class TimeframeState:
    timeframe: str
    current_strat_type: str  # 1, 2u, 2d, 3
    trend_direction: ContinuityDirection
    momentum_score: float  # 1-10
    structure_score: float  # 1-10
    recent_break: Optional[str]  # Type of recent break
    daily_open_relationship: str  # above/below/at daily open
    weekly_open_relationship: str  # above/below/at weekly open
    monthly_open_relationship: str  # above/below/at monthly open
    last_update: datetime

@dataclass
class ContinuityAnalysis:
    symbol: str
    analysis_time: datetime
    timeframe_states: Dict[str, TimeframeState]
    overall_continuity: TimeframeContinuity
    primary_direction: ContinuityDirection
    alignment_strength: AlignmentStrength
    alignment_score: float  # 0-100
    actionable_signals: List[Dict]
    confluence_levels: List[float]  # Price levels with multiple timeframe confluence
    risk_factors: List[str]
    opportunity_score: float  # 0-100 overall opportunity rating

@dataclass
class FTFCSignal:
    signal_id: str
    symbol: str
    signal_type: str  # 'FTFC_ALIGNMENT', 'CONFLUENCE_BREAK', etc.
    direction: ContinuityDirection
    confidence: float  # 0-100
    timeframes_aligned: List[str]
    alignment_score: float
    entry_level: float
    stop_loss: float
    target_levels: List[float]
    time_horizon: str  # Expected time to target
    risk_reward_ratio: float
    metadata: Dict

class FTFCContinuityAgent:
    """
    Full Timeframe Continuity Agent for STRAT methodology
    
    Analyzes:
    - Multi-timeframe STRAT pattern alignment
    - Daily/Weekly/Monthly open relationships
    - Timeframe momentum confluence
    - Market structure continuity
    - Actionable alignment opportunities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.timeframe_states = {}  # {symbol: {timeframe: TimeframeState}}
        self.continuity_history = {}  # {symbol: deque[ContinuityAnalysis]}
        self.open_levels_cache = {}  # {symbol: {daily_open, weekly_open, monthly_open}}
        
        # Performance tracking
        self.alignment_accuracy = deque(maxlen=100)
        self.signal_success_rate = deque(maxlen=100)
        self.confluence_hit_rate = deque(maxlen=100)
        
        self.logger.info("🔗 FTFC Continuity Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            'timeframes': ['5m', '15m', '1h', '4h', '1d'],
            'primary_timeframe': '15m',
            'higher_timeframe': '1h',
            'lower_timeframe': '5m',
            'min_alignment_score': 70,
            'strong_alignment_threshold': 80,
            'perfect_alignment_threshold': 95,
            'momentum_weight': 0.3,
            'structure_weight': 0.4,
            'open_relationship_weight': 0.3,
            'confluence_tolerance_pct': 0.005,  # 0.5% for confluence levels
            'min_timeframes_for_signal': 3,
            'max_risk_reward_ratio': 4.0,
            'min_risk_reward_ratio': 1.5,
            'opportunity_decay_hours': 4,  # How long opportunities remain valid
            'recency_weight': 0.6,  # Weight for recent vs historical alignment
        }
    
    def analyze_ftfc_continuity(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> ContinuityAnalysis:
        """
        Comprehensive FTFC continuity analysis across all timeframes
        """
        try:
            # Initialize symbol tracking
            if symbol not in self.timeframe_states:
                self.timeframe_states[symbol] = {}
                self.continuity_history[symbol] = deque(maxlen=50)
                self.open_levels_cache[symbol] = {}
            
            # Step 1: Analyze each timeframe
            timeframe_states = {}
            for tf, data in timeframe_data.items():
                if tf in self.config['timeframes'] and not data.empty:
                    state = self._analyze_timeframe_state(symbol, tf, data)
                    timeframe_states[tf] = state
                    self.timeframe_states[symbol][tf] = state
            
            # Step 2: Calculate overall continuity metrics
            overall_continuity = self._calculate_overall_continuity(timeframe_states)
            primary_direction = self._determine_primary_direction(timeframe_states)
            alignment_strength = self._calculate_alignment_strength(timeframe_states)
            alignment_score = self._calculate_alignment_score(timeframe_states)
            
            # Step 3: Find confluence levels
            confluence_levels = self._identify_confluence_levels(symbol, timeframe_data)
            
            # Step 4: Generate actionable signals
            actionable_signals = self._generate_ftfc_signals(
                symbol, timeframe_states, overall_continuity, 
                primary_direction, confluence_levels
            )
            
            # Step 5: Assess risk factors
            risk_factors = self._assess_risk_factors(timeframe_states, confluence_levels)
            
            # Step 6: Calculate opportunity score
            opportunity_score = self._calculate_opportunity_score(
                timeframe_states, alignment_score, len(actionable_signals)
            )
            
            analysis = ContinuityAnalysis(
                symbol=symbol,
                analysis_time=datetime.now(),
                timeframe_states=timeframe_states,
                overall_continuity=overall_continuity,
                primary_direction=primary_direction,
                alignment_strength=alignment_strength,
                alignment_score=alignment_score,
                actionable_signals=actionable_signals,
                confluence_levels=confluence_levels,
                risk_factors=risk_factors,
                opportunity_score=opportunity_score
            )
            
            # Store analysis
            self.continuity_history[symbol].append(analysis)
            
            self.logger.debug(f"🔗 FTFC analysis complete: {symbol} - "
                            f"Alignment: {overall_continuity.value}, Score: {alignment_score:.1f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing FTFC continuity for {symbol}: {e}")
            return self._empty_analysis(symbol)
    
    def _analyze_timeframe_state(self, symbol: str, timeframe: str, data: pd.DataFrame) -> TimeframeState:
        """Analyze the state of a specific timeframe"""
        try:
            if len(data) < 5:
                return self._default_timeframe_state(timeframe)
            
            # Get current STRAT type
            current_strat_type = self._classify_strat_type(data)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(data)
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(data)
            
            # Calculate structure score (how well-defined the structure is)
            structure_score = self._calculate_structure_score(data)
            
            # Check for recent breaks
            recent_break = self._check_recent_breaks(data)
            
            # Get open relationships (requires open level data)
            daily_open_rel, weekly_open_rel, monthly_open_rel = self._get_open_relationships(
                symbol, data.iloc[-1]['close']
            )
            
            return TimeframeState(
                timeframe=timeframe,
                current_strat_type=current_strat_type,
                trend_direction=trend_direction,
                momentum_score=momentum_score,
                structure_score=structure_score,
                recent_break=recent_break,
                daily_open_relationship=daily_open_rel,
                weekly_open_relationship=weekly_open_rel,
                monthly_open_relationship=monthly_open_rel,
                last_update=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe state {timeframe}: {e}")
            return self._default_timeframe_state(timeframe)
    
    def _classify_strat_type(self, data: pd.DataFrame) -> str:
        """Classify current STRAT candle type"""
        try:
            if len(data) < 2:
                return "1"
            
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
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
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> ContinuityDirection:
        """Determine trend direction for the timeframe"""
        try:
            if len(data) < 20:
                return ContinuityDirection.NEUTRAL
            
            # Use multiple EMAs for trend determination
            ema_8 = data['close'].ewm(span=8).mean()
            ema_21 = data['close'].ewm(span=21).mean()
            ema_50 = data['close'].ewm(span=50) if len(data) >= 50 else ema_21
            
            current_price = data['close'].iloc[-1]
            current_ema_8 = ema_8.iloc[-1]
            current_ema_21 = ema_21.iloc[-1]
            current_ema_50 = ema_50.iloc[-1] if len(data) >= 50 else current_ema_21
            
            # Strong bullish: Price > EMA8 > EMA21 > EMA50
            if current_price > current_ema_8 > current_ema_21 > current_ema_50:
                return ContinuityDirection.BULLISH
            
            # Strong bearish: Price < EMA8 < EMA21 < EMA50
            elif current_price < current_ema_8 < current_ema_21 < current_ema_50:
                return ContinuityDirection.BEARISH
            
            # Moderate signals
            elif current_price > current_ema_8 and current_ema_8 > current_ema_21:
                return ContinuityDirection.BULLISH
            elif current_price < current_ema_8 and current_ema_8 < current_ema_21:
                return ContinuityDirection.BEARISH
            else:
                return ContinuityDirection.NEUTRAL
                
        except Exception:
            return ContinuityDirection.NEUTRAL
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score 1-10"""
        try:
            if len(data) < 10:
                return 5.0
            
            # Price momentum (rate of change)
            roc_5 = (data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6]
            roc_10 = (data['close'].iloc[-1] - data['close'].iloc[-11]) / data['close'].iloc[-11]
            
            # Volume momentum
            volume_momentum = 1.0
            if 'volume' in data.columns:
                avg_vol = data['volume'].rolling(window=20).mean().iloc[-1]
                recent_vol = data['volume'].rolling(window=3).mean().iloc[-1]
                volume_momentum = min(recent_vol / avg_vol, 3.0) if avg_vol > 0 else 1.0
            
            # Combine momentum factors
            price_momentum = (abs(roc_5) * 100 + abs(roc_10) * 50) / 2
            momentum_score = price_momentum * 20 + volume_momentum * 2
            
            return min(max(momentum_score, 1.0), 10.0)
            
        except Exception:
            return 5.0
    
    def _calculate_structure_score(self, data: pd.DataFrame) -> float:
        """Calculate how well-defined the market structure is"""
        try:
            if len(data) < 20:
                return 5.0
            
            structure_score = 0
            
            # Higher highs and higher lows in uptrend
            highs = data['high'].rolling(window=5, center=True).max()
            lows = data['low'].rolling(window=5, center=True).min()
            
            # Check for consistent structure
            recent_highs = highs[-10:].dropna()
            recent_lows = lows[-10:].dropna()
            
            if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                # Uptrend structure
                if recent_highs.is_monotonic_increasing and recent_lows.is_monotonic_increasing:
                    structure_score = 8.5
                # Downtrend structure  
                elif recent_highs.is_monotonic_decreasing and recent_lows.is_monotonic_decreasing:
                    structure_score = 8.5
                # Mixed but trending
                elif recent_highs.iloc[-1] > recent_highs.iloc[0] and recent_lows.iloc[-1] > recent_lows.iloc[0]:
                    structure_score = 6.5
                elif recent_highs.iloc[-1] < recent_highs.iloc[0] and recent_lows.iloc[-1] < recent_lows.iloc[0]:
                    structure_score = 6.5
                else:
                    structure_score = 3.5  # Choppy/sideways
            else:
                structure_score = 5.0
            
            return structure_score
            
        except Exception:
            return 5.0
    
    def _check_recent_breaks(self, data: pd.DataFrame) -> Optional[str]:
        """Check for recent significant breaks"""
        try:
            if len(data) < 10:
                return None
            
            # Check last few candles for significant breaks
            recent_data = data.tail(5)
            
            # Look for breakouts from consolidation
            consolidation_range = data.tail(20)
            high_resistance = consolidation_range['high'].quantile(0.95)
            low_support = consolidation_range['low'].quantile(0.05)
            
            current_high = recent_data['high'].max()
            current_low = recent_data['low'].min()
            
            if current_high > high_resistance:
                return "resistance_break"
            elif current_low < low_support:
                return "support_break"
            
            # Check for STRAT pattern breaks
            if len(data) >= 3:
                last_3_candles = data.tail(3)
                if any(self._classify_strat_type(last_3_candles.iloc[i:i+2]) in ['2u', '3'] for i in range(2)):
                    return "strat_2u_break"
                elif any(self._classify_strat_type(last_3_candles.iloc[i:i+2]) in ['2d', '3'] for i in range(2)):
                    return "strat_2d_break"
            
            return None
            
        except Exception:
            return None
    
    def _get_open_relationships(self, symbol: str, current_price: float) -> Tuple[str, str, str]:
        """Get relationship to daily, weekly, monthly opens"""
        try:
            # This would integrate with actual open level data
            # For now, using cached/estimated values
            open_levels = self.open_levels_cache.get(symbol, {})
            
            daily_open = open_levels.get('daily_open', current_price * 0.995)
            weekly_open = open_levels.get('weekly_open', current_price * 0.99)
            monthly_open = open_levels.get('monthly_open', current_price * 0.985)
            
            tolerance = current_price * 0.001  # 0.1% tolerance
            
            # Daily relationship
            if current_price > daily_open + tolerance:
                daily_rel = "above"
            elif current_price < daily_open - tolerance:
                daily_rel = "below"
            else:
                daily_rel = "at"
            
            # Weekly relationship
            if current_price > weekly_open + tolerance:
                weekly_rel = "above"
            elif current_price < weekly_open - tolerance:
                weekly_rel = "below"
            else:
                weekly_rel = "at"
            
            # Monthly relationship
            if current_price > monthly_open + tolerance:
                monthly_rel = "above"
            elif current_price < monthly_open - tolerance:
                monthly_rel = "below"
            else:
                monthly_rel = "at"
            
            return daily_rel, weekly_rel, monthly_rel
            
        except Exception:
            return "unknown", "unknown", "unknown"
    
    def _calculate_overall_continuity(self, timeframe_states: Dict[str, TimeframeState]) -> TimeframeContinuity:
        """Calculate overall timeframe continuity"""
        try:
            if not timeframe_states:
                return TimeframeContinuity.NO_ALIGNMENT
            
            # Count aligned timeframes (same direction)
            directions = [state.trend_direction for state in timeframe_states.values()]
            
            if not directions:
                return TimeframeContinuity.NO_ALIGNMENT
            
            # Count bullish vs bearish vs neutral
            bullish_count = directions.count(ContinuityDirection.BULLISH)
            bearish_count = directions.count(ContinuityDirection.BEARISH)
            total_count = len(directions)
            
            # Calculate alignment percentage
            max_aligned = max(bullish_count, bearish_count)
            alignment_pct = (max_aligned / total_count) * 100
            
            if alignment_pct >= 95:
                return TimeframeContinuity.PERFECT_ALIGNMENT
            elif alignment_pct >= 80:
                return TimeframeContinuity.STRONG_ALIGNMENT
            elif alignment_pct >= 60:
                return TimeframeContinuity.MODERATE_ALIGNMENT
            elif alignment_pct >= 40:
                return TimeframeContinuity.WEAK_ALIGNMENT
            else:
                return TimeframeContinuity.NO_ALIGNMENT
                
        except Exception:
            return TimeframeContinuity.NO_ALIGNMENT
    
    def _determine_primary_direction(self, timeframe_states: Dict[str, TimeframeState]) -> ContinuityDirection:
        """Determine the primary direction across timeframes"""
        try:
            if not timeframe_states:
                return ContinuityDirection.NEUTRAL
            
            # Weight higher timeframes more heavily
            timeframe_weights = {
                '1d': 5.0,
                '4h': 4.0,
                '1h': 3.0,
                '15m': 2.0,
                '5m': 1.0
            }
            
            weighted_bullish = 0
            weighted_bearish = 0
            total_weight = 0
            
            for tf, state in timeframe_states.items():
                weight = timeframe_weights.get(tf, 1.0)
                total_weight += weight
                
                if state.trend_direction == ContinuityDirection.BULLISH:
                    weighted_bullish += weight
                elif state.trend_direction == ContinuityDirection.BEARISH:
                    weighted_bearish += weight
            
            if total_weight == 0:
                return ContinuityDirection.NEUTRAL
            
            bullish_pct = weighted_bullish / total_weight
            bearish_pct = weighted_bearish / total_weight
            
            if bullish_pct > 0.6:
                return ContinuityDirection.BULLISH
            elif bearish_pct > 0.6:
                return ContinuityDirection.BEARISH
            else:
                return ContinuityDirection.NEUTRAL
                
        except Exception:
            return ContinuityDirection.NEUTRAL
    
    def _calculate_alignment_strength(self, timeframe_states: Dict[str, TimeframeState]) -> AlignmentStrength:
        """Calculate alignment strength classification"""
        try:
            alignment_score = self._calculate_alignment_score(timeframe_states)
            
            if alignment_score >= 90:
                return AlignmentStrength.VERY_STRONG
            elif alignment_score >= 75:
                return AlignmentStrength.STRONG
            elif alignment_score >= 50:
                return AlignmentStrength.MODERATE
            elif alignment_score >= 25:
                return AlignmentStrength.WEAK
            else:
                return AlignmentStrength.VERY_WEAK
                
        except Exception:
            return AlignmentStrength.WEAK
    
    def _calculate_alignment_score(self, timeframe_states: Dict[str, TimeframeState]) -> float:
        """Calculate numerical alignment score 0-100"""
        try:
            if not timeframe_states:
                return 0
            
            score = 0
            total_possible = 0
            
            for tf, state in timeframe_states.items():
                # Weight factors
                momentum_weight = self.config['momentum_weight']
                structure_weight = self.config['structure_weight']
                open_rel_weight = self.config['open_relationship_weight']
                
                # Momentum contribution
                momentum_contrib = (state.momentum_score / 10) * momentum_weight * 100
                
                # Structure contribution
                structure_contrib = (state.structure_score / 10) * structure_weight * 100
                
                # Open relationship contribution (bonus for alignment)
                open_alignment_bonus = 0
                if state.daily_open_relationship == "above" and state.trend_direction == ContinuityDirection.BULLISH:
                    open_alignment_bonus = 20
                elif state.daily_open_relationship == "below" and state.trend_direction == ContinuityDirection.BEARISH:
                    open_alignment_bonus = 20
                
                open_contrib = open_alignment_bonus * open_rel_weight
                
                tf_score = momentum_contrib + structure_contrib + open_contrib
                score += tf_score
                total_possible += 100
            
            if total_possible == 0:
                return 0
            
            final_score = (score / total_possible) * 100
            return min(max(final_score, 0), 100)
            
        except Exception:
            return 0
    
    def _identify_confluence_levels(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> List[float]:
        """Identify price levels with multiple timeframe confluence"""
        confluence_levels = []
        
        try:
            all_levels = []
            
            # Collect significant levels from each timeframe
            for tf, data in timeframe_data.items():
                if data.empty:
                    continue
                
                # Get recent highs/lows
                recent_highs = data['high'].rolling(window=5, center=True).max().dropna()
                recent_lows = data['low'].rolling(window=5, center=True).min().dropna()
                
                # Add significant levels
                if len(recent_highs) > 0:
                    all_levels.extend(recent_highs.tail(3).tolist())
                if len(recent_lows) > 0:
                    all_levels.extend(recent_lows.tail(3).tolist())
                
                # Add EMAs as potential confluence
                if len(data) >= 21:
                    ema_21 = data['close'].ewm(span=21).mean().iloc[-1]
                    all_levels.append(ema_21)
            
            if not all_levels:
                return confluence_levels
            
            # Find clusters of levels (confluence)
            tolerance = self.config['confluence_tolerance_pct']
            all_levels.sort()
            
            i = 0
            while i < len(all_levels):
                cluster = [all_levels[i]]
                j = i + 1
                
                # Find all levels within tolerance
                while j < len(all_levels):
                    if abs(all_levels[j] - all_levels[i]) / all_levels[i] <= tolerance:
                        cluster.append(all_levels[j])
                        j += 1
                    else:
                        break
                
                # If cluster has multiple levels, add average as confluence level
                if len(cluster) >= 2:
                    confluence_level = np.mean(cluster)
                    confluence_levels.append(confluence_level)
                
                i = j
            
            return confluence_levels[:10]  # Limit to top 10
            
        except Exception as e:
            self.logger.error(f"Error identifying confluence levels: {e}")
            return []
    
    def _generate_ftfc_signals(self, symbol: str, timeframe_states: Dict[str, TimeframeState],
                              overall_continuity: TimeframeContinuity, 
                              primary_direction: ContinuityDirection,
                              confluence_levels: List[float]) -> List[Dict]:
        """Generate actionable FTFC signals"""
        signals = []
        
        try:
            # Only generate signals for good alignment
            if overall_continuity in [TimeframeContinuity.NO_ALIGNMENT, TimeframeContinuity.WEAK_ALIGNMENT]:
                return signals
            
            if primary_direction == ContinuityDirection.NEUTRAL:
                return signals
            
            # Get current price from primary timeframe
            primary_tf = self.config['primary_timeframe']
            if primary_tf not in timeframe_states:
                return signals
            
            current_price = 0  # Would get from data
            
            # Check for aligned timeframes with recent breakouts
            aligned_timeframes = []
            breakout_timeframes = []
            
            for tf, state in timeframe_states.items():
                if state.trend_direction == primary_direction:
                    aligned_timeframes.append(tf)
                    
                    if state.recent_break and (
                        (primary_direction == ContinuityDirection.BULLISH and "resistance" in state.recent_break) or
                        (primary_direction == ContinuityDirection.BEARISH and "support" in state.recent_break)
                    ):
                        breakout_timeframes.append(tf)
            
            # Generate alignment signal
            if len(aligned_timeframes) >= self.config['min_timeframes_for_signal']:
                alignment_score = self._calculate_alignment_score(timeframe_states)
                
                if alignment_score >= self.config['min_alignment_score']:
                    signal = self._create_alignment_signal(
                        symbol, primary_direction, aligned_timeframes, 
                        alignment_score, current_price
                    )
                    signals.append(signal)
            
            # Generate confluence break signals
            if confluence_levels and breakout_timeframes:
                for level in confluence_levels:
                    confluence_signal = self._create_confluence_signal(
                        symbol, primary_direction, level, breakout_timeframes,
                        alignment_score, current_price
                    )
                    if confluence_signal:
                        signals.append(confluence_signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating FTFC signals: {e}")
            return []
    
    def _create_alignment_signal(self, symbol: str, direction: ContinuityDirection, 
                               aligned_tfs: List[str], alignment_score: float, 
                               current_price: float) -> Dict:
        """Create timeframe alignment signal"""
        try:
            signal_id = f"ftfc_alignment_{symbol}_{int(datetime.now().timestamp())}"
            
            # Calculate risk/reward based on alignment strength
            if direction == ContinuityDirection.BULLISH:
                entry_level = current_price * 1.002  # 0.2% above current
                stop_loss = current_price * 0.985   # 1.5% stop
                targets = [current_price * 1.02, current_price * 1.04, current_price * 1.06]
            else:
                entry_level = current_price * 0.998  # 0.2% below current
                stop_loss = current_price * 1.015   # 1.5% stop
                targets = [current_price * 0.98, current_price * 0.96, current_price * 0.94]
            
            risk_reward = abs(targets[0] - entry_level) / abs(entry_level - stop_loss)
            
            return {
                'signal_id': signal_id,
                'signal_type': 'FTFC_ALIGNMENT',
                'symbol': symbol,
                'direction': direction.value,
                'confidence': alignment_score,
                'timeframes_aligned': aligned_tfs,
                'alignment_score': alignment_score,
                'entry_level': entry_level,
                'stop_loss': stop_loss,
                'target_levels': targets,
                'risk_reward_ratio': risk_reward,
                'time_horizon': self._estimate_time_horizon(aligned_tfs),
                'actionable': True,
                'timestamp': datetime.now(),
                'metadata': {
                    'continuity_type': 'timeframe_alignment',
                    'aligned_count': len(aligned_tfs),
                    'signal_strength': 'high' if alignment_score > 85 else 'medium'
                }
            }
            
        except Exception:
            return {}
    
    def _create_confluence_signal(self, symbol: str, direction: ContinuityDirection, 
                                 confluence_level: float, breakout_tfs: List[str],
                                 alignment_score: float, current_price: float) -> Optional[Dict]:
        """Create confluence level signal"""
        try:
            # Check if confluence level is actionable
            distance_to_level = abs(current_price - confluence_level) / current_price
            
            if distance_to_level > 0.02:  # More than 2% away
                return None
            
            signal_id = f"ftfc_confluence_{symbol}_{int(confluence_level)}_{int(datetime.now().timestamp())}"
            
            if direction == ContinuityDirection.BULLISH:
                entry_level = confluence_level * 1.001
                stop_loss = confluence_level * 0.99
                targets = [confluence_level * 1.015, confluence_level * 1.025, confluence_level * 1.04]
            else:
                entry_level = confluence_level * 0.999
                stop_loss = confluence_level * 1.01
                targets = [confluence_level * 0.985, confluence_level * 0.975, confluence_level * 0.96]
            
            risk_reward = abs(targets[0] - entry_level) / abs(entry_level - stop_loss)
            
            # Only create signal if risk/reward is acceptable
            if (risk_reward >= self.config['min_risk_reward_ratio'] and 
                risk_reward <= self.config['max_risk_reward_ratio']):
                
                return {
                    'signal_id': signal_id,
                    'signal_type': 'FTFC_CONFLUENCE',
                    'symbol': symbol,
                    'direction': direction.value,
                    'confidence': min(alignment_score * 1.1, 100),  # Bonus for confluence
                    'timeframes_aligned': breakout_tfs,
                    'confluence_level': confluence_level,
                    'entry_level': entry_level,
                    'stop_loss': stop_loss,
                    'target_levels': targets,
                    'risk_reward_ratio': risk_reward,
                    'time_horizon': self._estimate_time_horizon(breakout_tfs),
                    'actionable': True,
                    'timestamp': datetime.now(),
                    'metadata': {
                        'continuity_type': 'confluence_break',
                        'distance_to_level_pct': distance_to_level * 100,
                        'breakout_timeframes': len(breakout_tfs)
                    }
                }
            
            return None
            
        except Exception:
            return None
    
    def _estimate_time_horizon(self, timeframes: List[str]) -> str:
        """Estimate time horizon based on aligned timeframes"""
        try:
            # Get the highest timeframe
            tf_order = {'5m': 1, '15m': 2, '1h': 3, '4h': 4, '1d': 5}
            max_tf_value = max(tf_order.get(tf, 1) for tf in timeframes)
            
            if max_tf_value >= 5:  # Daily
                return "days"
            elif max_tf_value >= 4:  # 4H
                return "hours"
            elif max_tf_value >= 3:  # 1H
                return "hours"
            else:
                return "minutes"
                
        except Exception:
            return "hours"
    
    def _assess_risk_factors(self, timeframe_states: Dict[str, TimeframeState], 
                           confluence_levels: List[float]) -> List[str]:
        """Assess risk factors for the current setup"""
        risk_factors = []
        
        try:
            # Check for conflicting timeframes
            directions = [state.trend_direction for state in timeframe_states.values()]
            if ContinuityDirection.BULLISH in directions and ContinuityDirection.BEARISH in directions:
                risk_factors.append("conflicting_timeframe_signals")
            
            # Check for low momentum timeframes
            low_momentum_count = sum(1 for state in timeframe_states.values() if state.momentum_score < 4)
            if low_momentum_count > len(timeframe_states) / 2:
                risk_factors.append("low_momentum_environment")
            
            # Check for poor structure
            poor_structure_count = sum(1 for state in timeframe_states.values() if state.structure_score < 4)
            if poor_structure_count > len(timeframe_states) / 2:
                risk_factors.append("poor_market_structure")
            
            # Check for conflicting open relationships
            open_conflicts = 0
            for state in timeframe_states.values():
                if (state.trend_direction == ContinuityDirection.BULLISH and 
                    state.daily_open_relationship == "below"):
                    open_conflicts += 1
                elif (state.trend_direction == ContinuityDirection.BEARISH and 
                      state.daily_open_relationship == "above"):
                    open_conflicts += 1
            
            if open_conflicts > 0:
                risk_factors.append("conflicting_open_relationships")
            
            # Check confluence level clustering (too many levels = confusion)
            if len(confluence_levels) > 5:
                risk_factors.append("excessive_confluence_levels")
            
            return risk_factors
            
        except Exception:
            return ["analysis_error"]
    
    def _calculate_opportunity_score(self, timeframe_states: Dict[str, TimeframeState], 
                                   alignment_score: float, signal_count: int) -> float:
        """Calculate overall opportunity score"""
        try:
            opportunity = 0
            
            # Base score from alignment
            opportunity += alignment_score * 0.4
            
            # Bonus for multiple actionable signals
            opportunity += min(signal_count * 15, 30)
            
            # Momentum factor
            avg_momentum = np.mean([state.momentum_score for state in timeframe_states.values()])
            opportunity += (avg_momentum / 10) * 20
            
            # Structure factor
            avg_structure = np.mean([state.structure_score for state in timeframe_states.values()])
            opportunity += (avg_structure / 10) * 10
            
            return min(opportunity, 100)
            
        except Exception:
            return 50
    
    def _default_timeframe_state(self, timeframe: str) -> TimeframeState:
        """Return default timeframe state"""
        return TimeframeState(
            timeframe=timeframe,
            current_strat_type="1",
            trend_direction=ContinuityDirection.NEUTRAL,
            momentum_score=5.0,
            structure_score=5.0,
            recent_break=None,
            daily_open_relationship="unknown",
            weekly_open_relationship="unknown",
            monthly_open_relationship="unknown",
            last_update=datetime.now()
        )
    
    def _empty_analysis(self, symbol: str) -> ContinuityAnalysis:
        """Return empty analysis for error cases"""
        return ContinuityAnalysis(
            symbol=symbol,
            analysis_time=datetime.now(),
            timeframe_states={},
            overall_continuity=TimeframeContinuity.NO_ALIGNMENT,
            primary_direction=ContinuityDirection.NEUTRAL,
            alignment_strength=AlignmentStrength.VERY_WEAK,
            alignment_score=0,
            actionable_signals=[],
            confluence_levels=[],
            risk_factors=["insufficient_data"],
            opportunity_score=0
        )
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            total_symbols = len(self.timeframe_states)
            total_analyses = sum(len(history) for history in self.continuity_history.values())
            
            avg_accuracy = np.mean(self.alignment_accuracy) if self.alignment_accuracy else 0
            signal_success = np.mean(self.signal_success_rate) if self.signal_success_rate else 0
            
            return {
                'agent_name': 'FTFC Continuity Agent',
                'status': 'active',
                'symbols_tracked': total_symbols,
                'total_analyses': total_analyses,
                'alignment_accuracy': f"{avg_accuracy:.1%}",
                'signal_success_rate': f"{signal_success:.1%}",
                'min_alignment_threshold': self.config['min_alignment_score'],
                'timeframes_monitored': self.config['timeframes'],
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'FTFC Continuity Agent', 'status': 'error'}
    
    def update_signal_outcome(self, signal_id: str, success: bool, pnl: float):
        """Update signal outcome for performance tracking"""
        try:
            self.signal_success_rate.append(1.0 if success else 0.0)
            
            # Update alignment accuracy if it was an alignment signal
            if "alignment" in signal_id.lower():
                self.alignment_accuracy.append(1.0 if success else 0.0)
            
            # Update confluence hit rate
            if "confluence" in signal_id.lower():
                self.confluence_hit_rate.append(1.0 if success else 0.0)
                
        except Exception as e:
            self.logger.error(f"Error updating signal outcome: {e}")
    
    def update_open_levels(self, symbol: str, daily_open: float, 
                          weekly_open: float, monthly_open: float):
        """Update cached open levels for a symbol"""
        if symbol not in self.open_levels_cache:
            self.open_levels_cache[symbol] = {}
        
        self.open_levels_cache[symbol].update({
            'daily_open': daily_open,
            'weekly_open': weekly_open,
            'monthly_open': monthly_open,
            'last_update': datetime.now()
        })
        
        self.logger.debug(f"🔗 Updated open levels for {symbol}")
    
    def get_current_continuity_status(self, symbol: str) -> Optional[Dict]:
        """Get current continuity status summary for a symbol"""
        try:
            if (symbol not in self.continuity_history or 
                not self.continuity_history[symbol]):
                return None
            
            latest_analysis = self.continuity_history[symbol][-1]
            
            return {
                'symbol': symbol,
                'overall_continuity': latest_analysis.overall_continuity.value,
                'primary_direction': latest_analysis.primary_direction.value,
                'alignment_score': latest_analysis.alignment_score,
                'opportunity_score': latest_analysis.opportunity_score,
                'active_signals': len(latest_analysis.actionable_signals),
                'confluence_levels': len(latest_analysis.confluence_levels),
                'risk_factors': latest_analysis.risk_factors,
                'last_analysis': latest_analysis.analysis_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting continuity status for {symbol}: {e}")
            return None