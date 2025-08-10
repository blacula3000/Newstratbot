"""
Reversal Setup Agent - Advanced Exhaustion Pattern Detection
Identifies reversal opportunities through STRAT exhaustion patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

class ReversalType(Enum):
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    NO_REVERSAL = "no_reversal"

class ExhaustionPattern(Enum):
    TRIPLE_TAP = "triple_tap"  # Three tests of support/resistance
    FAILED_BREAKOUT = "failed_breakout"  # False break and reversal
    EXHAUSTION_GAP = "exhaustion_gap"  # Gap exhaustion
    CLIMAX_REVERSAL = "climax_reversal"  # Volume climax reversal
    DIVERGENCE_REVERSAL = "divergence_reversal"  # Price/momentum divergence
    TRAP_REVERSAL = "trap_reversal"  # Bull/bear trap pattern
    SQUEEZE_REVERSAL = "squeeze_reversal"  # Volatility squeeze breakout

class ReversalStrength(Enum):
    VERY_STRONG = "very_strong"  # 90-100% confidence
    STRONG = "strong"  # 75-90% confidence
    MODERATE = "moderate"  # 60-75% confidence
    WEAK = "weak"  # 40-60% confidence
    VERY_WEAK = "very_weak"  # Below 40% confidence

@dataclass
class ExhaustionSignal:
    pattern_type: ExhaustionPattern
    reversal_type: ReversalType
    strength: ReversalStrength
    confidence: float  # 0-100
    entry_zone: Tuple[float, float]  # Entry price range
    stop_loss: float
    targets: List[float]
    volume_confirmation: bool
    structure_confirmation: bool
    divergence_present: bool
    time_window: str  # Expected reversal timeframe
    risk_reward: float
    metadata: Dict

@dataclass
class ReversalSetup:
    symbol: str
    timeframe: str
    setup_type: ReversalType
    patterns_detected: List[ExhaustionSignal]
    combined_confidence: float
    optimal_entry: float
    stop_loss: float
    profit_targets: List[float]
    risk_score: float  # 1-10 risk assessment
    opportunity_score: float  # 1-10 opportunity rating
    actionable: bool
    timestamp: datetime

class ReversalSetupAgent:
    """
    Advanced Reversal Setup Agent for STRAT methodology
    
    Detects:
    - Triple tap exhaustion patterns
    - Failed breakout reversals
    - Volume climax reversals
    - Momentum divergences
    - Bull/bear trap setups
    - Volatility squeeze reversals
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.reversal_history = {}  # {symbol: deque[ReversalSetup]}
        self.pattern_cache = {}  # {symbol: {timeframe: recent_patterns}}
        self.support_resistance_levels = {}  # {symbol: {support: [], resistance: []}}
        
        # Performance tracking
        self.reversal_accuracy = deque(maxlen=100)
        self.pattern_success_rate = defaultdict(lambda: deque(maxlen=50))
        self.false_signal_rate = deque(maxlen=100)
        
        self.logger.info("🔄 Reversal Setup Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            'min_confidence': 65,
            'min_touches_for_level': 3,
            'level_tolerance_pct': 0.003,  # 0.3% tolerance for level tests
            'divergence_lookback': 14,
            'volume_spike_threshold': 2.0,  # 2x average volume
            'exhaustion_gap_threshold_pct': 0.01,  # 1% gap
            'trap_reversal_threshold_pct': 0.005,  # 0.5% trap threshold
            'squeeze_periods': 20,  # Bollinger Band periods
            'squeeze_threshold': 0.5,  # BB width threshold
            'max_pattern_age_hours': 24,
            'risk_reward_min': 2.0,
            'timeframes': ['5m', '15m', '1h', '4h'],
            'momentum_indicators': ['rsi', 'macd', 'stochastic'],
            'structure_lookback': 50
        }
    
    def analyze_reversal_setup(self, symbol: str, timeframe: str, data: pd.DataFrame) -> ReversalSetup:
        """
        Comprehensive reversal setup analysis
        """
        try:
            # Initialize tracking
            if symbol not in self.reversal_history:
                self.reversal_history[symbol] = deque(maxlen=50)
                self.pattern_cache[symbol] = {}
                self.support_resistance_levels[symbol] = {'support': [], 'resistance': []}
            
            if timeframe not in self.pattern_cache[symbol]:
                self.pattern_cache[symbol][timeframe] = deque(maxlen=20)
            
            # Step 1: Update support/resistance levels
            self._update_support_resistance(symbol, data)
            
            # Step 2: Detect exhaustion patterns
            patterns = self._detect_exhaustion_patterns(symbol, timeframe, data)
            
            # Step 3: Filter and validate patterns
            valid_patterns = self._validate_patterns(patterns, data)
            
            # Step 4: Determine reversal type and strength
            reversal_type = self._determine_reversal_type(valid_patterns)
            combined_confidence = self._calculate_combined_confidence(valid_patterns)
            
            # Step 5: Calculate optimal entry and risk parameters
            optimal_entry, stop_loss, targets = self._calculate_trade_parameters(
                valid_patterns, reversal_type, data
            )
            
            # Step 6: Assess risk and opportunity
            risk_score = self._assess_risk_score(valid_patterns, data)
            opportunity_score = self._assess_opportunity_score(
                valid_patterns, combined_confidence, risk_score
            )
            
            # Step 7: Determine if setup is actionable
            actionable = (
                combined_confidence >= self.config['min_confidence'] and
                opportunity_score >= 6 and
                risk_score <= 7 and
                len(valid_patterns) > 0
            )
            
            setup = ReversalSetup(
                symbol=symbol,
                timeframe=timeframe,
                setup_type=reversal_type,
                patterns_detected=valid_patterns,
                combined_confidence=combined_confidence,
                optimal_entry=optimal_entry,
                stop_loss=stop_loss,
                profit_targets=targets,
                risk_score=risk_score,
                opportunity_score=opportunity_score,
                actionable=actionable,
                timestamp=datetime.now()
            )
            
            # Store setup
            self.reversal_history[symbol].append(setup)
            
            self.logger.debug(f"🔄 Reversal analysis complete: {symbol} {timeframe} - "
                            f"Type: {reversal_type.value}, Confidence: {combined_confidence:.1f}")
            
            return setup
            
        except Exception as e:
            self.logger.error(f"Error analyzing reversal setup for {symbol} {timeframe}: {e}")
            return self._empty_setup(symbol, timeframe)
    
    def _update_support_resistance(self, symbol: str, data: pd.DataFrame):
        """Update support and resistance levels"""
        try:
            if len(data) < 20:
                return
            
            # Find local highs and lows
            highs = data['high'].rolling(window=5, center=True).max()
            lows = data['low'].rolling(window=5, center=True).min()
            
            support_levels = []
            resistance_levels = []
            
            for i in range(10, len(data) - 5):
                # Local high (resistance)
                if data['high'].iloc[i] == highs.iloc[i]:
                    if self._is_significant_level(data, i, data['high'].iloc[i], 'resistance'):
                        resistance_levels.append(data['high'].iloc[i])
                
                # Local low (support)
                if data['low'].iloc[i] == lows.iloc[i]:
                    if self._is_significant_level(data, i, data['low'].iloc[i], 'support'):
                        support_levels.append(data['low'].iloc[i])
            
            # Cluster and filter levels
            self.support_resistance_levels[symbol]['support'] = self._cluster_levels(support_levels)
            self.support_resistance_levels[symbol]['resistance'] = self._cluster_levels(resistance_levels)
            
        except Exception as e:
            self.logger.error(f"Error updating support/resistance: {e}")
    
    def _is_significant_level(self, data: pd.DataFrame, index: int, level: float, level_type: str) -> bool:
        """Check if a level is significant based on touches and reactions"""
        try:
            tolerance = level * self.config['level_tolerance_pct']
            touches = 0
            
            # Count touches before and after
            for i in range(max(0, index - 20), min(len(data), index + 20)):
                if level_type == 'resistance':
                    if abs(data['high'].iloc[i] - level) <= tolerance:
                        touches += 1
                else:
                    if abs(data['low'].iloc[i] - level) <= tolerance:
                        touches += 1
            
            return touches >= self.config['min_touches_for_level']
            
        except Exception:
            return False
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster nearby levels into single significant levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] <= self.config['level_tolerance_pct']:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered[:10]  # Keep top 10 levels
    
    def _detect_exhaustion_patterns(self, symbol: str, timeframe: str, data: pd.DataFrame) -> List[ExhaustionSignal]:
        """Detect various exhaustion patterns"""
        patterns = []
        
        try:
            if len(data) < 50:
                return patterns
            
            # Pattern 1: Triple Tap
            triple_tap = self._detect_triple_tap(symbol, data)
            if triple_tap:
                patterns.append(triple_tap)
            
            # Pattern 2: Failed Breakout
            failed_breakout = self._detect_failed_breakout(symbol, data)
            if failed_breakout:
                patterns.append(failed_breakout)
            
            # Pattern 3: Exhaustion Gap
            exhaustion_gap = self._detect_exhaustion_gap(data)
            if exhaustion_gap:
                patterns.append(exhaustion_gap)
            
            # Pattern 4: Climax Reversal
            climax_reversal = self._detect_climax_reversal(data)
            if climax_reversal:
                patterns.append(climax_reversal)
            
            # Pattern 5: Divergence Reversal
            divergence_reversal = self._detect_divergence_reversal(data)
            if divergence_reversal:
                patterns.append(divergence_reversal)
            
            # Pattern 6: Trap Reversal
            trap_reversal = self._detect_trap_reversal(symbol, data)
            if trap_reversal:
                patterns.append(trap_reversal)
            
            # Pattern 7: Squeeze Reversal
            squeeze_reversal = self._detect_squeeze_reversal(data)
            if squeeze_reversal:
                patterns.append(squeeze_reversal)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting exhaustion patterns: {e}")
            return []
    
    def _detect_triple_tap(self, symbol: str, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect triple tap exhaustion pattern"""
        try:
            support_levels = self.support_resistance_levels[symbol]['support']
            resistance_levels = self.support_resistance_levels[symbol]['resistance']
            
            current_price = data['close'].iloc[-1]
            
            # Check for triple tap at support
            for support in support_levels:
                taps = self._count_level_taps(data, support, 'support')
                if taps >= 3:
                    # Recent tap within last 5 candles
                    recent_low = data['low'].iloc[-5:].min()
                    if abs(recent_low - support) / support <= self.config['level_tolerance_pct']:
                        return self._create_triple_tap_signal(
                            support, 'support', current_price, data
                        )
            
            # Check for triple tap at resistance
            for resistance in resistance_levels:
                taps = self._count_level_taps(data, resistance, 'resistance')
                if taps >= 3:
                    recent_high = data['high'].iloc[-5:].max()
                    if abs(recent_high - resistance) / resistance <= self.config['level_tolerance_pct']:
                        return self._create_triple_tap_signal(
                            resistance, 'resistance', current_price, data
                        )
            
            return None
            
        except Exception:
            return None
    
    def _count_level_taps(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """Count number of taps at a level"""
        tolerance = level * self.config['level_tolerance_pct']
        taps = 0
        
        for i in range(len(data) - 30, len(data)):
            if i < 0:
                continue
            
            if level_type == 'support':
                if abs(data['low'].iloc[i] - level) <= tolerance:
                    taps += 1
            else:
                if abs(data['high'].iloc[i] - level) <= tolerance:
                    taps += 1
        
        return taps
    
    def _create_triple_tap_signal(self, level: float, level_type: str, 
                                 current_price: float, data: pd.DataFrame) -> ExhaustionSignal:
        """Create triple tap exhaustion signal"""
        if level_type == 'support':
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (level * 0.998, level * 1.002)
            stop_loss = level * 0.99
            targets = [level * 1.01, level * 1.02, level * 1.03]
        else:
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (level * 0.998, level * 1.002)
            stop_loss = level * 1.01
            targets = [level * 0.99, level * 0.98, level * 0.97]
        
        volume_confirmation = self._check_volume_confirmation(data)
        
        confidence = 75  # Base confidence for triple tap
        if volume_confirmation:
            confidence += 10
        
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.TRIPLE_TAP,
            reversal_type=reversal_type,
            strength=ReversalStrength.STRONG if confidence >= 75 else ReversalStrength.MODERATE,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=volume_confirmation,
            structure_confirmation=True,
            divergence_present=False,
            time_window="1-4 hours",
            risk_reward=risk_reward,
            metadata={'level': level, 'level_type': level_type}
        )
    
    def _detect_failed_breakout(self, symbol: str, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect failed breakout reversal pattern"""
        try:
            if len(data) < 10:
                return None
            
            recent_data = data.iloc[-10:]
            
            # Check for recent breakout that failed
            resistance_levels = self.support_resistance_levels[symbol]['resistance']
            support_levels = self.support_resistance_levels[symbol]['support']
            
            # Failed resistance breakout (bearish reversal)
            for resistance in resistance_levels:
                # Check if price broke above then fell back
                max_high = recent_data['high'].max()
                current_close = recent_data['close'].iloc[-1]
                
                if (max_high > resistance * 1.003 and  # Broke above
                    current_close < resistance * 0.997):  # Fell back below
                    
                    return self._create_failed_breakout_signal(
                        resistance, 'resistance', current_close, data
                    )
            
            # Failed support breakdown (bullish reversal)
            for support in support_levels:
                min_low = recent_data['low'].min()
                current_close = recent_data['close'].iloc[-1]
                
                if (min_low < support * 0.997 and  # Broke below
                    current_close > support * 1.003):  # Recovered above
                    
                    return self._create_failed_breakout_signal(
                        support, 'support', current_close, data
                    )
            
            return None
            
        except Exception:
            return None
    
    def _create_failed_breakout_signal(self, level: float, breakout_type: str, 
                                      current_price: float, data: pd.DataFrame) -> ExhaustionSignal:
        """Create failed breakout signal"""
        if breakout_type == 'resistance':
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (level * 0.995, level * 1.001)
            stop_loss = level * 1.015
            targets = [level * 0.985, level * 0.975, level * 0.965]
        else:
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (level * 0.999, level * 1.005)
            stop_loss = level * 0.985
            targets = [level * 1.015, level * 1.025, level * 1.035]
        
        volume_confirmation = self._check_volume_confirmation(data)
        divergence = self._check_divergence(data, reversal_type)
        
        confidence = 70
        if volume_confirmation:
            confidence += 10
        if divergence:
            confidence += 10
        
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.FAILED_BREAKOUT,
            reversal_type=reversal_type,
            strength=ReversalStrength.STRONG if confidence >= 75 else ReversalStrength.MODERATE,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=volume_confirmation,
            structure_confirmation=True,
            divergence_present=divergence,
            time_window="30min-2hours",
            risk_reward=risk_reward,
            metadata={'breakout_level': level, 'breakout_type': breakout_type}
        )
    
    def _detect_exhaustion_gap(self, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect exhaustion gap pattern"""
        try:
            if len(data) < 5:
                return None
            
            # Check for gap in recent candles
            for i in range(len(data) - 5, len(data) - 1):
                gap_size = abs(data['open'].iloc[i+1] - data['close'].iloc[i]) / data['close'].iloc[i]
                
                if gap_size >= self.config['exhaustion_gap_threshold_pct']:
                    # Check if gap is being filled
                    gap_direction = 'up' if data['open'].iloc[i+1] > data['close'].iloc[i] else 'down'
                    current_price = data['close'].iloc[-1]
                    
                    if gap_direction == 'up' and current_price < data['open'].iloc[i+1]:
                        # Gap up being filled - bearish
                        return self._create_gap_exhaustion_signal('bearish', data['open'].iloc[i+1], data)
                    elif gap_direction == 'down' and current_price > data['open'].iloc[i+1]:
                        # Gap down being filled - bullish
                        return self._create_gap_exhaustion_signal('bullish', data['open'].iloc[i+1], data)
            
            return None
            
        except Exception:
            return None
    
    def _create_gap_exhaustion_signal(self, direction: str, gap_level: float, 
                                     data: pd.DataFrame) -> ExhaustionSignal:
        """Create exhaustion gap signal"""
        current_price = data['close'].iloc[-1]
        
        if direction == 'bullish':
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (current_price * 0.999, current_price * 1.001)
            stop_loss = gap_level * 0.99
            targets = [current_price * 1.01, current_price * 1.02, current_price * 1.03]
        else:
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (current_price * 0.999, current_price * 1.001)
            stop_loss = gap_level * 1.01
            targets = [current_price * 0.99, current_price * 0.98, current_price * 0.97]
        
        confidence = 65  # Base confidence for gap exhaustion
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.EXHAUSTION_GAP,
            reversal_type=reversal_type,
            strength=ReversalStrength.MODERATE,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=False,
            structure_confirmation=False,
            divergence_present=False,
            time_window="15min-1hour",
            risk_reward=risk_reward,
            metadata={'gap_level': gap_level}
        )
    
    def _detect_climax_reversal(self, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect volume climax reversal"""
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return None
            
            avg_volume = data['volume'].rolling(window=20).mean()
            recent_volume = data['volume'].iloc[-5:]
            
            # Check for volume spike
            max_recent_volume = recent_volume.max()
            avg_recent = avg_volume.iloc[-1]
            
            if max_recent_volume > avg_recent * self.config['volume_spike_threshold']:
                # Determine direction of climax
                climax_index = recent_volume.idxmax()
                climax_candle = data.loc[climax_index]
                
                if climax_candle['close'] < climax_candle['open']:
                    # Selling climax - potential bullish reversal
                    return self._create_climax_signal('bullish', climax_candle, data)
                else:
                    # Buying climax - potential bearish reversal
                    return self._create_climax_signal('bearish', climax_candle, data)
            
            return None
            
        except Exception:
            return None
    
    def _create_climax_signal(self, direction: str, climax_candle: pd.Series, 
                             data: pd.DataFrame) -> ExhaustionSignal:
        """Create volume climax signal"""
        current_price = data['close'].iloc[-1]
        
        if direction == 'bullish':
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (climax_candle['low'] * 0.998, climax_candle['low'] * 1.002)
            stop_loss = climax_candle['low'] * 0.99
            targets = [current_price * 1.01, current_price * 1.02, current_price * 1.035]
        else:
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (climax_candle['high'] * 0.998, climax_candle['high'] * 1.002)
            stop_loss = climax_candle['high'] * 1.01
            targets = [current_price * 0.99, current_price * 0.98, current_price * 0.965]
        
        confidence = 72  # Base confidence for climax
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.CLIMAX_REVERSAL,
            reversal_type=reversal_type,
            strength=ReversalStrength.STRONG,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=True,
            structure_confirmation=False,
            divergence_present=False,
            time_window="30min-4hours",
            risk_reward=risk_reward,
            metadata={'climax_high': climax_candle['high'], 'climax_low': climax_candle['low']}
        )
    
    def _detect_divergence_reversal(self, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect price/momentum divergence"""
        try:
            if len(data) < self.config['divergence_lookback'] + 5:
                return None
            
            # Calculate RSI
            rsi = self._calculate_rsi(data['close'], 14)
            if rsi is None or len(rsi) < 10:
                return None
            
            # Look for divergences
            price_highs = data['high'].rolling(window=5, center=True).max()
            price_lows = data['low'].rolling(window=5, center=True).min()
            
            # Bullish divergence: lower lows in price, higher lows in RSI
            recent_lows_idx = []
            for i in range(len(data) - 20, len(data) - 2):
                if data['low'].iloc[i] == price_lows.iloc[i]:
                    recent_lows_idx.append(i)
            
            if len(recent_lows_idx) >= 2:
                if (data['low'].iloc[recent_lows_idx[-1]] < data['low'].iloc[recent_lows_idx[-2]] and
                    rsi.iloc[recent_lows_idx[-1]] > rsi.iloc[recent_lows_idx[-2]]):
                    return self._create_divergence_signal('bullish', data)
            
            # Bearish divergence: higher highs in price, lower highs in RSI
            recent_highs_idx = []
            for i in range(len(data) - 20, len(data) - 2):
                if data['high'].iloc[i] == price_highs.iloc[i]:
                    recent_highs_idx.append(i)
            
            if len(recent_highs_idx) >= 2:
                if (data['high'].iloc[recent_highs_idx[-1]] > data['high'].iloc[recent_highs_idx[-2]] and
                    rsi.iloc[recent_highs_idx[-1]] < rsi.iloc[recent_highs_idx[-2]]):
                    return self._create_divergence_signal('bearish', data)
            
            return None
            
        except Exception:
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[pd.Series]:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return None
    
    def _create_divergence_signal(self, direction: str, data: pd.DataFrame) -> ExhaustionSignal:
        """Create divergence reversal signal"""
        current_price = data['close'].iloc[-1]
        
        if direction == 'bullish':
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (current_price * 0.998, current_price * 1.002)
            stop_loss = data['low'].iloc[-10:].min() * 0.99
            targets = [current_price * 1.015, current_price * 1.025, current_price * 1.04]
        else:
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (current_price * 0.998, current_price * 1.002)
            stop_loss = data['high'].iloc[-10:].max() * 1.01
            targets = [current_price * 0.985, current_price * 0.975, current_price * 0.96]
        
        confidence = 78  # High confidence for divergence
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.DIVERGENCE_REVERSAL,
            reversal_type=reversal_type,
            strength=ReversalStrength.STRONG,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=False,
            structure_confirmation=True,
            divergence_present=True,
            time_window="1-6hours",
            risk_reward=risk_reward,
            metadata={'divergence_type': direction}
        )
    
    def _detect_trap_reversal(self, symbol: str, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect bull/bear trap reversal"""
        try:
            if len(data) < 10:
                return None
            
            support_levels = self.support_resistance_levels[symbol]['support']
            resistance_levels = self.support_resistance_levels[symbol]['resistance']
            
            # Bear trap: false breakdown below support
            for support in support_levels:
                recent_low = data['low'].iloc[-5:].min()
                current_close = data['close'].iloc[-1]
                
                if (recent_low < support * (1 - self.config['trap_reversal_threshold_pct']) and
                    current_close > support):
                    return self._create_trap_signal('bull_trap', support, data)
            
            # Bull trap: false breakout above resistance
            for resistance in resistance_levels:
                recent_high = data['high'].iloc[-5:].max()
                current_close = data['close'].iloc[-1]
                
                if (recent_high > resistance * (1 + self.config['trap_reversal_threshold_pct']) and
                    current_close < resistance):
                    return self._create_trap_signal('bear_trap', resistance, data)
            
            return None
            
        except Exception:
            return None
    
    def _create_trap_signal(self, trap_type: str, level: float, 
                           data: pd.DataFrame) -> ExhaustionSignal:
        """Create trap reversal signal"""
        current_price = data['close'].iloc[-1]
        
        if trap_type == 'bull_trap':
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (level * 1.001, level * 1.003)
            stop_loss = data['low'].iloc[-5:].min() * 0.995
            targets = [level * 1.015, level * 1.025, level * 1.035]
        else:
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (level * 0.997, level * 0.999)
            stop_loss = data['high'].iloc[-5:].max() * 1.005
            targets = [level * 0.985, level * 0.975, level * 0.965]
        
        confidence = 73  # Good confidence for trap patterns
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.TRAP_REVERSAL,
            reversal_type=reversal_type,
            strength=ReversalStrength.STRONG,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=self._check_volume_confirmation(data),
            structure_confirmation=True,
            divergence_present=False,
            time_window="15min-2hours",
            risk_reward=risk_reward,
            metadata={'trap_type': trap_type, 'trap_level': level}
        )
    
    def _detect_squeeze_reversal(self, data: pd.DataFrame) -> Optional[ExhaustionSignal]:
        """Detect volatility squeeze reversal"""
        try:
            if len(data) < self.config['squeeze_periods'] + 5:
                return None
            
            # Calculate Bollinger Bands
            sma = data['close'].rolling(window=self.config['squeeze_periods']).mean()
            std = data['close'].rolling(window=self.config['squeeze_periods']).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            band_width = (upper_band - lower_band) / sma
            
            # Check for squeeze (narrow bands)
            recent_width = band_width.iloc[-10:]
            if recent_width.iloc[-1] < self.config['squeeze_threshold'] * recent_width.mean():
                # Squeeze detected, check for breakout
                current_close = data['close'].iloc[-1]
                
                if current_close > upper_band.iloc[-1]:
                    return self._create_squeeze_signal('bullish', upper_band.iloc[-1], data)
                elif current_close < lower_band.iloc[-1]:
                    return self._create_squeeze_signal('bearish', lower_band.iloc[-1], data)
            
            return None
            
        except Exception:
            return None
    
    def _create_squeeze_signal(self, direction: str, band_level: float, 
                              data: pd.DataFrame) -> ExhaustionSignal:
        """Create squeeze breakout signal"""
        current_price = data['close'].iloc[-1]
        
        if direction == 'bullish':
            reversal_type = ReversalType.BULLISH_REVERSAL
            entry_zone = (current_price * 0.999, current_price * 1.001)
            stop_loss = band_level * 0.99
            targets = [current_price * 1.02, current_price * 1.03, current_price * 1.045]
        else:
            reversal_type = ReversalType.BEARISH_REVERSAL
            entry_zone = (current_price * 0.999, current_price * 1.001)
            stop_loss = band_level * 1.01
            targets = [current_price * 0.98, current_price * 0.97, current_price * 0.955]
        
        confidence = 70  # Base confidence for squeeze
        risk_reward = abs(targets[0] - entry_zone[0]) / abs(entry_zone[0] - stop_loss)
        
        return ExhaustionSignal(
            pattern_type=ExhaustionPattern.SQUEEZE_REVERSAL,
            reversal_type=reversal_type,
            strength=ReversalStrength.MODERATE,
            confidence=confidence,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            volume_confirmation=self._check_volume_confirmation(data),
            structure_confirmation=False,
            divergence_present=False,
            time_window="30min-4hours",
            risk_reward=risk_reward,
            metadata={'band_level': band_level, 'direction': direction}
        )
    
    def _check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Check if volume confirms the pattern"""
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return False
            
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            recent_volume = data['volume'].iloc[-3:].mean()
            
            return recent_volume > avg_volume * 1.2
            
        except Exception:
            return False
    
    def _check_divergence(self, data: pd.DataFrame, reversal_type: ReversalType) -> bool:
        """Quick divergence check"""
        try:
            rsi = self._calculate_rsi(data['close'], 14)
            if rsi is None or len(rsi) < 5:
                return False
            
            if reversal_type == ReversalType.BULLISH_REVERSAL:
                # Check for bullish divergence
                return rsi.iloc[-1] > rsi.iloc[-5] and data['close'].iloc[-1] < data['close'].iloc[-5]
            else:
                # Check for bearish divergence
                return rsi.iloc[-1] < rsi.iloc[-5] and data['close'].iloc[-1] > data['close'].iloc[-5]
                
        except Exception:
            return False
    
    def _validate_patterns(self, patterns: List[ExhaustionSignal], data: pd.DataFrame) -> List[ExhaustionSignal]:
        """Validate and filter patterns"""
        valid_patterns = []
        
        for pattern in patterns:
            # Check risk/reward ratio
            if pattern.risk_reward < self.config['risk_reward_min']:
                continue
            
            # Check confidence threshold
            if pattern.confidence < self.config['min_confidence']:
                continue
            
            # Additional validation based on pattern type
            if pattern.pattern_type == ExhaustionPattern.TRIPLE_TAP:
                # Ensure recent price action supports the pattern
                if not self._validate_triple_tap(pattern, data):
                    continue
            
            valid_patterns.append(pattern)
        
        return valid_patterns
    
    def _validate_triple_tap(self, pattern: ExhaustionSignal, data: pd.DataFrame) -> bool:
        """Additional validation for triple tap pattern"""
        try:
            # Check if price is still near the level
            current_price = data['close'].iloc[-1]
            level = pattern.metadata.get('level', 0)
            
            if level > 0:
                distance = abs(current_price - level) / level
                return distance < 0.02  # Within 2% of level
            
            return True
            
        except Exception:
            return False
    
    def _determine_reversal_type(self, patterns: List[ExhaustionSignal]) -> ReversalType:
        """Determine overall reversal type from patterns"""
        if not patterns:
            return ReversalType.NO_REVERSAL
        
        bullish_count = sum(1 for p in patterns if p.reversal_type == ReversalType.BULLISH_REVERSAL)
        bearish_count = sum(1 for p in patterns if p.reversal_type == ReversalType.BEARISH_REVERSAL)
        
        if bullish_count > bearish_count:
            return ReversalType.BULLISH_REVERSAL
        elif bearish_count > bullish_count:
            return ReversalType.BEARISH_REVERSAL
        else:
            return ReversalType.NO_REVERSAL
    
    def _calculate_combined_confidence(self, patterns: List[ExhaustionSignal]) -> float:
        """Calculate combined confidence from multiple patterns"""
        if not patterns:
            return 0
        
        # Weight patterns by their individual confidence
        total_confidence = 0
        total_weight = 0
        
        for pattern in patterns:
            weight = 1.0
            # Give extra weight to certain patterns
            if pattern.pattern_type in [ExhaustionPattern.DIVERGENCE_REVERSAL, ExhaustionPattern.TRIPLE_TAP]:
                weight = 1.5
            
            total_confidence += pattern.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        # Average confidence with bonus for multiple patterns
        base_confidence = total_confidence / total_weight
        pattern_bonus = min(len(patterns) * 5, 15)  # Max 15% bonus
        
        return min(base_confidence + pattern_bonus, 100)
    
    def _calculate_trade_parameters(self, patterns: List[ExhaustionSignal], 
                                  reversal_type: ReversalType, 
                                  data: pd.DataFrame) -> Tuple[float, float, List[float]]:
        """Calculate optimal entry, stop loss, and targets"""
        if not patterns:
            current_price = data['close'].iloc[-1]
            return current_price, current_price * 0.98, [current_price * 1.02]
        
        # Average entry zones
        entry_zones = [p.entry_zone for p in patterns]
        optimal_entry = np.mean([np.mean(zone) for zone in entry_zones])
        
        # Most conservative stop loss
        if reversal_type == ReversalType.BULLISH_REVERSAL:
            stop_loss = min(p.stop_loss for p in patterns)
        else:
            stop_loss = max(p.stop_loss for p in patterns)
        
        # Average targets
        all_targets = []
        for p in patterns:
            all_targets.extend(p.targets)
        
        # Group into 3 target levels
        targets = [
            np.percentile(all_targets, 33),
            np.percentile(all_targets, 66),
            np.percentile(all_targets, 90)
        ]
        
        return optimal_entry, stop_loss, targets
    
    def _assess_risk_score(self, patterns: List[ExhaustionSignal], data: pd.DataFrame) -> float:
        """Assess risk score 1-10"""
        risk_score = 5  # Base risk
        
        try:
            # Factor 1: Pattern conflicts
            reversal_types = [p.reversal_type for p in patterns]
            if len(set(reversal_types)) > 1:
                risk_score += 2  # Conflicting signals increase risk
            
            # Factor 2: Volume confirmation
            volume_confirmed = sum(1 for p in patterns if p.volume_confirmation)
            if volume_confirmed < len(patterns) / 2:
                risk_score += 1
            
            # Factor 3: Market volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            if volatility > 0.03:  # High volatility
                risk_score += 1.5
            
            # Factor 4: Distance from support/resistance
            current_price = data['close'].iloc[-1]
            recent_high = data['high'].iloc[-20:].max()
            recent_low = data['low'].iloc[-20:].min()
            
            position_in_range = (current_price - recent_low) / (recent_high - recent_low)
            if position_in_range > 0.8 or position_in_range < 0.2:
                risk_score += 1  # Near extremes is riskier
            
            return min(risk_score, 10)
            
        except Exception:
            return 7  # Default moderate-high risk
    
    def _assess_opportunity_score(self, patterns: List[ExhaustionSignal], 
                                 confidence: float, risk_score: float) -> float:
        """Assess opportunity score 1-10"""
        opportunity = 0
        
        # Base opportunity from confidence
        opportunity += (confidence / 100) * 5
        
        # Pattern quality bonus
        high_quality_patterns = sum(1 for p in patterns if p.strength in [ReversalStrength.STRONG, ReversalStrength.VERY_STRONG])
        opportunity += min(high_quality_patterns * 1.5, 3)
        
        # Risk-adjusted opportunity
        risk_adjustment = (10 - risk_score) / 10
        opportunity *= risk_adjustment
        
        # Risk/reward bonus
        avg_rr = np.mean([p.risk_reward for p in patterns]) if patterns else 1
        if avg_rr > 3:
            opportunity += 1.5
        elif avg_rr > 2:
            opportunity += 1
        
        return min(max(opportunity, 1), 10)
    
    def _empty_setup(self, symbol: str, timeframe: str) -> ReversalSetup:
        """Return empty setup for error cases"""
        return ReversalSetup(
            symbol=symbol,
            timeframe=timeframe,
            setup_type=ReversalType.NO_REVERSAL,
            patterns_detected=[],
            combined_confidence=0,
            optimal_entry=0,
            stop_loss=0,
            profit_targets=[],
            risk_score=10,
            opportunity_score=0,
            actionable=False,
            timestamp=datetime.now()
        )
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            total_setups = sum(len(history) for history in self.reversal_history.values())
            
            avg_accuracy = np.mean(self.reversal_accuracy) if self.reversal_accuracy else 0
            false_signal_rate = np.mean(self.false_signal_rate) if self.false_signal_rate else 0
            
            pattern_performance = {}
            for pattern_type, results in self.pattern_success_rate.items():
                if results:
                    pattern_performance[pattern_type] = f"{np.mean(results):.1%}"
            
            return {
                'agent_name': 'Reversal Setup Agent',
                'status': 'active',
                'total_setups_detected': total_setups,
                'reversal_accuracy': f"{avg_accuracy:.1%}",
                'false_signal_rate': f"{false_signal_rate:.1%}",
                'pattern_performance': pattern_performance,
                'min_confidence_threshold': self.config['min_confidence'],
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Reversal Setup Agent', 'status': 'error'}
    
    def update_pattern_outcome(self, pattern_type: str, success: bool, pnl: float):
        """Update pattern outcome for performance tracking"""
        try:
            self.pattern_success_rate[pattern_type].append(1.0 if success else 0.0)
            self.reversal_accuracy.append(1.0 if success else 0.0)
            
            if not success and pnl < 0:
                self.false_signal_rate.append(1.0)
            else:
                self.false_signal_rate.append(0.0)
                
        except Exception as e:
            self.logger.error(f"Error updating pattern outcome: {e}")