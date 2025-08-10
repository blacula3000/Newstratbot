"""
Magnet Level Agent - Key Price Level Detection and Analysis
Identifies and monitors critical price levels that act as magnets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque, defaultdict

class MagnetType(Enum):
    PRIOR_HIGH_LOW = "prior_high_low"
    WEEKLY_OPEN = "weekly_open"
    MONTHLY_OPEN = "monthly_open"
    DAILY_OPEN = "daily_open"
    PIVOT_POINT = "pivot_point"
    VWAP = "vwap"
    ROUND_NUMBER = "round_number"
    INSTITUTIONAL_LEVEL = "institutional_level"
    FIBONACCI_LEVEL = "fibonacci_level"
    VOLUME_POC = "volume_poc"  # Point of Control

class MagnetStrength(Enum):
    VERY_STRONG = "very_strong"  # 90-100 strength
    STRONG = "strong"  # 75-90 strength
    MODERATE = "moderate"  # 50-75 strength
    WEAK = "weak"  # 25-50 strength
    VERY_WEAK = "very_weak"  # 0-25 strength

class MagnetStatus(Enum):
    ACTIVE = "active"  # Currently influencing price
    TESTED = "tested"  # Recently tested
    BROKEN = "broken"  # Level broken through
    DORMANT = "dormant"  # Not currently active

@dataclass
class MagnetLevel:
    level: float
    magnet_type: MagnetType
    strength: float  # 0-100 strength score
    status: MagnetStatus
    created_time: datetime
    last_test_time: Optional[datetime]
    test_count: int
    rejection_count: int  # How many times price rejected from level
    break_count: int  # How many times level was broken
    distance_score: float  # How close current price is
    volume_profile: Dict  # Volume distribution around level
    timeframe: str
    metadata: Dict

@dataclass
class MagnetInteraction:
    magnet_level: MagnetLevel
    interaction_type: str  # 'approach', 'test', 'rejection', 'break'
    price: float
    timestamp: datetime
    volume: float
    strength: float  # Strength of interaction
    follow_through: Optional[float]  # Price movement after interaction

@dataclass
class MagnetAnalysis:
    symbol: str
    timeframe: str
    current_price: float
    active_magnets: List[MagnetLevel]
    nearest_magnets: List[Tuple[MagnetLevel, float]]  # (magnet, distance)
    magnet_interactions: List[MagnetInteraction]
    dominant_magnet: Optional[MagnetLevel]
    magnet_confluence: List[List[MagnetLevel]]  # Groups of confluent levels
    price_attraction_score: float  # Overall magnetic pull
    breakout_probability: float  # Probability of breaking key levels
    actionable_levels: List[Dict]
    timestamp: datetime

class MagnetLevelAgent:
    """
    Magnet Level Agent for identifying and monitoring key price levels
    
    Monitors:
    - Prior highs/lows with significance scoring
    - Daily/Weekly/Monthly opens
    - Pivot points and VWAP levels
    - Round numbers and psychological levels
    - Volume-based levels (POC)
    - Fibonacci retracements/extensions
    - Level confluence and magnetic fields
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.magnet_levels = {}  # {symbol: {timeframe: [MagnetLevel]}}
        self.interaction_history = {}  # {symbol: deque[MagnetInteraction]}
        self.open_levels_cache = {}  # Cached open levels
        self.vwap_cache = {}  # VWAP calculations
        
        # Performance tracking
        self.level_accuracy = deque(maxlen=100)
        self.rejection_success_rate = deque(maxlen=100)
        self.breakout_success_rate = deque(maxlen=100)
        
        self.logger.info("🧲 Magnet Level Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            'min_strength': 40,
            'confluence_tolerance_pct': 0.005,  # 0.5% for confluence
            'level_test_tolerance_pct': 0.003,  # 0.3% for level test
            'min_touches_for_strength': 2,
            'round_number_levels': [0.00, 0.25, 0.50, 0.75],  # Round number endings
            'fibonacci_ratios': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618],
            'pivot_calculation_period': 1,  # Days for pivot calculation
            'vwap_period': 20,  # VWAP calculation period
            'volume_profile_bins': 20,  # Number of price bins for volume profile
            'max_level_age_days': 30,
            'distance_decay_factor': 0.8,  # How distance affects strength
            'timeframes': ['5m', '15m', '1h', '4h', '1d'],
            'min_volume_for_poc': 1000,
            'institutional_round_numbers': [10, 25, 50, 100, 250, 500, 1000]
        }
    
    def analyze_magnet_levels(self, symbol: str, timeframe: str, 
                             data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> MagnetAnalysis:
        """
        Comprehensive magnet level analysis
        """
        try:
            # Initialize tracking
            if symbol not in self.magnet_levels:
                self.magnet_levels[symbol] = {}
                self.interaction_history[symbol] = deque(maxlen=200)
                self.open_levels_cache[symbol] = {}
                self.vwap_cache[symbol] = {}
            
            if timeframe not in self.magnet_levels[symbol]:
                self.magnet_levels[symbol][timeframe] = []
            
            # Current price
            current_price = data['close'].iloc[-1]
            
            # Step 1: Identify and update magnet levels
            magnet_levels = self._identify_magnet_levels(symbol, timeframe, data, volume_data)
            
            # Step 2: Update level status based on recent price action
            updated_levels = self._update_level_status(magnet_levels, data)
            
            # Step 3: Find magnet interactions
            interactions = self._detect_magnet_interactions(symbol, updated_levels, data)
            
            # Step 4: Find nearest and most significant magnets
            nearest_magnets = self._find_nearest_magnets(current_price, updated_levels)
            
            # Step 5: Identify dominant magnet (strongest influence)
            dominant_magnet = self._identify_dominant_magnet(current_price, updated_levels)
            
            # Step 6: Find magnet confluence zones
            confluence_zones = self._find_magnet_confluence(updated_levels)
            
            # Step 7: Calculate price attraction score
            attraction_score = self._calculate_attraction_score(current_price, updated_levels)
            
            # Step 8: Assess breakout probability
            breakout_probability = self._assess_breakout_probability(
                current_price, updated_levels, data
            )
            
            # Step 9: Generate actionable levels
            actionable_levels = self._generate_actionable_levels(
                current_price, updated_levels, confluence_zones
            )
            
            analysis = MagnetAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                active_magnets=updated_levels,
                nearest_magnets=nearest_magnets,
                magnet_interactions=interactions,
                dominant_magnet=dominant_magnet,
                magnet_confluence=confluence_zones,
                price_attraction_score=attraction_score,
                breakout_probability=breakout_probability,
                actionable_levels=actionable_levels,
                timestamp=datetime.now()
            )
            
            # Store updated levels
            self.magnet_levels[symbol][timeframe] = updated_levels
            
            # Store interactions
            self.interaction_history[symbol].extend(interactions)
            
            self.logger.debug(f"🧲 Magnet analysis complete: {symbol} {timeframe} - "
                            f"{len(updated_levels)} levels, attraction: {attraction_score:.1f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing magnet levels for {symbol} {timeframe}: {e}")
            return self._empty_analysis(symbol, timeframe)
    
    def _identify_magnet_levels(self, symbol: str, timeframe: str, 
                              data: pd.DataFrame, volume_data: Optional[pd.DataFrame]) -> List[MagnetLevel]:
        """Identify various types of magnet levels"""
        magnet_levels = []
        current_time = datetime.now()
        
        try:
            # 1. Prior highs and lows
            prior_levels = self._identify_prior_high_low_levels(data, timeframe)
            magnet_levels.extend(prior_levels)
            
            # 2. Open levels (daily, weekly, monthly)
            open_levels = self._identify_open_levels(symbol, data, timeframe)
            magnet_levels.extend(open_levels)
            
            # 3. Pivot points
            pivot_levels = self._identify_pivot_points(data, timeframe)
            magnet_levels.extend(pivot_levels)
            
            # 4. VWAP levels
            vwap_levels = self._identify_vwap_levels(symbol, data, timeframe)
            magnet_levels.extend(vwap_levels)
            
            # 5. Round numbers
            round_number_levels = self._identify_round_numbers(data, timeframe)
            magnet_levels.extend(round_number_levels)
            
            # 6. Volume-based levels
            if volume_data is not None:
                volume_levels = self._identify_volume_levels(data, volume_data, timeframe)
                magnet_levels.extend(volume_levels)
            
            # 7. Fibonacci levels
            fibonacci_levels = self._identify_fibonacci_levels(data, timeframe)
            magnet_levels.extend(fibonacci_levels)
            
            # Clean up old levels and merge similar ones
            valid_levels = self._cleanup_and_merge_levels(magnet_levels)
            
            return valid_levels
            
        except Exception as e:
            self.logger.error(f"Error identifying magnet levels: {e}")
            return []
    
    def _identify_prior_high_low_levels(self, data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify prior high/low levels with significance scoring"""
        levels = []
        
        try:
            # Find local highs and lows with different lookback periods
            for lookback in [5, 10, 20]:
                highs = data['high'].rolling(window=lookback, center=True).max()
                lows = data['low'].rolling(window=lookback, center=True).min()
                
                for i in range(lookback, len(data) - lookback):
                    # Local high
                    if data['high'].iloc[i] == highs.iloc[i]:
                        strength = self._calculate_level_strength(data, i, data['high'].iloc[i], 'high')
                        if strength >= self.config['min_strength']:
                            level = MagnetLevel(
                                level=data['high'].iloc[i],
                                magnet_type=MagnetType.PRIOR_HIGH_LOW,
                                strength=strength,
                                status=MagnetStatus.ACTIVE,
                                created_time=datetime.now(),
                                last_test_time=None,
                                test_count=0,
                                rejection_count=0,
                                break_count=0,
                                distance_score=0,
                                volume_profile={},
                                timeframe=timeframe,
                                metadata={'level_type': 'high', 'lookback': lookback, 'index': i}
                            )
                            levels.append(level)
                    
                    # Local low
                    if data['low'].iloc[i] == lows.iloc[i]:
                        strength = self._calculate_level_strength(data, i, data['low'].iloc[i], 'low')
                        if strength >= self.config['min_strength']:
                            level = MagnetLevel(
                                level=data['low'].iloc[i],
                                magnet_type=MagnetType.PRIOR_HIGH_LOW,
                                strength=strength,
                                status=MagnetStatus.ACTIVE,
                                created_time=datetime.now(),
                                last_test_time=None,
                                test_count=0,
                                rejection_count=0,
                                break_count=0,
                                distance_score=0,
                                volume_profile={},
                                timeframe=timeframe,
                                metadata={'level_type': 'low', 'lookback': lookback, 'index': i}
                            )
                            levels.append(level)
            
            return levels
            
        except Exception:
            return []
    
    def _calculate_level_strength(self, data: pd.DataFrame, index: int, level: float, level_type: str) -> float:
        """Calculate the strength of a price level"""
        try:
            strength = 0
            
            # Factor 1: Volume at the level
            if 'volume' in data.columns:
                level_volume = data['volume'].iloc[index]
                avg_volume = data['volume'].rolling(window=20).mean().iloc[index]
                if level_volume > avg_volume:
                    strength += min(level_volume / avg_volume, 3) * 15
            
            # Factor 2: Number of times tested
            tolerance = level * self.config['level_test_tolerance_pct']
            test_count = 0
            
            for i in range(max(0, index - 50), min(len(data), index + 50)):
                if level_type == 'high':
                    if abs(data['high'].iloc[i] - level) <= tolerance:
                        test_count += 1
                else:
                    if abs(data['low'].iloc[i] - level) <= tolerance:
                        test_count += 1
            
            strength += min(test_count * 8, 40)
            
            # Factor 3: Reaction strength (how much price moved away)
            reaction_strength = 0
            window = min(10, len(data) - index - 1)
            if window > 0:
                if level_type == 'high':
                    for i in range(1, window + 1):
                        if index + i < len(data):
                            drop = (level - data['low'].iloc[index + i]) / level
                            reaction_strength = max(reaction_strength, drop * 100)
                else:
                    for i in range(1, window + 1):
                        if index + i < len(data):
                            rise = (data['high'].iloc[index + i] - level) / level
                            reaction_strength = max(reaction_strength, rise * 100)
            
            strength += min(reaction_strength * 3, 30)
            
            # Factor 4: Time significance (weekly/monthly highs get bonus)
            if timeframe in ['1d', '1w']:
                strength += 15
            
            return min(strength, 100)
            
        except Exception:
            return 50  # Default moderate strength
    
    def _identify_open_levels(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify daily, weekly, monthly open levels"""
        levels = []
        
        try:
            # This would integrate with actual open level data
            # For now, estimating based on available data
            current_price = data['close'].iloc[-1]
            
            # Daily open (using first candle of recent period)
            if len(data) > 20:
                daily_open = data['open'].iloc[-20]  # Approximate daily open
                levels.append(MagnetLevel(
                    level=daily_open,
                    magnet_type=MagnetType.DAILY_OPEN,
                    strength=70,
                    status=MagnetStatus.ACTIVE,
                    created_time=datetime.now() - timedelta(days=1),
                    last_test_time=None,
                    test_count=0,
                    rejection_count=0,
                    break_count=0,
                    distance_score=0,
                    volume_profile={},
                    timeframe=timeframe,
                    metadata={'open_type': 'daily'}
                ))
            
            # Weekly open (estimated)
            if len(data) > 100 and timeframe in ['1h', '4h', '1d']:
                weekly_open = data['open'].iloc[-100]
                levels.append(MagnetLevel(
                    level=weekly_open,
                    magnet_type=MagnetType.WEEKLY_OPEN,
                    strength=85,
                    status=MagnetStatus.ACTIVE,
                    created_time=datetime.now() - timedelta(weeks=1),
                    last_test_time=None,
                    test_count=0,
                    rejection_count=0,
                    break_count=0,
                    distance_score=0,
                    volume_profile={},
                    timeframe=timeframe,
                    metadata={'open_type': 'weekly'}
                ))
            
            return levels
            
        except Exception:
            return []
    
    def _identify_pivot_points(self, data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify pivot point levels"""
        levels = []
        
        try:
            if len(data) < 10:
                return levels
            
            # Standard pivot points
            if timeframe in ['15m', '1h', '4h', '1d']:
                high = data['high'].iloc[-2]  # Previous period high
                low = data['low'].iloc[-2]   # Previous period low
                close = data['close'].iloc[-2]  # Previous period close
                
                # Calculate pivot levels
                pivot = (high + low + close) / 3
                r1 = (2 * pivot) - low
                s1 = (2 * pivot) - high
                r2 = pivot + (high - low)
                s2 = pivot - (high - low)
                
                pivot_levels = [
                    (pivot, 80, 'pivot'),
                    (r1, 70, 'r1'),
                    (s1, 70, 's1'),
                    (r2, 60, 'r2'),
                    (s2, 60, 's2')
                ]
                
                for level_price, strength, level_name in pivot_levels:
                    levels.append(MagnetLevel(
                        level=level_price,
                        magnet_type=MagnetType.PIVOT_POINT,
                        strength=strength,
                        status=MagnetStatus.ACTIVE,
                        created_time=datetime.now(),
                        last_test_time=None,
                        test_count=0,
                        rejection_count=0,
                        break_count=0,
                        distance_score=0,
                        volume_profile={},
                        timeframe=timeframe,
                        metadata={'pivot_type': level_name}
                    ))
            
            return levels
            
        except Exception:
            return []
    
    def _identify_vwap_levels(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify VWAP-based levels"""
        levels = []
        
        try:
            if 'volume' not in data.columns or len(data) < self.config['vwap_period']:
                return levels
            
            # Calculate VWAP
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            
            current_vwap = vwap.iloc[-1]
            
            levels.append(MagnetLevel(
                level=current_vwap,
                magnet_type=MagnetType.VWAP,
                strength=75,
                status=MagnetStatus.ACTIVE,
                created_time=datetime.now(),
                last_test_time=None,
                test_count=0,
                rejection_count=0,
                break_count=0,
                distance_score=0,
                volume_profile={},
                timeframe=timeframe,
                metadata={'vwap_period': self.config['vwap_period']}
            ))
            
            return levels
            
        except Exception:
            return []
    
    def _identify_round_numbers(self, data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify round number levels"""
        levels = []
        
        try:
            current_price = data['close'].iloc[-1]
            price_range = data['high'].max() - data['low'].min()
            
            # Find relevant round numbers
            for base in self.config['institutional_round_numbers']:
                # Scale based on price
                if current_price > base:
                    # Find round numbers around current price
                    lower_bound = current_price - price_range
                    upper_bound = current_price + price_range
                    
                    # Generate round numbers in range
                    start = int(lower_bound / base) * base
                    end = int(upper_bound / base) * base + base
                    
                    for round_price in range(int(start), int(end) + 1, int(base)):
                        if lower_bound <= round_price <= upper_bound:
                            # Calculate strength based on psychological significance
                            strength = self._calculate_round_number_strength(round_price, base)
                            
                            if strength >= self.config['min_strength']:
                                levels.append(MagnetLevel(
                                    level=float(round_price),
                                    magnet_type=MagnetType.ROUND_NUMBER,
                                    strength=strength,
                                    status=MagnetStatus.ACTIVE,
                                    created_time=datetime.now(),
                                    last_test_time=None,
                                    test_count=0,
                                    rejection_count=0,
                                    break_count=0,
                                    distance_score=0,
                                    volume_profile={},
                                    timeframe=timeframe,
                                    metadata={'base': base, 'round_number': round_price}
                                ))
            
            return levels[:10]  # Limit to top 10 round numbers
            
        except Exception:
            return []
    
    def _calculate_round_number_strength(self, price: float, base: float) -> float:
        """Calculate psychological strength of round numbers"""
        strength = 40  # Base strength
        
        # Bigger round numbers are stronger
        if base >= 1000:
            strength += 30
        elif base >= 100:
            strength += 20
        elif base >= 50:
            strength += 15
        elif base >= 10:
            strength += 10
        
        # Even rounder numbers (00, 000) get bonus
        if price % (base * 10) == 0:
            strength += 15
        elif price % (base * 5) == 0:
            strength += 10
        elif price % (base * 2) == 0:
            strength += 5
        
        return min(strength, 100)
    
    def _identify_volume_levels(self, data: pd.DataFrame, volume_data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify volume-based levels (Point of Control)"""
        levels = []
        
        try:
            if len(data) < 20:
                return levels
            
            # Create volume profile
            price_range = data['high'].max() - data['low'].min()
            bin_size = price_range / self.config['volume_profile_bins']
            
            volume_profile = {}
            
            for i in range(len(data)):
                price_level = round(data['close'].iloc[i] / bin_size) * bin_size
                volume = data['volume'].iloc[i] if 'volume' in data.columns else 1
                
                if price_level in volume_profile:
                    volume_profile[price_level] += volume
                else:
                    volume_profile[price_level] = volume
            
            # Find Point of Control (highest volume level)
            if volume_profile:
                poc_price = max(volume_profile, key=volume_profile.get)
                poc_volume = volume_profile[poc_price]
                
                if poc_volume >= self.config['min_volume_for_poc']:
                    strength = min(70 + (poc_volume / max(volume_profile.values())) * 30, 100)
                    
                    levels.append(MagnetLevel(
                        level=poc_price,
                        magnet_type=MagnetType.VOLUME_POC,
                        strength=strength,
                        status=MagnetStatus.ACTIVE,
                        created_time=datetime.now(),
                        last_test_time=None,
                        test_count=0,
                        rejection_count=0,
                        break_count=0,
                        distance_score=0,
                        volume_profile=volume_profile,
                        timeframe=timeframe,
                        metadata={'poc_volume': poc_volume}
                    ))
            
            return levels
            
        except Exception:
            return []
    
    def _identify_fibonacci_levels(self, data: pd.DataFrame, timeframe: str) -> List[MagnetLevel]:
        """Identify Fibonacci retracement/extension levels"""
        levels = []
        
        try:
            if len(data) < 50:
                return levels
            
            # Find significant swing high and low
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            swing_range = swing_high - swing_low
            
            # Calculate Fibonacci levels
            for ratio in self.config['fibonacci_ratios']:
                # Retracement levels
                if swing_range > 0:
                    fib_level = swing_high - (swing_range * ratio)
                    
                    strength = self._calculate_fibonacci_strength(ratio)
                    
                    levels.append(MagnetLevel(
                        level=fib_level,
                        magnet_type=MagnetType.FIBONACCI_LEVEL,
                        strength=strength,
                        status=MagnetStatus.ACTIVE,
                        created_time=datetime.now(),
                        last_test_time=None,
                        test_count=0,
                        rejection_count=0,
                        break_count=0,
                        distance_score=0,
                        volume_profile={},
                        timeframe=timeframe,
                        metadata={
                            'fibonacci_ratio': ratio,
                            'swing_high': swing_high,
                            'swing_low': swing_low,
                            'level_type': 'retracement'
                        }
                    ))
            
            return levels
            
        except Exception:
            return []
    
    def _calculate_fibonacci_strength(self, ratio: float) -> float:
        """Calculate strength of Fibonacci levels based on ratio significance"""
        strength_map = {
            0.236: 50,
            0.382: 65,
            0.5: 70,
            0.618: 80,  # Golden ratio - strongest
            0.786: 65,
            1.0: 75,    # 100% retracement
            1.618: 70   # Golden extension
        }
        
        return strength_map.get(ratio, 50)
    
    def _cleanup_and_merge_levels(self, levels: List[MagnetLevel]) -> List[MagnetLevel]:
        """Clean up old levels and merge similar ones"""
        try:
            # Remove levels that are too old
            max_age = timedelta(days=self.config['max_level_age_days'])
            current_time = datetime.now()
            
            valid_levels = [
                level for level in levels
                if current_time - level.created_time <= max_age
            ]
            
            # Merge similar levels
            merged_levels = []
            tolerance = self.config['confluence_tolerance_pct']
            
            for level in valid_levels:
                merged = False
                
                for existing in merged_levels:
                    if abs(level.level - existing.level) / existing.level <= tolerance:
                        # Merge levels - keep the stronger one
                        if level.strength > existing.strength:
                            existing.level = level.level
                            existing.strength = level.strength
                        merged = True
                        break
                
                if not merged:
                    merged_levels.append(level)
            
            return merged_levels
            
        except Exception:
            return levels
    
    def _update_level_status(self, levels: List[MagnetLevel], data: pd.DataFrame) -> List[MagnetLevel]:
        """Update level status based on recent price action"""
        try:
            current_price = data['close'].iloc[-1]
            tolerance = self.config['level_test_tolerance_pct']
            
            for level in levels:
                # Update distance score
                distance = abs(current_price - level.level) / level.level
                level.distance_score = max(0, 100 - (distance * 100 / 0.1))  # Decay over 10%
                
                # Check for recent tests
                recent_data = data.tail(10)
                
                for i in range(len(recent_data)):
                    candle = recent_data.iloc[i]
                    
                    # Check if level was tested
                    if abs(candle['high'] - level.level) / level.level <= tolerance or \
                       abs(candle['low'] - level.level) / level.level <= tolerance:
                        
                        level.test_count += 1
                        level.last_test_time = datetime.now()
                        level.status = MagnetStatus.TESTED
                        
                        # Check for rejection or break
                        if candle['close'] > level.level * (1 + tolerance) or \
                           candle['close'] < level.level * (1 - tolerance):
                            level.break_count += 1
                            level.status = MagnetStatus.BROKEN
                        else:
                            level.rejection_count += 1
                
                # Update strength based on tests and rejections
                if level.test_count > 0:
                    rejection_ratio = level.rejection_count / level.test_count
                    level.strength *= (0.8 + 0.4 * rejection_ratio)  # Boost for rejections
                
            return levels
            
        except Exception:
            return levels
    
    def _detect_magnet_interactions(self, symbol: str, levels: List[MagnetLevel], 
                                   data: pd.DataFrame) -> List[MagnetInteraction]:
        """Detect recent interactions with magnet levels"""
        interactions = []
        
        try:
            recent_data = data.tail(5)
            tolerance = self.config['level_test_tolerance_pct']
            
            for level in levels:
                for i in range(len(recent_data)):
                    candle = recent_data.iloc[i]
                    
                    # Check for interaction
                    interaction_type = None
                    price = 0
                    
                    if abs(candle['high'] - level.level) / level.level <= tolerance:
                        interaction_type = 'test'
                        price = candle['high']
                    elif abs(candle['low'] - level.level) / level.level <= tolerance:
                        interaction_type = 'test'
                        price = candle['low']
                    elif abs(candle['close'] - level.level) / level.level <= 0.02:  # Within 2%
                        interaction_type = 'approach'
                        price = candle['close']
                    
                    if interaction_type:
                        # Calculate interaction strength
                        volume = candle.get('volume', 0)
                        strength = self._calculate_interaction_strength(candle, level, interaction_type)
                        
                        # Calculate follow-through
                        follow_through = None
                        if i < len(recent_data) - 1:
                            next_candle = recent_data.iloc[i + 1]
                            follow_through = (next_candle['close'] - candle['close']) / candle['close']
                        
                        interaction = MagnetInteraction(
                            magnet_level=level,
                            interaction_type=interaction_type,
                            price=price,
                            timestamp=datetime.now(),
                            volume=volume,
                            strength=strength,
                            follow_through=follow_through
                        )
                        
                        interactions.append(interaction)
            
            return interactions
            
        except Exception:
            return []
    
    def _calculate_interaction_strength(self, candle: pd.Series, level: MagnetLevel, 
                                      interaction_type: str) -> float:
        """Calculate strength of magnet interaction"""
        try:
            strength = 50  # Base strength
            
            # Volume factor
            if 'volume' in candle and candle['volume'] > 0:
                # Would compare to average volume
                strength += 10
            
            # Level strength factor
            strength += (level.strength / 100) * 30
            
            # Interaction type factor
            if interaction_type == 'test':
                strength += 20
            elif interaction_type == 'rejection':
                strength += 30
            elif interaction_type == 'break':
                strength += 25
            
            # Wick factor (rejections at level)
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range > 0:
                wick_ratio = (total_range - body_size) / total_range
                if wick_ratio > 0.5:  # Large wicks indicate rejection
                    strength += 15
            
            return min(strength, 100)
            
        except Exception:
            return 50
    
    def _find_nearest_magnets(self, current_price: float, levels: List[MagnetLevel]) -> List[Tuple[MagnetLevel, float]]:
        """Find nearest magnet levels to current price"""
        nearest = []
        
        for level in levels:
            distance = abs(current_price - level.level) / current_price
            nearest.append((level, distance))
        
        # Sort by distance and return top 5
        nearest.sort(key=lambda x: x[1])
        return nearest[:5]
    
    def _identify_dominant_magnet(self, current_price: float, levels: List[MagnetLevel]) -> Optional[MagnetLevel]:
        """Identify the most influential magnet level"""
        try:
            if not levels:
                return None
            
            max_influence = 0
            dominant = None
            
            for level in levels:
                # Calculate influence score
                distance = abs(current_price - level.level) / current_price
                distance_factor = max(0, 1 - distance * 10)  # Decay with distance
                
                influence = level.strength * distance_factor
                
                if influence > max_influence:
                    max_influence = influence
                    dominant = level
            
            return dominant
            
        except Exception:
            return None
    
    def _find_magnet_confluence(self, levels: List[MagnetLevel]) -> List[List[MagnetLevel]]:
        """Find groups of confluent magnet levels"""
        confluence_zones = []
        
        try:
            tolerance = self.config['confluence_tolerance_pct']
            processed = set()
            
            for i, level1 in enumerate(levels):
                if i in processed:
                    continue
                
                confluence_group = [level1]
                processed.add(i)
                
                for j, level2 in enumerate(levels):
                    if j == i or j in processed:
                        continue
                    
                    if abs(level1.level - level2.level) / level1.level <= tolerance:
                        confluence_group.append(level2)
                        processed.add(j)
                
                if len(confluence_group) >= 2:
                    confluence_zones.append(confluence_group)
            
            return confluence_zones
            
        except Exception:
            return []
    
    def _calculate_attraction_score(self, current_price: float, levels: List[MagnetLevel]) -> float:
        """Calculate overall price attraction score"""
        try:
            total_attraction = 0
            
            for level in levels:
                distance = abs(current_price - level.level) / current_price
                distance_factor = max(0, 1 - distance * 5)  # Stronger decay
                
                attraction = level.strength * distance_factor * (level.distance_score / 100)
                total_attraction += attraction
            
            return min(total_attraction / 10, 100)  # Normalize to 0-100
            
        except Exception:
            return 50
    
    def _assess_breakout_probability(self, current_price: float, levels: List[MagnetLevel], 
                                   data: pd.DataFrame) -> float:
        """Assess probability of breaking through key levels"""
        try:
            # Find nearest resistance/support
            nearest_resistance = None
            nearest_support = None
            
            for level in levels:
                if level.level > current_price:
                    if nearest_resistance is None or level.level < nearest_resistance.level:
                        nearest_resistance = level
                elif level.level < current_price:
                    if nearest_support is None or level.level > nearest_support.level:
                        nearest_support = level
            
            # Calculate momentum
            returns = data['close'].pct_change().dropna()
            momentum = returns.iloc[-5:].mean()  # Recent momentum
            
            # Calculate breakout probability
            breakout_prob = 50  # Base probability
            
            # Momentum factor
            if momentum > 0.01:  # Strong upward momentum
                breakout_prob += 20
            elif momentum < -0.01:  # Strong downward momentum
                breakout_prob += 20
            
            # Level strength factor (weaker levels more likely to break)
            relevant_level = nearest_resistance if momentum > 0 else nearest_support
            if relevant_level:
                level_factor = (100 - relevant_level.strength) / 100
                breakout_prob += level_factor * 20
            
            # Volume factor
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                recent_volume = data['volume'].iloc[-3:].mean()
                if recent_volume > avg_volume * 1.5:
                    breakout_prob += 15
            
            return min(max(breakout_prob, 0), 100)
            
        except Exception:
            return 50
    
    def _generate_actionable_levels(self, current_price: float, levels: List[MagnetLevel], 
                                   confluence_zones: List[List[MagnetLevel]]) -> List[Dict]:
        """Generate actionable trading levels"""
        actionable = []
        
        try:
            # Confluence zones are high priority
            for zone in confluence_zones:
                if len(zone) >= 2:
                    avg_level = np.mean([l.level for l in zone])
                    combined_strength = np.mean([l.strength for l in zone])
                    
                    distance = abs(current_price - avg_level) / current_price
                    
                    if distance <= 0.05:  # Within 5%
                        action_type = 'resistance' if avg_level > current_price else 'support'
                        
                        actionable.append({
                            'level': avg_level,
                            'type': 'confluence_zone',
                            'action': action_type,
                            'strength': combined_strength,
                            'distance_pct': distance * 100,
                            'magnet_count': len(zone),
                            'confidence': min(combined_strength + len(zone) * 5, 95)
                        })
            
            # Individual strong levels
            for level in levels:
                if level.strength >= 70:
                    distance = abs(current_price - level.level) / current_price
                    
                    if distance <= 0.03:  # Within 3%
                        action_type = 'resistance' if level.level > current_price else 'support'
                        
                        actionable.append({
                            'level': level.level,
                            'type': level.magnet_type.value,
                            'action': action_type,
                            'strength': level.strength,
                            'distance_pct': distance * 100,
                            'test_count': level.test_count,
                            'confidence': level.strength
                        })
            
            # Sort by confidence and distance
            actionable.sort(key=lambda x: (x['confidence'], -x['distance_pct']), reverse=True)
            
            return actionable[:10]  # Top 10 actionable levels
            
        except Exception:
            return []
    
    def _empty_analysis(self, symbol: str, timeframe: str) -> MagnetAnalysis:
        """Return empty analysis for error cases"""
        return MagnetAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            current_price=0,
            active_magnets=[],
            nearest_magnets=[],
            magnet_interactions=[],
            dominant_magnet=None,
            magnet_confluence=[],
            price_attraction_score=0,
            breakout_probability=50,
            actionable_levels=[],
            timestamp=datetime.now()
        )
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            total_levels = sum(
                len(timeframes.get(tf, []))
                for timeframes in self.magnet_levels.values()
                for tf in timeframes
            )
            
            total_interactions = sum(len(interactions) for interactions in self.interaction_history.values())
            
            avg_accuracy = np.mean(self.level_accuracy) if self.level_accuracy else 0
            rejection_success = np.mean(self.rejection_success_rate) if self.rejection_success_rate else 0
            
            return {
                'agent_name': 'Magnet Level Agent',
                'status': 'active',
                'total_levels_tracked': total_levels,
                'total_interactions': total_interactions,
                'level_accuracy': f"{avg_accuracy:.1%}",
                'rejection_success_rate': f"{rejection_success:.1%}",
                'min_strength_threshold': self.config['min_strength'],
                'confluence_tolerance': f"{self.config['confluence_tolerance_pct'] * 100:.1f}%",
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Magnet Level Agent', 'status': 'error'}
    
    def update_level_outcome(self, level: float, level_type: str, success: bool, interaction_type: str):
        """Update level outcome for performance tracking"""
        try:
            self.level_accuracy.append(1.0 if success else 0.0)
            
            if interaction_type == 'rejection':
                self.rejection_success_rate.append(1.0 if success else 0.0)
            elif interaction_type == 'breakout':
                self.breakout_success_rate.append(1.0 if success else 0.0)
                
        except Exception as e:
            self.logger.error(f"Error updating level outcome: {e}")
    
    def get_confluence_zones(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get current confluence zones for a symbol"""
        try:
            if (symbol not in self.magnet_levels or 
                timeframe not in self.magnet_levels[symbol]):
                return []
            
            levels = self.magnet_levels[symbol][timeframe]
            confluence_zones = self._find_magnet_confluence(levels)
            
            result = []
            for zone in confluence_zones:
                avg_level = np.mean([l.level for l in zone])
                combined_strength = np.mean([l.strength for l in zone])
                
                result.append({
                    'level': avg_level,
                    'strength': combined_strength,
                    'magnet_count': len(zone),
                    'types': [l.magnet_type.value for l in zone]
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting confluence zones: {e}")
            return []