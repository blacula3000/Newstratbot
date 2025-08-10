"""
Entry Timing Agent - Precise Entry Logic and Market Timing
Advanced entry timing optimization for maximum edge capture
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

class EntryTiming(Enum):
    IMMEDIATE = "immediate"          # Enter immediately
    ON_PULLBACK = "on_pullback"      # Wait for pullback to enter
    ON_BREAKOUT = "on_breakout"      # Wait for breakout confirmation
    ON_RETEST = "on_retest"          # Wait for level retest
    SCALE_IN = "scale_in"            # Scale into position
    WAIT_SIGNAL = "wait_signal"      # Wait for additional confirmation

class EntryQuality(Enum):
    EXCELLENT = "excellent"   # 90-100% quality
    GOOD = "good"            # 70-90% quality
    FAIR = "fair"            # 50-70% quality
    POOR = "poor"            # 30-50% quality
    VERY_POOR = "very_poor"  # <30% quality

class MarketMicrostructure(Enum):
    STRONG_BUYERS = "strong_buyers"
    WEAK_BUYERS = "weak_buyers"
    STRONG_SELLERS = "strong_sellers"
    WEAK_SELLERS = "weak_sellers"
    BALANCED = "balanced"
    CHOPPY = "choppy"

@dataclass
class EntryOpportunity:
    symbol: str
    timeframe: str
    entry_type: EntryTiming
    target_price: float
    price_range: Tuple[float, float]  # (min_price, max_price)
    quality_score: float              # 0-100 entry quality
    urgency_score: float              # 0-100 how urgent is entry
    
    # Timing factors
    market_structure_score: float     # Current market structure favorability
    momentum_alignment: float         # Momentum alignment score
    volume_confirmation: bool         # Volume supports entry
    spread_favorability: float        # Bid-ask spread favorability
    
    # Risk factors
    slippage_estimate: float          # Expected slippage
    execution_risk: float             # Risk of poor execution
    timing_risk: float                # Risk of poor timing
    
    # Conditions
    entry_conditions: List[str]       # Conditions that must be met
    exit_conditions: List[str]        # Pre-defined exit conditions
    time_limit: Optional[datetime]    # Time limit for entry
    
    # Metadata
    confidence: float
    reasoning: str
    timestamp: datetime

@dataclass
class MarketMicrostructureAnalysis:
    symbol: str
    timeframe: str
    bid_ask_spread: float
    order_book_imbalance: float       # Bid vs ask depth imbalance
    recent_trade_flow: str            # "buying_pressure" or "selling_pressure"
    price_impact_estimate: float      # Estimated impact of our trade
    volume_profile: Dict[str, float]  # Volume at different price levels
    market_maker_behavior: str        # Passive, aggressive, or neutral
    retail_vs_institutional: float   # Flow composition estimate
    microstructure_regime: MarketMicrostructure
    timestamp: datetime

@dataclass
class EntryExecution:
    entry_opportunity: EntryOpportunity
    execution_price: float
    execution_time: datetime
    slippage_actual: float
    execution_quality: EntryQuality
    conditions_met: List[str]
    conditions_failed: List[str]
    success: bool
    notes: str

class EntryTimingAgent:
    """
    Advanced Entry Timing Agent for precise trade execution
    
    Analyzes:
    - Market microstructure and order flow
    - Optimal entry timing based on market conditions
    - Bid-ask spread dynamics and execution costs
    - Volume profile and support/resistance levels
    - Market maker vs retail flow patterns
    - Real-time entry opportunity scoring
    - Risk-adjusted entry optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.entry_opportunities = {}  # {symbol: {timeframe: [EntryOpportunity]}}
        self.execution_history = {}   # {symbol: deque[EntryExecution]}
        self.microstructure_cache = {}  # Recent microstructure analysis
        self.spread_history = {}      # Bid-ask spread tracking
        
        # Performance tracking
        self.entry_accuracy = deque(maxlen=100)
        self.slippage_accuracy = deque(maxlen=100)
        self.timing_performance = deque(maxlen=100)
        
        self.logger.info("🎯 Entry Timing Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            # Entry quality thresholds
            'min_entry_quality': 60,
            'excellent_quality_threshold': 90,
            'good_quality_threshold': 70,
            
            # Timing parameters
            'max_entry_delay_minutes': 30,
            'pullback_entry_threshold': 0.005,    # 0.5% pullback
            'breakout_confirmation_threshold': 0.003,  # 0.3% breakout
            'retest_tolerance': 0.002,             # 0.2% retest tolerance
            
            # Market microstructure
            'max_spread_bps': 20,                  # 20 basis points max spread
            'min_volume_confirmation': 1.5,       # 1.5x average volume
            'order_book_depth_threshold': 10000,   # Min order book depth
            'max_price_impact_bps': 10,           # 10 bps max price impact
            
            # Risk thresholds
            'max_slippage_bps': 5,                # 5 bps max expected slippage
            'max_execution_risk': 0.3,            # 30% max execution risk
            'urgency_decay_minutes': 15,          # Urgency decay time
            
            # Scale-in parameters
            'scale_in_levels': 3,                  # Number of scale-in levels
            'scale_in_spacing': 0.002,            # 0.2% spacing between levels
            'initial_scale_percentage': 0.4,       # 40% initial scale
            
            # Market hours consideration
            'market_open_buffer_minutes': 30,     # Buffer after market open
            'market_close_buffer_minutes': 60,    # Buffer before market close
            'lunch_hour_penalty': 0.8,            # Reduce quality during lunch hour
            
            # Advanced timing
            'momentum_alignment_weight': 0.3,
            'structure_alignment_weight': 0.4,
            'microstructure_weight': 0.3,
            
            # Exit conditions
            'profit_target_multiplier': 2.5,      # 2.5x risk for profit target
            'stop_loss_buffer': 1.1,              # 10% buffer for stop loss
            'time_stop_hours': 24                 # 24-hour time stop
        }
    
    def analyze_entry_timing(self, symbol: str, timeframe: str, signal_data: Dict, 
                           market_data: pd.DataFrame, level_2_data: Optional[Dict] = None) -> EntryOpportunity:
        """
        Comprehensive entry timing analysis
        """
        try:
            # Initialize tracking
            if symbol not in self.entry_opportunities:
                self.entry_opportunities[symbol] = {}
                self.execution_history[symbol] = deque(maxlen=50)
                self.microstructure_cache[symbol] = {}
                self.spread_history[symbol] = deque(maxlen=100)
            
            if timeframe not in self.entry_opportunities[symbol]:
                self.entry_opportunities[symbol][timeframe] = []
            
            # Step 1: Analyze market microstructure
            microstructure = self._analyze_market_microstructure(symbol, market_data, level_2_data)
            
            # Step 2: Determine optimal entry type
            entry_type = self._determine_entry_type(signal_data, market_data, microstructure)
            
            # Step 3: Calculate target entry price and range
            target_price, price_range = self._calculate_entry_price_range(
                signal_data, market_data, entry_type, microstructure
            )
            
            # Step 4: Assess entry quality
            quality_score = self._calculate_entry_quality(
                signal_data, market_data, microstructure, entry_type
            )
            
            # Step 5: Calculate urgency score
            urgency_score = self._calculate_urgency_score(
                signal_data, market_data, entry_type
            )
            
            # Step 6: Analyze timing factors
            timing_factors = self._analyze_timing_factors(
                signal_data, market_data, microstructure
            )
            
            # Step 7: Estimate execution risks
            execution_risks = self._estimate_execution_risks(
                symbol, target_price, microstructure, signal_data
            )
            
            # Step 8: Define entry and exit conditions
            entry_conditions = self._define_entry_conditions(entry_type, signal_data, microstructure)
            exit_conditions = self._define_exit_conditions(signal_data, target_price)
            
            # Step 9: Set time limits
            time_limit = self._calculate_time_limit(entry_type, urgency_score)
            
            # Step 10: Generate reasoning and confidence
            reasoning, confidence = self._generate_entry_reasoning(
                entry_type, quality_score, urgency_score, timing_factors
            )
            
            opportunity = EntryOpportunity(
                symbol=symbol,
                timeframe=timeframe,
                entry_type=entry_type,
                target_price=target_price,
                price_range=price_range,
                quality_score=quality_score,
                urgency_score=urgency_score,
                
                # Timing factors
                market_structure_score=timing_factors['structure_score'],
                momentum_alignment=timing_factors['momentum_alignment'],
                volume_confirmation=timing_factors['volume_confirmation'],
                spread_favorability=timing_factors['spread_favorability'],
                
                # Risk factors
                slippage_estimate=execution_risks['slippage_estimate'],
                execution_risk=execution_risks['execution_risk'],
                timing_risk=execution_risks['timing_risk'],
                
                # Conditions
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                time_limit=time_limit,
                
                # Metadata
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            # Store opportunity
            self.entry_opportunities[symbol][timeframe].append(opportunity)
            
            # Cache microstructure analysis
            self.microstructure_cache[symbol][timeframe] = microstructure
            
            self.logger.debug(f"🎯 Entry timing analysis complete: {symbol} {timeframe} - "
                            f"Type: {entry_type.value}, Quality: {quality_score:.1f}")
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error analyzing entry timing for {symbol} {timeframe}: {e}")
            return self._default_entry_opportunity(symbol, timeframe)
    
    def _analyze_market_microstructure(self, symbol: str, market_data: pd.DataFrame, 
                                     level_2_data: Optional[Dict]) -> MarketMicrostructureAnalysis:
        """Analyze market microstructure and order flow"""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Bid-ask spread analysis
            if level_2_data and 'bid' in level_2_data and 'ask' in level_2_data:
                bid = level_2_data['bid']
                ask = level_2_data['ask']
                spread = (ask - bid) / current_price
            else:
                # Estimate spread from high-low range
                recent_hl = market_data[['high', 'low']].iloc[-5:]
                avg_range = recent_hl.apply(lambda x: (x['high'] - x['low']) / x['high'], axis=1).mean()
                spread = avg_range * 0.1  # Rough estimate
            
            # Order book imbalance
            if level_2_data and 'bid_size' in level_2_data and 'ask_size' in level_2_data:
                bid_size = level_2_data['bid_size']
                ask_size = level_2_data['ask_size']
                imbalance = (bid_size - ask_size) / (bid_size + ask_size)
            else:
                # Estimate from volume and price action
                recent_volume = market_data['volume'].iloc[-5:]
                recent_price_change = market_data['close'].pct_change().iloc[-5:]
                volume_price_corr = recent_volume.corr(recent_price_change)
                imbalance = volume_price_corr * 0.5  # Rough estimate
            
            # Recent trade flow
            if len(market_data) >= 10:
                price_changes = market_data['close'].pct_change().iloc[-10:]
                volume_changes = market_data['volume'].pct_change().iloc[-10:]
                
                # Positive correlation suggests buying pressure
                flow_correlation = price_changes.corr(volume_changes)
                if flow_correlation > 0.3:
                    trade_flow = "buying_pressure"
                elif flow_correlation < -0.3:
                    trade_flow = "selling_pressure"
                else:
                    trade_flow = "neutral"
            else:
                trade_flow = "neutral"
            
            # Price impact estimate
            recent_volume = market_data['volume'].iloc[-10:].mean()
            avg_trade_size = recent_volume / 100  # Rough estimate
            if avg_trade_size > 0:
                price_impact = 0.001 * np.sqrt(1000 / avg_trade_size)  # Square root law
            else:
                price_impact = 0.005
            
            # Volume profile (simplified)
            if len(market_data) >= 20:
                price_levels = pd.cut(market_data['close'].iloc[-20:], bins=5)
                volume_profile = market_data['volume'].iloc[-20:].groupby(price_levels).sum().to_dict()
                # Convert to string keys
                volume_profile = {str(k): v for k, v in volume_profile.items()}
            else:
                volume_profile = {"current_level": market_data['volume'].iloc[-1]}
            
            # Market maker behavior (simplified heuristic)
            if spread < 0.001:  # Tight spread
                mm_behavior = "aggressive"
            elif spread > 0.005:  # Wide spread
                mm_behavior = "passive"
            else:
                mm_behavior = "neutral"
            
            # Retail vs institutional flow (rough estimate)
            avg_volume = market_data['volume'].iloc[-20:].mean()
            recent_volume = market_data['volume'].iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 2:
                retail_vs_institutional = 0.3  # More institutional
            elif volume_ratio < 0.5:
                retail_vs_institutional = 0.7  # More retail
            else:
                retail_vs_institutional = 0.5  # Balanced
            
            # Determine microstructure regime
            if imbalance > 0.3 and trade_flow == "buying_pressure":
                regime = MarketMicrostructure.STRONG_BUYERS
            elif imbalance > 0.1 and trade_flow == "buying_pressure":
                regime = MarketMicrostructure.WEAK_BUYERS
            elif imbalance < -0.3 and trade_flow == "selling_pressure":
                regime = MarketMicrostructure.STRONG_SELLERS
            elif imbalance < -0.1 and trade_flow == "selling_pressure":
                regime = MarketMicrostructure.WEAK_SELLERS
            elif abs(imbalance) < 0.1:
                regime = MarketMicrostructure.BALANCED
            else:
                regime = MarketMicrostructure.CHOPPY
            
            return MarketMicrostructureAnalysis(
                symbol=symbol,
                timeframe="1m",  # Microstructure typically analyzed on short timeframes
                bid_ask_spread=spread,
                order_book_imbalance=imbalance,
                recent_trade_flow=trade_flow,
                price_impact_estimate=price_impact,
                volume_profile=volume_profile,
                market_maker_behavior=mm_behavior,
                retail_vs_institutional=retail_vs_institutional,
                microstructure_regime=regime,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market microstructure: {e}")
            return self._default_microstructure_analysis(symbol)
    
    def _determine_entry_type(self, signal_data: Dict, market_data: pd.DataFrame, 
                            microstructure: MarketMicrostructureAnalysis) -> EntryTiming:
        """Determine optimal entry timing type"""
        try:
            signal_strength = signal_data.get('confidence_score', 70)
            signal_type = signal_data.get('signal_type', 'breakout')
            urgency = signal_data.get('urgency', 'medium')
            
            # Immediate entry conditions
            if (signal_strength >= 85 and 
                microstructure.microstructure_regime in [MarketMicrostructure.STRONG_BUYERS, MarketMicrostructure.STRONG_SELLERS] and
                microstructure.bid_ask_spread < 0.002):
                return EntryTiming.IMMEDIATE
            
            # Breakout confirmation entry
            if (signal_type in ['breakout', 'trigger_break'] and 
                signal_strength >= 75):
                return EntryTiming.ON_BREAKOUT
            
            # Pullback entry for strong signals in unfavorable microstructure
            if (signal_strength >= 80 and 
                microstructure.bid_ask_spread > 0.003):
                return EntryTiming.ON_PULLBACK
            
            # Retest entry for reversal signals
            if signal_type in ['reversal', 'exhaustion'] and signal_strength >= 70:
                return EntryTiming.ON_RETEST
            
            # Scale-in for high conviction but uncertain timing
            if (signal_strength >= 75 and 
                microstructure.microstructure_regime == MarketMicrostructure.CHOPPY):
                return EntryTiming.SCALE_IN
            
            # Wait for additional signals
            if signal_strength < 70:
                return EntryTiming.WAIT_SIGNAL
            
            # Default to pullback entry
            return EntryTiming.ON_PULLBACK
            
        except Exception:
            return EntryTiming.ON_PULLBACK
    
    def _calculate_entry_price_range(self, signal_data: Dict, market_data: pd.DataFrame, 
                                   entry_type: EntryTiming, microstructure: MarketMicrostructureAnalysis) -> Tuple[float, Tuple[float, float]]:
        """Calculate target entry price and acceptable range"""
        try:
            current_price = market_data['close'].iloc[-1]
            signal_direction = signal_data.get('direction', 'LONG')
            
            if entry_type == EntryTiming.IMMEDIATE:
                target_price = current_price
                price_range = (
                    current_price * (1 - microstructure.bid_ask_spread/2),
                    current_price * (1 + microstructure.bid_ask_spread/2)
                )
            
            elif entry_type == EntryTiming.ON_PULLBACK:
                pullback_pct = self.config['pullback_entry_threshold']
                if signal_direction == 'LONG':
                    target_price = current_price * (1 - pullback_pct)
                    price_range = (
                        current_price * (1 - pullback_pct - 0.001),
                        current_price * (1 - pullback_pct + 0.001)
                    )
                else:
                    target_price = current_price * (1 + pullback_pct)
                    price_range = (
                        current_price * (1 + pullback_pct - 0.001),
                        current_price * (1 + pullback_pct + 0.001)
                    )
            
            elif entry_type == EntryTiming.ON_BREAKOUT:
                breakout_pct = self.config['breakout_confirmation_threshold']
                trigger_level = signal_data.get('trigger_level', current_price)
                
                if signal_direction == 'LONG':
                    target_price = trigger_level * (1 + breakout_pct)
                    price_range = (target_price * 0.999, target_price * 1.002)
                else:
                    target_price = trigger_level * (1 - breakout_pct)
                    price_range = (target_price * 0.998, target_price * 1.001)
            
            elif entry_type == EntryTiming.ON_RETEST:
                retest_level = signal_data.get('support_resistance_level', current_price)
                tolerance = self.config['retest_tolerance']
                
                target_price = retest_level
                price_range = (
                    retest_level * (1 - tolerance),
                    retest_level * (1 + tolerance)
                )
            
            elif entry_type == EntryTiming.SCALE_IN:
                target_price = current_price  # Start scaling from current price
                spacing = self.config['scale_in_spacing']
                
                if signal_direction == 'LONG':
                    price_range = (
                        current_price * (1 - spacing * 2),
                        current_price * (1 + spacing)
                    )
                else:
                    price_range = (
                        current_price * (1 - spacing),
                        current_price * (1 + spacing * 2)
                    )
            
            else:  # WAIT_SIGNAL
                target_price = current_price
                price_range = (current_price * 0.99, current_price * 1.01)
            
            return target_price, price_range
            
        except Exception as e:
            self.logger.error(f"Error calculating entry price range: {e}")
            current_price = market_data['close'].iloc[-1]
            return current_price, (current_price * 0.995, current_price * 1.005)
    
    def _calculate_entry_quality(self, signal_data: Dict, market_data: pd.DataFrame, 
                               microstructure: MarketMicrostructureAnalysis, entry_type: EntryTiming) -> float:
        """Calculate entry quality score"""
        try:
            quality = 50  # Base quality
            
            # Signal strength factor
            signal_strength = signal_data.get('confidence_score', 70)
            quality += (signal_strength - 70) * 0.5
            
            # Microstructure factors
            if microstructure.bid_ask_spread < 0.001:
                quality += 15  # Tight spread bonus
            elif microstructure.bid_ask_spread > 0.005:
                quality -= 10  # Wide spread penalty
            
            # Order book imbalance
            if abs(microstructure.order_book_imbalance) > 0.3:
                quality += 10  # Strong imbalance supports entry
            
            # Trade flow alignment
            signal_direction = signal_data.get('direction', 'LONG')
            if ((signal_direction == 'LONG' and microstructure.recent_trade_flow == 'buying_pressure') or
                (signal_direction == 'SHORT' and microstructure.recent_trade_flow == 'selling_pressure')):
                quality += 15
            
            # Volume confirmation
            if 'volume' in market_data.columns:
                recent_volume = market_data['volume'].iloc[-3:].mean()
                avg_volume = market_data['volume'].iloc[-20:].mean()
                
                if recent_volume > avg_volume * self.config['min_volume_confirmation']:
                    quality += 10
                elif recent_volume < avg_volume * 0.5:
                    quality -= 5
            
            # Entry type specific adjustments
            if entry_type == EntryTiming.IMMEDIATE:
                if microstructure.microstructure_regime in [MarketMicrostructure.STRONG_BUYERS, MarketMicrostructure.STRONG_SELLERS]:
                    quality += 10
                else:
                    quality -= 5  # Immediate entry without strong flow is risky
            
            elif entry_type == EntryTiming.ON_PULLBACK:
                quality += 5  # Pullback entries generally higher quality
            
            elif entry_type == EntryTiming.SCALE_IN:
                quality += 8  # Scale-in reduces timing risk
            
            # Market conditions
            returns = market_data['close'].pct_change().dropna()
            if len(returns) >= 10:
                volatility = returns.iloc[-10:].std()
                if volatility > 0.03:  # High volatility
                    quality -= 8
                elif volatility < 0.01:  # Low volatility
                    quality += 5
            
            return min(max(quality, 0), 100)
            
        except Exception:
            return 60  # Default moderate quality
    
    def _calculate_urgency_score(self, signal_data: Dict, market_data: pd.DataFrame, 
                               entry_type: EntryTiming) -> float:
        """Calculate urgency score for entry timing"""
        try:
            urgency = 50  # Base urgency
            
            # Signal-based urgency
            signal_strength = signal_data.get('confidence_score', 70)
            signal_age_minutes = signal_data.get('signal_age_minutes', 5)
            
            # Higher strength = higher urgency
            urgency += (signal_strength - 70) * 0.3
            
            # Time decay of urgency
            if signal_age_minutes > self.config['urgency_decay_minutes']:
                decay = (signal_age_minutes - self.config['urgency_decay_minutes']) * 2
                urgency = max(urgency - decay, 10)
            
            # Market momentum urgency
            if len(market_data) >= 5:
                recent_momentum = market_data['close'].pct_change().iloc[-5:].sum()
                urgency += abs(recent_momentum) * 1000  # Convert to urgency points
            
            # Entry type adjustments
            type_urgency = {
                EntryTiming.IMMEDIATE: 90,
                EntryTiming.ON_BREAKOUT: 80,
                EntryTiming.ON_PULLBACK: 60,
                EntryTiming.ON_RETEST: 50,
                EntryTiming.SCALE_IN: 40,
                EntryTiming.WAIT_SIGNAL: 20
            }
            urgency = (urgency + type_urgency.get(entry_type, 50)) / 2
            
            # Market hours considerations
            current_time = datetime.now().time()
            
            # Higher urgency near market close
            if current_time.hour >= 15:  # Afternoon
                urgency += 10
            
            # Lower urgency during lunch hours
            if 12 <= current_time.hour <= 13:
                urgency *= self.config['lunch_hour_penalty']
            
            return min(max(urgency, 0), 100)
            
        except Exception:
            return 50
    
    def _analyze_timing_factors(self, signal_data: Dict, market_data: pd.DataFrame, 
                              microstructure: MarketMicrostructureAnalysis) -> Dict[str, Any]:
        """Analyze various timing factors"""
        try:
            # Market structure score
            structure_score = 70  # Base score
            
            # Trend alignment
            if len(market_data) >= 20:
                short_ma = market_data['close'].rolling(5).mean().iloc[-1]
                long_ma = market_data['close'].rolling(20).mean().iloc[-1]
                current_price = market_data['close'].iloc[-1]
                
                if short_ma > long_ma and current_price > short_ma:
                    structure_score += 15  # Strong uptrend structure
                elif short_ma < long_ma and current_price < short_ma:
                    structure_score += 15  # Strong downtrend structure
                elif abs(short_ma - long_ma) / current_price < 0.01:
                    structure_score -= 10  # Choppy structure
            
            # Momentum alignment
            momentum_alignment = 50
            if len(market_data) >= 10:
                short_momentum = market_data['close'].pct_change().iloc[-5:].mean()
                long_momentum = market_data['close'].pct_change().iloc[-10:].mean()
                
                if short_momentum * long_momentum > 0:  # Same direction
                    momentum_alignment = 75 + min(abs(short_momentum) * 500, 20)
                else:  # Opposing momentum
                    momentum_alignment = 25
            
            # Volume confirmation
            volume_confirmation = False
            if 'volume' in market_data.columns and len(market_data) >= 10:
                recent_volume = market_data['volume'].iloc[-3:].mean()
                avg_volume = market_data['volume'].iloc[-20:].mean()
                volume_confirmation = recent_volume > avg_volume * self.config['min_volume_confirmation']
            
            # Spread favorability
            spread_favorability = 100 - min(microstructure.bid_ask_spread * 10000, 50)  # Convert to basis points
            
            return {
                'structure_score': min(max(structure_score, 0), 100),
                'momentum_alignment': min(max(momentum_alignment, 0), 100),
                'volume_confirmation': volume_confirmation,
                'spread_favorability': min(max(spread_favorability, 0), 100)
            }
            
        except Exception:
            return {
                'structure_score': 60,
                'momentum_alignment': 50,
                'volume_confirmation': False,
                'spread_favorability': 70
            }
    
    def _estimate_execution_risks(self, symbol: str, target_price: float, 
                                microstructure: MarketMicrostructureAnalysis, signal_data: Dict) -> Dict[str, float]:
        """Estimate execution-related risks"""
        try:
            # Slippage estimate
            base_slippage = microstructure.bid_ask_spread / 2  # Half spread
            
            # Add market impact
            position_size = signal_data.get('position_size', 0.03)
            market_impact = microstructure.price_impact_estimate * np.sqrt(position_size / 0.01)
            
            slippage_estimate = base_slippage + market_impact
            
            # Execution risk (probability of poor execution)
            execution_risk = 0.1  # Base 10% risk
            
            # Higher risk for wide spreads
            if microstructure.bid_ask_spread > 0.005:
                execution_risk += 0.2
            
            # Higher risk for large position sizes
            if position_size > 0.05:
                execution_risk += 0.15
            
            # Higher risk in choppy markets
            if microstructure.microstructure_regime == MarketMicrostructure.CHOPPY:
                execution_risk += 0.1
            
            # Timing risk (risk of poor entry timing)
            timing_risk = 0.15  # Base 15% timing risk
            
            # Signal strength affects timing risk
            signal_strength = signal_data.get('confidence_score', 70)
            timing_risk = timing_risk * (1 - (signal_strength - 50) / 100)
            
            # Market volatility affects timing risk
            returns = signal_data.get('recent_returns', [0])
            if len(returns) > 5:
                volatility = np.std(returns)
                if volatility > 0.02:  # High volatility
                    timing_risk += 0.1
            
            return {
                'slippage_estimate': min(slippage_estimate, 0.01),  # Cap at 1%
                'execution_risk': min(max(execution_risk, 0), 0.5),
                'timing_risk': min(max(timing_risk, 0), 0.4)
            }
            
        except Exception:
            return {
                'slippage_estimate': 0.002,  # 0.2% default
                'execution_risk': 0.2,      # 20% default
                'timing_risk': 0.15         # 15% default
            }
    
    def _define_entry_conditions(self, entry_type: EntryTiming, signal_data: Dict, 
                               microstructure: MarketMicrostructureAnalysis) -> List[str]:
        """Define specific conditions that must be met for entry"""
        conditions = []
        
        try:
            if entry_type == EntryTiming.IMMEDIATE:
                conditions = [
                    "Price within 0.1% of target",
                    f"Spread < {self.config['max_spread_bps']} bps",
                    "Market open and liquid"
                ]
            
            elif entry_type == EntryTiming.ON_PULLBACK:
                conditions = [
                    f"Price pulls back {self.config['pullback_entry_threshold']:.1%}",
                    "Volume > 50% of recent average",
                    "Pullback doesn't break key support/resistance"
                ]
            
            elif entry_type == EntryTiming.ON_BREAKOUT:
                conditions = [
                    f"Price breaks trigger level by {self.config['breakout_confirmation_threshold']:.1%}",
                    "Breakout confirmed with volume",
                    "No immediate rejection from breakout level"
                ]
            
            elif entry_type == EntryTiming.ON_RETEST:
                conditions = [
                    f"Price retests level within {self.config['retest_tolerance']:.1%}",
                    "Retest shows support/resistance holding",
                    "Volume pattern supports retest"
                ]
            
            elif entry_type == EntryTiming.SCALE_IN:
                conditions = [
                    "Initial entry at current price",
                    "Subsequent entries at predetermined levels",
                    "Stop loss applies to full position"
                ]
            
            elif entry_type == EntryTiming.WAIT_SIGNAL:
                conditions = [
                    "Additional confirmation signal",
                    "Improved market microstructure",
                    "Signal strength > 75%"
                ]
            
            # Add common conditions
            conditions.extend([
                "Position size within risk limits",
                "No major news events pending",
                "Adequate account balance"
            ])
            
            return conditions
            
        except Exception:
            return ["Standard entry conditions apply"]
    
    def _define_exit_conditions(self, signal_data: Dict, target_price: float) -> List[str]:
        """Define pre-planned exit conditions"""
        try:
            conditions = []
            
            # Profit target
            profit_target_pct = signal_data.get('profit_target_pct', 0.025)  # 2.5% default
            conditions.append(f"Profit target: {profit_target_pct:.1%}")
            
            # Stop loss
            stop_loss_pct = signal_data.get('stop_loss_pct', 0.015)  # 1.5% default
            conditions.append(f"Stop loss: {stop_loss_pct:.1%}")
            
            # Time stop
            time_stop_hours = self.config['time_stop_hours']
            conditions.append(f"Time stop: {time_stop_hours} hours")
            
            # Signal invalidation
            conditions.append("Exit if signal invalidated")
            
            # Risk management
            conditions.append("Exit if portfolio heat > 75%")
            conditions.append("Exit if correlation risk exceeds limits")
            
            return conditions
            
        except Exception:
            return ["Standard exit conditions apply"]
    
    def _calculate_time_limit(self, entry_type: EntryTiming, urgency_score: float) -> Optional[datetime]:
        """Calculate time limit for entry opportunity"""
        try:
            current_time = datetime.now()
            
            if entry_type == EntryTiming.IMMEDIATE:
                return current_time + timedelta(minutes=2)  # Very short window
            
            elif entry_type == EntryTiming.ON_BREAKOUT:
                return current_time + timedelta(minutes=15)  # Moderate window
            
            elif entry_type == EntryTiming.ON_PULLBACK:
                base_minutes = self.config['max_entry_delay_minutes']
                # Adjust based on urgency
                adjusted_minutes = base_minutes * (2 - urgency_score / 100)
                return current_time + timedelta(minutes=adjusted_minutes)
            
            elif entry_type == EntryTiming.ON_RETEST:
                return current_time + timedelta(hours=4)  # Longer window for retest
            
            elif entry_type == EntryTiming.SCALE_IN:
                return current_time + timedelta(hours=8)  # Extended window for scaling
            
            elif entry_type == EntryTiming.WAIT_SIGNAL:
                return current_time + timedelta(hours=24)  # Long window for additional signals
            
            return None
            
        except Exception:
            return datetime.now() + timedelta(minutes=30)
    
    def _generate_entry_reasoning(self, entry_type: EntryTiming, quality_score: float, 
                                urgency_score: float, timing_factors: Dict) -> Tuple[str, float]:
        """Generate reasoning and confidence for entry timing"""
        try:
            reasoning_parts = []
            confidence = 70  # Base confidence
            
            # Entry type reasoning
            type_descriptions = {
                EntryTiming.IMMEDIATE: "Strong microstructure supports immediate entry",
                EntryTiming.ON_PULLBACK: "Waiting for pullback to improve risk/reward",
                EntryTiming.ON_BREAKOUT: "Waiting for breakout confirmation",
                EntryTiming.ON_RETEST: "Waiting for level retest for optimal entry",
                EntryTiming.SCALE_IN: "Scaling in to reduce timing risk",
                EntryTiming.WAIT_SIGNAL: "Waiting for additional confirmation"
            }
            reasoning_parts.append(type_descriptions.get(entry_type, "Standard entry timing"))
            
            # Quality factors
            if quality_score >= 80:
                reasoning_parts.append("High entry quality conditions")
                confidence += 15
            elif quality_score < 50:
                reasoning_parts.append("Below-average entry conditions")
                confidence -= 10
            
            # Urgency factors
            if urgency_score >= 80:
                reasoning_parts.append("High urgency due to signal strength")
                confidence += 5
            elif urgency_score < 30:
                reasoning_parts.append("Low urgency allows patience")
                confidence += 5
            
            # Timing factors
            if timing_factors['volume_confirmation']:
                reasoning_parts.append("Volume confirms timing")
                confidence += 10
            
            if timing_factors['momentum_alignment'] >= 75:
                reasoning_parts.append("Strong momentum alignment")
                confidence += 10
            
            if timing_factors['spread_favorability'] >= 80:
                reasoning_parts.append("Favorable spread conditions")
                confidence += 5
            
            reasoning = "; ".join(reasoning_parts)
            return reasoning, min(max(confidence, 30), 95)
            
        except Exception:
            return "Standard entry timing analysis", 65
    
    def monitor_entry_opportunity(self, opportunity: EntryOpportunity, 
                                current_price: float, current_time: datetime) -> Dict[str, Any]:
        """Monitor an entry opportunity and provide status updates"""
        try:
            status = {
                'opportunity_id': f"{opportunity.symbol}_{opportunity.timeframe}_{int(opportunity.timestamp.timestamp())}",
                'status': 'active',
                'conditions_met': [],
                'conditions_pending': [],
                'should_enter': False,
                'time_remaining': None,
                'updated_quality': opportunity.quality_score,
                'notes': []
            }
            
            # Check time limit
            if opportunity.time_limit and current_time > opportunity.time_limit:
                status['status'] = 'expired'
                status['notes'].append("Time limit exceeded")
                return status
            
            if opportunity.time_limit:
                time_remaining = (opportunity.time_limit - current_time).total_seconds() / 60
                status['time_remaining'] = f"{time_remaining:.1f} minutes"
            
            # Check entry conditions based on type
            if opportunity.entry_type == EntryTiming.IMMEDIATE:
                if opportunity.price_range[0] <= current_price <= opportunity.price_range[1]:
                    status['conditions_met'].append("Price within range")
                    status['should_enter'] = True
                else:
                    status['conditions_pending'].append("Price outside range")
            
            elif opportunity.entry_type == EntryTiming.ON_PULLBACK:
                if current_price <= opportunity.target_price:
                    status['conditions_met'].append("Pullback achieved")
                    status['should_enter'] = True
                else:
                    status['conditions_pending'].append("Waiting for pullback")
            
            elif opportunity.entry_type == EntryTiming.ON_BREAKOUT:
                if current_price >= opportunity.target_price:  # Assuming long breakout
                    status['conditions_met'].append("Breakout confirmed")
                    status['should_enter'] = True
                else:
                    status['conditions_pending'].append("Waiting for breakout")
            
            # Update quality based on current conditions
            quality_decay = max(0, (current_time - opportunity.timestamp).total_seconds() / 3600 * 5)
            status['updated_quality'] = max(opportunity.quality_score - quality_decay, 30)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error monitoring entry opportunity: {e}")
            return {'status': 'error', 'should_enter': False}
    
    def execute_entry(self, opportunity: EntryOpportunity, execution_price: float) -> EntryExecution:
        """Record entry execution details"""
        try:
            # Calculate actual slippage
            target_price = opportunity.target_price
            actual_slippage = abs(execution_price - target_price) / target_price
            
            # Determine execution quality
            slippage_bps = actual_slippage * 10000
            
            if slippage_bps <= 2:
                exec_quality = EntryQuality.EXCELLENT
            elif slippage_bps <= 5:
                exec_quality = EntryQuality.GOOD
            elif slippage_bps <= 10:
                exec_quality = EntryQuality.FAIR
            elif slippage_bps <= 20:
                exec_quality = EntryQuality.POOR
            else:
                exec_quality = EntryQuality.VERY_POOR
            
            # Check which conditions were met
            conditions_met = ["Entry executed"]  # Basic condition
            conditions_failed = []
            
            # Determine success
            success = (exec_quality in [EntryQuality.EXCELLENT, EntryQuality.GOOD, EntryQuality.FAIR] and
                      actual_slippage <= opportunity.slippage_estimate * 2)
            
            execution = EntryExecution(
                entry_opportunity=opportunity,
                execution_price=execution_price,
                execution_time=datetime.now(),
                slippage_actual=actual_slippage,
                execution_quality=exec_quality,
                conditions_met=conditions_met,
                conditions_failed=conditions_failed,
                success=success,
                notes=f"Executed with {slippage_bps:.1f} bps slippage"
            )
            
            # Store execution
            if opportunity.symbol in self.execution_history:
                self.execution_history[opportunity.symbol].append(execution)
            
            # Update performance tracking
            self.slippage_accuracy.append(
                1.0 - min(abs(actual_slippage - opportunity.slippage_estimate) / opportunity.slippage_estimate, 1.0)
            )
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Error recording entry execution: {e}")
            return self._default_execution(opportunity)
    
    def _default_entry_opportunity(self, symbol: str, timeframe: str) -> EntryOpportunity:
        """Return default entry opportunity for error cases"""
        return EntryOpportunity(
            symbol=symbol,
            timeframe=timeframe,
            entry_type=EntryTiming.ON_PULLBACK,
            target_price=100.0,
            price_range=(99.0, 101.0),
            quality_score=60.0,
            urgency_score=50.0,
            market_structure_score=60.0,
            momentum_alignment=50.0,
            volume_confirmation=False,
            spread_favorability=70.0,
            slippage_estimate=0.002,
            execution_risk=0.2,
            timing_risk=0.15,
            entry_conditions=["Standard conditions"],
            exit_conditions=["Standard exits"],
            time_limit=datetime.now() + timedelta(hours=1),
            confidence=60.0,
            reasoning="Default entry timing applied",
            timestamp=datetime.now()
        )
    
    def _default_microstructure_analysis(self, symbol: str) -> MarketMicrostructureAnalysis:
        """Return default microstructure analysis"""
        return MarketMicrostructureAnalysis(
            symbol=symbol,
            timeframe="1m",
            bid_ask_spread=0.002,
            order_book_imbalance=0.0,
            recent_trade_flow="neutral",
            price_impact_estimate=0.001,
            volume_profile={"current": 1000},
            market_maker_behavior="neutral",
            retail_vs_institutional=0.5,
            microstructure_regime=MarketMicrostructure.BALANCED,
            timestamp=datetime.now()
        )
    
    def _default_execution(self, opportunity: EntryOpportunity) -> EntryExecution:
        """Return default execution for error cases"""
        return EntryExecution(
            entry_opportunity=opportunity,
            execution_price=opportunity.target_price,
            execution_time=datetime.now(),
            slippage_actual=0.002,
            execution_quality=EntryQuality.FAIR,
            conditions_met=["Executed"],
            conditions_failed=[],
            success=True,
            notes="Default execution recorded"
        )
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            total_opportunities = sum(
                len(timeframes.get(tf, []))
                for timeframes in self.entry_opportunities.values()
                for tf in timeframes
            )
            
            total_executions = sum(len(executions) for executions in self.execution_history.values())
            
            avg_slippage_accuracy = np.mean(self.slippage_accuracy) if self.slippage_accuracy else 0
            avg_timing_performance = np.mean(self.timing_performance) if self.timing_performance else 0
            
            return {
                'agent_name': 'Entry Timing Agent',
                'status': 'active',
                'total_opportunities': total_opportunities,
                'total_executions': total_executions,
                'slippage_prediction_accuracy': f"{avg_slippage_accuracy:.1%}",
                'timing_performance': f"{avg_timing_performance:.1%}",
                'min_entry_quality': self.config['min_entry_quality'],
                'max_spread_bps': self.config['max_spread_bps'],
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Entry Timing Agent', 'status': 'error'}
    
    def update_execution_outcome(self, execution_id: str, success: bool, final_pnl: float):
        """Update execution outcome for performance tracking"""
        try:
            self.entry_accuracy.append(1.0 if success else 0.0)
            
            # Update timing performance based on PnL
            if final_pnl != 0:
                # Normalize PnL to performance score
                performance = min(max((final_pnl + 0.02) / 0.04, 0), 1)  # -2% to +2% range
                self.timing_performance.append(performance)
                
        except Exception as e:
            self.logger.error(f"Error updating execution outcome: {e}")