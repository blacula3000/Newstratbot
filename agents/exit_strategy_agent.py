"""
Exit Strategy Agent - Dynamic Exit Management and Trade Optimization
Sophisticated exit timing and target management for maximum profit extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import math

class ExitTrigger(Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_DETERIORATION = "liquidity_deterioration"
    RISK_MANAGEMENT = "risk_management"
    PARTIAL_PROFIT = "partial_profit"

class ExitQuality(Enum):
    EXCELLENT = "excellent"  # 90-100% of potential captured
    GOOD = "good"           # 75-90% captured
    FAIR = "fair"           # 60-75% captured
    POOR = "poor"           # 40-60% captured
    VERY_POOR = "very_poor" # <40% captured

class TradeStage(Enum):
    EARLY = "early"         # 0-25% of expected duration
    DEVELOPING = "developing" # 25-50% of duration
    MATURE = "mature"       # 50-75% of duration
    LATE = "late"          # 75-100% of duration
    EXTENDED = "extended"   # Beyond expected duration

@dataclass
class ExitLevel:
    level_type: ExitTrigger
    price: float
    percentage_of_position: float  # 0.0-1.0 (what % of position to close)
    priority: int                  # 1=highest priority
    conditions: List[str]          # Conditions that must be met
    time_validity: Optional[datetime]  # When this exit expires
    is_active: bool
    created_time: datetime

@dataclass
class TradeMonitoring:
    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    direction: str  # 'LONG' or 'SHORT'
    
    # Performance metrics
    unrealized_pnl: float
    unrealized_pnl_pct: float
    max_favorable_excursion: float  # Best price reached
    max_adverse_excursion: float    # Worst price reached
    
    # Trade progression
    trade_stage: TradeStage
    time_in_trade: timedelta
    expected_duration: timedelta
    
    # Risk metrics
    current_risk: float
    risk_adjusted_return: float
    volatility_since_entry: float
    correlation_drift: float
    
    # Exit levels
    active_exits: List[ExitLevel]
    triggered_exits: List[ExitLevel]
    
    # Market conditions
    current_trend_strength: float
    momentum_deterioration: float
    volume_pattern_change: float
    microstructure_quality: float
    
    timestamp: datetime

@dataclass
class ExitDecision:
    symbol: str
    exit_trigger: ExitTrigger
    exit_price: float
    exit_percentage: float  # What % of position to close (0.0-1.0)
    urgency: float         # 0-100 urgency score
    quality_score: float   # 0-100 quality of exit decision
    
    # Reasoning
    primary_reason: str
    supporting_reasons: List[str]
    risk_factors: List[str]
    
    # Execution details
    execution_method: str  # 'market', 'limit', 'stop_market', etc.
    limit_price: Optional[float]
    time_in_force: str
    
    # Expected outcomes
    expected_slippage: float
    profit_potential_remaining: float
    risk_reduction: float
    
    # Metadata
    confidence: float
    timestamp: datetime

class ExitStrategyAgent:
    """
    Advanced Exit Strategy Agent for dynamic trade management
    
    Features:
    - Multiple exit level management (profit targets, stops, time exits)
    - Dynamic trailing stop adjustment
    - Partial profit-taking strategies
    - Volatility-based exit timing
    - Signal reversal detection
    - Risk-adjusted exit optimization
    - Market microstructure consideration
    - Trade stage-based exit rules
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.active_trades = {}  # {symbol: TradeMonitoring}
        self.exit_history = {}   # {symbol: deque[ExitDecision]}
        self.performance_cache = {}  # Performance tracking per symbol
        
        # Performance tracking
        self.exit_quality_scores = deque(maxlen=100)
        self.profit_capture_ratio = deque(maxlen=100)
        self.exit_timing_accuracy = deque(maxlen=100)
        
        self.logger.info("🚪 Exit Strategy Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            # Basic exit parameters
            'default_profit_target_r': 2.5,      # 2.5R profit target
            'default_stop_loss_r': 1.0,          # 1R stop loss
            'max_risk_per_trade': 0.02,          # 2% max risk
            
            # Trailing stop parameters
            'trailing_stop_activation_r': 1.5,   # Start trailing after 1.5R profit
            'trailing_stop_distance_pct': 0.015, # 1.5% trailing distance
            'trailing_step_size': 0.005,         # 0.5% step size
            
            # Partial profit taking
            'partial_profit_levels': [1.0, 2.0, 3.0],  # R multiples for partial profits
            'partial_profit_percentages': [0.3, 0.3, 0.4],  # % to close at each level
            
            # Time-based exits
            'max_trade_duration_hours': 72,      # 72-hour max trade duration
            'profit_time_decay_hours': 24,       # Start time decay after 24 hours
            'weekend_exit_buffer_hours': 2,      # Exit before weekend
            
            # Volatility exits
            'volatility_spike_threshold': 2.0,   # 2x normal volatility
            'volatility_exit_percentage': 0.5,   # Close 50% on vol spike
            
            # Signal reversal
            'reversal_confidence_threshold': 75, # 75% confidence for reversal exit
            'reversal_exit_percentage': 0.7,     # Close 70% on signal reversal
            
            # Risk management
            'max_adverse_excursion_pct': 0.05,   # 5% max adverse excursion
            'correlation_exit_threshold': 0.8,   # Exit if correlation breaks down
            'liquidity_exit_threshold': 50,      # Exit if liquidity score drops below 50
            
            # Market microstructure
            'spread_deterioration_threshold': 3.0,  # 3x normal spread
            'volume_deterioration_threshold': 0.3,  # 30% of normal volume
            
            # Quality thresholds
            'min_exit_quality': 60,
            'excellent_exit_threshold': 90,
            'emergency_exit_quality': 40,
            
            # Advanced features
            'momentum_exit_sensitivity': 0.7,
            'trend_strength_exit_threshold': 0.3,
            'mean_reversion_exit_factor': 1.2
        }
    
    def monitor_trade(self, symbol: str, entry_data: Dict, current_market_data: pd.DataFrame, 
                     volatility_data: Optional[Dict] = None, signal_data: Optional[Dict] = None) -> TradeMonitoring:
        """
        Comprehensive trade monitoring and exit level management
        """
        try:
            # Initialize trade monitoring if new
            if symbol not in self.active_trades:
                self.active_trades[symbol] = self._initialize_trade_monitoring(symbol, entry_data)
                self.exit_history[symbol] = deque(maxlen=50)
                self.performance_cache[symbol] = {}
            
            trade = self.active_trades[symbol]
            current_price = current_market_data['close'].iloc[-1]
            
            # Step 1: Update basic trade metrics
            self._update_trade_metrics(trade, current_price, current_market_data)
            
            # Step 2: Update trade stage
            self._update_trade_stage(trade)
            
            # Step 3: Calculate risk metrics
            self._update_risk_metrics(trade, current_market_data, volatility_data)
            
            # Step 4: Update market condition assessments
            self._update_market_conditions(trade, current_market_data, signal_data)
            
            # Step 5: Update exit levels dynamically
            self._update_exit_levels(trade, current_market_data, volatility_data)
            
            # Step 6: Check for triggered exits
            triggered_exits = self._check_exit_triggers(trade, current_price)
            trade.triggered_exits.extend(triggered_exits)
            
            # Step 7: Update performance tracking
            self._update_performance_tracking(symbol, trade)
            
            trade.timestamp = datetime.now()
            
            self.logger.debug(f"🚪 Trade monitoring updated: {symbol} - "
                            f"PnL: {trade.unrealized_pnl_pct:.1%}, Stage: {trade.trade_stage.value}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error monitoring trade for {symbol}: {e}")
            return self._default_trade_monitoring(symbol)
    
    def _initialize_trade_monitoring(self, symbol: str, entry_data: Dict) -> TradeMonitoring:
        """Initialize trade monitoring structure"""
        try:
            entry_price = entry_data['entry_price']
            position_size = entry_data['position_size']
            direction = entry_data['direction']
            
            # Create initial exit levels
            initial_exits = self._create_initial_exit_levels(entry_data)
            
            # Expected duration based on signal type
            signal_type = entry_data.get('signal_type', 'breakout')
            expected_hours = {
                'breakout': 24, 'reversal': 48, 'continuation': 36, 'mean_reversion': 12
            }.get(signal_type, 24)
            
            return TradeMonitoring(
                symbol=symbol,
                entry_price=entry_price,
                current_price=entry_price,
                position_size=position_size,
                direction=direction,
                
                # Performance (initialized)
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                max_favorable_excursion=0.0,
                max_adverse_excursion=0.0,
                
                # Progression
                trade_stage=TradeStage.EARLY,
                time_in_trade=timedelta(0),
                expected_duration=timedelta(hours=expected_hours),
                
                # Risk (initialized)
                current_risk=entry_data.get('initial_risk', 0.02),
                risk_adjusted_return=0.0,
                volatility_since_entry=0.0,
                correlation_drift=0.0,
                
                # Exits
                active_exits=initial_exits,
                triggered_exits=[],
                
                # Market conditions (initialized)
                current_trend_strength=0.5,
                momentum_deterioration=0.0,
                volume_pattern_change=0.0,
                microstructure_quality=0.7,
                
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing trade monitoring: {e}")
            return self._default_trade_monitoring(symbol)
    
    def _create_initial_exit_levels(self, entry_data: Dict) -> List[ExitLevel]:
        """Create initial exit levels for the trade"""
        exits = []
        
        try:
            entry_price = entry_data['entry_price']
            direction = entry_data['direction']
            risk_amount = entry_data.get('risk_amount', entry_price * 0.02)
            
            # Calculate R (risk unit)
            r_value = risk_amount
            
            # Stop Loss
            if direction == 'LONG':
                stop_price = entry_price - r_value
            else:
                stop_price = entry_price + r_value
            
            exits.append(ExitLevel(
                level_type=ExitTrigger.STOP_LOSS,
                price=stop_price,
                percentage_of_position=1.0,
                priority=1,
                conditions=["Stop loss hit"],
                time_validity=None,
                is_active=True,
                created_time=datetime.now()
            ))
            
            # Profit Targets
            profit_r = self.config['default_profit_target_r']
            if direction == 'LONG':
                profit_price = entry_price + (r_value * profit_r)
            else:
                profit_price = entry_price - (r_value * profit_r)
            
            exits.append(ExitLevel(
                level_type=ExitTrigger.PROFIT_TARGET,
                price=profit_price,
                percentage_of_position=0.5,  # Take 50% at first target
                priority=3,
                conditions=["Profit target reached"],
                time_validity=None,
                is_active=True,
                created_time=datetime.now()
            ))
            
            # Partial Profit Levels
            for i, (r_multiple, percentage) in enumerate(zip(
                self.config['partial_profit_levels'],
                self.config['partial_profit_percentages']
            )):
                if direction == 'LONG':
                    partial_price = entry_price + (r_value * r_multiple)
                else:
                    partial_price = entry_price - (r_value * r_multiple)
                
                exits.append(ExitLevel(
                    level_type=ExitTrigger.PARTIAL_PROFIT,
                    price=partial_price,
                    percentage_of_position=percentage,
                    priority=4 + i,
                    conditions=[f"Partial profit at {r_multiple}R"],
                    time_validity=None,
                    is_active=True,
                    created_time=datetime.now()
                ))
            
            # Time Stop
            max_duration = timedelta(hours=self.config['max_trade_duration_hours'])
            time_exit_time = datetime.now() + max_duration
            
            exits.append(ExitLevel(
                level_type=ExitTrigger.TIME_STOP,
                price=0,  # Market exit
                percentage_of_position=1.0,
                priority=10,
                conditions=["Time limit reached"],
                time_validity=time_exit_time,
                is_active=True,
                created_time=datetime.now()
            ))
            
            return exits
            
        except Exception as e:
            self.logger.error(f"Error creating initial exit levels: {e}")
            return []
    
    def _update_trade_metrics(self, trade: TradeMonitoring, current_price: float, 
                            market_data: pd.DataFrame):
        """Update basic trade performance metrics"""
        try:
            trade.current_price = current_price
            
            # Calculate P&L
            if trade.direction == 'LONG':
                pnl_per_share = current_price - trade.entry_price
            else:
                pnl_per_share = trade.entry_price - current_price
            
            trade.unrealized_pnl = pnl_per_share * trade.position_size
            trade.unrealized_pnl_pct = pnl_per_share / trade.entry_price
            
            # Update excursions
            if trade.direction == 'LONG':
                favorable_excursion = max(market_data['high'].iloc[-10:].max() - trade.entry_price, 0)
                adverse_excursion = max(trade.entry_price - market_data['low'].iloc[-10:].min(), 0)
            else:
                favorable_excursion = max(trade.entry_price - market_data['low'].iloc[-10:].min(), 0)
                adverse_excursion = max(market_data['high'].iloc[-10:].max() - trade.entry_price, 0)
            
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable_excursion / trade.entry_price)
            trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse_excursion / trade.entry_price)
            
            # Update time in trade
            trade.time_in_trade = datetime.now() - trade.active_exits[0].created_time
            
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {e}")
    
    def _update_trade_stage(self, trade: TradeMonitoring):
        """Update the current stage of the trade"""
        try:
            time_ratio = trade.time_in_trade.total_seconds() / trade.expected_duration.total_seconds()
            
            if time_ratio <= 0.25:
                trade.trade_stage = TradeStage.EARLY
            elif time_ratio <= 0.5:
                trade.trade_stage = TradeStage.DEVELOPING
            elif time_ratio <= 0.75:
                trade.trade_stage = TradeStage.MATURE
            elif time_ratio <= 1.0:
                trade.trade_stage = TradeStage.LATE
            else:
                trade.trade_stage = TradeStage.EXTENDED
                
        except Exception as e:
            self.logger.error(f"Error updating trade stage: {e}")
    
    def _update_risk_metrics(self, trade: TradeMonitoring, market_data: pd.DataFrame, 
                           volatility_data: Optional[Dict]):
        """Update risk-related metrics"""
        try:
            # Volatility since entry
            if len(market_data) >= 10:
                recent_returns = market_data['close'].pct_change().iloc[-10:]
                trade.volatility_since_entry = recent_returns.std() * np.sqrt(252)  # Annualized
            
            # Risk-adjusted return
            if trade.volatility_since_entry > 0:
                trade.risk_adjusted_return = trade.unrealized_pnl_pct / trade.volatility_since_entry
            else:
                trade.risk_adjusted_return = 0
            
            # Update current risk (distance to stop loss)
            stop_loss_exits = [e for e in trade.active_exits if e.level_type == ExitTrigger.STOP_LOSS]
            if stop_loss_exits:
                stop_price = stop_loss_exits[0].price
                if trade.direction == 'LONG':
                    trade.current_risk = (trade.current_price - stop_price) / trade.current_price
                else:
                    trade.current_risk = (stop_price - trade.current_price) / trade.current_price
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    def _update_market_conditions(self, trade: TradeMonitoring, market_data: pd.DataFrame, 
                                signal_data: Optional[Dict]):
        """Update market condition assessments"""
        try:
            # Trend strength
            if len(market_data) >= 20:
                short_ma = market_data['close'].rolling(5).mean().iloc[-1]
                long_ma = market_data['close'].rolling(20).mean().iloc[-1]
                
                trend_strength = abs(short_ma - long_ma) / trade.current_price
                trade.current_trend_strength = min(trend_strength * 10, 1.0)  # Normalize
            
            # Momentum deterioration
            if len(market_data) >= 10:
                recent_momentum = market_data['close'].pct_change().iloc[-5:].mean()
                earlier_momentum = market_data['close'].pct_change().iloc[-10:-5].mean()
                
                if trade.direction == 'LONG':
                    momentum_change = recent_momentum - earlier_momentum
                    trade.momentum_deterioration = max(-momentum_change, 0)
                else:
                    momentum_change = earlier_momentum - recent_momentum
                    trade.momentum_deterioration = max(-momentum_change, 0)
            
            # Volume pattern change
            if 'volume' in market_data.columns and len(market_data) >= 20:
                recent_avg_volume = market_data['volume'].iloc[-5:].mean()
                historical_avg_volume = market_data['volume'].iloc[-20:-5].mean()
                
                if historical_avg_volume > 0:
                    volume_change = (historical_avg_volume - recent_avg_volume) / historical_avg_volume
                    trade.volume_pattern_change = max(volume_change, 0)  # Deterioration only
            
            # Microstructure quality (simplified)
            if len(market_data) >= 5:
                price_range = market_data['high'].iloc[-5:].max() - market_data['low'].iloc[-5:].min()
                avg_range = price_range / 5
                current_spread_proxy = avg_range / trade.current_price
                
                # Lower spread proxy indicates better microstructure
                trade.microstructure_quality = max(0, 1 - current_spread_proxy * 100)
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")
    
    def _update_exit_levels(self, trade: TradeMonitoring, market_data: pd.DataFrame, 
                          volatility_data: Optional[Dict]):
        """Dynamically update exit levels"""
        try:
            current_price = trade.current_price
            
            # Update trailing stops
            self._update_trailing_stops(trade, current_price)
            
            # Add volatility-based exits
            if volatility_data and 'volatility_regime' in volatility_data:
                vol_regime = volatility_data['volatility_regime']
                if vol_regime in ['high', 'extremely_high']:
                    self._add_volatility_exit(trade, volatility_data)
            
            # Add time decay exits as trade matures
            if trade.trade_stage in [TradeStage.MATURE, TradeStage.LATE]:
                self._add_time_decay_exits(trade)
            
            # Remove expired exits
            current_time = datetime.now()
            trade.active_exits = [
                exit_level for exit_level in trade.active_exits
                if exit_level.time_validity is None or exit_level.time_validity > current_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error updating exit levels: {e}")
    
    def _update_trailing_stops(self, trade: TradeMonitoring, current_price: float):
        """Update trailing stop levels"""
        try:
            # Check if we should activate trailing stops
            activation_r = self.config['trailing_stop_activation_r']
            initial_risk = trade.current_risk
            
            if trade.unrealized_pnl_pct > activation_r * initial_risk:
                # Find existing trailing stop
                trailing_stops = [e for e in trade.active_exits if e.level_type == ExitTrigger.TRAILING_STOP]
                
                if not trailing_stops:
                    # Create initial trailing stop
                    trailing_distance = self.config['trailing_stop_distance_pct']
                    
                    if trade.direction == 'LONG':
                        trailing_price = current_price * (1 - trailing_distance)
                    else:
                        trailing_price = current_price * (1 + trailing_distance)
                    
                    new_trailing_stop = ExitLevel(
                        level_type=ExitTrigger.TRAILING_STOP,
                        price=trailing_price,
                        percentage_of_position=1.0,
                        priority=2,
                        conditions=["Trailing stop triggered"],
                        time_validity=None,
                        is_active=True,
                        created_time=datetime.now()
                    )
                    trade.active_exits.append(new_trailing_stop)
                    
                    # Deactivate original stop loss
                    for exit_level in trade.active_exits:
                        if exit_level.level_type == ExitTrigger.STOP_LOSS:
                            exit_level.is_active = False
                
                else:
                    # Update existing trailing stop
                    trailing_stop = trailing_stops[0]
                    trailing_distance = self.config['trailing_stop_distance_pct']
                    
                    if trade.direction == 'LONG':
                        new_trailing_price = current_price * (1 - trailing_distance)
                        if new_trailing_price > trailing_stop.price:
                            trailing_stop.price = new_trailing_price
                    else:
                        new_trailing_price = current_price * (1 + trailing_distance)
                        if new_trailing_price < trailing_stop.price:
                            trailing_stop.price = new_trailing_price
            
        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")
    
    def _add_volatility_exit(self, trade: TradeMonitoring, volatility_data: Dict):
        """Add volatility spike exit"""
        try:
            current_vol = volatility_data.get('realized_volatility', 0.2)
            normal_vol = volatility_data.get('normal_volatility', current_vol)
            
            vol_spike_threshold = self.config['volatility_spike_threshold']
            
            if current_vol > normal_vol * vol_spike_threshold:
                # Check if volatility exit already exists
                vol_exits = [e for e in trade.active_exits if e.level_type == ExitTrigger.VOLATILITY_SPIKE]
                
                if not vol_exits:
                    vol_exit = ExitLevel(
                        level_type=ExitTrigger.VOLATILITY_SPIKE,
                        price=0,  # Market exit
                        percentage_of_position=self.config['volatility_exit_percentage'],
                        priority=5,
                        conditions=["Volatility spike detected"],
                        time_validity=datetime.now() + timedelta(minutes=30),
                        is_active=True,
                        created_time=datetime.now()
                    )
                    trade.active_exits.append(vol_exit)
            
        except Exception as e:
            self.logger.error(f"Error adding volatility exit: {e}")
    
    def _add_time_decay_exits(self, trade: TradeMonitoring):
        """Add time decay exits for mature trades"""
        try:
            if trade.trade_stage == TradeStage.MATURE:
                # Check for time decay exit
                time_exits = [e for e in trade.active_exits if "time_decay" in str(e.conditions)]
                
                if not time_exits and trade.unrealized_pnl_pct > 0:
                    # Add time decay exit at reduced profit
                    time_decay_exit = ExitLevel(
                        level_type=ExitTrigger.RISK_MANAGEMENT,
                        price=0,  # Market exit
                        percentage_of_position=0.5,
                        priority=6,
                        conditions=["Time decay - lock in profits"],
                        time_validity=datetime.now() + timedelta(hours=6),
                        is_active=True,
                        created_time=datetime.now()
                    )
                    trade.active_exits.append(time_decay_exit)
            
        except Exception as e:
            self.logger.error(f"Error adding time decay exits: {e}")
    
    def _check_exit_triggers(self, trade: TradeMonitoring, current_price: float) -> List[ExitLevel]:
        """Check which exit triggers have been hit"""
        triggered_exits = []
        
        try:
            current_time = datetime.now()
            
            for exit_level in trade.active_exits:
                if not exit_level.is_active:
                    continue
                
                triggered = False
                
                if exit_level.level_type == ExitTrigger.TIME_STOP:
                    if exit_level.time_validity and current_time >= exit_level.time_validity:
                        triggered = True
                
                elif exit_level.level_type in [ExitTrigger.STOP_LOSS, ExitTrigger.TRAILING_STOP]:
                    if trade.direction == 'LONG':
                        triggered = current_price <= exit_level.price
                    else:
                        triggered = current_price >= exit_level.price
                
                elif exit_level.level_type in [ExitTrigger.PROFIT_TARGET, ExitTrigger.PARTIAL_PROFIT]:
                    if trade.direction == 'LONG':
                        triggered = current_price >= exit_level.price
                    else:
                        triggered = current_price <= exit_level.price
                
                elif exit_level.level_type == ExitTrigger.VOLATILITY_SPIKE:
                    # Always trigger volatility exits when they're active
                    triggered = True
                
                elif exit_level.level_type == ExitTrigger.RISK_MANAGEMENT:
                    # Risk management exits trigger based on conditions
                    triggered = True
                
                if triggered:
                    triggered_exits.append(exit_level)
                    exit_level.is_active = False  # Deactivate after triggering
            
            return triggered_exits
            
        except Exception as e:
            self.logger.error(f"Error checking exit triggers: {e}")
            return []
    
    def generate_exit_decision(self, symbol: str, triggered_exits: List[ExitLevel], 
                             trade: TradeMonitoring, market_data: pd.DataFrame) -> Optional[ExitDecision]:
        """Generate exit decision based on triggered exits"""
        try:
            if not triggered_exits:
                return None
            
            # Sort by priority (lower number = higher priority)
            triggered_exits.sort(key=lambda x: x.priority)
            primary_exit = triggered_exits[0]
            
            current_price = trade.current_price
            
            # Determine exit percentage (sum of all triggered exits)
            total_exit_percentage = min(sum(e.percentage_of_position for e in triggered_exits), 1.0)
            
            # Calculate quality score
            quality_score = self._calculate_exit_quality(trade, primary_exit)
            
            # Calculate urgency
            urgency = self._calculate_exit_urgency(trade, primary_exit)
            
            # Generate reasoning
            primary_reason = f"{primary_exit.level_type.value.replace('_', ' ').title()} triggered"
            supporting_reasons = [f"{e.level_type.value}" for e in triggered_exits[1:]]
            risk_factors = self._identify_exit_risk_factors(trade, market_data)
            
            # Determine execution method
            execution_method, limit_price = self._determine_execution_method(primary_exit, current_price, urgency)
            
            # Calculate expected outcomes
            expected_slippage = self._estimate_exit_slippage(trade, execution_method)
            profit_potential = self._estimate_remaining_profit_potential(trade)
            risk_reduction = total_exit_percentage * trade.current_risk
            
            decision = ExitDecision(
                symbol=symbol,
                exit_trigger=primary_exit.level_type,
                exit_price=current_price,  # Will be updated with actual execution price
                exit_percentage=total_exit_percentage,
                urgency=urgency,
                quality_score=quality_score,
                
                # Reasoning
                primary_reason=primary_reason,
                supporting_reasons=supporting_reasons,
                risk_factors=risk_factors,
                
                # Execution
                execution_method=execution_method,
                limit_price=limit_price,
                time_in_force="IOC",  # Immediate or Cancel for exits
                
                # Outcomes
                expected_slippage=expected_slippage,
                profit_potential_remaining=profit_potential,
                risk_reduction=risk_reduction,
                
                # Metadata
                confidence=self._calculate_exit_confidence(quality_score, urgency, risk_factors),
                timestamp=datetime.now()
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error generating exit decision: {e}")
            return None
    
    def _calculate_exit_quality(self, trade: TradeMonitoring, primary_exit: ExitLevel) -> float:
        """Calculate the quality score for the exit decision"""
        try:
            quality = 60  # Base quality
            
            # Profit capture quality
            if trade.unrealized_pnl_pct > 0:
                # Compare to max favorable excursion
                if trade.max_favorable_excursion > 0:
                    capture_ratio = trade.unrealized_pnl_pct / trade.max_favorable_excursion
                    quality += capture_ratio * 30
                else:
                    quality += 15  # Bonus for any profit
            
            # Exit type quality
            exit_type_scores = {
                ExitTrigger.PROFIT_TARGET: 25,
                ExitTrigger.PARTIAL_PROFIT: 20,
                ExitTrigger.TRAILING_STOP: 15,
                ExitTrigger.STOP_LOSS: -10,
                ExitTrigger.TIME_STOP: 5,
                ExitTrigger.RISK_MANAGEMENT: 10
            }
            quality += exit_type_scores.get(primary_exit.level_type, 0)
            
            # Trade stage considerations
            stage_scores = {
                TradeStage.EARLY: -5,      # Early exits usually suboptimal
                TradeStage.DEVELOPING: 0,
                TradeStage.MATURE: 10,     # Good time to exit
                TradeStage.LATE: 5,
                TradeStage.EXTENDED: -10   # Should have exited earlier
            }
            quality += stage_scores.get(trade.trade_stage, 0)
            
            # Market conditions
            if trade.momentum_deterioration > 0.02:  # 2% momentum loss
                quality += 15  # Good to exit when momentum fails
            
            if trade.volume_pattern_change > 0.5:  # 50% volume drop
                quality += 10  # Good to exit when volume dries up
            
            return min(max(quality, 0), 100)
            
        except Exception:
            return 60
    
    def _calculate_exit_urgency(self, trade: TradeMonitoring, primary_exit: ExitLevel) -> float:
        """Calculate urgency score for exit execution"""
        try:
            urgency = 50  # Base urgency
            
            # Exit type urgency
            type_urgency = {
                ExitTrigger.STOP_LOSS: 90,
                ExitTrigger.TRAILING_STOP: 85,
                ExitTrigger.VOLATILITY_SPIKE: 95,
                ExitTrigger.LIQUIDITY_DETERIORATION: 80,
                ExitTrigger.RISK_MANAGEMENT: 70,
                ExitTrigger.PROFIT_TARGET: 60,
                ExitTrigger.TIME_STOP: 40
            }
            urgency = type_urgency.get(primary_exit.level_type, urgency)
            
            # Market condition adjustments
            if trade.volatility_since_entry > 0.4:  # High volatility
                urgency += 15
            
            if trade.microstructure_quality < 0.3:  # Poor liquidity
                urgency += 20
            
            # Adverse excursion
            if trade.max_adverse_excursion > 0.03:  # 3% adverse excursion
                urgency += 10
            
            return min(max(urgency, 10), 100)
            
        except Exception:
            return 60
    
    def _identify_exit_risk_factors(self, trade: TradeMonitoring, market_data: pd.DataFrame) -> List[str]:
        """Identify risk factors affecting the exit"""
        risk_factors = []
        
        try:
            # Volatility risk
            if trade.volatility_since_entry > 0.35:
                risk_factors.append("High volatility environment")
            
            # Liquidity risk
            if trade.microstructure_quality < 0.4:
                risk_factors.append("Deteriorating market liquidity")
            
            # Time risk
            if trade.trade_stage == TradeStage.EXTENDED:
                risk_factors.append("Trade duration exceeded expectations")
            
            # Momentum risk
            if trade.momentum_deterioration > 0.03:
                risk_factors.append("Momentum deterioration detected")
            
            # Weekend risk
            current_time = datetime.now()
            if current_time.weekday() >= 4 and current_time.hour >= 14:  # Friday afternoon
                risk_factors.append("Weekend gap risk")
            
            # Volume risk
            if trade.volume_pattern_change > 0.6:
                risk_factors.append("Volume pattern deterioration")
            
            return risk_factors
            
        except Exception:
            return ["Standard exit risks apply"]
    
    def _determine_execution_method(self, exit_level: ExitLevel, current_price: float, urgency: float) -> Tuple[str, Optional[float]]:
        """Determine optimal execution method for exit"""
        try:
            # High urgency = market orders
            if urgency >= 85 or exit_level.level_type in [ExitTrigger.STOP_LOSS, ExitTrigger.VOLATILITY_SPIKE]:
                return "market", None
            
            # Medium urgency = limit orders with small buffer
            elif urgency >= 60:
                buffer = 0.001  # 0.1% buffer
                if exit_level.level_type in [ExitTrigger.PROFIT_TARGET, ExitTrigger.PARTIAL_PROFIT]:
                    limit_price = current_price * (1 - buffer)  # Sell slightly below market
                else:
                    limit_price = current_price * (1 + buffer)  # Buy slightly above market
                return "limit", limit_price
            
            # Low urgency = limit orders at better prices
            else:
                buffer = 0.002  # 0.2% buffer for better price
                if exit_level.level_type in [ExitTrigger.PROFIT_TARGET, ExitTrigger.PARTIAL_PROFIT]:
                    limit_price = current_price * (1 + buffer)  # Try to sell above market
                else:
                    limit_price = current_price * (1 - buffer)  # Try to buy below market
                return "limit", limit_price
                
        except Exception:
            return "market", None
    
    def _estimate_exit_slippage(self, trade: TradeMonitoring, execution_method: str) -> float:
        """Estimate slippage for exit execution"""
        try:
            base_slippage = 0.001  # 0.1% base slippage
            
            # Execution method impact
            if execution_method == "market":
                base_slippage *= 2  # Higher slippage for market orders
            
            # Microstructure impact
            if trade.microstructure_quality < 0.5:
                base_slippage *= (1.5 - trade.microstructure_quality)
            
            # Volatility impact
            vol_multiplier = 1 + trade.volatility_since_entry
            base_slippage *= vol_multiplier
            
            # Position size impact (larger positions have more slippage)
            if trade.position_size > 0.05:  # Large position
                base_slippage *= 1.5
            
            return min(base_slippage, 0.01)  # Cap at 1%
            
        except Exception:
            return 0.002  # Default 0.2%
    
    def _estimate_remaining_profit_potential(self, trade: TradeMonitoring) -> float:
        """Estimate remaining profit potential"""
        try:
            # Simple heuristic: profit potential decreases with trade age and stage
            base_potential = 0.02  # 2% base potential
            
            # Stage-based adjustment
            stage_multipliers = {
                TradeStage.EARLY: 1.2,
                TradeStage.DEVELOPING: 1.0,
                TradeStage.MATURE: 0.7,
                TradeStage.LATE: 0.4,
                TradeStage.EXTENDED: 0.2
            }
            
            potential = base_potential * stage_multipliers.get(trade.trade_stage, 0.5)
            
            # Trend strength adjustment
            potential *= trade.current_trend_strength
            
            # Momentum adjustment
            if trade.momentum_deterioration > 0.02:
                potential *= 0.5  # Reduce potential if momentum is failing
            
            return max(potential, 0)
            
        except Exception:
            return 0.01  # 1% default potential
    
    def _calculate_exit_confidence(self, quality_score: float, urgency: float, risk_factors: List[str]) -> float:
        """Calculate confidence in exit decision"""
        try:
            confidence = 70  # Base confidence
            
            # Quality contribution
            confidence += (quality_score - 60) * 0.3
            
            # Urgency contribution (high urgency = higher confidence for risk management)
            if urgency >= 85:
                confidence += 15
            elif urgency <= 30:
                confidence -= 10
            
            # Risk factor penalty
            confidence -= len(risk_factors) * 5
            
            return min(max(confidence, 30), 95)
            
        except Exception:
            return 60
    
    def execute_exit(self, decision: ExitDecision, actual_exit_price: float) -> Dict[str, Any]:
        """Record exit execution and update performance tracking"""
        try:
            # Calculate actual slippage
            expected_price = decision.exit_price
            actual_slippage = abs(actual_exit_price - expected_price) / expected_price
            
            # Determine execution quality
            slippage_bps = actual_slippage * 10000
            
            if slippage_bps <= 5:
                exec_quality = ExitQuality.EXCELLENT
            elif slippage_bps <= 15:
                exec_quality = ExitQuality.GOOD
            elif slippage_bps <= 30:
                exec_quality = ExitQuality.FAIR
            elif slippage_bps <= 50:
                exec_quality = ExitQuality.POOR
            else:
                exec_quality = ExitQuality.VERY_POOR
            
            # Record execution
            execution_record = {
                'decision': decision,
                'actual_exit_price': actual_exit_price,
                'actual_slippage': actual_slippage,
                'execution_quality': exec_quality,
                'execution_time': datetime.now(),
                'success': exec_quality in [ExitQuality.EXCELLENT, ExitQuality.GOOD, ExitQuality.FAIR]
            }
            
            # Store execution
            if decision.symbol in self.exit_history:
                self.exit_history[decision.symbol].append(decision)
            
            # Update performance tracking
            self.exit_quality_scores.append(decision.quality_score)
            
            # Calculate profit capture ratio if applicable
            symbol = decision.symbol
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                if trade.max_favorable_excursion > 0:
                    capture_ratio = trade.unrealized_pnl_pct / trade.max_favorable_excursion
                    self.profit_capture_ratio.append(max(0, min(capture_ratio, 1.0)))
            
            # Update timing accuracy
            timing_score = 1.0 - (actual_slippage / max(decision.expected_slippage, 0.001))
            self.exit_timing_accuracy.append(max(0, min(timing_score, 1.0)))
            
            return execution_record
            
        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            return {'success': False, 'error': str(e)}
    
    def _default_trade_monitoring(self, symbol: str) -> TradeMonitoring:
        """Return default trade monitoring for error cases"""
        return TradeMonitoring(
            symbol=symbol,
            entry_price=100.0,
            current_price=100.0,
            position_size=0.03,
            direction='LONG',
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            max_favorable_excursion=0.0,
            max_adverse_excursion=0.0,
            trade_stage=TradeStage.EARLY,
            time_in_trade=timedelta(0),
            expected_duration=timedelta(hours=24),
            current_risk=0.02,
            risk_adjusted_return=0.0,
            volatility_since_entry=0.2,
            correlation_drift=0.0,
            active_exits=[],
            triggered_exits=[],
            current_trend_strength=0.5,
            momentum_deterioration=0.0,
            volume_pattern_change=0.0,
            microstructure_quality=0.7,
            timestamp=datetime.now()
        )
    
    def _update_performance_tracking(self, symbol: str, trade: TradeMonitoring):
        """Update performance tracking for the trade"""
        try:
            if symbol not in self.performance_cache:
                self.performance_cache[symbol] = {
                    'peak_pnl': trade.unrealized_pnl_pct,
                    'trough_pnl': trade.unrealized_pnl_pct,
                    'best_exit_opportunity': trade.unrealized_pnl_pct
                }
            
            cache = self.performance_cache[symbol]
            
            # Update peak and trough
            cache['peak_pnl'] = max(cache['peak_pnl'], trade.unrealized_pnl_pct)
            cache['trough_pnl'] = min(cache['trough_pnl'], trade.unrealized_pnl_pct)
            
            # Track best exit opportunity (when conditions were optimal)
            exit_conditions_score = (
                trade.current_trend_strength * 0.3 +
                (1 - trade.momentum_deterioration) * 0.3 +
                trade.microstructure_quality * 0.2 +
                (1 - trade.volume_pattern_change) * 0.2
            )
            
            if exit_conditions_score > 0.8 and trade.unrealized_pnl_pct > 0:
                cache['best_exit_opportunity'] = max(
                    cache['best_exit_opportunity'],
                    trade.unrealized_pnl_pct
                )
            
        except Exception as e:
            self.logger.error(f"Error updating performance tracking: {e}")
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            active_trades_count = len(self.active_trades)
            total_exits = sum(len(exits) for exits in self.exit_history.values())
            
            avg_exit_quality = np.mean(self.exit_quality_scores) if self.exit_quality_scores else 0
            avg_profit_capture = np.mean(self.profit_capture_ratio) if self.profit_capture_ratio else 0
            avg_timing_accuracy = np.mean(self.exit_timing_accuracy) if self.exit_timing_accuracy else 0
            
            return {
                'agent_name': 'Exit Strategy Agent',
                'status': 'active',
                'active_trades': active_trades_count,
                'total_exits_managed': total_exits,
                'avg_exit_quality': f"{avg_exit_quality:.1f}",
                'profit_capture_ratio': f"{avg_profit_capture:.1%}",
                'timing_accuracy': f"{avg_timing_accuracy:.1%}",
                'default_profit_target': f"{self.config['default_profit_target_r']:.1f}R",
                'trailing_activation': f"{self.config['trailing_stop_activation_r']:.1f}R",
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Exit Strategy Agent', 'status': 'error'}
    
    def close_trade(self, symbol: str):
        """Close and archive a trade"""
        try:
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                
                # Archive final performance metrics
                final_performance = {
                    'final_pnl': trade.unrealized_pnl_pct,
                    'max_favorable': trade.max_favorable_excursion,
                    'max_adverse': trade.max_adverse_excursion,
                    'trade_duration': trade.time_in_trade,
                    'final_stage': trade.trade_stage.value
                }
                
                self.performance_cache[symbol]['final_performance'] = final_performance
                
                # Remove from active trades
                del self.active_trades[symbol]
                
                self.logger.info(f"🚪 Trade closed and archived: {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error closing trade {symbol}: {e}")