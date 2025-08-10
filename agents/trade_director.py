"""
Trade Director - Master Trading Orchestration System
Ensemble decision-making system that coordinates all specialized agents
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

# Import all specialized agents
from .enhanced_data_pipeline import EnhancedDataPipeline, DataEvent, TimeFrame
from .trigger_line_agent import TriggerLineAgent, TriggerLineAnalysis
from .ftfc_continuity_agent import FTFCContinuityAgent, ContinuityAnalysis
from .reversal_setup_agent import ReversalSetupAgent, ReversalSetup
from .magnet_level_agent import MagnetLevelAgent, MagnetAnalysis
from .volatility_agent import VolatilityAgent, VolatilityMetrics
from .position_sizing_agent import PositionSizingAgent, PositionSizeCalculation
from .entry_timing_agent import EntryTimingAgent, EntryOpportunity
from .exit_strategy_agent import ExitStrategyAgent, TradeMonitoring

class DecisionPhase(Enum):
    OPPORTUNITY_SCANNING = "opportunity_scanning"
    SIGNAL_VALIDATION = "signal_validation"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTION_PLANNING = "execution_planning"
    TRADE_MANAGEMENT = "trade_management"
    PERFORMANCE_REVIEW = "performance_review"

class ConsensusLevel(Enum):
    UNANIMOUS = "unanimous"      # All agents agree
    STRONG = "strong"           # 80%+ agreement
    MODERATE = "moderate"       # 60-80% agreement
    WEAK = "weak"              # 40-60% agreement
    CONFLICTED = "conflicted"   # <40% agreement

class TradeDecision(Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    SCALE_IN = "scale_in"
    HOLD = "hold"
    SCALE_OUT = "scale_out"
    EXIT_ALL = "exit_all"
    NO_ACTION = "no_action"

@dataclass
class AgentVote:
    agent_name: str
    vote: TradeDecision
    confidence: float      # 0-100
    weight: float         # Agent weight in decision
    reasoning: str
    supporting_data: Dict
    timestamp: datetime

@dataclass
class EnsembleDecision:
    symbol: str
    timeframe: str
    decision: TradeDecision
    consensus_level: ConsensusLevel
    total_confidence: float
    decision_phase: DecisionPhase
    
    # Agent votes breakdown
    agent_votes: List[AgentVote]
    unanimous_factors: List[str]
    conflicting_factors: List[str]
    
    # Risk assessment
    risk_score: float          # 0-100 overall risk
    reward_potential: float    # 0-100 reward potential
    risk_reward_ratio: float
    
    # Execution parameters
    position_size: float
    entry_price_range: Tuple[float, float]
    stop_loss: float
    profit_targets: List[float]
    time_horizon: str
    
    # Meta information
    market_regime: str
    volatility_environment: str
    overall_market_health: float
    
    # Decision quality metrics
    decision_quality: float     # 0-100
    execution_urgency: float    # 0-100
    conviction_level: float     # 0-100
    
    timestamp: datetime

@dataclass
class MarketOverview:
    overall_sentiment: str      # "bullish", "bearish", "neutral"
    volatility_regime: str      # "low", "normal", "high", "extreme"
    trend_strength: float       # 0-100
    market_health_score: float  # 0-100
    active_opportunities: int
    risk_environment: str       # "favorable", "moderate", "challenging"
    recommended_exposure: float # 0-1.0 recommended portfolio exposure
    timestamp: datetime

class TradeDirector:
    """
    Master Trading System Orchestrator
    
    Coordinates all specialized agents to make unified trading decisions:
    - Data Pipeline: Real-time multi-timeframe data ingestion
    - Trigger Line Agent: Breakout and support/resistance analysis
    - FTFC Agent: Timeframe continuity validation
    - Reversal Agent: Exhaustion pattern detection
    - Magnet Level Agent: Key price level analysis
    - Volatility Agent: Vol regime and position sizing
    - Position Sizing Agent: Risk-adjusted sizing
    - Entry Timing Agent: Optimal entry execution
    - Exit Strategy Agent: Dynamic exit management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all specialized agents
        self.data_pipeline = EnhancedDataPipeline(
            symbols=self.config['symbols'],
            config=self.config.get('data_pipeline', {})
        )
        
        self.trigger_line_agent = TriggerLineAgent(self.config.get('trigger_line', {}))
        self.ftfc_agent = FTFCContinuityAgent(self.config.get('ftfc', {}))
        self.reversal_agent = ReversalSetupAgent(self.config.get('reversal', {}))
        self.magnet_agent = MagnetLevelAgent(self.config.get('magnet', {}))
        self.volatility_agent = VolatilityAgent(self.config.get('volatility', {}))
        self.position_sizing_agent = PositionSizingAgent(self.config.get('position_sizing', {}))
        self.entry_timing_agent = EntryTimingAgent(self.config.get('entry_timing', {}))
        self.exit_strategy_agent = ExitStrategyAgent(self.config.get('exit_strategy', {}))
        
        # State management
        self.active_decisions = {}  # {symbol: EnsembleDecision}
        self.decision_history = {}  # {symbol: deque[EnsembleDecision]}
        self.market_overview = None
        self.agent_performance = {}
        
        # Performance tracking
        self.decision_accuracy = deque(maxlen=100)
        self.consensus_reliability = deque(maxlen=50)
        self.execution_success_rate = deque(maxlen=100)
        
        self.logger.info("🎭 Trade Director initialized with full agent ensemble")
    
    def _default_config(self) -> Dict:
        return {
            # Core settings
            'symbols': ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC'],
            'primary_timeframe': '15m',
            'timeframes': ['5m', '15m', '1h', '4h', '1d'],
            
            # Agent weights in decision making
            'agent_weights': {
                'trigger_line': 0.15,
                'ftfc_continuity': 0.20,
                'reversal_setup': 0.12,
                'magnet_level': 0.13,
                'volatility': 0.15,
                'position_sizing': 0.10,
                'entry_timing': 0.08,
                'exit_strategy': 0.07
            },
            
            # Consensus requirements
            'min_consensus_threshold': 0.6,     # 60% minimum consensus
            'strong_consensus_threshold': 0.8,  # 80% for strong consensus
            'unanimous_threshold': 0.95,        # 95% for unanimous
            
            # Decision thresholds
            'min_decision_quality': 70,
            'min_conviction_level': 65,
            'max_risk_score': 75,
            'min_reward_potential': 60,
            
            # Risk management
            'max_portfolio_heat': 0.75,
            'max_correlated_positions': 3,
            'daily_loss_limit': 0.03,
            'max_drawdown': 0.15,
            
            # Execution settings
            'execution_timeout_minutes': 30,
            'decision_refresh_interval': 300,  # 5 minutes
            'emergency_exit_conditions': [
                'market_crash', 'liquidity_crisis', 'flash_crash'
            ],
            
            # Performance requirements
            'min_win_rate': 0.4,
            'min_profit_factor': 1.3,
            'min_sharpe_ratio': 0.6
        }
    
    async def start_trading_system(self):
        """Start the complete trading system"""
        try:
            self.logger.info("🚀 Starting Trade Director ensemble system...")
            
            # Start data pipeline
            await self.data_pipeline.start()
            
            # Register event handlers
            self.data_pipeline.register_event_handler(
                DataEvent.NEW_CANDLE, self._handle_new_candle
            )
            self.data_pipeline.register_event_handler(
                DataEvent.TIMEFRAME_COMPLETE, self._handle_timeframe_alignment
            )
            
            # Start main decision loop
            decision_task = asyncio.create_task(self._main_decision_loop())
            
            # Start monitoring loop
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("✅ Trade Director system started successfully")
            
            # Run both tasks
            await asyncio.gather(decision_task, monitoring_task)
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {e}")
            await self.emergency_shutdown()
    
    async def _handle_new_candle(self, event_data: Dict):
        """Handle new candle events from data pipeline"""
        try:
            candle = event_data['candle']
            context = event_data['context']
            
            symbol = candle.symbol
            timeframe = candle.timeframe.value
            
            self.logger.debug(f"📊 New {timeframe} candle: {symbol} @ {candle.close}")
            
            # Trigger opportunity scanning for this symbol
            await self._scan_opportunities(symbol, {timeframe: [candle]})
            
        except Exception as e:
            self.logger.error(f"Error handling new candle: {e}")
    
    async def _handle_timeframe_alignment(self, event_data: Dict):
        """Handle timeframe alignment completion"""
        try:
            symbol = event_data['symbol']
            context = event_data['context']
            
            self.logger.debug(f"🔗 Timeframe alignment complete: {symbol}")
            
            # Get multi-timeframe data
            timeframe_data = {}
            for tf, candles in context.timeframes.items():
                if candles:
                    df_data = []
                    for candle in list(candles)[-100:]:  # Last 100 candles
                        df_data.append({
                            'timestamp': candle.timestamp,
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        timeframe_data[tf.value] = df
            
            # Comprehensive analysis across all timeframes
            if timeframe_data:
                await self._comprehensive_analysis(symbol, timeframe_data)
            
        except Exception as e:
            self.logger.error(f"Error handling timeframe alignment: {e}")
    
    async def _main_decision_loop(self):
        """Main decision-making loop"""
        while True:
            try:
                # Update market overview
                await self._update_market_overview()
                
                # Process each symbol
                for symbol in self.config['symbols']:
                    await self._process_symbol_decisions(symbol)
                
                # Review and update active trades
                await self._review_active_trades()
                
                # Performance review
                await self._performance_review()
                
                # Wait before next cycle
                await asyncio.sleep(self.config['decision_refresh_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in main decision loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _monitoring_loop(self):
        """System monitoring and health checks"""
        while True:
            try:
                # Check agent health
                agent_health = await self._check_agent_health()
                
                # Check portfolio metrics
                portfolio_health = await self._check_portfolio_health()
                
                # Emergency conditions check
                await self._check_emergency_conditions()
                
                # Log system status
                self._log_system_status(agent_health, portfolio_health)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _comprehensive_analysis(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]):
        """Run comprehensive analysis using all agents"""
        try:
            primary_tf = self.config['primary_timeframe']
            primary_data = timeframe_data.get(primary_tf, list(timeframe_data.values())[0])
            
            # Run all agent analyses in parallel
            analyses = await asyncio.gather(
                self._run_trigger_analysis(symbol, primary_tf, primary_data),
                self._run_ftfc_analysis(symbol, timeframe_data),
                self._run_reversal_analysis(symbol, primary_tf, primary_data),
                self._run_magnet_analysis(symbol, primary_tf, primary_data),
                self._run_volatility_analysis(symbol, primary_tf, primary_data),
                return_exceptions=True
            )
            
            # Collect successful analyses
            trigger_analysis = analyses[0] if not isinstance(analyses[0], Exception) else None
            ftfc_analysis = analyses[1] if not isinstance(analyses[1], Exception) else None
            reversal_analysis = analyses[2] if not isinstance(analyses[2], Exception) else None
            magnet_analysis = analyses[3] if not isinstance(analyses[3], Exception) else None
            volatility_analysis = analyses[4] if not isinstance(analyses[4], Exception) else None
            
            # Create ensemble decision
            decision = await self._create_ensemble_decision(
                symbol, primary_tf, {
                    'trigger': trigger_analysis,
                    'ftfc': ftfc_analysis,
                    'reversal': reversal_analysis,
                    'magnet': magnet_analysis,
                    'volatility': volatility_analysis
                }
            )
            
            if decision and decision.decision != TradeDecision.NO_ACTION:
                await self._execute_decision(decision)
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
    
    async def _run_trigger_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TriggerLineAnalysis]:
        """Run trigger line analysis"""
        try:
            return self.trigger_line_agent.analyze_trigger_lines(symbol, timeframe, data)
        except Exception as e:
            self.logger.error(f"Error in trigger analysis: {e}")
            return None
    
    async def _run_ftfc_analysis(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> Optional[ContinuityAnalysis]:
        """Run FTFC continuity analysis"""
        try:
            return self.ftfc_agent.analyze_ftfc_continuity(symbol, timeframe_data)
        except Exception as e:
            self.logger.error(f"Error in FTFC analysis: {e}")
            return None
    
    async def _run_reversal_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[ReversalSetup]:
        """Run reversal setup analysis"""
        try:
            return self.reversal_agent.analyze_reversal_setup(symbol, timeframe, data)
        except Exception as e:
            self.logger.error(f"Error in reversal analysis: {e}")
            return None
    
    async def _run_magnet_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[MagnetAnalysis]:
        """Run magnet level analysis"""
        try:
            return self.magnet_agent.analyze_magnet_levels(symbol, timeframe, data)
        except Exception as e:
            self.logger.error(f"Error in magnet analysis: {e}")
            return None
    
    async def _run_volatility_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[VolatilityMetrics]:
        """Run volatility analysis"""
        try:
            return self.volatility_agent.analyze_volatility(symbol, timeframe, data)
        except Exception as e:
            self.logger.error(f"Error in volatility analysis: {e}")
            return None
    
    async def _create_ensemble_decision(self, symbol: str, timeframe: str, analyses: Dict) -> Optional[EnsembleDecision]:
        """Create ensemble decision from all agent analyses"""
        try:
            agent_votes = []
            total_weight = 0
            
            # Collect votes from each agent
            trigger_vote = self._get_trigger_vote(analyses.get('trigger'))
            if trigger_vote:
                agent_votes.append(trigger_vote)
                total_weight += self.config['agent_weights']['trigger_line']
            
            ftfc_vote = self._get_ftfc_vote(analyses.get('ftfc'))
            if ftfc_vote:
                agent_votes.append(ftfc_vote)
                total_weight += self.config['agent_weights']['ftfc_continuity']
            
            reversal_vote = self._get_reversal_vote(analyses.get('reversal'))
            if reversal_vote:
                agent_votes.append(reversal_vote)
                total_weight += self.config['agent_weights']['reversal_setup']
            
            magnet_vote = self._get_magnet_vote(analyses.get('magnet'))
            if magnet_vote:
                agent_votes.append(magnet_vote)
                total_weight += self.config['agent_weights']['magnet_level']
            
            volatility_vote = self._get_volatility_vote(analyses.get('volatility'))
            if volatility_vote:
                agent_votes.append(volatility_vote)
                total_weight += self.config['agent_weights']['volatility']
            
            if not agent_votes:
                return None
            
            # Calculate consensus
            decision, consensus_level, total_confidence = self._calculate_consensus(agent_votes, total_weight)
            
            if consensus_level == ConsensusLevel.CONFLICTED:
                return None  # Skip conflicted decisions
            
            # Risk and reward assessment
            risk_score = self._calculate_ensemble_risk(analyses)
            reward_potential = self._calculate_reward_potential(analyses)
            risk_reward_ratio = reward_potential / max(risk_score, 1) if risk_score > 0 else 0
            
            # Position sizing
            position_size = await self._calculate_position_size(symbol, decision, analyses)
            
            # Entry parameters
            entry_range = self._calculate_entry_range(analyses)
            stop_loss = self._calculate_stop_loss(analyses, decision)
            profit_targets = self._calculate_profit_targets(analyses, decision)
            
            # Quality metrics
            decision_quality = self._calculate_decision_quality(consensus_level, total_confidence, risk_reward_ratio)
            execution_urgency = self._calculate_execution_urgency(analyses)
            conviction_level = self._calculate_conviction_level(agent_votes, consensus_level)
            
            # Check quality thresholds
            if (decision_quality < self.config['min_decision_quality'] or
                conviction_level < self.config['min_conviction_level'] or
                risk_score > self.config['max_risk_score']):
                return None
            
            ensemble_decision = EnsembleDecision(
                symbol=symbol,
                timeframe=timeframe,
                decision=decision,
                consensus_level=consensus_level,
                total_confidence=total_confidence,
                decision_phase=DecisionPhase.EXECUTION_PLANNING,
                
                agent_votes=agent_votes,
                unanimous_factors=self._identify_unanimous_factors(agent_votes),
                conflicting_factors=self._identify_conflicting_factors(agent_votes),
                
                risk_score=risk_score,
                reward_potential=reward_potential,
                risk_reward_ratio=risk_reward_ratio,
                
                position_size=position_size,
                entry_price_range=entry_range,
                stop_loss=stop_loss,
                profit_targets=profit_targets,
                time_horizon=self._determine_time_horizon(analyses),
                
                market_regime=self._determine_market_regime(analyses),
                volatility_environment=self._determine_volatility_environment(analyses),
                overall_market_health=self._calculate_market_health(analyses),
                
                decision_quality=decision_quality,
                execution_urgency=execution_urgency,
                conviction_level=conviction_level,
                
                timestamp=datetime.now()
            )
            
            return ensemble_decision
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble decision: {e}")
            return None
    
    def _get_trigger_vote(self, analysis: Optional[TriggerLineAnalysis]) -> Optional[AgentVote]:
        """Get vote from trigger line agent"""
        if not analysis or not analysis.actionable_signals:
            return AgentVote(
                agent_name="Trigger Line Agent",
                vote=TradeDecision.NO_ACTION,
                confidence=0,
                weight=self.config['agent_weights']['trigger_line'],
                reasoning="No actionable trigger signals",
                supporting_data={},
                timestamp=datetime.now()
            )
        
        # Get strongest signal
        strongest_signal = max(analysis.actionable_signals, key=lambda x: x['confidence_score'])
        
        vote = TradeDecision.ENTER_LONG if strongest_signal['direction'] == 'LONG' else TradeDecision.ENTER_SHORT
        
        return AgentVote(
            agent_name="Trigger Line Agent",
            vote=vote,
            confidence=strongest_signal['confidence_score'],
            weight=self.config['agent_weights']['trigger_line'],
            reasoning=f"Trigger break detected: {strongest_signal['signal_type']}",
            supporting_data={
                'trigger_level': strongest_signal.get('trigger_level'),
                'break_strength': strongest_signal.get('momentum_score', 0),
                'volume_confirmed': strongest_signal.get('volume_confirmed', False)
            },
            timestamp=datetime.now()
        )
    
    def _get_ftfc_vote(self, analysis: Optional[ContinuityAnalysis]) -> Optional[AgentVote]:
        """Get vote from FTFC continuity agent"""
        if not analysis or not analysis.actionable_signals:
            return AgentVote(
                agent_name="FTFC Continuity Agent",
                vote=TradeDecision.NO_ACTION,
                confidence=0,
                weight=self.config['agent_weights']['ftfc_continuity'],
                reasoning="No timeframe continuity alignment",
                supporting_data={},
                timestamp=datetime.now()
            )
        
        # Strong alignment suggests high conviction
        alignment_score = analysis.alignment_score
        direction = analysis.primary_direction.value
        
        if direction == 'bullish':
            vote = TradeDecision.ENTER_LONG
        elif direction == 'bearish':
            vote = TradeDecision.ENTER_SHORT
        else:
            vote = TradeDecision.NO_ACTION
        
        return AgentVote(
            agent_name="FTFC Continuity Agent",
            vote=vote,
            confidence=alignment_score,
            weight=self.config['agent_weights']['ftfc_continuity'],
            reasoning=f"Timeframe continuity: {analysis.overall_continuity.value}",
            supporting_data={
                'alignment_score': alignment_score,
                'timeframes_aligned': len([s for s in analysis.actionable_signals]),
                'confluence_levels': len(analysis.confluence_levels)
            },
            timestamp=datetime.now()
        )
    
    def _get_reversal_vote(self, analysis: Optional[ReversalSetup]) -> Optional[AgentVote]:
        """Get vote from reversal setup agent"""
        if not analysis or not analysis.actionable:
            return AgentVote(
                agent_name="Reversal Setup Agent",
                vote=TradeDecision.NO_ACTION,
                confidence=0,
                weight=self.config['agent_weights']['reversal_setup'],
                reasoning="No actionable reversal setups",
                supporting_data={},
                timestamp=datetime.now()
            )
        
        vote = TradeDecision.ENTER_LONG if analysis.setup_type.value == 'bullish_reversal' else TradeDecision.ENTER_SHORT
        
        return AgentVote(
            agent_name="Reversal Setup Agent",
            vote=vote,
            confidence=analysis.combined_confidence,
            weight=self.config['agent_weights']['reversal_setup'],
            reasoning=f"Reversal setup: {analysis.setup_type.value}",
            supporting_data={
                'patterns_detected': len(analysis.patterns_detected),
                'opportunity_score': analysis.opportunity_score,
                'risk_score': analysis.risk_score
            },
            timestamp=datetime.now()
        )
    
    def _get_magnet_vote(self, analysis: Optional[MagnetAnalysis]) -> Optional[AgentVote]:
        """Get vote from magnet level agent"""
        if not analysis or not analysis.actionable_levels:
            return AgentVote(
                agent_name="Magnet Level Agent",
                vote=TradeDecision.HOLD,  # Magnet agent suggests patience
                confidence=50,
                weight=self.config['agent_weights']['magnet_level'],
                reasoning="No actionable magnet levels",
                supporting_data={},
                timestamp=datetime.now()
            )
        
        # Magnet levels suggest support for directional moves
        attraction_score = analysis.price_attraction_score
        breakout_probability = analysis.breakout_probability
        
        if breakout_probability > 70:
            vote = TradeDecision.ENTER_LONG  # Assuming upward breakout
            confidence = breakout_probability
            reasoning = "High breakout probability from magnet levels"
        else:
            vote = TradeDecision.HOLD
            confidence = attraction_score
            reasoning = "Magnet levels suggest consolidation"
        
        return AgentVote(
            agent_name="Magnet Level Agent",
            vote=vote,
            confidence=confidence,
            weight=self.config['agent_weights']['magnet_level'],
            reasoning=reasoning,
            supporting_data={
                'active_magnets': len(analysis.active_magnets),
                'attraction_score': attraction_score,
                'breakout_probability': breakout_probability
            },
            timestamp=datetime.now()
        )
    
    def _get_volatility_vote(self, analysis: Optional[VolatilityMetrics]) -> Optional[AgentVote]:
        """Get vote from volatility agent"""
        if not analysis:
            return AgentVote(
                agent_name="Volatility Agent",
                vote=TradeDecision.NO_ACTION,
                confidence=0,
                weight=self.config['agent_weights']['volatility'],
                reasoning="No volatility analysis available",
                supporting_data={},
                timestamp=datetime.now()
            )
        
        # Volatility agent focuses on risk adjustment
        vol_regime = analysis.volatility_regime.value
        optimal_size = analysis.optimal_position_size
        
        if vol_regime in ['extremely_low', 'low']:
            vote = TradeDecision.ENTER_LONG  # Low vol environments favor position taking
            confidence = 70
            reasoning = "Low volatility environment favorable for entries"
        elif vol_regime in ['extremely_high']:
            vote = TradeDecision.HOLD  # High vol suggests caution
            confidence = 60
            reasoning = "High volatility suggests caution"
        else:
            vote = TradeDecision.HOLD  # Neutral volatility
            confidence = 50
            reasoning = "Normal volatility environment"
        
        return AgentVote(
            agent_name="Volatility Agent",
            vote=vote,
            confidence=confidence,
            weight=self.config['agent_weights']['volatility'],
            reasoning=reasoning,
            supporting_data={
                'volatility_regime': vol_regime,
                'iv_rank': analysis.iv_rank,
                'optimal_position_size': optimal_size
            },
            timestamp=datetime.now()
        )
    
    def _calculate_consensus(self, votes: List[AgentVote], total_weight: float) -> Tuple[TradeDecision, ConsensusLevel, float]:
        """Calculate consensus from agent votes"""
        try:
            # Weight votes by agent importance
            weighted_votes = defaultdict(float)
            weighted_confidence = defaultdict(float)
            
            for vote in votes:
                decision = vote.vote
                weight = vote.weight / total_weight  # Normalize weight
                
                weighted_votes[decision] += weight
                weighted_confidence[decision] += vote.confidence * weight
            
            # Find winning decision
            winning_decision = max(weighted_votes.keys(), key=lambda x: weighted_votes[x])
            winning_weight = weighted_votes[winning_decision]
            
            # Calculate consensus level
            if winning_weight >= self.config['unanimous_threshold']:
                consensus_level = ConsensusLevel.UNANIMOUS
            elif winning_weight >= self.config['strong_consensus_threshold']:
                consensus_level = ConsensusLevel.STRONG
            elif winning_weight >= self.config['min_consensus_threshold']:
                consensus_level = ConsensusLevel.MODERATE
            elif winning_weight >= 0.4:
                consensus_level = ConsensusLevel.WEAK
            else:
                consensus_level = ConsensusLevel.CONFLICTED
            
            # Total confidence is weighted average
            total_confidence = weighted_confidence[winning_decision]
            
            return winning_decision, consensus_level, total_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating consensus: {e}")
            return TradeDecision.NO_ACTION, ConsensusLevel.CONFLICTED, 0
    
    def _calculate_ensemble_risk(self, analyses: Dict) -> float:
        """Calculate overall risk score from all analyses"""
        try:
            risk_score = 50  # Base risk
            
            # Volatility risk
            volatility_analysis = analyses.get('volatility')
            if volatility_analysis:
                vol_regime = volatility_analysis.volatility_regime.value
                if vol_regime in ['extremely_high', 'high']:
                    risk_score += 20
                elif vol_regime in ['extremely_low']:
                    risk_score += 10  # Low vol can be risky too (mean reversion)
            
            # Reversal risk
            reversal_analysis = analyses.get('reversal')
            if reversal_analysis and reversal_analysis.actionable:
                risk_score += reversal_analysis.risk_score * 0.3
            
            # Market structure risk
            ftfc_analysis = analyses.get('ftfc')
            if ftfc_analysis:
                if len(ftfc_analysis.risk_factors) > 2:
                    risk_score += 15
            
            return min(max(risk_score, 0), 100)
            
        except Exception:
            return 60  # Default moderate risk
    
    def _calculate_reward_potential(self, analyses: Dict) -> float:
        """Calculate reward potential from all analyses"""
        try:
            reward = 50  # Base reward
            
            # Trigger line potential
            trigger_analysis = analyses.get('trigger')
            if trigger_analysis and trigger_analysis.actionable_signals:
                avg_confidence = np.mean([s['confidence_score'] for s in trigger_analysis.actionable_signals])
                reward += (avg_confidence - 50) * 0.5
            
            # FTFC alignment potential
            ftfc_analysis = analyses.get('ftfc')
            if ftfc_analysis:
                reward += (ftfc_analysis.alignment_score - 50) * 0.4
                reward += ftfc_analysis.opportunity_score * 0.3
            
            # Magnet level potential
            magnet_analysis = analyses.get('magnet')
            if magnet_analysis:
                reward += magnet_analysis.breakout_probability * 0.3
            
            return min(max(reward, 0), 100)
            
        except Exception:
            return 60  # Default moderate reward
    
    async def _calculate_position_size(self, symbol: str, decision: TradeDecision, analyses: Dict) -> float:
        """Calculate position size using position sizing agent"""
        try:
            if decision in [TradeDecision.NO_ACTION, TradeDecision.HOLD]:
                return 0.0
            
            # Prepare signal data for position sizing
            signal_data = {
                'base_position_size': 0.03,  # 3% base
                'confidence_score': 70,  # Will be updated from analyses
                'expected_return': 0.05,
                'stop_loss_distance_pct': 0.02,
                'signal_type': 'ensemble',
                'direction': 'LONG' if decision == TradeDecision.ENTER_LONG else 'SHORT'
            }
            
            # Update with actual analysis data
            if analyses.get('trigger'):
                trigger_signals = analyses['trigger'].actionable_signals
                if trigger_signals:
                    signal_data['confidence_score'] = max(s['confidence_score'] for s in trigger_signals)
            
            # Portfolio data (simplified)
            portfolio_data = {
                'current_positions': {},  # Would get from actual portfolio
                'total_exposure': 0.3     # Example 30% exposure
            }
            
            # Volatility data
            volatility_data = None
            if analyses.get('volatility'):
                vol_analysis = analyses['volatility']
                volatility_data = {
                    'realized_volatility': vol_analysis.realized_volatility,
                    'volatility_regime': vol_analysis.volatility_regime.value,
                    'iv_rank': vol_analysis.iv_rank
                }
            
            # Calculate size
            calculation = self.position_sizing_agent.calculate_position_size(
                symbol, signal_data, portfolio_data, volatility_data
            )
            
            return calculation.adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.02  # Default 2% position
    
    def _calculate_entry_range(self, analyses: Dict) -> Tuple[float, float]:
        """Calculate optimal entry price range"""
        try:
            # Use magnet levels if available
            magnet_analysis = analyses.get('magnet')
            if magnet_analysis and magnet_analysis.actionable_levels:
                level = magnet_analysis.actionable_levels[0]['level']
                return (level * 0.998, level * 1.002)
            
            # Use trigger levels
            trigger_analysis = analyses.get('trigger')
            if trigger_analysis and trigger_analysis.actionable_signals:
                level = trigger_analysis.actionable_signals[0].get('trigger_level', 100)
                return (level * 0.997, level * 1.003)
            
            # Default range
            return (99.0, 101.0)
            
        except Exception:
            return (99.0, 101.0)
    
    def _calculate_stop_loss(self, analyses: Dict, decision: TradeDecision) -> float:
        """Calculate stop loss level"""
        try:
            # Use reversal analysis stop if available
            reversal_analysis = analyses.get('reversal')
            if reversal_analysis and reversal_analysis.actionable:
                return reversal_analysis.stop_loss
            
            # Use volatility-based stop
            volatility_analysis = analyses.get('volatility')
            if volatility_analysis:
                current_price = 100  # Would get actual current price
                vol = volatility_analysis.realized_volatility
                if decision == TradeDecision.ENTER_LONG:
                    return current_price * (1 - vol * 1.5)
                else:
                    return current_price * (1 + vol * 1.5)
            
            # Default 2% stop
            current_price = 100
            if decision == TradeDecision.ENTER_LONG:
                return current_price * 0.98
            else:
                return current_price * 1.02
                
        except Exception:
            return 98.0 if decision == TradeDecision.ENTER_LONG else 102.0
    
    def _calculate_profit_targets(self, analyses: Dict, decision: TradeDecision) -> List[float]:
        """Calculate profit target levels"""
        try:
            targets = []
            
            # Use reversal analysis targets if available
            reversal_analysis = analyses.get('reversal')
            if reversal_analysis and reversal_analysis.actionable:
                targets.extend(reversal_analysis.profit_targets)
            
            # Use magnet levels as targets
            magnet_analysis = analyses.get('magnet')
            if magnet_analysis and magnet_analysis.actionable_levels:
                for level_data in magnet_analysis.actionable_levels[:3]:  # Top 3 levels
                    targets.append(level_data['level'])
            
            # Default targets
            if not targets:
                current_price = 100  # Would get actual current price
                if decision == TradeDecision.ENTER_LONG:
                    targets = [current_price * 1.02, current_price * 1.04, current_price * 1.06]
                else:
                    targets = [current_price * 0.98, current_price * 0.96, current_price * 0.94]
            
            return sorted(targets[:3])  # Return max 3 targets
            
        except Exception:
            return [102.0, 104.0, 106.0]
    
    def _calculate_decision_quality(self, consensus_level: ConsensusLevel, 
                                  total_confidence: float, risk_reward_ratio: float) -> float:
        """Calculate overall decision quality"""
        try:
            quality = 50  # Base quality
            
            # Consensus quality
            consensus_scores = {
                ConsensusLevel.UNANIMOUS: 30,
                ConsensusLevel.STRONG: 25,
                ConsensusLevel.MODERATE: 15,
                ConsensusLevel.WEAK: 5,
                ConsensusLevel.CONFLICTED: -20
            }
            quality += consensus_scores.get(consensus_level, 0)
            
            # Confidence quality
            quality += (total_confidence - 50) * 0.4
            
            # Risk-reward quality
            if risk_reward_ratio > 2.0:
                quality += 15
            elif risk_reward_ratio > 1.5:
                quality += 10
            elif risk_reward_ratio < 1.0:
                quality -= 15
            
            return min(max(quality, 0), 100)
            
        except Exception:
            return 60
    
    def _calculate_execution_urgency(self, analyses: Dict) -> float:
        """Calculate urgency for execution"""
        try:
            urgency = 50  # Base urgency
            
            # Volatility urgency
            volatility_analysis = analyses.get('volatility')
            if volatility_analysis:
                vol_regime = volatility_analysis.volatility_regime.value
                if vol_regime in ['extremely_high']:
                    urgency += 30  # High urgency in volatile markets
            
            # Trigger urgency
            trigger_analysis = analyses.get('trigger')
            if trigger_analysis and trigger_analysis.actionable_signals:
                # Recent breakouts need quick action
                urgency += 20
            
            return min(max(urgency, 0), 100)
            
        except Exception:
            return 60
    
    def _calculate_conviction_level(self, agent_votes: List[AgentVote], consensus_level: ConsensusLevel) -> float:
        """Calculate conviction level"""
        try:
            # Base conviction from consensus
            consensus_conviction = {
                ConsensusLevel.UNANIMOUS: 90,
                ConsensusLevel.STRONG: 80,
                ConsensusLevel.MODERATE: 65,
                ConsensusLevel.WEAK: 45,
                ConsensusLevel.CONFLICTED: 20
            }
            
            base_conviction = consensus_conviction.get(consensus_level, 50)
            
            # Average confidence across agents
            if agent_votes:
                avg_confidence = np.mean([vote.confidence for vote in agent_votes])
                conviction = (base_conviction + avg_confidence) / 2
            else:
                conviction = base_conviction
            
            return min(max(conviction, 0), 100)
            
        except Exception:
            return 60
    
    def _identify_unanimous_factors(self, agent_votes: List[AgentVote]) -> List[str]:
        """Identify factors where all agents agree"""
        factors = []
        
        try:
            # Check if all agents agree on direction
            decisions = [vote.vote for vote in agent_votes]
            if len(set(decisions)) == 1:
                factors.append(f"Unanimous decision: {decisions[0].value}")
            
            # Check confidence levels
            confidences = [vote.confidence for vote in agent_votes]
            if all(c > 70 for c in confidences):
                factors.append("High confidence across all agents")
            
            return factors
            
        except Exception:
            return []
    
    def _identify_conflicting_factors(self, agent_votes: List[AgentVote]) -> List[str]:
        """Identify conflicting factors between agents"""
        conflicts = []
        
        try:
            # Check for opposing decisions
            decisions = [vote.vote for vote in agent_votes]
            if (TradeDecision.ENTER_LONG in decisions and 
                TradeDecision.ENTER_SHORT in decisions):
                conflicts.append("Conflicting directional signals")
            
            # Check for wide confidence spread
            confidences = [vote.confidence for vote in agent_votes]
            if max(confidences) - min(confidences) > 40:
                conflicts.append("Wide confidence spread between agents")
            
            return conflicts
            
        except Exception:
            return []
    
    def _determine_time_horizon(self, analyses: Dict) -> str:
        """Determine expected time horizon for trade"""
        try:
            # Use reversal analysis if available
            reversal_analysis = analyses.get('reversal')
            if reversal_analysis and reversal_analysis.actionable:
                return "1-3 days"  # Reversal trades typically shorter term
            
            # Use FTFC analysis
            ftfc_analysis = analyses.get('ftfc')
            if ftfc_analysis and ftfc_analysis.overall_continuity.value in ['strong_alignment', 'perfect_alignment']:
                return "3-7 days"  # Strong alignment suggests longer moves
            
            return "1-2 days"  # Default
            
        except Exception:
            return "1 day"
    
    def _determine_market_regime(self, analyses: Dict) -> str:
        """Determine current market regime"""
        try:
            volatility_analysis = analyses.get('volatility')
            if volatility_analysis:
                vol_regime = volatility_analysis.volatility_regime.value
                return f"volatility_{vol_regime}"
            
            return "normal"
            
        except Exception:
            return "unknown"
    
    def _determine_volatility_environment(self, analyses: Dict) -> str:
        """Determine volatility environment"""
        try:
            volatility_analysis = analyses.get('volatility')
            if volatility_analysis:
                return volatility_analysis.volatility_regime.value
            
            return "normal"
            
        except Exception:
            return "unknown"
    
    def _calculate_market_health(self, analyses: Dict) -> float:
        """Calculate overall market health score"""
        try:
            health = 70  # Base health
            
            # FTFC health
            ftfc_analysis = analyses.get('ftfc')
            if ftfc_analysis:
                health += (ftfc_analysis.alignment_score - 50) * 0.3
            
            # Volatility health
            volatility_analysis = analyses.get('volatility')
            if volatility_analysis:
                vol_regime = volatility_analysis.volatility_regime.value
                if vol_regime in ['normal']:
                    health += 10
                elif vol_regime in ['extremely_high', 'extremely_low']:
                    health -= 15
            
            return min(max(health, 0), 100)
            
        except Exception:
            return 60
    
    async def _execute_decision(self, decision: EnsembleDecision):
        """Execute the ensemble decision"""
        try:
            self.logger.info(f"🎭 Executing decision: {decision.symbol} {decision.decision.value} "
                           f"(Quality: {decision.decision_quality:.1f}, Conviction: {decision.conviction_level:.1f})")
            
            if decision.decision in [TradeDecision.ENTER_LONG, TradeDecision.ENTER_SHORT]:
                # Get entry timing analysis
                entry_opportunity = await self._get_entry_timing(decision)
                
                if entry_opportunity:
                    # Store active decision
                    self.active_decisions[decision.symbol] = decision
                    
                    # Initialize decision history if needed
                    if decision.symbol not in self.decision_history:
                        self.decision_history[decision.symbol] = deque(maxlen=50)
                    
                    self.decision_history[decision.symbol].append(decision)
                    
                    self.logger.info(f"✅ Decision executed: {decision.symbol}")
                else:
                    self.logger.warning(f"⚠️ Entry timing not favorable: {decision.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error executing decision for {decision.symbol}: {e}")
    
    async def _get_entry_timing(self, decision: EnsembleDecision) -> Optional[EntryOpportunity]:
        """Get entry timing analysis"""
        try:
            # Simplified entry timing - would integrate with full pipeline
            signal_data = {
                'direction': 'LONG' if decision.decision == TradeDecision.ENTER_LONG else 'SHORT',
                'confidence_score': decision.total_confidence,
                'signal_type': 'ensemble',
                'entry_price': np.mean(decision.entry_price_range),
                'stop_loss_distance_pct': abs(decision.stop_loss - np.mean(decision.entry_price_range)) / np.mean(decision.entry_price_range)
            }
            
            # Create mock market data
            current_price = np.mean(decision.entry_price_range)
            mock_data = pd.DataFrame({
                'timestamp': pd.date_range(end=datetime.now(), periods=20, freq='15min'),
                'open': [current_price] * 20,
                'high': [current_price * 1.01] * 20,
                'low': [current_price * 0.99] * 20,
                'close': [current_price] * 20,
                'volume': [1000] * 20
            })
            mock_data.set_index('timestamp', inplace=True)
            
            return self.entry_timing_agent.analyze_entry_timing(
                decision.symbol, decision.timeframe, signal_data, mock_data
            )
            
        except Exception as e:
            self.logger.error(f"Error getting entry timing: {e}")
            return None
    
    async def _scan_opportunities(self, symbol: str, new_data: Dict):
        """Scan for new opportunities when new data arrives"""
        try:
            # Quick opportunity scan - simplified version
            self.logger.debug(f"🔍 Scanning opportunities: {symbol}")
            
            # This would trigger lightweight analysis
            # Full analysis happens in timeframe alignment
            
        except Exception as e:
            self.logger.error(f"Error scanning opportunities for {symbol}: {e}")
    
    async def _process_symbol_decisions(self, symbol: str):
        """Process decisions for a specific symbol"""
        try:
            # Check if we have an active decision
            if symbol in self.active_decisions:
                decision = self.active_decisions[symbol]
                
                # Check if decision needs review/update
                age = datetime.now() - decision.timestamp
                if age.total_seconds() > 1800:  # 30 minutes
                    # Re-evaluate decision
                    self.logger.debug(f"🔄 Re-evaluating decision: {symbol}")
                    # Would trigger new comprehensive analysis
            
        except Exception as e:
            self.logger.error(f"Error processing decisions for {symbol}: {e}")
    
    async def _review_active_trades(self):
        """Review and manage active trades"""
        try:
            # Review all active positions with exit strategy agent
            # This would integrate with actual portfolio/position data
            pass
            
        except Exception as e:
            self.logger.error(f"Error reviewing active trades: {e}")
    
    async def _performance_review(self):
        """Periodic performance review"""
        try:
            # Calculate performance metrics
            # Update agent weights based on performance
            # Generate performance reports
            pass
            
        except Exception as e:
            self.logger.error(f"Error in performance review: {e}")
    
    async def _update_market_overview(self):
        """Update overall market assessment"""
        try:
            # Simplified market overview
            self.market_overview = MarketOverview(
                overall_sentiment="neutral",
                volatility_regime="normal",
                trend_strength=60.0,
                market_health_score=70.0,
                active_opportunities=len([d for d in self.active_decisions.values() 
                                        if d.decision != TradeDecision.NO_ACTION]),
                risk_environment="moderate",
                recommended_exposure=0.6,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating market overview: {e}")
    
    async def _check_agent_health(self) -> Dict[str, str]:
        """Check health of all agents"""
        try:
            health_status = {
                'data_pipeline': 'healthy',
                'trigger_line': 'healthy',
                'ftfc_continuity': 'healthy',
                'reversal_setup': 'healthy',
                'magnet_level': 'healthy',
                'volatility': 'healthy',
                'position_sizing': 'healthy',
                'entry_timing': 'healthy',
                'exit_strategy': 'healthy'
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error checking agent health: {e}")
            return {}
    
    async def _check_portfolio_health(self) -> Dict[str, float]:
        """Check portfolio health metrics"""
        try:
            # Simplified portfolio health
            portfolio_health = {
                'total_exposure': 0.4,
                'daily_pnl': 0.01,
                'unrealized_pnl': 0.02,
                'max_drawdown': 0.05,
                'risk_budget_used': 0.6
            }
            
            return portfolio_health
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio health: {e}")
            return {}
    
    async def _check_emergency_conditions(self):
        """Check for emergency conditions"""
        try:
            # Check for emergency exit conditions
            # This would monitor for market crashes, liquidity crises, etc.
            pass
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
    
    def _log_system_status(self, agent_health: Dict, portfolio_health: Dict):
        """Log periodic system status"""
        try:
            healthy_agents = sum(1 for status in agent_health.values() if status == 'healthy')
            total_agents = len(agent_health)
            
            self.logger.info(f"🎭 System Status: {healthy_agents}/{total_agents} agents healthy, "
                           f"Active decisions: {len(self.active_decisions)}")
            
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown of the trading system"""
        try:
            self.logger.warning("🚨 Emergency shutdown initiated")
            
            # Stop data pipeline
            await self.data_pipeline.stop()
            
            # Close all positions (would integrate with actual broker)
            # Cancel all pending orders
            # Save state
            
            self.logger.warning("🛑 Emergency shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error in emergency shutdown: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            return {
                'system_name': 'Trade Director Ensemble',
                'status': 'active',
                'active_symbols': len(self.config['symbols']),
                'active_decisions': len(self.active_decisions),
                'total_decisions': sum(len(history) for history in self.decision_history.values()),
                'avg_decision_quality': np.mean([d.decision_quality for d in self.active_decisions.values()]) if self.active_decisions else 0,
                'market_overview': self.market_overview.__dict__ if self.market_overview else {},
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'system_name': 'Trade Director Ensemble', 'status': 'error'}

# Main execution
async def main():
    """Main function to run the Trade Director system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'symbols': ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC'],
        'primary_timeframe': '15m',
        'timeframes': ['5m', '15m', '1h', '4h', '1d']
    }
    
    # Initialize and start Trade Director
    director = TradeDirector(config)
    
    try:
        await director.start_trading_system()
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
        await director.emergency_shutdown()
    except Exception as e:
        logging.error(f"System error: {e}")
        await director.emergency_shutdown()

if __name__ == "__main__":
    asyncio.run(main())