"""
Agent Coordinator - Master integration system for all trading agents
Orchestrates data flow, decision making, and agent communication
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd

# Import all agents
from .data_quality_agent import DataQualityAgent, DataQualityReport
from .market_regime_agent import MarketRegimeAgent, RegimeAnalysis, MarketRegime
from .liquidity_microstructure_agent import LiquidityMicrostructureAgent, MicrostructureReport
from .execution_agent import ExecutionAgent, ExecutionReport
from .order_health_monitor import OrderHealthMonitor, HealthAlert
from .risk_governance_agent import RiskGovernanceAgent, RiskAssessment, RiskAction
from .compliance_journal_agent import ComplianceJournalAgent, JournalEventType, AuditAction
from .attribution_drift_agent import AttributionDriftAgent, SignalType

class SystemStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class TradingDecision:
    decision_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold', 'close'
    size: float
    confidence: float
    reasoning: str
    supporting_agents: List[str]
    risk_adjusted_size: Optional[float]
    execution_params: Dict
    compliance_approved: bool
    timestamp: datetime

@dataclass
class SystemHealth:
    overall_status: SystemStatus
    agent_statuses: Dict[str, str]
    active_alerts: List[Dict]
    performance_metrics: Dict[str, float]
    last_update: datetime
    recommendations: List[str]

class AgentCoordinator:
    """
    Master coordinator that orchestrates all trading agents
    """
    
    def __init__(self, config: Optional[Dict] = None, exchange_client=None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.exchange_client = exchange_client
        
        # Initialize all agents
        self.agents = self._initialize_agents()
        
        # System state
        self.system_status = SystemStatus.HEALTHY
        self.active_decisions = {}
        self.agent_health = {}
        self.performance_metrics = {}
        
        # Decision pipeline
        self.decision_pipeline = [
            self._data_quality_check,
            self._regime_analysis,
            self._liquidity_check,
            self._risk_assessment,
            self._compliance_check,
            self._execution_planning,
            self._final_decision
        ]
        
        # Background monitoring
        self._start_background_monitoring()
    
    def _default_config(self) -> Dict:
        return {
            # System settings
            'decision_timeout_seconds': 30,
            'max_concurrent_decisions': 10,
            'health_check_interval': 60,  # 1 minute
            
            # Agent weights for decision making
            'agent_weights': {
                'data_quality': 0.25,
                'market_regime': 0.20,
                'liquidity': 0.15,
                'risk_governance': 0.25,
                'execution': 0.10,
                'compliance': 0.05
            },
            
            # Decision thresholds
            'min_confidence_threshold': 0.6,
            'max_risk_score': 80,
            'min_data_quality_score': 70,
            'min_liquidity_score': 60,
            
            # Emergency settings
            'emergency_stop_conditions': [
                'multiple_critical_alerts',
                'system_wide_failure',
                'regulatory_breach'
            ],
            
            # Performance tracking
            'performance_update_interval': 300,  # 5 minutes
            'agent_timeout': 10,  # seconds
        }
    
    def _initialize_agents(self) -> Dict:
        """Initialize all trading agents"""
        agents = {
            'data_quality': DataQualityAgent(),
            'market_regime': MarketRegimeAgent(),
            'liquidity': LiquidityMicrostructureAgent(self.exchange_client),
            'execution': ExecutionAgent(self.exchange_client),
            'order_health': OrderHealthMonitor(),
            'risk_governance': RiskGovernanceAgent(),
            'compliance': ComplianceJournalAgent(),
            'attribution': AttributionDriftAgent()
        }
        
        self.logger.info(f"Initialized {len(agents)} trading agents")
        return agents
    
    async def process_trading_signal(self, signal: Dict) -> TradingDecision:
        """
        Process a trading signal through the complete agent pipeline
        """
        decision_id = f"TD_{int(datetime.now().timestamp())}_{signal['symbol']}"
        
        self.logger.info(f"Processing trading signal: {decision_id}")
        
        # Initialize decision context
        decision_context = {
            'decision_id': decision_id,
            'original_signal': signal,
            'symbol': signal['symbol'],
            'timestamp': datetime.now(),
            'pipeline_results': {},
            'agent_votes': {},
            'warnings': [],
            'blocking_issues': []
        }
        
        # Execute decision pipeline
        try:
            for stage in self.decision_pipeline:
                result = await stage(decision_context)
                if result and result.get('block_execution'):
                    # Pipeline stage blocked execution
                    break
            
            # Create final decision
            decision = self._create_trading_decision(decision_context)
            
            # Record decision for tracking
            self.active_decisions[decision_id] = decision
            
            # Journal the decision
            await self._journal_decision(decision, decision_context)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing trading signal {decision_id}: {e}")
            # Create rejection decision
            return self._create_rejection_decision(decision_id, signal, str(e))
    
    async def _data_quality_check(self, context: Dict) -> Dict:
        """Stage 1: Data quality validation"""
        symbol = context['symbol']
        
        try:
            # Get recent market data (mock - would get from data feed)
            market_data = pd.DataFrame({
                'open': [100, 101, 99],
                'high': [102, 103, 101],
                'low': [99, 100, 98],
                'close': [101, 99, 100],
                'volume': [10000, 12000, 8000]
            })
            
            # Validate data quality
            report = self.agents['data_quality'].validate_dataframe(market_data, symbol)
            
            quality_score = self.agents['data_quality'].get_quality_score(symbol)
            should_trade, reason = self.agents['data_quality'].should_trade(symbol)
            
            context['pipeline_results']['data_quality'] = {
                'quality_score': quality_score,
                'should_trade': should_trade,
                'reason': reason,
                'issues': [r.issues for r in report]
            }
            
            # Block if quality too low
            if quality_score < self.config['min_data_quality_score']:
                context['blocking_issues'].append(f"Data quality too low: {quality_score}")
                return {'block_execution': True}
            
            context['agent_votes']['data_quality'] = quality_score / 100
            
        except Exception as e:
            self.logger.error(f"Data quality check failed: {e}")
            context['warnings'].append(f"Data quality check failed: {e}")
        
        return {}
    
    async def _regime_analysis(self, context: Dict) -> Dict:
        """Stage 2: Market regime analysis"""
        symbol = context['symbol']
        
        try:
            # Mock timeframe data
            timeframe_data = {
                '15m': pd.DataFrame({'close': [100, 101, 99, 100], 'volume': [1000] * 4}),
                '1h': pd.DataFrame({'close': [98, 100, 101, 100], 'volume': [5000] * 4}),
                '4h': pd.DataFrame({'close': [95, 98, 100, 100], 'volume': [20000] * 4})
            }
            
            # Analyze regime
            regime_analysis = self.agents['market_regime'].analyze_regime(timeframe_data, symbol)
            
            context['pipeline_results']['market_regime'] = {
                'current_regime': regime_analysis.current_regime.value,
                'strength': regime_analysis.regime_strength,
                'recommended_strategy': regime_analysis.recommended_strategy,
                'agent_adjustments': regime_analysis.agent_adjustments
            }
            
            # Apply regime adjustments to other agents
            self._apply_regime_adjustments(regime_analysis.agent_adjustments, context)
            
            # Vote based on regime strength
            regime_vote = regime_analysis.regime_strength / 100
            if regime_analysis.current_regime in [MarketRegime.COMPRESSION, MarketRegime.REVERSAL_WATCH]:
                regime_vote *= 0.5  # Reduce confidence in uncertain regimes
            
            context['agent_votes']['market_regime'] = regime_vote
            
        except Exception as e:
            self.logger.error(f"Regime analysis failed: {e}")
            context['warnings'].append(f"Regime analysis failed: {e}")
        
        return {}
    
    async def _liquidity_check(self, context: Dict) -> Dict:
        """Stage 3: Liquidity and microstructure analysis"""
        symbol = context['symbol']
        
        try:
            # Analyze liquidity
            liquidity_report = await self.agents['liquidity'].analyze_liquidity(symbol)
            
            context['pipeline_results']['liquidity'] = {
                'liquidity_state': liquidity_report.liquidity_state.value,
                'liquidity_score': liquidity_report.liquidity_score,
                'execution_cost': liquidity_report.execution_cost_estimate,
                'max_safe_size': liquidity_report.max_safe_size,
                'recommended_order_type': liquidity_report.recommended_order_type,
                'warnings': liquidity_report.warnings
            }
            
            # Block if liquidity too poor
            if liquidity_report.liquidity_score < self.config['min_liquidity_score']:
                context['blocking_issues'].append(f"Liquidity too poor: {liquidity_report.liquidity_score}")
                return {'block_execution': True}
            
            context['agent_votes']['liquidity'] = liquidity_report.liquidity_score / 100
            
        except Exception as e:
            self.logger.error(f"Liquidity check failed: {e}")
            context['warnings'].append(f"Liquidity check failed: {e}")
        
        return {}
    
    async def _risk_assessment(self, context: Dict) -> Dict:
        """Stage 4: Risk governance assessment"""
        signal = context['original_signal']
        
        try:
            # Prepare trade request for risk assessment
            trade_request = {
                'symbol': signal['symbol'],
                'size': signal.get('size', 100),
                'side': signal.get('side', 'buy'),
                'entry_price': signal.get('entry_price', 100),
                'stop_loss': signal.get('stop_loss', 98)
            }
            
            # Assess risk
            risk_assessment = self.agents['risk_governance'].assess_trade(trade_request)
            
            context['pipeline_results']['risk_governance'] = {
                'action': risk_assessment.action.value,
                'risk_score': risk_assessment.risk_score,
                'violations': [v.value for v in risk_assessment.violations],
                'adjusted_size': risk_assessment.adjusted_size,
                'warnings': risk_assessment.warnings,
                'emergency': risk_assessment.emergency
            }
            
            # Handle risk actions
            if risk_assessment.action == RiskAction.BLOCK_TRADE:
                context['blocking_issues'].append(f"Risk governance blocked: {risk_assessment.warnings}")
                return {'block_execution': True}
            elif risk_assessment.action == RiskAction.EMERGENCY_STOP:
                context['blocking_issues'].append("EMERGENCY STOP triggered")
                self.system_status = SystemStatus.EMERGENCY_STOP
                return {'block_execution': True}
            
            # Apply size adjustment if needed
            if risk_assessment.adjusted_size:
                context['risk_adjusted_size'] = risk_assessment.adjusted_size
            
            # Vote inversely to risk score
            risk_vote = max(0, (100 - risk_assessment.risk_score) / 100)
            context['agent_votes']['risk_governance'] = risk_vote
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            context['warnings'].append(f"Risk assessment failed: {e}")
        
        return {}
    
    async def _compliance_check(self, context: Dict) -> Dict:
        """Stage 5: Compliance validation"""
        try:
            # Journal the trading decision for compliance
            entry_id = self.agents['compliance'].journal_event(
                event_type=JournalEventType.STRATEGY_DECISION,
                action=AuditAction.TRADE_EXECUTED,
                symbol=context['symbol'],
                rationale=f"Automated trading decision: {context['decision_id']}",
                context={
                    'decision_id': context['decision_id'],
                    'pipeline_results': context['pipeline_results'],
                    'agent_votes': context['agent_votes'],
                    'warnings': context['warnings']
                },
                strategy_id="strat_bot_v1",
                take_screenshot=False  # Skip screenshot for automated decisions
            )
            
            context['pipeline_results']['compliance'] = {
                'journal_entry_id': entry_id,
                'compliance_approved': True
            }
            
            context['agent_votes']['compliance'] = 1.0  # Compliance always votes yes if no violations
            
        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            context['warnings'].append(f"Compliance check failed: {e}")
            context['blocking_issues'].append(f"Compliance validation failed: {e}")
            return {'block_execution': True}
        
        return {}
    
    async def _execution_planning(self, context: Dict) -> Dict:
        """Stage 6: Execution planning"""
        try:
            # Get execution parameters from liquidity analysis
            liquidity_results = context['pipeline_results'].get('liquidity', {})
            
            execution_params = {
                'order_type': 'limit',
                'urgency': 'medium',
                'max_slippage_bps': 10,
                'execution_algo': 'standard'
            }
            
            # Adjust based on liquidity
            if liquidity_results.get('liquidity_score', 100) < 70:
                execution_params['execution_algo'] = 'patient'
                execution_params['urgency'] = 'low'
            elif liquidity_results.get('execution_cost', 0) > 15:
                execution_params['execution_algo'] = 'stealth'
            
            context['pipeline_results']['execution'] = execution_params
            context['agent_votes']['execution'] = 0.8  # Neutral execution vote
            
        except Exception as e:
            self.logger.error(f"Execution planning failed: {e}")
            context['warnings'].append(f"Execution planning failed: {e}")
        
        return {}
    
    async def _final_decision(self, context: Dict) -> Dict:
        """Stage 7: Final decision synthesis"""
        try:
            # Calculate weighted confidence score
            total_weight = 0
            weighted_score = 0
            
            for agent, vote in context['agent_votes'].items():
                weight = self.config['agent_weights'].get(agent, 0.1)
                weighted_score += vote * weight
                total_weight += weight
            
            final_confidence = weighted_score / total_weight if total_weight > 0 else 0
            
            context['final_confidence'] = final_confidence
            context['decision_approved'] = (
                final_confidence >= self.config['min_confidence_threshold'] and
                not context['blocking_issues']
            )
            
        except Exception as e:
            self.logger.error(f"Final decision synthesis failed: {e}")
            context['warnings'].append(f"Final decision failed: {e}")
            context['decision_approved'] = False
        
        return {}
    
    def _apply_regime_adjustments(self, adjustments: Dict, context: Dict):
        """Apply regime-specific adjustments to other agents"""
        # Would update other agent configurations based on regime
        context['regime_adjustments'] = adjustments
    
    def _create_trading_decision(self, context: Dict) -> TradingDecision:
        """Create final trading decision from pipeline results"""
        signal = context['original_signal']
        
        if not context.get('decision_approved', False):
            action = 'hold'
            size = 0
            reasoning = f"Decision blocked: {', '.join(context['blocking_issues'])}"
        else:
            action = signal.get('action', 'buy')
            size = context.get('risk_adjusted_size', signal.get('size', 100))
            reasoning = f"Approved by agent pipeline. Confidence: {context['final_confidence']:.1%}"
        
        return TradingDecision(
            decision_id=context['decision_id'],
            symbol=context['symbol'],
            action=action,
            size=size,
            confidence=context.get('final_confidence', 0),
            reasoning=reasoning,
            supporting_agents=list(context['agent_votes'].keys()),
            risk_adjusted_size=context.get('risk_adjusted_size'),
            execution_params=context['pipeline_results'].get('execution', {}),
            compliance_approved=context['pipeline_results'].get('compliance', {}).get('compliance_approved', False),
            timestamp=context['timestamp']
        )
    
    def _create_rejection_decision(self, decision_id: str, signal: Dict, error: str) -> TradingDecision:
        """Create rejection decision for failed signals"""
        return TradingDecision(
            decision_id=decision_id,
            symbol=signal['symbol'],
            action='hold',
            size=0,
            confidence=0.0,
            reasoning=f"Signal rejected due to error: {error}",
            supporting_agents=[],
            risk_adjusted_size=None,
            execution_params={},
            compliance_approved=False,
            timestamp=datetime.now()
        )
    
    async def _journal_decision(self, decision: TradingDecision, context: Dict):
        """Journal the final trading decision"""
        try:
            self.agents['compliance'].journal_event(
                event_type=JournalEventType.ORDER_PLACEMENT,
                action=AuditAction.TRADE_EXECUTED if decision.action != 'hold' else AuditAction.MANUAL_INTERVENTION,
                symbol=decision.symbol,
                rationale=decision.reasoning,
                context={
                    'decision': {
                        'action': decision.action,
                        'size': decision.size,
                        'confidence': decision.confidence
                    },
                    'pipeline_results': context['pipeline_results'],
                    'agent_votes': context['agent_votes'],
                    'warnings': context['warnings']
                },
                strategy_id="strat_bot_v1"
            )
        except Exception as e:
            self.logger.error(f"Failed to journal decision: {e}")
    
    async def execute_decision(self, decision: TradingDecision) -> ExecutionReport:
        """Execute a trading decision"""
        if decision.action == 'hold' or decision.size <= 0:
            self.logger.info(f"Skipping execution for {decision.decision_id}: {decision.reasoning}")
            return None
        
        try:
            # Prepare order request
            order_request = {
                'symbol': decision.symbol,
                'side': decision.action,
                'size': decision.size,
                'order_type': decision.execution_params.get('order_type', 'limit')
            }
            
            # Execute through execution agent
            execution_report = await self.agents['execution'].execute_order(
                order_request, decision.execution_params
            )
            
            # Record result for attribution
            if execution_report.success:
                # Would record actual P&L when position is closed
                pass
            
            return execution_report
            
        except Exception as e:
            self.logger.error(f"Execution failed for {decision.decision_id}: {e}")
            return None
    
    def _start_background_monitoring(self):
        """Start background system monitoring"""
        async def monitor():
            while True:
                try:
                    await self._system_health_check()
                    await asyncio.sleep(self.config['health_check_interval'])
                except Exception as e:
                    self.logger.error(f"Background monitoring error: {e}")
                    await asyncio.sleep(5)
        
        # Would start the monitoring task
        pass
    
    async def _system_health_check(self):
        """Perform system health check"""
        try:
            # Check each agent's health
            agent_statuses = {}
            active_alerts = []
            
            # Data quality agent health
            try:
                quality_score = self.agents['data_quality'].get_quality_score("SYSTEM")
                agent_statuses['data_quality'] = 'healthy' if quality_score > 80 else 'warning'
            except:
                agent_statuses['data_quality'] = 'error'
            
            # Risk governance health
            try:
                risk_report = self.agents['risk_governance'].get_risk_report()
                if risk_report.get('emergency_mode'):
                    agent_statuses['risk_governance'] = 'critical'
                    active_alerts.append({'type': 'emergency_mode', 'agent': 'risk_governance'})
                else:
                    agent_statuses['risk_governance'] = 'healthy'
            except:
                agent_statuses['risk_governance'] = 'error'
            
            # Order health monitor
            try:
                health_summary = self.agents['order_health'].get_health_summary()
                if health_summary.get('system_health') == 'critical':
                    agent_statuses['order_health'] = 'critical'
                else:
                    agent_statuses['order_health'] = 'healthy'
            except:
                agent_statuses['order_health'] = 'error'
            
            # Determine overall system status
            if any(status == 'critical' for status in agent_statuses.values()):
                self.system_status = SystemStatus.CRITICAL
            elif any(status == 'error' for status in agent_statuses.values()):
                self.system_status = SystemStatus.DEGRADED
            elif any(status == 'warning' for status in agent_statuses.values()):
                self.system_status = SystemStatus.WARNING
            else:
                self.system_status = SystemStatus.HEALTHY
            
            # Update health tracking
            self.agent_health = agent_statuses
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            self.system_status = SystemStatus.DEGRADED
    
    def get_system_status(self) -> SystemHealth:
        """Get current system health status"""
        return SystemHealth(
            overall_status=self.system_status,
            agent_statuses=self.agent_health,
            active_alerts=[],  # Would collect from all agents
            performance_metrics=self.performance_metrics,
            last_update=datetime.now(),
            recommendations=self._generate_system_recommendations()
        )
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        if self.system_status == SystemStatus.CRITICAL:
            recommendations.append("URGENT: Address critical system issues immediately")
        elif self.system_status == SystemStatus.DEGRADED:
            recommendations.append("System performance degraded - review agent health")
        elif self.system_status == SystemStatus.WARNING:
            recommendations.append("Monitor system closely - some agents reporting issues")
        
        # Agent-specific recommendations
        if self.agent_health.get('risk_governance') == 'critical':
            recommendations.append("Risk governance in emergency mode - halt trading")
        
        if not recommendations:
            recommendations.append("System operating normally")
        
        return recommendations
    
    async def shutdown(self):
        """Graceful shutdown of all agents"""
        self.logger.info("Shutting down agent coordinator...")
        
        try:
            # Stop background monitoring
            # Cancel any active tasks
            
            # Shutdown individual agents
            if 'order_health' in self.agents:
                self.agents['order_health'].stop_monitoring()
            
            # Final compliance journal
            if 'compliance' in self.agents:
                self.agents['compliance'].journal_event(
                    event_type=JournalEventType.SYSTEM_EVENT,
                    action=AuditAction.STRATEGY_STOPPED,
                    symbol="SYSTEM",
                    rationale="System shutdown initiated",
                    context={'shutdown_time': datetime.now().isoformat()}
                )
            
            self.logger.info("Agent coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_agent_performance_summary(self) -> Dict:
        """Get performance summary for all agents"""
        try:
            attribution_report = self.agents['attribution'].generate_attribution_report(days=7)
            health_summary = self.agents['order_health'].get_health_summary()
            risk_report = self.agents['risk_governance'].get_risk_report()
            
            return {
                'system_status': self.system_status.value,
                'attribution_summary': {
                    'total_pnl': attribution_report.total_pnl,
                    'top_performers': attribution_report.top_performers[:3],
                    'drift_alerts': len(attribution_report.drift_alerts)
                },
                'execution_health': {
                    'system_health': health_summary.get('system_health'),
                    'recent_performance': health_summary.get('recent_performance')
                },
                'risk_metrics': {
                    'emergency_mode': risk_report.get('emergency_mode'),
                    'current_violations': risk_report.get('current_violations'),
                    'risk_score': risk_report.get('risk_score')
                },
                'active_decisions': len(self.active_decisions)
            }
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}