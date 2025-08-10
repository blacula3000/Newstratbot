"""
Attribution & Drift Detection Agent - Performance tracking and signal decay detection
Monitors: sub-signal effectiveness, strategy drift, edge attribution, precision tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    STRAT_PATTERN = "strat_pattern"
    TECHNICAL_INDICATOR = "technical_indicator"
    VOLUME_PROFILE = "volume_profile"
    MARKET_STRUCTURE = "market_structure"
    REGIME_FILTER = "regime_filter"
    TIMEFRAME_CONFLUENCE = "timeframe_confluence"
    RISK_ADJUSTMENT = "risk_adjustment"

class PerformanceMetric(Enum):
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    AVG_WINNER = "avg_winner"
    AVG_LOSER = "avg_loser"
    EXPECTANCY = "expectancy"
    HIT_RATE = "hit_rate"

class DriftType(Enum):
    SIGNAL_DECAY = "signal_decay"
    MARKET_REGIME_SHIFT = "market_regime_shift"
    VOLATILITY_CHANGE = "volatility_change"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    EXECUTION_DEGRADATION = "execution_degradation"

@dataclass
class SignalPerformance:
    signal_id: str
    signal_type: SignalType
    symbol: str
    timeframe: str
    total_signals: int
    winning_signals: int
    losing_signals: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    sharpe_ratio: float
    last_signal_time: datetime
    performance_window: List[float]  # Recent PnL history
    confidence_score: float  # 0-100
    effectiveness_trend: str  # 'improving', 'stable', 'declining'

@dataclass
class DriftAlert:
    alert_id: str
    drift_type: DriftType
    signal_id: str
    symbol: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    metrics: Dict
    detection_time: datetime
    confidence: float
    recommended_action: str
    historical_comparison: Dict

@dataclass
class AttributionReport:
    period_start: datetime
    period_end: datetime
    total_pnl: float
    signal_contributions: Dict[str, float]
    top_performers: List[Tuple[str, float]]
    bottom_performers: List[Tuple[str, float]]
    regime_performance: Dict[str, Dict]
    timeframe_performance: Dict[str, Dict]
    drift_alerts: List[DriftAlert]
    recommendations: List[str]

class AttributionDriftAgent:
    """
    Tracks signal performance attribution and detects strategy drift
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.signal_performance = {}  # signal_id -> SignalPerformance
        self.trade_attribution = deque(maxlen=1000)  # Recent trades with signal attribution
        self.drift_history = deque(maxlen=100)
        
        # Analysis windows
        self.performance_windows = {
            'daily': deque(maxlen=30),
            'weekly': deque(maxlen=12),
            'monthly': deque(maxlen=6)
        }
        
        # Market regime tracking
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.current_regime = "unknown"
        
        # Drift detection
        self.drift_detectors = self._initialize_drift_detectors()
        self.baseline_metrics = {}
        
    def _default_config(self) -> Dict:
        return {
            # Performance tracking
            'min_samples_for_analysis': 20,
            'performance_window_size': 50,
            'attribution_lookback_days': 30,
            
            # Drift detection thresholds
            'win_rate_drift_threshold': 0.1,  # 10% change
            'profit_factor_drift_threshold': 0.2,  # 20% change
            'sharpe_drift_threshold': 0.3,  # 30% change
            'consecutive_loss_threshold': 5,
            'drawdown_drift_threshold': 0.15,  # 15% increase in drawdown
            
            # Signal effectiveness thresholds
            'min_win_rate': 0.4,  # 40% minimum win rate
            'min_profit_factor': 1.1,  # 1.1 minimum profit factor
            'min_sharpe_ratio': 0.5,  # 0.5 minimum Sharpe ratio
            'max_consecutive_losses': 7,
            
            # Drift alert settings
            'drift_check_interval': 3600,  # 1 hour in seconds
            'alert_cooldown': 7200,  # 2 hours between similar alerts
            'confidence_threshold': 0.7,  # Minimum confidence for alerts
            
            # Auto-reweighting
            'enable_auto_reweight': True,
            'reweight_threshold': 0.3,  # Reweight if performance drops 30%
            'max_signal_weight': 0.3,  # Maximum weight per signal
            'min_signal_weight': 0.05,  # Minimum weight per signal
            
            # Baseline establishment
            'baseline_period_days': 90,
            'baseline_update_frequency': 30  # days
        }
    
    def _initialize_drift_detectors(self) -> Dict:
        """Initialize drift detection algorithms"""
        return {
            'statistical': self._detect_statistical_drift,
            'trend': self._detect_trend_drift,
            'regime': self._detect_regime_drift,
            'correlation': self._detect_correlation_drift,
            'volatility': self._detect_volatility_drift
        }
    
    def record_signal_result(self, signal_id: str, signal_type: SignalType,
                           symbol: str, timeframe: str, pnl: float,
                           entry_time: datetime, exit_time: datetime,
                           metadata: Optional[Dict] = None):
        """
        Record the result of a trading signal for attribution analysis
        """
        # Initialize signal performance tracking if new
        if signal_id not in self.signal_performance:
            self.signal_performance[signal_id] = SignalPerformance(
                signal_id=signal_id,
                signal_type=signal_type,
                symbol=symbol,
                timeframe=timeframe,
                total_signals=0,
                winning_signals=0,
                losing_signals=0,
                total_pnl=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                sharpe_ratio=0.0,
                last_signal_time=entry_time,
                performance_window=[],
                confidence_score=50.0,
                effectiveness_trend='stable'
            )
        
        # Update performance metrics
        perf = self.signal_performance[signal_id]
        perf.total_signals += 1
        perf.total_pnl += pnl
        perf.last_signal_time = exit_time
        
        # Add to performance window
        perf.performance_window.append(pnl)
        if len(perf.performance_window) > self.config['performance_window_size']:
            perf.performance_window.pop(0)
        
        # Update win/loss tracking
        if pnl > 0:
            perf.winning_signals += 1
        else:
            perf.losing_signals += 1
        
        # Recalculate metrics
        self._update_signal_metrics(signal_id)
        
        # Record trade attribution
        self.trade_attribution.append({
            'signal_id': signal_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'pnl': pnl,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'regime': self.current_regime,
            'metadata': metadata or {}
        })
        
        # Update regime performance
        self.regime_performance[self.current_regime][signal_id].append(pnl)
        
        # Check for drift if enough samples
        if perf.total_signals >= self.config['min_samples_for_analysis']:
            self._check_signal_drift(signal_id)
        
        self.logger.debug(f"Recorded signal result: {signal_id} -> {pnl:.2f}")
    
    def _update_signal_metrics(self, signal_id: str):
        """Update calculated metrics for a signal"""
        perf = self.signal_performance[signal_id]
        
        if perf.total_signals == 0:
            return
        
        # Win rate
        perf.win_rate = perf.winning_signals / perf.total_signals
        
        # Average win/loss
        if perf.winning_signals > 0:
            winning_trades = [pnl for pnl in perf.performance_window if pnl > 0]
            perf.avg_win = statistics.mean(winning_trades) if winning_trades else 0
        
        if perf.losing_signals > 0:
            losing_trades = [abs(pnl) for pnl in perf.performance_window if pnl < 0]
            perf.avg_loss = statistics.mean(losing_trades) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(pnl for pnl in perf.performance_window if pnl > 0)
        total_losses = sum(abs(pnl) for pnl in perf.performance_window if pnl < 0)
        perf.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Consecutive wins/losses
        perf.max_consecutive_wins = self._calculate_max_consecutive(perf.performance_window, True)
        perf.max_consecutive_losses = self._calculate_max_consecutive(perf.performance_window, False)
        
        # Sharpe ratio (simplified)
        if len(perf.performance_window) >= 10:
            returns = perf.performance_window
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            perf.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Confidence score
        perf.confidence_score = self._calculate_confidence_score(perf)
        
        # Effectiveness trend
        perf.effectiveness_trend = self._determine_effectiveness_trend(perf)
    
    def _calculate_max_consecutive(self, pnl_history: List[float], wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not pnl_history:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_history:
            if (wins and pnl > 0) or (not wins and pnl <= 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_confidence_score(self, perf: SignalPerformance) -> float:
        """Calculate confidence score for signal effectiveness"""
        score = 50.0  # Base score
        
        # Sample size factor
        if perf.total_signals >= 50:
            score += 20
        elif perf.total_signals >= 30:
            score += 15
        elif perf.total_signals >= 20:
            score += 10
        
        # Win rate factor
        if perf.win_rate > 0.6:
            score += 15
        elif perf.win_rate > 0.5:
            score += 10
        elif perf.win_rate > 0.4:
            score += 5
        elif perf.win_rate < 0.3:
            score -= 20
        
        # Profit factor
        if perf.profit_factor > 2.0:
            score += 15
        elif perf.profit_factor > 1.5:
            score += 10
        elif perf.profit_factor > 1.1:
            score += 5
        elif perf.profit_factor < 1.0:
            score -= 25
        
        # Consistency (low consecutive losses)
        if perf.max_consecutive_losses > 7:
            score -= 15
        elif perf.max_consecutive_losses > 5:
            score -= 10
        
        # Recent performance
        if len(perf.performance_window) >= 10:
            recent_avg = statistics.mean(perf.performance_window[-10:])
            overall_avg = statistics.mean(perf.performance_window)
            
            if recent_avg > overall_avg * 1.2:
                score += 10  # Improving
            elif recent_avg < overall_avg * 0.8:
                score -= 15  # Declining
        
        return max(0, min(100, score))
    
    def _determine_effectiveness_trend(self, perf: SignalPerformance) -> str:
        """Determine if signal effectiveness is improving, stable, or declining"""
        if len(perf.performance_window) < 20:
            return 'stable'
        
        # Split into two halves
        mid_point = len(perf.performance_window) // 2
        first_half = perf.performance_window[:mid_point]
        second_half = perf.performance_window[mid_point:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        improvement = (second_avg - first_avg) / abs(first_avg) if first_avg != 0 else 0
        
        if improvement > 0.1:  # 10% improvement
            return 'improving'
        elif improvement < -0.1:  # 10% decline
            return 'declining'
        else:
            return 'stable'
    
    def _check_signal_drift(self, signal_id: str):
        """Check for performance drift in a signal"""
        for detector_name, detector_func in self.drift_detectors.items():
            try:
                drift_alert = detector_func(signal_id)
                if drift_alert and drift_alert.confidence >= self.config['confidence_threshold']:
                    self._record_drift_alert(drift_alert)
            except Exception as e:
                self.logger.error(f"Drift detection failed for {detector_name}: {e}")
    
    def _detect_statistical_drift(self, signal_id: str) -> Optional[DriftAlert]:
        """Detect statistical drift in signal performance"""
        perf = self.signal_performance[signal_id]
        
        if len(perf.performance_window) < 30:
            return None
        
        # Split into baseline and recent periods
        baseline_size = len(perf.performance_window) // 2
        baseline = perf.performance_window[:baseline_size]
        recent = perf.performance_window[baseline_size:]
        
        # Statistical tests
        baseline_mean = statistics.mean(baseline)
        recent_mean = statistics.mean(recent)
        
        baseline_wr = sum(1 for x in baseline if x > 0) / len(baseline)
        recent_wr = sum(1 for x in recent if x > 0) / len(recent)
        
        # Check for significant changes
        mean_change = (recent_mean - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else 0
        wr_change = abs(recent_wr - baseline_wr)
        
        if abs(mean_change) > self.config['profit_factor_drift_threshold'] or \
           wr_change > self.config['win_rate_drift_threshold']:
            
            severity = 'critical' if abs(mean_change) > 0.5 else 'high'
            
            return DriftAlert(
                alert_id=f"STAT_DRIFT_{signal_id}_{int(datetime.now().timestamp())}",
                drift_type=DriftType.SIGNAL_DECAY,
                signal_id=signal_id,
                symbol=perf.symbol,
                severity=severity,
                description=f"Statistical drift detected: Mean PnL changed {mean_change:.1%}, Win rate changed {wr_change:.1%}",
                metrics={
                    'baseline_mean': baseline_mean,
                    'recent_mean': recent_mean,
                    'mean_change_pct': mean_change * 100,
                    'baseline_win_rate': baseline_wr,
                    'recent_win_rate': recent_wr,
                    'wr_change': wr_change
                },
                detection_time=datetime.now(),
                confidence=min(0.9, abs(mean_change) + wr_change),
                recommended_action='reduce_weight' if mean_change < -0.2 else 'monitor',
                historical_comparison={'baseline_period': len(baseline), 'recent_period': len(recent)}
            )
        
        return None
    
    def _detect_trend_drift(self, signal_id: str) -> Optional[DriftAlert]:
        """Detect trend-based drift using linear regression"""
        perf = self.signal_performance[signal_id]
        
        if len(perf.performance_window) < 20:
            return None
        
        # Fit linear regression to performance
        x = np.arange(len(perf.performance_window)).reshape(-1, 1)
        y = np.array(perf.performance_window)
        
        reg = LinearRegression().fit(x, y)
        trend_slope = reg.coef_[0]
        r_squared = r2_score(y, reg.predict(x))
        
        # Check for significant negative trend
        if trend_slope < -0.01 and r_squared > 0.3:  # Strong negative trend
            severity = 'high' if trend_slope < -0.05 else 'medium'
            
            return DriftAlert(
                alert_id=f"TREND_DRIFT_{signal_id}_{int(datetime.now().timestamp())}",
                drift_type=DriftType.SIGNAL_DECAY,
                signal_id=signal_id,
                symbol=perf.symbol,
                severity=severity,
                description=f"Negative performance trend detected: slope={trend_slope:.4f}",
                metrics={
                    'trend_slope': trend_slope,
                    'r_squared': r_squared,
                    'projected_decline': trend_slope * len(perf.performance_window)
                },
                detection_time=datetime.now(),
                confidence=min(0.9, r_squared),
                recommended_action='reduce_weight' if trend_slope < -0.03 else 'monitor',
                historical_comparison={'sample_size': len(perf.performance_window)}
            )
        
        return None
    
    def _detect_regime_drift(self, signal_id: str) -> Optional[DriftAlert]:
        """Detect performance drift due to market regime changes"""
        if signal_id not in self.signal_performance:
            return None
        
        # Check performance across different regimes
        regime_performance = {}
        for regime, signals in self.regime_performance.items():
            if signal_id in signals:
                regime_perf = signals[signal_id]
                if len(regime_perf) >= 5:  # Minimum samples
                    regime_performance[regime] = {
                        'mean_pnl': statistics.mean(regime_perf),
                        'win_rate': sum(1 for x in regime_perf if x > 0) / len(regime_perf),
                        'sample_size': len(regime_perf)
                    }
        
        if len(regime_performance) < 2:
            return None
        
        # Find best and worst performing regimes
        regimes_by_performance = sorted(regime_performance.items(), 
                                      key=lambda x: x[1]['mean_pnl'], reverse=True)
        
        best_regime = regimes_by_performance[0]
        worst_regime = regimes_by_performance[-1]
        
        performance_gap = best_regime[1]['mean_pnl'] - worst_regime[1]['mean_pnl']
        
        # Check if current regime is problematic
        if self.current_regime == worst_regime[0] and performance_gap > 0.1:
            return DriftAlert(
                alert_id=f"REGIME_DRIFT_{signal_id}_{int(datetime.now().timestamp())}",
                drift_type=DriftType.MARKET_REGIME_SHIFT,
                signal_id=signal_id,
                symbol=self.signal_performance[signal_id].symbol,
                severity='medium',
                description=f"Signal performs poorly in current regime ({self.current_regime})",
                metrics={
                    'current_regime': self.current_regime,
                    'current_regime_performance': regime_performance[self.current_regime],
                    'best_regime': best_regime[0],
                    'best_regime_performance': best_regime[1],
                    'performance_gap': performance_gap
                },
                detection_time=datetime.now(),
                confidence=0.7,
                recommended_action='reduce_weight_in_regime',
                historical_comparison=regime_performance
            )
        
        return None
    
    def _detect_correlation_drift(self, signal_id: str) -> Optional[DriftAlert]:
        """Detect correlation breakdown between signals"""
        # Simplified implementation
        # Would analyze correlation between this signal and other signals
        return None
    
    def _detect_volatility_drift(self, signal_id: str) -> Optional[DriftAlert]:
        """Detect changes in signal performance volatility"""
        perf = self.signal_performance[signal_id]
        
        if len(perf.performance_window) < 30:
            return None
        
        # Split into periods
        mid_point = len(perf.performance_window) // 2
        early_period = perf.performance_window[:mid_point]
        recent_period = perf.performance_window[mid_point:]
        
        early_vol = statistics.stdev(early_period) if len(early_period) > 1 else 0
        recent_vol = statistics.stdev(recent_period) if len(recent_period) > 1 else 0
        
        if early_vol > 0:
            vol_change = (recent_vol - early_vol) / early_vol
            
            if abs(vol_change) > 0.5:  # 50% change in volatility
                return DriftAlert(
                    alert_id=f"VOL_DRIFT_{signal_id}_{int(datetime.now().timestamp())}",
                    drift_type=DriftType.VOLATILITY_CHANGE,
                    signal_id=signal_id,
                    symbol=perf.symbol,
                    severity='medium',
                    description=f"Signal volatility changed {vol_change:.1%}",
                    metrics={
                        'early_volatility': early_vol,
                        'recent_volatility': recent_vol,
                        'volatility_change_pct': vol_change * 100
                    },
                    detection_time=datetime.now(),
                    confidence=0.6,
                    recommended_action='adjust_position_sizing',
                    historical_comparison={'early_samples': len(early_period), 'recent_samples': len(recent_period)}
                )
        
        return None
    
    def _record_drift_alert(self, alert: DriftAlert):
        """Record and potentially act on drift alert"""
        # Check cooldown
        recent_similar = [a for a in self.drift_history 
                         if a.signal_id == alert.signal_id 
                         and a.drift_type == alert.drift_type
                         and (datetime.now() - a.detection_time).seconds < self.config['alert_cooldown']]
        
        if recent_similar:
            return  # Skip duplicate alerts
        
        self.drift_history.append(alert)
        
        # Log alert
        log_func = {
            'low': self.logger.info,
            'medium': self.logger.warning,
            'high': self.logger.error,
            'critical': self.logger.critical
        }.get(alert.severity, self.logger.warning)
        
        log_func(f"DRIFT ALERT [{alert.severity.upper()}]: {alert.description}")
        
        # Execute recommended action if configured
        if self.config['enable_auto_reweight'] and alert.recommended_action in ['reduce_weight', 'reduce_weight_in_regime']:
            self._auto_reweight_signal(alert.signal_id, alert.recommended_action)
    
    def _auto_reweight_signal(self, signal_id: str, action: str):
        """Automatically reweight signal based on drift detection"""
        try:
            current_weight = self._get_signal_weight(signal_id)
            
            if action == 'reduce_weight':
                new_weight = max(self.config['min_signal_weight'], current_weight * 0.7)
            elif action == 'increase_weight':
                new_weight = min(self.config['max_signal_weight'], current_weight * 1.3)
            else:
                new_weight = current_weight
            
            # Would integrate with signal weighting system
            self.logger.warning(f"AUTO-REWEIGHT: {signal_id} weight {current_weight:.3f} -> {new_weight:.3f}")
            
        except Exception as e:
            self.logger.error(f"Auto-reweight failed for {signal_id}: {e}")
    
    def _get_signal_weight(self, signal_id: str) -> float:
        """Get current weight of signal in strategy"""
        # Mock implementation - would integrate with strategy manager
        return 0.1
    
    def update_market_regime(self, regime: str):
        """Update current market regime"""
        self.current_regime = regime
        self.logger.debug(f"Market regime updated to: {regime}")
    
    def generate_attribution_report(self, days: int = 30) -> AttributionReport:
        """Generate comprehensive attribution report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter recent trades
        recent_trades = [
            trade for trade in self.trade_attribution
            if trade['exit_time'] >= start_date
        ]
        
        # Calculate signal contributions
        signal_contributions = defaultdict(float)
        for trade in recent_trades:
            signal_contributions[trade['signal_id']] += trade['pnl']
        
        total_pnl = sum(signal_contributions.values())
        
        # Top/bottom performers
        sorted_signals = sorted(signal_contributions.items(), key=lambda x: x[1], reverse=True)
        top_performers = sorted_signals[:5]
        bottom_performers = sorted_signals[-5:]
        
        # Regime performance
        regime_perf = {}
        for regime, signals in self.regime_performance.items():
            regime_pnl = 0
            regime_trades = 0
            for signal_id, pnl_list in signals.items():
                recent_pnl = [pnl for pnl, trade in zip(pnl_list, self.trade_attribution) 
                             if trade['exit_time'] >= start_date]
                regime_pnl += sum(recent_pnl)
                regime_trades += len(recent_pnl)
            
            regime_perf[regime] = {
                'total_pnl': regime_pnl,
                'trade_count': regime_trades,
                'avg_pnl_per_trade': regime_pnl / regime_trades if regime_trades > 0 else 0
            }
        
        # Timeframe performance
        timeframe_perf = defaultdict(lambda: {'pnl': 0, 'trades': 0})
        for trade in recent_trades:
            tf = trade['timeframe']
            timeframe_perf[tf]['pnl'] += trade['pnl']
            timeframe_perf[tf]['trades'] += 1
        
        # Recent drift alerts
        recent_alerts = [alert for alert in self.drift_history 
                        if alert.detection_time >= start_date]
        
        # Generate recommendations
        recommendations = self._generate_attribution_recommendations(
            signal_contributions, recent_alerts, regime_perf
        )
        
        return AttributionReport(
            period_start=start_date,
            period_end=end_date,
            total_pnl=total_pnl,
            signal_contributions=dict(signal_contributions),
            top_performers=top_performers,
            bottom_performers=bottom_performers,
            regime_performance=regime_perf,
            timeframe_performance=dict(timeframe_perf),
            drift_alerts=recent_alerts,
            recommendations=recommendations
        )
    
    def _generate_attribution_recommendations(self, contributions: Dict, 
                                           alerts: List[DriftAlert], 
                                           regime_perf: Dict) -> List[str]:
        """Generate recommendations based on attribution analysis"""
        recommendations = []
        
        # Signal performance recommendations
        if contributions:
            total_pnl = sum(contributions.values())
            negative_contributors = {k: v for k, v in contributions.items() if v < 0}
            
            if len(negative_contributors) > len(contributions) / 2:
                recommendations.append("Review strategy - majority of signals are unprofitable")
            
            # Top contributor analysis
            sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            if sorted_contrib and sorted_contrib[0][1] > total_pnl * 0.6:
                recommendations.append(f"Signal {sorted_contrib[0][0]} dominates performance - consider diversification")
        
        # Drift alert recommendations
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical drift alerts immediately")
        
        high_alerts = [a for a in alerts if a.severity == 'high']
        if high_alerts:
            recommendations.append(f"Review {len(high_alerts)} high-severity drift alerts")
        
        # Regime performance recommendations
        if regime_perf:
            worst_regime = min(regime_perf.items(), key=lambda x: x[1]['avg_pnl_per_trade'])
            if worst_regime[1]['avg_pnl_per_trade'] < -0.01:
                recommendations.append(f"Poor performance in {worst_regime[0]} regime - consider regime-specific adjustments")
        
        if not recommendations:
            recommendations.append("Performance attribution looks healthy - maintain current strategy")
        
        return recommendations
    
    def get_signal_rankings(self, metric: PerformanceMetric = PerformanceMetric.PROFIT_FACTOR) -> List[Tuple[str, float]]:
        """Get signals ranked by specified metric"""
        rankings = []
        
        for signal_id, perf in self.signal_performance.items():
            if perf.total_signals < self.config['min_samples_for_analysis']:
                continue
            
            if metric == PerformanceMetric.WIN_RATE:
                value = perf.win_rate
            elif metric == PerformanceMetric.PROFIT_FACTOR:
                value = perf.profit_factor
            elif metric == PerformanceMetric.SHARPE_RATIO:
                value = perf.sharpe_ratio
            elif metric == PerformanceMetric.EXPECTANCY:
                value = perf.total_pnl / perf.total_signals
            else:
                value = perf.confidence_score
            
            rankings.append((signal_id, value))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_underperforming_signals(self) -> List[str]:
        """Get list of signals that are underperforming"""
        underperforming = []
        
        for signal_id, perf in self.signal_performance.items():
            if perf.total_signals < self.config['min_samples_for_analysis']:
                continue
            
            # Check multiple criteria
            criteria_failed = 0
            
            if perf.win_rate < self.config['min_win_rate']:
                criteria_failed += 1
            
            if perf.profit_factor < self.config['min_profit_factor']:
                criteria_failed += 1
            
            if perf.sharpe_ratio < self.config['min_sharpe_ratio']:
                criteria_failed += 1
            
            if perf.max_consecutive_losses > self.config['max_consecutive_losses']:
                criteria_failed += 1
            
            if criteria_failed >= 2:  # Fail multiple criteria
                underperforming.append(signal_id)
        
        return underperforming
    
    def reset_signal_tracking(self, signal_id: str):
        """Reset tracking for a specific signal"""
        if signal_id in self.signal_performance:
            del self.signal_performance[signal_id]
        
        # Remove from trade attribution
        self.trade_attribution = deque([
            trade for trade in self.trade_attribution 
            if trade['signal_id'] != signal_id
        ], maxlen=1000)
        
        self.logger.info(f"Reset tracking for signal: {signal_id}")
    
    def export_performance_data(self) -> Dict:
        """Export performance data for external analysis"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'signals': {},
            'trade_attribution': list(self.trade_attribution),
            'drift_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'drift_type': alert.drift_type.value,
                    'signal_id': alert.signal_id,
                    'severity': alert.severity,
                    'description': alert.description,
                    'detection_time': alert.detection_time.isoformat(),
                    'metrics': alert.metrics
                }
                for alert in self.drift_history
            ]
        }
        
        # Export signal performance
        for signal_id, perf in self.signal_performance.items():
            export_data['signals'][signal_id] = {
                'signal_type': perf.signal_type.value,
                'symbol': perf.symbol,
                'timeframe': perf.timeframe,
                'total_signals': perf.total_signals,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'sharpe_ratio': perf.sharpe_ratio,
                'total_pnl': perf.total_pnl,
                'confidence_score': perf.confidence_score,
                'effectiveness_trend': perf.effectiveness_trend,
                'performance_window': perf.performance_window
            }
        
        return export_data