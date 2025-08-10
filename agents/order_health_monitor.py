"""
Order Health Monitor Agent - Verifies order fills vs. intents
Monitors: fill rates, slippage, latency, partial fills, execution quality
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class AlertType(Enum):
    HIGH_SLIPPAGE = "high_slippage"
    POOR_FILL_RATE = "poor_fill_rate"
    HIGH_LATENCY = "high_latency"
    FREQUENT_REJECTIONS = "frequent_rejections"
    PRICE_DEVIATION = "price_deviation"
    PARTIAL_FILL_RATE = "partial_fill_rate"
    EXECUTION_DELAY = "execution_delay"
    LIQUIDITY_ISSUE = "liquidity_issue"

@dataclass
class OrderIntent:
    """Original order intent for comparison"""
    order_id: str
    symbol: str
    side: str
    intended_size: float
    intended_price: Optional[float]
    max_slippage_bps: float
    expected_fill_time_ms: float
    timestamp: datetime
    strategy_context: Dict
    priority: str  # 'high', 'medium', 'low'

@dataclass
class ExecutionMetrics:
    """Actual execution metrics"""
    order_id: str
    filled_size: float
    avg_fill_price: float
    total_latency_ms: float
    slippage_bps: float
    partial_fills: int
    rejections: int
    execution_venue: str
    timestamp: datetime

@dataclass
class HealthAlert:
    alert_type: AlertType
    severity: str  # 'low', 'medium', 'high', 'critical'
    order_id: str
    symbol: str
    message: str
    metrics: Dict
    timestamp: datetime
    auto_action: Optional[str]  # Recommended automatic action

@dataclass
class VenueHealthReport:
    venue_name: str
    status: HealthStatus
    fill_rate_pct: float
    avg_latency_ms: float
    avg_slippage_bps: float
    rejection_rate_pct: float
    uptime_pct: float
    last_update: datetime
    issues: List[str]
    recommendations: List[str]

class OrderHealthMonitor:
    """
    Monitors order execution health and raises alerts for anomalies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Tracking structures
        self.order_intents = {}  # order_id -> OrderIntent
        self.execution_metrics = {}  # order_id -> ExecutionMetrics
        self.venue_metrics = defaultdict(lambda: {
            'fills': deque(maxlen=100),
            'latencies': deque(maxlen=100),
            'slippages': deque(maxlen=100),
            'rejections': deque(maxlen=100)
        })
        
        # Health monitoring
        self.alerts = deque(maxlen=1000)
        self.venue_health = {}
        self.system_health = HealthStatus.HEALTHY
        
        # Performance windows
        self.performance_windows = {
            'short': deque(maxlen=20),   # Last 20 orders
            'medium': deque(maxlen=100), # Last 100 orders
            'long': deque(maxlen=500)    # Last 500 orders
        }
        
        # Background monitoring
        self._monitoring_task = None
        self._start_monitoring()
    
    def _default_config(self) -> Dict:
        return {
            # Thresholds
            'max_slippage_bps': 15,
            'max_latency_ms': 1000,
            'min_fill_rate': 0.85,  # 85% fill rate
            'max_rejection_rate': 0.1,  # 10% rejection rate
            'max_partial_fill_rate': 0.3,  # 30% partial fills
            
            # Warning thresholds (before critical)
            'warning_slippage_bps': 10,
            'warning_latency_ms': 500,
            'warning_fill_rate': 0.9,
            
            # Monitoring intervals
            'health_check_interval': 30,  # seconds
            'alert_cooldown': 60,  # seconds between similar alerts
            'venue_timeout': 5000,  # ms to consider venue offline
            
            # Sample sizes
            'min_sample_size': 10,
            'rolling_window_size': 100,
            
            # Auto-actions
            'enable_auto_venue_switch': True,
            'enable_auto_size_reduction': True,
            'enable_auto_strategy_pause': True
        }
    
    def register_order_intent(self, intent: OrderIntent):
        """Register an order intent for monitoring"""
        self.order_intents[intent.order_id] = intent
        self.logger.debug(f"Registered intent for order {intent.order_id}")
    
    def record_execution_metrics(self, metrics: ExecutionMetrics):
        """Record actual execution metrics"""
        self.execution_metrics[metrics.order_id] = metrics
        
        # Add to venue tracking
        venue = metrics.execution_venue
        self.venue_metrics[venue]['fills'].append(metrics.filled_size > 0)
        self.venue_metrics[venue]['latencies'].append(metrics.total_latency_ms)
        self.venue_metrics[venue]['slippages'].append(metrics.slippage_bps)
        self.venue_metrics[venue]['rejections'].append(metrics.rejections > 0)
        
        # Add to performance windows
        perf_data = {
            'order_id': metrics.order_id,
            'fill_rate': metrics.filled_size / self.order_intents.get(metrics.order_id, intent).intended_size if metrics.order_id in self.order_intents else 0,
            'slippage': metrics.slippage_bps,
            'latency': metrics.total_latency_ms,
            'timestamp': metrics.timestamp
        }
        
        for window in self.performance_windows.values():
            window.append(perf_data)
        
        # Check for immediate alerts
        self._check_order_health(metrics.order_id)
        
        self.logger.debug(f"Recorded execution metrics for order {metrics.order_id}")
    
    def _check_order_health(self, order_id: str):
        """Check individual order health against intent"""
        if order_id not in self.order_intents or order_id not in self.execution_metrics:
            return
        
        intent = self.order_intents[order_id]
        actual = self.execution_metrics[order_id]
        alerts = []
        
        # Check slippage
        if actual.slippage_bps > intent.max_slippage_bps:
            severity = 'critical' if actual.slippage_bps > self.config['max_slippage_bps'] else 'high'
            alerts.append(HealthAlert(
                alert_type=AlertType.HIGH_SLIPPAGE,
                severity=severity,
                order_id=order_id,
                symbol=intent.symbol,
                message=f"High slippage: {actual.slippage_bps:.1f} bps > {intent.max_slippage_bps:.1f} bps",
                metrics={'actual_slippage': actual.slippage_bps, 'max_allowed': intent.max_slippage_bps},
                timestamp=datetime.now(),
                auto_action='reduce_position_size' if self.config['enable_auto_size_reduction'] else None
            ))
        
        # Check latency
        if actual.total_latency_ms > intent.expected_fill_time_ms * 2:
            severity = 'high' if actual.total_latency_ms > self.config['max_latency_ms'] else 'medium'
            alerts.append(HealthAlert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=severity,
                order_id=order_id,
                symbol=intent.symbol,
                message=f"High latency: {actual.total_latency_ms:.0f}ms > {intent.expected_fill_time_ms:.0f}ms",
                metrics={'actual_latency': actual.total_latency_ms, 'expected': intent.expected_fill_time_ms},
                timestamp=datetime.now(),
                auto_action='switch_venue' if self.config['enable_auto_venue_switch'] else None
            ))
        
        # Check fill rate
        fill_rate = actual.filled_size / intent.intended_size
        if fill_rate < self.config['min_fill_rate']:
            severity = 'high' if fill_rate < 0.7 else 'medium'
            alerts.append(HealthAlert(
                alert_type=AlertType.POOR_FILL_RATE,
                severity=severity,
                order_id=order_id,
                symbol=intent.symbol,
                message=f"Poor fill rate: {fill_rate:.1%} < {self.config['min_fill_rate']:.1%}",
                metrics={'fill_rate': fill_rate, 'min_required': self.config['min_fill_rate']},
                timestamp=datetime.now(),
                auto_action='pause_strategy' if fill_rate < 0.5 else None
            ))
        
        # Check rejections
        if actual.rejections > 0:
            alerts.append(HealthAlert(
                alert_type=AlertType.FREQUENT_REJECTIONS,
                severity='medium',
                order_id=order_id,
                symbol=intent.symbol,
                message=f"Order rejected {actual.rejections} times",
                metrics={'rejections': actual.rejections},
                timestamp=datetime.now(),
                auto_action='review_order_params'
            ))
        
        # Check price deviation (if limit order)
        if intent.intended_price and actual.avg_fill_price > 0:
            if intent.side == 'buy':
                deviation = (actual.avg_fill_price - intent.intended_price) / intent.intended_price
            else:
                deviation = (intent.intended_price - actual.avg_fill_price) / intent.intended_price
            
            deviation_bps = deviation * 10000
            
            if abs(deviation_bps) > 20:  # 20 bps deviation
                alerts.append(HealthAlert(
                    alert_type=AlertType.PRICE_DEVIATION,
                    severity='medium',
                    order_id=order_id,
                    symbol=intent.symbol,
                    message=f"Price deviation: {deviation_bps:.1f} bps from intent",
                    metrics={'deviation_bps': deviation_bps, 'intended': intent.intended_price, 'actual': actual.avg_fill_price},
                    timestamp=datetime.now(),
                    auto_action=None
                ))
        
        # Record alerts
        for alert in alerts:
            self._record_alert(alert)
    
    def _record_alert(self, alert: HealthAlert):
        """Record and potentially act on alert"""
        # Check cooldown
        recent_similar = [a for a in list(self.alerts)[-10:]  # Last 10 alerts
                         if a.alert_type == alert.alert_type 
                         and a.symbol == alert.symbol
                         and (datetime.now() - a.timestamp).seconds < self.config['alert_cooldown']]
        
        if recent_similar:
            return  # Skip duplicate alerts in cooldown period
        
        self.alerts.append(alert)
        
        # Log alert
        log_func = {
            'low': self.logger.info,
            'medium': self.logger.warning,
            'high': self.logger.error,
            'critical': self.logger.critical
        }.get(alert.severity, self.logger.warning)
        
        log_func(f"HEALTH ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Execute auto-action if configured
        if alert.auto_action and self._should_execute_auto_action(alert):
            self._execute_auto_action(alert)
    
    def _should_execute_auto_action(self, alert: HealthAlert) -> bool:
        """Determine if auto-action should be executed"""
        # Only execute for high/critical severity
        if alert.severity not in ['high', 'critical']:
            return False
        
        # Check if similar auto-actions were recently executed
        recent_actions = [a for a in list(self.alerts)[-20:]
                         if a.auto_action == alert.auto_action
                         and (datetime.now() - a.timestamp).seconds < 300]  # 5 minutes
        
        return len(recent_actions) < 3  # Max 3 similar actions in 5 minutes
    
    def _execute_auto_action(self, alert: HealthAlert):
        """Execute automatic remedial action"""
        try:
            if alert.auto_action == 'reduce_position_size':
                self.logger.warning(f"AUTO-ACTION: Reducing position sizes for {alert.symbol}")
                # Would integrate with position manager
                
            elif alert.auto_action == 'switch_venue':
                self.logger.warning(f"AUTO-ACTION: Switching venue for {alert.symbol}")
                # Would integrate with venue router
                
            elif alert.auto_action == 'pause_strategy':
                self.logger.warning(f"AUTO-ACTION: Pausing strategy for {alert.symbol}")
                # Would integrate with strategy manager
                
            elif alert.auto_action == 'review_order_params':
                self.logger.warning(f"AUTO-ACTION: Reviewing order parameters for {alert.symbol}")
                # Would trigger parameter review
                
        except Exception as e:
            self.logger.error(f"Failed to execute auto-action {alert.auto_action}: {e}")
    
    def _start_monitoring(self):
        """Start background health monitoring"""
        async def monitor():
            while True:
                try:
                    await self._periodic_health_check()
                    await asyncio.sleep(self.config['health_check_interval'])
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(5)
        
        if asyncio.get_event_loop().is_running():
            self._monitoring_task = asyncio.create_task(monitor())
        else:
            self._monitoring_task = None
    
    async def _periodic_health_check(self):
        """Periodic system health assessment"""
        # Update venue health
        for venue in self.venue_metrics:
            self._update_venue_health(venue)
        
        # Update system health
        self._update_system_health()
        
        # Check for degradation patterns
        self._check_degradation_patterns()
    
    def _update_venue_health(self, venue: str):
        """Update health status for a venue"""
        metrics = self.venue_metrics[venue]
        
        if not any(len(m) >= self.config['min_sample_size'] for m in metrics.values()):
            return  # Not enough data
        
        # Calculate metrics
        fill_rate = statistics.mean(metrics['fills']) if metrics['fills'] else 0
        avg_latency = statistics.mean(metrics['latencies']) if metrics['latencies'] else 0
        avg_slippage = statistics.mean(metrics['slippages']) if metrics['slippages'] else 0
        rejection_rate = statistics.mean(metrics['rejections']) if metrics['rejections'] else 0
        
        # Determine status
        issues = []
        recommendations = []
        
        if fill_rate < self.config['min_fill_rate']:
            issues.append(f"Low fill rate: {fill_rate:.1%}")
            status = HealthStatus.DEGRADED
        
        if avg_latency > self.config['max_latency_ms']:
            issues.append(f"High latency: {avg_latency:.0f}ms")
            status = HealthStatus.DEGRADED
        
        if avg_slippage > self.config['max_slippage_bps']:
            issues.append(f"High slippage: {avg_slippage:.1f} bps")
            status = HealthStatus.DEGRADED
        
        if rejection_rate > self.config['max_rejection_rate']:
            issues.append(f"High rejection rate: {rejection_rate:.1%}")
            status = HealthStatus.DEGRADED
        
        # Overall status determination
        if len(issues) >= 3:
            status = HealthStatus.CRITICAL
            recommendations.append("Consider switching primary venue")
        elif len(issues) >= 2:
            status = HealthStatus.DEGRADED
            recommendations.append("Monitor closely, consider backup venue")
        elif len(issues) == 1:
            status = HealthStatus.WARNING
            recommendations.append("Monitor and optimize order parameters")
        else:
            status = HealthStatus.HEALTHY
        
        # Create report
        self.venue_health[venue] = VenueHealthReport(
            venue_name=venue,
            status=status,
            fill_rate_pct=fill_rate * 100,
            avg_latency_ms=avg_latency,
            avg_slippage_bps=avg_slippage,
            rejection_rate_pct=rejection_rate * 100,
            uptime_pct=100,  # Would calculate from connection data
            last_update=datetime.now(),
            issues=issues,
            recommendations=recommendations
        )
    
    def _update_system_health(self):
        """Update overall system health status"""
        if not self.venue_health:
            self.system_health = HealthStatus.OFFLINE
            return
        
        venue_statuses = [v.status for v in self.venue_health.values()]
        
        if any(s == HealthStatus.CRITICAL for s in venue_statuses):
            self.system_health = HealthStatus.CRITICAL
        elif any(s == HealthStatus.DEGRADED for s in venue_statuses):
            self.system_health = HealthStatus.DEGRADED
        elif any(s == HealthStatus.WARNING for s in venue_statuses):
            self.system_health = HealthStatus.WARNING
        else:
            self.system_health = HealthStatus.HEALTHY
    
    def _check_degradation_patterns(self):
        """Check for systematic degradation patterns"""
        if len(self.performance_windows['medium']) < 20:
            return
        
        recent_data = list(self.performance_windows['medium'])[-20:]
        older_data = list(self.performance_windows['medium'])[-40:-20] if len(self.performance_windows['medium']) >= 40 else []
        
        if not older_data:
            return
        
        # Compare recent vs. older performance
        recent_fill_rate = statistics.mean([d['fill_rate'] for d in recent_data])
        older_fill_rate = statistics.mean([d['fill_rate'] for d in older_data])
        
        recent_slippage = statistics.mean([d['slippage'] for d in recent_data])
        older_slippage = statistics.mean([d['slippage'] for d in older_data])
        
        # Detect degradation
        fill_rate_degradation = (older_fill_rate - recent_fill_rate) / older_fill_rate > 0.1
        slippage_degradation = (recent_slippage - older_slippage) / older_slippage > 0.2
        
        if fill_rate_degradation or slippage_degradation:
            alert = HealthAlert(
                alert_type=AlertType.EXECUTION_DELAY,
                severity='high',
                order_id='SYSTEM',
                symbol='ALL',
                message=f"System performance degradation detected",
                metrics={
                    'recent_fill_rate': recent_fill_rate,
                    'older_fill_rate': older_fill_rate,
                    'recent_slippage': recent_slippage,
                    'older_slippage': older_slippage
                },
                timestamp=datetime.now(),
                auto_action='review_system_config'
            )
            self._record_alert(alert)
    
    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary"""
        # Recent performance
        if self.performance_windows['short']:
            recent_data = list(self.performance_windows['short'])
            recent_fill_rate = statistics.mean([d['fill_rate'] for d in recent_data])
            recent_slippage = statistics.mean([d['slippage'] for d in recent_data])
            recent_latency = statistics.mean([d['latency'] for d in recent_data])
        else:
            recent_fill_rate = recent_slippage = recent_latency = 0
        
        # Alert summary
        recent_alerts = [a for a in list(self.alerts)[-50:] 
                        if (datetime.now() - a.timestamp).seconds < 3600]  # Last hour
        
        alert_summary = {}
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            if alert_type not in alert_summary:
                alert_summary[alert_type] = {'count': 0, 'max_severity': 'low'}
            alert_summary[alert_type]['count'] += 1
            if alert.severity == 'critical':
                alert_summary[alert_type]['max_severity'] = 'critical'
            elif alert.severity == 'high' and alert_summary[alert_type]['max_severity'] != 'critical':
                alert_summary[alert_type]['max_severity'] = 'high'
        
        return {
            'system_health': self.system_health.value,
            'venue_health': {v: report.status.value for v, report in self.venue_health.items()},
            'recent_performance': {
                'fill_rate_pct': recent_fill_rate * 100,
                'avg_slippage_bps': recent_slippage,
                'avg_latency_ms': recent_latency
            },
            'alert_summary': alert_summary,
            'total_orders_monitored': len(self.execution_metrics),
            'active_venues': len(self.venue_health),
            'last_update': datetime.now().isoformat()
        }
    
    def get_venue_report(self, venue: str) -> Optional[VenueHealthReport]:
        """Get detailed venue health report"""
        return self.venue_health.get(venue)
    
    def get_recent_alerts(self, hours: int = 1) -> List[HealthAlert]:
        """Get recent alerts within specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]
    
    def get_execution_analytics(self, symbol: Optional[str] = None, 
                               hours: int = 24) -> Dict:
        """Get detailed execution analytics"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter data
        relevant_orders = []
        for order_id, metrics in self.execution_metrics.items():
            if metrics.timestamp > cutoff:
                if symbol is None or self.order_intents.get(order_id, {}).symbol == symbol:
                    relevant_orders.append(order_id)
        
        if not relevant_orders:
            return {}
        
        # Calculate analytics
        fill_rates = []
        slippages = []
        latencies = []
        venues = defaultdict(int)
        
        for order_id in relevant_orders:
            metrics = self.execution_metrics[order_id]
            intent = self.order_intents.get(order_id)
            
            if intent:
                fill_rate = metrics.filled_size / intent.intended_size
                fill_rates.append(fill_rate)
            
            slippages.append(metrics.slippage_bps)
            latencies.append(metrics.total_latency_ms)
            venues[metrics.execution_venue] += 1
        
        return {
            'period_hours': hours,
            'total_orders': len(relevant_orders),
            'avg_fill_rate_pct': statistics.mean(fill_rates) * 100 if fill_rates else 0,
            'avg_slippage_bps': statistics.mean(slippages) if slippages else 0,
            'max_slippage_bps': max(slippages) if slippages else 0,
            'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'venue_distribution': dict(venues),
            'success_rate_pct': (sum(1 for r in fill_rates if r > 0.95) / len(fill_rates) * 100) if fill_rates else 0
        }
    
    def reset_metrics(self, venue: Optional[str] = None):
        """Reset metrics for venue or all venues"""
        if venue:
            if venue in self.venue_metrics:
                for metric_queue in self.venue_metrics[venue].values():
                    metric_queue.clear()
                if venue in self.venue_health:
                    del self.venue_health[venue]
        else:
            self.venue_metrics.clear()
            self.venue_health.clear()
            for window in self.performance_windows.values():
                window.clear()
            self.alerts.clear()
        
        self.logger.info(f"Reset health metrics for {'all venues' if not venue else venue}")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None