"""
Risk Governance Agent - Enforces hard risk limits and portfolio guardrails
Controls: max daily loss, position limits, concentration, R-multiples, time blocks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, time
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

class RiskViolationType(Enum):
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_POSITIONS = "max_positions"
    POSITION_SIZE = "position_size"
    SECTOR_CONCENTRATION = "sector_concentration"
    CORRELATION_LIMIT = "correlation_limit"
    TIME_RESTRICTION = "time_restriction"
    LEVERAGE_LIMIT = "leverage_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    WIN_RATE_THRESHOLD = "win_rate_threshold"
    R_MULTIPLE_LIMIT = "r_multiple_limit"

class RiskAction(Enum):
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK_TRADE = "block_trade"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"
    WARNING = "warning"

@dataclass
class RiskLimits:
    max_daily_loss_pct: float = 2.0  # 2% max daily loss
    max_daily_loss_usd: float = 10000  # $10k max daily loss
    max_positions: int = 10
    max_position_size_pct: float = 5.0  # 5% per position
    max_sector_exposure_pct: float = 30.0  # 30% in one sector
    max_correlation: float = 0.7  # Max correlation between positions
    max_leverage: float = 2.0  # 2x leverage max
    max_drawdown_pct: float = 10.0  # 10% max drawdown
    min_win_rate: float = 0.35  # 35% minimum win rate
    max_r_per_trade: float = 2.0  # Max 2R risk per trade
    restricted_times: List[Tuple[time, time]] = field(default_factory=list)
    blacklist_symbols: List[str] = field(default_factory=list)
    require_confirmation_above: float = 10000  # Require confirmation > $10k

@dataclass
class RiskAssessment:
    timestamp: datetime
    trade_id: str
    symbol: str
    action: RiskAction
    violations: List[RiskViolationType]
    current_metrics: Dict[str, float]
    risk_score: float  # 0-100, higher is riskier
    adjusted_size: Optional[float]
    warnings: List[str]
    requirements: List[str]
    emergency: bool

class RiskGovernanceAgent:
    """
    Enforces strict risk management rules with hard guardrails
    """
    
    def __init__(self, config: Optional[RiskLimits] = None):
        self.limits = config or RiskLimits()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.daily_pnl = 0
        self.open_positions = {}
        self.closed_trades_today = []
        self.peak_balance = 0
        self.current_balance = 0
        self.trade_history = []
        self.violations_history = []
        self.emergency_mode = False
        
        # Performance metrics
        self.performance_window = 100  # Last 100 trades
        self.sector_map = self._load_sector_map()
        
    def _load_sector_map(self) -> Dict:
        """Load symbol to sector mapping"""
        # Mock sector mapping - replace with actual data
        return {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'XOM': 'Energy',
            'CVX': 'Energy',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'AMZN': 'Consumer',
            'TSLA': 'Automotive',
            'BTC/USDT': 'Crypto',
            'ETH/USDT': 'Crypto'
        }
    
    def assess_trade(self, trade_request: Dict) -> RiskAssessment:
        """
        Comprehensive risk assessment for a trade request
        """
        violations = []
        warnings = []
        requirements = []
        
        # Extract trade details
        symbol = trade_request['symbol']
        size = trade_request['size']
        side = trade_request['side']  # 'buy' or 'sell'
        entry_price = trade_request.get('entry_price', 0)
        stop_loss = trade_request.get('stop_loss', 0)
        
        # Calculate position value and risk
        position_value = size * entry_price
        risk_amount = abs(entry_price - stop_loss) * size if stop_loss else position_value * 0.02
        
        # Check time restrictions
        if self._is_restricted_time():
            violations.append(RiskViolationType.TIME_RESTRICTION)
            warnings.append(f"Trading restricted at current time")
        
        # Check blacklist
        if symbol in self.limits.blacklist_symbols:
            violations.append(RiskViolationType.POSITION_SIZE)
            warnings.append(f"{symbol} is blacklisted")
        
        # Check daily loss limit
        daily_loss_check = self._check_daily_loss_limit(risk_amount)
        if daily_loss_check['violated']:
            violations.append(RiskViolationType.DAILY_LOSS_LIMIT)
            warnings.append(daily_loss_check['message'])
        
        # Check position limits
        position_check = self._check_position_limits(symbol, position_value)
        if position_check['violated']:
            violations.append(RiskViolationType.MAX_POSITIONS)
            warnings.append(position_check['message'])
        
        # Check position size
        size_check = self._check_position_size(position_value)
        if size_check['violated']:
            violations.append(RiskViolationType.POSITION_SIZE)
            warnings.append(size_check['message'])
        
        # Check sector concentration
        sector_check = self._check_sector_concentration(symbol, position_value)
        if sector_check['violated']:
            violations.append(RiskViolationType.SECTOR_CONCENTRATION)
            warnings.append(sector_check['message'])
        
        # Check correlation
        correlation_check = self._check_correlation_limits(symbol)
        if correlation_check['violated']:
            violations.append(RiskViolationType.CORRELATION_LIMIT)
            warnings.append(correlation_check['message'])
        
        # Check leverage
        leverage_check = self._check_leverage()
        if leverage_check['violated']:
            violations.append(RiskViolationType.LEVERAGE_LIMIT)
            warnings.append(leverage_check['message'])
        
        # Check drawdown
        drawdown_check = self._check_drawdown()
        if drawdown_check['violated']:
            violations.append(RiskViolationType.DRAWDOWN_LIMIT)
            warnings.append(drawdown_check['message'])
        
        # Check R-multiple
        r_check = self._check_r_multiple(risk_amount)
        if r_check['violated']:
            violations.append(RiskViolationType.R_MULTIPLE_LIMIT)
            warnings.append(r_check['message'])
        
        # Check win rate
        winrate_check = self._check_win_rate()
        if winrate_check['below_threshold']:
            warnings.append(winrate_check['message'])
        
        # Determine action
        action = self._determine_action(violations, position_value)
        
        # Calculate adjusted size if needed
        adjusted_size = self._calculate_adjusted_size(
            size, violations, position_value
        ) if action == RiskAction.REDUCE_SIZE else None
        
        # Generate requirements
        if position_value > self.limits.require_confirmation_above:
            requirements.append(f"Manual confirmation required for ${position_value:,.2f} trade")
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            violations, warnings, trade_request
        )
        
        # Check for emergency conditions
        emergency = self._check_emergency_conditions()
        
        assessment = RiskAssessment(
            timestamp=datetime.now(),
            trade_id=trade_request.get('id', 'unknown'),
            symbol=symbol,
            action=action,
            violations=violations,
            current_metrics=self._get_current_metrics(),
            risk_score=risk_score,
            adjusted_size=adjusted_size,
            warnings=warnings,
            requirements=requirements,
            emergency=emergency
        )
        
        # Log assessment
        self.violations_history.append(assessment)
        
        return assessment
    
    def _is_restricted_time(self) -> bool:
        """Check if current time is restricted"""
        current_time = datetime.now().time()
        
        for start_time, end_time in self.limits.restricted_times:
            if start_time <= current_time <= end_time:
                return True
        
        # Default restrictions (first/last 5 minutes of regular trading)
        if time(9, 30) <= current_time <= time(9, 35):
            return True
        if time(15, 55) <= current_time <= time(16, 0):
            return True
            
        return False
    
    def _check_daily_loss_limit(self, new_risk: float) -> Dict:
        """Check if daily loss limit would be exceeded"""
        potential_daily_loss = abs(self.daily_pnl) + new_risk
        
        # Check percentage limit
        if self.current_balance > 0:
            loss_pct = (potential_daily_loss / self.current_balance) * 100
            if loss_pct > self.limits.max_daily_loss_pct:
                return {
                    'violated': True,
                    'message': f"Would exceed daily loss limit: {loss_pct:.1f}% > {self.limits.max_daily_loss_pct}%"
                }
        
        # Check absolute limit
        if potential_daily_loss > self.limits.max_daily_loss_usd:
            return {
                'violated': True,
                'message': f"Would exceed daily loss limit: ${potential_daily_loss:,.0f} > ${self.limits.max_daily_loss_usd:,.0f}"
            }
        
        return {'violated': False}
    
    def _check_position_limits(self, symbol: str, value: float) -> Dict:
        """Check position count limits"""
        current_positions = len(self.open_positions)
        
        if symbol not in self.open_positions:
            if current_positions >= self.limits.max_positions:
                return {
                    'violated': True,
                    'message': f"Max positions reached: {current_positions}/{self.limits.max_positions}"
                }
        
        return {'violated': False}
    
    def _check_position_size(self, position_value: float) -> Dict:
        """Check position size limits"""
        if self.current_balance <= 0:
            return {'violated': False}
        
        position_pct = (position_value / self.current_balance) * 100
        
        if position_pct > self.limits.max_position_size_pct:
            return {
                'violated': True,
                'message': f"Position too large: {position_pct:.1f}% > {self.limits.max_position_size_pct}%"
            }
        
        return {'violated': False}
    
    def _check_sector_concentration(self, symbol: str, value: float) -> Dict:
        """Check sector concentration limits"""
        sector = self.sector_map.get(symbol, 'Unknown')
        
        # Calculate current sector exposure
        sector_exposure = 0
        for sym, pos in self.open_positions.items():
            if self.sector_map.get(sym) == sector:
                sector_exposure += pos['value']
        
        # Add new position
        total_sector = sector_exposure + value
        
        if self.current_balance > 0:
            sector_pct = (total_sector / self.current_balance) * 100
            
            if sector_pct > self.limits.max_sector_exposure_pct:
                return {
                    'violated': True,
                    'message': f"Sector concentration too high: {sector} = {sector_pct:.1f}%"
                }
        
        return {'violated': False}
    
    def _check_correlation_limits(self, symbol: str) -> Dict:
        """Check correlation between positions"""
        # This would need historical price data to calculate actual correlations
        # For now, use a simplified approach
        
        high_correlation_symbols = {
            'SPY': ['QQQ', 'IWM'],
            'GLD': ['SLV', 'GDX'],
            'BTC/USDT': ['ETH/USDT', 'BNB/USDT'],
            'XOM': ['CVX', 'COP']
        }
        
        for sym in self.open_positions:
            correlated = high_correlation_symbols.get(sym, [])
            if symbol in correlated:
                return {
                    'violated': True,
                    'message': f"High correlation with existing position: {sym}"
                }
        
        return {'violated': False}
    
    def _check_leverage(self) -> Dict:
        """Check leverage limits"""
        total_exposure = sum(pos['value'] for pos in self.open_positions.values())
        
        if self.current_balance > 0:
            leverage = total_exposure / self.current_balance
            
            if leverage > self.limits.max_leverage:
                return {
                    'violated': True,
                    'message': f"Leverage too high: {leverage:.1f}x > {self.limits.max_leverage}x"
                }
        
        return {'violated': False}
    
    def _check_drawdown(self) -> Dict:
        """Check drawdown limits"""
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
            
            if drawdown > self.limits.max_drawdown_pct:
                return {
                    'violated': True,
                    'message': f"In maximum drawdown: {drawdown:.1f}%"
                }
        
        return {'violated': False}
    
    def _check_r_multiple(self, risk_amount: float) -> Dict:
        """Check R-multiple limits"""
        if self.current_balance > 0:
            r_size = risk_amount / (self.current_balance * 0.01)  # 1R = 1% of account
            
            if r_size > self.limits.max_r_per_trade:
                return {
                    'violated': True,
                    'message': f"Risk too high: {r_size:.1f}R > {self.limits.max_r_per_trade}R"
                }
        
        return {'violated': False}
    
    def _check_win_rate(self) -> Dict:
        """Check recent win rate"""
        if len(self.trade_history) < 20:
            return {'below_threshold': False}
        
        recent_trades = self.trade_history[-self.performance_window:]
        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        win_rate = wins / len(recent_trades)
        
        if win_rate < self.limits.min_win_rate:
            return {
                'below_threshold': True,
                'message': f"Win rate below threshold: {win_rate:.1%} < {self.limits.min_win_rate:.1%}"
            }
        
        return {'below_threshold': False}
    
    def _determine_action(self, violations: List[RiskViolationType], position_value: float) -> RiskAction:
        """Determine action based on violations"""
        
        # Critical violations - block trade
        critical = [
            RiskViolationType.DAILY_LOSS_LIMIT,
            RiskViolationType.DRAWDOWN_LIMIT,
            RiskViolationType.TIME_RESTRICTION
        ]
        
        if any(v in critical for v in violations):
            return RiskAction.BLOCK_TRADE
        
        # Major violations - reduce size
        major = [
            RiskViolationType.POSITION_SIZE,
            RiskViolationType.LEVERAGE_LIMIT,
            RiskViolationType.R_MULTIPLE_LIMIT
        ]
        
        if any(v in major for v in violations):
            return RiskAction.REDUCE_SIZE
        
        # Minor violations - warning
        if violations:
            return RiskAction.WARNING
        
        return RiskAction.ALLOW
    
    def _calculate_adjusted_size(self, original_size: float, violations: List, 
                                position_value: float) -> float:
        """Calculate adjusted position size to meet risk limits"""
        adjusted = original_size
        
        # Reduce for position size violations
        if RiskViolationType.POSITION_SIZE in violations:
            max_value = self.current_balance * (self.limits.max_position_size_pct / 100)
            reduction_factor = max_value / position_value
            adjusted *= reduction_factor
        
        # Reduce for R-multiple violations
        if RiskViolationType.R_MULTIPLE_LIMIT in violations:
            max_risk = self.current_balance * 0.01 * self.limits.max_r_per_trade
            # Estimate risk as 2% of position
            current_risk = position_value * 0.02
            if current_risk > 0:
                reduction_factor = max_risk / current_risk
                adjusted *= reduction_factor
        
        # Ensure minimum size
        adjusted = max(adjusted, 1)
        
        return round(adjusted, 2)
    
    def _calculate_risk_score(self, violations: List, warnings: List, trade: Dict) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0
        
        # Violation scoring
        violation_scores = {
            RiskViolationType.DAILY_LOSS_LIMIT: 30,
            RiskViolationType.DRAWDOWN_LIMIT: 25,
            RiskViolationType.LEVERAGE_LIMIT: 20,
            RiskViolationType.POSITION_SIZE: 15,
            RiskViolationType.R_MULTIPLE_LIMIT: 15,
            RiskViolationType.MAX_POSITIONS: 10,
            RiskViolationType.SECTOR_CONCENTRATION: 10,
            RiskViolationType.CORRELATION_LIMIT: 10,
            RiskViolationType.TIME_RESTRICTION: 20
        }
        
        for violation in violations:
            score += violation_scores.get(violation, 5)
        
        # Warning scoring
        score += len(warnings) * 3
        
        # Account health scoring
        if self.current_balance > 0:
            # Drawdown contribution
            drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
            score += min(20, drawdown * 2)
            
            # Daily loss contribution
            daily_loss_pct = abs(self.daily_pnl / self.current_balance) * 100
            score += min(15, daily_loss_pct * 5)
        
        # Win rate contribution
        if len(self.trade_history) >= 20:
            recent_trades = self.trade_history[-20:]
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            win_rate = wins / len(recent_trades)
            if win_rate < 0.4:
                score += 10
        
        return min(100, max(0, score))
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        metrics = {
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'total_exposure': sum(pos['value'] for pos in self.open_positions.values())
        }
        
        if self.current_balance > 0:
            metrics['daily_pnl_pct'] = (self.daily_pnl / self.current_balance) * 100
            metrics['leverage'] = metrics['total_exposure'] / self.current_balance
            metrics['drawdown_pct'] = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
        
        if len(self.trade_history) >= 20:
            recent = self.trade_history[-20:]
            metrics['recent_win_rate'] = sum(1 for t in recent if t['pnl'] > 0) / len(recent)
            metrics['avg_win'] = np.mean([t['pnl'] for t in recent if t['pnl'] > 0] or [0])
            metrics['avg_loss'] = np.mean([abs(t['pnl']) for t in recent if t['pnl'] < 0] or [0])
        
        return metrics
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions requiring immediate action"""
        
        # Daily loss approaching limit
        if self.current_balance > 0:
            daily_loss_pct = abs(self.daily_pnl / self.current_balance) * 100
            if daily_loss_pct > self.limits.max_daily_loss_pct * 0.8:
                self.emergency_mode = True
                return True
        
        # Severe drawdown
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
            if drawdown > self.limits.max_drawdown_pct * 0.9:
                self.emergency_mode = True
                return True
        
        # Consecutive losses
        if len(self.trade_history) >= 5:
            last_5 = self.trade_history[-5:]
            if all(t['pnl'] < 0 for t in last_5):
                self.emergency_mode = True
                return True
        
        return False
    
    def update_position(self, symbol: str, position_data: Dict):
        """Update open position"""
        self.open_positions[symbol] = position_data
    
    def close_position(self, symbol: str, pnl: float):
        """Record closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        
        trade_record = {
            'symbol': symbol,
            'pnl': pnl,
            'timestamp': datetime.now()
        }
        
        self.trade_history.append(trade_record)
        self.closed_trades_today.append(trade_record)
        self.daily_pnl += pnl
        
        # Update peak balance
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of trading day)"""
        self.daily_pnl = 0
        self.closed_trades_today = []
        self.emergency_mode = False
    
    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        metrics = self._get_current_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'emergency_mode': self.emergency_mode,
            'metrics': metrics,
            'limits': {
                'daily_loss': f"{self.limits.max_daily_loss_pct}% / ${self.limits.max_daily_loss_usd:,.0f}",
                'max_positions': self.limits.max_positions,
                'max_leverage': f"{self.limits.max_leverage}x",
                'max_drawdown': f"{self.limits.max_drawdown_pct}%"
            },
            'current_violations': [v.value for v in self.violations_history[-1].violations] if self.violations_history else [],
            'risk_score': self.violations_history[-1].risk_score if self.violations_history else 0,
            'recommendations': self._get_risk_recommendations()
        }
    
    def _get_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if self.emergency_mode:
            recommendations.append("EMERGENCY: Close all positions and stop trading")
        
        metrics = self._get_current_metrics()
        
        if metrics.get('daily_pnl_pct', 0) < -1:
            recommendations.append("Reduce position sizes for remainder of day")
        
        if metrics.get('leverage', 0) > 1.5:
            recommendations.append("Consider reducing leverage")
        
        if metrics.get('recent_win_rate', 1) < 0.4:
            recommendations.append("Review strategy - win rate below acceptable threshold")
        
        if len(self.open_positions) > self.limits.max_positions * 0.8:
            recommendations.append("Approaching position limit - be selective with new trades")
        
        return recommendations