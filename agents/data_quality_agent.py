"""
Data Quality Agent - Ensures data integrity and flags anomalies
Monitors: bad ticks, split/dividend adjustments, illiquid candles, wide spreads
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class DataQualityIssue(Enum):
    BAD_TICK = "bad_tick"
    WIDE_SPREAD = "wide_spread"
    ILLIQUID_CANDLE = "illiquid_candle"
    PRICE_SPIKE = "price_spike"
    MISSING_DATA = "missing_data"
    SPLIT_DETECTED = "split_detected"
    DIVIDEND_DETECTED = "dividend_detected"
    STALE_DATA = "stale_data"
    NEGATIVE_PRICE = "negative_price"
    ZERO_VOLUME = "zero_volume"

@dataclass
class DataQualityReport:
    timestamp: datetime
    symbol: str
    timeframe: str
    issues: List[DataQualityIssue]
    severity: str  # 'critical', 'warning', 'info'
    details: Dict
    action_required: bool
    recommended_action: str

class DataQualityAgent:
    """
    Validates market data integrity and flags anomalies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.issue_history = []
        self.split_dividend_cache = {}
        
    def _default_config(self) -> Dict:
        return {
            'max_spread_pct': 0.5,  # 0.5% max spread
            'min_volume_threshold': 100,  # Minimum volume for liquidity
            'spike_threshold': 0.1,  # 10% price spike threshold
            'stale_data_minutes': 5,  # Data older than 5 minutes is stale
            'lookback_periods': 20,  # Periods for statistical validation
            'z_score_threshold': 3,  # Standard deviations for outlier detection
            'split_detection_threshold': 0.4,  # 40% price change for split detection
            'max_issues_before_halt': 5,  # Stop trading after this many critical issues
        }
    
    def validate_candle(self, candle: Dict, historical_data: pd.DataFrame = None) -> DataQualityReport:
        """
        Comprehensive validation of a single candle
        """
        issues = []
        details = {}
        severity = 'info'
        
        # Basic data validation
        if candle['high'] < candle['low']:
            issues.append(DataQualityIssue.BAD_TICK)
            details['high_low_mismatch'] = f"High {candle['high']} < Low {candle['low']}"
            severity = 'critical'
            
        if candle['close'] < 0 or candle['open'] < 0:
            issues.append(DataQualityIssue.NEGATIVE_PRICE)
            severity = 'critical'
            
        if candle['volume'] == 0:
            issues.append(DataQualityIssue.ZERO_VOLUME)
            details['zero_volume'] = True
            severity = 'warning'
            
        # Spread validation
        spread = self._calculate_spread(candle)
        if spread > self.config['max_spread_pct']:
            issues.append(DataQualityIssue.WIDE_SPREAD)
            details['spread_pct'] = spread
            severity = 'warning'
            
        # Liquidity check
        if candle['volume'] < self.config['min_volume_threshold']:
            issues.append(DataQualityIssue.ILLIQUID_CANDLE)
            details['volume'] = candle['volume']
            severity = 'warning'
            
        # Statistical anomaly detection
        if historical_data is not None and len(historical_data) > self.config['lookback_periods']:
            anomalies = self._detect_statistical_anomalies(candle, historical_data)
            issues.extend(anomalies['issues'])
            details.update(anomalies['details'])
            if anomalies['issues']:
                severity = max(severity, 'warning')
                
        # Create report
        action_required = severity == 'critical'
        recommended_action = self._get_recommended_action(issues, severity)
        
        report = DataQualityReport(
            timestamp=datetime.now(),
            symbol=candle.get('symbol', 'UNKNOWN'),
            timeframe=candle.get('timeframe', '1m'),
            issues=issues,
            severity=severity,
            details=details,
            action_required=action_required,
            recommended_action=recommended_action
        )
        
        self.issue_history.append(report)
        return report
    
    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> List[DataQualityReport]:
        """
        Validate entire dataframe of OHLCV data
        """
        reports = []
        
        # Check for missing data
        if df.isnull().any().any():
            missing_report = DataQualityReport(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe='unknown',
                issues=[DataQualityIssue.MISSING_DATA],
                severity='warning',
                details={'missing_data': df.isnull().sum().to_dict()},
                action_required=False,
                recommended_action="Fill or interpolate missing data"
            )
            reports.append(missing_report)
        
        # Check for potential splits/dividends
        split_div_check = self._detect_splits_dividends(df, symbol)
        if split_div_check:
            reports.append(split_div_check)
        
        # Validate individual candles
        for idx, row in df.iterrows():
            candle = row.to_dict()
            candle['symbol'] = symbol
            
            # Get historical context
            hist_data = df.loc[:idx].tail(self.config['lookback_periods'] + 1)
            report = self.validate_candle(candle, hist_data)
            
            if report.issues:
                reports.append(report)
                
        return reports
    
    def _calculate_spread(self, candle: Dict) -> float:
        """
        Calculate bid-ask spread as percentage
        """
        if 'bid' in candle and 'ask' in candle:
            mid_price = (candle['bid'] + candle['ask']) / 2
            spread = (candle['ask'] - candle['bid']) / mid_price * 100
        else:
            # Estimate from high-low
            spread = (candle['high'] - candle['low']) / candle['close'] * 100
        return spread
    
    def _detect_statistical_anomalies(self, candle: Dict, historical_data: pd.DataFrame) -> Dict:
        """
        Detect statistical anomalies using z-score and other methods
        """
        issues = []
        details = {}
        
        # Calculate statistics
        close_prices = historical_data['close'].values
        mean_close = np.mean(close_prices)
        std_close = np.std(close_prices)
        
        # Z-score check
        if std_close > 0:
            z_score = (candle['close'] - mean_close) / std_close
            if abs(z_score) > self.config['z_score_threshold']:
                issues.append(DataQualityIssue.PRICE_SPIKE)
                details['z_score'] = z_score
                details['expected_range'] = (
                    mean_close - self.config['z_score_threshold'] * std_close,
                    mean_close + self.config['z_score_threshold'] * std_close
                )
        
        # Volume anomaly
        volume_mean = historical_data['volume'].mean()
        volume_std = historical_data['volume'].std()
        if volume_std > 0:
            volume_z_score = (candle['volume'] - volume_mean) / volume_std
            if abs(volume_z_score) > self.config['z_score_threshold']:
                details['volume_anomaly'] = {
                    'z_score': volume_z_score,
                    'expected_volume': volume_mean
                }
        
        return {'issues': issues, 'details': details}
    
    def _detect_splits_dividends(self, df: pd.DataFrame, symbol: str) -> Optional[DataQualityReport]:
        """
        Detect potential stock splits or dividend adjustments
        """
        if len(df) < 2:
            return None
            
        # Calculate daily returns
        returns = df['close'].pct_change()
        
        # Look for extreme price changes
        extreme_changes = returns[abs(returns) > self.config['split_detection_threshold']]
        
        if not extreme_changes.empty:
            issues = []
            details = {}
            
            for idx, change in extreme_changes.items():
                if change < -self.config['split_detection_threshold']:
                    issues.append(DataQualityIssue.SPLIT_DETECTED)
                    details[f'potential_split_{idx}'] = {
                        'date': idx,
                        'price_change_pct': change * 100,
                        'ratio_estimate': 1 / (1 + change)
                    }
                elif change > self.config['split_detection_threshold']:
                    issues.append(DataQualityIssue.DIVIDEND_DETECTED)
                    details[f'potential_dividend_{idx}'] = {
                        'date': idx,
                        'price_change_pct': change * 100
                    }
            
            if issues:
                return DataQualityReport(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    timeframe='daily',
                    issues=issues,
                    severity='critical',
                    details=details,
                    action_required=True,
                    recommended_action="Verify and adjust for corporate actions"
                )
        
        return None
    
    def _get_recommended_action(self, issues: List[DataQualityIssue], severity: str) -> str:
        """
        Provide recommended action based on issues found
        """
        if severity == 'critical':
            if DataQualityIssue.BAD_TICK in issues:
                return "HALT TRADING - Bad tick detected, verify data source"
            elif DataQualityIssue.SPLIT_DETECTED in issues:
                return "HALT TRADING - Potential split detected, adjust positions"
            elif DataQualityIssue.NEGATIVE_PRICE in issues:
                return "HALT TRADING - Invalid price data"
        elif severity == 'warning':
            if DataQualityIssue.WIDE_SPREAD in issues:
                return "Use limit orders only, widen stops"
            elif DataQualityIssue.ILLIQUID_CANDLE in issues:
                return "Reduce position size or avoid trading"
            elif DataQualityIssue.PRICE_SPIKE in issues:
                return "Wait for confirmation, check news"
        
        return "Monitor closely"
    
    def check_data_freshness(self, last_update: datetime) -> bool:
        """
        Check if data is fresh enough for trading
        """
        age_minutes = (datetime.now() - last_update).total_seconds() / 60
        if age_minutes > self.config['stale_data_minutes']:
            report = DataQualityReport(
                timestamp=datetime.now(),
                symbol='SYSTEM',
                timeframe='N/A',
                issues=[DataQualityIssue.STALE_DATA],
                severity='critical',
                details={'data_age_minutes': age_minutes},
                action_required=True,
                recommended_action="Refresh data connection"
            )
            self.issue_history.append(report)
            return False
        return True
    
    def get_quality_score(self, symbol: str, lookback_hours: int = 24) -> float:
        """
        Calculate overall data quality score for a symbol
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_issues = [
            r for r in self.issue_history 
            if r.symbol == symbol and r.timestamp > cutoff_time
        ]
        
        if not recent_issues:
            return 100.0
        
        # Calculate score based on severity and frequency
        total_issues = len(recent_issues)
        critical_issues = sum(1 for r in recent_issues if r.severity == 'critical')
        warning_issues = sum(1 for r in recent_issues if r.severity == 'warning')
        
        # Weighted scoring
        score = 100.0
        score -= critical_issues * 20  # Heavy penalty for critical issues
        score -= warning_issues * 5    # Moderate penalty for warnings
        score -= (total_issues - critical_issues - warning_issues) * 1  # Light penalty for info
        
        return max(0, min(100, score))
    
    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Determine if trading should proceed based on data quality
        """
        quality_score = self.get_quality_score(symbol, lookback_hours=1)
        
        # Check recent critical issues
        recent_critical = [
            r for r in self.issue_history[-10:]
            if r.symbol == symbol and r.severity == 'critical'
        ]
        
        if recent_critical:
            return False, f"Critical data quality issues: {recent_critical[0].recommended_action}"
        
        if quality_score < 70:
            return False, f"Data quality score too low: {quality_score:.1f}"
        
        if quality_score < 85:
            return True, f"Proceed with caution - quality score: {quality_score:.1f}"
        
        return True, f"Data quality good: {quality_score:.1f}"