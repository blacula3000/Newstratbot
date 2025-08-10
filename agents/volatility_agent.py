"""
Volatility Agent - Advanced Volatility Analysis and IV Rank Integration
Comprehensive volatility monitoring for dynamic risk management
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

class VolatilityRegime(Enum):
    EXTREMELY_LOW = "extremely_low"  # <10th percentile
    LOW = "low"  # 10th-25th percentile
    NORMAL = "normal"  # 25th-75th percentile
    HIGH = "high"  # 75th-90th percentile
    EXTREMELY_HIGH = "extremely_high"  # >90th percentile

class VolatilityTrend(Enum):
    EXPANDING = "expanding"  # Volatility increasing
    CONTRACTING = "contracting"  # Volatility decreasing
    STABLE = "stable"  # Volatility stable

class IVRankLevel(Enum):
    VERY_LOW = "very_low"  # 0-20
    LOW = "low"  # 20-40
    MODERATE = "moderate"  # 40-60
    HIGH = "high"  # 60-80
    VERY_HIGH = "very_high"  # 80-100

@dataclass
class VolatilityMetrics:
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Core volatility measures
    realized_volatility: float  # Historical volatility
    garch_volatility: float  # GARCH model volatility
    parkinson_volatility: float  # Parkinson estimator (high-low)
    garman_klass_volatility: float  # Garman-Klass estimator
    
    # IV Rank and percentiles
    iv_rank: float  # 0-100 implied volatility rank
    iv_percentile: float  # Current IV percentile vs history
    realized_vs_implied: float  # RV vs IV differential
    
    # Volatility structure
    volatility_regime: VolatilityRegime
    volatility_trend: VolatilityTrend
    trend_strength: float  # 1-10 strength of vol trend
    
    # Risk metrics
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    max_drawdown_risk: float  # Expected max drawdown
    tail_risk: float  # Tail risk measure
    
    # Volatility surface metrics
    term_structure: Dict[str, float]  # Vol across time horizons
    skew: float  # Volatility skew
    smile: float  # Volatility smile coefficient
    
    # Trading implications
    optimal_position_size: float  # Vol-adjusted position size
    risk_budget: float  # Available risk budget
    sharpe_expectation: float  # Expected Sharpe ratio
    kelly_fraction: float  # Kelly criterion optimal size

@dataclass
class VolatilitySignal:
    signal_id: str
    signal_type: str  # 'vol_expansion', 'vol_contraction', 'regime_change'
    symbol: str
    timeframe: str
    confidence: float  # 0-100
    direction: str  # 'bullish_vol', 'bearish_vol', 'neutral'
    current_iv_rank: float
    expected_move: float  # Expected price move
    time_horizon: str
    risk_adjustment: float  # Position size multiplier
    opportunity_score: float  # 1-10 opportunity rating
    metadata: Dict
    timestamp: datetime

class VolatilityAgent:
    """
    Advanced Volatility Agent for comprehensive volatility analysis
    
    Analyzes:
    - Multiple volatility estimators (realized, GARCH, Parkinson, etc.)
    - Implied volatility rank and percentiles
    - Volatility regime classification
    - Term structure and volatility surface
    - Risk metrics (VaR, CVaR, tail risk)
    - Position sizing based on volatility
    - Volatility-based trading signals
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.volatility_history = {}  # {symbol: {timeframe: deque[VolatilityMetrics]}}
        self.iv_history = {}  # {symbol: deque[float]} - IV history for rank calculation
        self.regime_changes = {}  # {symbol: deque[regime_change_events]}
        
        # Model state
        self.garch_models = {}  # {symbol: garch_parameters}
        self.vol_forecasts = {}  # {symbol: forecast_data}
        
        # Performance tracking
        self.vol_forecast_accuracy = deque(maxlen=100)
        self.regime_prediction_accuracy = deque(maxlen=50)
        self.position_sizing_performance = deque(maxlen=100)
        
        self.logger.info("📊 Volatility Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            'lookback_periods': {
                'short': 10,
                'medium': 30,
                'long': 252  # ~1 year of trading days
            },
            'iv_rank_period': 252,  # 1 year for IV rank calculation
            'volatility_window': 20,  # Rolling window for vol calculation
            'garch_lags': {'p': 1, 'q': 1},  # GARCH(1,1) default
            'var_confidence': 0.95,  # 95% VaR confidence level
            'regime_change_threshold': 1.5,  # Standard deviations for regime change
            'vol_expansion_threshold': 1.3,  # Multiplier for vol expansion detection
            'vol_contraction_threshold': 0.7,  # Multiplier for vol contraction
            'min_signal_confidence': 70,
            'position_sizing_method': 'kelly_fractional',  # 'fixed', 'vol_target', 'kelly'
            'base_vol_target': 0.15,  # 15% annualized vol target
            'max_position_size': 0.1,  # 10% max position size
            'min_position_size': 0.01,  # 1% min position size
            'risk_free_rate': 0.02,  # 2% risk-free rate
            'business_days_per_year': 252,
            'rebalance_threshold': 0.2  # 20% change triggers rebalance
        }
    
    def analyze_volatility(self, symbol: str, timeframe: str, 
                          data: pd.DataFrame, iv_data: Optional[pd.DataFrame] = None) -> VolatilityMetrics:
        """
        Comprehensive volatility analysis
        """
        try:
            # Initialize tracking
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = {}
                self.iv_history[symbol] = deque(maxlen=self.config['iv_rank_period'])
                self.regime_changes[symbol] = deque(maxlen=50)
                self.garch_models[symbol] = None
                self.vol_forecasts[symbol] = {}
            
            if timeframe not in self.volatility_history[symbol]:
                self.volatility_history[symbol][timeframe] = deque(maxlen=100)
            
            # Step 1: Calculate core volatility measures
            vol_measures = self._calculate_volatility_measures(data)
            
            # Step 2: Calculate IV metrics if available
            iv_metrics = self._calculate_iv_metrics(symbol, iv_data)
            
            # Step 3: Determine volatility regime and trend
            regime, trend, trend_strength = self._analyze_volatility_regime(
                symbol, timeframe, vol_measures['realized_volatility']
            )
            
            # Step 4: Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(data, vol_measures)
            
            # Step 5: Analyze volatility structure
            structure_metrics = self._analyze_volatility_structure(data, vol_measures)
            
            # Step 6: Calculate trading implications
            trading_metrics = self._calculate_trading_implications(
                vol_measures, risk_metrics, data
            )
            
            # Create comprehensive metrics
            metrics = VolatilityMetrics(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                
                # Core measures
                realized_volatility=vol_measures['realized_volatility'],
                garch_volatility=vol_measures['garch_volatility'],
                parkinson_volatility=vol_measures['parkinson_volatility'],
                garman_klass_volatility=vol_measures['garman_klass_volatility'],
                
                # IV metrics
                iv_rank=iv_metrics['iv_rank'],
                iv_percentile=iv_metrics['iv_percentile'],
                realized_vs_implied=iv_metrics['rv_vs_iv'],
                
                # Structure
                volatility_regime=regime,
                volatility_trend=trend,
                trend_strength=trend_strength,
                
                # Risk
                var_95=risk_metrics['var_95'],
                cvar_95=risk_metrics['cvar_95'],
                max_drawdown_risk=risk_metrics['max_drawdown_risk'],
                tail_risk=risk_metrics['tail_risk'],
                
                # Structure metrics
                term_structure=structure_metrics['term_structure'],
                skew=structure_metrics['skew'],
                smile=structure_metrics['smile'],
                
                # Trading implications
                optimal_position_size=trading_metrics['optimal_position_size'],
                risk_budget=trading_metrics['risk_budget'],
                sharpe_expectation=trading_metrics['sharpe_expectation'],
                kelly_fraction=trading_metrics['kelly_fraction']
            )
            
            # Store metrics
            self.volatility_history[symbol][timeframe].append(metrics)
            
            # Update IV history if available
            if iv_metrics['current_iv'] is not None:
                self.iv_history[symbol].append(iv_metrics['current_iv'])
            
            self.logger.debug(f"📊 Volatility analysis complete: {symbol} {timeframe} - "
                            f"RV: {vol_measures['realized_volatility']:.1%}, "
                            f"Regime: {regime.value}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility for {symbol} {timeframe}: {e}")
            return self._empty_metrics(symbol, timeframe)
    
    def _calculate_volatility_measures(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volatility estimators"""
        measures = {}
        
        try:
            if len(data) < 20:
                return {key: 0.0 for key in ['realized_volatility', 'garch_volatility', 
                                           'parkinson_volatility', 'garman_klass_volatility']}
            
            # 1. Realized Volatility (Close-to-Close)
            returns = data['close'].pct_change().dropna()
            measures['realized_volatility'] = returns.std() * np.sqrt(self.config['business_days_per_year'])
            
            # 2. Parkinson Volatility (High-Low)
            if 'high' in data.columns and 'low' in data.columns:
                hl_ratio = np.log(data['high'] / data['low'])
                parkinson_var = (hl_ratio ** 2).mean() / (4 * np.log(2))
                measures['parkinson_volatility'] = np.sqrt(parkinson_var * self.config['business_days_per_year'])
            else:
                measures['parkinson_volatility'] = measures['realized_volatility']
            
            # 3. Garman-Klass Volatility
            if all(col in data.columns for col in ['high', 'low', 'open', 'close']):
                gk_var = self._calculate_garman_klass(data)
                measures['garman_klass_volatility'] = np.sqrt(gk_var * self.config['business_days_per_year'])
            else:
                measures['garman_klass_volatility'] = measures['realized_volatility']
            
            # 4. GARCH Volatility
            measures['garch_volatility'] = self._calculate_garch_volatility(returns)
            
            return measures
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility measures: {e}")
            return {key: 0.15 for key in ['realized_volatility', 'garch_volatility', 
                                        'parkinson_volatility', 'garman_klass_volatility']}
    
    def _calculate_garman_klass(self, data: pd.DataFrame) -> float:
        """Calculate Garman-Klass volatility estimator"""
        try:
            ln_hl = np.log(data['high'] / data['low'])
            ln_co = np.log(data['close'] / data['open'])
            
            gk_var = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)
            return gk_var.mean()
            
        except Exception:
            return (data['close'].pct_change() ** 2).mean()
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """Calculate GARCH(1,1) volatility forecast"""
        try:
            if len(returns) < 30:
                return returns.std() * np.sqrt(self.config['business_days_per_year'])
            
            # Simplified GARCH(1,1) estimation
            # In production, would use proper GARCH library
            
            # Initial parameters (typical GARCH values)
            omega = 0.000001  # Long-term variance
            alpha = 0.1       # ARCH coefficient
            beta = 0.85       # GARCH coefficient
            
            # Calculate conditional variance
            returns_squared = returns ** 2
            conditional_var = np.zeros(len(returns))
            
            # Initialize with sample variance
            conditional_var[0] = returns_squared.iloc[0]
            
            for i in range(1, len(returns)):
                conditional_var[i] = (omega + 
                                    alpha * returns_squared.iloc[i-1] + 
                                    beta * conditional_var[i-1])
            
            # Current volatility forecast
            current_vol = np.sqrt(conditional_var[-1] * self.config['business_days_per_year'])
            
            return current_vol
            
        except Exception:
            return returns.std() * np.sqrt(self.config['business_days_per_year'])
    
    def _calculate_iv_metrics(self, symbol: str, iv_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate IV rank and related metrics"""
        metrics = {
            'iv_rank': 50.0,
            'iv_percentile': 50.0,
            'rv_vs_iv': 0.0,
            'current_iv': None
        }
        
        try:
            if iv_data is None or iv_data.empty:
                return metrics
            
            current_iv = iv_data['implied_volatility'].iloc[-1] if 'implied_volatility' in iv_data.columns else None
            
            if current_iv is not None:
                metrics['current_iv'] = current_iv
                
                # Calculate IV Rank (requires historical IV data)
                if len(self.iv_history[symbol]) > 20:
                    iv_history = list(self.iv_history[symbol])
                    iv_rank = (sum(1 for iv in iv_history if iv < current_iv) / len(iv_history)) * 100
                    metrics['iv_rank'] = iv_rank
                
                # Calculate IV Percentile (more recent focus)
                recent_iv = list(self.iv_history[symbol])[-60:] if len(self.iv_history[symbol]) >= 60 else list(self.iv_history[symbol])
                if len(recent_iv) > 10:
                    iv_percentile = (sum(1 for iv in recent_iv if iv < current_iv) / len(recent_iv)) * 100
                    metrics['iv_percentile'] = iv_percentile
                
                # RV vs IV (if we have realized vol)
                if symbol in self.volatility_history:
                    for timeframe_data in self.volatility_history[symbol].values():
                        if timeframe_data:
                            latest_rv = timeframe_data[-1].realized_volatility
                            metrics['rv_vs_iv'] = (current_iv - latest_rv) / latest_rv
                            break
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating IV metrics: {e}")
            return metrics
    
    def _analyze_volatility_regime(self, symbol: str, timeframe: str, current_vol: float) -> Tuple[VolatilityRegime, VolatilityTrend, float]:
        """Analyze volatility regime and trend"""
        try:
            regime = VolatilityRegime.NORMAL
            trend = VolatilityTrend.STABLE
            trend_strength = 5.0
            
            # Get historical volatility data
            if (symbol in self.volatility_history and 
                timeframe in self.volatility_history[symbol] and
                len(self.volatility_history[symbol][timeframe]) > 10):
                
                historical_vols = [m.realized_volatility for m in self.volatility_history[symbol][timeframe]]
                
                # Calculate percentiles for regime classification
                vol_10th = np.percentile(historical_vols, 10)
                vol_25th = np.percentile(historical_vols, 25)
                vol_75th = np.percentile(historical_vols, 75)
                vol_90th = np.percentile(historical_vols, 90)
                
                # Classify regime
                if current_vol <= vol_10th:
                    regime = VolatilityRegime.EXTREMELY_LOW
                elif current_vol <= vol_25th:
                    regime = VolatilityRegime.LOW
                elif current_vol <= vol_75th:
                    regime = VolatilityRegime.NORMAL
                elif current_vol <= vol_90th:
                    regime = VolatilityRegime.HIGH
                else:
                    regime = VolatilityRegime.EXTREMELY_HIGH
                
                # Analyze trend
                if len(historical_vols) >= 5:
                    recent_vols = historical_vols[-5:]
                    vol_trend = np.polyfit(range(len(recent_vols)), recent_vols, 1)[0]
                    
                    vol_mean = np.mean(historical_vols)
                    normalized_trend = vol_trend / vol_mean if vol_mean > 0 else 0
                    
                    if normalized_trend > 0.1:
                        trend = VolatilityTrend.EXPANDING
                        trend_strength = min(normalized_trend * 50, 10)
                    elif normalized_trend < -0.1:
                        trend = VolatilityTrend.CONTRACTING
                        trend_strength = min(abs(normalized_trend) * 50, 10)
                    else:
                        trend = VolatilityTrend.STABLE
                        trend_strength = 5 - abs(normalized_trend) * 25
            
            return regime, trend, trend_strength
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regime: {e}")
            return VolatilityRegime.NORMAL, VolatilityTrend.STABLE, 5.0
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, vol_measures: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return {
                    'var_95': 0.02,
                    'cvar_95': 0.03,
                    'max_drawdown_risk': 0.1,
                    'tail_risk': 0.05
                }
            
            # 1. Value at Risk (95%)
            var_95 = np.percentile(returns, (1 - self.config['var_confidence']) * 100)
            risk_metrics['var_95'] = abs(var_95)
            
            # 2. Conditional Value at Risk (Expected Shortfall)
            tail_returns = returns[returns <= var_95]
            cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
            risk_metrics['cvar_95'] = abs(cvar_95)
            
            # 3. Maximum Drawdown Risk (based on volatility)
            current_vol = vol_measures.get('realized_volatility', 0.2)
            # Rough estimate: max drawdown ≈ 2.5 * volatility for normal markets
            risk_metrics['max_drawdown_risk'] = current_vol * 2.5
            
            # 4. Tail Risk (excess kurtosis measure)
            if len(returns) >= 30:
                kurtosis = returns.kurtosis()
                tail_risk = max(0, (kurtosis - 3) / 10)  # Normalize excess kurtosis
                risk_metrics['tail_risk'] = min(tail_risk, 0.5)
            else:
                risk_metrics['tail_risk'] = 0.1
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {
                'var_95': 0.02,
                'cvar_95': 0.03,
                'max_drawdown_risk': 0.1,
                'tail_risk': 0.05
            }
    
    def _analyze_volatility_structure(self, data: pd.DataFrame, vol_measures: Dict[str, float]) -> Dict[str, Any]:
        """Analyze volatility term structure and surface characteristics"""
        structure = {
            'term_structure': {'1d': 0.0, '1w': 0.0, '1m': 0.0},
            'skew': 0.0,
            'smile': 0.0
        }
        
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 30:
                return structure
            
            # Term structure (different time horizons)
            for period, days in [('1d', 1), ('1w', 5), ('1m', 20)]:
                if len(returns) >= days * 2:
                    period_vol = returns.rolling(window=days).std().iloc[-1] * np.sqrt(self.config['business_days_per_year'])
                    structure['term_structure'][period] = period_vol if not pd.isna(period_vol) else vol_measures['realized_volatility']
                else:
                    structure['term_structure'][period] = vol_measures['realized_volatility']
            
            # Volatility skew (asymmetry in return distribution)
            structure['skew'] = returns.skew() if len(returns) >= 20 else 0.0
            
            # Volatility smile (simplified measure based on tail behavior)
            if len(returns) >= 50:
                left_tail = returns.quantile(0.1)
                right_tail = returns.quantile(0.9)
                center = returns.median()
                
                # Measure how extreme tails compare to center
                smile_measure = (abs(left_tail - center) + abs(right_tail - center)) / (2 * returns.std())
                structure['smile'] = max(0, smile_measure - 1)  # Excess beyond normal distribution
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility structure: {e}")
            return structure
    
    def _calculate_trading_implications(self, vol_measures: Dict[str, float], 
                                      risk_metrics: Dict[str, float], 
                                      data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading implications from volatility analysis"""
        implications = {}
        
        try:
            returns = data['close'].pct_change().dropna()
            current_vol = vol_measures.get('realized_volatility', 0.2)
            
            # 1. Optimal Position Size (Vol Targeting)
            target_vol = self.config['base_vol_target']
            vol_scalar = target_vol / current_vol if current_vol > 0 else 1.0
            base_position_size = vol_scalar * 0.05  # 5% base position
            
            # Apply position size limits
            optimal_position_size = max(
                self.config['min_position_size'],
                min(base_position_size, self.config['max_position_size'])
            )
            implications['optimal_position_size'] = optimal_position_size
            
            # 2. Risk Budget (based on VaR)
            var_95 = risk_metrics.get('var_95', 0.02)
            daily_risk_budget = var_95 * optimal_position_size
            implications['risk_budget'] = daily_risk_budget
            
            # 3. Sharpe Expectation (based on vol regime)
            # Lower vol environments typically have lower expected Sharpe ratios
            vol_percentile = min(current_vol / 0.4, 1.0)  # Normalize to 40% max vol
            base_sharpe = 0.5  # Base expected Sharpe ratio
            vol_adjustment = 1 - (vol_percentile * 0.3)  # Reduce Sharpe in high vol
            implications['sharpe_expectation'] = base_sharpe * vol_adjustment
            
            # 4. Kelly Fraction (simplified Kelly criterion)
            if len(returns) >= 20:
                mean_return = returns.mean()
                return_variance = returns.var()
                
                if return_variance > 0:
                    # Kelly fraction = (mean - rf) / variance
                    excess_return = mean_return - (self.config['risk_free_rate'] / 252)
                    kelly_fraction = excess_return / return_variance
                    
                    # Apply Kelly fraction limits (usually cap at 25%)
                    kelly_fraction = max(-0.05, min(kelly_fraction, 0.25))
                    implications['kelly_fraction'] = kelly_fraction
                else:
                    implications['kelly_fraction'] = 0.05
            else:
                implications['kelly_fraction'] = 0.05
            
            return implications
            
        except Exception as e:
            self.logger.error(f"Error calculating trading implications: {e}")
            return {
                'optimal_position_size': 0.03,
                'risk_budget': 0.01,
                'sharpe_expectation': 0.4,
                'kelly_fraction': 0.05
            }
    
    def generate_volatility_signals(self, symbol: str, timeframe: str, 
                                   metrics: VolatilityMetrics, 
                                   price_data: pd.DataFrame) -> List[VolatilitySignal]:
        """Generate volatility-based trading signals"""
        signals = []
        
        try:
            current_price = price_data['close'].iloc[-1]
            
            # Signal 1: Volatility Expansion
            if (metrics.volatility_trend == VolatilityTrend.EXPANDING and 
                metrics.trend_strength >= 7 and
                metrics.iv_rank < 50):  # Low IV rank suggests expansion opportunity
                
                signal = self._create_volatility_signal(
                    symbol, timeframe, 'vol_expansion', metrics, current_price
                )
                if signal.confidence >= self.config['min_signal_confidence']:
                    signals.append(signal)
            
            # Signal 2: Volatility Contraction
            if (metrics.volatility_trend == VolatilityTrend.CONTRACTING and
                metrics.trend_strength >= 7 and
                metrics.iv_rank > 70):  # High IV rank suggests contraction opportunity
                
                signal = self._create_volatility_signal(
                    symbol, timeframe, 'vol_contraction', metrics, current_price
                )
                if signal.confidence >= self.config['min_signal_confidence']:
                    signals.append(signal)
            
            # Signal 3: Regime Change
            if self._detect_regime_change(symbol, timeframe, metrics):
                signal = self._create_volatility_signal(
                    symbol, timeframe, 'regime_change', metrics, current_price
                )
                if signal.confidence >= self.config['min_signal_confidence']:
                    signals.append(signal)
            
            # Signal 4: Mean Reversion
            if (metrics.volatility_regime in [VolatilityRegime.EXTREMELY_HIGH, VolatilityRegime.EXTREMELY_LOW] and
                metrics.volatility_trend == VolatilityTrend.STABLE):
                
                signal = self._create_volatility_signal(
                    symbol, timeframe, 'vol_mean_reversion', metrics, current_price
                )
                if signal.confidence >= self.config['min_signal_confidence']:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating volatility signals: {e}")
            return []
    
    def _create_volatility_signal(self, symbol: str, timeframe: str, signal_type: str, 
                                 metrics: VolatilityMetrics, current_price: float) -> VolatilitySignal:
        """Create a specific volatility signal"""
        try:
            signal_id = f"vol_{signal_type}_{symbol}_{timeframe}_{int(datetime.now().timestamp())}"
            
            # Base confidence from trend strength and regime
            confidence = 60 + (metrics.trend_strength / 10) * 20
            
            # Signal-specific logic
            if signal_type == 'vol_expansion':
                direction = 'bullish_vol'
                expected_move = metrics.realized_volatility * 1.5
                risk_adjustment = 0.8  # Reduce position size for vol expansion
                opportunity_score = min(8, 4 + metrics.trend_strength / 2)
                confidence += 10 if metrics.iv_rank < 30 else 0
                
            elif signal_type == 'vol_contraction':
                direction = 'bearish_vol'
                expected_move = metrics.realized_volatility * 0.7
                risk_adjustment = 1.2  # Increase position size for vol contraction
                opportunity_score = min(8, 4 + metrics.trend_strength / 2)
                confidence += 10 if metrics.iv_rank > 70 else 0
                
            elif signal_type == 'regime_change':
                direction = 'neutral'
                expected_move = metrics.realized_volatility * 1.2
                risk_adjustment = 0.6  # Very conservative for regime changes
                opportunity_score = 7  # High opportunity but high risk
                confidence += 15  # Regime changes are significant
                
            elif signal_type == 'vol_mean_reversion':
                if metrics.volatility_regime == VolatilityRegime.EXTREMELY_HIGH:
                    direction = 'bearish_vol'
                    expected_move = metrics.realized_volatility * 0.8
                    risk_adjustment = 0.9
                else:
                    direction = 'bullish_vol'
                    expected_move = metrics.realized_volatility * 1.1
                    risk_adjustment = 1.1
                    
                opportunity_score = 6
                confidence += 5  # Moderate boost for mean reversion
            
            else:
                direction = 'neutral'
                expected_move = metrics.realized_volatility
                risk_adjustment = 1.0
                opportunity_score = 5
            
            # Time horizon based on signal type
            time_horizon_map = {
                'vol_expansion': '1-3 days',
                'vol_contraction': '3-7 days',
                'regime_change': '1-2 weeks',
                'vol_mean_reversion': '3-10 days'
            }
            
            return VolatilitySignal(
                signal_id=signal_id,
                signal_type=signal_type,
                symbol=symbol,
                timeframe=timeframe,
                confidence=min(confidence, 95),
                direction=direction,
                current_iv_rank=metrics.iv_rank,
                expected_move=expected_move,
                time_horizon=time_horizon_map.get(signal_type, '1 week'),
                risk_adjustment=risk_adjustment,
                opportunity_score=opportunity_score,
                metadata={
                    'volatility_regime': metrics.volatility_regime.value,
                    'volatility_trend': metrics.volatility_trend.value,
                    'trend_strength': metrics.trend_strength,
                    'current_vol': metrics.realized_volatility,
                    'optimal_position_size': metrics.optimal_position_size
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating volatility signal: {e}")
            return self._empty_signal(symbol, timeframe)
    
    def _detect_regime_change(self, symbol: str, timeframe: str, metrics: VolatilityMetrics) -> bool:
        """Detect if a volatility regime change is occurring"""
        try:
            if (symbol not in self.volatility_history or 
                timeframe not in self.volatility_history[symbol] or
                len(self.volatility_history[symbol][timeframe]) < 5):
                return False
            
            recent_metrics = list(self.volatility_history[symbol][timeframe])[-5:]
            
            # Check for regime consistency in recent history
            recent_regimes = [m.volatility_regime for m in recent_metrics]
            current_regime = metrics.volatility_regime
            
            # If current regime is different from majority of recent regimes
            regime_counts = {regime: recent_regimes.count(regime) for regime in set(recent_regimes)}
            most_common_regime = max(regime_counts, key=regime_counts.get)
            
            return current_regime != most_common_regime and regime_counts[most_common_regime] >= 3
            
        except Exception:
            return False
    
    def get_position_size_recommendation(self, symbol: str, base_position_size: float, 
                                       current_vol: Optional[float] = None) -> float:
        """Get volatility-adjusted position size recommendation"""
        try:
            # Get latest volatility metrics
            if (symbol in self.volatility_history and 
                self.volatility_history[symbol]):
                
                # Get most recent metrics across all timeframes
                latest_metrics = None
                for timeframe_data in self.volatility_history[symbol].values():
                    if timeframe_data:
                        if latest_metrics is None or timeframe_data[-1].timestamp > latest_metrics.timestamp:
                            latest_metrics = timeframe_data[-1]
                
                if latest_metrics:
                    return latest_metrics.optimal_position_size * (base_position_size / 0.05)  # Scale from base
            
            # Fallback: simple vol scaling
            if current_vol:
                target_vol = self.config['base_vol_target']
                vol_scalar = target_vol / current_vol if current_vol > 0 else 1.0
                adjusted_size = base_position_size * vol_scalar
                
                return max(
                    self.config['min_position_size'],
                    min(adjusted_size, self.config['max_position_size'])
                )
            
            return base_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size recommendation: {e}")
            return base_position_size
    
    def get_risk_budget(self, symbol: str, position_size: float) -> Dict[str, float]:
        """Get risk budget recommendations"""
        try:
            risk_budget = {
                'daily_var': position_size * 0.02,  # Default 2% daily VaR
                'max_loss': position_size * 0.05,   # Default 5% max loss
                'portfolio_heat': position_size * 0.15  # Default 15% portfolio heat
            }
            
            # Get latest volatility metrics for more precise risk budgeting
            if (symbol in self.volatility_history and 
                self.volatility_history[symbol]):
                
                latest_metrics = None
                for timeframe_data in self.volatility_history[symbol].values():
                    if timeframe_data:
                        if latest_metrics is None or timeframe_data[-1].timestamp > latest_metrics.timestamp:
                            latest_metrics = timeframe_data[-1]
                
                if latest_metrics:
                    risk_budget.update({
                        'daily_var': position_size * latest_metrics.var_95,
                        'max_loss': position_size * latest_metrics.cvar_95 * 2,
                        'portfolio_heat': position_size * latest_metrics.max_drawdown_risk
                    })
            
            return risk_budget
            
        except Exception as e:
            self.logger.error(f"Error calculating risk budget: {e}")
            return {
                'daily_var': position_size * 0.02,
                'max_loss': position_size * 0.05,
                'portfolio_heat': position_size * 0.15
            }
    
    def _empty_metrics(self, symbol: str, timeframe: str) -> VolatilityMetrics:
        """Return empty metrics for error cases"""
        return VolatilityMetrics(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            realized_volatility=0.2,
            garch_volatility=0.2,
            parkinson_volatility=0.2,
            garman_klass_volatility=0.2,
            iv_rank=50.0,
            iv_percentile=50.0,
            realized_vs_implied=0.0,
            volatility_regime=VolatilityRegime.NORMAL,
            volatility_trend=VolatilityTrend.STABLE,
            trend_strength=5.0,
            var_95=0.02,
            cvar_95=0.03,
            max_drawdown_risk=0.1,
            tail_risk=0.05,
            term_structure={'1d': 0.2, '1w': 0.2, '1m': 0.2},
            skew=0.0,
            smile=0.0,
            optimal_position_size=0.03,
            risk_budget=0.01,
            sharpe_expectation=0.4,
            kelly_fraction=0.05
        )
    
    def _empty_signal(self, symbol: str, timeframe: str) -> VolatilitySignal:
        """Return empty signal for error cases"""
        return VolatilitySignal(
            signal_id=f"empty_{symbol}_{timeframe}",
            signal_type='none',
            symbol=symbol,
            timeframe=timeframe,
            confidence=0,
            direction='neutral',
            current_iv_rank=50,
            expected_move=0.02,
            time_horizon='1 day',
            risk_adjustment=1.0,
            opportunity_score=0,
            metadata={},
            timestamp=datetime.now()
        )
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            total_symbols = len(self.volatility_history)
            total_analyses = sum(
                len(timeframes.get(tf, []))
                for timeframes in self.volatility_history.values()
                for tf in timeframes
            )
            
            forecast_accuracy = np.mean(self.vol_forecast_accuracy) if self.vol_forecast_accuracy else 0
            regime_accuracy = np.mean(self.regime_prediction_accuracy) if self.regime_prediction_accuracy else 0
            
            return {
                'agent_name': 'Volatility Agent',
                'status': 'active',
                'symbols_tracked': total_symbols,
                'total_analyses': total_analyses,
                'vol_forecast_accuracy': f"{forecast_accuracy:.1%}",
                'regime_prediction_accuracy': f"{regime_accuracy:.1%}",
                'base_vol_target': f"{self.config['base_vol_target']:.1%}",
                'iv_rank_period': self.config['iv_rank_period'],
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Volatility Agent', 'status': 'error'}
    
    def update_forecast_accuracy(self, predicted_vol: float, actual_vol: float):
        """Update volatility forecast accuracy tracking"""
        try:
            if predicted_vol > 0 and actual_vol > 0:
                accuracy = 1 - abs(predicted_vol - actual_vol) / actual_vol
                self.vol_forecast_accuracy.append(max(0, accuracy))
                
        except Exception as e:
            self.logger.error(f"Error updating forecast accuracy: {e}")
    
    def update_regime_prediction(self, predicted_regime: str, actual_regime: str):
        """Update regime prediction accuracy"""
        try:
            accuracy = 1.0 if predicted_regime == actual_regime else 0.0
            self.regime_prediction_accuracy.append(accuracy)
            
        except Exception as e:
            self.logger.error(f"Error updating regime prediction: {e}")