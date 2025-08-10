"""
Position Sizing Agent - Advanced Portfolio Risk Management
Dynamic position sizing with volatility normalization and multi-factor risk controls
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

class SizingMethod(Enum):
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_TARGET = "volatility_target"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    MAX_DRAWDOWN_TARGET = "max_drawdown_target"
    SHARPE_OPTIMIZATION = "sharpe_optimization"

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class PortfolioHeat(Enum):
    COLD = "cold"        # <25% heat
    WARM = "warm"        # 25-50% heat
    HOT = "hot"          # 50-75% heat
    OVERHEATED = "overheated"  # >75% heat

@dataclass
class PositionSizeCalculation:
    symbol: str
    base_size: float
    adjusted_size: float
    sizing_method: SizingMethod
    risk_level: RiskLevel
    
    # Risk factors
    volatility_adjustment: float
    correlation_adjustment: float
    liquidity_adjustment: float
    concentration_adjustment: float
    heat_adjustment: float
    
    # Risk metrics
    expected_var: float
    expected_max_loss: float
    portfolio_contribution: float
    marginal_var: float
    
    # Constraints applied
    min_size_constraint: bool
    max_size_constraint: bool
    portfolio_limit_constraint: bool
    correlation_limit_constraint: bool
    
    # Metadata
    confidence: float
    reasoning: str
    timestamp: datetime

@dataclass
class PortfolioRiskMetrics:
    total_exposure: float
    portfolio_heat: PortfolioHeat
    portfolio_var: float
    portfolio_sharpe: float
    max_drawdown_estimate: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    tail_risk: float
    risk_budget_used: float
    diversification_ratio: float
    active_positions: int
    timestamp: datetime

@dataclass
class RiskBudget:
    daily_var_budget: float
    max_loss_budget: float
    drawdown_budget: float
    concentration_budget: float
    sector_budget: Dict[str, float]
    correlation_budget: float
    used_budgets: Dict[str, float]
    available_budgets: Dict[str, float]

class PositionSizingAgent:
    """
    Advanced Position Sizing Agent with comprehensive risk management
    
    Features:
    - Multiple sizing methodologies (Kelly, Vol Target, Risk Parity)
    - Volatility normalization and regime adjustment
    - Portfolio heat monitoring and constraint enforcement
    - Correlation and concentration risk management
    - Dynamic risk budgeting and allocation
    - Multi-factor position adjustment
    - Real-time portfolio risk monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.current_positions = {}  # {symbol: position_data}
        self.portfolio_history = deque(maxlen=252)  # Daily portfolio snapshots
        self.risk_budgets = {}  # Risk budget tracking
        self.correlation_matrix = {}  # Symbol correlation matrix
        
        # Performance tracking
        self.sizing_accuracy = deque(maxlen=100)
        self.risk_adjusted_returns = deque(maxlen=252)
        self.drawdown_predictions = deque(maxlen=50)
        
        self.logger.info("📏 Position Sizing Agent initialized")
    
    def _default_config(self) -> Dict:
        return {
            # Base sizing parameters
            'default_risk_per_trade': 0.02,  # 2% risk per trade
            'max_position_size': 0.08,       # 8% max position
            'min_position_size': 0.005,      # 0.5% min position
            'max_portfolio_heat': 0.75,      # 75% max portfolio heat
            
            # Risk budgets
            'daily_var_budget': 0.03,        # 3% daily VaR budget
            'max_loss_budget': 0.15,         # 15% max loss budget
            'drawdown_budget': 0.20,         # 20% max drawdown
            'concentration_budget': 0.25,     # 25% max in single position
            
            # Volatility targeting
            'target_volatility': 0.15,       # 15% target portfolio vol
            'vol_lookback': 20,              # Days for vol calculation
            'vol_adjustment_factor': 1.5,    # Vol adjustment sensitivity
            
            # Correlation controls
            'max_correlation': 0.7,          # Max correlation between positions
            'correlation_penalty': 0.8,      # Position size reduction for correlation
            'correlation_lookback': 60,      # Days for correlation calculation
            
            # Liquidity controls
            'min_liquidity_score': 50,       # Minimum liquidity score
            'liquidity_adjustment_factor': 0.3,  # Liquidity adjustment sensitivity
            
            # Kelly criterion parameters
            'kelly_lookback': 252,           # 1 year for Kelly calculation
            'kelly_safety_factor': 0.5,     # Reduce Kelly by 50% for safety
            'max_kelly_fraction': 0.15,     # 15% max Kelly fraction
            
            # Risk level mappings
            'risk_level_multipliers': {
                RiskLevel.CONSERVATIVE: 0.5,
                RiskLevel.MODERATE: 1.0,
                RiskLevel.AGGRESSIVE: 1.5,
                RiskLevel.MAXIMUM: 2.0
            },
            
            # Rebalancing
            'rebalance_threshold': 0.25,     # 25% change triggers rebalance
            'rebalance_frequency_hours': 24, # Daily rebalancing
            
            # Business parameters
            'business_days_per_year': 252,
            'risk_free_rate': 0.02
        }
    
    def calculate_position_size(self, symbol: str, signal_data: Dict, 
                              portfolio_data: Dict, volatility_data: Optional[Dict] = None) -> PositionSizeCalculation:
        """
        Calculate optimal position size using multiple factors
        """
        try:
            # Extract signal information
            base_size = signal_data.get('base_position_size', 0.03)
            sizing_method = SizingMethod(signal_data.get('sizing_method', 'volatility_target'))
            risk_level = RiskLevel(signal_data.get('risk_level', 'moderate'))
            
            # Step 1: Calculate base position size using chosen method
            method_size = self._calculate_method_size(symbol, signal_data, sizing_method, volatility_data)
            
            # Step 2: Apply volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(symbol, volatility_data)
            vol_adjusted_size = method_size * volatility_adjustment
            
            # Step 3: Apply correlation adjustment
            correlation_adjustment = self._calculate_correlation_adjustment(symbol, portfolio_data)
            corr_adjusted_size = vol_adjusted_size * correlation_adjustment
            
            # Step 4: Apply liquidity adjustment
            liquidity_adjustment = self._calculate_liquidity_adjustment(symbol, signal_data)
            liq_adjusted_size = corr_adjusted_size * liquidity_adjustment
            
            # Step 5: Apply concentration adjustment
            concentration_adjustment = self._calculate_concentration_adjustment(symbol, portfolio_data)
            conc_adjusted_size = liq_adjusted_size * concentration_adjustment
            
            # Step 6: Apply portfolio heat adjustment
            heat_adjustment = self._calculate_heat_adjustment(portfolio_data)
            heat_adjusted_size = conc_adjusted_size * heat_adjustment
            
            # Step 7: Apply risk level multiplier
            risk_multiplier = self.config['risk_level_multipliers'][risk_level]
            risk_adjusted_size = heat_adjusted_size * risk_multiplier
            
            # Step 8: Apply hard constraints
            final_size, constraints_applied = self._apply_size_constraints(
                risk_adjusted_size, symbol, portfolio_data
            )
            
            # Step 9: Calculate risk metrics
            risk_metrics = self._calculate_position_risk_metrics(
                symbol, final_size, portfolio_data, volatility_data
            )
            
            # Step 10: Generate reasoning and confidence
            reasoning, confidence = self._generate_sizing_reasoning(
                symbol, base_size, final_size, volatility_adjustment, 
                correlation_adjustment, constraints_applied
            )
            
            calculation = PositionSizeCalculation(
                symbol=symbol,
                base_size=base_size,
                adjusted_size=final_size,
                sizing_method=sizing_method,
                risk_level=risk_level,
                
                # Adjustments
                volatility_adjustment=volatility_adjustment,
                correlation_adjustment=correlation_adjustment,
                liquidity_adjustment=liquidity_adjustment,
                concentration_adjustment=concentration_adjustment,
                heat_adjustment=heat_adjustment,
                
                # Risk metrics
                expected_var=risk_metrics['expected_var'],
                expected_max_loss=risk_metrics['expected_max_loss'],
                portfolio_contribution=risk_metrics['portfolio_contribution'],
                marginal_var=risk_metrics['marginal_var'],
                
                # Constraints
                min_size_constraint=constraints_applied['min_size'],
                max_size_constraint=constraints_applied['max_size'],
                portfolio_limit_constraint=constraints_applied['portfolio_limit'],
                correlation_limit_constraint=constraints_applied['correlation_limit'],
                
                # Metadata
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            self.logger.debug(f"📏 Position sizing complete: {symbol} - "
                            f"Base: {base_size:.1%} → Final: {final_size:.1%}")
            
            return calculation
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return self._default_position_calculation(symbol)
    
    def _calculate_method_size(self, symbol: str, signal_data: Dict, 
                             method: SizingMethod, volatility_data: Optional[Dict]) -> float:
        """Calculate position size based on chosen method"""
        try:
            if method == SizingMethod.FIXED_PERCENT:
                return signal_data.get('base_position_size', 0.03)
            
            elif method == SizingMethod.VOLATILITY_TARGET:
                return self._calculate_vol_target_size(symbol, signal_data, volatility_data)
            
            elif method == SizingMethod.KELLY_CRITERION:
                return self._calculate_kelly_size(symbol, signal_data)
            
            elif method == SizingMethod.RISK_PARITY:
                return self._calculate_risk_parity_size(symbol, signal_data, volatility_data)
            
            elif method == SizingMethod.MAX_DRAWDOWN_TARGET:
                return self._calculate_drawdown_target_size(symbol, signal_data, volatility_data)
            
            elif method == SizingMethod.SHARPE_OPTIMIZATION:
                return self._calculate_sharpe_optimal_size(symbol, signal_data, volatility_data)
            
            else:
                return signal_data.get('base_position_size', 0.03)
                
        except Exception:
            return signal_data.get('base_position_size', 0.03)
    
    def _calculate_vol_target_size(self, symbol: str, signal_data: Dict, 
                                  volatility_data: Optional[Dict]) -> float:
        """Calculate size based on volatility targeting"""
        try:
            target_vol = self.config['target_volatility']
            
            # Get asset volatility
            if volatility_data and 'realized_volatility' in volatility_data:
                asset_vol = volatility_data['realized_volatility']
            else:
                # Fallback: estimate from price data or use default
                asset_vol = 0.25  # Default 25% volatility
            
            # Calculate position size to achieve target volatility
            if asset_vol > 0:
                vol_target_size = target_vol / asset_vol
                # Scale by typical position size (e.g., 5% base)
                position_size = vol_target_size * 0.05
            else:
                position_size = 0.03
            
            return min(position_size, 0.15)  # Cap at 15%
            
        except Exception:
            return 0.03
    
    def _calculate_kelly_size(self, symbol: str, signal_data: Dict) -> float:
        """Calculate Kelly criterion optimal size"""
        try:
            # Kelly fraction = (expected_return - risk_free_rate) / variance
            expected_return = signal_data.get('expected_return', 0.05)  # 5% expected return
            confidence = signal_data.get('confidence_score', 70) / 100
            
            # Adjust expected return by confidence
            adjusted_return = expected_return * confidence
            
            # Estimate return variance (simplified)
            return_variance = signal_data.get('return_variance', 0.04)  # Default variance
            
            # Calculate Kelly fraction
            excess_return = adjusted_return - (self.config['risk_free_rate'] / 252)
            
            if return_variance > 0:
                kelly_fraction = excess_return / return_variance
                # Apply safety factor and constraints
                safe_kelly = kelly_fraction * self.config['kelly_safety_factor']
                return min(max(safe_kelly, 0), self.config['max_kelly_fraction'])
            
            return 0.03
            
        except Exception:
            return 0.03
    
    def _calculate_risk_parity_size(self, symbol: str, signal_data: Dict, 
                                   volatility_data: Optional[Dict]) -> float:
        """Calculate risk parity position size"""
        try:
            # Risk parity: equal risk contribution from each position
            target_risk_contribution = 1.0 / signal_data.get('total_positions', 10)
            
            # Get asset volatility
            if volatility_data and 'realized_volatility' in volatility_data:
                asset_vol = volatility_data['realized_volatility']
            else:
                asset_vol = 0.25
            
            # Calculate size for equal risk contribution
            if asset_vol > 0:
                risk_parity_size = (target_risk_contribution * 0.15) / asset_vol
                return min(risk_parity_size, 0.08)
            
            return 0.03
            
        except Exception:
            return 0.03
    
    def _calculate_drawdown_target_size(self, symbol: str, signal_data: Dict, 
                                       volatility_data: Optional[Dict]) -> float:
        """Calculate size based on maximum drawdown target"""
        try:
            max_drawdown_target = self.config['drawdown_budget']
            stop_loss_distance = signal_data.get('stop_loss_distance_pct', 0.03)  # 3% default
            
            if stop_loss_distance > 0:
                # Position size = max_drawdown_target / stop_loss_distance
                drawdown_size = max_drawdown_target / stop_loss_distance
                return min(drawdown_size, 0.10)
            
            return 0.03
            
        except Exception:
            return 0.03
    
    def _calculate_sharpe_optimal_size(self, symbol: str, signal_data: Dict, 
                                      volatility_data: Optional[Dict]) -> float:
        """Calculate Sharpe ratio optimized position size"""
        try:
            expected_return = signal_data.get('expected_return', 0.05)
            expected_sharpe = signal_data.get('expected_sharpe', 0.5)
            
            # Get volatility
            if volatility_data and 'realized_volatility' in volatility_data:
                asset_vol = volatility_data['realized_volatility']
            else:
                asset_vol = expected_return / expected_sharpe if expected_sharpe > 0 else 0.25
            
            # Optimal size for Sharpe maximization (simplified)
            if asset_vol > 0 and expected_sharpe > 0:
                # Size proportional to Sharpe ratio
                sharpe_size = (expected_sharpe / 2.0) * 0.08  # Scale by 8% max
                return min(max(sharpe_size, 0.01), 0.08)
            
            return 0.03
            
        except Exception:
            return 0.03
    
    def _calculate_volatility_adjustment(self, symbol: str, volatility_data: Optional[Dict]) -> float:
        """Calculate volatility-based position adjustment"""
        try:
            if not volatility_data:
                return 1.0
            
            current_vol = volatility_data.get('realized_volatility', 0.2)
            vol_regime = volatility_data.get('volatility_regime', 'normal')
            vol_trend = volatility_data.get('volatility_trend', 'stable')
            
            # Base adjustment based on volatility level
            target_vol = self.config['target_volatility']
            if current_vol > 0:
                vol_ratio = target_vol / current_vol
                base_adjustment = 0.5 + 0.5 * vol_ratio  # Scale between 0.5 and 1.5
            else:
                base_adjustment = 1.0
            
            # Regime adjustments
            regime_multipliers = {
                'extremely_low': 1.2,
                'low': 1.1,
                'normal': 1.0,
                'high': 0.9,
                'extremely_high': 0.7
            }
            regime_adjustment = regime_multipliers.get(vol_regime, 1.0)
            
            # Trend adjustments
            trend_multipliers = {
                'expanding': 0.8,    # Reduce size when vol is expanding
                'stable': 1.0,
                'contracting': 1.1   # Increase size when vol is contracting
            }
            trend_adjustment = trend_multipliers.get(vol_trend, 1.0)
            
            # Combine adjustments
            final_adjustment = base_adjustment * regime_adjustment * trend_adjustment
            
            # Apply bounds
            return min(max(final_adjustment, 0.3), 2.0)
            
        except Exception:
            return 1.0
    
    def _calculate_correlation_adjustment(self, symbol: str, portfolio_data: Dict) -> float:
        """Calculate correlation-based position adjustment"""
        try:
            current_positions = portfolio_data.get('current_positions', {})
            
            if not current_positions:
                return 1.0
            
            # Calculate average correlation with existing positions
            correlations = []
            for existing_symbol, position_info in current_positions.items():
                if existing_symbol != symbol:
                    correlation = self._get_correlation(symbol, existing_symbol)
                    if correlation is not None:
                        # Weight by position size
                        weight = position_info.get('size', 0)
                        correlations.append(correlation * weight)
            
            if not correlations:
                return 1.0
            
            avg_correlation = np.mean([abs(c) for c in correlations])
            
            # Apply penalty for high correlation
            if avg_correlation > self.config['max_correlation']:
                penalty = 1 - ((avg_correlation - self.config['max_correlation']) * 2)
                return max(penalty, 0.3)  # Minimum 30% size
            
            # Gradual reduction as correlation increases
            correlation_adjustment = 1 - (avg_correlation * self.config['correlation_penalty'] * 0.5)
            return min(max(correlation_adjustment, 0.5), 1.0)
            
        except Exception:
            return 1.0
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get correlation between two symbols"""
        try:
            # This would integrate with actual correlation calculation
            # For now, return a placeholder
            if symbol1 == symbol2:
                return 1.0
            
            # Mock correlation based on symbol similarity
            # In production, would use actual price correlation
            if (symbol1.startswith('BTC') and symbol2.startswith('BTC')) or \
               (symbol1.startswith('ETH') and symbol2.startswith('ETH')):
                return 0.8
            elif symbol1[:3] == symbol2[:3]:  # Same base currency
                return 0.6
            else:
                return 0.2
                
        except Exception:
            return 0.3  # Default moderate correlation
    
    def _calculate_liquidity_adjustment(self, symbol: str, signal_data: Dict) -> float:
        """Calculate liquidity-based adjustment"""
        try:
            liquidity_score = signal_data.get('liquidity_score', 75)
            min_liquidity = self.config['min_liquidity_score']
            
            if liquidity_score < min_liquidity:
                # Reduce position size for low liquidity
                liquidity_ratio = liquidity_score / min_liquidity
                adjustment = 0.5 + 0.5 * liquidity_ratio
                return max(adjustment, 0.2)
            
            # Bonus for high liquidity
            if liquidity_score > 90:
                return 1.1
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_concentration_adjustment(self, symbol: str, portfolio_data: Dict) -> float:
        """Calculate concentration-based adjustment"""
        try:
            current_positions = portfolio_data.get('current_positions', {})
            total_exposure = sum(pos.get('size', 0) for pos in current_positions.values())
            
            concentration_budget = self.config['concentration_budget']
            current_symbol_exposure = current_positions.get(symbol, {}).get('size', 0)
            
            # Check if adding this position would exceed concentration limit
            new_total_exposure = total_exposure
            available_concentration = concentration_budget - current_symbol_exposure
            
            if available_concentration <= 0:
                return 0.1  # Very small position if already at limit
            
            # Gradual reduction as we approach concentration limit
            concentration_usage = current_symbol_exposure / concentration_budget
            if concentration_usage > 0.8:  # Above 80% of limit
                reduction = 1 - ((concentration_usage - 0.8) * 2)
                return max(reduction, 0.3)
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_heat_adjustment(self, portfolio_data: Dict) -> float:
        """Calculate portfolio heat-based adjustment"""
        try:
            portfolio_metrics = self._calculate_portfolio_heat(portfolio_data)
            current_heat = portfolio_metrics['heat_percentage']
            max_heat = self.config['max_portfolio_heat']
            
            # Reduce position sizes as portfolio gets hot
            heat_ratio = current_heat / max_heat
            
            if heat_ratio > 1.0:  # Overheated
                return 0.2  # Very small positions only
            elif heat_ratio > 0.8:  # Hot
                return 0.5
            elif heat_ratio > 0.6:  # Warm
                return 0.8
            else:  # Cold
                return 1.0
                
        except Exception:
            return 0.8  # Conservative default
    
    def _calculate_portfolio_heat(self, portfolio_data: Dict) -> Dict[str, float]:
        """Calculate current portfolio heat metrics"""
        try:
            current_positions = portfolio_data.get('current_positions', {})
            
            total_exposure = 0
            total_risk = 0
            
            for symbol, position_info in current_positions.items():
                position_size = position_info.get('size', 0)
                position_risk = position_info.get('var_contribution', position_size * 0.02)
                
                total_exposure += position_size
                total_risk += position_risk
            
            # Heat metrics
            heat_percentage = total_exposure
            risk_percentage = total_risk
            
            return {
                'heat_percentage': heat_percentage,
                'risk_percentage': risk_percentage,
                'exposure_count': len(current_positions),
                'avg_position_size': total_exposure / max(len(current_positions), 1)
            }
            
        except Exception:
            return {
                'heat_percentage': 0.3,
                'risk_percentage': 0.02,
                'exposure_count': 0,
                'avg_position_size': 0
            }
    
    def _apply_size_constraints(self, size: float, symbol: str, portfolio_data: Dict) -> Tuple[float, Dict[str, bool]]:
        """Apply hard position size constraints"""
        try:
            original_size = size
            constraints_applied = {
                'min_size': False,
                'max_size': False,
                'portfolio_limit': False,
                'correlation_limit': False
            }
            
            # Minimum size constraint
            if size < self.config['min_position_size']:
                size = self.config['min_position_size']
                constraints_applied['min_size'] = True
            
            # Maximum size constraint
            if size > self.config['max_position_size']:
                size = self.config['max_position_size']
                constraints_applied['max_size'] = True
            
            # Portfolio heat constraint
            current_positions = portfolio_data.get('current_positions', {})
            total_exposure = sum(pos.get('size', 0) for pos in current_positions.values())
            
            if total_exposure + size > self.config['max_portfolio_heat']:
                available_capacity = max(0, self.config['max_portfolio_heat'] - total_exposure)
                if available_capacity < size:
                    size = available_capacity
                    constraints_applied['portfolio_limit'] = True
            
            # Concentration constraint
            concentration_limit = self.config['concentration_budget']
            current_symbol_size = current_positions.get(symbol, {}).get('size', 0)
            
            if current_symbol_size + size > concentration_limit:
                available_concentration = max(0, concentration_limit - current_symbol_size)
                if available_concentration < size:
                    size = available_concentration
                    constraints_applied['correlation_limit'] = True
            
            return max(size, 0), constraints_applied
            
        except Exception:
            return max(size, self.config['min_position_size']), {
                'min_size': False, 'max_size': False,
                'portfolio_limit': False, 'correlation_limit': False
            }
    
    def _calculate_position_risk_metrics(self, symbol: str, position_size: float, 
                                        portfolio_data: Dict, volatility_data: Optional[Dict]) -> Dict[str, float]:
        """Calculate risk metrics for the position"""
        try:
            # Get volatility
            if volatility_data and 'realized_volatility' in volatility_data:
                asset_vol = volatility_data['realized_volatility']
                var_95 = volatility_data.get('var_95', asset_vol * 1.645)  # 95% VaR
            else:
                asset_vol = 0.25
                var_95 = asset_vol * 1.645
            
            # Expected VaR contribution
            expected_var = position_size * var_95
            
            # Expected maximum loss (using CVaR)
            cvar_multiplier = 1.5  # CVaR is typically 1.5x VaR
            expected_max_loss = expected_var * cvar_multiplier
            
            # Portfolio contribution
            current_positions = portfolio_data.get('current_positions', {})
            total_portfolio_size = sum(pos.get('size', 0) for pos in current_positions.values())
            portfolio_contribution = position_size / max(total_portfolio_size + position_size, 0.01)
            
            # Marginal VaR (simplified)
            marginal_var = expected_var * (1 + portfolio_contribution)
            
            return {
                'expected_var': expected_var,
                'expected_max_loss': expected_max_loss,
                'portfolio_contribution': portfolio_contribution,
                'marginal_var': marginal_var
            }
            
        except Exception:
            return {
                'expected_var': position_size * 0.02,
                'expected_max_loss': position_size * 0.05,
                'portfolio_contribution': 0.1,
                'marginal_var': position_size * 0.025
            }
    
    def _generate_sizing_reasoning(self, symbol: str, base_size: float, final_size: float,
                                  vol_adj: float, corr_adj: float, constraints: Dict[str, bool]) -> Tuple[str, float]:
        """Generate reasoning and confidence for position sizing"""
        try:
            reasoning_parts = []
            confidence = 80  # Base confidence
            
            # Size change reasoning
            size_change = (final_size - base_size) / base_size if base_size > 0 else 0
            
            if size_change > 0.2:
                reasoning_parts.append(f"Increased size by {size_change:.1%}")
                confidence += 5
            elif size_change < -0.2:
                reasoning_parts.append(f"Reduced size by {abs(size_change):.1%}")
                confidence -= 5
            
            # Adjustment reasoning
            if vol_adj < 0.8:
                reasoning_parts.append("High volatility reduction applied")
                confidence += 10
            elif vol_adj > 1.2:
                reasoning_parts.append("Low volatility increase applied")
                confidence += 5
            
            if corr_adj < 0.8:
                reasoning_parts.append("Correlation penalty applied")
                confidence -= 5
            
            # Constraint reasoning
            active_constraints = [k for k, v in constraints.items() if v]
            if active_constraints:
                reasoning_parts.append(f"Constraints: {', '.join(active_constraints)}")
                confidence -= len(active_constraints) * 3
            
            # Generate final reasoning
            if reasoning_parts:
                reasoning = f"{symbol} sizing: " + "; ".join(reasoning_parts)
            else:
                reasoning = f"{symbol} sizing: Standard allocation applied"
            
            return reasoning, min(max(confidence, 30), 95)
            
        except Exception:
            return f"{symbol} sizing: Default calculation used", 70
    
    def calculate_portfolio_metrics(self, portfolio_data: Dict) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            current_positions = portfolio_data.get('current_positions', {})
            
            # Basic exposure metrics
            total_exposure = sum(pos.get('size', 0) for pos in current_positions.values())
            active_positions = len([pos for pos in current_positions.values() if pos.get('size', 0) > 0])
            
            # Portfolio heat classification
            heat_pct = total_exposure
            if heat_pct > 0.75:
                portfolio_heat = PortfolioHeat.OVERHEATED
            elif heat_pct > 0.50:
                portfolio_heat = PortfolioHeat.HOT
            elif heat_pct > 0.25:
                portfolio_heat = PortfolioHeat.WARM
            else:
                portfolio_heat = PortfolioHeat.COLD
            
            # Risk metrics (simplified calculations)
            portfolio_var = self._calculate_portfolio_var(current_positions)
            portfolio_sharpe = self._estimate_portfolio_sharpe(current_positions)
            max_drawdown_estimate = self._estimate_max_drawdown(current_positions)
            
            # Risk decomposition
            correlation_risk = self._calculate_correlation_risk(current_positions)
            concentration_risk = self._calculate_concentration_risk(current_positions)
            liquidity_risk = self._calculate_liquidity_risk(current_positions)
            tail_risk = self._calculate_tail_risk(current_positions)
            
            # Risk budget utilization
            risk_budget_used = portfolio_var / self.config['daily_var_budget']
            
            # Diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(current_positions)
            
            return PortfolioRiskMetrics(
                total_exposure=total_exposure,
                portfolio_heat=portfolio_heat,
                portfolio_var=portfolio_var,
                portfolio_sharpe=portfolio_sharpe,
                max_drawdown_estimate=max_drawdown_estimate,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                tail_risk=tail_risk,
                risk_budget_used=risk_budget_used,
                diversification_ratio=diversification_ratio,
                active_positions=active_positions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return self._default_portfolio_metrics()
    
    def _calculate_portfolio_var(self, positions: Dict) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            total_var = 0
            
            for symbol, position in positions.items():
                position_size = position.get('size', 0)
                position_vol = position.get('volatility', 0.25)
                position_var = position_size * position_vol * 1.645  # 95% VaR
                total_var += position_var ** 2  # Sum of squared VaRs (simplified)
            
            return math.sqrt(total_var)
            
        except Exception:
            return 0.02  # 2% default VaR
    
    def _estimate_portfolio_sharpe(self, positions: Dict) -> float:
        """Estimate portfolio Sharpe ratio"""
        try:
            if not positions:
                return 0.0
            
            weighted_sharpe = 0
            total_weight = 0
            
            for symbol, position in positions.items():
                weight = position.get('size', 0)
                expected_sharpe = position.get('expected_sharpe', 0.5)
                
                weighted_sharpe += weight * expected_sharpe
                total_weight += weight
            
            return weighted_sharpe / max(total_weight, 0.01)
            
        except Exception:
            return 0.4  # Default Sharpe estimate
    
    def _estimate_max_drawdown(self, positions: Dict) -> float:
        """Estimate maximum portfolio drawdown"""
        try:
            total_exposure = sum(pos.get('size', 0) for pos in positions.values())
            avg_volatility = np.mean([pos.get('volatility', 0.25) for pos in positions.values()])
            
            # Rough estimate: max drawdown ≈ 3 * portfolio volatility
            estimated_drawdown = total_exposure * avg_volatility * 3
            return min(estimated_drawdown, 0.5)  # Cap at 50%
            
        except Exception:
            return 0.15  # 15% default estimate
    
    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """Calculate portfolio correlation risk"""
        try:
            if len(positions) < 2:
                return 0.0
            
            symbols = list(positions.keys())
            total_correlation_risk = 0
            pair_count = 0
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    corr = abs(self._get_correlation(symbols[i], symbols[j]) or 0.3)
                    size_i = positions[symbols[i]].get('size', 0)
                    size_j = positions[symbols[j]].get('size', 0)
                    
                    # Correlation risk = correlation * product of position sizes
                    pair_risk = corr * size_i * size_j
                    total_correlation_risk += pair_risk
                    pair_count += 1
            
            return total_correlation_risk / max(pair_count, 1)
            
        except Exception:
            return 0.1  # Default moderate correlation risk
    
    def _calculate_concentration_risk(self, positions: Dict) -> float:
        """Calculate portfolio concentration risk"""
        try:
            if not positions:
                return 0.0
            
            position_sizes = [pos.get('size', 0) for pos in positions.values()]
            total_size = sum(position_sizes)
            
            if total_size == 0:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index for concentration
            hhi = sum((size / total_size) ** 2 for size in position_sizes)
            
            # Normalize HHI to risk score (0 = perfectly diversified, 1 = fully concentrated)
            concentration_risk = (hhi - 1/len(positions)) / (1 - 1/len(positions)) if len(positions) > 1 else 1.0
            
            return max(0, min(concentration_risk, 1.0))
            
        except Exception:
            return 0.2  # Default moderate concentration
    
    def _calculate_liquidity_risk(self, positions: Dict) -> float:
        """Calculate portfolio liquidity risk"""
        try:
            if not positions:
                return 0.0
            
            weighted_liquidity_risk = 0
            total_weight = 0
            
            for symbol, position in positions.items():
                weight = position.get('size', 0)
                liquidity_score = position.get('liquidity_score', 75)
                
                # Convert liquidity score to risk (higher score = lower risk)
                liquidity_risk = (100 - liquidity_score) / 100
                
                weighted_liquidity_risk += weight * liquidity_risk
                total_weight += weight
            
            return weighted_liquidity_risk / max(total_weight, 0.01)
            
        except Exception:
            return 0.15  # Default moderate liquidity risk
    
    def _calculate_tail_risk(self, positions: Dict) -> float:
        """Calculate portfolio tail risk"""
        try:
            if not positions:
                return 0.0
            
            # Simplified tail risk based on position concentration and volatility
            position_sizes = [pos.get('size', 0) for pos in positions.values()]
            position_vols = [pos.get('volatility', 0.25) for pos in positions.values()]
            
            max_position = max(position_sizes) if position_sizes else 0
            avg_volatility = np.mean(position_vols) if position_vols else 0.25
            
            # Tail risk increases with concentration and volatility
            tail_risk = max_position * avg_volatility * 2
            
            return min(tail_risk, 0.3)  # Cap at 30%
            
        except Exception:
            return 0.1  # Default tail risk
    
    def _calculate_diversification_ratio(self, positions: Dict) -> float:
        """Calculate portfolio diversification ratio"""
        try:
            if len(positions) <= 1:
                return 1.0
            
            # Simplified diversification ratio
            # Perfect diversification = 1/sqrt(n), no diversification = 1
            n_positions = len(positions)
            theoretical_max = 1.0 / math.sqrt(n_positions)
            
            # Adjust based on correlation
            avg_correlation = 0.3  # Simplified average correlation
            actual_diversification = theoretical_max + (1 - theoretical_max) * avg_correlation
            
            return 1.0 / actual_diversification
            
        except Exception:
            return 1.0
    
    def _default_position_calculation(self, symbol: str) -> PositionSizeCalculation:
        """Return default position calculation for error cases"""
        return PositionSizeCalculation(
            symbol=symbol,
            base_size=0.03,
            adjusted_size=0.03,
            sizing_method=SizingMethod.FIXED_PERCENT,
            risk_level=RiskLevel.MODERATE,
            volatility_adjustment=1.0,
            correlation_adjustment=1.0,
            liquidity_adjustment=1.0,
            concentration_adjustment=1.0,
            heat_adjustment=1.0,
            expected_var=0.006,
            expected_max_loss=0.015,
            portfolio_contribution=0.1,
            marginal_var=0.007,
            min_size_constraint=False,
            max_size_constraint=False,
            portfolio_limit_constraint=False,
            correlation_limit_constraint=False,
            confidence=60,
            reasoning="Default sizing applied due to calculation error",
            timestamp=datetime.now()
        )
    
    def _default_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Return default portfolio metrics for error cases"""
        return PortfolioRiskMetrics(
            total_exposure=0.0,
            portfolio_heat=PortfolioHeat.COLD,
            portfolio_var=0.02,
            portfolio_sharpe=0.4,
            max_drawdown_estimate=0.15,
            correlation_risk=0.1,
            concentration_risk=0.2,
            liquidity_risk=0.15,
            tail_risk=0.1,
            risk_budget_used=0.5,
            diversification_ratio=1.0,
            active_positions=0,
            timestamp=datetime.now()
        )
    
    def get_agent_status(self) -> Dict:
        """Get agent status and performance"""
        try:
            total_calculations = len(self.sizing_accuracy)
            avg_accuracy = np.mean(self.sizing_accuracy) if self.sizing_accuracy else 0
            
            risk_adjusted_performance = np.mean(self.risk_adjusted_returns) if self.risk_adjusted_returns else 0
            
            return {
                'agent_name': 'Position Sizing Agent',
                'status': 'active',
                'total_calculations': total_calculations,
                'sizing_accuracy': f"{avg_accuracy:.1%}",
                'risk_adjusted_performance': f"{risk_adjusted_performance:.1%}",
                'default_risk_per_trade': f"{self.config['default_risk_per_trade']:.1%}",
                'max_position_size': f"{self.config['max_position_size']:.1%}",
                'target_volatility': f"{self.config['target_volatility']:.1%}",
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {'agent_name': 'Position Sizing Agent', 'status': 'error'}
    
    def update_sizing_performance(self, symbol: str, predicted_risk: float, actual_risk: float):
        """Update sizing performance tracking"""
        try:
            if predicted_risk > 0 and actual_risk >= 0:
                accuracy = 1 - abs(predicted_risk - actual_risk) / predicted_risk
                self.sizing_accuracy.append(max(0, min(accuracy, 1)))
                
        except Exception as e:
            self.logger.error(f"Error updating sizing performance: {e}")
    
    def rebalance_portfolio(self, current_positions: Dict, target_weights: Dict) -> Dict[str, float]:
        """Calculate rebalancing trades needed"""
        try:
            rebalance_trades = {}
            total_current = sum(pos.get('size', 0) for pos in current_positions.values())
            
            for symbol, target_weight in target_weights.items():
                current_weight = current_positions.get(symbol, {}).get('size', 0)
                weight_diff = target_weight - current_weight
                
                # Only rebalance if change exceeds threshold
                if abs(weight_diff) > self.config['rebalance_threshold'] * target_weight:
                    rebalance_trades[symbol] = weight_diff
            
            return rebalance_trades
            
        except Exception as e:
            self.logger.error(f"Error calculating rebalancing: {e}")
            return {}