"""
Liquidity & Microstructure Agent - Analyzes market liquidity and microstructure
Monitors: spread, queue depth, halt status, auction periods, options OI/IV
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, time
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp

class LiquidityState(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    ILLIQUID = "illiquid"
    HALTED = "halted"
    AUCTION = "auction"
    CLOSED = "closed"

class MarketPhase(Enum):
    PRE_MARKET = "pre_market"
    OPENING_AUCTION = "opening_auction"
    CONTINUOUS_TRADING = "continuous_trading"
    CLOSING_AUCTION = "closing_auction"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

@dataclass
class MicrostructureReport:
    timestamp: datetime
    symbol: str
    liquidity_state: LiquidityState
    market_phase: MarketPhase
    bid_ask_spread: float
    spread_percentage: float
    bid_size: float
    ask_size: float
    order_book_imbalance: float
    average_spread_10min: float
    depth_at_levels: Dict[int, Dict[str, float]]  # Level -> {bid_size, ask_size}
    trade_frequency: float  # Trades per minute
    average_trade_size: float
    large_trade_ratio: float  # Ratio of large trades
    halt_status: bool
    halt_reason: Optional[str]
    options_data: Optional[Dict]
    execution_cost_estimate: float
    recommended_order_type: str
    max_safe_size: float
    liquidity_score: float  # 0-100
    warnings: List[str]

class LiquidityMicrostructureAgent:
    """
    Analyzes market liquidity conditions and microstructure for optimal execution
    """
    
    def __init__(self, exchange_client=None, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.exchange_client = exchange_client
        self.liquidity_cache = {}
        self.market_hours = self._initialize_market_hours()
        self.halt_cache = {}
        self.options_cache = {}
        
    def _default_config(self) -> Dict:
        return {
            'max_spread_bps': 10,  # Max 10 basis points
            'min_depth_usd': 10000,  # Minimum $10k depth
            'large_trade_threshold': 10000,  # USD value
            'depth_levels': 5,  # Order book levels to analyze
            'liquidity_window': 600,  # 10 minutes in seconds
            'stale_quote_seconds': 5,
            'imbalance_threshold': 0.3,  # 30% imbalance warning
            'min_trades_per_minute': 1,
            'options_iv_percentile': 80,  # High IV warning
            'max_execution_cost_bps': 5,  # Max 5 bps execution cost
            'halt_check_interval': 30,  # seconds
            'market_hours': {
                'pre_market': (time(4, 0), time(9, 30)),
                'regular': (time(9, 30), time(16, 0)),
                'after_hours': (time(16, 0), time(20, 0))
            }
        }
    
    def _initialize_market_hours(self) -> Dict:
        """Initialize market hours for different exchanges"""
        return {
            'NYSE': {
                'timezone': 'America/New_York',
                'pre_market': (time(4, 0), time(9, 30)),
                'regular': (time(9, 30), time(16, 0)),
                'after_hours': (time(16, 0), time(20, 0))
            },
            'NASDAQ': {
                'timezone': 'America/New_York',
                'pre_market': (time(4, 0), time(9, 30)),
                'regular': (time(9, 30), time(16, 0)),
                'after_hours': (time(16, 0), time(20, 0))
            },
            'CRYPTO': {
                'timezone': 'UTC',
                'regular': (time(0, 0), time(23, 59))  # 24/7
            }
        }
    
    async def analyze_liquidity(self, symbol: str, exchange: str = 'NYSE') -> MicrostructureReport:
        """
        Comprehensive liquidity and microstructure analysis
        """
        warnings = []
        
        # Get market phase
        market_phase = self._get_market_phase(exchange)
        
        # Check for halts
        halt_status, halt_reason = await self._check_halt_status(symbol)
        if halt_status:
            return self._create_halt_report(symbol, halt_reason)
        
        # Get order book data
        order_book = await self._get_order_book(symbol)
        if not order_book:
            warnings.append("Unable to fetch order book data")
            return self._create_illiquid_report(symbol, warnings)
        
        # Calculate spread metrics
        spread_metrics = self._calculate_spread_metrics(order_book)
        
        # Analyze order book depth
        depth_analysis = self._analyze_depth(order_book)
        
        # Get recent trades
        recent_trades = await self._get_recent_trades(symbol)
        trade_metrics = self._analyze_trades(recent_trades) if recent_trades else {}
        
        # Get options data if available
        options_data = await self._get_options_data(symbol) if self._is_optionable(symbol) else None
        
        # Calculate liquidity state
        liquidity_state = self._determine_liquidity_state(
            spread_metrics, depth_analysis, trade_metrics, market_phase
        )
        
        # Calculate execution costs
        execution_cost = self._estimate_execution_cost(
            spread_metrics, depth_analysis, trade_metrics
        )
        
        # Determine recommended order type
        recommended_order = self._recommend_order_type(
            liquidity_state, spread_metrics, market_phase
        )
        
        # Calculate maximum safe order size
        max_safe_size = self._calculate_max_safe_size(
            depth_analysis, trade_metrics, liquidity_state
        )
        
        # Calculate overall liquidity score
        liquidity_score = self._calculate_liquidity_score(
            spread_metrics, depth_analysis, trade_metrics, market_phase
        )
        
        # Generate warnings
        warnings.extend(self._generate_warnings(
            spread_metrics, depth_analysis, trade_metrics, options_data
        ))
        
        report = MicrostructureReport(
            timestamp=datetime.now(),
            symbol=symbol,
            liquidity_state=liquidity_state,
            market_phase=market_phase,
            bid_ask_spread=spread_metrics.get('spread', 0),
            spread_percentage=spread_metrics.get('spread_pct', 0),
            bid_size=spread_metrics.get('bid_size', 0),
            ask_size=spread_metrics.get('ask_size', 0),
            order_book_imbalance=depth_analysis.get('imbalance', 0),
            average_spread_10min=spread_metrics.get('avg_spread', 0),
            depth_at_levels=depth_analysis.get('levels', {}),
            trade_frequency=trade_metrics.get('frequency', 0),
            average_trade_size=trade_metrics.get('avg_size', 0),
            large_trade_ratio=trade_metrics.get('large_ratio', 0),
            halt_status=halt_status,
            halt_reason=halt_reason,
            options_data=options_data,
            execution_cost_estimate=execution_cost,
            recommended_order_type=recommended_order,
            max_safe_size=max_safe_size,
            liquidity_score=liquidity_score,
            warnings=warnings
        )
        
        # Cache the report
        self.liquidity_cache[symbol] = report
        
        return report
    
    def _get_market_phase(self, exchange: str) -> MarketPhase:
        """
        Determine current market phase
        """
        if exchange == 'CRYPTO':
            return MarketPhase.CONTINUOUS_TRADING
        
        from datetime import datetime
        import pytz
        
        market_info = self.market_hours.get(exchange, self.market_hours['NYSE'])
        tz = pytz.timezone(market_info['timezone'])
        now = datetime.now(tz).time()
        
        # Check if weekend
        if datetime.now(tz).weekday() >= 5:
            return MarketPhase.CLOSED
        
        # Check market phases
        if market_info['pre_market'][0] <= now < market_info['pre_market'][1]:
            return MarketPhase.PRE_MARKET
        elif now < time(9, 35) and now >= market_info['regular'][0]:
            return MarketPhase.OPENING_AUCTION
        elif market_info['regular'][0] <= now < market_info['regular'][1]:
            return MarketPhase.CONTINUOUS_TRADING
        elif time(15, 55) <= now < market_info['regular'][1]:
            return MarketPhase.CLOSING_AUCTION
        elif market_info['after_hours'][0] <= now < market_info['after_hours'][1]:
            return MarketPhase.AFTER_HOURS
        else:
            return MarketPhase.CLOSED
    
    async def _check_halt_status(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol is halted
        """
        # Check cache first
        if symbol in self.halt_cache:
            cached_time, status, reason = self.halt_cache[symbol]
            if (datetime.now() - cached_time).seconds < self.config['halt_check_interval']:
                return status, reason
        
        # Mock implementation - replace with actual exchange API
        try:
            if self.exchange_client:
                # Implement actual halt check via exchange API
                pass
        except Exception as e:
            self.logger.error(f"Error checking halt status: {e}")
        
        # Update cache
        self.halt_cache[symbol] = (datetime.now(), False, None)
        return False, None
    
    async def _get_order_book(self, symbol: str, levels: int = 5) -> Optional[Dict]:
        """
        Get order book data
        """
        try:
            if self.exchange_client:
                # Implement actual order book fetch
                pass
            
            # Mock data for demonstration
            return {
                'bids': [
                    {'price': 100.00, 'size': 1000},
                    {'price': 99.99, 'size': 1500},
                    {'price': 99.98, 'size': 2000},
                    {'price': 99.97, 'size': 2500},
                    {'price': 99.96, 'size': 3000}
                ],
                'asks': [
                    {'price': 100.01, 'size': 1000},
                    {'price': 100.02, 'size': 1500},
                    {'price': 100.03, 'size': 2000},
                    {'price': 100.04, 'size': 2500},
                    {'price': 100.05, 'size': 3000}
                ],
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return None
    
    async def _get_recent_trades(self, symbol: str, minutes: int = 10) -> Optional[List[Dict]]:
        """
        Get recent trades data
        """
        try:
            if self.exchange_client:
                # Implement actual trades fetch
                pass
            
            # Mock data
            return [
                {'price': 100.01, 'size': 100, 'time': datetime.now() - timedelta(seconds=i)}
                for i in range(0, minutes * 60, 10)
            ]
        except Exception as e:
            self.logger.error(f"Error fetching trades: {e}")
            return None
    
    async def _get_options_data(self, symbol: str) -> Optional[Dict]:
        """
        Get options data (OI, IV, etc.)
        """
        try:
            # Check cache
            if symbol in self.options_cache:
                cached_time, data = self.options_cache[symbol]
                if (datetime.now() - cached_time).seconds < 300:  # 5 min cache
                    return data
            
            # Mock options data
            options_data = {
                'iv_rank': 65,  # Implied volatility rank
                'iv_percentile': 75,
                'put_call_ratio': 0.8,
                'total_oi': 50000,
                'atm_iv': 0.25,
                'skew': -0.05,  # Negative = put skew
                'term_structure': 'normal'  # normal, inverted, flat
            }
            
            # Update cache
            self.options_cache[symbol] = (datetime.now(), options_data)
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching options data: {e}")
            return None
    
    def _calculate_spread_metrics(self, order_book: Dict) -> Dict:
        """
        Calculate spread-related metrics
        """
        if not order_book or not order_book.get('bids') or not order_book.get('asks'):
            return {}
        
        best_bid = order_book['bids'][0]
        best_ask = order_book['asks'][0]
        
        spread = best_ask['price'] - best_bid['price']
        mid_price = (best_ask['price'] + best_bid['price']) / 2
        spread_pct = (spread / mid_price) * 100
        spread_bps = spread_pct * 100  # Basis points
        
        return {
            'spread': spread,
            'spread_pct': spread_pct,
            'spread_bps': spread_bps,
            'mid_price': mid_price,
            'bid_price': best_bid['price'],
            'ask_price': best_ask['price'],
            'bid_size': best_bid['size'],
            'ask_size': best_ask['size'],
            'avg_spread': spread  # Would calculate from historical data
        }
    
    def _analyze_depth(self, order_book: Dict) -> Dict:
        """
        Analyze order book depth
        """
        if not order_book:
            return {}
        
        levels = {}
        total_bid_size = 0
        total_ask_size = 0
        total_bid_value = 0
        total_ask_value = 0
        
        for i, (bid, ask) in enumerate(zip(order_book.get('bids', []), order_book.get('asks', []))):
            levels[i + 1] = {
                'bid_size': bid['size'],
                'bid_price': bid['price'],
                'ask_size': ask['size'],
                'ask_price': ask['price']
            }
            total_bid_size += bid['size']
            total_ask_size += ask['size']
            total_bid_value += bid['size'] * bid['price']
            total_ask_value += ask['size'] * ask['price']
        
        # Calculate imbalance
        total_size = total_bid_size + total_ask_size
        imbalance = (total_bid_size - total_ask_size) / total_size if total_size > 0 else 0
        
        return {
            'levels': levels,
            'total_bid_size': total_bid_size,
            'total_ask_size': total_ask_size,
            'total_bid_value': total_bid_value,
            'total_ask_value': total_ask_value,
            'imbalance': imbalance,
            'depth_quality': self._assess_depth_quality(levels)
        }
    
    def _assess_depth_quality(self, levels: Dict) -> str:
        """
        Assess quality of order book depth
        """
        if not levels:
            return 'poor'
        
        # Check for consistent depth across levels
        sizes = []
        for level in levels.values():
            sizes.append(level['bid_size'] + level['ask_size'])
        
        if not sizes:
            return 'poor'
        
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        cv = std_size / avg_size if avg_size > 0 else 1
        
        if cv < 0.3 and avg_size > 1000:
            return 'excellent'
        elif cv < 0.5 and avg_size > 500:
            return 'good'
        elif cv < 0.7 and avg_size > 100:
            return 'fair'
        else:
            return 'poor'
    
    def _analyze_trades(self, trades: List[Dict]) -> Dict:
        """
        Analyze recent trade patterns
        """
        if not trades:
            return {}
        
        # Calculate metrics
        trade_sizes = [t['size'] for t in trades]
        trade_values = [t['size'] * t['price'] for t in trades]
        
        # Time analysis
        time_diffs = []
        for i in range(1, len(trades)):
            diff = (trades[i-1]['time'] - trades[i]['time']).total_seconds()
            time_diffs.append(diff)
        
        avg_time_between = np.mean(time_diffs) if time_diffs else 0
        trades_per_minute = 60 / avg_time_between if avg_time_between > 0 else 0
        
        # Size analysis
        avg_size = np.mean(trade_sizes)
        large_trades = [v for v in trade_values if v > self.config['large_trade_threshold']]
        large_ratio = len(large_trades) / len(trades) if trades else 0
        
        return {
            'frequency': trades_per_minute,
            'avg_size': avg_size,
            'total_volume': sum(trade_sizes),
            'large_ratio': large_ratio,
            'trade_count': len(trades)
        }
    
    def _determine_liquidity_state(self, spread: Dict, depth: Dict, 
                                  trades: Dict, phase: MarketPhase) -> LiquidityState:
        """
        Determine overall liquidity state
        """
        if phase == MarketPhase.CLOSED:
            return LiquidityState.CLOSED
        
        if phase in [MarketPhase.OPENING_AUCTION, MarketPhase.CLOSING_AUCTION]:
            return LiquidityState.AUCTION
        
        # Score components
        score = 0
        
        # Spread scoring
        if spread.get('spread_bps', float('inf')) < 5:
            score += 30
        elif spread.get('spread_bps', float('inf')) < 10:
            score += 20
        elif spread.get('spread_bps', float('inf')) < 20:
            score += 10
        
        # Depth scoring
        depth_quality = depth.get('depth_quality', 'poor')
        if depth_quality == 'excellent':
            score += 30
        elif depth_quality == 'good':
            score += 20
        elif depth_quality == 'fair':
            score += 10
        
        # Trade frequency scoring
        if trades.get('frequency', 0) > 10:
            score += 30
        elif trades.get('frequency', 0) > 5:
            score += 20
        elif trades.get('frequency', 0) > 1:
            score += 10
        
        # Phase adjustment
        if phase in [MarketPhase.PRE_MARKET, MarketPhase.AFTER_HOURS]:
            score *= 0.7
        
        # Determine state
        if score >= 80:
            return LiquidityState.EXCELLENT
        elif score >= 60:
            return LiquidityState.GOOD
        elif score >= 40:
            return LiquidityState.FAIR
        elif score >= 20:
            return LiquidityState.POOR
        else:
            return LiquidityState.ILLIQUID
    
    def _estimate_execution_cost(self, spread: Dict, depth: Dict, trades: Dict) -> float:
        """
        Estimate execution cost in basis points
        """
        base_cost = spread.get('spread_bps', 0) / 2  # Half spread
        
        # Impact cost based on depth
        if depth.get('depth_quality') == 'poor':
            base_cost *= 1.5
        elif depth.get('depth_quality') == 'fair':
            base_cost *= 1.2
        
        # Adjust for trade frequency
        if trades.get('frequency', 0) < 1:
            base_cost *= 1.3
        
        return base_cost
    
    def _recommend_order_type(self, state: LiquidityState, spread: Dict, phase: MarketPhase) -> str:
        """
        Recommend optimal order type
        """
        if state == LiquidityState.EXCELLENT:
            if spread.get('spread_bps', 0) < 3:
                return "MARKET - Excellent liquidity"
            else:
                return "LIMIT - Capture spread"
        
        elif state == LiquidityState.GOOD:
            return "LIMIT - Good liquidity, optimize entry"
        
        elif state in [LiquidityState.FAIR, LiquidityState.POOR]:
            return "LIMIT_PATIENT - Wait for liquidity"
        
        elif state == LiquidityState.ILLIQUID:
            return "AVOID - Insufficient liquidity"
        
        elif state == LiquidityState.AUCTION:
            return "AUCTION_LIMIT - Participate in auction"
        
        else:
            return "NONE - Market closed"
    
    def _calculate_max_safe_size(self, depth: Dict, trades: Dict, state: LiquidityState) -> float:
        """
        Calculate maximum safe order size
        """
        if state in [LiquidityState.ILLIQUID, LiquidityState.CLOSED, LiquidityState.HALTED]:
            return 0
        
        # Base on average trade size and depth
        avg_trade = trades.get('avg_size', 100)
        total_depth = depth.get('total_bid_size', 0) + depth.get('total_ask_size', 0)
        
        # Conservative approach: 10% of visible liquidity or 5x average trade
        max_size = min(total_depth * 0.1, avg_trade * 5)
        
        # Adjust for liquidity state
        state_multipliers = {
            LiquidityState.EXCELLENT: 1.0,
            LiquidityState.GOOD: 0.8,
            LiquidityState.FAIR: 0.5,
            LiquidityState.POOR: 0.2,
            LiquidityState.AUCTION: 0.3
        }
        
        multiplier = state_multipliers.get(state, 0.1)
        return max_size * multiplier
    
    def _calculate_liquidity_score(self, spread: Dict, depth: Dict, 
                                  trades: Dict, phase: MarketPhase) -> float:
        """
        Calculate overall liquidity score (0-100)
        """
        score = 0
        
        # Spread component (30 points)
        if spread.get('spread_bps', float('inf')) < 5:
            score += 30
        elif spread.get('spread_bps', float('inf')) < 10:
            score += 20
        elif spread.get('spread_bps', float('inf')) < 20:
            score += 10
        
        # Depth component (30 points)
        depth_quality = depth.get('depth_quality', 'poor')
        quality_scores = {'excellent': 30, 'good': 20, 'fair': 10, 'poor': 0}
        score += quality_scores.get(depth_quality, 0)
        
        # Trade activity (20 points)
        freq = trades.get('frequency', 0)
        if freq > 10:
            score += 20
        elif freq > 5:
            score += 15
        elif freq > 1:
            score += 10
        elif freq > 0.5:
            score += 5
        
        # Imbalance (10 points)
        imbalance = abs(depth.get('imbalance', 0))
        if imbalance < 0.1:
            score += 10
        elif imbalance < 0.2:
            score += 5
        
        # Market phase (10 points)
        phase_scores = {
            MarketPhase.CONTINUOUS_TRADING: 10,
            MarketPhase.PRE_MARKET: 5,
            MarketPhase.AFTER_HOURS: 5,
            MarketPhase.OPENING_AUCTION: 3,
            MarketPhase.CLOSING_AUCTION: 3,
            MarketPhase.CLOSED: 0
        }
        score += phase_scores.get(phase, 0)
        
        return min(100, max(0, score))
    
    def _generate_warnings(self, spread: Dict, depth: Dict, 
                          trades: Dict, options: Optional[Dict]) -> List[str]:
        """
        Generate liquidity warnings
        """
        warnings = []
        
        # Spread warnings
        if spread.get('spread_bps', 0) > self.config['max_spread_bps']:
            warnings.append(f"Wide spread: {spread.get('spread_bps', 0):.1f} bps")
        
        # Depth warnings
        if depth.get('total_bid_value', 0) < self.config['min_depth_usd']:
            warnings.append("Insufficient bid depth")
        if depth.get('total_ask_value', 0) < self.config['min_depth_usd']:
            warnings.append("Insufficient ask depth")
        
        # Imbalance warning
        if abs(depth.get('imbalance', 0)) > self.config['imbalance_threshold']:
            direction = "bid" if depth.get('imbalance', 0) > 0 else "ask"
            warnings.append(f"Order book imbalanced to {direction} side")
        
        # Trade frequency warning
        if trades.get('frequency', 0) < self.config['min_trades_per_minute']:
            warnings.append("Low trading activity")
        
        # Options warnings
        if options:
            if options.get('iv_percentile', 0) > self.config['options_iv_percentile']:
                warnings.append(f"High IV percentile: {options['iv_percentile']}%")
            if abs(options.get('skew', 0)) > 0.1:
                warnings.append(f"Options skew detected: {options['skew']:.2f}")
        
        return warnings
    
    def _is_optionable(self, symbol: str) -> bool:
        """
        Check if symbol has options
        """
        # Simple check - could be enhanced with actual data
        return not symbol.endswith('USDT') and not symbol.endswith('USD')
    
    def _create_halt_report(self, symbol: str, reason: str) -> MicrostructureReport:
        """
        Create report for halted symbol
        """
        return MicrostructureReport(
            timestamp=datetime.now(),
            symbol=symbol,
            liquidity_state=LiquidityState.HALTED,
            market_phase=MarketPhase.CONTINUOUS_TRADING,
            bid_ask_spread=0,
            spread_percentage=0,
            bid_size=0,
            ask_size=0,
            order_book_imbalance=0,
            average_spread_10min=0,
            depth_at_levels={},
            trade_frequency=0,
            average_trade_size=0,
            large_trade_ratio=0,
            halt_status=True,
            halt_reason=reason,
            options_data=None,
            execution_cost_estimate=0,
            recommended_order_type="NONE - Symbol halted",
            max_safe_size=0,
            liquidity_score=0,
            warnings=[f"Symbol halted: {reason}"]
        )
    
    def _create_illiquid_report(self, symbol: str, warnings: List[str]) -> MicrostructureReport:
        """
        Create report for illiquid symbol
        """
        return MicrostructureReport(
            timestamp=datetime.now(),
            symbol=symbol,
            liquidity_state=LiquidityState.ILLIQUID,
            market_phase=MarketPhase.CONTINUOUS_TRADING,
            bid_ask_spread=0,
            spread_percentage=0,
            bid_size=0,
            ask_size=0,
            order_book_imbalance=0,
            average_spread_10min=0,
            depth_at_levels={},
            trade_frequency=0,
            average_trade_size=0,
            large_trade_ratio=0,
            halt_status=False,
            halt_reason=None,
            options_data=None,
            execution_cost_estimate=0,
            recommended_order_type="AVOID - Insufficient liquidity",
            max_safe_size=0,
            liquidity_score=0,
            warnings=warnings + ["Symbol appears illiquid"]
        )
    
    def get_execution_params(self, symbol: str) -> Dict:
        """
        Get execution parameters based on current liquidity
        """
        if symbol not in self.liquidity_cache:
            return {
                'order_type': 'LIMIT',
                'time_in_force': 'GTC',
                'price_buffer': 0.001,
                'max_size': 100,
                'split_orders': False
            }
        
        report = self.liquidity_cache[symbol]
        
        params = {
            'order_type': 'LIMIT' if 'LIMIT' in report.recommended_order_type else 'MARKET',
            'time_in_force': 'IOC' if report.liquidity_state == LiquidityState.POOR else 'GTC',
            'price_buffer': report.spread_percentage / 200,  # Half spread
            'max_size': report.max_safe_size,
            'split_orders': report.liquidity_state in [LiquidityState.POOR, LiquidityState.FAIR],
            'execution_algo': self._select_execution_algo(report)
        }
        
        return params
    
    def _select_execution_algo(self, report: MicrostructureReport) -> str:
        """
        Select appropriate execution algorithm
        """
        if report.liquidity_state == LiquidityState.EXCELLENT:
            return "AGGRESSIVE"
        elif report.liquidity_state == LiquidityState.GOOD:
            return "STANDARD"
        elif report.liquidity_state in [LiquidityState.FAIR, LiquidityState.POOR]:
            return "PATIENT"
        else:
            return "NONE"