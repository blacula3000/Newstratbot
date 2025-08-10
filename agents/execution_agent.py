"""
Execution Agent - Smart order routing and execution management
Handles: order types, slippage control, partial fills, cancel/replace logic
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import deque

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_IF_TOUCHED = "limit_if_touched"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"

class ExecutionAlgo(Enum):
    AGGRESSIVE = "aggressive"
    STANDARD = "standard"
    PATIENT = "patient"
    PASSIVE = "passive"
    STEALTH = "stealth"
    SWEEP = "sweep"

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK, DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0
    avg_fill_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    order_id: str
    symbol: str
    total_size: float
    execution_algo: ExecutionAlgo
    order_type: OrderType
    slices: List[Dict]  # List of order slices
    max_slippage_bps: float
    urgency: str  # 'high', 'medium', 'low'
    constraints: Dict
    estimated_cost_bps: float
    estimated_time_seconds: float

@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    requested_size: float
    filled_size: float
    avg_price: float
    slippage_bps: float
    execution_time_ms: float
    fills: List[Dict]
    rejections: List[Dict]
    cost_bps: float
    success: bool
    notes: List[str]

class ExecutionAgent:
    """
    Manages intelligent order execution with slippage control and smart routing
    """
    
    def __init__(self, exchange_client=None, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.exchange_client = exchange_client
        
        # Order management
        self.active_orders = {}
        self.order_history = deque(maxlen=1000)
        self.execution_plans = {}
        
        # Performance tracking
        self.slippage_history = deque(maxlen=100)
        self.fill_rate_history = deque(maxlen=100)
        
        # Market data cache
        self.market_data = {}
        self.order_book_cache = {}
        
    def _default_config(self) -> Dict:
        return {
            'max_slippage_bps': 10,  # 10 basis points max slippage
            'default_urgency': 'medium',
            'max_order_age_seconds': 300,  # 5 minutes
            'partial_fill_threshold': 0.5,  # Cancel if less than 50% filled
            'retry_attempts': 3,
            'retry_delay_ms': 500,
            'iceberg_show_pct': 0.2,  # Show 20% of iceberg order
            'twap_intervals': 10,  # Split TWAP into 10 intervals
            'vwap_lookback_minutes': 30,
            'sweep_levels': 3,  # Sweep top 3 price levels
            'stealth_max_size_pct': 0.1,  # Max 10% of recent volume
            'price_improvement_bps': 1,  # Try to improve price by 1 bp
            'cancel_replace_threshold_bps': 5,  # Replace if price moves 5 bps
        }
    
    async def execute_order(self, order_request: Dict, 
                           execution_params: Optional[Dict] = None) -> ExecutionReport:
        """
        Execute an order with intelligent routing and slippage control
        """
        # Create order
        order = self._create_order(order_request)
        
        # Get execution parameters
        params = execution_params or self._get_default_execution_params(order_request)
        
        # Create execution plan
        plan = await self._create_execution_plan(order, params)
        self.execution_plans[order.id] = plan
        
        # Execute based on algorithm
        if plan.execution_algo == ExecutionAlgo.AGGRESSIVE:
            report = await self._execute_aggressive(order, plan)
        elif plan.execution_algo == ExecutionAlgo.PATIENT:
            report = await self._execute_patient(order, plan)
        elif plan.execution_algo == ExecutionAlgo.PASSIVE:
            report = await self._execute_passive(order, plan)
        elif plan.execution_algo == ExecutionAlgo.STEALTH:
            report = await self._execute_stealth(order, plan)
        elif plan.execution_algo == ExecutionAlgo.SWEEP:
            report = await self._execute_sweep(order, plan)
        else:
            report = await self._execute_standard(order, plan)
        
        # Update history
        self.order_history.append(order)
        self._update_performance_metrics(report)
        
        return report
    
    def _create_order(self, request: Dict) -> Order:
        """Create order object from request"""
        order_type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'bracket': OrderType.BRACKET
        }
        
        return Order(
            id=str(uuid.uuid4()),
            symbol=request['symbol'],
            side=request['side'],
            size=request['size'],
            order_type=order_type_map.get(request.get('order_type', 'limit'), OrderType.LIMIT),
            limit_price=request.get('limit_price'),
            stop_price=request.get('stop_price'),
            time_in_force=request.get('time_in_force', 'GTC'),
            metadata=request.get('metadata', {})
        )
    
    def _get_default_execution_params(self, request: Dict) -> Dict:
        """Get default execution parameters based on order characteristics"""
        size = request['size']
        symbol = request['symbol']
        
        # Get recent volume
        avg_volume = self._get_average_volume(symbol)
        size_pct = (size / avg_volume * 100) if avg_volume > 0 else 0
        
        # Determine urgency
        if request.get('urgent', False):
            urgency = 'high'
        elif size_pct > 5:
            urgency = 'low'  # Large order, be patient
        else:
            urgency = 'medium'
        
        # Select algorithm
        if urgency == 'high':
            algo = ExecutionAlgo.AGGRESSIVE
        elif size_pct > 10:
            algo = ExecutionAlgo.STEALTH
        elif size_pct > 5:
            algo = ExecutionAlgo.PATIENT
        else:
            algo = ExecutionAlgo.STANDARD
        
        return {
            'urgency': urgency,
            'execution_algo': algo,
            'max_slippage_bps': self.config['max_slippage_bps'],
            'allow_partial': True,
            'smart_routing': True
        }
    
    async def _create_execution_plan(self, order: Order, params: Dict) -> ExecutionPlan:
        """Create detailed execution plan"""
        # Get market data
        market_data = await self._get_market_data(order.symbol)
        
        # Calculate slices based on algorithm
        if params['execution_algo'] == ExecutionAlgo.STEALTH:
            slices = self._create_stealth_slices(order, market_data)
        elif params['execution_algo'] == ExecutionAlgo.PATIENT:
            slices = self._create_time_slices(order, 'twap')
        else:
            slices = [{'size': order.size, 'price': None, 'delay': 0}]
        
        # Estimate costs
        estimated_cost = self._estimate_execution_cost(order, params, market_data)
        estimated_time = self._estimate_execution_time(order, params)
        
        # Define constraints
        constraints = {
            'max_slippage_bps': params.get('max_slippage_bps', self.config['max_slippage_bps']),
            'max_time_seconds': params.get('max_time', 300),
            'min_fill_size': params.get('min_fill', order.size * 0.5),
            'allow_partial': params.get('allow_partial', True)
        }
        
        return ExecutionPlan(
            order_id=order.id,
            symbol=order.symbol,
            total_size=order.size,
            execution_algo=params['execution_algo'],
            order_type=order.order_type,
            slices=slices,
            max_slippage_bps=params['max_slippage_bps'],
            urgency=params['urgency'],
            constraints=constraints,
            estimated_cost_bps=estimated_cost,
            estimated_time_seconds=estimated_time
        )
    
    async def _execute_aggressive(self, order: Order, plan: ExecutionPlan) -> ExecutionReport:
        """
        Aggressive execution - prioritize speed over price
        """
        fills = []
        rejections = []
        start_time = datetime.now()
        
        try:
            # Use market order or aggressive limit
            if order.order_type == OrderType.LIMIT:
                # Cross the spread aggressively
                market_data = await self._get_market_data(order.symbol)
                if order.side == 'buy':
                    order.limit_price = market_data['ask'] * 1.001  # Pay 0.1% above ask
                else:
                    order.limit_price = market_data['bid'] * 0.999  # Sell 0.1% below bid
            
            # Submit order
            response = await self._submit_order(order)
            
            if response['status'] == 'filled':
                fills.append({
                    'size': response['filled_size'],
                    'price': response['fill_price'],
                    'time': datetime.now()
                })
            elif response['status'] == 'rejected':
                rejections.append(response)
            
            # If partial fill, sweep remaining liquidity
            if response.get('filled_size', 0) < order.size:
                remaining = order.size - response.get('filled_size', 0)
                sweep_fills = await self._sweep_liquidity(order.symbol, order.side, remaining)
                fills.extend(sweep_fills)
        
        except Exception as e:
            self.logger.error(f"Aggressive execution failed: {e}")
            rejections.append({'reason': str(e)})
        
        return self._create_execution_report(order, plan, fills, rejections, start_time)
    
    async def _execute_patient(self, order: Order, plan: ExecutionPlan) -> ExecutionReport:
        """
        Patient execution - prioritize price over speed
        """
        fills = []
        rejections = []
        start_time = datetime.now()
        
        try:
            # Split into time slices
            for slice_info in plan.slices:
                # Place limit order at mid or better
                market_data = await self._get_market_data(order.symbol)
                mid_price = (market_data['bid'] + market_data['ask']) / 2
                
                slice_order = Order(
                    id=f"{order.id}_slice_{len(fills)}",
                    symbol=order.symbol,
                    side=order.side,
                    size=slice_info['size'],
                    order_type=OrderType.LIMIT,
                    limit_price=mid_price if order.side == 'buy' else mid_price,
                    time_in_force='IOC'
                )
                
                response = await self._submit_order(slice_order)
                
                if response.get('filled_size', 0) > 0:
                    fills.append({
                        'size': response['filled_size'],
                        'price': response['fill_price'],
                        'time': datetime.now()
                    })
                
                # Wait before next slice
                if slice_info.get('delay', 0) > 0:
                    await asyncio.sleep(slice_info['delay'] / 1000)
                
                # Check if we've filled enough
                total_filled = sum(f['size'] for f in fills)
                if total_filled >= order.size * 0.95:
                    break
        
        except Exception as e:
            self.logger.error(f"Patient execution failed: {e}")
            rejections.append({'reason': str(e)})
        
        return self._create_execution_report(order, plan, fills, rejections, start_time)
    
    async def _execute_passive(self, order: Order, plan: ExecutionPlan) -> ExecutionReport:
        """
        Passive execution - provide liquidity, never take
        """
        fills = []
        rejections = []
        start_time = datetime.now()
        
        try:
            market_data = await self._get_market_data(order.symbol)
            
            # Place limit order behind the spread
            if order.side == 'buy':
                order.limit_price = market_data['bid'] - (market_data['spread'] * 0.1)
            else:
                order.limit_price = market_data['ask'] + (market_data['spread'] * 0.1)
            
            order.time_in_force = 'GTC'
            response = await self._submit_order(order)
            
            # Monitor and adjust
            max_wait = plan.constraints['max_time_seconds']
            check_interval = 5  # seconds
            elapsed = 0
            
            while elapsed < max_wait and response['status'] not in ['filled', 'cancelled']:
                await asyncio.sleep(check_interval)
                elapsed += check_interval
                
                # Check fill status
                status = await self._check_order_status(order.id)
                
                if status['filled_size'] > 0:
                    fills.append({
                        'size': status['filled_size'],
                        'price': status['avg_price'],
                        'time': datetime.now()
                    })
                
                # Adjust price if market moved
                if await self._should_reprice(order, market_data):
                    await self._cancel_replace_order(order)
        
        except Exception as e:
            self.logger.error(f"Passive execution failed: {e}")
            rejections.append({'reason': str(e)})
        
        return self._create_execution_report(order, plan, fills, rejections, start_time)
    
    async def _execute_stealth(self, order: Order, plan: ExecutionPlan) -> ExecutionReport:
        """
        Stealth execution - minimize market impact for large orders
        """
        fills = []
        rejections = []
        start_time = datetime.now()
        
        try:
            # Execute in small, randomized slices
            for slice_info in plan.slices:
                # Randomize timing
                delay = np.random.uniform(1, 5)  # 1-5 seconds
                await asyncio.sleep(delay)
                
                # Use iceberg order or small visible size
                slice_order = Order(
                    id=f"{order.id}_stealth_{len(fills)}",
                    symbol=order.symbol,
                    side=order.side,
                    size=slice_info['size'],
                    order_type=OrderType.LIMIT,
                    limit_price=slice_info.get('price'),
                    time_in_force='IOC',
                    metadata={'visible_size': slice_info['size'] * 0.2}  # Show only 20%
                )
                
                response = await self._submit_order(slice_order)
                
                if response.get('filled_size', 0) > 0:
                    fills.append({
                        'size': response['filled_size'],
                        'price': response['fill_price'],
                        'time': datetime.now()
                    })
                
                # Adapt based on fill rate
                if len(fills) > 2:
                    fill_rate = sum(f['size'] for f in fills) / (len(fills) * slice_info['size'])
                    if fill_rate < 0.5:
                        # Poor fill rate, be more aggressive
                        slice_info['price'] = await self._get_aggressive_price(order.symbol, order.side)
        
        except Exception as e:
            self.logger.error(f"Stealth execution failed: {e}")
            rejections.append({'reason': str(e)})
        
        return self._create_execution_report(order, plan, fills, rejections, start_time)
    
    async def _execute_sweep(self, order: Order, plan: ExecutionPlan) -> ExecutionReport:
        """
        Sweep execution - take multiple price levels immediately
        """
        fills = []
        rejections = []
        start_time = datetime.now()
        
        try:
            # Get order book
            order_book = await self._get_order_book(order.symbol)
            
            # Determine levels to sweep
            levels_to_sweep = self.config['sweep_levels']
            remaining_size = order.size
            
            book_side = 'asks' if order.side == 'buy' else 'bids'
            
            for i, level in enumerate(order_book[book_side][:levels_to_sweep]):
                if remaining_size <= 0:
                    break
                
                sweep_size = min(remaining_size, level['size'])
                
                sweep_order = Order(
                    id=f"{order.id}_sweep_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    size=sweep_size,
                    order_type=OrderType.LIMIT,
                    limit_price=level['price'] * (1.001 if order.side == 'buy' else 0.999),
                    time_in_force='IOC'
                )
                
                response = await self._submit_order(sweep_order)
                
                if response.get('filled_size', 0) > 0:
                    fills.append({
                        'size': response['filled_size'],
                        'price': response['fill_price'],
                        'time': datetime.now()
                    })
                    remaining_size -= response['filled_size']
        
        except Exception as e:
            self.logger.error(f"Sweep execution failed: {e}")
            rejections.append({'reason': str(e)})
        
        return self._create_execution_report(order, plan, fills, rejections, start_time)
    
    async def _execute_standard(self, order: Order, plan: ExecutionPlan) -> ExecutionReport:
        """
        Standard execution - balanced approach
        """
        fills = []
        rejections = []
        start_time = datetime.now()
        
        try:
            # Get current market
            market_data = await self._get_market_data(order.symbol)
            
            # Set competitive price
            if order.order_type == OrderType.LIMIT:
                if order.side == 'buy':
                    order.limit_price = market_data['bid'] + (market_data['spread'] * 0.25)
                else:
                    order.limit_price = market_data['ask'] - (market_data['spread'] * 0.25)
            
            # Submit order
            response = await self._submit_order(order)
            
            # Handle response
            if response['status'] == 'filled':
                fills.append({
                    'size': response['filled_size'],
                    'price': response['fill_price'],
                    'time': datetime.now()
                })
            elif response['status'] == 'partial':
                fills.append({
                    'size': response['filled_size'],
                    'price': response['avg_price'],
                    'time': datetime.now()
                })
                
                # Try to fill remainder
                if plan.constraints['allow_partial']:
                    remaining = order.size - response['filled_size']
                    remainder_order = Order(
                        id=f"{order.id}_remainder",
                        symbol=order.symbol,
                        side=order.side,
                        size=remaining,
                        order_type=OrderType.MARKET,
                        time_in_force='IOC'
                    )
                    
                    remainder_response = await self._submit_order(remainder_order)
                    if remainder_response.get('filled_size', 0) > 0:
                        fills.append({
                            'size': remainder_response['filled_size'],
                            'price': remainder_response['fill_price'],
                            'time': datetime.now()
                        })
            elif response['status'] == 'rejected':
                rejections.append(response)
        
        except Exception as e:
            self.logger.error(f"Standard execution failed: {e}")
            rejections.append({'reason': str(e)})
        
        return self._create_execution_report(order, plan, fills, rejections, start_time)
    
    def _create_stealth_slices(self, order: Order, market_data: Dict) -> List[Dict]:
        """Create stealth execution slices"""
        avg_volume = market_data.get('avg_volume_1min', 1000)
        max_slice = avg_volume * self.config['stealth_max_size_pct']
        
        slices = []
        remaining = order.size
        
        while remaining > 0:
            # Randomize slice size
            slice_size = min(remaining, np.random.uniform(max_slice * 0.5, max_slice))
            slices.append({
                'size': slice_size,
                'price': None,  # Will be determined at execution
                'delay': np.random.uniform(1000, 5000)  # 1-5 seconds
            })
            remaining -= slice_size
        
        return slices
    
    def _create_time_slices(self, order: Order, algo_type: str) -> List[Dict]:
        """Create time-based execution slices (TWAP/VWAP)"""
        if algo_type == 'twap':
            num_slices = self.config['twap_intervals']
            slice_size = order.size / num_slices
            interval = 60000 / num_slices  # Distribute over 1 minute
            
            return [
                {'size': slice_size, 'price': None, 'delay': i * interval}
                for i in range(num_slices)
            ]
        else:  # VWAP
            # Would need volume profile, simplified here
            return self._create_time_slices(order, 'twap')
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data"""
        # Cache check
        if symbol in self.market_data:
            cached_time, data = self.market_data[symbol]
            if (datetime.now() - cached_time).seconds < 1:
                return data
        
        # Mock implementation
        data = {
            'bid': 100.00,
            'ask': 100.02,
            'spread': 0.02,
            'last': 100.01,
            'volume': 100000,
            'avg_volume_1min': 1000
        }
        
        self.market_data[symbol] = (datetime.now(), data)
        return data
    
    async def _get_order_book(self, symbol: str) -> Dict:
        """Get order book data"""
        # Mock implementation
        return {
            'bids': [
                {'price': 99.99 - i * 0.01, 'size': 1000 * (i + 1)}
                for i in range(5)
            ],
            'asks': [
                {'price': 100.01 + i * 0.01, 'size': 1000 * (i + 1)}
                for i in range(5)
            ]
        }
    
    async def _submit_order(self, order: Order) -> Dict:
        """Submit order to exchange"""
        # Mock implementation
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate different outcomes
        import random
        outcome = random.choice(['filled', 'partial', 'rejected'])
        
        if outcome == 'filled':
            return {
                'status': 'filled',
                'filled_size': order.size,
                'fill_price': order.limit_price or 100.01,
                'order_id': order.id
            }
        elif outcome == 'partial':
            return {
                'status': 'partial',
                'filled_size': order.size * 0.6,
                'avg_price': order.limit_price or 100.01,
                'order_id': order.id
            }
        else:
            return {
                'status': 'rejected',
                'reason': 'Insufficient liquidity',
                'order_id': order.id
            }
    
    async def _check_order_status(self, order_id: str) -> Dict:
        """Check order status"""
        # Mock implementation
        return {
            'order_id': order_id,
            'status': 'partial',
            'filled_size': 50,
            'avg_price': 100.01
        }
    
    async def _should_reprice(self, order: Order, old_market: Dict) -> bool:
        """Determine if order should be repriced"""
        new_market = await self._get_market_data(order.symbol)
        
        if order.side == 'buy':
            price_change = (new_market['bid'] - old_market['bid']) / old_market['bid']
        else:
            price_change = (new_market['ask'] - old_market['ask']) / old_market['ask']
        
        return abs(price_change) * 10000 > self.config['cancel_replace_threshold_bps']
    
    async def _cancel_replace_order(self, order: Order) -> bool:
        """Cancel and replace order with new price"""
        try:
            # Cancel existing
            await self._cancel_order(order.id)
            
            # Get new price
            market_data = await self._get_market_data(order.symbol)
            if order.side == 'buy':
                order.limit_price = market_data['bid']
            else:
                order.limit_price = market_data['ask']
            
            # Submit new order
            order.id = f"{order.id}_replaced"
            await self._submit_order(order)
            
            return True
        except Exception as e:
            self.logger.error(f"Cancel/replace failed: {e}")
            return False
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        # Mock implementation
        return True
    
    async def _sweep_liquidity(self, symbol: str, side: str, size: float) -> List[Dict]:
        """Sweep available liquidity"""
        fills = []
        order_book = await self._get_order_book(symbol)
        
        book_side = 'asks' if side == 'buy' else 'bids'
        remaining = size
        
        for level in order_book[book_side]:
            if remaining <= 0:
                break
            
            fill_size = min(remaining, level['size'])
            fills.append({
                'size': fill_size,
                'price': level['price'],
                'time': datetime.now()
            })
            remaining -= fill_size
        
        return fills
    
    async def _get_aggressive_price(self, symbol: str, side: str) -> float:
        """Get aggressive price for immediate execution"""
        market_data = await self._get_market_data(symbol)
        
        if side == 'buy':
            return market_data['ask'] * 1.002  # 0.2% above ask
        else:
            return market_data['bid'] * 0.998  # 0.2% below bid
    
    def _get_average_volume(self, symbol: str) -> float:
        """Get average volume for symbol"""
        # Mock implementation
        return 10000
    
    def _estimate_execution_cost(self, order: Order, params: Dict, market: Dict) -> float:
        """Estimate execution cost in basis points"""
        base_cost = market['spread'] / market['last'] * 5000  # Half spread in bps
        
        # Adjust for algorithm
        algo_multipliers = {
            ExecutionAlgo.AGGRESSIVE: 1.5,
            ExecutionAlgo.STANDARD: 1.0,
            ExecutionAlgo.PATIENT: 0.7,
            ExecutionAlgo.PASSIVE: 0.3,
            ExecutionAlgo.STEALTH: 0.8,
            ExecutionAlgo.SWEEP: 2.0
        }
        
        multiplier = algo_multipliers.get(params['execution_algo'], 1.0)
        
        # Adjust for size
        avg_volume = self._get_average_volume(order.symbol)
        if avg_volume > 0:
            size_impact = (order.size / avg_volume) * 10  # 10 bps per 100% of volume
            base_cost += size_impact
        
        return base_cost * multiplier
    
    def _estimate_execution_time(self, order: Order, params: Dict) -> float:
        """Estimate execution time in seconds"""
        base_times = {
            ExecutionAlgo.AGGRESSIVE: 1,
            ExecutionAlgo.STANDARD: 5,
            ExecutionAlgo.PATIENT: 60,
            ExecutionAlgo.PASSIVE: 120,
            ExecutionAlgo.STEALTH: 180,
            ExecutionAlgo.SWEEP: 2
        }
        
        return base_times.get(params['execution_algo'], 10)
    
    def _create_execution_report(self, order: Order, plan: ExecutionPlan, 
                                fills: List[Dict], rejections: List[Dict],
                                start_time: datetime) -> ExecutionReport:
        """Create execution report"""
        total_filled = sum(f['size'] for f in fills)
        
        if total_filled > 0:
            weighted_sum = sum(f['size'] * f['price'] for f in fills)
            avg_price = weighted_sum / total_filled
        else:
            avg_price = 0
        
        # Calculate slippage
        if order.limit_price and avg_price > 0:
            if order.side == 'buy':
                slippage = (avg_price - order.limit_price) / order.limit_price
            else:
                slippage = (order.limit_price - avg_price) / order.limit_price
            slippage_bps = slippage * 10000
        else:
            slippage_bps = 0
        
        # Calculate actual cost
        market_mid = (plan.estimated_cost_bps / 10000) * avg_price if avg_price > 0 else 0
        actual_cost_bps = abs(slippage_bps)
        
        # Execution time
        exec_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Success determination
        success = total_filled >= plan.constraints.get('min_fill_size', order.size * 0.5)
        
        # Notes
        notes = []
        if total_filled < order.size:
            notes.append(f"Partial fill: {total_filled/order.size:.1%}")
        if slippage_bps > plan.max_slippage_bps:
            notes.append(f"Slippage exceeded: {slippage_bps:.1f} bps")
        if rejections:
            notes.append(f"Rejections: {len(rejections)}")
        
        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            requested_size=order.size,
            filled_size=total_filled,
            avg_price=avg_price,
            slippage_bps=slippage_bps,
            execution_time_ms=exec_time_ms,
            fills=fills,
            rejections=rejections,
            cost_bps=actual_cost_bps,
            success=success,
            notes=notes
        )
    
    def _update_performance_metrics(self, report: ExecutionReport):
        """Update performance tracking"""
        self.slippage_history.append(report.slippage_bps)
        fill_rate = report.filled_size / report.requested_size if report.requested_size > 0 else 0
        self.fill_rate_history.append(fill_rate)
    
    def get_execution_analytics(self) -> Dict:
        """Get execution performance analytics"""
        if not self.slippage_history:
            return {}
        
        return {
            'avg_slippage_bps': np.mean(self.slippage_history),
            'max_slippage_bps': max(self.slippage_history),
            'avg_fill_rate': np.mean(self.fill_rate_history),
            'total_orders': len(self.order_history),
            'success_rate': sum(1 for r in self.fill_rate_history if r > 0.9) / len(self.fill_rate_history)
        }