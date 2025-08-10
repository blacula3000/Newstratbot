"""
Enhanced Multi-Timeframe Data Ingestion Pipeline
Event-driven processing for tick/1m/5m/30m/1h/D data
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

class TimeFrame(Enum):
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"

class DataEvent(Enum):
    NEW_TICK = "new_tick"
    NEW_CANDLE = "new_candle"
    CANDLE_UPDATE = "candle_update"
    TIMEFRAME_COMPLETE = "timeframe_complete"

@dataclass
class MarketTick:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    metadata: Dict = None

@dataclass
class Candle:
    symbol: str
    timeframe: TimeFrame
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 0
    strat_type: Optional[str] = None  # '1', '2u', '2d', '3'
    is_complete: bool = False

@dataclass
class DataContext:
    symbol: str
    timeframes: Dict[TimeFrame, deque]  # Recent candles per timeframe
    current_candles: Dict[TimeFrame, Candle]  # Building candles
    last_tick: Optional[MarketTick]
    last_update: datetime

class EnhancedDataPipeline:
    """
    Event-driven multi-timeframe data pipeline for professional STRAT analysis
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.symbols = symbols
        self.data_contexts = {symbol: self._create_data_context(symbol) for symbol in symbols}
        
        # Event handlers
        self.event_handlers = defaultdict(list)
        self.processing_queue = asyncio.Queue(maxsize=10000)
        
        # Pipeline components
        self.running = False
        self.processor_tasks = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.tick_count = 0
        self.candle_count = defaultdict(int)
        self.processing_times = deque(maxlen=1000)
        
        self.logger.info(f"Enhanced data pipeline initialized for {len(symbols)} symbols")
    
    def _default_config(self) -> Dict:
        return {
            'max_candles_per_timeframe': 500,
            'tick_buffer_size': 1000,
            'processing_threads': 4,
            'candle_completion_delay_ms': 100,
            'data_quality_checks': True,
            'strat_classification': True,
            'event_batch_size': 10,
            'timeframe_alignment_tolerance_ms': 500
        }
    
    def _create_data_context(self, symbol: str) -> DataContext:
        """Create data context for a symbol"""
        timeframes = {}
        current_candles = {}
        
        for tf in TimeFrame:
            if tf != TimeFrame.TICK:
                timeframes[tf] = deque(maxlen=self.config['max_candles_per_timeframe'])
                current_candles[tf] = None
        
        return DataContext(
            symbol=symbol,
            timeframes=timeframes,
            current_candles=current_candles,
            last_tick=None,
            last_update=datetime.now()
        )
    
    def register_event_handler(self, event_type: DataEvent, handler: Callable):
        """Register event handler for specific data events"""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for {event_type.value}")
    
    async def start(self):
        """Start the data pipeline"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("🚀 Starting enhanced data pipeline...")
        
        # Start processing tasks
        for i in range(self.config['processing_threads']):
            task = asyncio.create_task(self._process_events())
            self.processor_tasks.append(task)
        
        # Start candle completion monitor
        completion_task = asyncio.create_task(self._monitor_candle_completion())
        self.processor_tasks.append(completion_task)
        
        self.logger.info("✅ Data pipeline started")
    
    async def stop(self):
        """Stop the data pipeline"""
        self.running = False
        
        # Cancel all tasks
        for task in self.processor_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processor_tasks, return_exceptions=True)
        
        self.thread_pool.shutdown(wait=True)
        self.logger.info("⏹️ Data pipeline stopped")
    
    async def ingest_tick(self, tick: MarketTick):
        """Ingest a new market tick"""
        if not self.running:
            return
        
        # Queue tick for processing
        try:
            await self.processing_queue.put(('tick', tick))
            self.tick_count += 1
        except asyncio.QueueFull:
            self.logger.warning(f"Processing queue full, dropping tick for {tick.symbol}")
    
    async def ingest_candle(self, candle: Candle):
        """Ingest a completed candle"""
        if not self.running:
            return
        
        try:
            await self.processing_queue.put(('candle', candle))
        except asyncio.QueueFull:
            self.logger.warning(f"Processing queue full, dropping candle for {candle.symbol}")
    
    async def _process_events(self):
        """Main event processing loop"""
        while self.running:
            try:
                # Get event from queue
                event_type, data = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )
                
                start_time = datetime.now()
                
                if event_type == 'tick':
                    await self._process_tick(data)
                elif event_type == 'candle':
                    await self._process_candle(data)
                
                # Track processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.processing_times.append(processing_time)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    async def _process_tick(self, tick: MarketTick):
        """Process a market tick"""
        context = self.data_contexts.get(tick.symbol)
        if not context:
            return
        
        # Update context
        context.last_tick = tick
        context.last_update = tick.timestamp
        
        # Update current candles with tick
        await self._update_candles_with_tick(context, tick)
        
        # Emit tick event
        await self._emit_event(DataEvent.NEW_TICK, {
            'tick': tick,
            'context': context
        })
    
    async def _process_candle(self, candle: Candle):
        """Process a completed candle"""
        context = self.data_contexts.get(candle.symbol)
        if not context:
            return
        
        # Add STRAT classification
        if self.config['strat_classification']:
            candle.strat_type = await self._classify_strat_candle(candle, context)
        
        # Add to timeframe history
        if candle.timeframe in context.timeframes:
            context.timeframes[candle.timeframe].append(candle)
            self.candle_count[candle.timeframe] += 1
        
        # Update current candle
        context.current_candles[candle.timeframe] = None
        
        # Emit candle event
        await self._emit_event(DataEvent.NEW_CANDLE, {
            'candle': candle,
            'context': context
        })
        
        # Check for timeframe completion (all TFs have new candles)
        if self._check_timeframe_alignment(context):
            await self._emit_event(DataEvent.TIMEFRAME_COMPLETE, {
                'symbol': candle.symbol,
                'context': context
            })
    
    async def _update_candles_with_tick(self, context: DataContext, tick: MarketTick):
        """Update building candles with new tick data"""
        current_time = tick.timestamp
        
        for tf in TimeFrame:
            if tf == TimeFrame.TICK:
                continue
            
            # Get or create current candle for timeframe
            current_candle = context.current_candles[tf]
            candle_start_time = self._get_candle_start_time(current_time, tf)
            
            if current_candle is None or current_candle.timestamp != candle_start_time:
                # Start new candle
                current_candle = Candle(
                    symbol=tick.symbol,
                    timeframe=tf,
                    timestamp=candle_start_time,
                    open=tick.last,
                    high=tick.last,
                    low=tick.last,
                    close=tick.last,
                    volume=tick.volume,
                    tick_count=1,
                    is_complete=False
                )
                context.current_candles[tf] = current_candle
                
                # Emit candle update event
                await self._emit_event(DataEvent.CANDLE_UPDATE, {
                    'candle': current_candle,
                    'context': context,
                    'is_new': True
                })
            else:
                # Update existing candle
                current_candle.high = max(current_candle.high, tick.last)
                current_candle.low = min(current_candle.low, tick.last)
                current_candle.close = tick.last
                current_candle.volume += tick.volume
                current_candle.tick_count += 1
                
                # Emit candle update event
                await self._emit_event(DataEvent.CANDLE_UPDATE, {
                    'candle': current_candle,
                    'context': context,
                    'is_new': False
                })
    
    def _get_candle_start_time(self, timestamp: datetime, timeframe: TimeFrame) -> datetime:
        """Get the start time for a candle based on timeframe"""
        if timeframe == TimeFrame.M1:
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == TimeFrame.M5:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.M30:
            minute = (timestamp.minute // 30) * 30
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == TimeFrame.H1:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.D1:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp
    
    async def _monitor_candle_completion(self):
        """Monitor and complete candles based on time"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for context in self.data_contexts.values():
                    for tf, current_candle in context.current_candles.items():
                        if current_candle and not current_candle.is_complete:
                            # Check if candle should be completed
                            if self._should_complete_candle(current_candle, current_time):
                                # Mark as complete and process
                                current_candle.is_complete = True
                                completed_candle = self._copy_candle(current_candle)
                                await self.ingest_candle(completed_candle)
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                self.logger.error(f"Error in candle completion monitor: {e}")
                await asyncio.sleep(1)
    
    def _should_complete_candle(self, candle: Candle, current_time: datetime) -> bool:
        """Determine if a candle should be completed based on time"""
        if candle.timeframe == TimeFrame.M1:
            next_candle_time = candle.timestamp + timedelta(minutes=1)
        elif candle.timeframe == TimeFrame.M5:
            next_candle_time = candle.timestamp + timedelta(minutes=5)
        elif candle.timeframe == TimeFrame.M30:
            next_candle_time = candle.timestamp + timedelta(minutes=30)
        elif candle.timeframe == TimeFrame.H1:
            next_candle_time = candle.timestamp + timedelta(hours=1)
        elif candle.timeframe == TimeFrame.D1:
            next_candle_time = candle.timestamp + timedelta(days=1)
        else:
            return False
        
        return current_time >= next_candle_time
    
    def _copy_candle(self, candle: Candle) -> Candle:
        """Create a copy of a candle"""
        return Candle(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            tick_count=candle.tick_count,
            strat_type=candle.strat_type,
            is_complete=True
        )
    
    async def _classify_strat_candle(self, candle: Candle, context: DataContext) -> str:
        """Classify candle as STRAT type (1, 2u, 2d, 3)"""
        timeframe_history = context.timeframes.get(candle.timeframe)
        
        if not timeframe_history or len(timeframe_history) == 0:
            return '1'  # Default to inside bar
        
        prev_candle = timeframe_history[-1]
        
        # STRAT classification logic
        if candle.high > prev_candle.high and candle.low < prev_candle.low:
            return '3'  # Outside bar
        elif candle.high > prev_candle.high:
            return '2u'  # Directional up
        elif candle.low < prev_candle.low:
            return '2d'  # Directional down
        else:
            return '1'  # Inside bar
    
    def _check_timeframe_alignment(self, context: DataContext) -> bool:
        """Check if all timeframes have aligned candles"""
        # Simple check - can be made more sophisticated
        return len([tf for tf, candles in context.timeframes.items() if candles]) >= 3
    
    async def _emit_event(self, event_type: DataEvent, data: Dict):
        """Emit event to registered handlers"""
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    # Run synchronous handlers in thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool, handler, data
                    )
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type.value}: {e}")
    
    def get_latest_candles(self, symbol: str, timeframe: TimeFrame, count: int = 100) -> List[Candle]:
        """Get latest candles for a symbol and timeframe"""
        context = self.data_contexts.get(symbol)
        if not context or timeframe not in context.timeframes:
            return []
        
        candles = list(context.timeframes[timeframe])
        return candles[-count:] if count else candles
    
    def get_current_candle(self, symbol: str, timeframe: TimeFrame) -> Optional[Candle]:
        """Get currently building candle"""
        context = self.data_contexts.get(symbol)
        if not context:
            return None
        
        return context.current_candles.get(timeframe)
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline performance statistics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'total_ticks_processed': self.tick_count,
            'candles_by_timeframe': dict(self.candle_count),
            'avg_processing_time_ms': avg_processing_time,
            'queue_size': self.processing_queue.qsize(),
            'active_symbols': len(self.symbols),
            'running': self.running
        }

# Example usage and integration
class StratDataHandler:
    """Example handler that processes STRAT events"""
    
    def __init__(self):
        self.logger = logging.getLogger('strat_handler')
    
    async def handle_new_candle(self, data: Dict):
        """Handle new completed candle"""
        candle = data['candle']
        context = data['context']
        
        # Log STRAT classification
        self.logger.info(f"New {candle.timeframe.value} {candle.strat_type} candle for {candle.symbol}: "
                        f"O:{candle.open:.2f} H:{candle.high:.2f} L:{candle.low:.2f} C:{candle.close:.2f}")
        
        # Trigger STRAT analysis if needed
        if candle.strat_type in ['2u', '2d', '3']:
            await self._trigger_strat_analysis(candle, context)
    
    async def handle_timeframe_complete(self, data: Dict):
        """Handle timeframe alignment completion"""
        symbol = data['symbol']
        context = data['context']
        
        self.logger.info(f"Timeframe alignment complete for {symbol} - triggering full analysis")
        # This is where you'd trigger the complete STRAT analysis pipeline
    
    async def _trigger_strat_analysis(self, candle: Candle, context: DataContext):
        """Trigger STRAT pattern analysis"""
        # This would integrate with your STRAT agents
        pass