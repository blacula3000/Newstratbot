#!/usr/bin/env python3
"""
Local Agent System Runner
Runs the enhanced event-driven trading agents locally for testing and development.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import signal
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.enhanced_data_pipeline import EnhancedDataPipeline, TimeFrame
from agents.trigger_line_agent import TriggerLineAgent
from agents.ftfc_continuity_agent import FTFCContinuityAgent
from agents.volatility_agent import VolatilityAgent
from agents.reversal_setup_agent import ReversalSetupAgent
from agents.magnet_level_agent import MagnetLevelAgent
from agents.position_sizing_agent import PositionSizingAgent
from agents.entry_timing_agent import EntryTimingAgent
from agents.exit_strategy_agent import ExitStrategyAgent
from agents.trade_director import TradeDirector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('local_agents.log')
    ]
)
logger = logging.getLogger(__name__)

class LocalAgentRunner:
    def __init__(self):
        self.running = True
        self.agents = {}
        self.data_pipeline = None
        self.trade_director = None
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize_agents(self, config):
        """Initialize all trading agents"""
        logger.info("Initializing trading agents...")
        
        try:
            # Enhanced Data Pipeline
            self.data_pipeline = EnhancedDataPipeline()
            logger.info("✅ Enhanced Data Pipeline initialized")
            
            # Core Analysis Agents
            self.agents['trigger_line'] = TriggerLineAgent()
            self.agents['ftfc_continuity'] = FTFCContinuityAgent()
            self.agents['volatility'] = VolatilityAgent()
            self.agents['reversal_setup'] = ReversalSetupAgent()
            self.agents['magnet_level'] = MagnetLevelAgent()
            
            # Risk & Execution Agents
            self.agents['position_sizing'] = PositionSizingAgent()
            self.agents['entry_timing'] = EntryTimingAgent()
            self.agents['exit_strategy'] = ExitStrategyAgent()
            
            # Trade Director (Master Orchestrator)
            self.trade_director = TradeDirector(config)
            
            logger.info(f"✅ Initialized {len(self.agents)} specialized agents")
            logger.info("✅ Trade Director initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            return False
    
    async def run_demo_analysis(self, symbol='BTCUSDT', timeframe='5m'):
        """Run a demonstration of the agent system"""
        logger.info(f"🚀 Starting demo analysis for {symbol} on {timeframe}")
        
        try:
            # Simulate market data (in real implementation, this comes from exchanges)
            import yfinance as yf
            import pandas as pd
            
            # Get sample data
            ticker = yf.Ticker(symbol.replace('USDT', '-USD'))
            data = ticker.history(period='5d', interval=timeframe)
            
            if data.empty:
                logger.warning("No data available, using synthetic data for demo")
                data = self.generate_synthetic_data()
            
            logger.info(f"📊 Loaded {len(data)} candles of market data")
            
            # Run individual agent analysis
            await self.demonstrate_agents(symbol, timeframe, data)
            
        except Exception as e:
            logger.error(f"❌ Demo analysis failed: {e}")
    
    async def demonstrate_agents(self, symbol, timeframe, data):
        """Demonstrate each agent's capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🤖 AGENT ANALYSIS DEMONSTRATION")
        logger.info("="*60)
        
        try:
            # Trigger Line Agent Demo
            logger.info("\n🎯 TRIGGER LINE AGENT ANALYSIS:")
            trigger_analysis = self.agents['trigger_line'].analyze_trigger_lines(symbol, timeframe, data)
            logger.info(f"   Active Triggers: {len(trigger_analysis.active_triggers)}")
            logger.info(f"   Break Detected: {trigger_analysis.break_detected}")
            if trigger_analysis.break_detected:
                logger.info(f"   Break Type: {trigger_analysis.break_type}")
                logger.info(f"   Confidence: {trigger_analysis.confidence:.2f}")
            
            # FTFC Continuity Agent Demo
            logger.info("\n📈 FTFC CONTINUITY AGENT ANALYSIS:")
            # For demo, we'll analyze multiple timeframes
            continuity_analysis = self.agents['ftfc_continuity'].analyze_continuity(
                symbol, timeframe, data, ['1h', '4h', '1d']
            )
            logger.info(f"   Continuity Type: {continuity_analysis.continuity_type}")
            logger.info(f"   Alignment Score: {continuity_analysis.alignment_score:.2f}")
            logger.info(f"   Direction: {continuity_analysis.direction}")
            
            # Volatility Agent Demo
            logger.info("\n📊 VOLATILITY AGENT ANALYSIS:")
            vol_metrics = self.agents['volatility'].analyze_volatility(symbol, timeframe, data)
            logger.info(f"   Current Volatility: {vol_metrics.current_volatility:.2%}")
            logger.info(f"   Volatility Regime: {vol_metrics.regime}")
            logger.info(f"   IV Rank: {vol_metrics.iv_rank:.2f}")
            
            # Reversal Setup Agent Demo
            logger.info("\n🔄 REVERSAL SETUP AGENT ANALYSIS:")
            reversal_setup = self.agents['reversal_setup'].analyze_reversal_setup(symbol, timeframe, data)
            logger.info(f"   Reversal Pattern: {reversal_setup.pattern_type}")
            logger.info(f"   Pattern Confidence: {reversal_setup.confidence:.2f}")
            logger.info(f"   Setup Quality: {reversal_setup.setup_quality}")
            
            # Magnet Level Agent Demo
            logger.info("\n🧲 MAGNET LEVEL AGENT ANALYSIS:")
            magnet_analysis = self.agents['magnet_level'].analyze_magnet_levels(symbol, timeframe, data)
            logger.info(f"   Key Levels Found: {len(magnet_analysis.levels)}")
            logger.info(f"   Confluence Zones: {len(magnet_analysis.confluence_zones)}")
            if magnet_analysis.levels:
                strongest = max(magnet_analysis.levels, key=lambda x: x.strength)
                logger.info(f"   Strongest Level: {strongest.price:.2f} ({strongest.level_type})")
            
            # Position Sizing Agent Demo
            logger.info("\n💰 POSITION SIZING AGENT ANALYSIS:")
            signal_data = {
                'direction': 'long',
                'confidence': 0.75,
                'stop_loss': float(data['Low'].iloc[-1] * 0.98),
                'entry_price': float(data['Close'].iloc[-1])
            }
            portfolio_data = {
                'balance': 10000,
                'current_positions': {},
                'max_risk_per_trade': 0.02
            }
            position_calc = self.agents['position_sizing'].calculate_position_size(
                symbol, signal_data, portfolio_data, {'volatility': vol_metrics.current_volatility}
            )
            logger.info(f"   Recommended Size: {position_calc.final_position_size:.4f}")
            logger.info(f"   Risk Amount: ${position_calc.risk_amount:.2f}")
            logger.info(f"   Sizing Method: {position_calc.sizing_method}")
            
        except Exception as e:
            logger.error(f"❌ Agent demonstration failed: {e}")
    
    def generate_synthetic_data(self):
        """Generate synthetic market data for demo purposes"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate 200 periods of synthetic OHLCV data
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=200, freq='5T')
        
        # Simulate price movement with some volatility
        base_price = 50000
        returns = np.random.normal(0, 0.02, 200)
        prices = base_price * (1 + returns).cumprod()
        
        # Create OHLCV structure
        data = pd.DataFrame(index=dates)
        data['Open'] = prices * (1 + np.random.normal(0, 0.001, 200))
        data['High'] = prices * (1 + np.abs(np.random.normal(0, 0.005, 200)))
        data['Low'] = prices * (1 - np.abs(np.random.normal(0, 0.005, 200)))
        data['Close'] = prices
        data['Volume'] = np.random.normal(1000000, 200000, 200)
        
        return data
    
    async def run_continuous_monitoring(self):
        """Run continuous monitoring mode"""
        logger.info("🔄 Starting continuous monitoring mode...")
        logger.info("💡 This would connect to live market data in production")
        
        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"\n🔍 Monitoring Cycle {iteration} - {datetime.now().strftime('%H:%M:%S')}")
                
                # In production, this would process real-time market events
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if iteration % 10 == 0:  # Every 5 minutes, run full analysis
                    await self.run_demo_analysis()
                    
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def run(self, mode='demo'):
        """Main execution method"""
        logger.info("🚀 Starting Local Agent System...")
        self.setup_signal_handlers()
        
        # Configuration
        config = {
            'exchange': 'demo',  # Demo mode
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'base_timeframe': '5m',
            'risk_per_trade': 0.02,
            'demo_mode': True
        }
        
        # Initialize agents
        if not await self.initialize_agents(config):
            logger.error("❌ Failed to initialize agents, exiting...")
            return
        
        logger.info("✅ All agents initialized successfully!")
        
        # Run based on mode
        if mode == 'demo':
            logger.info("🎮 Running in DEMO mode - analyzing sample data")
            await self.run_demo_analysis()
        elif mode == 'monitor':
            logger.info("📡 Running in MONITORING mode - continuous analysis")
            await self.run_continuous_monitoring()
        
        logger.info("👋 Local agent system shutdown complete")

async def main():
    """Main entry point"""
    print("🤖 Newstratbot - Enhanced Event-Driven Trading Agents")
    print("=" * 60)
    print("Choose running mode:")
    print("1. Demo Mode - Analyze sample data and demonstrate capabilities")
    print("2. Monitor Mode - Continuous monitoring (simulated)")
    print("3. Exit")
    
    try:
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == '3':
            print("👋 Goodbye!")
            return
        elif choice == '1':
            mode = 'demo'
        elif choice == '2':
            mode = 'monitor'
        else:
            print("❌ Invalid choice, using demo mode")
            mode = 'demo'
        
        runner = LocalAgentRunner()
        await runner.run(mode)
        
    except KeyboardInterrupt:
        print("\n👋 Graceful shutdown requested")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())