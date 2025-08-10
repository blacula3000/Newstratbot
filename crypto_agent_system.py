"""
Crypto Agent System - Integration of professional agents with Bybit trading
Optimized for cryptocurrency STRAT trading with institutional controls
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# Import the new agent system
from agents.agent_coordinator import AgentCoordinator
from agents.data_quality_agent import DataQualityAgent
from agents.market_regime_agent import MarketRegimeAgent, MarketRegime
from agents.liquidity_microstructure_agent import LiquidityMicrostructureAgent
from agents.execution_agent import ExecutionAgent
from agents.order_health_monitor import OrderHealthMonitor
from agents.risk_governance_agent import RiskGovernanceAgent, RiskLimits
from agents.compliance_journal_agent import ComplianceJournalAgent
from agents.attribution_drift_agent import AttributionDriftAgent, SignalType

# Import existing components
from bybit_trader import BybitTrader
from strat_signal_engine import StratSignalEngine
from config import get_config

class CryptoAgentSystem:
    """
    Complete crypto trading system with institutional-grade agents
    """
    
    def __init__(self, config_overrides: Optional[Dict] = None):
        self.config = get_config()
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.config, key, value)
        
        self.logger = logging.getLogger('crypto_agent_system')
        
        # Initialize crypto-specific risk limits
        crypto_risk_limits = RiskLimits(
            max_daily_loss_pct=3.0,  # 3% max daily loss for crypto volatility
            max_daily_loss_usd=self.config.DAILY_LOSS_LIMIT,
            max_positions=self.config.MAX_CONCURRENT_POSITIONS,
            max_position_size_pct=8.0,  # 8% per position for crypto
            max_sector_exposure_pct=50.0,  # 50% crypto sector (all crypto)
            max_correlation=0.8,  # Higher correlation tolerance for crypto
            max_leverage=3.0,  # 3x leverage max for crypto
            max_drawdown_pct=15.0,  # 15% max drawdown
            min_win_rate=0.35,  # 35% minimum win rate
            max_r_per_trade=3.0,  # Max 3R risk per crypto trade
            restricted_times=[],  # 24/7 crypto trading
            blacklist_symbols=[],
            require_confirmation_above=50000  # $50k crypto trades
        )
        
        # Initialize Bybit client for agent system
        self.bybit_client = None
        if self.config.TRADING_MODE != 'paper':
            self.bybit_client = HTTP(
                testnet=self.config.BYBIT_TESTNET,
                api_key=self.config.BYBIT_API_KEY,
                api_secret=self.config.BYBIT_API_SECRET
            )
        
        # Initialize agent coordinator with crypto configuration
        agent_config = {
            'decision_timeout_seconds': 60,  # Longer timeout for crypto analysis
            'max_concurrent_decisions': 5,
            'min_confidence_threshold': 0.65,  # Higher confidence for crypto volatility
            'max_risk_score': 75,  # More conservative risk for crypto
            'min_data_quality_score': 75,
            'min_liquidity_score': 65,
            'agent_weights': {
                'data_quality': 0.20,
                'market_regime': 0.25,  # Higher weight for crypto regime
                'liquidity': 0.15,
                'risk_governance': 0.30,  # Higher risk focus
                'execution': 0.08,
                'compliance': 0.02
            }
        }
        
        self.agent_coordinator = AgentCoordinator(agent_config, self.bybit_client)
        
        # Override risk governance agent with crypto-specific limits
        self.agent_coordinator.agents['risk_governance'] = RiskGovernanceAgent(crypto_risk_limits)
        
        # Initialize legacy components
        self.bybit_trader = BybitTrader()
        self.signal_engine = StratSignalEngine()
        
        # Trading state
        self.active_symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'DOT', 'ADA', 'LINK']
        self.trading_active = False
        self.last_signal_check = {}
        
        # Performance tracking
        self.session_stats = {
            'signals_processed': 0,
            'trades_executed': 0,
            'trades_successful': 0,
            'total_pnl': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Crypto Agent System initialized with institutional controls")
    
    async def start_trading(self):
        """Start the crypto trading system"""
        self.logger.info("🚀 Starting Crypto Agent Trading System...")
        
        # Test connections
        if not await self._test_connections():
            self.logger.error("❌ Connection tests failed - aborting startup")
            return False
        
        # Initialize baseline performance metrics
        await self._initialize_baselines()
        
        self.trading_active = True
        
        # Start background monitoring
        monitor_task = asyncio.create_task(self._background_monitoring())
        
        # Start main trading loop
        trading_task = asyncio.create_task(self._trading_loop())
        
        try:
            await asyncio.gather(monitor_task, trading_task)
        except KeyboardInterrupt:
            self.logger.info("👋 Shutting down trading system...")
            await self.stop_trading()
    
    async def _test_connections(self) -> bool:
        """Test all system connections"""
        success = True
        
        # Test Bybit connection
        if not self.bybit_trader.connect():
            self.logger.error("❌ Bybit connection failed")
            success = False
        else:
            self.logger.info("✅ Bybit connected successfully")
        
        # Test agent system health
        try:
            health = self.agent_coordinator.get_system_status()
            if health.overall_status.value in ['healthy', 'warning']:
                self.logger.info(f"✅ Agent system status: {health.overall_status.value}")
            else:
                self.logger.error(f"❌ Agent system unhealthy: {health.overall_status.value}")
                success = False
        except Exception as e:
            self.logger.error(f"❌ Agent system test failed: {e}")
            success = False
        
        return success
    
    async def _initialize_baselines(self):
        """Initialize performance baselines for agents"""
        try:
            # Get recent market data for each symbol
            for symbol in self.active_symbols:
                # Initialize last signal check
                self.last_signal_check[symbol] = datetime.now() - timedelta(minutes=30)
                
                # Update market regime for the symbol
                regime_agent = self.agent_coordinator.agents['market_regime']
                regime_agent.current_regime = "crypto_normal"  # Default crypto regime
            
            self.logger.info("✅ Agent baselines initialized")
        except Exception as e:
            self.logger.error(f"Error initializing baselines: {e}")
    
    async def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("🔄 Starting main trading loop...")
        
        while self.trading_active:
            try:
                # Process each active symbol
                for symbol in self.active_symbols:
                    await self._process_symbol(symbol)
                    
                    # Small delay between symbols
                    await asyncio.sleep(1)
                
                # Wait before next full cycle
                await asyncio.sleep(30)  # 30-second cycle for crypto
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _process_symbol(self, symbol: str):
        """Process trading signals for a symbol"""
        try:
            # Check if enough time has passed since last signal
            if self.last_signal_check.get(symbol):
                time_since = (datetime.now() - self.last_signal_check[symbol]).total_seconds()
                if time_since < 300:  # 5 minutes minimum between signals
                    return
            
            # Get market data (mock implementation - integrate with your data source)
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return
            
            # Generate STRAT signals
            strat_signals = self._generate_strat_signals(symbol, market_data)
            
            # Process each signal through agent pipeline
            for signal in strat_signals:
                await self._process_trading_signal(symbol, signal)
                
                self.session_stats['signals_processed'] += 1
            
            self.last_signal_check[symbol] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data for symbol"""
        try:
            # If using live mode, get from Bybit
            if self.config.TRADING_MODE != 'paper' and self.bybit_client:
                # Get recent kline data
                klines = self.bybit_client.get_kline(
                    category="linear",
                    symbol=f"{symbol}USDT",
                    interval="15",  # 15-minute candles
                    limit=100
                )
                
                if klines['retCode'] == 0:
                    # Convert to DataFrame format expected by agents
                    df_data = []
                    for kline in klines['result']['list']:
                        df_data.append({
                            'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        })
                    
                    df = pd.DataFrame(df_data).sort_values('timestamp')
                    return {
                        'symbol': symbol,
                        'data': df,
                        'current_price': df.iloc[-1]['close']
                    }
            
            # Mock data for paper trading
            return self._generate_mock_data(symbol)
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _generate_mock_data(self, symbol: str) -> Dict:
        """Generate mock market data for testing"""
        # Create realistic crypto price data
        base_prices = {'BTC': 43000, 'ETH': 2600, 'SOL': 100, 'AVAX': 25, 'MATIC': 0.8}
        base_price = base_prices.get(symbol, 100)
        
        # Generate 100 candles of mock data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        prices = []
        
        current_price = base_price
        for i in range(100):
            # Random walk with crypto-like volatility
            change_pct = np.random.normal(0, 0.015)  # 1.5% standard deviation
            current_price *= (1 + change_pct)
            
            high = current_price * (1 + abs(np.random.normal(0, 0.008)))
            low = current_price * (1 - abs(np.random.normal(0, 0.008)))
            volume = np.random.randint(1000, 10000)
            
            prices.append({
                'timestamp': dates[i],
                'open': current_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })
        
        df = pd.DataFrame(prices)
        return {
            'symbol': symbol,
            'data': df,
            'current_price': current_price
        }
    
    def _generate_strat_signals(self, symbol: str, market_data: Dict) -> List[Dict]:
        """Generate STRAT signals for the symbol"""
        signals = []
        
        try:
            df = market_data['data']
            current_price = market_data['current_price']
            
            # Use existing signal engine to detect STRAT patterns
            strat_analysis = self.signal_engine.analyze_symbol_data(df)
            
            # Convert to agent-compatible signal format
            if strat_analysis and 'signals' in strat_analysis:
                for signal in strat_analysis['signals']:
                    if signal.get('confidence_score', 0) > 60:  # Minimum confidence
                        
                        # Calculate stop loss and target based on STRAT rules
                        entry_price = current_price
                        atr = self._calculate_atr(df)
                        
                        if signal['direction'] == 'LONG':
                            stop_loss = entry_price - (atr * 1.5)
                            target = entry_price + (atr * 2.5)
                        else:
                            stop_loss = entry_price + (atr * 1.5)
                            target = entry_price - (atr * 2.5)
                        
                        crypto_signal = {
                            'symbol': symbol,
                            'action': 'buy' if signal['direction'] == 'LONG' else 'sell',
                            'side': signal['direction'],
                            'size': 0,  # Will be calculated by risk agent
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'signal_type': signal.get('pattern_type', 'STRAT'),
                            'confidence_score': signal.get('confidence_score', 0),
                            'timeframe': '15m',
                            'rationale': f"{signal.get('pattern_type', 'STRAT')} pattern detected with {signal.get('confidence_score', 0)}% confidence",
                            'metadata': {
                                'strat_pattern': signal.get('pattern_type'),
                                'ftfc_score': signal.get('ftfc_analysis', {}).get('continuity_score', 0),
                                'market_structure': signal.get('market_structure', 'unknown'),
                                'crypto_symbol': symbol
                            }
                        }
                        
                        signals.append(crypto_signal)
            
        except Exception as e:
            self.logger.error(f"Error generating STRAT signals for {symbol}: {e}")
        
        return signals
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low'] 
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else df['close'].iloc[-1] * 0.02  # 2% fallback
            
        except Exception:
            return df['close'].iloc[-1] * 0.02  # 2% fallback
    
    async def _process_trading_signal(self, symbol: str, signal: Dict):
        """Process a trading signal through the agent pipeline"""
        try:
            # Update market regime based on crypto conditions
            await self._update_crypto_regime(symbol)
            
            # Process through agent coordinator
            decision = await self.agent_coordinator.process_trading_signal(signal)
            
            self.logger.info(f"🔍 Agent Decision for {symbol}: {decision.action} "
                           f"(Confidence: {decision.confidence:.1%})")
            
            # Execute if approved
            if decision.action in ['buy', 'sell'] and decision.size > 0:
                await self._execute_crypto_trade(decision)
            else:
                self.logger.info(f"⏸️ Trade blocked for {symbol}: {decision.reasoning}")
                
        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    async def _update_crypto_regime(self, symbol: str):
        """Update market regime specific to crypto conditions"""
        try:
            # Get regime agent
            regime_agent = self.agent_coordinator.agents['market_regime']
            
            # Simple crypto regime classification
            # In production, you'd use more sophisticated crypto-specific indicators
            market_data = await self._get_market_data(symbol)
            if market_data:
                df = market_data['data']
                
                # Calculate volatility
                returns = df['close'].pct_change().dropna()
                volatility = returns.std()
                
                # Classify crypto regime
                if volatility > 0.05:  # 5% volatility
                    crypto_regime = "high_volatility"
                elif volatility > 0.03:  # 3% volatility
                    crypto_regime = "normal_volatility"
                else:
                    crypto_regime = "low_volatility"
                
                regime_agent.current_regime = crypto_regime
                
        except Exception as e:
            self.logger.error(f"Error updating crypto regime: {e}")
    
    async def _execute_crypto_trade(self, decision):
        """Execute crypto trade using Bybit trader"""
        try:
            # Convert agent decision to Bybit trader format
            bybit_signal = {
                'symbol': decision.symbol,
                'direction': 'LONG' if decision.action == 'buy' else 'SHORT',
                'signal_type': decision.metadata.get('strat_pattern', 'STRAT'),
                'confidence_score': decision.confidence * 100,
                'entry_price': decision.execution_params.get('entry_price', 0),
                'stop_loss': decision.execution_params.get('stop_loss', 0),
                'target': decision.execution_params.get('target', 0),
                'ftfc_analysis': {
                    'continuity_score': decision.metadata.get('ftfc_score', 75)
                }
            }
            
            # Execute through Bybit trader
            result = self.bybit_trader.place_order(bybit_signal)
            
            if result['success']:
                self.session_stats['trades_executed'] += 1
                self.logger.info(f"✅ Trade executed: {result['message']}")
                
                # Record for attribution analysis
                attribution_agent = self.agent_coordinator.agents['attribution']
                attribution_agent.record_signal_result(
                    signal_id=f"{decision.symbol}_{decision.decision_id}",
                    signal_type=SignalType.STRAT_PATTERN,
                    symbol=decision.symbol,
                    timeframe='15m',
                    pnl=0,  # Will be updated when position closes
                    entry_time=datetime.now(),
                    exit_time=datetime.now(),  # Placeholder
                    metadata=decision.metadata
                )
                
            else:
                self.logger.error(f"❌ Trade failed: {result['message']}")
                
        except Exception as e:
            self.logger.error(f"Error executing crypto trade: {e}")
    
    async def _background_monitoring(self):
        """Background system monitoring"""
        while self.trading_active:
            try:
                # Update system health
                health = self.agent_coordinator.get_system_status()
                
                if health.overall_status.value == 'critical':
                    self.logger.error("🚨 CRITICAL system health - stopping trading")
                    await self.emergency_stop()
                    break
                elif health.overall_status.value == 'degraded':
                    self.logger.warning("⚠️ System health degraded")
                
                # Update positions and P&L
                self.bybit_trader.update_positions()
                
                # Log periodic status
                self._log_session_stats()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(30)
    
    def _log_session_stats(self):
        """Log session statistics"""
        uptime = datetime.now() - self.session_stats['start_time']
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        self.logger.info(f"📊 Session Stats: {self.session_stats['signals_processed']} signals, "
                        f"{self.session_stats['trades_executed']} trades, "
                        f"Uptime: {uptime_str}")
    
    async def emergency_stop(self):
        """Emergency stop all trading"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        
        self.trading_active = False
        
        # Stop Bybit trader
        self.bybit_trader.emergency_stop()
        
        # Shutdown agent coordinator
        await self.agent_coordinator.shutdown()
    
    async def stop_trading(self):
        """Graceful shutdown"""
        self.logger.info("⏹️ Stopping trading system...")
        
        self.trading_active = False
        
        # Final session report
        self._log_session_stats()
        
        # Generate final reports
        await self._generate_final_reports()
        
        # Shutdown components
        await self.agent_coordinator.shutdown()
        
        self.logger.info("👋 Trading system stopped")
    
    async def _generate_final_reports(self):
        """Generate final session reports"""
        try:
            # Attribution report
            attribution_agent = self.agent_coordinator.agents['attribution']
            attribution_report = attribution_agent.generate_attribution_report(days=1)
            
            # Trading status
            trading_status = self.bybit_trader.get_trading_status()
            
            # System performance
            performance = self.agent_coordinator.get_agent_performance_summary()
            
            self.logger.info("📋 Final Session Report:")
            self.logger.info(f"  Total Signals: {self.session_stats['signals_processed']}")
            self.logger.info(f"  Trades Executed: {self.session_stats['trades_executed']}")
            self.logger.info(f"  Account Balance: ${trading_status.get('account_balance', 0):.2f}")
            self.logger.info(f"  Daily P&L: ${trading_status.get('daily_pnl', 0):.2f}")
            self.logger.info(f"  System Status: {performance.get('system_status', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error generating final reports: {e}")

# Main execution
async def main():
    """Main function to run the crypto agent system"""
    
    # Crypto-specific configuration overrides
    crypto_config = {
        'MAX_CONCURRENT_POSITIONS': 5,  # Max 5 crypto positions
        'DAILY_LOSS_LIMIT': 2000,  # $2000 daily loss limit
        'DEFAULT_POSITION_SIZE': 0.08,  # 8% position size for crypto
        'MAX_RISK_PER_TRADE': 0.02,  # 2% risk per trade
        'TRADING_MODE': 'paper'  # Start with paper trading
    }
    
    # Initialize system
    crypto_system = CryptoAgentSystem(crypto_config)
    
    try:
        # Start trading
        await crypto_system.start_trading()
    except KeyboardInterrupt:
        await crypto_system.stop_trading()
    except Exception as e:
        logging.error(f"System error: {e}")
        await crypto_system.emergency_stop()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the system
    asyncio.run(main())