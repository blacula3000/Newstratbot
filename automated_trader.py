"""
Automated STRAT Trading System
Combines signal detection with live Bybit execution
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import schedule

from strat_signal_engine import StratSignalEngine
from bybit_trader import BybitTrader
from config import get_config

class AutomatedSTRATTrader:
    """Main automated trading system"""
    
    def __init__(self, watchlist: List[str] = None):
        self.config = get_config()
        self.logger = logging.getLogger('automated_trader')
        
        # Initialize components
        self.signal_engine = StratSignalEngine()
        self.trader = BybitTrader()
        
        # Watchlist
        self.watchlist = watchlist or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
        
        # System state
        self.running = False
        self.last_scan_time = None
        self.scan_interval = 300  # 5 minutes
        self.signals_found = []
        self.executed_trades = []
        
        # Performance tracking
        self.total_signals = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        self.logger.info("Automated STRAT Trader initialized")
    
    def start(self):
        """Start the automated trading system"""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("🚀 Starting Automated STRAT Trading System")
        
        # Test connections first
        if not self.trader.connect():
            self.logger.error("❌ Failed to connect to Bybit - cannot start trading")
            return False
        
        # Validate configuration
        validation = self.config.validate_config()
        if not validation['valid']:
            self.logger.error("❌ Configuration validation failed:")
            for error in validation['errors']:
                self.logger.error(f"   - {error}")
            return False
        
        self.running = True
        
        # Schedule regular scans
        schedule.every(5).minutes.do(self._scan_and_trade)
        schedule.every(1).hours.do(self._update_positions)
        schedule.every(1).days.at("00:00").do(self._daily_reset)
        
        # Start scheduler in separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Initial scan
        self._scan_and_trade()
        
        self.logger.info("✅ Automated trading system started successfully")
        self.logger.info(f"📊 Monitoring {len(self.watchlist)} symbols")
        self.logger.info(f"🔄 Scanning every {self.scan_interval//60} minutes")
        self.logger.info(f"💰 Trading mode: {self.config.TRADING_MODE.upper()}")
        
        return True
    
    def stop(self):
        """Stop the automated trading system"""
        if not self.running:
            return
            
        self.logger.info("🛑 Stopping Automated STRAT Trading System")
        self.running = False
        
        # Clear scheduled jobs
        schedule.clear()
        
        self.logger.info("✅ Automated trading system stopped")
    
    def _run_scheduler(self):
        """Run the job scheduler"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scan_and_trade(self):
        """Main scanning and trading logic"""
        if not self.running:
            return
            
        try:
            self.logger.info("🔍 Scanning for STRAT signals...")
            scan_start = time.time()
            
            # Scan for signals
            signals = self.signal_engine.scan_multiple_symbols(self.watchlist, '15m')
            
            scan_time = time.time() - scan_start
            self.last_scan_time = datetime.now()
            
            self.logger.info(f"📊 Scan completed in {scan_time:.1f}s - Found {len(signals)} signals")
            
            # Process each signal
            for signal in signals:
                self._process_signal(signal)
            
            # Update statistics
            self.total_signals += len(signals)
            
            # Log summary
            self._log_scan_summary(signals, scan_time)
            
        except Exception as e:
            self.logger.error(f"Error in scan_and_trade: {str(e)}")
    
    def _process_signal(self, signal: Dict):
        """Process an individual signal"""
        try:
            symbol = signal['symbol']
            
            # Check if we already have a position
            current_positions = self.trader.get_positions()
            if symbol in current_positions:
                self.logger.info(f"⏭️  Skipping {symbol} - already have position")
                return
            
            # Check if we've already traded this signal recently
            if self._recently_traded(signal):
                self.logger.info(f"⏭️  Skipping {symbol} - recently traded")
                return
            
            # Execute the trade
            self.logger.info(f"🎯 Processing signal: {symbol} {signal['direction']} "
                           f"({signal['signal_type']}) - Confidence: {signal['confidence_score']}%")
            
            trade_result = self.trader.place_order(signal)
            
            if trade_result['success']:
                self.successful_trades += 1
                self.executed_trades.append({
                    'signal': signal,
                    'result': trade_result,
                    'timestamp': datetime.now()
                })
                
                self.logger.info(f"✅ Trade executed: {symbol} {signal['direction']} "
                               f"@ ${signal['entry_price']:.2f}")
                
                # Send notification if configured
                self._send_trade_notification(signal, trade_result)
                
            else:
                self.failed_trades += 1
                self.logger.warning(f"❌ Trade failed: {symbol} - {trade_result['message']}")
                
        except Exception as e:
            self.logger.error(f"Error processing signal for {signal.get('symbol', 'unknown')}: {str(e)}")
    
    def _recently_traded(self, signal: Dict) -> bool:
        """Check if we've recently traded this symbol/signal"""
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        
        # Check last 4 hours for same signal
        recent_cutoff = datetime.now() - timedelta(hours=4)
        
        for trade in self.executed_trades:
            if (trade['timestamp'] > recent_cutoff and 
                trade['signal']['symbol'] == symbol and
                trade['signal']['signal_type'] == signal_type):
                return True
                
        return False
    
    def _update_positions(self):
        """Update position information and check for exits"""
        if not self.running:
            return
            
        try:
            self.logger.info("📊 Updating positions...")
            
            # Update trader positions
            self.trader.update_positions()
            
            # Get current status
            status = self.trader.get_trading_status()
            
            self.logger.info(f"💼 Portfolio Status: {status['active_positions']} positions, "
                           f"Daily P&L: ${status['daily_pnl']:.2f}")
            
            # Check for risk limits
            if abs(status['daily_pnl']) >= status['daily_loss_limit']:
                self.logger.warning("🚨 Daily loss limit reached - stopping trading")
                self.trader.trading_enabled = False
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    def _daily_reset(self):
        """Daily reset of counters and limits"""
        if not self.running:
            return
            
        self.logger.info("🔄 Daily reset - resetting counters and limits")
        
        # Reset daily counters
        self.trader.daily_pnl = 0
        self.trader.trade_count = 0
        self.trader.trading_enabled = True
        
        # Clear old executed trades (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(days=1)
        self.executed_trades = [
            trade for trade in self.executed_trades 
            if trade['timestamp'] > cutoff_time
        ]
        
        # Log daily summary
        self._log_daily_summary()
    
    def _log_scan_summary(self, signals: List[Dict], scan_time: float):
        """Log scanning summary"""
        if signals:
            avg_confidence = sum(s['confidence_score'] for s in signals) / len(signals)
            
            self.logger.info(f"📈 Signals Summary:")
            self.logger.info(f"   • Total signals: {len(signals)}")
            self.logger.info(f"   • Avg confidence: {avg_confidence:.1f}%")
            self.logger.info(f"   • Scan time: {scan_time:.1f}s")
            
            for signal in signals[:3]:  # Log top 3 signals
                ftfc_score = signal.get('ftfc_analysis', {}).get('continuity_score', 0)
                self.logger.info(f"   • {signal['symbol']} {signal['direction']} "
                               f"({signal['confidence_score']}% conf, {ftfc_score:.0f}% FTFC)")
    
    def _log_daily_summary(self):
        """Log daily performance summary"""
        total_trades = self.successful_trades + self.failed_trades
        success_rate = (self.successful_trades / total_trades * 100) if total_trades > 0 else 0
        
        status = self.trader.get_trading_status()
        
        self.logger.info(f"📊 Daily Summary:")
        self.logger.info(f"   • Total signals: {self.total_signals}")
        self.logger.info(f"   • Trades executed: {total_trades}")
        self.logger.info(f"   • Success rate: {success_rate:.1f}%")
        self.logger.info(f"   • Daily P&L: ${status['daily_pnl']:.2f}")
        self.logger.info(f"   • Active positions: {status['active_positions']}")
    
    def _send_trade_notification(self, signal: Dict, result: Dict):
        """Send trade notification (Discord/Telegram if configured)"""
        try:
            message = (f"🎯 STRAT Trade Executed\n"
                      f"Symbol: {signal['symbol']}\n"
                      f"Direction: {signal['direction']}\n"
                      f"Pattern: {signal['signal_type']}\n"
                      f"Entry: ${signal['entry_price']:.2f}\n"
                      f"Stop: ${signal['stop_loss']:.2f}\n"
                      f"Target: ${signal['target']:.2f}\n"
                      f"Confidence: {signal['confidence_score']}%\n"
                      f"Mode: {self.config.TRADING_MODE.upper()}")
            
            # Here you would implement Discord/Telegram notifications
            # For now, just log the notification
            self.logger.info(f"📱 Notification: {message}")
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        trader_status = self.trader.get_trading_status()
        
        return {
            'running': self.running,
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'watchlist_size': len(self.watchlist),
            'total_signals': self.total_signals,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': (self.successful_trades / (self.successful_trades + self.failed_trades) * 100) if (self.successful_trades + self.failed_trades) > 0 else 0,
            'trader_status': trader_status,
            'recent_trades': self.executed_trades[-5:] if self.executed_trades else []
        }
    
    def add_symbol(self, symbol: str):
        """Add symbol to watchlist"""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol.upper())
            self.logger.info(f"Added {symbol} to watchlist")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from watchlist"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol.upper())
            self.logger.info(f"Removed {symbol} from watchlist")
    
    def emergency_stop(self):
        """Emergency stop all trading and close positions"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        
        # Stop the system
        self.stop()
        
        # Emergency stop trader
        self.trader.emergency_stop()

# Command line interface
if __name__ == "__main__":
    import sys
    from config import validate_setup
    
    print("🤖 Automated STRAT Trading System")
    print("=" * 50)
    
    # Validate setup first
    if not validate_setup():
        print("❌ Setup validation failed! Run setup_api.py first.")
        sys.exit(1)
    
    # Initialize system
    watchlist = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT']
    system = AutomatedSTRATTrader(watchlist)
    
    try:
        if system.start():
            print(f"✅ System started successfully!")
            print(f"📊 Monitoring: {', '.join(watchlist)}")
            print(f"⚡ Mode: {system.config.TRADING_MODE.upper()}")
            print("\nPress Ctrl+C to stop...")
            
            # Keep running
            while system.running:
                time.sleep(10)
                
                # Print status every 5 minutes
                if int(time.time()) % 300 == 0:
                    status = system.get_status()
                    print(f"\n📊 Status: {status['successful_trades']} successful trades, "
                          f"{status['trader_status']['active_positions']} positions")
        else:
            print("❌ Failed to start system!")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        system.stop()
        print("👋 System stopped.")
    
    except Exception as e:
        print(f"❌ System error: {e}")
        system.emergency_stop()