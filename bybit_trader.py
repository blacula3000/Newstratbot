"""
Bybit Trading Integration for STRAT Signals
Handles live trading execution, position management, and risk controls
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pybit.unified_trading import HTTP
import json
from config import get_config
from strat_signal_engine import StratSignalEngine

class BybitTrader:
    """Bybit trading client with STRAT integration"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger('bybit_trader')
        
        # Initialize Bybit client
        if self.config.TRADING_MODE != 'paper':
            self.client = HTTP(
                testnet=self.config.BYBIT_TESTNET,
                api_key=self.config.BYBIT_API_KEY,
                api_secret=self.config.BYBIT_API_SECRET
            )
        else:
            self.client = None
            self.logger.info("Paper trading mode - no live connection")
        
        # Trading state
        self.positions = {}
        self.daily_pnl = 0
        self.trade_count = 0
        self.last_balance_check = None
        self.account_balance = 0
        
        # Risk management
        self.max_positions = self.config.MAX_CONCURRENT_POSITIONS
        self.daily_loss_limit = self.config.DAILY_LOSS_LIMIT
        self.trading_enabled = True
        
        self.logger.info(f"Bybit Trader initialized - Mode: {self.config.TRADING_MODE}")
    
    def connect(self) -> bool:
        """Test connection to Bybit API"""
        if self.config.TRADING_MODE == 'paper':
            self.logger.info("Paper trading mode - simulated connection OK")
            return True
            
        try:
            # Test connection with account info request
            account_info = self.client.get_wallet_balance(accountType="UNIFIED")
            
            if account_info['retCode'] == 0:
                self.logger.info("✅ Successfully connected to Bybit API")
                self._update_account_balance()
                return True
            else:
                self.logger.error(f"❌ Bybit API error: {account_info['retMsg']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {str(e)}")
            return False
    
    def _update_account_balance(self):
        """Update account balance information"""
        if self.config.TRADING_MODE == 'paper':
            self.account_balance = 10000  # Demo balance
            return
            
        try:
            balance_info = self.client.get_wallet_balance(accountType="UNIFIED")
            
            if balance_info['retCode'] == 0:
                # Get USDT balance
                for coin in balance_info['result']['list'][0]['coin']:
                    if coin['coin'] == 'USDT':
                        self.account_balance = float(coin['availableToWithdraw'])
                        break
                        
                self.last_balance_check = datetime.now()
                self.logger.info(f"Account balance updated: ${self.account_balance:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error updating balance: {str(e)}")
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        if self.config.TRADING_MODE == 'paper':
            return self.positions
            
        try:
            positions_info = self.client.get_positions(category="linear")
            
            if positions_info['retCode'] == 0:
                active_positions = {}
                
                for position in positions_info['result']['list']:
                    if float(position['size']) > 0:  # Only active positions
                        symbol = position['symbol']
                        active_positions[symbol] = {
                            'symbol': symbol,
                            'side': position['side'],
                            'size': float(position['size']),
                            'entry_price': float(position['avgPrice']),
                            'mark_price': float(position['markPrice']),
                            'unrealized_pnl': float(position['unrealisedPnl']),
                            'percentage': float(position['unrealisedPnl']) / float(position['positionValue']) * 100
                        }
                
                self.positions = active_positions
                return active_positions
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    def place_order(self, signal: Dict) -> Dict:
        """Place an order based on STRAT signal"""
        result = {
            'success': False,
            'order_id': None,
            'message': '',
            'signal': signal
        }
        
        try:
            # Pre-trade validation
            if not self._validate_trade(signal):
                result['message'] = "Trade validation failed"
                return result
            
            symbol = signal['symbol'] + 'USDT'  # Bybit format
            side = 'Buy' if signal['direction'] == 'LONG' else 'Sell'
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                result['message'] = "Invalid position size calculated"
                return result
            
            # Paper trading simulation
            if self.config.TRADING_MODE == 'paper':
                return self._simulate_order(signal, side, position_size)
            
            # Place actual order
            order_response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(position_size),
                timeInForce="IOC"  # Immediate or Cancel
            )
            
            if order_response['retCode'] == 0:
                order_id = order_response['result']['orderId']
                
                # Set stop loss and take profit
                self._set_stop_loss_take_profit(symbol, signal, side, position_size)
                
                result.update({
                    'success': True,
                    'order_id': order_id,
                    'message': f"Order placed successfully for {symbol}",
                    'position_size': position_size
                })
                
                # Log the trade
                self._log_trade(signal, result)
                
            else:
                result['message'] = f"Order failed: {order_response['retMsg']}"
                
        except Exception as e:
            result['message'] = f"Order execution error: {str(e)}"
            self.logger.error(result['message'])
        
        return result
    
    def _simulate_order(self, signal: Dict, side: str, position_size: float) -> Dict:
        """Simulate order for paper trading"""
        order_id = f"PAPER_{int(time.time())}_{signal['symbol']}"
        
        # Add to paper positions
        self.positions[signal['symbol']] = {
            'symbol': signal['symbol'],
            'side': signal['direction'],
            'size': position_size,
            'entry_price': signal['entry_price'],
            'mark_price': signal['entry_price'],
            'unrealized_pnl': 0,
            'percentage': 0,
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"📄 PAPER TRADE: {side} {position_size} {signal['symbol']} at ${signal['entry_price']:.2f}")
        
        return {
            'success': True,
            'order_id': order_id,
            'message': f"Paper trade executed for {signal['symbol']}",
            'position_size': position_size
        }
    
    def _validate_trade(self, signal: Dict) -> bool:
        """Validate if trade can be executed"""
        
        # Check if trading is enabled
        if not self.trading_enabled:
            self.logger.warning("Trading is disabled")
            return False
        
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.daily_loss_limit:
            self.logger.warning(f"Daily loss limit reached: ${abs(self.daily_pnl):.2f}")
            self.trading_enabled = False
            return False
        
        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            self.logger.warning(f"Maximum positions limit reached: {len(self.positions)}")
            return False
        
        # Check if already have position in this symbol
        if signal['symbol'] in self.positions:
            self.logger.warning(f"Already have position in {signal['symbol']}")
            return False
        
        # Check account balance
        if self.account_balance < self.config.ACCOUNT_BALANCE_THRESHOLD:
            self.logger.warning(f"Account balance too low: ${self.account_balance:.2f}")
            return False
        
        # Check signal confidence
        if signal['confidence_score'] < 70:
            self.logger.warning(f"Signal confidence too low: {signal['confidence_score']}%")
            return False
        
        # Check FTFC score
        ftfc_score = signal.get('ftfc_analysis', {}).get('continuity_score', 0)
        if ftfc_score < 70:
            self.logger.warning(f"FTFC score too low: {ftfc_score}%")
            return False
        
        return True
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on risk management"""
        
        # Get risk amount (percentage of account balance)
        risk_amount = self.account_balance * self.config.MAX_RISK_PER_TRADE
        
        # Calculate stop distance
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        # Calculate position size based on risk
        position_size = risk_amount / stop_distance
        
        # Apply maximum position size limit
        max_position_value = self.account_balance * self.config.DEFAULT_POSITION_SIZE
        max_size = max_position_value / entry_price
        
        position_size = min(position_size, max_size)
        
        # Round to appropriate decimal places
        position_size = round(position_size, 3)
        
        self.logger.info(f"Position size calculated: {position_size} (Risk: ${risk_amount:.2f})")
        
        return position_size
    
    def _set_stop_loss_take_profit(self, symbol: str, signal: Dict, side: str, qty: float):
        """Set stop loss and take profit orders"""
        try:
            # Stop loss order
            stop_side = 'Sell' if side == 'Buy' else 'Buy'
            
            stop_order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=stop_side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(signal['stop_loss']),
                timeInForce="GTC"
            )
            
            # Take profit order
            tp_order = self.client.place_order(
                category="linear", 
                symbol=symbol,
                side=stop_side,
                orderType="Limit",
                qty=str(qty),
                price=str(signal['target']),
                timeInForce="GTC"
            )
            
            if stop_order['retCode'] == 0 and tp_order['retCode'] == 0:
                self.logger.info(f"Stop loss and take profit set for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error setting stop/TP for {symbol}: {str(e)}")
    
    def _log_trade(self, signal: Dict, result: Dict):
        """Log trade details"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'signal_type': signal['signal_type'],
            'confidence': signal['confidence_score'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'position_size': result.get('position_size', 0),
            'order_id': result.get('order_id', ''),
            'success': result['success'],
            'trading_mode': self.config.TRADING_MODE
        }
        
        # Log to trade logger
        if self.config.LOG_TRADES:
            trade_logger = logging.getLogger('trades')
            trade_logger.info(f"TRADE: {json.dumps(trade_log)}")
        
        # Log to console
        self.logger.info(f"🔄 TRADE EXECUTED: {signal['direction']} {signal['symbol']} "
                        f"@ ${signal['entry_price']:.2f} (Confidence: {signal['confidence_score']}%)")
    
    def update_positions(self):
        """Update position information and P&L"""
        current_positions = self.get_positions()
        
        # Calculate daily P&L
        total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in current_positions.values())
        self.daily_pnl = total_pnl
        
        # Check for stop loss hits or targets reached (paper trading)
        if self.config.TRADING_MODE == 'paper':
            self._check_paper_positions()
    
    def _check_paper_positions(self):
        """Check paper trading positions for stop/target hits"""
        # This would simulate price movements and check stop/target levels
        # For now, just log position status
        for symbol, position in self.positions.items():
            self.logger.debug(f"Paper position: {symbol} - Entry: ${position['entry_price']:.2f}")
    
    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        self.update_positions()
        
        return {
            'trading_enabled': self.trading_enabled,
            'trading_mode': self.config.TRADING_MODE,
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.positions),
            'max_positions': self.max_positions,
            'daily_loss_limit': self.daily_loss_limit,
            'trade_count': self.trade_count,
            'positions': list(self.positions.values()) if self.positions else []
        }
    
    def close_position(self, symbol: str) -> Dict:
        """Close a specific position"""
        if symbol not in self.positions:
            return {'success': False, 'message': f'No position found for {symbol}'}
        
        try:
            if self.config.TRADING_MODE == 'paper':
                # Remove from paper positions
                removed_position = self.positions.pop(symbol, None)
                if removed_position:
                    self.logger.info(f"📄 PAPER CLOSE: {symbol} position closed")
                    return {'success': True, 'message': f'Paper position closed for {symbol}'}
            else:
                # Close actual position
                position = self.positions[symbol]
                side = 'Sell' if position['side'] == 'LONG' else 'Buy'
                
                close_order = self.client.place_order(
                    category="linear",
                    symbol=f"{symbol}USDT",
                    side=side,
                    orderType="Market",
                    qty=str(position['size']),
                    timeInForce="IOC"
                )
                
                if close_order['retCode'] == 0:
                    self.logger.info(f"Position closed for {symbol}")
                    return {'success': True, 'message': f'Position closed for {symbol}'}
                else:
                    return {'success': False, 'message': f'Failed to close {symbol}: {close_order["retMsg"]}'}
                    
        except Exception as e:
            error_msg = f"Error closing position for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        self.trading_enabled = False
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            result = self.close_position(symbol)
            self.logger.info(f"Emergency close {symbol}: {result['message']}")

# Example usage and testing
if __name__ == "__main__":
    # Test the trader setup
    from config import validate_setup
    
    if not validate_setup():
        print("❌ Configuration validation failed!")
        exit(1)
    
    # Initialize trader
    trader = BybitTrader()
    
    # Test connection
    if trader.connect():
        print("✅ Trader connected successfully")
        
        # Get trading status
        status = trader.get_trading_status()
        print(f"Trading Status: {status}")
        
        # Test with a demo signal
        demo_signal = {
            'symbol': 'BTCUSDT',
            'signal_type': '2-1-2 Reversal',
            'direction': 'LONG',
            'confidence_score': 85,
            'entry_price': 45000.0,
            'stop_loss': 44000.0,
            'target': 47000.0,
            'ftfc_analysis': {'continuity_score': 78}
        }
        
        print(f"\n🧪 Testing with demo signal...")
        if trader.config.TRADING_MODE == 'paper':
            result = trader.place_order(demo_signal)
            print(f"Trade result: {result}")
        else:
            print("⚠️  Live trading mode - skipping demo trade")
    else:
        print("❌ Trader connection failed!")