import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid
import pandas as pd
import numpy as np

@dataclass
class Position:
    """Represents a paper trading position"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    status: str  # 'open', 'closed'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class Trade:
    """Represents a completed trade"""
    position_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    duration_minutes: int
    reason: str  # 'stop_loss', 'take_profit', 'manual', 'signal'

class CryptoPaperTrader:
    """Paper trading system for cryptocurrency with The Strat methodology"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.max_position_size = 0.02  # 2% risk per trade
        self.leverage = 1  # Can be adjusted for futures trading
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Risk management
        self.max_open_positions = 3
        self.daily_loss_limit = initial_balance * 0.05  # 5% daily loss limit
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = self.current_balance * self.max_position_size
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # Position size in base currency (e.g., BTC)
        position_size = risk_amount / (entry_price * stop_distance)
        
        # Apply leverage
        position_size *= self.leverage
        
        return round(position_size, 8)
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                     stop_loss: float, take_profit: float, 
                     notes: str = "") -> Optional[Position]:
        """Open a new paper trading position"""
        
        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            print(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return None
        
        # Check max open positions
        open_positions = [p for p in self.positions.values() if p.status == 'open']
        if len(open_positions) >= self.max_open_positions:
            print(f"Maximum open positions ({self.max_open_positions}) reached")
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        # Check if we have enough balance
        required_margin = entry_price * position_size / self.leverage
        if required_margin > self.current_balance:
            print(f"Insufficient balance. Required: ${required_margin:.2f}, Available: ${self.current_balance:.2f}")
            return None
        
        # Create position
        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=position_size,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            status='open',
            notes=notes
        )
        
        self.positions[position.id] = position
        
        print(f"\n📈 POSITION OPENED:")
        print(f"Symbol: {symbol}")
        print(f"Side: {side.upper()}")
        print(f"Entry: ${entry_price:.2f}")
        print(f"Size: {position_size:.8f}")
        print(f"Stop Loss: ${stop_loss:.2f}")
        print(f"Take Profit: ${take_profit:.2f}")
        print(f"Risk: ${self.current_balance * self.max_position_size:.2f}")
        
        return position
    
    def close_position(self, position_id: str, exit_price: float, reason: str = 'manual') -> Optional[Trade]:
        """Close an open position"""
        
        if position_id not in self.positions:
            print(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        if position.status == 'closed':
            print(f"Position {position_id} is already closed")
            return None
        
        # Calculate P&L
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size
        
        pnl_percent = (pnl / (position.entry_price * position.size)) * 100
        
        # Update position
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.pnl = pnl
        position.pnl_percent = pnl_percent
        position.status = 'closed'
        
        # Create trade record
        duration = int((position.exit_time - position.entry_time).total_seconds() / 60)
        trade = Trade(
            position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            pnl=pnl,
            pnl_percent=pnl_percent,
            duration_minutes=duration,
            reason=reason
        )
        
        self.trade_history.append(trade)
        
        # Update balance and metrics
        self.current_balance += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update peak balance and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Print trade result
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"\n{emoji} POSITION CLOSED:")
        print(f"Symbol: {position.symbol}")
        print(f"Side: {position.side.upper()}")
        print(f"Entry: ${position.entry_price:.2f}")
        print(f"Exit: ${exit_price:.2f}")
        print(f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
        print(f"Duration: {duration} minutes")
        print(f"Reason: {reason}")
        print(f"Balance: ${self.current_balance:.2f}")
        
        return trade
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]):
        """Check all open positions for stop loss or take profit hits"""
        
        for position_id, position in list(self.positions.items()):
            if position.status != 'open':
                continue
            
            if position.symbol not in current_prices:
                continue
            
            current_price = current_prices[position.symbol]
            
            # Check stop loss
            if position.side == 'long':
                if current_price <= position.stop_loss:
                    self.close_position(position_id, position.stop_loss, 'stop_loss')
                elif current_price >= position.take_profit:
                    self.close_position(position_id, position.take_profit, 'take_profit')
            else:  # short
                if current_price >= position.stop_loss:
                    self.close_position(position_id, position.stop_loss, 'stop_loss')
                elif current_price <= position.take_profit:
                    self.close_position(position_id, position.take_profit, 'take_profit')
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.status == 'open']
    
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_win = avg_loss = 0
        
        if self.trade_history:
            wins = [t.pnl for t in self.trade_history if t.pnl > 0]
            losses = [t.pnl for t in self.trade_history if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss != 0 and self.losing_trades > 0 else 0
        
        return {
            'current_balance': self.current_balance,
            'total_pnl': self.total_pnl,
            'total_pnl_percent': (self.total_pnl / self.initial_balance * 100),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'daily_pnl': self.daily_pnl
        }
    
    def reset_daily_limits(self):
        """Reset daily loss limits (call this at the start of each trading day)"""
        
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            print(f"Daily limits reset for {current_date}")
    
    def export_trade_history(self, filename: str = 'trade_history.json'):
        """Export trade history to JSON file"""
        
        trades_data = [asdict(trade) for trade in self.trade_history]
        
        # Convert datetime objects to strings
        for trade in trades_data:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump({
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'trades': trades_data,
                'performance': self.get_performance_metrics()
            }, f, indent=2)
        
        print(f"Trade history exported to {filename}")

class StratCryptoSignalGenerator:
    """Generates trading signals based on The Strat methodology for crypto"""
    
    def __init__(self, paper_trader: CryptoPaperTrader):
        self.paper_trader = paper_trader
        self.active_signals = {}
        
    def process_strat_signal(self, data: Dict) -> Optional[Position]:
        """Process a Strat signal and open a paper trade if appropriate"""
        
        symbol = data.get('symbol')
        pattern = data.get('pattern')
        direction = data.get('direction')
        current_price = data.get('current_price')
        confidence = data.get('confidence', 0)
        
        if not all([symbol, pattern, direction, current_price]):
            return None
        
        # Only trade high confidence signals
        if confidence < 70:
            return None
        
        # Calculate stop loss and take profit based on ATR or fixed percentage
        if direction == 'BULLISH':
            side = 'long'
            # For crypto, use tighter stops due to volatility
            stop_loss = current_price * 0.98  # 2% stop loss
            take_profit = current_price * 1.06  # 6% take profit (3:1 RR)
        else:
            side = 'short'
            stop_loss = current_price * 1.02  # 2% stop loss
            take_profit = current_price * 0.94  # 6% take profit (3:1 RR)
        
        # Add pattern-specific adjustments
        if '3_1_2' in pattern:  # Strongest pattern
            # Wider targets for stronger patterns
            if side == 'long':
                take_profit = current_price * 1.08  # 8% target
            else:
                take_profit = current_price * 0.92
        
        notes = f"Strat Pattern: {pattern}, Confidence: {confidence}%"
        
        # Open the position
        position = self.paper_trader.open_position(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notes=notes
        )
        
        return position