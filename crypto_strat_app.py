from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime
import os
from strat_trading_bot import StratTradingBot, StratScenario, StratPatterns, TimeframeContinuity
from crypto_paper_trading import CryptoPaperTrader, StratCryptoSignalGenerator
from tradingview_webhook import TradingViewWebhook
import pandas as pd
import yfinance as yf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crypto-strat-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
bot_instance = None
bot_thread = None
bot_running = False
paper_trader = CryptoPaperTrader(initial_balance=10000)
signal_generator = StratCryptoSignalGenerator(paper_trader)
webhook_handler = TradingViewWebhook()

# Popular crypto pairs
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum', 
    'BNB-USD': 'Binance Coin',
    'SOL-USD': 'Solana',
    'ADA-USD': 'Cardano',
    'AVAX-USD': 'Avalanche',
    'DOT-USD': 'Polkadot',
    'MATIC-USD': 'Polygon',
    'LINK-USD': 'Chainlink',
    'UNI-USD': 'Uniswap'
}

class CryptoWebStratBot(StratTradingBot):
    def __init__(self, symbol: str, timeframe: str = '5m'):
        super().__init__(symbol, timeframe)
        self.web_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': 0,
            'bar_type': '',
            'sequence': [],
            'pattern': None,
            'strat_signals': {},
            'timeframe_continuity': {},
            'trade_signal': {},
            'last_update': '',
            'status': 'stopped',
            'positions': [],
            'performance': {}
        }
    
    def emit_data(self):
        """Emit data to web interface via SocketIO"""
        # Add current positions and performance
        self.web_data['positions'] = [
            {
                'id': p.id,
                'symbol': p.symbol,
                'side': p.side,
                'entry_price': p.entry_price,
                'size': p.size,
                'current_price': self.web_data['current_price'],
                'unrealized_pnl': self.calculate_unrealized_pnl(p),
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'entry_time': p.entry_time.isoformat()
            }
            for p in paper_trader.get_open_positions()
        ]
        self.web_data['performance'] = paper_trader.get_performance_metrics()
        
        socketio.emit('crypto_strat_update', self.web_data)
    
    def calculate_unrealized_pnl(self, position):
        """Calculate unrealized P&L for a position"""
        current_price = self.web_data['current_price']
        if position.side == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - current_price) * position.size
        return round(pnl, 2)
    
    def run_web(self):
        """Modified run method for web interface with paper trading"""
        global bot_running
        print(f"Starting Crypto Strat Web Bot for {self.symbol}...")
        self.web_data['status'] = 'running'
        
        while bot_running:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                self.current_data = self.fetch_data()
                
                if not self.current_data.empty:
                    # Get Strat patterns
                    strat_signals = self.analyze_strat_patterns(self.current_data)
                    
                    # Check timeframe continuity
                    continuity = self.timeframe_continuity.check_continuity()
                    
                    # Generate trade signal
                    trade_signal = self.generate_trade_signal(strat_signals, continuity)
                    
                    # Current price
                    current_price = self.current_data['Close'].iloc[-1]
                    
                    # Update prices for stop loss/take profit monitoring
                    paper_trader.check_stop_loss_take_profit({self.symbol: current_price})
                    
                    # Update web data
                    self.web_data.update({
                        'current_price': float(current_price),
                        'bar_type': strat_signals['bar_type'],
                        'sequence': strat_signals['sequence'][-3:] if strat_signals['sequence'] else [],
                        'pattern': strat_signals['pattern'],
                        'strat_signals': {
                            'actionable': strat_signals['actionable'],
                            'direction': strat_signals['direction'],
                            'strength': strat_signals['strength']
                        },
                        'timeframe_continuity': continuity,
                        'trade_signal': {
                            'action': trade_signal['action'],
                            'entry': float(trade_signal['entry']) if trade_signal['entry'] else None,
                            'stop_loss': float(trade_signal['stop_loss']) if trade_signal['stop_loss'] else None,
                            'target': float(trade_signal['target']) if trade_signal['target'] else None,
                            'confidence': trade_signal['confidence']
                        },
                        'last_update': current_time,
                        'status': 'running'
                    })
                    
                    # Process signal for paper trading
                    if strat_signals['actionable'] and self.should_send_signal():
                        signal_data = {
                            'symbol': self.symbol,
                            'pattern': strat_signals['pattern'],
                            'direction': strat_signals['direction'],
                            'current_price': current_price,
                            'confidence': trade_signal['confidence']
                        }
                        
                        position = signal_generator.process_strat_signal(signal_data)
                        if position:
                            self.last_signal_time = time.time()
                    
                    # Emit to web interface
                    self.emit_data()
                
                # Reset daily limits at midnight
                paper_trader.reset_daily_limits()
                
                time.sleep(30)  # 30 seconds update
                
            except Exception as e:
                print(f"Error: {e}")
                self.web_data['status'] = 'error'
                self.emit_data()
                time.sleep(60)
        
        self.web_data['status'] = 'stopped'
        self.emit_data()

@app.route('/')
def dashboard():
    """Main crypto Strat dashboard page"""
    return render_template('crypto_dashboard.html', crypto_symbols=CRYPTO_SYMBOLS)

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the crypto Strat trading bot"""
    global bot_instance, bot_thread, bot_running
    
    data = request.get_json()
    symbol = data.get('symbol', 'BTC-USD').upper()
    timeframe = data.get('timeframe', '5m')
    
    if not bot_running:
        bot_instance = CryptoWebStratBot(symbol, timeframe)
        bot_running = True
        bot_thread = threading.Thread(target=bot_instance.run_web)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({'status': 'success', 'message': f'Crypto Strat Bot started for {symbol}'})
    else:
        return jsonify({'status': 'error', 'message': 'Bot is already running'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_running
    
    bot_running = False
    return jsonify({'status': 'success', 'message': 'Crypto Strat Bot stopped'})

@app.route('/close_position/<position_id>', methods=['POST'])
def close_position(position_id):
    """Manually close a position"""
    
    current_price = bot_instance.web_data['current_price'] if bot_instance else 0
    
    if current_price <= 0:
        return jsonify({'error': 'No current price available'}), 400
    
    trade = paper_trader.close_position(position_id, current_price, 'manual')
    
    if trade:
        return jsonify({
            'status': 'closed',
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent
        })
    else:
        return jsonify({'error': 'Position not found or already closed'}), 404

@app.route('/export_trades', methods=['GET'])
def export_trades():
    """Export trade history"""
    
    paper_trader.export_trade_history('crypto_trades.json')
    return jsonify({'status': 'exported', 'filename': 'crypto_trades.json'})

# TradingView webhook endpoint
@app.route('/webhook/tradingview', methods=['POST'])
def tradingview_webhook():
    """Handle TradingView alerts"""
    
    try:
        webhook_data = request.get_json()
        result = webhook_handler.process_webhook(webhook_data)
        
        # Emit update if bot is running
        if bot_instance:
            bot_instance.emit_data()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    if bot_instance:
        emit('crypto_strat_update', bot_instance.web_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)