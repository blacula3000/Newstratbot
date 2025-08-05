from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime
import os
from strat_trading_bot import StratTradingBot, StratScenario, StratPatterns, TimeframeContinuity
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'strat-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
bot_instance = None
bot_thread = None
bot_running = False
current_data = {}

class WebStratTradingBot(StratTradingBot):
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
            'status': 'stopped'
        }
    
    def emit_data(self):
        """Emit data to web interface via SocketIO"""
        socketio.emit('strat_update', self.web_data)
    
    def run_web(self):
        """Modified run method for web interface"""
        global bot_running
        print(f"Starting Strat Web Bot for {self.symbol}...")
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
                    
                    # Emit to web interface
                    self.emit_data()
                    
                    # Log signals
                    if strat_signals['actionable'] and self.should_send_signal():
                        print(f"Strat Signal: {strat_signals['pattern']} @ ${current_price:.2f}")
                        self.last_signal_time = time.time()
                
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
    """Main Strat dashboard page"""
    return render_template('strat_dashboard.html')

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the Strat trading bot"""
    global bot_instance, bot_thread, bot_running
    
    data = request.get_json()
    symbol = data.get('symbol', 'SPY').upper()
    timeframe = data.get('timeframe', '5m')
    
    if not bot_running:
        bot_instance = WebStratTradingBot(symbol, timeframe)
        bot_running = True
        bot_thread = threading.Thread(target=bot_instance.run_web)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({'status': 'success', 'message': f'Strat Bot started for {symbol}'})
    else:
        return jsonify({'status': 'error', 'message': 'Bot is already running'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_running
    
    bot_running = False
    return jsonify({'status': 'success', 'message': 'Strat Bot stopped'})

@app.route('/get_status')
def get_status():
    """Get current bot status"""
    if bot_instance:
        return jsonify(bot_instance.web_data)
    else:
        return jsonify({
            'status': 'stopped', 
            'symbol': '', 
            'current_price': 0,
            'timeframe_continuity': {},
            'strat_signals': {}
        })

@app.route('/logs')
def logs():
    """View trading logs"""
    try:
        log_file = 'strat_trading_bot.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = f.readlines()[-100:]  # Last 100 lines
            return render_template('strat_logs.html', logs=logs)
        else:
            return render_template('strat_logs.html', logs=['No logs available'])
    except Exception as e:
        return render_template('strat_logs.html', logs=[f'Error reading logs: {e}'])

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    if bot_instance:
        emit('strat_update', bot_instance.web_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)