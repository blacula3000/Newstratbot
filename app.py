from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime
import os
from trading_bot import TradingBot, CandleStickPattern
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
bot_instance = None
bot_thread = None
bot_running = False
current_data = {}

class WebTradingBot(TradingBot):
    def __init__(self, symbol: str, timeframe: str = '5m'):
        super().__init__(symbol, timeframe)
        self.web_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': 0,
            'signals': {'bullish_reversal': False, 'bearish_reversal': False},
            'last_update': '',
            'status': 'stopped'
        }
    
    def emit_data(self):
        """Emit data to web interface via SocketIO"""
        socketio.emit('trading_update', self.web_data)
    
    def run_web(self):
        """Modified run method for web interface"""
        global bot_running
        print(f"Starting web bot for {self.symbol}...")
        self.web_data['status'] = 'running'
        
        while bot_running:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                data = self.fetch_data()
                
                if not data.empty:
                    signals = self.analyze_pattern(data)
                    current_price = data['Close'].iloc[-1]
                    
                    # Update web data
                    self.web_data.update({
                        'current_price': float(current_price),
                        'signals': signals,
                        'last_update': current_time,
                        'status': 'running'
                    })
                    
                    # Emit to web interface
                    self.emit_data()
                    
                    print(f"Time: {current_time}, Price: ${current_price:.2f}")
                    if signals['bullish_reversal']:
                        print("🟢 Bullish reversal detected!")
                    elif signals['bearish_reversal']:
                        print("🔴 Bearish reversal detected!")
                
                time.sleep(30)  # Reduced to 30 seconds for web interface
                
            except Exception as e:
                print(f"Error: {e}")
                self.web_data['status'] = 'error'
                self.emit_data()
                time.sleep(60)
        
        self.web_data['status'] = 'stopped'
        self.emit_data()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/advanced')
def advanced_dashboard():
    """Advanced dashboard with watchlist and sector analysis"""
    return render_template('advanced_dashboard.html')

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot_instance, bot_thread, bot_running
    
    symbol = request.form.get('symbol', 'SPY').upper()
    timeframe = request.form.get('timeframe', '5m')
    
    if not bot_running:
        bot_instance = WebTradingBot(symbol, timeframe)
        bot_running = True
        bot_thread = threading.Thread(target=bot_instance.run_web)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({'status': 'success', 'message': f'Bot started for {symbol}'})
    else:
        return jsonify({'status': 'error', 'message': 'Bot is already running'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_running
    
    bot_running = False
    return jsonify({'status': 'success', 'message': 'Bot stopped'})

@app.route('/get_status')
def get_status():
    """Get current bot status"""
    if bot_instance:
        return jsonify(bot_instance.web_data)
    else:
        return jsonify({'status': 'stopped', 'symbol': '', 'current_price': 0})

@app.route('/logs')
def logs():
    """View trading logs"""
    try:
        if os.path.exists('trading_bot.log'):
            with open('trading_bot.log', 'r') as f:
                logs = f.readlines()[-50:]  # Last 50 lines
            return render_template('logs.html', logs=logs)
        else:
            return render_template('logs.html', logs=['No logs available'])
    except Exception as e:
        return render_template('logs.html', logs=[f'Error reading logs: {e}'])

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    if bot_instance:
        emit('trading_update', bot_instance.web_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)