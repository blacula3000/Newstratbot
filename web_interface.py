from flask import Flask, render_template, jsonify, request
import threading
from trading_bot import TradingBot
import logging

app = Flask(__name__)

# Global bot instance
bot = None
bot_thread = None

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot."""
    global bot, bot_thread
    
    try:
        if bot_thread and bot_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'Bot is already running'})
        
        config = {
            'api_key': request.json.get('api_key'),
            'api_secret': request.json.get('api_secret'),
            'exchange': request.json.get('exchange', 'binance'),
            'base_url': request.json.get('base_url', 'https://paper-api.alpaca.markets')
        }
        
        symbols = request.json.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        market_type = request.json.get('market_type', 'futures')
        timeframe = request.json.get('timeframe', '1h')
        
        bot = TradingBot(market_type=market_type, timeframe=timeframe, config=config)
        bot_thread = threading.Thread(target=bot.run, args=(symbols,))
        bot_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Bot started successfully'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot."""
    global bot
    
    try:
        if bot:
            bot.stop_trading = True  # We'll add this flag to the TradingBot class
            return jsonify({'status': 'success', 'message': 'Bot stopped successfully'})
        return jsonify({'status': 'error', 'message': 'Bot is not running'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/status')
def get_status():
    """Get current bot status and positions."""
    global bot
    
    if not bot:
        return jsonify({'status': 'inactive'})
    
    try:
        return jsonify({
            'status': 'active',
            'active_trades': bot.active_trades,
            'stop_losses': bot.stop_losses,
            'take_profits': bot.take_profits
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/patterns')
def get_patterns():
    """Get recent pattern history."""
    global bot
    
    if not bot:
        return jsonify({'patterns': []})
    
    try:
        return jsonify({
            'patterns': bot.pattern_history
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 