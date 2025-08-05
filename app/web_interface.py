from flask import Flask, render_template, jsonify, request
import threading
from .trading_bot import TradingBot
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Global bot instance
bot = None
bot_thread = None

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('dashboard.html')

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot."""
    global bot, bot_thread
    
    try:
        if bot_thread and bot_thread.is_alive():
            return jsonify({'status': 'error', 'message': 'Bot is already running'})
        
        config = {
            'api_key': os.getenv('API_KEY'),
            'api_secret': os.getenv('API_SECRET'),
            'base_url': os.getenv('BASE_URL'),
            'exchange': os.getenv('EXCHANGE')
        }
        
        symbols = request.json.get('symbols', ['BTCUSDT'])
        timeframe = request.json.get('timeframe', '1h')
        
        bot = TradingBot(timeframe=timeframe, config=config)
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
            bot.stop_trading = True
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
            'take_profits': bot.take_profits,
            'balance': bot.session.get_wallet_balance()['result']['USDT']['available_balance']
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
            'patterns': bot.pattern_history[-50:]  # Last 50 patterns
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/market-data')
def get_market_data():
    """Get current market data."""
    global bot
    if not bot:
        return jsonify({'status': 'inactive'})
    
    try:
        ticker = bot.session.latest_information_for_symbol(symbol="BTCUSDT")
        if ticker['ret_code'] == 0 and ticker['result']:
            data = ticker['result'][0]
            return jsonify({
                'price': float(data['last_price']),
                'change': float(data['price_24h_pcnt']),
                'volume': float(data['volume_24h']),
                'market_mode': 'Bullish' if float(data['price_24h_pcnt']) > 0 else 'Bearish'
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/chart-data')
def get_chart_data():
    """Get chart data for visualization."""
    global bot
    if not bot:
        return jsonify({'status': 'inactive'})
    
    try:
        symbol = "BTCUSDT"
        timeframe = request.args.get('timeframe', '1h')
        
        # Get historical data
        df = bot.get_historical_data(symbol)
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data available'})
        
        # Format data for chart
        chart_data = []
        for index, row in df.iterrows():
            chart_data.append({
                'time': index.timestamp(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume'])
            })
        
        return jsonify({
            'status': 'success',
            'data': chart_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/trade-history')
def get_trade_history():
    """Get completed trade history."""
    global bot
    if not bot:
        return jsonify({'trades': []})
    
    try:
        # Get trade history from bot (you'll need to implement this in the bot class)
        trades = bot.get_trade_history()
        return jsonify({'trades': trades})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/performance')
def get_performance():
    """Get performance metrics."""
    global bot
    if not bot:
        return jsonify({'status': 'inactive'})
    
    try:
        metrics = bot.get_performance_metrics()
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/risk')
def get_risk_metrics():
    """Get risk metrics."""
    global bot
    if not bot:
        return jsonify({'status': 'inactive'})
    
    try:
        metrics = bot.get_risk_metrics()
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/performance-charts')
def get_performance_charts():
    """Get data for performance charts."""
    global bot
    if not bot:
        return jsonify({'status': 'inactive'})
    
    try:
        # Get trade history
        trades = bot.trade_history
        
        # Calculate monthly performance
        monthly_perf = {}
        equity_curve = []
        cumulative_pnl = 0
        
        for trade in trades:
            # Monthly performance
            date = datetime.fromisoformat(trade['exit_time'])
            month_key = f"{date.year}-{date.month:02d}"
            monthly_perf[month_key] = monthly_perf.get(month_key, 0) + trade['pnl']
            
            # Equity curve
            cumulative_pnl += trade['pnl']
            equity_curve.append({
                'time': trade['exit_time'],
                'value': cumulative_pnl
            })
        
        # Calculate win/loss distribution
        win_loss_dist = {
            'wins': len([t for t in trades if t['pnl'] > 0]),
            'losses': len([t for t in trades if t['pnl'] <= 0])
        }
        
        return jsonify({
            'status': 'success',
            'monthly_performance': monthly_perf,
            'equity_curve': equity_curve,
            'win_loss_distribution': win_loss_dist
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/update-risk-settings', methods=['POST'])
def update_risk_settings():
    """Update risk management settings."""
    global bot
    if not bot:
        return jsonify({'status': 'error', 'message': 'Bot is not running'})
    
    try:
        settings = request.json
        bot.update_risk_settings(
            risk_per_trade=settings['risk_per_trade'],
            max_drawdown_limit=settings['max_drawdown_limit'],
            daily_loss_limit=settings['daily_loss_limit']
        )
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 