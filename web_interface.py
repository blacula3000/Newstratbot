from flask import Flask, render_template, jsonify, request
import threading
from trading_bot import TradingBot
from multi_timeframe_analyzer import MultiTimeframeAnalyzer
import logging

app = Flask(__name__)

# Global bot instance
bot = None
bot_thread = None
mtf_analyzer = MultiTimeframeAnalyzer()

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
    """Get recent pattern history including agent-detected patterns."""
    global bot
    
    if not bot:
        return jsonify({'patterns': [], 'agent_patterns': []})
    
    try:
        # Separate patterns by source
        basic_patterns = [p for p in bot.pattern_history if p.get('source') != 'agent']
        agent_patterns = [p for p in bot.pattern_history if p.get('source') == 'agent']
        
        return jsonify({
            'patterns': bot.pattern_history,
            'basic_patterns': basic_patterns,
            'agent_patterns': agent_patterns,
            'total_patterns': len(bot.pattern_history),
            'agent_enabled': bot.use_agent_patterns if hasattr(bot, 'use_agent_patterns') else False
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/agent/status')
def get_agent_status():
    """Get status of the quant-pattern-analyst agent."""
    global bot
    
    if not bot:
        return jsonify({'agent_active': False})
    
    try:
        if hasattr(bot, 'pattern_agent'):
            return jsonify({
                'agent_active': True,
                'agent_name': 'quant-pattern-analyst',
                'min_confidence': bot.pattern_agent.min_pattern_confidence,
                'cached_symbols': list(bot.pattern_agent.pattern_cache.keys()),
                'last_analysis': bot.pattern_agent.last_analysis_time.isoformat() if bot.pattern_agent.last_analysis_time else None
            })
        else:
            return jsonify({'agent_active': False})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/agent/analyze', methods=['POST'])
def analyze_with_agent():
    """Trigger agent analysis for a specific symbol."""
    global bot
    
    if not bot or not hasattr(bot, 'pattern_agent'):
        return jsonify({'status': 'error', 'message': 'Agent not available'})
    
    try:
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({'status': 'error', 'message': 'Symbol required'})
        
        # Get historical data for the symbol
        df = bot.get_historical_data(symbol)
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data available for symbol'})
        
        # Run agent analysis
        analysis = bot.pattern_agent.analyze_patterns(symbol, df, bot.timeframe)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/mtf')
def multi_timeframe():
    """Render the multi-timeframe analysis page."""
    return render_template('multi_timeframe.html')

@app.route('/api/mtf/analyze', methods=['POST'])
def analyze_multi_timeframe():
    """Analyze an asset across multiple timeframes."""
    try:
        data = request.json
        symbol = data.get('symbol')
        timeframes = data.get('timeframes', ['15m', '30m', '1h', '1d'])
        
        if not symbol:
            return jsonify({'status': 'error', 'message': 'Symbol required'})
        
        # Get multi-timeframe analysis
        analysis = mtf_analyzer.get_multi_timeframe_analysis(symbol, timeframes)
        
        return jsonify({
            'status': 'success',
            'data': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/mtf/confluence', methods=['POST'])
def get_timeframe_confluence():
    """Get confluence analysis across timeframes."""
    try:
        data = request.json
        symbol = data.get('symbol')
        timeframes = data.get('timeframes', ['15m', '30m', '1h', '1d'])
        
        if not symbol:
            return jsonify({'status': 'error', 'message': 'Symbol required'})
        
        # Get confluence analysis
        confluence = mtf_analyzer.get_timeframe_confluence(symbol, timeframes)
        
        return jsonify({
            'status': 'success',
            'data': confluence
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/mtf/batch', methods=['POST'])
def analyze_batch_mtf():
    """Analyze multiple symbols across timeframes."""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        timeframes = data.get('timeframes', ['15m', '30m', '1h', '1d'])
        
        if not symbols:
            return jsonify({'status': 'error', 'message': 'Symbols required'})
        
        results = {}
        for symbol in symbols[:10]:  # Limit to 10 symbols
            try:
                analysis = mtf_analyzer.get_multi_timeframe_analysis(symbol, timeframes)
                results[symbol] = analysis
            except Exception as e:
                results[symbol] = {'error': str(e)}
        
        return jsonify({
            'status': 'success',
            'data': results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 