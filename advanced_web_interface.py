"""
Advanced Web Interface for STRAT Trading Bot
Includes watchlist management, sector analysis, and daily STRAT performance tracking
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
from collections import defaultdict
from strat_signal_engine import StratSignalEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize STRAT signal engine
signal_engine = StratSignalEngine()

# Global data storage
class MarketData:
    def __init__(self):
        self.watchlist = []
        self.symbol_data = {}
        self.sector_data = {}
        self.daily_strat_results = {}
        self.market_sentiment = 50
        self.active_patterns = defaultdict(list)
        self.connected_clients = set()
        
market_data = MarketData()

# Sector ETF mapping for sector analysis
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Finance': 'XLF',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication': 'XLC'
}

class STRATAnalyzer:
    """Analyzes candlestick patterns using STRAT methodology"""
    
    @staticmethod
    def identify_candle_type(open_price, high, low, close):
        """Identify if candle is Type 1 (Inside), 2 (Directional), or 3 (Outside)"""
        # This is simplified - actual implementation would compare with previous candle
        if abs(close - open_price) < (high - low) * 0.3:
            return 1  # Inside bar
        elif close > open_price:
            return 2  # Directional up
        elif close < open_price:
            return 2  # Directional down
        else:
            return 3  # Outside bar
            
    @staticmethod
    def identify_pattern(candles):
        """Identify STRAT patterns in candle sequence"""
        if len(candles) < 3:
            return None
            
        patterns = []
        
        # Check for 2-1-2 reversal
        if len(candles) >= 3:
            last_three = candles[-3:]
            types = [STRATAnalyzer.identify_candle_type(c['Open'], c['High'], c['Low'], c['Close']) 
                    for c in last_three]
            
            if types == [2, 1, 2]:
                patterns.append({
                    'type': '2-1-2 Reversal',
                    'confidence': 85,
                    'direction': 'Bullish' if last_three[-1]['Close'] > last_three[-1]['Open'] else 'Bearish'
                })
                
        # Check for 3-1-2 pattern
        if len(candles) >= 3:
            last_three = candles[-3:]
            types = [STRATAnalyzer.identify_candle_type(c['Open'], c['High'], c['Low'], c['Close']) 
                    for c in last_three]
            
            if types == [3, 1, 2]:
                patterns.append({
                    'type': '3-1-2 Combo',
                    'confidence': 78,
                    'direction': 'Bullish' if last_three[-1]['Close'] > last_three[-1]['Open'] else 'Bearish'
                })
                
        return patterns

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('advanced_dashboard.html')

@app.route('/signals')
def signals_dashboard():
    """Serve the STRAT signals dashboard"""
    return render_template('strat_signals_dashboard.html')

@app.route('/api/watchlist', methods=['GET', 'POST'])
def manage_watchlist():
    """Manage watchlist symbols"""
    if request.method == 'POST':
        data = request.json
        symbol = data.get('symbol', '').upper()
        
        if symbol and symbol not in market_data.watchlist:
            market_data.watchlist.append(symbol)
            # Start monitoring the new symbol
            threading.Thread(target=monitor_symbol, args=(symbol,), daemon=True).start()
            return jsonify({'status': 'success', 'symbol': symbol})
        return jsonify({'status': 'error', 'message': 'Symbol already in watchlist or invalid'})
    
    return jsonify({'watchlist': market_data.watchlist})

@app.route('/api/sector-analysis')
def sector_analysis():
    """Analyze sector performance"""
    sector_performance = {}
    
    for sector, etf in SECTOR_ETFS.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period='1d', interval='1d')
            if not hist.empty:
                change = ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                sector_performance[sector] = {
                    'change': round(change, 2),
                    'bullish': change > 0,
                    'strength': min(100, abs(change) * 20)  # Convert to 0-100 scale
                }
        except:
            sector_performance[sector] = {'change': 0, 'bullish': False, 'strength': 50}
    
    # Calculate overall market sentiment
    bullish_sectors = sum(1 for s in sector_performance.values() if s['bullish'])
    market_sentiment = (bullish_sectors / len(sector_performance)) * 100
    
    return jsonify({
        'sectors': sector_performance,
        'market_sentiment': round(market_sentiment)
    })

@app.route('/api/daily-strat-results')
def daily_strat_results():
    """Get daily STRAT analysis results with proper actionable signals"""
    results = {}
    
    for symbol in market_data.watchlist[:15]:  # Limit to first 15 symbols for performance
        try:
            # Use the proper STRAT signal engine
            signal_analysis = signal_engine.identify_actionable_signal(symbol, '15m')
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='15m')
            
            if not hist.empty:
                daily_change = ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                
                results[symbol] = {
                    'has_signal': signal_analysis['has_signal'],
                    'signal_type': signal_analysis.get('signal_type', 'No Signal'),
                    'direction': signal_analysis.get('direction', 'None'),
                    'confidence_score': signal_analysis['confidence_score'],
                    'pattern_sequence': ' -> '.join(signal_analysis['pattern_sequence']) if signal_analysis['pattern_sequence'] else 'No Pattern',
                    'ftfc_score': signal_analysis.get('ftfc_analysis', {}).get('continuity_score', 0),
                    'trigger_broken': signal_analysis.get('trigger_info', {}).get('trigger_broken', False),
                    'daily_change': round(daily_change, 2),
                    'close_price': round(hist['Close'].iloc[-1], 2),
                    'volume': int(hist['Volume'].sum()),
                    'high': round(hist['High'].max(), 2),
                    'low': round(hist['Low'].min(), 2),
                    'entry_price': signal_analysis.get('entry_price', 0),
                    'stop_loss': signal_analysis.get('stop_loss', 0),
                    'target': signal_analysis.get('target', 0)
                }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            results[symbol] = {
                'has_signal': False,
                'signal_type': 'Error',
                'direction': 'None',
                'confidence_score': 0,
                'pattern_sequence': 'Error',
                'ftfc_score': 0,
                'trigger_broken': False,
                'daily_change': 0,
                'close_price': 0,
                'volume': 0,
                'high': 0,
                'low': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'target': 0
            }
    
    return jsonify(results)

@app.route('/api/actionable-signals')
def actionable_signals():
    """Get current actionable STRAT signals from watchlist"""
    try:
        # Scan watchlist for actionable signals
        signals = signal_engine.scan_multiple_symbols(market_data.watchlist[:10], '15m')
        
        # Format for frontend
        formatted_signals = []
        for signal in signals:
            if signal['has_signal'] and signal['confidence_score'] >= 70:
                formatted_signals.append({
                    'symbol': signal['symbol'],
                    'signal_type': signal['signal_type'],
                    'direction': signal['direction'],
                    'confidence_score': signal['confidence_score'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'target': signal['target'],
                    'risk_reward': round((signal['target'] - signal['entry_price']) / (signal['entry_price'] - signal['stop_loss']), 2) if signal['direction'] == 'LONG' else round((signal['entry_price'] - signal['target']) / (signal['stop_loss'] - signal['entry_price']), 2),
                    'ftfc_score': signal['ftfc_analysis'].get('continuity_score', 0),
                    'pattern_sequence': signal['pattern_sequence'],
                    'timestamp': signal['timestamp'].isoformat()
                })
        
        return jsonify({
            'signals': formatted_signals,
            'total_signals': len(formatted_signals),
            'scan_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbol/<symbol>/strat-analysis')
def symbol_strat_analysis(symbol):
    """Get detailed STRAT analysis for a specific symbol"""
    try:
        ticker = yf.Ticker(symbol.upper())
        
        # Get multiple timeframes
        timeframes = {
            '5m': ticker.history(period='1d', interval='5m'),
            '15m': ticker.history(period='5d', interval='15m'),
            '1h': ticker.history(period='1mo', interval='1h'),
            '1d': ticker.history(period='3mo', interval='1d')
        }
        
        analysis = {}
        
        for tf, data in timeframes.items():
            if not data.empty:
                candles = data.reset_index().tail(10).to_dict('records')
                patterns = STRATAnalyzer.identify_pattern(candles)
                
                analysis[tf] = {
                    'patterns': patterns if patterns else [],
                    'last_candle_type': STRATAnalyzer.identify_candle_type(
                        data['Open'].iloc[-1],
                        data['High'].iloc[-1],
                        data['Low'].iloc[-1],
                        data['Close'].iloc[-1]
                    ),
                    'trend': 'Bullish' if data['Close'].iloc[-1] > data['Close'].iloc[-5] else 'Bearish'
                }
        
        # Calculate confluence score
        bullish_signals = sum(1 for tf in analysis.values() if tf['trend'] == 'Bullish')
        confluence_score = (bullish_signals / len(analysis)) * 100
        
        return jsonify({
            'symbol': symbol.upper(),
            'timeframe_analysis': analysis,
            'confluence_score': round(confluence_score),
            'recommendation': 'Buy' if confluence_score > 70 else 'Hold' if confluence_score > 30 else 'Sell'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def monitor_symbol(symbol):
    """Monitor a symbol for real-time updates"""
    while symbol in market_data.watchlist:
        try:
            ticker = yf.Ticker(symbol)
            current = ticker.history(period='1d', interval='1m').tail(1)
            
            if not current.empty:
                price_data = {
                    'symbol': symbol,
                    'price': round(current['Close'].iloc[-1], 2),
                    'volume': int(current['Volume'].iloc[-1]),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store data
                market_data.symbol_data[symbol] = price_data
                
                # Emit to connected clients
                socketio.emit('price_update', price_data)
                
                # Check for patterns
                hist = ticker.history(period='1d', interval='5m')
                if len(hist) > 3:
                    candles = hist.reset_index().tail(5).to_dict('records')
                    patterns = STRATAnalyzer.identify_pattern(candles)
                    
                    if patterns:
                        pattern_data = {
                            'symbol': symbol,
                            'patterns': patterns,
                            'timestamp': datetime.now().isoformat()
                        }
                        market_data.active_patterns[symbol] = patterns
                        socketio.emit('pattern_detected', pattern_data)
                        
        except Exception as e:
            print(f"Error monitoring {symbol}: {e}")
        
        time.sleep(60)  # Update every minute

def update_market_sentiment():
    """Update overall market sentiment periodically"""
    while True:
        try:
            # Get sector data
            sector_performance = {}
            for sector, etf in list(SECTOR_ETFS.items())[:5]:  # Sample first 5 sectors
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period='1d', interval='1d')
                    if not hist.empty:
                        change = ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                        sector_performance[sector] = change
                except:
                    sector_performance[sector] = 0
            
            # Calculate sentiment
            avg_change = np.mean(list(sector_performance.values()))
            # Convert to 0-100 scale (assuming ±5% is max daily change)
            sentiment = 50 + (avg_change * 10)
            sentiment = max(0, min(100, sentiment))
            
            market_data.market_sentiment = round(sentiment)
            
            # Emit update
            socketio.emit('market_sentiment', {'value': market_data.market_sentiment})
            
        except Exception as e:
            print(f"Error updating market sentiment: {e}")
        
        time.sleep(300)  # Update every 5 minutes

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    market_data.connected_clients.add(request.sid)
    
    # Send initial data
    emit('initial_data', {
        'watchlist': market_data.watchlist,
        'market_sentiment': market_data.market_sentiment
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")
    market_data.connected_clients.discard(request.sid)

@socketio.on('subscribe_symbol')
def handle_subscribe(data):
    """Subscribe to symbol updates"""
    symbol = data.get('symbol', '').upper()
    if symbol and symbol not in market_data.watchlist:
        market_data.watchlist.append(symbol)
        threading.Thread(target=monitor_symbol, args=(symbol,), daemon=True).start()
        emit('subscription_confirmed', {'symbol': symbol})

@socketio.on('load_symbol')
def handle_load_symbol(data):
    """Load detailed data for a symbol"""
    symbol = data.get('symbol', '').upper()
    timeframe = data.get('timeframe', '15m')
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Map timeframe to period
        period_map = {
            '1m': '1d', '5m': '5d', '15m': '5d',
            '1h': '1mo', '4h': '3mo', '1d': '1y'
        }
        
        period = period_map.get(timeframe, '5d')
        hist = ticker.history(period=period, interval=timeframe)
        
        if not hist.empty:
            chart_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': hist.reset_index().to_dict('records'),
                'current_price': round(hist['Close'].iloc[-1], 2)
            }
            emit('symbol_data', chart_data)
            
    except Exception as e:
        emit('error', {'message': f"Error loading {symbol}: {str(e)}"})

def start_background_tasks():
    """Start background monitoring tasks"""
    # Start market sentiment updater
    sentiment_thread = threading.Thread(target=update_market_sentiment, daemon=True)
    sentiment_thread.start()
    
    # Load default watchlist
    default_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    for symbol in default_symbols:
        if symbol not in market_data.watchlist:
            market_data.watchlist.append(symbol)
            threading.Thread(target=monitor_symbol, args=(symbol,), daemon=True).start()

if __name__ == '__main__':
    print("Starting Advanced STRAT Trading Bot Dashboard...")
    print("Access the dashboard at: http://localhost:5000")
    
    # Start background tasks
    start_background_tasks()
    
    # Run the Flask-SocketIO app
    socketio.run(app, debug=True, port=5000)