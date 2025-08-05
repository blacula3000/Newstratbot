from flask import Flask, request, jsonify
import json
import hmac
import hashlib
from datetime import datetime
import logging
from crypto_paper_trading import CryptoPaperTrader, StratCryptoSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tradingview_webhook.log'),
        logging.StreamHandler()
    ]
)

class TradingViewWebhook:
    """Handles incoming webhooks from TradingView alerts"""
    
    def __init__(self, secret_key: str = "your-secret-key-here"):
        self.secret_key = secret_key
        self.paper_trader = CryptoPaperTrader(initial_balance=10000)
        self.signal_generator = StratCryptoSignalGenerator(self.paper_trader)
        
    def verify_webhook(self, data: bytes, signature: str) -> bool:
        """Verify webhook signature for security"""
        expected_signature = hmac.new(
            self.secret_key.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    def parse_alert(self, alert_data: Dict) -> Dict:
        """Parse TradingView alert format"""
        
        # Expected format from TradingView:
        # {
        #     "symbol": "BTCUSDT",
        #     "price": "45000",
        #     "action": "buy/sell",
        #     "pattern": "3_1_2_COMBO",
        #     "timeframe": "15",
        #     "exchange": "BINANCE"
        # }
        
        parsed_data = {
            'symbol': alert_data.get('symbol', '').upper(),
            'current_price': float(alert_data.get('price', 0)),
            'action': alert_data.get('action', '').lower(),
            'pattern': alert_data.get('pattern', ''),
            'timeframe': alert_data.get('timeframe', ''),
            'exchange': alert_data.get('exchange', 'BINANCE'),
            'timestamp': datetime.now()
        }
        
        # Convert action to direction
        if parsed_data['action'] == 'buy':
            parsed_data['direction'] = 'BULLISH'
        elif parsed_data['action'] == 'sell':
            parsed_data['direction'] = 'BEARISH'
        else:
            parsed_data['direction'] = None
        
        # Assign confidence based on pattern
        pattern_confidence = {
            '3_1_2_COMBO': 85,
            '2_2_REVERSAL': 75,
            '1_2_2_CONTINUATION': 70,
            'FTFC_BULLISH': 80,
            'FTFC_BEARISH': 80
        }
        
        parsed_data['confidence'] = pattern_confidence.get(parsed_data['pattern'], 65)
        
        return parsed_data
    
    def process_webhook(self, webhook_data: Dict) -> Dict:
        """Process incoming webhook and execute paper trade"""
        
        try:
            # Parse the alert
            signal_data = self.parse_alert(webhook_data)
            
            logging.info(f"Received signal: {signal_data}")
            
            # Check if we should act on this signal
            if not signal_data['direction']:
                return {'status': 'ignored', 'reason': 'No clear direction'}
            
            # Process the signal
            position = self.signal_generator.process_strat_signal(signal_data)
            
            if position:
                return {
                    'status': 'success',
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                }
            else:
                return {
                    'status': 'rejected',
                    'reason': 'Signal did not meet criteria or risk limits exceeded'
                }
                
        except Exception as e:
            logging.error(f"Error processing webhook: {e}")
            return {'status': 'error', 'message': str(e)}

# Flask app for webhook endpoint
app = Flask(__name__)
webhook_handler = TradingViewWebhook()

@app.route('/webhook/tradingview', methods=['POST'])
def tradingview_webhook():
    """Endpoint for TradingView webhooks"""
    
    try:
        # Get webhook data
        data = request.get_data()
        signature = request.headers.get('X-Webhook-Signature', '')
        
        # Verify signature (optional but recommended)
        # if not webhook_handler.verify_webhook(data, signature):
        #     return jsonify({'error': 'Invalid signature'}), 403
        
        # Parse JSON data
        webhook_data = request.get_json()
        
        # Process the webhook
        result = webhook_handler.process_webhook(webhook_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/positions', methods=['GET'])
def get_positions():
    """Get all open positions"""
    
    positions = webhook_handler.paper_trader.get_open_positions()
    return jsonify({
        'positions': [
            {
                'id': p.id,
                'symbol': p.symbol,
                'side': p.side,
                'entry_price': p.entry_price,
                'size': p.size,
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'entry_time': p.entry_time.isoformat()
            }
            for p in positions
        ]
    })

@app.route('/performance', methods=['GET'])
def get_performance():
    """Get trading performance metrics"""
    
    metrics = webhook_handler.paper_trader.get_performance_metrics()
    return jsonify(metrics)

@app.route('/close/<position_id>', methods=['POST'])
def close_position(position_id):
    """Manually close a position"""
    
    data = request.get_json()
    exit_price = float(data.get('price', 0))
    
    if exit_price <= 0:
        return jsonify({'error': 'Invalid exit price'}), 400
    
    trade = webhook_handler.paper_trader.close_position(
        position_id, 
        exit_price, 
        'manual'
    )
    
    if trade:
        return jsonify({
            'status': 'closed',
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent
        })
    else:
        return jsonify({'error': 'Position not found or already closed'}), 404

@app.route('/price_update', methods=['POST'])
def price_update():
    """Update current prices for stop loss/take profit monitoring"""
    
    prices = request.get_json()
    webhook_handler.paper_trader.check_stop_loss_take_profit(prices)
    return jsonify({'status': 'updated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)