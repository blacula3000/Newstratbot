import os
from dotenv import load_dotenv
from app.trading_bot import TradingBot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_connection():
    load_dotenv()
    
    config = {
        'api_key': os.getenv('API_KEY'),
        'api_secret': os.getenv('API_SECRET'),
        'base_url': os.getenv('BASE_URL'),
        'exchange': os.getenv('EXCHANGE')
    }
    
    try:
        bot = TradingBot(config=config)
        
        # Test market data fetching
        price = bot.get_current_price('BTCUSDT')
        print(f"\nCurrent BTC price: ${price}")
        
        # Test account connection
        wallet = bot.session.get_wallet_balance()
        if wallet['ret_code'] == 0:
            balance = wallet['result']['USDT']['available_balance']
            print(f"Wallet balance: {balance} USDT")
        
        print("\nConnection test successful!")
        return True
        
    except Exception as e:
        print(f"\nError testing connection: {e}")
        return False

if __name__ == "__main__":
    test_connection() 