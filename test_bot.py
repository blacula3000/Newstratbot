from app.trading_bot import TradingBot
from dotenv import load_dotenv
import os

def test_bot():
    load_dotenv()
    
    config = {
        'api_key': os.getenv('API_KEY'),
        'api_secret': os.getenv('API_SECRET'),
        'base_url': os.getenv('BASE_URL'),
        'exchange': os.getenv('EXCHANGE')
    }
    
    bot = TradingBot(config=config)
    
    # Test basic functions
    symbol = 'BTCUSDT'
    
    # Test price fetching
    price = bot.get_current_price(symbol)
    print(f"Current {symbol} price: {price}")
    
    # Test historical data
    df = bot.get_historical_data(symbol)
    print(f"\nHistorical data shape: {df.shape}")
    print(df.tail())
    
    # Test technical analysis
    signals = bot.analyze_technical_indicators(df)
    print("\nTechnical signals:", signals)
    
    # Test position size calculation
    size = bot.calculate_position_size(symbol)
    print(f"\nCalculated position size: {size}")

if __name__ == "__main__":
    test_bot() 