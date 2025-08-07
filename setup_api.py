"""
Secure API Key Setup for Bybit Trading
Interactive script to safely configure your API credentials
"""

import os
import getpass
from config import validate_setup, get_config

def setup_bybit_api():
    """Interactive setup for Bybit API credentials"""
    
    print("🔐 Bybit API Configuration Setup")
    print("=" * 50)
    print()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("📝 Found existing .env file")
        choice = input("Do you want to update it? (y/n): ").lower()
        if choice != 'y':
            print("Setup cancelled.")
            return
    else:
        print("📝 Creating new .env configuration file...")
    
    print("\n🔑 API Credentials Setup")
    print("⚠️  IMPORTANT SECURITY NOTES:")
    print("   • Never share your API keys with anyone")
    print("   • Keep your secret key absolutely private")
    print("   • Start with testnet for safety")
    print("   • Use API keys with trading permissions only")
    print()
    
    # Get API credentials
    api_key = input("Enter your Bybit API Key: ").strip()
    
    if not api_key:
        print("❌ API Key is required!")
        return
    
    print("Enter your Bybit API Secret (input will be hidden): ")
    api_secret = getpass.getpass().strip()
    
    if not api_secret:
        print("❌ API Secret is required!")
        return
    
    print("\n🧪 Trading Mode Configuration")
    print("1. Paper Trading (Safe - No real money)")
    print("2. Testnet (Safe - Fake money on real API)")
    print("3. Live Trading (⚠️  REAL MONEY - BE CAREFUL!)")
    
    mode_choice = input("Choose trading mode (1-3): ").strip()
    
    if mode_choice == "1":
        trading_mode = "paper"
        testnet = "true"
        print("✅ Selected: Paper Trading (Safe)")
    elif mode_choice == "2":
        trading_mode = "testnet"
        testnet = "true"
        print("✅ Selected: Testnet (Safe)")
    elif mode_choice == "3":
        trading_mode = "live"
        testnet = "false"
        print("⚠️  Selected: Live Trading (REAL MONEY!)")
        
        confirm = input("Are you SURE you want live trading? Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Live trading cancelled. Defaulting to testnet.")
            trading_mode = "testnet"
            testnet = "true"
    else:
        print("Invalid choice. Defaulting to paper trading.")
        trading_mode = "paper"
        testnet = "true"
    
    print("\n💰 Risk Management Configuration")
    
    # Position size
    position_size = input("Default position size (0.01): ").strip()
    if not position_size:
        position_size = "0.01"
    
    # Risk per trade
    risk_per_trade = input("Max risk per trade as decimal (0.02 = 2%): ").strip()
    if not risk_per_trade:
        risk_per_trade = "0.02"
    
    # Daily loss limit
    daily_loss_limit = input("Daily loss limit in USD (100): ").strip()
    if not daily_loss_limit:
        daily_loss_limit = "100"
    
    # Max positions
    max_positions = input("Maximum concurrent positions (5): ").strip()
    if not max_positions:
        max_positions = "5"
    
    # Create .env content
    env_content = f"""# Bybit API Configuration
BYBIT_API_KEY={api_key}
BYBIT_API_SECRET={api_secret}

# Trading Configuration
BYBIT_TESTNET={testnet}
TRADING_MODE={trading_mode}
DEFAULT_POSITION_SIZE={position_size}
MAX_RISK_PER_TRADE={risk_per_trade}
MAX_CONCURRENT_POSITIONS={max_positions}

# Risk Management
STOP_LOSS_PERCENTAGE=2.0
TAKE_PROFIT_RATIO=2.0
DAILY_LOSS_LIMIT={daily_loss_limit}
ACCOUNT_BALANCE_THRESHOLD=1000

# Alert Settings (Optional)
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_TRADES=true
"""
    
    # Write to .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ Configuration saved to .env file!")
        
        # Validate the configuration
        print("\n🔍 Validating configuration...")
        if validate_setup():
            print("✅ Configuration is valid!")
            
            # Test connection
            print("\n🔌 Testing connection...")
            test_connection()
            
        else:
            print("❌ Configuration validation failed!")
            
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")

def test_connection():
    """Test the Bybit connection with current configuration"""
    try:
        from bybit_trader import BybitTrader
        
        trader = BybitTrader()
        
        if trader.connect():
            print("✅ Successfully connected to Bybit!")
            
            # Get trading status
            status = trader.get_trading_status()
            print(f"\n📊 Trading Status:")
            print(f"   Mode: {status['trading_mode'].upper()}")
            print(f"   Balance: ${status['account_balance']:.2f}")
            print(f"   Positions: {status['active_positions']}/{status['max_positions']}")
            print(f"   Daily P&L: ${status['daily_pnl']:.2f}")
            
        else:
            print("❌ Connection failed!")
            print("Please check your API credentials and network connection.")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

def show_current_config():
    """Display current configuration (without sensitive data)"""
    try:
        config = get_config()
        safe_config = config.get_safe_config_display()
        
        print("📊 Current Configuration:")
        print("=" * 30)
        print(f"Trading Mode: {safe_config['trading_mode'].upper()}")
        print(f"Testnet: {'Yes' if safe_config['testnet_enabled'] else 'No'}")
        print(f"API Key: {'Configured' if safe_config['api_key_configured'] else 'Missing'}")
        print(f"API Secret: {'Configured' if safe_config['api_secret_configured'] else 'Missing'}")
        print(f"Position Size: {safe_config['default_position_size']}")
        print(f"Max Risk/Trade: {safe_config['max_risk_per_trade']*100}%")
        print(f"Max Positions: {safe_config['max_concurrent_positions']}")
        print(f"Daily Loss Limit: ${safe_config['daily_loss_limit']}")
        
    except Exception as e:
        print(f"Error reading configuration: {e}")

def main():
    """Main setup menu"""
    
    while True:
        print("\n🤖 STRAT Trading Bot - API Setup")
        print("=" * 40)
        print("1. Setup/Update Bybit API")
        print("2. Test Current Configuration")
        print("3. Show Current Configuration")
        print("4. Validate Setup")
        print("5. Exit")
        print()
        
        choice = input("Choose an option (1-5): ").strip()
        
        if choice == "1":
            setup_bybit_api()
        elif choice == "2":
            test_connection()
        elif choice == "3":
            show_current_config()
        elif choice == "4":
            validate_setup()
        elif choice == "5":
            print("👋 Setup complete!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user.")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        input("Press Enter to exit...")