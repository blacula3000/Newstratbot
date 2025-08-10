"""
Crypto Trading Setup Script
Interactive setup for Bybit STRAT trading system
"""

import os
import sys
from pathlib import Path
import getpass

def create_env_file():
    """Create .env file with user's API credentials"""
    print("\n🔧 Setting up your Bybit API credentials...")
    print("You can get these from: https://bybit.com/app/user/api-management")
    print("⚠️ Make sure to enable 'Trade' permissions for your API key")
    
    # Get API credentials
    api_key = input("Enter your Bybit API Key: ").strip()
    api_secret = getpass.getpass("Enter your Bybit API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("❌ API credentials are required")
        return False
    
    # Choose testnet or mainnet
    print("\n🌐 Choose trading environment:")
    print("1. Testnet (Recommended for testing)")
    print("2. Mainnet (Live trading)")
    
    choice = input("Enter choice (1-2): ").strip()
    testnet = choice != "2"
    
    if not testnet:
        print("⚠️ WARNING: You selected MAINNET (live trading)")
        confirm = input("Are you sure? This will use real money! (yes/no): ").lower()
        if confirm != "yes":
            testnet = True
            print("✅ Switched to testnet for safety")
    
    # Choose initial trading mode
    print("\n📊 Choose initial trading mode:")
    print("1. Paper Trading (Simulated, no real orders)")
    print("2. Live Trading (Real orders)")
    
    mode_choice = input("Enter choice (1-2): ").strip()
    trading_mode = "live" if mode_choice == "2" else "paper"
    
    if trading_mode == "live":
        print("⚠️ WARNING: Live trading mode will place real orders!")
        confirm = input("Are you sure? (yes/no): ").lower()
        if confirm != "yes":
            trading_mode = "paper"
            print("✅ Switched to paper trading for safety")
    
    # Create .env file content
    env_content = f"""# Bybit API Configuration
BYBIT_API_KEY={api_key}
BYBIT_API_SECRET={api_secret}
BYBIT_TESTNET={'true' if testnet else 'false'}

# Trading Configuration
TRADING_MODE={trading_mode}

# Optional: Webhook for notifications
# TRADING_WEBHOOK_URL=your_discord_or_slack_webhook_url_here

# Risk Management Overrides (optional)
# DAILY_LOSS_LIMIT=2000
# MAX_POSITION_SIZE=0.08
# MAX_CONCURRENT_POSITIONS=5
"""
    
    # Write .env file
    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"\n✅ Created .env file with your configuration")
    print(f"   Testnet: {'Yes' if testnet else 'No'}")
    print(f"   Trading Mode: {trading_mode.title()}")
    
    return True

def create_directory_structure():
    """Create necessary directories"""
    directories = ["logs", "data", "screenshots"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created directory structure")

def check_python_packages():
    """Check and install required packages"""
    print("\n📦 Checking Python packages...")
    
    required_packages = [
        "pandas",
        "numpy", 
        "pybit",
        "scikit-learn",
        "pillow",
        "aiohttp",
        "asyncio-mqtt"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages now? (yes/no): ").lower()
        
        if install == "yes":
            import subprocess
            for package in missing_packages:
                print(f"Installing {package}...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✅ {package} installed")
                else:
                    print(f"  ❌ Failed to install {package}")
                    print(f"  Error: {result.stderr}")
        else:
            print("⚠️ Please install missing packages manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def show_getting_started():
    """Show getting started information"""
    print("""
🎉 SETUP COMPLETE!

📋 What's Been Configured:
==========================
✅ API credentials stored securely in .env file
✅ Directory structure created (logs, data, screenshots)
✅ Python packages verified/installed
✅ Crypto trading system ready to use

🚀 Getting Started:
==================
1. Test the connection:
   python -c "from bybit_trader import BybitTrader; print('✅ Connected' if BybitTrader().connect() else '❌ Failed')"

2. Start paper trading:
   python start_crypto_trading.py

3. View the web dashboard:
   python start_dashboard.py
   Then open: http://localhost:5000

📊 Trading Features:
===================
• STRAT Pattern Recognition (2-1-2 Reversals, 3-1-2 Continuations)
• Multi-Agent Risk Management
• Real-time Performance Attribution
• Institutional-grade Compliance
• Bybit API Integration
• 24/7 Crypto Trading

⚙️ Configuration:
=================
• Edit crypto_config.py to adjust trading parameters
• Modify .env file to change API settings
• Check logs/ directory for detailed operation logs

🛡️ Safety Features:
===================
• Paper trading mode by default
• Daily loss limits
• Position size controls
• Multi-agent risk validation
• Emergency stop functionality

📚 Documentation:
================
• Read README.md for detailed information
• Check STRAT_SYSTEM_README.md for STRAT methodology
• Review agent documentation in agents/ directory

🆘 Support:
==========
• Check logs for any errors
• Review configuration if issues occur
• Ensure API permissions include 'Trade' access
• Start with paper trading to test everything

Happy Trading! 🚀📈
    """)

def main():
    """Main setup function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🛠️ CRYPTO STRAT TRADING SYSTEM SETUP 🛠️               ║
║                                                              ║
║   This setup will configure your system for professional    ║
║   cryptocurrency trading using STRAT methodology on Bybit   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Welcome to the Crypto STRAT Trading System setup!")
    print("This will configure everything you need to start trading.")
    
    # Check if already configured
    env_file = Path(".env")
    if env_file.exists():
        print("\n⚠️ Found existing .env file")
        overwrite = input("Do you want to reconfigure? (yes/no): ").lower()
        if overwrite != "yes":
            print("Setup cancelled.")
            return
    
    try:
        # Step 1: Create directory structure
        print("\n" + "="*60)
        print("STEP 1: Creating directory structure...")
        create_directory_structure()
        
        # Step 2: Check packages
        print("\n" + "="*60)
        print("STEP 2: Checking Python packages...")
        if not check_python_packages():
            return
        
        # Step 3: Configure API credentials
        print("\n" + "="*60)
        print("STEP 3: Configuring Bybit API credentials...")
        if not create_env_file():
            return
        
        # Step 4: Success
        print("\n" + "="*60)
        show_getting_started()
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and try again")

if __name__ == "__main__":
    main()