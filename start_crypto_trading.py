"""
Crypto Trading Startup Script
Quick start for Bybit STRAT trading with agent system
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_config import CryptoConfig, validate_crypto_config
from crypto_agent_system import CryptoAgentSystem

def setup_logging():
    """Setup comprehensive logging"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Main logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(logs_dir / f"crypto_trading_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Trade-specific logger
    trade_logger = logging.getLogger('trades')
    trade_handler = logging.FileHandler(logs_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.log")
    trade_handler.setFormatter(logging.Formatter(log_format, date_format))
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)
    
    # Agent logger
    agent_logger = logging.getLogger('agents')
    agent_handler = logging.FileHandler(logs_dir / f"agents_{datetime.now().strftime('%Y%m%d')}.log")
    agent_handler.setFormatter(logging.Formatter(log_format, date_format))
    agent_logger.addHandler(agent_handler)
    agent_logger.setLevel(logging.DEBUG)

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          🚀 CRYPTO STRAT TRADING SYSTEM 🚀                  ║
    ║                                                              ║
    ║    Professional Algorithmic Trading with Institutional      ║
    ║    Grade Risk Management and Compliance Controls             ║
    ║                                                              ║
    ║    • STRAT Pattern Recognition                               ║
    ║    • Multi-Agent Decision Making                             ║
    ║    • Bybit Exchange Integration                              ║
    ║    • Real-time Risk Management                               ║
    ║    • Compliance Monitoring                                   ║
    ║    • Performance Attribution                                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'pybit', 'asyncio', 'logging'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ System requirements met")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print("⚙️ Setting up environment...")
    
    # Check for environment file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Found .env file")
        # Load environment variables from .env file
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    else:
        print("⚠️ No .env file found - using system environment variables")
    
    # Validate configuration
    if not validate_crypto_config():
        print("❌ Configuration validation failed")
        return False
    
    print("✅ Environment configured")
    return True

def get_trading_mode():
    """Get trading mode from user"""
    config = CryptoConfig()
    
    print(f"\n📋 Current trading mode: {config.TRADING_MODE}")
    
    if config.TRADING_MODE == 'live':
        print("⚠️  LIVE TRADING MODE - Real money will be used!")
        confirm = input("Are you sure you want to proceed with live trading? (yes/no): ").lower()
        if confirm != 'yes':
            print("Switching to paper trading mode for safety...")
            return 'paper'
        
        # Additional confirmation for live trading
        print("\n🚨 FINAL WARNING: This will place real trades with real money!")
        final_confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if final_confirm != 'CONFIRM LIVE TRADING':
            print("Live trading cancelled. Switching to paper mode.")
            return 'paper'
    
    return config.TRADING_MODE

def show_configuration():
    """Show current configuration"""
    config = CryptoConfig()
    
    print(f"""
📊 TRADING CONFIGURATION
========================
Exchange: {config.EXCHANGE}
Mode: {config.TRADING_MODE.upper()}
Testnet: {config.BYBIT_TESTNET}

Symbols: {', '.join(config.ACTIVE_SYMBOLS)}
Primary Timeframe: {config.STRAT_TIMEFRAMES['primary']} minutes
Max Positions: {config.MAX_CONCURRENT_POSITIONS}
Position Size: {config.DEFAULT_POSITION_SIZE * 100}%
Daily Loss Limit: ${config.DAILY_LOSS_LIMIT:,}

Risk Management:
- Max Portfolio Risk: {config.MAX_PORTFOLIO_RISK * 100}%
- Max Risk Per Trade: {config.MAX_RISK_PER_TRADE * 100}%
- Stop Loss ATR: {config.STOP_LOSS_ATR_MULTIPLIER}x
- Take Profit ATR: {config.TAKE_PROFIT_ATR_MULTIPLIER}x

Agent System:
- Data Quality Monitoring: ✅
- Market Regime Analysis: ✅
- Liquidity Assessment: ✅
- Risk Governance: ✅
- Order Health Monitoring: ✅
- Compliance Journaling: ✅
- Performance Attribution: ✅
    """)

async def main():
    """Main startup function"""
    
    # Print banner
    print_banner()
    
    # System checks
    if not check_requirements():
        print("❌ System check failed")
        return
    
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger('crypto_trading')
    logger.info("🚀 Crypto trading system starting up...")
    
    # Show configuration
    show_configuration()
    
    # Get trading mode confirmation
    trading_mode = get_trading_mode()
    
    # Create crypto-specific configuration
    crypto_config_overrides = {
        'TRADING_MODE': trading_mode,
        'LOG_TRADES': True,
        'LOG_SIGNALS': True,
    }
    
    try:
        # Initialize the crypto agent system
        logger.info("🤖 Initializing crypto agent system...")
        crypto_system = CryptoAgentSystem(crypto_config_overrides)
        
        print(f"\n🎯 Starting trading in {trading_mode.upper()} mode...")
        print("Press Ctrl+C to stop trading gracefully")
        
        # Start trading
        await crypto_system.start_trading()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
        print("\n⏹️ Graceful shutdown initiated...")
        
    except Exception as e:
        logger.error(f"💥 System error: {e}")
        print(f"\n❌ System error: {e}")
        print("Check logs for detailed error information")
    
    finally:
        logger.info("🏁 Crypto trading system shutdown complete")

if __name__ == "__main__":
    """
    Crypto Trading System Startup
    
    Usage:
    python start_crypto_trading.py
    
    Environment Variables:
    - BYBIT_API_KEY: Your Bybit API key
    - BYBIT_API_SECRET: Your Bybit API secret
    - TRADING_MODE: 'paper' or 'live'
    """
    
    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)