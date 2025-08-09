"""
Configuration Management for STRAT Trading Bot
Handles API keys, trading settings, and risk management parameters
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    """Central configuration management"""
    
    def __init__(self):
        self.load_config()
        
    def load_config(self):
        """Load all configuration settings"""
        
        # API Configuration
        self.BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
        self.BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
        
        # Trading Mode Configuration
        self.BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        self.TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # paper, testnet, live
        
        # Position and Risk Configuration
        self.DEFAULT_POSITION_SIZE = float(os.getenv('DEFAULT_POSITION_SIZE', '0.01'))
        self.MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))
        self.MAX_CONCURRENT_POSITIONS = int(os.getenv('MAX_CONCURRENT_POSITIONS', '5'))
        
        # Risk Management
        self.STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0'))
        self.TAKE_PROFIT_RATIO = float(os.getenv('TAKE_PROFIT_RATIO', '2.0'))
        self.DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', '100'))
        self.ACCOUNT_BALANCE_THRESHOLD = float(os.getenv('ACCOUNT_BALANCE_THRESHOLD', '1000'))
        
        # Alert Settings
        self.DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
        self.LOG_TRADES = os.getenv('LOG_TRADES', 'true').lower() == 'true'
        
        # Agent Configuration
        self.USE_PATTERN_AGENT = os.getenv('USE_PATTERN_AGENT', 'true').lower() == 'true'
        self.AGENT_MIN_CONFIDENCE = float(os.getenv('AGENT_MIN_CONFIDENCE', '0.75'))
        self.AGENT_MAX_PATTERNS = int(os.getenv('AGENT_MAX_PATTERNS', '10'))
        self.AGENT_CACHE_DURATION = int(os.getenv('AGENT_CACHE_DURATION', '900'))
        
        # Bybit API URLs
        if self.BYBIT_TESTNET:
            self.BYBIT_HTTP_URL = "https://api-testnet.bybit.com"
            self.BYBIT_WS_URL = "wss://stream-testnet.bybit.com"
        else:
            self.BYBIT_HTTP_URL = "https://api.bybit.com"
            self.BYBIT_WS_URL = "wss://stream.bybit.com"
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required API credentials for non-paper trading
        if self.TRADING_MODE != 'paper':
            if not self.BYBIT_API_KEY:
                validation_result['valid'] = False
                validation_result['errors'].append("BYBIT_API_KEY is required for live/testnet trading")
                
            if not self.BYBIT_API_SECRET:
                validation_result['valid'] = False
                validation_result['errors'].append("BYBIT_API_SECRET is required for live/testnet trading")
        
        # Validate risk parameters
        if self.MAX_RISK_PER_TRADE > 0.05:  # 5%
            validation_result['warnings'].append("MAX_RISK_PER_TRADE is above 5% - this is very risky")
            
        if self.DEFAULT_POSITION_SIZE > 1.0:
            validation_result['warnings'].append("DEFAULT_POSITION_SIZE is very large")
            
        # Check trading mode consistency
        if self.TRADING_MODE == 'live' and self.BYBIT_TESTNET:
            validation_result['errors'].append("Cannot use live trading with testnet URLs")
            validation_result['valid'] = False
            
        if self.TRADING_MODE == 'testnet' and not self.BYBIT_TESTNET:
            validation_result['errors'].append("Testnet mode requires BYBIT_TESTNET=true")
            validation_result['valid'] = False
            
        return validation_result
    
    def get_safe_config_display(self) -> Dict[str, Any]:
        """Get configuration for display (without sensitive data)"""
        return {
            'trading_mode': self.TRADING_MODE,
            'testnet_enabled': self.BYBIT_TESTNET,
            'api_key_configured': bool(self.BYBIT_API_KEY),
            'api_secret_configured': bool(self.BYBIT_API_SECRET),
            'default_position_size': self.DEFAULT_POSITION_SIZE,
            'max_risk_per_trade': self.MAX_RISK_PER_TRADE,
            'max_concurrent_positions': self.MAX_CONCURRENT_POSITIONS,
            'stop_loss_percentage': self.STOP_LOSS_PERCENTAGE,
            'take_profit_ratio': self.TAKE_PROFIT_RATIO,
            'daily_loss_limit': self.DAILY_LOSS_LIMIT,
            'alerts_configured': bool(self.DISCORD_WEBHOOK_URL or self.TELEGRAM_BOT_TOKEN),
            'http_url': self.BYBIT_HTTP_URL,
            'pattern_agent_enabled': self.USE_PATTERN_AGENT,
            'agent_min_confidence': self.AGENT_MIN_CONFIDENCE,
            'agent_max_patterns': self.AGENT_MAX_PATTERNS
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        
        # Create logs directory if it doesn't exist
        if self.LOG_TO_FILE:
            os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if self.LOG_TO_FILE:
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[
                    logging.FileHandler('logs/trading_bot.log'),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=log_level, format=log_format)
        
        # Create specialized loggers
        self.trade_logger = logging.getLogger('trades')
        if self.LOG_TRADES and self.LOG_TO_FILE:
            trade_handler = logging.FileHandler('logs/trades.log')
            trade_handler.setFormatter(logging.Formatter(log_format))
            self.trade_logger.addHandler(trade_handler)
            
        return logging.getLogger('trading_bot')

# Global config instance
config = TradingConfig()

def get_config() -> TradingConfig:
    """Get the global configuration instance"""
    return config

def create_env_file():
    """Create a .env file from the example if it doesn't exist"""
    if not os.path.exists('.env'):
        print("🔧 Creating .env file from example...")
        try:
            with open('.env.example', 'r') as example:
                with open('.env', 'w') as env_file:
                    env_file.write(example.read())
            print("✅ .env file created! Please edit it with your API credentials.")
            print("⚠️  Remember to set BYBIT_TESTNET=true for safe testing!")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    return True

def validate_setup():
    """Validate the current setup and display status"""
    print("🔍 Validating Trading Bot Configuration...")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        if create_env_file():
            print("📝 Please edit the .env file with your API credentials.")
        return False
    
    # Load and validate config
    config = get_config()
    validation = config.validate_config()
    
    # Display safe configuration
    safe_config = config.get_safe_config_display()
    
    print("📊 Current Configuration:")
    print(f"  Trading Mode: {safe_config['trading_mode'].upper()}")
    print(f"  Testnet: {'Enabled' if safe_config['testnet_enabled'] else 'Disabled'}")
    print(f"  API Key: {'✅ Configured' if safe_config['api_key_configured'] else '❌ Missing'}")
    print(f"  API Secret: {'✅ Configured' if safe_config['api_secret_configured'] else '❌ Missing'}")
    print(f"  Position Size: {safe_config['default_position_size']}")
    print(f"  Max Risk/Trade: {safe_config['max_risk_per_trade']*100:.1f}%")
    print(f"  Max Positions: {safe_config['max_concurrent_positions']}")
    print(f"  Daily Loss Limit: ${safe_config['daily_loss_limit']}")
    print(f"\n🤖 Agent Configuration:")
    print(f"  Pattern Agent: {'✅ Enabled' if safe_config['pattern_agent_enabled'] else '❌ Disabled'}")
    print(f"  Min Confidence: {safe_config['agent_min_confidence']*100:.0f}%")
    print(f"  Max Patterns: {safe_config['agent_max_patterns']}")
    
    # Show validation results
    print("\n🔍 Validation Results:")
    if validation['valid']:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    return validation['valid']

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_setup()