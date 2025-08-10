"""
Crypto-Specific Configuration for Bybit STRAT Trading
Optimized settings for cryptocurrency trading with institutional controls
"""

import os
from datetime import time
from typing import List, Dict, Tuple

class CryptoConfig:
    """Configuration optimized for crypto trading on Bybit"""
    
    # ========== EXCHANGE SETTINGS ==========
    EXCHANGE = "BYBIT"
    BYBIT_TESTNET = True  # Set to False for live trading
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    
    # ========== TRADING MODE ==========
    TRADING_MODE = 'paper'  # 'paper' or 'live'
    
    # ========== CRYPTO SYMBOLS ==========
    # Primary crypto pairs to trade (will be converted to USDT pairs)
    ACTIVE_SYMBOLS = [
        'BTC',    # Bitcoin
        'ETH',    # Ethereum  
        'SOL',    # Solana
        'AVAX',   # Avalanche
        'MATIC',  # Polygon
        'DOT',    # Polkadot
        'ADA',    # Cardano
        'LINK',   # Chainlink
        'ATOM',   # Cosmos
        'NEAR'    # Near Protocol
    ]
    
    # High-priority symbols (get priority in signal processing)
    PRIORITY_SYMBOLS = ['BTC', 'ETH', 'SOL']
    
    # ========== STRAT SETTINGS ==========
    # Timeframes for STRAT analysis (Bybit supported intervals)
    STRAT_TIMEFRAMES = {
        'primary': '15',    # 15-minute primary timeframe
        'higher': '60',     # 1-hour higher timeframe
        'lower': '5'        # 5-minute lower timeframe
    }
    
    # STRAT pattern confidence thresholds
    MIN_STRAT_CONFIDENCE = 65      # Minimum confidence for signal generation
    HIGH_CONFIDENCE_THRESHOLD = 80  # High confidence signals
    
    # Timeframe Continuity (FTFC) settings
    MIN_FTFC_SCORE = 70           # Minimum FTFC alignment score
    FTFC_LOOKBACK_PERIODS = 20    # Periods to analyze for continuity
    
    # ========== POSITION SIZING (Crypto-Optimized) ==========
    MAX_PORTFOLIO_RISK = 0.15          # 15% of portfolio at risk
    MAX_POSITION_SIZE = 0.08           # 8% max per position
    DEFAULT_POSITION_SIZE = 0.05       # 5% default position size
    MAX_RISK_PER_TRADE = 0.02          # 2% max risk per trade
    
    # Volatility-based sizing
    HIGH_VOL_SIZE_MULTIPLIER = 0.7     # Reduce size by 30% for high volatility
    LOW_VOL_SIZE_MULTIPLIER = 1.2      # Increase size by 20% for low volatility
    VOLATILITY_THRESHOLD = 0.04        # 4% daily volatility threshold
    
    # ========== RISK MANAGEMENT (Crypto-Specific) ==========
    DAILY_LOSS_LIMIT = 3000           # $3000 daily loss limit
    MAX_CONSECUTIVE_LOSSES = 5         # Stop after 5 consecutive losses
    MAX_DRAWDOWN = 0.20               # 20% max drawdown
    COOLDOWN_AFTER_LOSS_LIMIT = 24    # Hours to wait after hitting daily limit
    
    # Position limits
    MAX_CONCURRENT_POSITIONS = 6       # Max 6 crypto positions
    MAX_CORRELATED_POSITIONS = 3       # Max 3 highly correlated positions
    
    # Leverage settings (for margin trading)
    MAX_LEVERAGE = 3                   # 3x maximum leverage
    DEFAULT_LEVERAGE = 1               # No leverage by default
    
    # ========== STOP LOSS & TAKE PROFIT ==========
    # ATR-based stops (Average True Range multipliers)
    STOP_LOSS_ATR_MULTIPLIER = 1.5    # 1.5x ATR for stop loss
    TAKE_PROFIT_ATR_MULTIPLIER = 2.5   # 2.5x ATR for take profit
    
    # Crypto-specific stop settings
    MIN_STOP_DISTANCE_PCT = 0.02       # Minimum 2% stop distance
    MAX_STOP_DISTANCE_PCT = 0.08       # Maximum 8% stop distance
    TRAILING_STOP_TRIGGER = 0.03       # Start trailing after 3% profit
    TRAILING_STOP_DISTANCE = 0.015     # 1.5% trailing stop distance
    
    # ========== EXECUTION SETTINGS ==========
    ORDER_TYPE = 'Market'              # Default to market orders for crypto
    TIME_IN_FORCE = 'IOC'             # Immediate or Cancel
    SLIPPAGE_TOLERANCE = 0.001         # 0.1% slippage tolerance
    EXECUTION_TIMEOUT = 30             # 30 seconds execution timeout
    
    # Order size limits
    MIN_ORDER_SIZE_USDT = 10          # Minimum $10 orders
    MAX_ORDER_SIZE_USDT = 50000       # Maximum $50k orders
    
    # ========== MARKET REGIME DETECTION ==========
    # Crypto market regimes
    REGIME_INDICATORS = {
        'trend_strength': 0.6,         # ADX equivalent threshold
        'volatility_expansion': 0.05,   # 5% volatility for expansion
        'volume_surge': 2.0,           # 2x average volume for surge
        'correlation_breakdown': 0.3    # Correlation threshold
    }
    
    # Regime-specific adjustments
    REGIME_ADJUSTMENTS = {
        'high_volatility': {
            'position_size_multiplier': 0.7,
            'stop_multiplier': 1.5,
            'min_confidence': 75
        },
        'low_volatility': {
            'position_size_multiplier': 1.2,
            'stop_multiplier': 0.8,
            'min_confidence': 60
        },
        'trend_following': {
            'position_size_multiplier': 1.1,
            'stop_multiplier': 1.2,
            'min_confidence': 65
        },
        'range_bound': {
            'position_size_multiplier': 0.8,
            'stop_multiplier': 0.9,
            'min_confidence': 70
        }
    }
    
    # ========== DATA QUALITY THRESHOLDS ==========
    MIN_DATA_QUALITY_SCORE = 75       # Minimum data quality to trade
    MAX_SPREAD_BPS = 15               # Maximum 15 basis points spread
    MIN_VOLUME_THRESHOLD = 1000       # Minimum volume for liquidity
    STALE_DATA_THRESHOLD = 60         # Seconds before data considered stale
    
    # ========== LIQUIDITY REQUIREMENTS ==========
    MIN_LIQUIDITY_SCORE = 65          # Minimum liquidity score
    MIN_ORDER_BOOK_DEPTH = 10000      # Minimum $10k order book depth
    MAX_MARKET_IMPACT = 0.005         # Maximum 0.5% market impact
    
    # ========== COMPLIANCE & LOGGING ==========
    LOG_TRADES = True                  # Enable trade logging
    LOG_SIGNALS = True                 # Enable signal logging
    SCREENSHOT_COMPLIANCE = False      # Disable screenshots for crypto
    JOURNAL_RETENTION_DAYS = 365      # 1 year retention
    
    # Compliance thresholds
    LARGE_ORDER_THRESHOLD = 25000     # $25k for large order alerts
    SUSPICIOUS_ACTIVITY_THRESHOLD = 10 # 10+ rapid trades
    
    # ========== PERFORMANCE MONITORING ==========
    PERFORMANCE_CHECK_INTERVAL = 300   # 5 minutes
    HEALTH_CHECK_INTERVAL = 60         # 1 minute
    ATTRIBUTION_WINDOW_DAYS = 7        # 1 week attribution analysis
    
    # Performance thresholds
    MIN_WIN_RATE = 0.35               # 35% minimum win rate
    MIN_PROFIT_FACTOR = 1.1           # 1.1 minimum profit factor
    MIN_SHARPE_RATIO = 0.5            # 0.5 minimum Sharpe ratio
    
    # ========== NOTIFICATION SETTINGS ==========
    ENABLE_NOTIFICATIONS = True        # Enable notifications
    NOTIFICATION_LEVELS = ['critical', 'high']  # Notification levels
    
    # Discord/Slack webhook for notifications (optional)
    WEBHOOK_URL = os.getenv('TRADING_WEBHOOK_URL', '')
    
    # ========== CRYPTO-SPECIFIC FEATURES ==========
    # Fear & Greed Index integration
    USE_FEAR_GREED_INDEX = True
    FEAR_GREED_THRESHOLD = 25         # Extreme fear threshold
    GREED_THRESHOLD = 75              # Extreme greed threshold
    
    # Funding rate analysis
    MONITOR_FUNDING_RATES = True
    HIGH_FUNDING_THRESHOLD = 0.01     # 1% funding rate threshold
    
    # On-chain metrics (if available)
    USE_ONCHAIN_METRICS = False       # Enable on-chain analysis
    WHALE_ALERT_THRESHOLD = 1000000   # $1M whale movement alert
    
    # ========== TIME-BASED SETTINGS ==========
    # Trading hours (24/7 for crypto, but can restrict if needed)
    TRADING_HOURS = {
        'enabled': False,              # Disable time restrictions for crypto
        'start_time': time(0, 0),     # 00:00 UTC
        'end_time': time(23, 59),     # 23:59 UTC
        'timezone': 'UTC'
    }
    
    # Weekend trading
    WEEKEND_TRADING = True            # Enable weekend trading
    
    # ========== BACKUP & RECOVERY ==========
    ENABLE_BACKUPS = True
    BACKUP_INTERVAL_HOURS = 6         # Every 6 hours
    MAX_BACKUP_FILES = 48             # Keep 12 days of backups
    
    # Emergency settings
    EMERGENCY_STOP_CONDITIONS = [
        'system_failure',
        'api_disconnection',
        'extreme_drawdown',
        'multiple_critical_alerts'
    ]
    
    # ========== ADVANCED FEATURES ==========
    # Multi-exchange arbitrage (future feature)
    ENABLE_ARBITRAGE = False
    ARBITRAGE_MIN_PROFIT = 0.002      # 0.2% minimum arbitrage profit
    
    # Options trading (if supported)
    ENABLE_OPTIONS = False
    MAX_OPTIONS_ALLOCATION = 0.1      # 10% max in options
    
    # DeFi integration (future feature)
    ENABLE_DEFI = False
    DEFI_PROTOCOLS = ['uniswap', 'aave', 'compound']

# Validation functions
def validate_crypto_config() -> bool:
    """Validate crypto configuration"""
    config = CryptoConfig()
    
    # Check required API credentials
    if config.TRADING_MODE == 'live':
        if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
            print("❌ Missing Bybit API credentials for live trading")
            return False
    
    # Validate risk parameters
    if config.MAX_PORTFOLIO_RISK > 0.3:
        print("⚠️ Warning: Portfolio risk exceeds 30%")
    
    if config.MAX_POSITION_SIZE > 0.1:
        print("⚠️ Warning: Position size exceeds 10%")
    
    # Validate symbols
    if not config.ACTIVE_SYMBOLS:
        print("❌ No active symbols configured")
        return False
    
    print("✅ Crypto configuration validated")
    return True

def get_crypto_symbol_config(symbol: str) -> Dict:
    """Get symbol-specific configuration"""
    config = CryptoConfig()
    
    # Base configuration
    symbol_config = {
        'symbol': f"{symbol}USDT",
        'min_order_size': config.MIN_ORDER_SIZE_USDT,
        'max_order_size': config.MAX_ORDER_SIZE_USDT,
        'position_size': config.DEFAULT_POSITION_SIZE,
        'stop_loss_atr': config.STOP_LOSS_ATR_MULTIPLIER,
        'take_profit_atr': config.TAKE_PROFIT_ATR_MULTIPLIER
    }
    
    # Symbol-specific adjustments
    symbol_adjustments = {
        'BTC': {
            'position_size': config.DEFAULT_POSITION_SIZE * 1.2,  # Larger BTC positions
            'min_confidence': 60
        },
        'ETH': {
            'position_size': config.DEFAULT_POSITION_SIZE * 1.1,
            'min_confidence': 62
        },
        'SOL': {
            'position_size': config.DEFAULT_POSITION_SIZE * 0.9,  # Smaller for alt coins
            'min_confidence': 70
        }
    }
    
    if symbol in symbol_adjustments:
        symbol_config.update(symbol_adjustments[symbol])
    
    return symbol_config

def get_bybit_connection_config() -> Dict:
    """Get Bybit connection configuration"""
    config = CryptoConfig()
    
    return {
        'api_key': config.BYBIT_API_KEY,
        'api_secret': config.BYBIT_API_SECRET,
        'testnet': config.BYBIT_TESTNET,
        'trading_mode': config.TRADING_MODE,
        'timeout': config.EXECUTION_TIMEOUT,
        'max_retries': 3,
        'retry_delay': 1
    }

if __name__ == "__main__":
    # Test configuration
    if validate_crypto_config():
        config = CryptoConfig()
        print(f"📊 Configured for {len(config.ACTIVE_SYMBOLS)} crypto symbols")
        print(f"🎯 Primary timeframe: {config.STRAT_TIMEFRAMES['primary']} minutes")
        print(f"💰 Max position size: {config.MAX_POSITION_SIZE * 100}%")
        print(f"🛡️ Daily loss limit: ${config.DAILY_LOSS_LIMIT:,}")
        print(f"⚙️ Mode: {config.TRADING_MODE}")
    else:
        print("❌ Configuration validation failed")