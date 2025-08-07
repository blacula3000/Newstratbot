# 🚀 Bybit API Integration Setup Guide

## Overview
This guide will help you securely configure Bybit API integration for automated STRAT trading.

## ⚠️ IMPORTANT SECURITY WARNINGS

### Before You Start:
- **NEVER share your API keys with anyone**
- **Always start with testnet or paper trading**
- **Use API keys with minimal required permissions**
- **Monitor your trades closely, especially at first**
- **Set strict risk limits**

## 📋 Prerequisites

### 1. Bybit Account Setup
1. Create a Bybit account at https://bybit.com
2. Complete KYC verification if planning to use live trading
3. Enable 2FA (Two-Factor Authentication)

### 2. Create API Keys
1. Log in to Bybit
2. Go to **Account & Security** → **API Management**
3. Click **Create New Key**
4. **For Testing (Recommended):**
   - Select **Testnet** environment
   - Permissions: `Trade`, `Read`
   - IP Restriction: Add your IP for security
5. **For Live Trading (Advanced):**
   - Select **Mainnet** environment
   - Permissions: `Trade`, `Read` (NO Withdrawal!)
   - IP Restriction: **REQUIRED**

### 3. Install Dependencies
```bash
cd Newstratbot
uv add python-dotenv schedule
```

## 🔧 Configuration Setup

### Option 1: Interactive Setup (Recommended)
```bash
python setup_api.py
```

This will guide you through:
- API key configuration
- Trading mode selection
- Risk management settings

### Option 2: Manual Setup
1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your details:
   ```env
   # Bybit API Credentials
   BYBIT_API_KEY=your_api_key_here
   BYBIT_API_SECRET=your_api_secret_here

   # Trading Configuration
   BYBIT_TESTNET=true  # Start with true!
   TRADING_MODE=paper  # paper/testnet/live
   DEFAULT_POSITION_SIZE=0.01
   MAX_RISK_PER_TRADE=0.02  # 2% max risk

   # Risk Management
   DAILY_LOSS_LIMIT=100
   MAX_CONCURRENT_POSITIONS=5
   ```

## 🧪 Testing Your Setup

### 1. Validate Configuration
```bash
python config.py
```

Expected output:
```
✅ Configuration is valid!
📊 Current Configuration:
  Trading Mode: PAPER
  Testnet: Enabled
  API Key: ✅ Configured
  API Secret: ✅ Configured
```

### 2. Test Bybit Connection
```bash
python bybit_trader.py
```

Expected output:
```
✅ Trader connected successfully
Trading Status: {'trading_enabled': True, 'account_balance': 10000.0, ...}
```

### 3. Test Signal Engine
```bash
python test_strat_signals.py
```

This will test:
- STRAT pattern recognition
- Trigger level detection
- FTFC analysis
- Signal generation

### 4. Run Paper Trading Test
```bash
python automated_trader.py
```

## 🎛️ Trading Modes Explained

### 1. Paper Trading (Safest)
- **Mode**: `TRADING_MODE=paper`
- **Risk**: None - purely simulated
- **Purpose**: Learn and test strategies
- **Recommended for**: Beginners, strategy testing

### 2. Testnet Trading
- **Mode**: `TRADING_MODE=testnet`, `BYBIT_TESTNET=true`
- **Risk**: None - uses fake money
- **Purpose**: Test API integration with real Bybit testnet
- **Recommended for**: API testing, final validation

### 3. Live Trading ⚠️
- **Mode**: `TRADING_MODE=live`, `BYBIT_TESTNET=false`
- **Risk**: Real money
- **Purpose**: Actual trading
- **Recommended for**: Experienced users only

## 📊 Dashboard Access

### Signal Dashboard
```bash
python advanced_web_interface.py
```
Access: `http://localhost:5000/signals`

Features:
- Real-time signal detection
- Live position monitoring
- Trading controls
- Performance tracking

## 🛡️ Risk Management Settings

### Essential Settings
```env
# Maximum risk per single trade (2% recommended)
MAX_RISK_PER_TRADE=0.02

# Daily loss limit in USD
DAILY_LOSS_LIMIT=100

# Maximum concurrent positions
MAX_CONCURRENT_POSITIONS=5

# Minimum account balance to continue trading
ACCOUNT_BALANCE_THRESHOLD=1000
```

### Position Sizing
The system automatically calculates position sizes based on:
- Account balance
- Risk per trade setting
- Stop loss distance
- Maximum position size limits

## 🔄 Automated Trading

### Start Automated System
```bash
python automated_trader.py
```

The system will:
1. Scan for STRAT signals every 5 minutes
2. Execute trades based on signal criteria
3. Monitor positions and risk limits
4. Log all activities

### Signal Criteria
Trades are only executed when:
- ✅ Valid STRAT pattern (2-1-2, 3-1-2, etc.)
- ✅ Trigger level break confirmed
- ✅ FTFC score ≥70%
- ✅ Confidence score ≥70%
- ✅ Risk-reward ratio ≥1:2
- ✅ Within risk limits

## 📈 Monitoring Your Bot

### Log Files
- `logs/trading_bot.log` - General system logs
- `logs/trades.log` - Trade execution logs

### Real-time Status
Check status anytime:
```bash
# View current positions
python -c "from bybit_trader import BybitTrader; t=BybitTrader(); print(t.get_trading_status())"
```

### Dashboard Monitoring
Access the web dashboard at `http://localhost:5000/signals` for:
- Live signal detection
- Position monitoring
- Performance metrics
- Trade history

## 🚨 Emergency Procedures

### Emergency Stop
If you need to immediately stop all trading:
```python
from automated_trader import AutomatedSTRATTrader
system = AutomatedSTRATTrader()
system.emergency_stop()
```

### Manual Position Closure
```python
from bybit_trader import BybitTrader
trader = BybitTrader()
trader.close_position('BTCUSDT')  # Close specific position
```

## 🔍 Troubleshooting

### Common Issues

#### "Connection Failed"
- Check internet connection
- Verify API keys are correct
- Ensure IP is whitelisted (if set)
- Check if testnet/mainnet setting matches API keys

#### "Invalid API Key"
- Verify API key and secret are correctly copied
- Check if keys are from correct environment (testnet/mainnet)
- Ensure API has required permissions

#### "Position Size Too Small"
- Increase `DEFAULT_POSITION_SIZE` setting
- Check account balance
- Verify risk settings aren't too conservative

#### "No Signals Found"
- Market may not have suitable STRAT setups
- Check if confidence thresholds are too high
- Verify symbols in watchlist are active

### Debug Mode
Enable detailed logging:
```env
LOG_LEVEL=DEBUG
```

## 📞 Support

### Self-Help
1. Check the troubleshooting section above
2. Review log files in `logs/` directory
3. Validate configuration with `python config.py`
4. Test individual components

### Resources
- Bybit API Documentation: https://bybit-exchange.github.io/docs/
- STRAT Methodology: See `AGENT_ARCHITECTURE.md`
- Bot Features: See `ADVANCED_FEATURES.md`

## 🎯 Next Steps

1. **Start with Paper Trading** - Get familiar with the system
2. **Test on Testnet** - Validate API integration
3. **Start Small on Live** - Use minimal position sizes
4. **Monitor Closely** - Watch performance and adjust settings
5. **Scale Gradually** - Increase sizes as confidence grows

## 🔐 Security Checklist

- [ ] API keys stored securely in `.env` file
- [ ] `.env` file added to `.gitignore`
- [ ] IP restrictions enabled on API keys
- [ ] 2FA enabled on Bybit account
- [ ] Started with testnet/paper trading
- [ ] Risk limits set appropriately
- [ ] Emergency stop procedures understood
- [ ] Regular monitoring plan in place

---

**Remember: Trading involves risk. Never trade with money you can't afford to lose. Always test thoroughly before using real funds.**