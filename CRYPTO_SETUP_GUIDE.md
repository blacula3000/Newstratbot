# The Strat Crypto Paper Trading Setup Guide

## Overview
This is a complete paper trading system for cryptocurrency using The Strat methodology with TradingView integration.

## Features
- ✅ Real-time crypto price monitoring
- ✅ The Strat pattern detection (1-2-3 bar types)
- ✅ Multi-timeframe continuity analysis
- ✅ Paper trading with position management
- ✅ TradingView webhook integration
- ✅ Performance tracking and metrics
- ✅ Risk management (2% per trade, daily loss limits)
- ✅ Professional web dashboard

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python crypto_strat_app.py
```

### 3. Access Dashboard
Open your browser to: `http://localhost:5000`

## TradingView Integration

### 1. Add Pine Script Indicator
1. Open TradingView and go to Pine Editor
2. Copy the contents of `strat_indicator.pine`
3. Click "Add to Chart"
4. The indicator will show:
   - Bar type numbers (1, 2↑, 2↓, 3)
   - Pattern labels (2-2, 3-1-2, 1-2-2)
   - Timeframe continuity background

### 2. Create Alerts
1. Right-click on the chart → "Add Alert"
2. Condition: Choose your Strat indicator
3. Alert name: e.g., "BTC Strat Signal"
4. Message: Leave default (the indicator formats it)
5. Webhook URL: `http://your-ip:5000/webhook/tradingview`

### 3. For Local Testing (Recommended)
Use ngrok to expose your local server:
```bash
ngrok http 5000
```
Then use the ngrok URL in TradingView alerts.

## How It Works

### The Strat Methodology
- **Bar Type 1 (Inside)**: Consolidation - lower high, higher low
- **Bar Type 2 (Directional)**: Trending - breaks one side
- **Bar Type 3 (Outside)**: Expansion - higher high AND lower low

### Key Patterns
1. **3-1-2 Combo** (Strongest ⭐⭐⭐)
   - Outside bar → Inside bar → Directional break
   - Highest probability setup

2. **2-2 Reversal** (Strong ⭐⭐)
   - Two opposing directional bars
   - Indicates potential trend reversal

3. **1-2-2 Continuation** (Moderate ⭐)
   - Inside bar → Two directional bars same direction
   - Trend continuation pattern

### Paper Trading Rules
- **Position Sizing**: 2% risk per trade
- **Stop Loss**: 2% below entry (crypto volatility)
- **Take Profit**: 6% above entry (3:1 risk/reward)
- **Max Positions**: 3 concurrent trades
- **Daily Loss Limit**: 5% of account

## Trading Workflow

### Manual Trading
1. Select crypto pair (BTC-USD, ETH-USD, etc.)
2. Choose timeframe (5m recommended for day trading)
3. Click "START BOT"
4. Monitor for patterns and signals
5. Positions open automatically on high-confidence signals

### TradingView Integration
1. Set up alerts on multiple crypto pairs
2. Alerts trigger webhook to your bot
3. Bot evaluates signals and opens positions
4. Monitor performance on dashboard

## Dashboard Features

### Left Panel
- **Market Status**: Live price updates
- **Pattern Detection**: Current Strat patterns
- **Trade Setup**: Entry, stop loss, target levels

### Right Panel
- **Performance Metrics**: P&L, win rate, drawdown
- **Open Positions**: Active trades with unrealized P&L
- **Position Management**: Manual close option

## Risk Management

### Automated Features
- Stop loss on every trade
- Position sizing based on account risk
- Daily loss limits
- Maximum position limits

### Best Practices
1. Start with one crypto pair to learn patterns
2. Use 5m or 15m timeframes for clearer signals
3. Wait for high-confidence setups (70%+)
4. Don't override stop losses
5. Review trade history regularly

## Crypto-Specific Considerations

### Volatility
- Crypto moves faster than stocks
- Wider stops may be needed during high volatility
- Consider trading during active hours

### Recommended Pairs
- **BTC-USD**: Most liquid, cleaner patterns
- **ETH-USD**: Good for trend following
- **Major Alts**: SOL, BNB for more volatility

### Exchange Data
- Uses Yahoo Finance for real-time data
- TradingView provides more accurate data
- Consider exchange-specific nuances

## Performance Tracking

### Metrics Tracked
- Total P&L and percentage
- Win rate and profit factor
- Average win vs average loss
- Maximum drawdown
- Daily P&L

### Export Features
- Click "EXPORT" to save trade history
- JSON format for analysis
- Import to Excel for detailed review

## Troubleshooting

### Bot Not Starting
- Check Python dependencies
- Ensure port 5000 is available
- Verify symbol format (use -USD suffix)

### No Signals
- Patterns are rare - be patient
- Check timeframe continuity
- Ensure market is open

### Webhook Issues
- Verify webhook URL in alerts
- Check firewall settings
- Use ngrok for local testing

## Advanced Tips

### Optimize Entry
- Wait for pattern + timeframe alignment
- Higher timeframes = stronger signals
- Volume confirmation helps

### Position Management
- Scale out at multiple targets
- Move stops to breakeven
- Consider time-based exits

### Market Conditions
- Trending markets: Focus on 1-2-2
- Ranging markets: Focus on 2-2
- High volatility: Reduce position size

## Support
- Logs available in `tradingview_webhook.log`
- Export trades for analysis
- Monitor `strat_trading_bot.log` for signals

Remember: This is PAPER TRADING only. Test thoroughly before using real money!