# Local Setup & Running Guide

## Quick Start (Recommended)

### 1. Install Dependencies
```bash
pip install pandas numpy yfinance flask ccxt
```

### 2. Run the Demo
```bash
python quick_start.py
```

This will demonstrate all the enhanced agents working together with synthetic data.

## What You'll See

The demo showcases three core agents:

### 🎯 Trigger Line Agent
- Identifies key support/resistance levels  
- Detects STRAT 2u/2d/3 breakouts
- Analyzes momentum and confidence scores
- **Demo Output**: 20 active trigger lines, momentum analysis

### 📊 Volatility Agent  
- Calculates realized and GARCH volatility
- Determines volatility regime (low/normal/high)
- Analyzes volatility trends and IV rank
- **Demo Output**: 29.61% realized vol, normal regime

### 💰 Position Sizing Agent
- Dynamic position sizing with volatility adjustment
- Kelly Criterion and risk parity methods
- Risk level classification and constraints
- **Demo Output**: 0.0262 BTC position ($1,077 USD value)

## Sample Output
```
TRIGGER LINE AGENT ANALYSIS
------------------------------------------
Active Trigger Lines: 20
Directional Bias: NO_BREAK
Confidence Score: 59.53
Momentum Score: 40.00

VOLATILITY AGENT ANALYSIS  
------------------------------------------
Realized Volatility: 29.61%
GARCH Volatility: 18.25%
Volatility Regime: NORMAL
IV Rank: 50.00

POSITION SIZING AGENT ANALYSIS
------------------------------------------
Base Position Size: 0.0300
Adjusted Position Size: 0.0262
USD Value: $1,077.10
Expected Max Loss: $0.02
Sizing Method: volatility_target
```

## Full System Options

### Option 1: Advanced Demo
```bash
python run_local_agents.py
```
Then select:
- **Demo Mode**: Comprehensive analysis of all 10 agents
- **Monitor Mode**: Continuous monitoring simulation

### Option 2: Windows Shortcut
```bash
start_local.bat
```
Double-click to run the demo directly.

## Live Trading Setup

⚠️ **For actual trading (advanced users only)**:

1. Get API credentials from your exchange (Binance, Bybit, etc.)
2. Create `.env` file with your keys
3. Start with testnet/paper trading
4. Use small position sizes initially
5. Review all documentation thoroughly

## Architecture

This system implements 10 specialized agents:

**Core Analysis Agents:**
- Trigger Line Agent (breakout detection)
- FTFC Continuity Agent (timeframe alignment) 
- Reversal Setup Agent (exhaustion patterns)
- Magnet Level Agent (support/resistance)
- Volatility Agent (volatility analysis)

**Risk & Execution Agents:**
- Position Sizing Agent (dynamic sizing)
- Entry Timing Agent (precise entries)
- Exit Strategy Agent (dynamic exits)
- Trade Director (master orchestrator)
- Enhanced Data Pipeline (real-time data)

## Files Overview

- `quick_start.py` - Immediate demo (recommended)
- `run_local_agents.py` - Full system with options
- `setup_local_simple.py` - Dependency installer
- `start_local.bat` - Windows shortcut
- `agents/` - All 10 specialized agents
- `AGENT_ARCHITECTURE_GUIDE.md` - Technical documentation
- `AGENT_INTEGRATION_QUICKSTART.md` - Integration guide

## Next Steps

1. ✅ Run `quick_start.py` to see agents working
2. 📖 Read `AGENT_ARCHITECTURE_GUIDE.md` for details  
3. 🔧 Try `run_local_agents.py` for full system
4. 🚀 Configure live data feeds when ready

## Support

- Check logs in `local_agents.log` for detailed information
- All agents are designed to work independently and together
- System processes data efficiently with sophisticated analysis
- Start with demo mode - it's safe and educational

**Happy Trading! 🤖📈**