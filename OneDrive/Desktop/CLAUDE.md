# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based trading bot system implementing Rob Smith's STRAT methodology for algorithmic trading. The system features both standalone and web interface components, performing systematic trading across multiple markets (crypto, stocks, futures) using advanced STRAT pattern recognition and Timeframe Continuity (TFC) filtering.

### STRAT Methodology Implementation

The system implements a complete STRAT trading methodology with:
- **Candle Classification**: Type 1 (Inside), 2U/2D (Directional), 3 (Outside) labeling
- **Pattern Recognition**: 2-1-2 Reversals, 3-1-2 Continuations, 2-2/3-2 setups
- **Timeframe Continuity Filter (TFC)**: Higher timeframe alignment validation
- **Actionable Entry Conditions**: Precise entry, stop-loss, and target calculations
- **Risk Management**: Pattern-specific position sizing and risk controls

## Common Commands

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_bot.py

# Start the trading bot directly
python trading_bot.py

# Run STRAT methodology bots
python professional_strat_trading_bot.py      # Latest professional implementation
python actionable_strat_trading_bot.py        # Actionable entry patterns
python strat_enhanced_trading_bot.py          # Enhanced STRAT with pattern detection

# Start the web interface
python web_interface.py
# or
flask run

# Start with WSGI server
python wsgi.py

# Start the Flask app on production
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

### Testing Commands
```bash
# Test bot configuration and connection
python test_bot.py

# Test setup environment
python test_setup.py
```

### Deployment Commands
```bash
# Make deployment scripts executable
chmod +x deployment_steps.sh
chmod +x setup_instructions.sh
chmod +x final_setup.sh
chmod +x start_bot.bat

# Run setup
./setup_instructions.sh

# Monitor logs
tail -f logs/trading_bot.log
```

## Architecture Overview

### Core Components

1. **TradingBot Class** (`trading_bot.py`, `app/trading_bot.py`)
   - Main bot logic with technical analysis and pattern detection
   - Supports multiple exchanges (Binance, Bybit, Alpaca)
   - Handles position management, risk controls, and real-time trading
   - Uses threading for concurrent symbol processing

2. **Web Interface** (`web_interface.py`, `app/web_interface.py`) 
   - Flask-based REST API for bot control
   - Dashboard UI (`templates/index.html`) for monitoring trades and patterns
   - Real-time status updates and configuration management

3. **STRAT Pattern Detection System**
   - **Professional STRAT Implementation**: Complete Rob Smith STRAT methodology
   - **Candle Classification**: Precise 1, 2U, 2D, 3 candle type labeling
   - **Pattern Recognition**: 2-1-2 Reversals, 3-1-2 Continuations, 2-2/3-2 setups
   - **Timeframe Continuity Filter (TFC)**: Multi-timeframe alignment validation
   - **Actionable Signals**: Precise entry conditions, stop-loss, and profit targets

4. **Agent Integration System** (`agent_integration.py`)
   - Quant Pattern Analyst Agent for classical chart patterns
   - Head & Shoulders, Cup & Handle, Double Tops/Bottoms detection
   - Multi-timeframe confluence analysis
   - Pattern confidence scoring and caching system

### Key Features

- **STRAT Methodology**: Complete implementation of Rob Smith's STRAT system
- **Multi-Market Support**: Crypto (Binance, Bybit), Stocks/Futures (Alpaca) 
- **Timeframe Continuity (TFC)**: Higher timeframe alignment filtering
- **Professional Pattern Detection**: 8 core STRAT patterns with precise entry/exit rules
- **Technical Indicators**: EMA crossovers, RSI, MACD, custom pattern detection
- **Risk Management**: Pattern-specific stop-loss, profit targets, position sizing
- **Real-time Data**: Market hours detection, new candle validation
- **Web Dashboard**: Multi-timeframe analysis, real-time pattern monitoring
- **Comprehensive Backtesting**: Historical pattern performance validation

### Configuration

The bot uses environment variables and config dictionaries:
- API keys and secrets for exchanges
- Base URLs for testnet/production environments
- Risk parameters (position size, stop-loss percentages)
- Trading symbols and timeframes

### Data Flow

1. **Data Acquisition**: Historical data fetching (yfinance, exchange APIs) with proper interval mapping
2. **Market Hours Validation**: US market open/close checking with timezone awareness
3. **Candle Classification**: STRAT candle type labeling (1, 2U, 2D, 3) 
4. **TFC Calculation**: Daily/weekly open alignment for timeframe continuity
5. **Pattern Detection**: Multi-pattern STRAT signal generation with confidence scoring
6. **Entry Validation**: Actionable signal filtering with precise entry/stop/target levels
7. **Risk Assessment**: Pattern-specific position sizing and risk controls
8. **Signal Logging**: CSV export for backtesting and performance tracking
9. **Real-time Monitoring**: New candle detection and breakout monitoring

### Deployment Architecture

- **Local Development**: Direct Python execution with Flask dev server
- **Production**: Nginx reverse proxy → Gunicorn WSGI → Flask app
- **AWS EC2**: Ubuntu instances with systemd services
- **Monitoring**: Log files, supervisor for process management

## Important Notes

- **STRAT Implementation Files**:
  - `professional_strat_trading_bot.py`: Latest production-ready implementation with TFC filter
  - `actionable_strat_trading_bot.py`: Actionable entry patterns with breakout monitoring  
  - `strat_enhanced_trading_bot.py`: Enhanced STRAT with comprehensive backtesting
  - `trading_bot.py`: Original bot with agent integration for classical patterns

- **Market Data Handling**: 
  - Proper yfinance interval mapping ("1h" → "60m")
  - New candle detection to prevent duplicate signals
  - Market hours awareness with timezone handling

- **STRAT Methodology**: Complete implementation of Rob Smith's STRAT system
  - Precise candle classification and pattern recognition
  - Timeframe Continuity (TFC) filtering for higher probability trades
  - Pattern-specific risk management with stop-loss and profit targets

- **Production Features**:
  - Comprehensive logging with rotating file handlers
  - CSV signal export for performance tracking
  - Real-time pattern monitoring and breakout detection
  - Backtesting with TFC effectiveness analysis

## Cursor Rules Integration

The project follows data analysis best practices from `Cursor.rules/Cursor.rules.md`:
- Vectorized operations for performance
- Descriptive variable names
- Proper error handling and data validation
- Structured notebook-style analysis (applicable to data processing components)