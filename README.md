# 🤖 Newstratbot - Advanced Multi-Agent STRAT Trading Bot

A sophisticated Python trading bot implementing **The STRAT methodology** with a multi-agent architecture for intelligent market analysis and automated trading decisions. Features real-time pattern detection, multi-timeframe confluence analysis, and a modern web interface dashboard.

## ✨ Features

### Core Trading Features
- 🎯 **STRAT Pattern Detection**: Identifies 1s, 2s, 3s and complex combos (2-1-2, 3-1-2, etc.)
- 🤖 **Multi-Agent Architecture**: Specialized agents for scenario classification, timeframe confluence, and trigger monitoring
- 📊 **Web Dashboard**: Modern, responsive interface with live updates
- 📈 **Multi-Timeframe Analysis**: Monthly to 5-minute timeframe confluence
- 🎛️ **Trigger Line Monitoring**: Entry/exit signal validation with momentum analysis
- 📋 **Live Logging**: Real-time trading logs and agent communications
- 🔄 **Auto-refresh**: Real-time WebSocket updates
- 💰 **Multi-symbol Support**: Crypto (Binance, Bybit) and Stocks (Alpaca, any Yahoo Finance symbol)

### Agent System Features
- 🧠 **Intelligent Coordination**: Supervisor agent orchestrates multiple specialized sub-agents
- 📐 **Timeframe Confluence**: Analyzes alignment across 7+ timeframes
- 🎯 **High-Probability Setups**: Only trades with >70% confluence score
- ⚡ **Real-time Decision Making**: Sub-second latency for signal generation
- 🛡️ **Risk Management**: Automated position sizing and stop-loss placement

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Web Server**:
   ```bash
   python app.py
   ```

3. **Open Dashboard**:
   - Navigate to: `http://localhost:5000`
   - Enter stock symbol (e.g., SPY)
   - Select timeframe
   - Click "🚀 Start Bot"

### Option 2: Console Version

```bash
python trading_bot.py
```

## 📱 Web Interface Screenshots

The web dashboard includes:
- **Control Panel**: Start/stop bot with custom symbols and timeframes
- **Real-time Price Display**: Live market data updates
- **Signal Detection**: Visual indicators for bullish/bearish patterns
- **Trading Logs**: Comprehensive activity logging
- **Status Monitoring**: Bot status and connection indicators

## 🔧 Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/blacula3000/Newstratbot.git
   cd Newstratbot
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**:
   ```bash
   python app.py
   ```

## 🤖 Enhanced Event-Driven Multi-Agent Architecture

The system implements a sophisticated **Event-Driven Multi-Agent Architecture** for institutional-grade trading performance:

### Event-Driven System
- **Enhanced Data Pipeline**: Real-time multi-timeframe data ingestion and distribution
- **Event Processing**: Immediate response to market events rather than batch processing  
- **Asynchronous Operations**: Non-blocking data processing and analysis
- **State Management**: Persistent tracking of market conditions and trading states

### Specialized Trading Agents

#### Core Analysis Agents
- **Trigger Line Agent**: Advanced STRAT 2u/2d/3 breakout detection with momentum validation
- **FTFC Continuity Agent**: Full timeframe continuity analysis for higher probability trades
- **Reversal Setup Agent**: Exhaustion pattern detection with 7 reversal pattern types
- **Magnet Level Agent**: Comprehensive price level magnetism analysis (10 level types)
- **Volatility Agent**: Advanced volatility analysis with IV rank integration and GARCH modeling

#### Risk & Execution Agents  
- **Position Sizing Agent**: Dynamic position sizing with Kelly Criterion and volatility normalization
- **Entry Timing Agent**: Precise entry logic with market microstructure analysis
- **Exit Strategy Agent**: Dynamic exit management with 10 exit trigger types
- **Trade Director**: Master orchestration system with ensemble decision making

### Agent Architecture Flow
```
Market Data → Enhanced Data Pipeline → Specialized Agents → Trade Director → Execution
                      ↓
    ┌─────────────────┼─────────────────┐
    │                 │                 │
Trigger Line     Volatility        Reversal Setup
    │                 │                 │
FTFC Continuity  Position Sizing   Magnet Level  
    │                 │                 │
Entry Timing     Exit Strategy     Trade Director
    │                 │                 │
    └─────────────────┼─────────────────┘
                      ↓
            Ensemble Decision Making
```

### Multi-Agent Ensemble Features
- **Weighted Consensus**: Each agent contributes confidence-weighted analysis
- **Real-time Coordination**: Agents communicate and share insights continuously
- **Ensemble Decision Making**: Multiple agents validate signals for higher probability trades
- **Dynamic Risk Management**: Multi-layer risk controls across all agents

For detailed technical documentation, see [AGENT_ARCHITECTURE_GUIDE.md](AGENT_ARCHITECTURE_GUIDE.md)

## 📊 Technical Details

### STRAT Pattern Detection
- **Type 1 (Inside Bar)**: Consolidation pattern
- **Type 2 (Directional Bar)**: Trend continuation
- **Type 3 (Outside Bar)**: Volatility expansion
- **Reversal Combos**: 2-1-2, 3-1-2, 2-2, 3-2-1 patterns

### Data Sources
- **Crypto**: Binance, Bybit APIs
- **Stocks**: Alpaca API, Yahoo Finance
- **Real-time**: WebSocket connections
- **Historical**: REST API endpoints

### Technology Stack
- **Backend**: Flask + SocketIO for real-time updates
- **Frontend**: Modern HTML5 + CSS3 + JavaScript
- **Data Analysis**: Pandas + NumPy + TA-Lib
- **Trading APIs**: CCXT, Alpaca, PyBit
- **Agent System**: Event-driven multi-agent framework with ensemble decision making
- **Volatility Modeling**: GARCH, Parkinson, Garman-Klass estimators
- **Risk Management**: Kelly Criterion, volatility targeting, correlation analysis
- **Logging**: Python logging with real-time streaming

## 🌐 Web Interface Features

- **Real-time Updates**: WebSocket connections for instant data
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Controls**: Start/stop bot with custom parameters
- **Visual Signals**: Color-coded trading alerts
- **Log Viewer**: Dedicated page for trading activity logs

## 📝 Usage Examples

### Web Interface
1. Open `http://localhost:5000`
2. Enter symbol: `AAPL`
3. Select timeframe: `5m`
4. Click "Start Bot"
5. Monitor real-time signals and price updates

### Console
```python
bot = TradingBot(symbol="SPY", timeframe="5m")
bot.run()
```

## 🔍 Monitoring

- **Live Dashboard**: Real-time price and signal updates
- **Trading Logs**: Accessible via web interface at `/logs`
- **System Status**: Visual indicators for bot status
- **Error Handling**: Automatic retry and error logging

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

---

## Original Console Code:

```python
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class CandleStickPattern:
    def __init__(self):
        self.patterns = {
            'bullish_reversal': self._check_bullish_reversal,
            'bearish_reversal': self._check_bearish_reversal
        }
    
    def _check_bullish_reversal(self, candles: pd.DataFrame) -> bool:
        if len(candles) < 2:
            return False
        
        current = candles.iloc[-1]
        previous = candles.iloc[-2]
        
        return (previous['Close'] < previous['Open'] and
                current['Close'] > current['Open'] and
                current['Low'] < previous['Low'])
    
    def _check_bearish_reversal(self, candles: pd.DataFrame) -> bool:
        if len(candles) < 2:
            return False
            
        current = candles.iloc[-1]
        previous = candles.iloc[-2]
        
        return (previous['Close'] > previous['Open'] and
                current['Close'] < current['Open'] and
                current['High'] > previous['High'])

class TradingBot:
    def __init__(self, symbol: str, timeframe: str = '5m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.pattern_detector = CandleStickPattern()
        logging.info(f"Bot initialized for {symbol} with {timeframe} timeframe")
    
    def fetch_data(self) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period='1d', interval=self.timeframe)
            return data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def analyze_pattern(self, data: pd.DataFrame) -> dict:
        if data.empty:
            return {'bullish_reversal': False, 'bearish_reversal': False}
        
        signals = {
            'bullish_reversal': self.pattern_detector._check_bullish_reversal(data),
            'bearish_reversal': self.pattern_detector._check_bearish_reversal(data)
        }
        return signals

    def run(self):
        print(f"Starting bot for {self.symbol}...")
        logging.info(f"Bot started for {self.symbol}")
        
        while True:
            try:
                # Get current time
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Fetch current market data
                data = self.fetch_data()
                
                if not data.empty:
                    # Analyze patterns
                    signals = self.analyze_pattern(data)
                    
                    # Current price
                    current_price = data['Close'].iloc[-1]
                    
                    # Print status
                    print(f"\nTime: {current_time}")
                    print(f"Symbol: {self.symbol}")
                    print(f"Current Price: ${current_price:.2f}")
                    
                    if signals['bullish_reversal']:
                        print("🟢 Bullish reversal detected!")
                        logging.info(f"Bullish reversal detected for {self.symbol} at ${current_price:.2f}")
                    
                    elif signals['bearish_reversal']:
                        print("🔴 Bearish reversal detected!")
                        logging.info(f"Bearish reversal detected for {self.symbol} at ${current_price:.2f}")
                    
                    else:
                        print("No pattern detected")
                
                # Wait before next check
                time.sleep(60)  # 60 seconds delay
                
            except KeyboardInterrupt:
                print("\nBot stopped by user")
                logging.info("Bot stopped by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    # Create a bot instance (example with SPY)
    symbol = input("Enter stock symbol (e.g., SPY): ").upper()
    timeframe = input("Enter timeframe (1m, 5m, 15m, 1h): ")
    
    bot = TradingBot(symbol=symbol, timeframe=timeframe)
    bot.run()
