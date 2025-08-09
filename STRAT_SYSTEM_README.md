# STRAT Methodology Trading System

## Overview

This is a complete implementation of Rob Smith's STRAT methodology for systematic trading. The system provides precise candle classification, pattern recognition, and timeframe continuity filtering to identify high-probability trading opportunities.

## Core STRAT Concepts

### Candle Classification

Every candle is classified relative to the previous candle:

- **Type 1 (Inside Bar)**: High < Previous High AND Low > Previous Low
- **Type 2U (Directional Up)**: High > Previous High AND Low >= Previous Low  
- **Type 2D (Directional Down)**: Low < Previous Low AND High <= Previous High
- **Type 3 (Outside Bar)**: High > Previous High AND Low < Previous Low

### STRAT Patterns

The system detects 8 core STRAT patterns:

#### Reversal Patterns
1. **2-1-2 Bullish Reversal** (2D → 1 → 2U): Down move, inside bar, break higher
2. **2-1-2 Bearish Reversal** (2U → 1 → 2D): Up move, inside bar, break lower
3. **2-2 Bullish Reversal** (2D → 2U): Direct reversal from down to up
4. **2-2 Bearish Reversal** (2U → 2D): Direct reversal from up to down
5. **3-2 Bullish Reversal** (3 → 2U): Outside bar followed by upward break
6. **3-2 Bearish Reversal** (3 → 2D): Outside bar followed by downward break

#### Continuation Patterns  
7. **3-1-2 Bullish Continuation** (3 → 1 → 2U): Outside bar, consolidation, breakout up
8. **3-1-2 Bearish Continuation** (3 → 1 → 2D): Outside bar, consolidation, breakdown

### Timeframe Continuity (TFC) Filter

TFC ensures trades align with higher timeframe bias:

- **TFC Up**: Current Close > Daily Open AND Current Close > Weekly Open
- **TFC Down**: Current Close < Daily Open AND Current Close < Weekly Open

Only patterns that pass TFC filter are considered high-probability signals.

## System Architecture

### Core Files

- **`professional_strat_trading_bot.py`**: Latest production implementation
- **`actionable_strat_trading_bot.py`**: Actionable entry patterns with breakout monitoring
- **`strat_enhanced_trading_bot.py`**: Enhanced STRAT with comprehensive backtesting

### Key Classes

#### `StratDetector`
- Candle classification and labeling
- Pattern detection for all 8 STRAT patterns
- TFC calculation and validation
- Pivot level identification for targets

#### `ProfessionalStratTradingBot`  
- Real-time market monitoring
- Pattern scanning and signal generation
- Market hours awareness
- Signal logging and backtesting

#### `StratSignal` (Dataclass)
- Complete signal information
- Entry, stop-loss, and profit target levels
- TFC pass/fail status
- Pattern-specific notes

## Entry and Exit Rules

### 2-1-2 Reversals
- **Entry**: Breakout above/below inside bar high/low
- **Stop**: Inside bar opposite extreme  
- **Target**: 2:1 risk/reward from entry

### 3-1-2 Continuations
- **Entry**: Breakout above/below inside bar high/low
- **Stop**: Inside bar opposite extreme
- **Target**: 1.5:1 risk/reward from entry

### 2-2 & 3-2 Reversals
- **Entry**: Breakout above/below previous candle high/low
- **Stop**: Previous candle opposite extreme
- **Target**: 1.5:1 risk/reward from entry

## Usage Examples

### Live Trading
```python
from professional_strat_trading_bot import ProfessionalStratTradingBot

# Initialize bot
bot = ProfessionalStratTradingBot(symbol="SPY", timeframe="5m")

# Run live monitoring
bot.run()
```

### Backtesting
```python
# Run backtest
bot.backtest(days=10)

# Output shows:
# - Total signals and hit rates
# - TFC-filtered signals performance
# - Pattern-by-pattern breakdown
# - TFC improvement metrics
```

### Pattern Detection
```python
from professional_strat_trading_bot import StratDetector
import yfinance as yf

# Get data
data = yf.download("SPY", period="1d", interval="5m")

# Detect patterns
detector = StratDetector()
signals = detector.scan_all_patterns(data)

for signal in signals:
    if signal.tfc_pass:
        print(f"TFC PASS: {signal.pattern} - Entry: ${signal.entry:.2f}")
```

## Signal Output Format

The system outputs detailed signal information:

```
🟢✅ TFC PASS: 2-1-2 Bullish Reversal
   📍 Entry: $425.67 | 🛑 Stop: $424.23 | 🎯 Target: $427.55
   🎯 Target2: $428.83
   📊 TFC Status: PASS | Notes: Inside bar breakout, TFC: Pass
```

## CSV Export

All signals are logged to CSV files for analysis:
- Timestamp, symbol, pattern type
- Entry, stop, and target prices  
- TFC pass/fail status
- Pattern-specific notes

## Performance Tracking

The system tracks:
- Overall hit rates by pattern
- TFC filter effectiveness
- Risk/reward ratios achieved
- Pattern frequency and reliability

## Installation & Setup

```bash
# Install dependencies
pip install pandas numpy yfinance pytz

# Run the professional STRAT bot
python professional_strat_trading_bot.py

# Enter symbol: SPY
# Enter timeframe: 5m  
# Run mode: live (or backtest)
```

## Key Advantages

1. **Objective Patterns**: Eliminates subjective pattern interpretation
2. **TFC Filter**: Significantly improves win rates by filtering against trend
3. **Precise Entries**: Exact entry, stop, and target calculations
4. **Risk Management**: Pattern-specific risk/reward ratios
5. **Backtesting**: Historical validation of pattern performance
6. **Real-time Monitoring**: Live pattern detection and breakout alerts

## Best Practices

1. **Always Use TFC**: Only trade patterns that pass timeframe continuity
2. **Respect Stops**: Pattern-based stop losses are critical for risk management  
3. **Multiple Timeframes**: Confirm patterns across different timeframes
4. **Volume Confirmation**: Higher volume on breakouts increases reliability
5. **Market Hours**: Focus on active trading sessions for best results

## Disclaimer

This system is for educational and research purposes. All trading involves risk of loss. Past performance does not guarantee future results. Always paper trade extensively before risking real capital.