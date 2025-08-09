# Quant Pattern Analyst Agent Integration Documentation

## Overview

The **quant-pattern-analyst** agent has been successfully integrated into the trading bot system. This agent specializes in identifying, analyzing, and presenting classical chart patterns in financial data.

## Purpose

The agent enhances the trading bot's pattern detection capabilities by identifying:
- **Head and Shoulders** - Major reversal pattern
- **Cup and Handle** - Bullish continuation pattern
- **Double Tops/Bottoms** - Reversal patterns
- **Triangles** (Ascending/Descending/Symmetrical) - Continuation/reversal patterns
- **Flags and Pennants** - Short-term continuation patterns
- **Wedges** (Rising/Falling) - Reversal patterns

## Architecture

### 1. Agent Integration Module (`agent_integration.py`)
- `QuantPatternAnalystAgent` class wraps the Claude agent
- Handles pattern detection, caching, and signal generation
- Provides confidence scoring and pattern validation

### 2. Trading Bot Integration (`trading_bot.py`)
- Agent initialized in bot constructor
- `detect_patterns()` method enhanced to use agent when enabled
- Dual pattern detection: basic STRAT patterns + advanced agent patterns
- Pattern source tracking ('basic' vs 'agent')

### 3. Web Interface Integration (`web_interface.py`)
New API endpoints:
- `/api/patterns` - Enhanced to separate basic vs agent patterns
- `/api/agent/status` - Get agent status and configuration
- `/api/agent/analyze` - Trigger on-demand pattern analysis

### 4. Configuration (`config.py`)
New environment variables:
- `USE_PATTERN_AGENT` - Enable/disable agent (default: true)
- `AGENT_MIN_CONFIDENCE` - Minimum pattern confidence (default: 0.75)
- `AGENT_MAX_PATTERNS` - Max patterns to track per symbol (default: 10)
- `AGENT_CACHE_DURATION` - Cache duration in seconds (default: 900)

## Features

### Pattern Detection
- Runs alongside existing STRAT pattern detection
- Provides confidence scores for each pattern
- Detects complex multi-bar patterns
- Returns entry/exit points and risk/reward ratios

### Pattern Caching
- Caches patterns per symbol to reduce API calls
- Configurable cache duration
- Automatic cache invalidation

### Signal Generation
- Aggregates patterns into trading signals
- Bullish/bearish scoring system
- Configurable confidence thresholds

### Pattern Validation
- Validates patterns against current price
- Checks stop-loss and target violations
- Removes invalidated patterns

## Usage

### Basic Usage
```python
from trading_bot import TradingBot

# Bot automatically initializes agent if enabled
bot = TradingBot(
    market_type="futures",
    timeframe="1h",
    config={'use_agent_patterns': True}
)

# Run bot with symbols
bot.run(['BTC/USDT', 'ETH/USDT'])
```

### Direct Agent Usage
```python
from agent_integration import QuantPatternAnalystAgent
import pandas as pd

# Initialize agent
agent = QuantPatternAnalystAgent({
    'min_confidence': 0.8,
    'max_patterns': 5
})

# Analyze patterns
df = pd.DataFrame(price_data)  # Your OHLCV data
analysis = agent.analyze_patterns('BTC/USDT', df, '1h')

# Get trading signals
signals = agent.get_pattern_signals('BTC/USDT')
```

### Configuration
Set in `.env` file:
```env
# Enable/disable agent
USE_PATTERN_AGENT=true

# Agent parameters
AGENT_MIN_CONFIDENCE=0.75
AGENT_MAX_PATTERNS=10
AGENT_CACHE_DURATION=900
```

## API Endpoints

### GET /api/patterns
Returns all detected patterns with separation by source:
```json
{
  "patterns": [...],
  "basic_patterns": [...],
  "agent_patterns": [...],
  "total_patterns": 15,
  "agent_enabled": true
}
```

### GET /api/agent/status
Returns agent status and configuration:
```json
{
  "agent_active": true,
  "agent_name": "quant-pattern-analyst",
  "min_confidence": 0.75,
  "cached_symbols": ["BTC/USDT", "ETH/USDT"],
  "last_analysis": "2024-01-15T10:30:00"
}
```

### POST /api/agent/analyze
Trigger pattern analysis for a symbol:
```json
Request:
{
  "symbol": "BTC/USDT"
}

Response:
{
  "status": "success",
  "analysis": {
    "symbol": "BTC/USDT",
    "patterns": [...],
    "summary": "Detected 3 patterns with avg confidence 0.82"
  }
}
```

## Pattern Types and Actions

| Pattern Type | Trading Action | Typical Confidence |
|-------------|----------------|-------------------|
| Cup and Handle | Buy | 0.75-0.85 |
| Head and Shoulders | Sell | 0.80-0.90 |
| Double Bottom | Buy | 0.70-0.80 |
| Double Top | Sell | 0.70-0.80 |
| Ascending Triangle | Buy | 0.65-0.75 |
| Descending Triangle | Sell | 0.65-0.75 |
| Bull Flag | Buy | 0.70-0.80 |
| Bear Flag | Sell | 0.70-0.80 |
| Rising Wedge | Sell | 0.65-0.75 |
| Falling Wedge | Buy | 0.65-0.75 |

## Testing

Run the test suite:
```bash
python test_agent_integration.py
```

Tests cover:
- Agent initialization
- Bot integration
- Pattern detection
- Signal generation
- Pattern validation
- API endpoints
- Configuration loading

## Performance Considerations

1. **Caching**: Patterns are cached for 15 minutes by default
2. **Fallback**: If agent fails, system falls back to basic pattern detection
3. **Concurrency**: Agent calls are synchronous, may add 1-2s latency
4. **Rate Limiting**: Consider agent API limits when setting scan frequency

## Troubleshooting

### Agent Not Detecting Patterns
- Check `USE_PATTERN_AGENT=true` in config
- Verify sufficient data points (min 50 bars recommended)
- Check logs for agent invocation errors

### Low Confidence Patterns
- Adjust `AGENT_MIN_CONFIDENCE` lower (e.g., 0.65)
- Ensure clean price data without gaps
- Check timeframe appropriateness for patterns

### Agent Timeouts
- Increase timeout in `_invoke_agent()` method
- Reduce data points sent to agent
- Check system resources

## Future Enhancements

1. **Parallel Pattern Detection**: Run multiple agents for different pattern types
2. **Pattern Backtesting**: Historical performance tracking
3. **Machine Learning Integration**: Train on successful patterns
4. **Real-time Pattern Alerts**: WebSocket notifications
5. **Pattern Visualization**: Chart overlays in web interface
6. **Multi-Timeframe Confluence**: Cross-reference patterns across timeframes

## Conclusion

The quant-pattern-analyst agent significantly enhances the trading bot's pattern recognition capabilities. It provides sophisticated pattern detection that complements the existing STRAT methodology, offering traders more comprehensive market analysis and higher-confidence trading signals.