# STRAT Trading Bot - Multi-Agent Architecture

## Overview
This trading bot implements a sophisticated multi-agent system based on "The STRAT" methodology by Rob Smith. Each agent specializes in specific aspects of market analysis and trading decisions, working together to identify high-probability trading opportunities.

## Agent Hierarchy and Responsibilities

```
                    ┌─────────────────────┐
                    │  Supervisor Agent   │
                    │  (Orchestrator)     │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
    ┌───────────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
    │ Scenario Agent   │ │Timeframe │ │  Trigger    │
    │ (Pattern ID)     │ │Confluence│ │Line Monitor │
    └───────────┬──────┘ └────┬─────┘ └──────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Trade Director     │
                    │ (Decision Synthesis) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Execution Engine    │
                    │  (Order Management)  │
                    └─────────────────────┘
```

## Core Agents

### 1. Supervisor Agent (`supervisor_agent.py`)
**Role**: Master Orchestrator
- Coordinates all sub-agents
- Manages agent communication flow
- Aggregates signals from multiple agents
- Makes final go/no-go decisions on trades

**Key Functions**:
```python
- coordinate_agents()
- aggregate_signals()
- validate_trade_setup()
- manage_agent_lifecycle()
```

### 2. STRAT Scenario Classifier Agent (`strat_scenario_classifier.py`)
**Role**: Pattern Recognition Specialist
- Identifies STRAT scenarios (1s, 2s, 3s)
- Detects inside bars, outside bars, and directional bars
- Classifies reversal patterns (2-1-2, 3-1-2, etc.)
- Tracks scenario transitions

**Key Patterns Detected**:
- **Type 1**: Inside bar (consolidation)
- **Type 2**: Directional bar (trend continuation)
- **Type 3**: Outside bar (volatility expansion)
- **Reversal Combos**: 2-1-2, 3-1-2, 2-2, 3-2-1

### 3. Timeframe Confluence Agent (`strat_timeframe_confluence.py`)
**Role**: Multi-Timeframe Alignment Analyzer
- Analyzes alignment across multiple timeframes
- Calculates confluence strength scores
- Identifies timeframe continuity/discontinuity
- Monitors broadening formations

**Timeframes Analyzed**:
- Monthly → Weekly → Daily → 4H → 1H → 15m → 5m
- Provides confluence scores (0-100%)
- Tracks scenario alignment across timeframes

### 4. Trigger Line Monitor Agent (`trigger_line_agent.py`)
**Role**: Entry/Exit Signal Specialist
- Monitors trigger line breaks
- Calculates trigger line angles and momentum
- Identifies false breaks vs confirmed breaks
- Tracks trigger line clustering

**Key Metrics**:
- Trigger line slope/angle
- Break confirmation criteria
- Volume at break points
- Historical trigger effectiveness

### 5. Trade Director Agent (`strat_trade_director.py`)
**Role**: Trade Decision Synthesizer
- Combines signals from all agents
- Calculates position sizing
- Sets stop-loss and take-profit levels
- Manages risk parameters

**Decision Factors**:
- Confluence strength (minimum 70%)
- Risk-reward ratio (minimum 1:2)
- Account risk per trade (max 2%)
- Market conditions filter

## Agent Communication Flow

### Signal Generation Pipeline
```
1. Market Data Input
   ↓
2. Scenario Classification (all timeframes)
   ↓
3. Timeframe Confluence Analysis
   ↓
4. Trigger Line Validation
   ↓
5. Trade Director Synthesis
   ↓
6. Execution Decision
```

### Inter-Agent Messaging Protocol

Each agent communicates using standardized message format:
```json
{
  "agent_id": "scenario_classifier",
  "timestamp": "2024-01-15T10:30:00Z",
  "symbol": "AAPL",
  "timeframe": "1H",
  "signal": {
    "type": "reversal_setup",
    "pattern": "2-1-2",
    "confidence": 85,
    "details": {...}
  }
}
```

## Agent Interaction Examples

### Example 1: High-Probability Long Setup
```
1. Scenario Classifier detects 2-1-2 reversal on daily
2. Timeframe Confluence confirms alignment (85% score)
   - Weekly: Type 1 (inside)
   - Daily: 2-1-2 reversal
   - 4H: Type 2 up
3. Trigger Line Monitor confirms break above
4. Trade Director approves entry with:
   - Entry: $150.50
   - Stop: $148.00
   - Target: $156.00
   - Position Size: 100 shares
```

### Example 2: Signal Rejection
```
1. Scenario Classifier detects 3-2 combo on 1H
2. Timeframe Confluence shows poor alignment (45% score)
   - Daily: Type 2 down (conflicting)
   - 4H: Type 3 (high volatility)
3. Trade Director rejects trade due to:
   - Low confluence score
   - Conflicting higher timeframe
```

## Configuration and Tuning

### Agent Priority Weights
```python
AGENT_WEIGHTS = {
    'scenario_classifier': 0.30,
    'timeframe_confluence': 0.35,
    'trigger_line': 0.25,
    'market_conditions': 0.10
}
```

### Minimum Thresholds
```python
MIN_CONFLUENCE_SCORE = 70  # Minimum alignment score
MIN_CONFIDENCE = 75         # Minimum pattern confidence
MIN_RR_RATIO = 2.0         # Minimum risk-reward
MAX_RISK_PERCENT = 2.0     # Maximum risk per trade
```

## Real-Time Monitoring

### Dashboard Metrics
- Active scenarios per timeframe
- Current confluence scores
- Trigger line status
- Open positions and P&L
- Agent health status

### Alert System
- New high-probability setups
- Scenario transitions
- Trigger line breaks
- Risk threshold warnings

## Performance Optimization

### Agent Response Times
- Scenario Classifier: <100ms per symbol
- Timeframe Confluence: <200ms per analysis
- Trigger Line Monitor: <50ms per check
- Trade Director: <150ms per decision

### Scalability
- Supports monitoring 50+ symbols simultaneously
- Processes 1000+ candles per second
- Handles multiple exchange connections
- Maintains sub-second latency

## Error Handling and Failsafes

### Agent Failure Recovery
```python
- Automatic agent restart on failure
- Fallback to conservative mode
- Position freeze on critical errors
- Emergency stop-loss activation
```

### Data Validation
- Candle data integrity checks
- Pattern validation rules
- Confluence score verification
- Risk parameter boundaries

## Future Enhancements

### Planned Agent Additions
1. **Volume Profile Agent**: Analyze volume at price levels
2. **Market Regime Agent**: Identify trending vs ranging markets
3. **Sentiment Analysis Agent**: Incorporate news and social sentiment
4. **Machine Learning Agent**: Pattern success rate optimization

### Integration Roadmap
- TradingView webhook integration
- Multi-exchange arbitrage
- Options strategy overlay
- Portfolio rebalancing agent

## Usage Example

```python
from supervisor_agent import SupervisorAgent
from trading_bot import TradingBot

# Initialize the trading system
bot = TradingBot(
    api_key="your_api_key",
    api_secret="your_secret"
)

# Start the multi-agent system
supervisor = SupervisorAgent(bot)
supervisor.start_monitoring(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    timeframes=['1d', '4h', '1h', '15m']
)

# Agents now work autonomously
# Supervisor coordinates and executes trades
```

## Conclusion

This multi-agent architecture provides a robust, scalable, and intelligent trading system that leverages the power of specialized agents working in harmony. Each agent's expertise contributes to making informed, high-probability trading decisions while maintaining strict risk management protocols.

The modular design allows for easy enhancement and addition of new agents as trading strategies evolve, ensuring the system remains adaptive and competitive in changing market conditions.