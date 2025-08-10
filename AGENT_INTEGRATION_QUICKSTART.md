# Agent Integration Quick Start Guide

## Overview

This guide provides step-by-step instructions for integrating and running the enhanced event-driven trading agents within the Newstratbot system.

## System Requirements

- Python 3.8+
- All dependencies from `requirements.txt`
- Active market data connection (Binance, Bybit, or Alpaca)
- Sufficient system resources for real-time processing

## Quick Start

### 1. Basic Agent Integration

```python
# Import the Trade Director (main orchestrator)
from agents.trade_director import TradeDirector

# Initialize the system with your configuration
config = {
    'exchange': 'bybit',  # or 'binance', 'alpaca'
    'api_key': 'your_api_key',
    'api_secret': 'your_api_secret',
    'testnet': True,  # Start with testnet
    'symbols': ['BTCUSDT', 'ETHUSDT'],
    'base_timeframe': '5m',
    'risk_per_trade': 0.02  # 2% risk per trade
}

# Create and start the trading system
director = TradeDirector(config)
await director.start_trading_system()
```

### 2. Individual Agent Usage

Each agent can be used independently for analysis:

```python
# Volatility Analysis
from agents.volatility_agent import VolatilityAgent
import pandas as pd

vol_agent = VolatilityAgent()
data = get_market_data('BTCUSDT', '5m', 200)  # Your data source
vol_metrics = vol_agent.analyze_volatility('BTCUSDT', '5m', data)

print(f"Current Volatility: {vol_metrics.current_volatility:.2%}")
print(f"IV Rank: {vol_metrics.iv_rank:.2f}")
print(f"Regime: {vol_metrics.regime}")
```

```python
# Position Sizing
from agents.position_sizing_agent import PositionSizingAgent

sizing_agent = PositionSizingAgent()
signal_data = {
    'direction': 'long',
    'confidence': 0.8,
    'stop_loss': 45000,
    'entry_price': 50000
}
portfolio_data = {
    'balance': 10000,
    'current_positions': {},
    'max_risk_per_trade': 0.02
}

position_calc = sizing_agent.calculate_position_size('BTCUSDT', signal_data, portfolio_data)
print(f"Recommended Position Size: {position_calc.final_position_size}")
```

### 3. Event-Driven Data Pipeline

```python
from agents.enhanced_data_pipeline import EnhancedDataPipeline, TimeFrame

# Initialize data pipeline
pipeline = EnhancedDataPipeline()

# Subscribe to real-time data
symbols = ['BTCUSDT', 'ETHUSDT']
timeframes = [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1]

await pipeline.subscribe_to_symbols(symbols, timeframes)

# Process incoming data events
async for event in pipeline.get_events():
    if event.event_type == 'new_candle':
        print(f"New {event.timeframe} candle for {event.symbol}: {event.data}")
```

## Agent Configuration

### Core Agent Settings

```python
agent_configs = {
    'trigger_line': {
        'min_volume_ratio': 1.2,
        'momentum_threshold': 0.65,
        'confirmation_candles': 2
    },
    'ftfc_continuity': {
        'required_timeframes': 5,
        'min_alignment_score': 0.7,
        'weight_higher_tf': True
    },
    'volatility': {
        'lookback_period': 50,
        'garch_model': 'GARCH(1,1)',
        'iv_source': 'deribit'  # if available
    },
    'position_sizing': {
        'kelly_fraction': 0.25,
        'max_position_size': 0.10,
        'volatility_target': 0.15
    }
}
```

### Risk Management Settings

```python
risk_config = {
    'max_daily_loss': 0.05,  # 5% daily loss limit
    'max_correlation': 0.7,   # Maximum position correlation
    'max_portfolio_heat': 0.20,  # Total portfolio risk
    'stop_loss_multiplier': 2.0,  # ATR multiplier for stops
    'profit_target_ratio': 2.0    # Risk:Reward ratio
}
```

## Running the Complete System

### Development/Testing Mode

```python
import asyncio
from agents.trade_director import TradeDirector

async def run_trading_system():
    config = {
        'exchange': 'bybit',
        'testnet': True,
        'symbols': ['BTCUSDT'],
        'base_currency': 'USDT',
        'initial_balance': 10000,
        'risk_per_trade': 0.01,  # Conservative 1% risk
        'agent_configs': agent_configs,
        'risk_config': risk_config
    }
    
    director = TradeDirector(config)
    
    try:
        await director.start_trading_system()
    except KeyboardInterrupt:
        print("Shutting down trading system...")
        await director.shutdown()

# Run the system
asyncio.run(run_trading_system())
```

### Production Mode

```python
import asyncio
import logging
from agents.trade_director import TradeDirector

# Set up production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

async def run_production_system():
    config = {
        'exchange': 'bybit',
        'testnet': False,  # LIVE TRADING - USE WITH CAUTION
        'api_key': os.getenv('BYBIT_API_KEY'),
        'api_secret': os.getenv('BYBIT_API_SECRET'),
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        'base_currency': 'USDT',
        'risk_per_trade': 0.02,
        'max_daily_trades': 10,
        'agent_configs': agent_configs,
        'risk_config': risk_config,
        'monitoring': {
            'enable_alerts': True,
            'webhook_url': 'your_webhook_url',
            'alert_on_loss': True,
            'alert_on_system_error': True
        }
    }
    
    director = TradeDirector(config)
    
    try:
        logging.info("Starting production trading system...")
        await director.start_trading_system()
    except Exception as e:
        logging.error(f"System error: {e}")
        await director.emergency_shutdown()

# For production, consider using a process manager like systemd or supervisor
asyncio.run(run_production_system())
```

## Integration with Existing Web Interface

### Flask Integration

```python
from flask import Flask, jsonify, request
from agents.trade_director import TradeDirector
import asyncio

app = Flask(__name__)
director = None

@app.route('/api/start_agents', methods=['POST'])
def start_agents():
    global director
    config = request.json
    
    if director is None:
        director = TradeDirector(config)
        asyncio.create_task(director.start_trading_system())
        return jsonify({'status': 'started'})
    return jsonify({'error': 'Already running'})

@app.route('/api/agent_status')
def agent_status():
    if director:
        status = director.get_agent_status()
        return jsonify(status)
    return jsonify({'error': 'Not running'})

@app.route('/api/trading_signals')
def get_signals():
    if director:
        signals = director.get_recent_signals()
        return jsonify(signals)
    return jsonify([])
```

## Monitoring and Debugging

### Agent Health Monitoring

```python
# Check individual agent status
agent_status = director.get_agent_status()
for agent_name, status in agent_status.items():
    print(f"{agent_name}: {'✅' if status['healthy'] else '❌'}")
    if not status['healthy']:
        print(f"  Error: {status['error']}")
        print(f"  Last Update: {status['last_update']}")
```

### Performance Metrics

```python
# Get performance metrics
metrics = director.get_performance_metrics()
print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Average Return: {metrics['avg_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Real-time Logging

```python
# Set up structured logging for agents
import logging
import json

class AgentLogFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'agent': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'symbol': getattr(record, 'symbol', None),
            'action': getattr(record, 'action', None)
        }
        return json.dumps(log_entry)

# Apply to all agents
handler = logging.StreamHandler()
handler.setFormatter(AgentLogFormatter())
logging.getLogger('agents').addHandler(handler)
logging.getLogger('agents').setLevel(logging.INFO)
```

## Troubleshooting

### Common Issues

1. **Agent Not Starting**: Check API credentials and network connectivity
2. **High Memory Usage**: Reduce lookback periods and data retention
3. **Slow Performance**: Optimize agent update frequencies
4. **Missing Signals**: Verify data quality and agent configuration

### Debug Mode

```python
# Enable debug mode for detailed logging
config['debug_mode'] = True
config['log_level'] = 'DEBUG'

# This will provide detailed logs for each agent's decision process
director = TradeDirector(config)
```

## Next Steps

1. Start with testnet/paper trading
2. Monitor agent performance and adjust configurations
3. Gradually increase position sizes
4. Implement additional risk controls as needed
5. Consider deploying to cloud infrastructure for 24/7 operation

For detailed technical documentation, see [AGENT_ARCHITECTURE_GUIDE.md](AGENT_ARCHITECTURE_GUIDE.md).

For production deployment, see [AWS_EC2_DEPLOYMENT_GUIDE.md](AWS_EC2_DEPLOYMENT_GUIDE.md).