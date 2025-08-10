#!/usr/bin/env python3
"""
Quick Start - Direct Agent Demo
Run this to immediately see the enhanced agent system in action
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test if we can import basic dependencies"""
    print("Testing basic imports...")
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        print("[+] Basic dependencies OK")
        return True
    except ImportError as e:
        print(f"[-] Import error: {e}")
        print("Run: pip install pandas numpy yfinance")
        return False

def demo_individual_agents():
    """Demonstrate individual agents with synthetic data"""
    print("\n" + "="*60)
    print("NEWSTRATBOT ENHANCED AGENTS DEMO")
    print("="*60)
    
    try:
        # Import agents
        from agents.trigger_line_agent import TriggerLineAgent
        from agents.volatility_agent import VolatilityAgent
        from agents.position_sizing_agent import PositionSizingAgent
        
        print("\n[*] Loading agents...")
        
        # Initialize agents
        trigger_agent = TriggerLineAgent()
        volatility_agent = VolatilityAgent()
        position_agent = PositionSizingAgent()
        
        print("[+] Agents initialized successfully!")
        
        # Generate synthetic market data for demo
        import pandas as pd
        import numpy as np
        
        print("\n[*] Generating synthetic market data...")
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5T')
        np.random.seed(42)  # For reproducible results
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, 200)
        prices = base_price * (1 + returns).cumprod()
        
        data = pd.DataFrame(index=dates)
        data['open'] = prices * (1 + np.random.normal(0, 0.001, 200))
        data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.005, 200)))
        data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.005, 200)))
        data['close'] = prices
        data['volume'] = np.random.normal(1000000, 200000, 200)
        
        current_price = data['close'].iloc[-1]
        print(f"[+] Generated {len(data)} candles, current price: ${current_price:.2f}")
        
        print("\n" + "-"*50)
        print("TRIGGER LINE AGENT ANALYSIS")
        print("-"*50)
        
        # Test Trigger Line Agent
        trigger_analysis = trigger_agent.analyze_trigger_lines('BTCUSDT', '5m', data)
        print(f"Active Trigger Lines: {len(trigger_analysis.active_triggers)}")
        print(f"Recent Breaks: {len(trigger_analysis.recent_breaks)}")
        print(f"Directional Bias: {trigger_analysis.directional_bias}")
        print(f"Confidence Score: {trigger_analysis.confidence_score:.2f}")
        print(f"Momentum Score: {trigger_analysis.momentum_score:.2f}")
        if trigger_analysis.recent_breaks:
            latest_break = trigger_analysis.recent_breaks[0]
            print(f"Latest Break: {latest_break.break_type} at {latest_break.break_price:.2f}")
        else:
            print("No recent breakouts detected")
        
        print("\n" + "-"*50)
        print("VOLATILITY AGENT ANALYSIS")
        print("-"*50)
        
        # Test Volatility Agent
        vol_metrics = volatility_agent.analyze_volatility('BTCUSDT', '5m', data)
        print(f"Realized Volatility: {vol_metrics.realized_volatility:.2%}")
        print(f"GARCH Volatility: {vol_metrics.garch_volatility:.2%}")
        print(f"Volatility Regime: {vol_metrics.volatility_regime}")
        print(f"IV Percentile: {vol_metrics.iv_percentile:.1f}")
        print(f"IV Rank: {vol_metrics.iv_rank:.2f}")
        print(f"Volatility Trend: {vol_metrics.volatility_trend}")
        
        print("\n" + "-"*50)
        print("POSITION SIZING AGENT ANALYSIS")
        print("-"*50)
        
        # Test Position Sizing Agent
        signal_data = {
            'direction': 'long',
            'confidence': 0.75,
            'stop_loss': float(data['low'].iloc[-10:].min()),
            'entry_price': float(current_price)
        }
        
        portfolio_data = {
            'balance': 10000,
            'current_positions': {},
            'max_risk_per_trade': 0.02
        }
        
        volatility_data = {'volatility': vol_metrics.realized_volatility}
        
        position_calc = position_agent.calculate_position_size(
            'BTCUSDT', signal_data, portfolio_data, volatility_data
        )
        
        print(f"Signal: {signal_data['direction'].upper()} with {signal_data['confidence']:.1%} confidence")
        print(f"Entry Price: ${signal_data['entry_price']:.2f}")
        print(f"Stop Loss: ${signal_data['stop_loss']:.2f}")
        print(f"Risk per Trade: {portfolio_data['max_risk_per_trade']:.1%}")
        print(f"")
        print(f"RECOMMENDED POSITION:")
        print(f"Base Position Size: {position_calc.base_size:.4f}")
        print(f"Adjusted Position Size: {position_calc.adjusted_size:.4f}")
        print(f"USD Value: ${position_calc.adjusted_size * current_price:.2f}")
        print(f"Expected Max Loss: ${position_calc.expected_max_loss:.2f}")
        print(f"Sizing Method: {position_calc.sizing_method.value}")
        print(f"Risk Level: {position_calc.risk_level.value}")
        
        risk_reward = abs((signal_data['entry_price'] - signal_data['stop_loss']) / signal_data['entry_price'])
        print(f"Risk per Share: {risk_reward:.2%}")
        
        print("\n" + "="*60)
        print("DEMO COMPLETE - AGENT SYSTEM WORKING!")
        print("="*60)
        print("\nKEY INSIGHTS:")
        print(f"• Market volatility is {vol_metrics.volatility_regime.value.lower()}")
        print(f"• Position sizing accounts for current volatility")
        print(f"• Risk management limits exposure to ${position_calc.expected_max_loss:.2f}")
        print(f"• System processes {len(data)} data points efficiently")
        
        print(f"\nNext Steps:")
        print("• Connect to live data feeds (Binance, Bybit, etc.)")
        print("• Configure API keys for real trading")
        print("• Run with: python run_local_agents.py")
        print("• See full documentation in AGENT_ARCHITECTURE_GUIDE.md")
        
        return True
        
    except ImportError as e:
        print(f"[-] Could not import agents: {e}")
        print("Some agent files may be missing or have import issues")
        return False
    except Exception as e:
        print(f"[-] Demo failed: {e}")
        logger.exception("Demo error details:")
        return False

def main():
    """Main demo execution"""
    print("NEWSTRATBOT ENHANCED AGENTS - QUICK START")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print()
    
    # Test basic imports first
    if not test_basic_imports():
        print("\n[-] Please install required dependencies first")
        print("Run: pip install pandas numpy yfinance flask")
        return
    
    # Run agent demo
    if demo_individual_agents():
        print("\n[+] Demo completed successfully!")
    else:
        print("\n[-] Demo failed - check error messages above")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.exception("Unexpected error details:")