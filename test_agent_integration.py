"""
Test script for Quant Pattern Analyst Agent Integration
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agent_integration import QuantPatternAnalystAgent
from trading_bot import TradingBot
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgentIntegration(unittest.TestCase):
    """Test suite for agent integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = QuantPatternAnalystAgent({
            'min_confidence': 0.7,
            'max_patterns': 5
        })
        
        self.bot = TradingBot(
            market_type="futures",
            timeframe="1h",
            config={'use_agent_patterns': True}
        )
        
        # Create sample price data
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        
        # Generate synthetic price data with patterns
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({
            'Open': close_prices + np.random.randn(100) * 0.1,
            'High': close_prices + np.abs(np.random.randn(100) * 0.3),
            'Low': close_prices - np.abs(np.random.randn(100) * 0.3),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        df.attrs['symbol'] = 'TEST/USDT'
        return df
    
    def test_agent_initialization(self):
        """Test agent is properly initialized"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.agent_name, "quant-pattern-analyst")
        self.assertEqual(self.agent.min_pattern_confidence, 0.7)
    
    def test_bot_agent_integration(self):
        """Test bot properly integrates with agent"""
        self.assertIsNotNone(self.bot.pattern_agent)
        self.assertTrue(self.bot.use_agent_patterns)
        self.assertEqual(self.bot.pattern_agent.min_pattern_confidence, 0.75)
    
    def test_pattern_detection(self):
        """Test pattern detection functionality"""
        patterns = self.bot.detect_patterns(self.sample_data)
        
        # Check patterns are detected
        self.assertIsInstance(patterns, list)
        
        # Check pattern structure
        if patterns:
            pattern = patterns[0]
            self.assertIn('type', pattern)
            self.assertIn('confidence', pattern)
            self.assertIn('action', pattern)
            self.assertIn('source', pattern)
    
    def test_action_determination(self):
        """Test action determination from patterns"""
        test_patterns = [
            {'type': 'Cup and Handle'},
            {'type': 'Head and Shoulders'},
            {'type': 'Ascending Triangle'},
            {'type': 'Double Top'},
            {'type': 'Unknown Pattern'}
        ]
        
        expected_actions = ['buy', 'sell', 'buy', 'sell', 'hold']
        
        for pattern, expected in zip(test_patterns, expected_actions):
            action = self.bot._determine_action_from_pattern(pattern)
            self.assertEqual(action, expected)
    
    def test_pattern_caching(self):
        """Test pattern caching functionality"""
        symbol = 'TEST/USDT'
        
        # Initial analysis should cache results
        analysis = self.agent.analyze_patterns(symbol, self.sample_data, '1h')
        
        # Check cache exists
        self.assertIn(symbol, self.agent.pattern_cache)
        cached = self.agent.pattern_cache[symbol]
        
        self.assertIn('patterns', cached)
        self.assertIn('timestamp', cached)
        self.assertIn('timeframe', cached)
    
    def test_pattern_signals(self):
        """Test pattern signal generation"""
        symbol = 'TEST/USDT'
        
        # Add some test patterns to cache
        self.agent.pattern_cache[symbol] = {
            'patterns': [
                {'type': 'Cup and Handle', 'confidence': 0.8},
                {'type': 'Bullish Flag', 'confidence': 0.75}
            ],
            'timestamp': datetime.now(),
            'timeframe': '1h'
        }
        
        signals = self.agent.get_pattern_signals(symbol)
        
        self.assertIn('signal', signals)
        self.assertIn('patterns', signals)
        self.assertIn('bullish_score', signals)
        self.assertIn('bearish_score', signals)
        
        # With bullish patterns, signal should be buy
        self.assertEqual(signals['signal'], 'buy')
    
    def test_pattern_validation(self):
        """Test pattern validation"""
        pattern = {
            'stop_loss': 95,
            'target_price': 105
        }
        
        # Price within valid range
        self.assertTrue(self.agent.validate_pattern(pattern, 100))
        
        # Price below stop loss
        self.assertFalse(self.agent.validate_pattern(pattern, 94))
        
        # Price above target (exceeded)
        self.assertFalse(self.agent.validate_pattern(pattern, 116))
    
    def test_config_integration(self):
        """Test configuration integration"""
        from config import TradingConfig
        
        config = TradingConfig()
        
        # Check agent config is loaded
        self.assertTrue(hasattr(config, 'USE_PATTERN_AGENT'))
        self.assertTrue(hasattr(config, 'AGENT_MIN_CONFIDENCE'))
        self.assertTrue(hasattr(config, 'AGENT_MAX_PATTERNS'))
        self.assertTrue(hasattr(config, 'AGENT_CACHE_DURATION'))
    
    def test_web_interface_endpoints(self):
        """Test web interface agent endpoints"""
        from web_interface import app
        
        with app.test_client() as client:
            # Test agent status endpoint
            response = client.get('/api/agent/status')
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('agent_active', data)
            
            # Test patterns endpoint with agent support
            response = client.get('/api/patterns')
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('agent_patterns', data)
            self.assertIn('agent_enabled', data)

def run_tests():
    """Run all tests"""
    print("🧪 Running Quant Pattern Analyst Agent Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} tests failed")
        print(f"❌ {len(result.errors)} tests had errors")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)