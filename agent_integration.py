"""
Quant Pattern Analyst Agent Integration Module

This module integrates the quant-pattern-analyst agent into the trading system.
The agent specializes in identifying, analyzing, and presenting classical chart patterns
in financial data including:
- Head and Shoulders (reversal pattern)
- Cup and Handle (continuation pattern)
- Double Tops/Bottoms (reversal patterns)
- Triangles (Ascending/Descending/Symmetrical)
- Flags and Pennants (continuation patterns)
- Wedges (Rising/Falling)

The agent provides:
- Pattern detection with confidence scores
- Pattern measurements and projections
- Entry/exit points based on pattern completion
- Risk/reward ratios for pattern-based trades
"""

import subprocess
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

class QuantPatternAnalystAgent:
    """
    Integration wrapper for the quant-pattern-analyst agent.
    Provides pattern detection and analysis capabilities for trading decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Quant Pattern Analyst Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.agent_name = "quant-pattern-analyst"
        self.pattern_cache = {}
        self.last_analysis_time = None
        self.min_pattern_confidence = self.config.get('min_confidence', 0.7)
        
    def analyze_patterns(self, symbol: str, df: pd.DataFrame, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze price data for classical chart patterns using the agent.
        
        Args:
            symbol: Trading symbol to analyze
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary containing detected patterns and analysis
        """
        try:
            # Prepare data for agent analysis
            analysis_request = {
                "symbol": symbol,
                "timeframe": timeframe,
                "data_points": len(df),
                "price_range": {
                    "high": float(df['High'].max()),
                    "low": float(df['Low'].min()),
                    "current": float(df['Close'].iloc[-1])
                }
            }
            
            # Call the agent via subprocess
            result = self._invoke_agent(
                task="pattern_detection",
                data=analysis_request,
                price_data=df.to_json(orient='records')
            )
            
            # Parse and enhance results
            patterns = self._parse_agent_response(result)
            
            # Add confidence scoring and filtering
            filtered_patterns = [
                p for p in patterns 
                if p.get('confidence', 0) >= self.min_pattern_confidence
            ]
            
            # Cache results
            self.pattern_cache[symbol] = {
                'patterns': filtered_patterns,
                'timestamp': datetime.now(),
                'timeframe': timeframe
            }
            
            return {
                'symbol': symbol,
                'patterns': filtered_patterns,
                'analysis_time': datetime.now().isoformat(),
                'summary': self._generate_pattern_summary(filtered_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns for {symbol}: {e}")
            return {'symbol': symbol, 'patterns': [], 'error': str(e)}
    
    def _invoke_agent(self, task: str, data: Dict, price_data: str) -> str:
        """
        Invoke the quant-pattern-analyst agent via Claude CLI.
        
        Args:
            task: Task type for the agent
            data: Analysis parameters
            price_data: JSON string of price data
            
        Returns:
            Agent response string
        """
        try:
            # Build the agent prompt
            prompt = f"""
            Analyze the following price data for {data['symbol']} on {data['timeframe']} timeframe:
            
            Data points: {data['data_points']}
            Price range: High={data['price_range']['high']}, Low={data['price_range']['low']}, Current={data['price_range']['current']}
            
            Task: Identify all classical chart patterns including:
            - Head and Shoulders patterns
            - Cup and Handle formations
            - Double Tops/Bottoms
            - Triangle patterns (ascending, descending, symmetrical)
            - Flag and Pennant patterns
            - Wedge patterns (rising, falling)
            
            For each pattern found, provide:
            1. Pattern type and direction
            2. Confidence score (0-1)
            3. Key price levels (neckline, support, resistance)
            4. Pattern measurements and projections
            5. Suggested entry and exit points
            6. Risk/reward ratio
            
            Price data: {price_data[:1000]}...
            """
            
            # Execute agent command
            cmd = [
                "claude", "agents", "run",
                self.agent_name,
                "--task", task,
                "--prompt", prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Agent invocation failed: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            logger.error("Agent invocation timed out")
            return ""
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            return ""
    
    def _parse_agent_response(self, response: str) -> List[Dict]:
        """
        Parse the agent's response into structured pattern data.
        
        Args:
            response: Raw agent response string
            
        Returns:
            List of detected patterns with details
        """
        patterns = []
        
        try:
            # Parse response (assuming JSON format from agent)
            # In production, this would parse the actual agent output format
            lines = response.strip().split('\n')
            current_pattern = {}
            
            for line in lines:
                if 'Pattern:' in line:
                    if current_pattern:
                        patterns.append(current_pattern)
                    current_pattern = {'type': line.split('Pattern:')[1].strip()}
                elif 'Confidence:' in line:
                    current_pattern['confidence'] = float(line.split(':')[1].strip())
                elif 'Entry:' in line:
                    current_pattern['entry_price'] = float(line.split(':')[1].strip())
                elif 'Target:' in line:
                    current_pattern['target_price'] = float(line.split(':')[1].strip())
                elif 'Stop:' in line:
                    current_pattern['stop_loss'] = float(line.split(':')[1].strip())
                elif 'Risk/Reward:' in line:
                    current_pattern['risk_reward'] = float(line.split(':')[1].strip())
            
            if current_pattern:
                patterns.append(current_pattern)
                
        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
        
        return patterns
    
    def _generate_pattern_summary(self, patterns: List[Dict]) -> str:
        """
        Generate a summary of detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Summary string
        """
        if not patterns:
            return "No high-confidence patterns detected"
        
        pattern_types = [p['type'] for p in patterns]
        avg_confidence = sum(p.get('confidence', 0) for p in patterns) / len(patterns)
        
        bullish_patterns = sum(1 for p in patterns if 'bullish' in p['type'].lower())
        bearish_patterns = sum(1 for p in patterns if 'bearish' in p['type'].lower())
        
        summary = f"Detected {len(patterns)} patterns with avg confidence {avg_confidence:.2f}. "
        summary += f"Bullish: {bullish_patterns}, Bearish: {bearish_patterns}"
        
        return summary
    
    def get_pattern_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading signals based on detected patterns.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Trading signals based on pattern analysis
        """
        if symbol not in self.pattern_cache:
            return {'signal': 'neutral', 'patterns': []}
        
        cached_data = self.pattern_cache[symbol]
        patterns = cached_data['patterns']
        
        if not patterns:
            return {'signal': 'neutral', 'patterns': []}
        
        # Determine overall signal based on patterns
        bullish_score = sum(
            p.get('confidence', 0) 
            for p in patterns 
            if 'bullish' in p['type'].lower()
        )
        bearish_score = sum(
            p.get('confidence', 0) 
            for p in patterns 
            if 'bearish' in p['type'].lower()
        )
        
        if bullish_score > bearish_score * 1.5:
            signal = 'buy'
        elif bearish_score > bullish_score * 1.5:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'patterns': patterns,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'timestamp': cached_data['timestamp']
        }
    
    def validate_pattern(self, pattern: Dict, current_price: float) -> bool:
        """
        Validate if a pattern is still valid based on current price.
        
        Args:
            pattern: Pattern dictionary
            current_price: Current market price
            
        Returns:
            True if pattern is still valid
        """
        try:
            # Check if price hasn't violated pattern boundaries
            if 'stop_loss' in pattern and current_price < pattern['stop_loss']:
                return False
            
            # Check if pattern hasn't exceeded projection
            if 'target_price' in pattern and current_price > pattern['target_price'] * 1.1:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating pattern: {e}")
            return False
    
    def get_pattern_history(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get historical pattern detections for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of history to retrieve
            
        Returns:
            List of historical patterns
        """
        # In production, this would query a database
        # For now, return cached data if available
        if symbol in self.pattern_cache:
            return self.pattern_cache[symbol]['patterns']
        return []