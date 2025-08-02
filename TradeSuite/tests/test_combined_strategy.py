"""
Test combined strategy implementation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.strategy_builders import (
    PatternStrategy, RiskStrategy, CombinedStrategy,
    Action, StrategyFactory
)
from patterns.candlestick_patterns import PatternFactory
from core.data_structures import TimeRange

class TestCombinedStrategy(unittest.TestCase):
    """Test combined strategy functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        self.data = pd.DataFrame({
            'open': np.random.randn(100).cumsum(),
            'high': None,
            'low': None,
            'close': None,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        # Set high/low based on open
        self.data['close'] = self.data['open'] + np.random.randn(100) * 0.1
        self.data['high'] = np.maximum(self.data['open'], self.data['close']) + np.abs(np.random.randn(100) * 0.05)
        self.data['low'] = np.minimum(self.data['open'], self.data['close']) - np.abs(np.random.randn(100) * 0.05)
        
        # Create FVG pattern
        self.pattern = PatternFactory.create_pattern('FVG', timeframes=[TimeRange(1, 'minute')])
        
        # Create pattern strategy
        self.pattern_strategy = PatternStrategy(
            actions=[Action(pattern=self.pattern)],
            combination_logic='AND',
            min_actions_required=1
        )
        
        # Create risk strategy
        self.risk_strategy = RiskStrategy(
            risk_per_trade=0.02,
            stop_loss_atr=2.0,
            take_profit_atr=3.0,
            trailing_stop=False,
            trailing_stop_atr=1.5
        )
        
        # Create combined strategy
        self.combined_strategy = CombinedStrategy(
            pattern_strategy=self.pattern_strategy,
            risk_strategy=self.risk_strategy
        )
    
    def test_risk_parameters(self):
        """Test risk parameters are properly copied"""
        self.assertEqual(self.combined_strategy.risk_per_trade, 0.02)
        self.assertEqual(self.combined_strategy.stop_loss_atr, 2.0)
        self.assertEqual(self.combined_strategy.take_profit_atr, 3.0)
        self.assertEqual(self.combined_strategy.trailing_stop, False)
        self.assertEqual(self.combined_strategy.trailing_stop_atr, 1.5)
    
    def test_evaluate(self):
        """Test strategy evaluation"""
        signals, action_details = self.combined_strategy.evaluate(self.data)
        
        # Check signals
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Check action details
        self.assertIsInstance(action_details, pd.DataFrame)
        self.assertEqual(len(action_details), len(self.data))
        
        # Check risk columns are present
        self.assertIn('atr', action_details.columns)
        self.assertIn('stop_loss_atr', action_details.columns)
        self.assertIn('take_profit_atr', action_details.columns)
        self.assertIn('risk_per_trade', action_details.columns)
        self.assertIn('trailing_stop', action_details.columns)
        self.assertIn('trailing_stop_atr', action_details.columns)
        
        # Check risk values
        self.assertEqual(action_details['stop_loss_atr'].iloc[0], 2.0)
        self.assertEqual(action_details['take_profit_atr'].iloc[0], 3.0)
        self.assertEqual(action_details['risk_per_trade'].iloc[0], 0.02)
        self.assertEqual(action_details['trailing_stop'].iloc[0], False)
        self.assertEqual(action_details['trailing_stop_atr'].iloc[0], 1.5)
    
    def test_zones(self):
        """Test zone detection and properties"""
        signals, action_details = self.combined_strategy.evaluate(self.data)
        
        # Get zones
        zones = getattr(self.combined_strategy, 'simple_zones', [])
        
        # Check FVG zones
        fvg_zones = [z for z in zones if z.get('zone_type', '').lower() == 'fvg']
        for zone in fvg_zones:
            # Check required properties
            self.assertIn('zone_min', zone)
            self.assertIn('zone_max', zone)
            self.assertIn('zone_direction', zone)
            self.assertIn('creation_index', zone)
            self.assertIn('initial_strength', zone)
            self.assertIn('gamma', zone)
            self.assertIn('tau_bars', zone)
            self.assertIn('drop_threshold', zone)
            
            # Check values are reasonable
            self.assertGreater(zone['zone_max'], zone['zone_min'])
            self.assertIn(zone['zone_direction'], ['bullish', 'bearish'])
            self.assertGreaterEqual(zone['creation_index'], 0)
            self.assertLess(zone['creation_index'], len(self.data))
            self.assertGreater(zone['initial_strength'], 0)
            self.assertLess(zone['initial_strength'], 1)
            self.assertGreater(zone['gamma'], 0)
            self.assertLess(zone['gamma'], 1)
            self.assertGreater(zone['tau_bars'], 0)
            self.assertGreater(zone['drop_threshold'], 0)
            self.assertLess(zone['drop_threshold'], 1)

if __name__ == '__main__':
    unittest.main()