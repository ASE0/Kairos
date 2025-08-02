"""
Test backtest window functionality
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import sys

# Create QApplication instance
app = QApplication(sys.argv)

from gui.backtest_window import BacktestWindow
from strategies.strategy_builders import (
    PatternStrategy, CombinedStrategy,
    Action, StrategyFactory
)
from strategies.risk_strategy import RiskStrategy
from patterns.candlestick_patterns import PatternFactory
from core.data_structures import TimeRange

class TestBacktestWindow(unittest.TestCase):
    """Test backtest window functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create QApplication instance
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()
    
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
        
        # Create backtest window
        self.window = BacktestWindow()
    
    def test_plot_candlesticks(self):
        """Test candlestick plotting with DataFrame action_details"""
        # Create test results
        signals, action_details = self.combined_strategy.evaluate(self.data)
        results = {
            'data': self.data,
            'action_details': action_details,
            'strategy_params': {},
            'component_summary': {},
            'trades': []
        }
        
        # Update chart tab
        self.window._update_chart_tab(results)
        
        # Check that no error occurred
        self.assertTrue(True)  # If we got here without error, test passed
    
    def test_plot_candlesticks_with_vwap(self):
        """Test candlestick plotting with VWAP in action_details"""
        # Create test results with VWAP
        signals, action_details = self.combined_strategy.evaluate(self.data)
        action_details['vwap'] = (self.data['close'] * self.data['volume']).cumsum() / self.data['volume'].cumsum()
        results = {
            'data': self.data,
            'action_details': action_details,
            'strategy_params': {},
            'component_summary': {},
            'trades': []
        }
        
        # Update chart tab
        self.window._update_chart_tab(results)
        
        # Check that no error occurred
        self.assertTrue(True)  # If we got here without error, test passed
    
    def test_plot_candlesticks_with_zones(self):
        """Test candlestick plotting with zones"""
        # Create test results with zones
        signals, action_details = self.combined_strategy.evaluate(self.data)
        zones = [
            {
                'zone_type': 'FVG',
                'zone_min': self.data['low'].min(),
                'zone_max': self.data['high'].max(),
                'zone_direction': 'bullish',
                'creation_index': 0,
                'initial_strength': 0.8,
                'gamma': 0.95,
                'tau_bars': 50,
                'drop_threshold': 0.01,
                'bar_interval_minutes': 1,
                'zone_days_valid': 1
            }
        ]
        results = {
            'data': self.data,
            'action_details': action_details,
            'strategy_params': {},
            'component_summary': {},
            'trades': [],
            'zones': zones
        }
        
        # Update chart tab
        self.window._update_chart_tab(results)
        
        # Check that no error occurred
        self.assertTrue(True)  # If we got here without error, test passed
    
    def tearDown(self):
        """Clean up after each test"""
        self.window.close()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        cls.app.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    unittest.main()
    app.quit()