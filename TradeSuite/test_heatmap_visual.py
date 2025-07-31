#!/usr/bin/env python3
"""
Simple visual test for the heatmap functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.results_viewer_window import ResultsViewerWindow
from PyQt6.QtWidgets import QApplication

def create_simple_test_data():
    """Create simple test data to verify heatmap visualization"""
    
    # Create sample time series data
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    end_time = datetime(2024, 1, 1, 16, 0, 0)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create sample OHLC data
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.1, len(time_index))
    prices = base_price + np.cumsum(price_changes)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 0.5, len(time_index)),
        'low': prices - np.random.uniform(0, 0.5, len(time_index)),
        'close': prices + np.random.normal(0, 0.1, len(time_index)),
        'volume': np.random.randint(100, 1000, len(time_index))
    }, index=time_index)
    
    # Create sample trades
    trades = [
        {
            'entry_time': start_time + timedelta(minutes=10),
            'exit_time': start_time + timedelta(minutes=25),
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'size': 1.0
        },
        {
            'entry_time': start_time + timedelta(minutes=45),
            'exit_time': start_time + timedelta(minutes=60),
            'entry_price': 101.0,
            'exit_price': 100.8,
            'pnl': -0.2,
            'size': 1.0
        }
    ]
    
    # Create sample zones
    zones = [
        {
            'timestamp': start_time + timedelta(minutes=5),
            'zone_min': 100.0,
            'zone_max': 101.0,
            'direction': 'bullish'
        },
        {
            'timestamp': start_time + timedelta(minutes=40),
            'zone_min': 100.5,
            'zone_max': 101.5,
            'direction': 'bearish'
        }
    ]
    
    # Create action_details with clear signal patterns
    fvg_signals = pd.Series([True if i % 20 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 15 == 0 else False for i in range(len(time_index))], index=time_index)
    ob_signals = pd.Series([True if i % 25 == 0 else False for i in range(len(time_index))], index=time_index)
    
    action_details = {
        'fvg': str(fvg_signals),
        'vwap': str(vwap_signals),
        'order_block': str(ob_signals)
    }
    
    # Create result data
    result_data = {
        'strategy_name': 'Simple_Test_Strategy',
        'data': data,
        'trades': trades,
        'zones': zones,
        'action_details': action_details,
        'strategy_params': {
            'filters': [
                {'type': 'vwap'},
                {'type': 'volume'}
            ],
            'location_strategies': ['VWAP', 'Order Block', 'FVG']
        },
        'gates_enabled': {
            'location_gate': True,
            'volatility_gate': True
        },
        'total_trades': len(trades),
        'total_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'win_rate': 0.67,
        'equity_curve': [100000] + [100000 + i * 10 for i in range(1, len(time_index))],
        'timeframe': '1min',
        'interval': '1min',
        'result_display_name': 'Simple_Test_Strategy_20240101_090000'
    }
    
    return result_data

def test_heatmap_visual():
    """Test the heatmap visualization"""
    print("Testing heatmap visualization...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Create simple test data
    result_data = create_simple_test_data()
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Test_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened with simple test data.")
    print("Navigate to the 'Heatmap' tab to see the building blocks heatmap.")
    print("")
    print("Expected visualization:")
    print("- 3 building blocks: fvg, vwap, order_block")
    print("- fvg signals every 20 minutes")
    print("- vwap signals every 15 minutes") 
    print("- order_block signals every 25 minutes")
    print("- Clear signal patterns should be visible")
    print("- Darker colors when multiple signals fire together")
    print("")
    print("Test the controls:")
    print("- Change time binning (1min, 5min, 15min)")
    print("- Change color scheme (Viridis, Plasma, etc.)")
    print("- Toggle legend on/off")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_heatmap_visual() 