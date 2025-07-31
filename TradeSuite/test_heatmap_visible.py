#!/usr/bin/env python3
"""
Test script to verify heatmap is visible with obvious patterns
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

def create_obvious_pattern_data():
    """Create data with very obvious signal patterns"""
    
    # Create simple time series
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    end_time = datetime(2024, 1, 1, 10, 0, 0)  # Just 1 hour
    time_index = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # Create simple OHLC data
    data = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(len(time_index))],
        'high': [100.5 + i * 0.1 for i in range(len(time_index))],
        'low': [99.5 + i * 0.1 for i in range(len(time_index))],
        'close': [100.2 + i * 0.1 for i in range(len(time_index))],
        'volume': [1000 + i * 10 for i in range(len(time_index))]
    }, index=time_index)
    
    # Create very obvious signal patterns
    # fvg: every 2nd period
    # vwap: every 3rd period  
    # order_block: every 4th period
    fvg_signals = pd.Series([True if i % 2 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 3 == 0 else False for i in range(len(time_index))], index=time_index)
    ob_signals = pd.Series([True if i % 4 == 0 else False for i in range(len(time_index))], index=time_index)
    
    action_details = {
        'fvg': str(fvg_signals),
        'vwap': str(vwap_signals),
        'order_block': str(ob_signals)
    }
    
    # Create simple result data
    result_data = {
        'strategy_name': 'Obvious_Pattern_Strategy',
        'data': data,
        'trades': [],
        'zones': [],
        'action_details': action_details,
        'strategy_params': {
            'filters': [{'type': 'vwap'}],
            'location_strategies': ['VWAP', 'FVG', 'Order Block']
        },
        'gates_enabled': {
            'location_gate': True
        },
        'total_trades': 0,
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'equity_curve': [100000] * len(time_index),
        'timeframe': '5min',
        'interval': '5min',
        'result_display_name': 'Obvious_Pattern_Strategy_20240101_090000'
    }
    
    return result_data

def test_heatmap_visibility():
    """Test heatmap visibility with obvious patterns"""
    print("Testing heatmap visibility with obvious patterns...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Create obvious pattern data
    result_data = create_obvious_pattern_data()
    
    print(f"Created data with {len(result_data['data'])} time periods")
    print(f"Signal patterns:")
    print(f"- fvg: every 2nd period (should be very visible)")
    print(f"- vwap: every 3rd period (should be visible)")
    print(f"- order_block: every 4th period (should be visible)")
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Obvious_Pattern_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened.")
    print("Navigate to the 'Heatmap' tab.")
    print("You should see a clear pattern:")
    print("- fvg: dense pattern (every 2nd period)")
    print("- vwap: medium pattern (every 3rd period)")
    print("- order_block: sparse pattern (every 4th period)")
    print("If you don't see anything, there's a display issue.")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_heatmap_visibility() 