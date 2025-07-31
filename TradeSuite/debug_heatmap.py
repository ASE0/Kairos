#!/usr/bin/env python3
"""
Simple debug script to test heatmap functionality directly
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

def create_debug_data():
    """Create very simple debug data"""
    
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
    
    # Create simple trades
    trades = [
        {
            'entry_time': start_time + timedelta(minutes=10),
            'exit_time': start_time + timedelta(minutes=25),
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'size': 1.0
        }
    ]
    
    # Create simple zones
    zones = [
        {
            'timestamp': start_time + timedelta(minutes=5),
            'zone_min': 100.0,
            'zone_max': 101.0,
            'direction': 'bullish'
        }
    ]
    
    # Create very simple action_details with clear patterns
    fvg_signals = pd.Series([True if i % 3 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 4 == 0 else False for i in range(len(time_index))], index=time_index)
    
    action_details = {
        'fvg': str(fvg_signals),
        'vwap': str(vwap_signals)
    }
    
    # Create simple result data
    result_data = {
        'strategy_name': 'Debug_Strategy',
        'data': data,
        'trades': trades,
        'zones': zones,
        'action_details': action_details,
        'strategy_params': {
            'filters': [{'type': 'vwap'}],
            'location_strategies': ['VWAP', 'FVG']
        },
        'gates_enabled': {
            'location_gate': True
        },
        'total_trades': len(trades),
        'total_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'win_rate': 0.67,
        'equity_curve': [100000] + [100000 + i * 10 for i in range(1, len(time_index))],
        'timeframe': '5min',
        'interval': '5min',
        'result_display_name': 'Debug_Strategy_20240101_090000'
    }
    
    return result_data

def test_heatmap_directly():
    """Test heatmap directly"""
    print("Testing heatmap directly...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Create debug data
    result_data = create_debug_data()
    
    print(f"Debug data created with {len(result_data['data'])} time periods")
    print(f"Action details keys: {list(result_data['action_details'].keys())}")
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Debug_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened.")
    print("Navigate to the 'Heatmap' tab.")
    print("You should see:")
    print("- 2 building blocks: fvg, vwap")
    print("- fvg signals every 3 periods")
    print("- vwap signals every 4 periods")
    print("- Clear pattern in the heatmap")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_heatmap_directly() 