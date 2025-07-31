#!/usr/bin/env python3
"""
Test script for heatmap functionality with real result data
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.results_viewer_window import ResultsViewerWindow
from PyQt6.QtWidgets import QApplication

def load_real_result_data():
    """Load real result data from the workspace"""
    
    # Try to load a real result file
    result_files = [
        "workspaces/results/fvg1mh5m/result_fvg1mh5m_1m_20250702_184105.json",
        "workspaces/results/vwap/result_vwap_1m_20250728_113714.json",
        "workspaces/results/FVGGG/result_FVGGG_1m_20250713_170233.json"
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                print(f"Loaded result data from: {file_path}")
                return result_data
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue
    
    print("No real result files found, creating sample data...")
    return create_sample_result_data()

def create_sample_result_data():
    """Create sample result data that matches the real format"""
    
    # Create sample time series data
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    end_time = datetime(2024, 1, 1, 16, 0, 0)
    time_index = pd.date_range(start=start_time, end=end_time, freq='5min')
    
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
    
    # Create realistic action_details with pandas Series string representation
    fvg_signals = pd.Series([True if i % 4 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 3 == 0 else False for i in range(len(time_index))], index=time_index)
    ob_signals = pd.Series([True if i % 5 == 0 else False for i in range(len(time_index))], index=time_index)
    
    action_details = {
        'fvg': str(fvg_signals),
        'vwap': str(vwap_signals),
        'order_block': str(ob_signals)
    }
    
    # Create sample result data
    result_data = {
        'strategy_name': 'Test_Multi_Strategy',
        'data': data,
        'trades': trades,
        'zones': zones,
        'action_details': action_details,
        'strategy_params': {
            'filters': [
                {'type': 'vwap'},
                {'type': 'volume'},
                {'type': 'momentum'}
            ],
            'location_strategies': ['VWAP', 'Order Block', 'FVG']
        },
        'gates_enabled': {
            'location_gate': True,
            'volatility_gate': True,
            'momentum_gate': False
        },
        'total_trades': len(trades),
        'total_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'win_rate': 0.67,
        'equity_curve': [100000] + [100000 + i * 10 for i in range(1, len(time_index))],
        'timeframe': '5min',
        'interval': '5min',
        'result_display_name': 'Test_Multi_Strategy_20240101_090000'
    }
    
    return result_data

def test_heatmap():
    """Test the heatmap functionality with real data"""
    print("Testing heatmap functionality with real data...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Load real result data
    result_data = load_real_result_data()
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Test_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened with real data.")
    print("Navigate to the 'Heatmap' tab to see the building blocks heatmap.")
    print("The heatmap should show:")
    print("- Individual building blocks on the Y-axis")
    print("- Time periods on the X-axis")
    print("- Color intensity representing signal density")
    print("- Darker colors when more sub-strategies fire simultaneously")
    print("- Lighter colors when fewer sub-strategies fire")
    print("")
    print("You can adjust:")
    print("- Time binning (1min, 5min, 15min, etc.)")
    print("- Color scheme (Viridis, Plasma, etc.)")
    print("- Show/hide legend")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_heatmap() 