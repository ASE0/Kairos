#!/usr/bin/env python3
"""
Test script to simulate tick frequency strategy and debug heatmap
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

def create_tick_frequency_data():
    """Create data that simulates your tick frequency strategy"""
    
    # Create time series for a full day
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    end_time = datetime(2024, 1, 1, 16, 0, 0)  # 7 hours of trading
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': [100 + i * 0.01 for i in range(len(time_index))],
        'high': [100.5 + i * 0.01 for i in range(len(time_index))],
        'low': [99.5 + i * 0.01 for i in range(len(time_index))],
        'close': [100.2 + i * 0.01 for i in range(len(time_index))],
        'volume': [1000 + i for i in range(len(time_index))]
    }, index=time_index)
    
    # Create tick frequency signals (mmr building block)
    # Simulate tick frequency filter signals
    mmr_signals = pd.Series([True if i % 10 == 0 else False for i in range(len(time_index))], index=time_index)
    
    # Create some trades
    trades = [
        {
            'entry_time': start_time + timedelta(minutes=15),
            'exit_time': start_time + timedelta(minutes=25),
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'size': 1.0
        },
        {
            'entry_time': start_time + timedelta(minutes=45),
            'exit_time': start_time + timedelta(minutes=55),
            'entry_price': 101.0,
            'exit_price': 101.8,
            'pnl': 0.8,
            'size': 1.0
        }
    ]
    
    # Create result data that matches your strategy
    result_data = {
        'strategy_name': 'Tick_Frequency_Strategy',
        'data': data,
        'trades': trades,
        'zones': [],
        'action_details': {
            'mmr': mmr_signals  # This is a pandas Series, not a string
        },
        'strategy_params': {
            'bar_interval_minutes': 1,
            'tick_frequency_threshold': 100,
            'tick_frequency_enabled': True,
            'filters': [{'type': 'tick_frequency'}],
            'location_strategies': ['Tick_Frequency']
        },
        'gates_enabled': {
            'tick_validation_gate': True,
            'location_gate': True
        },
        'total_trades': len(trades),
        'total_return': 0.15,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.03,
        'win_rate': 0.75,
        'equity_curve': [100000] + [100000 + i * 0.1 for i in range(1, len(time_index))],
        'timeframe': '1min',
        'interval': '1min',
        'result_display_name': 'Tick_Frequency_Strategy_20240101_090000'
    }
    
    return result_data

def test_tick_frequency_heatmap():
    """Test heatmap with tick frequency strategy"""
    print("Testing heatmap with tick frequency strategy...")
    
    # Create tick frequency data
    result_data = create_tick_frequency_data()
    
    print(f"Created tick frequency data with {len(result_data['data'])} time periods")
    print(f"Action details keys: {list(result_data['action_details'].keys())}")
    print(f"mmr signals: {result_data['action_details']['mmr'].sum()} active signals")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Tick_Frequency_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened.")
    print("Navigate to the 'Heatmap' tab.")
    print("You should see:")
    print("- mmr building block (tick frequency filter)")
    print("- Clear signal pattern every 10 minutes")
    print("- Color intensity showing signal density")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_tick_frequency_heatmap() 