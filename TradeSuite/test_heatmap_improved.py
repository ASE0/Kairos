#!/usr/bin/env python3
"""
Improved test script for heatmap functionality with real data structure
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

def create_realistic_result_data():
    """Create realistic result data that matches the actual structure"""
    
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
        },
        {
            'entry_time': start_time + timedelta(minutes=120),
            'exit_time': start_time + timedelta(minutes=135),
            'entry_price': 100.8,
            'exit_price': 101.5,
            'pnl': 0.7,
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
    # This matches the actual format from the backtest results
    fvg_signals = pd.Series([True if i % 20 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 15 == 0 else False for i in range(len(time_index))], index=time_index)
    ob_signals = pd.Series([True if i % 25 == 0 else False for i in range(len(time_index))], index=time_index)
    
    # Print the signal data for debugging
    print(f"[DEBUG] FVG signals: {fvg_signals.sum()} active out of {len(fvg_signals)}")
    print(f"[DEBUG] VWAP signals: {vwap_signals.sum()} active out of {len(vwap_signals)}")
    print(f"[DEBUG] OB signals: {ob_signals.sum()} active out of {len(ob_signals)}")
    print(f"[DEBUG] Sample FVG signal string:")
    print(str(fvg_signals)[:200] + "...")
    print(f"[DEBUG] Full FVG signal string:")
    print(str(fvg_signals))
    print(f"[DEBUG] Contains 'dtype: bool': {'dtype: bool' in str(fvg_signals)}")
    print(f"[DEBUG] Contains 'datetime': {'datetime' in str(fvg_signals)}")
    
    action_details = {
        'fvg': str(fvg_signals),  # String representation like in real results
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
        'component_summary': {
            'filters': ['vwap', 'volume', 'momentum'],
            'patterns': ['hammer', 'engulfing'],
            'locations': ['VWAP', 'Order Block', 'FVG'],
            'gates': ['location_gate', 'volatility_gate']
        },
        'total_trades': len(trades),
        'total_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'win_rate': 0.67,
        'equity_curve': [100000] + [100000 + i * 10 for i in range(1, len(time_index))],
        'timeframe': '1min',
        'interval': '1min',
        'result_display_name': 'Test_Multi_Strategy_20240101_090000'
    }
    
    return result_data

def test_heatmap():
    """Test the improved heatmap functionality"""
    print("Testing improved heatmap functionality...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Create realistic sample data
    sample_data = create_realistic_result_data()
    
    # Add the sample result
    results_viewer.add_result(sample_data, "Test_Multi_Strategy")
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened with realistic sample data.")
    print("Navigate to the 'Heatmap' tab to see the building blocks heatmap.")
    print("The heatmap should now show:")
    print("- FVG signals (every 20 minutes)")
    print("- VWAP signals (every 15 minutes)") 
    print("- Order Block signals (every 25 minutes)")
    print("- Trade entries/exits")
    print("- Zone activations")
    print("- Various filters and gates")
    print("")
    print("The heatmap should display as a density chart with:")
    print("- Darker colors when more sub-strategies fire simultaneously")
    print("- Lighter colors when fewer sub-strategies fire")
    print("- Different building blocks on the Y-axis")
    print("- Time periods on the X-axis")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_heatmap() 