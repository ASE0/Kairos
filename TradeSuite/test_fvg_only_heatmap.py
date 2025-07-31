#!/usr/bin/env python3
"""
Test script to verify heatmap shows only real FVG signals
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_fvg_only_test_data():
    """Create test data with only FVG signals"""
    
    # Create realistic time index (1-minute intervals for 4 hours)
    start_time = datetime(2024, 1, 1, 9, 0, 0)  # 9:00 AM
    time_index = pd.date_range(start=start_time, periods=240, freq='1min')  # 4 hours
    
    # Create OHLC data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(240) * 0.1)
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(240) * 0.05,
        'high': close_prices + np.abs(np.random.randn(240) * 0.1),
        'low': close_prices - np.abs(np.random.randn(240) * 0.1),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 240)
    }, index=time_index)
    
    # Create FVG signals at specific times (realistic pattern)
    fvg_signals = pd.Series(False, index=time_index)
    
    # Add FVG signals at specific times (every 15-20 minutes)
    fvg_times = [
        start_time + timedelta(minutes=15),
        start_time + timedelta(minutes=32),
        start_time + timedelta(minutes=47),
        start_time + timedelta(minutes=65),
        start_time + timedelta(minutes=82),
        start_time + timedelta(minutes=98),
        start_time + timedelta(minutes=115),
        start_time + timedelta(minutes=132),
        start_time + timedelta(minutes=148),
        start_time + timedelta(minutes=165),
        start_time + timedelta(minutes=182),
        start_time + timedelta(minutes=198),
        start_time + timedelta(minutes=215),
        start_time + timedelta(minutes=232)
    ]
    
    for fvg_time in fvg_times:
        if fvg_time in time_index:
            fvg_signals.loc[fvg_time] = True
    
    # Create trades that align with some FVG signals
    trades = []
    for i, fvg_time in enumerate(fvg_times[::2]):  # Every other FVG signal
        if fvg_time in time_index:
            exit_time = fvg_time + timedelta(minutes=5)
            trades.append({
                'entry_time': fvg_time,
                'exit_time': exit_time,
                'entry_price': data.loc[fvg_time, 'close'],
                'exit_price': data.loc[fvg_time, 'close'] + np.random.randn() * 0.5,
                'pnl': np.random.randn() * 0.5,
                'size': 1
            })
    
    # Create result data with ONLY FVG in action_details
    result_data = {
        'strategy_name': 'FVG_Only_Strategy',
        'data': data,
        'trades': trades,
        'zones': [],
        'action_details': {
            'fvg': fvg_signals  # ONLY FVG signals
        },
        'strategy_params': {
            'fvg_epsilon': 0.001,
            'fvg_N': 10,
            'fvg_sigma': 0.5
        },
        'gates_enabled': {
            'fvg_gate': True,
            'location_gate': False,
            'volatility_gate': False
        },
        'total_trades': len(trades),
        'total_return': 2.1,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.05,
        'win_rate': 0.75,
        'equity_curve': pd.Series(np.cumsum([0.1, -0.05, 0.2, -0.1, 0.15]), index=time_index[:5]),
        'timeframe': '1m',
        'interval': '1min',
        'result_display_name': 'FVG Only Test'
    }
    
    return result_data

def test_fvg_only_heatmap():
    """Test the heatmap with FVG-only strategy"""
    
    print("=== TESTING FVG-ONLY HEATMAP ===")
    
    # Create test data
    result_data = create_fvg_only_test_data()
    
    print(f"Created strategy with:")
    print(f"  - {len(result_data['data'])} time periods")
    print(f"  - {len(result_data['action_details'])} action_details building blocks")
    print(f"  - {len(result_data['trades'])} trades")
    print(f"  - Action details keys: {list(result_data['action_details'].keys())}")
    
    # Test the GUI
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.results_viewer_window import ResultsViewerWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create results viewer
        viewer = ResultsViewerWindow()
        
        # Add the result
        viewer.add_result(result_data, "FVG_Only_Strategy")
        
        # Show the window
        viewer.show()
        
        print("✅ FVG-ONLY HEATMAP TEST COMPLETE")
        print("Expected heatmap should show:")
        print("✅ ONLY FVG building block (no synthetic data)")
        print("✅ Real FVG signals at specific times")
        print("✅ Proper timestamps on X-axis")
        print("✅ Correct signal density visualization")
        print("Navigate to the 'Heatmap' tab to see the FVG-only view!")
        
        # Keep the window open for a few seconds
        import time
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Error testing FVG-only heatmap: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fvg_only_heatmap() 