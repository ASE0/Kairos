#!/usr/bin/env python3
"""
Test script for the simplified heatmap functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_data():
    """Create realistic test data with FVG signals and trades"""
    print("[TEST] Creating test data...")
    
    # Create time index (2 hours of 1-minute data)
    start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=2)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create OHLC data
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    prices = []
    current_price = base_price
    
    for i in range(len(time_index)):
        # Random walk with slight upward bias
        change = np.random.normal(0, 0.1) + 0.001
        current_price += change
        
        # Create OHLC for this minute
        high = current_price + abs(np.random.normal(0, 0.05))
        low = current_price - abs(np.random.normal(0, 0.05))
        close = current_price + np.random.normal(0, 0.02)
        
        prices.append({
            'open': current_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 5000)
        })
        current_price = close
    
    # Create DataFrame
    data = pd.DataFrame(prices, index=time_index)
    print(f"[TEST] Created OHLC data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    # Create FVG signals (Fair Value Gaps)
    fvg_signals = pd.Series(False, index=time_index)
    
    # Add some FVG signals at specific times
    fvg_times = [
        start_time + timedelta(minutes=15),  # 9:45
        start_time + timedelta(minutes=45),  # 10:15
        start_time + timedelta(minutes=75),  # 10:45
        start_time + timedelta(minutes=105), # 11:15
    ]
    
    for fvg_time in fvg_times:
        if fvg_time in time_index:
            fvg_signals.loc[fvg_time] = True
    
    print(f"[TEST] Created FVG signals: {fvg_signals.sum()} signals at {fvg_times}")
    
    # Create trades that align with some FVG signals
    trades = []
    trade_times = [
        start_time + timedelta(minutes=16),  # Shortly after first FVG
        start_time + timedelta(minutes=76),  # Shortly after third FVG
    ]
    
    for i, trade_time in enumerate(trade_times):
        # Find closest time index
        closest_idx = data.index.get_indexer([trade_time], method='nearest')[0]
        closest_time = data.index[closest_idx]
        entry_price = data.loc[closest_time, 'open']
        
        exit_time = trade_time + timedelta(minutes=10)
        if exit_time in data.index:
            exit_price = data.loc[exit_time, 'close']
        else:
            # Find closest exit time
            exit_closest_idx = data.index.get_indexer([exit_time], method='nearest')[0]
            exit_closest_time = data.index[exit_closest_idx]
            exit_price = data.loc[exit_closest_time, 'close']
        
        trades.append({
            'entry_time': trade_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': exit_price - entry_price,
            'direction': 'long'
        })
    
    print(f"[TEST] Created {len(trades)} trades at {[t['entry_time'] for t in trades]}")
    
    # Create result_data structure
    result_data = {
        'data': data,
        'trades': trades,
        'action_details': {
            'FVG': fvg_signals  # ONLY FVG signals
        },
        'strategy_name': 'FVG-Only Strategy',
        'timeframe': '1min',
        'total_trades': len(trades),
        'win_rate': 100.0,  # Assume all profitable for test
        'total_return': sum(t['pnl'] for t in trades),
    }
    
    # Debug: Print what we're actually creating
    print(f"[TEST] action_details keys: {list(result_data['action_details'].keys())}")
    print(f"[TEST] FVG signals: {result_data['action_details']['FVG'].sum()} total signals")
    print(f"[TEST] FVG signal times: {result_data['action_details']['FVG'][result_data['action_details']['FVG'] == True].index.tolist()}")
    
    print(f"[TEST] Result data created with keys: {list(result_data.keys())}")
    print(f"[TEST] action_details keys: {list(result_data['action_details'].keys())}")
    print(f"[TEST] FVG signals: {result_data['action_details']['FVG'].sum()} total signals")
    
    return result_data

def test_heatmap():
    """Test the heatmap with realistic data"""
    from PyQt6.QtWidgets import QApplication
    from gui.results_viewer_window import ResultsViewerWindow
    
    # Create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create test data
    result_data = create_test_data()
    
    # Create results viewer window
    print("[TEST] Creating ResultsViewerWindow...")
    viewer = ResultsViewerWindow()
    
    # Add the result and display it
    print("[TEST] Adding result data...")
    viewer.add_result(result_data, "FVG-Only Test Strategy")
    
    # Show the window
    print("[TEST] Showing window...")
    viewer.show()
    viewer.resize(1200, 800)
    
    print("[TEST] Test complete! The heatmap should show:")
    print("  - Only FVG building block (no other components)")
    print("  - Signals at the correct times (9:45, 10:15, 10:45, 11:15)")
    print("  - Trades at different times (9:46, 10:16)")
    print("  - Time axis horizontal, building blocks vertical")
    print("  - White/bright colors where signals occurred")
    
    return app, viewer

if __name__ == "__main__":
    app, viewer = test_heatmap()
    
    # Keep the application running
    print("\n[TEST] Press Ctrl+C to exit...")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[TEST] Exiting...")
        sys.exit(0)