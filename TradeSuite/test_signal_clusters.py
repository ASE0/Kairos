#!/usr/bin/env python3
"""
Test script to verify signal generation and trade entry logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_signal_clusters():
    """Test how signal clusters affect trade entry"""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.randn(1000) + 100,
        'high': np.random.randn(1000) + 101,
        'low': np.random.randn(1000) + 99,
        'close': np.random.randn(1000) + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Create sample signals with clusters
    signals = pd.Series(False, index=dates)
    
    # Create signal clusters (consecutive True values)
    signal_clusters = [
        (10, 15),   # Bar 10-15: cluster 1
        (50, 55),   # Bar 50-55: cluster 2  
        (100, 105), # Bar 100-105: cluster 3
        (200, 210), # Bar 200-210: cluster 4
        (300, 305), # Bar 300-305: cluster 5
    ]
    
    for start, end in signal_clusters:
        signals.iloc[start:end+1] = True
    
    print(f"Created {len(signal_clusters)} signal clusters")
    print(f"Total signals: {signals.sum()}")
    
    # Test rising edge detection
    prev_signal = False
    trades = []
    position = None
    
    for i in range(len(data)):
        current_signal = signals.iloc[i]
        
        # Only enter on rising edge of signal
        if not position and current_signal and not prev_signal:
            entry_price = data.iloc[i]['close']
            print(f"[TRADE] Opened trade at bar {i}: price={entry_price:.2f}")
            position = {
                'entry_time': data.index[i],
                'entry_index': i,
                'entry_price': entry_price
            }
            trades.append(position)
            position = None  # Close immediately for testing
        
        prev_signal = current_signal
    
    print(f"Total trades opened: {len(trades)}")
    print(f"Expected trades: {len(signal_clusters)}")
    
    if len(trades) == len(signal_clusters):
        print("✅ SUCCESS: Rising edge detection works correctly")
    else:
        print("❌ FAILURE: Rising edge detection is not working")
        print(f"Expected {len(signal_clusters)} trades, got {len(trades)}")

if __name__ == "__main__":
    test_signal_clusters() 