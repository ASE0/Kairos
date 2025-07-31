#!/usr/bin/env python3
"""
Test script for the statistics analyzer with timeframe selection
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDate, QTime, Qt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.statistics_window import StatisticsWindow
from gui.main_hub import TradingStrategyHub

def create_test_data():
    """Create test dataset with specific date range"""
    print("[TEST] Creating test dataset...")
    
    # Create time index (1 month of 1-minute data)
    start_time = datetime.now().replace(year=2024, month=1, day=1, hour=9, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(days=30)
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
    print(f"[TEST] Created dataset: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    return data

def test_statistics_timeframe():
    """Test the statistics analyzer with timeframe selection"""
    print("[TEST] Testing Statistics Analyzer with Timeframe Selection...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Create test data
    test_data = create_test_data()
    
    # Add test data to hub with metadata
    hub.datasets['Test Dataset - Jan 2024'] = {
        'data': test_data,
        'metadata': {
            'name': 'Test Dataset - Jan 2024',
            'source': 'test',
            'selected_date_range': [
                '2024-01-01T09:30:00',
                '2024-01-15T16:00:00'
            ]
        }
    }
    
    # Create statistics window
    stats_window = StatisticsWindow(hub)
    stats_window.show()
    stats_window.resize(1200, 800)
    
    # Test timeframe functionality
    print("[TEST] Testing timeframe selection...")
    
    # Check if dataset combo is populated
    dataset_count = stats_window.dataset_combo.count()
    print(f"[TEST] Datasets in combo: {dataset_count}")
    
    # Select the test dataset
    if dataset_count > 1:
        stats_window.dataset_combo.setCurrentText('Test Dataset - Jan 2024')
        print("[TEST] Selected test dataset")
        
        # Check if timeframe was auto-adjusted
        start_date = stats_window.start_date.dateTime()
        end_date = stats_window.end_date.dateTime()
        print(f"[TEST] Auto-adjusted timeframe: {start_date.toString(Qt.DateFormat.ISODate)} to {end_date.toString(Qt.DateFormat.ISODate)}")
        
        # Test manual timeframe adjustment
        print("[TEST] Testing manual timeframe adjustment...")
        new_start = QDate(2024, 1, 5)
        new_start_time = QTime(10, 0, 0)
        new_start_dt = stats_window.start_date.dateTime()
        new_start_dt.setDate(new_start)
        new_start_dt.setTime(new_start_time)
        stats_window.start_date.setDateTime(new_start_dt)
        
        new_end = QDate(2024, 1, 10)
        new_end_time = QTime(15, 0, 0)
        new_end_dt = stats_window.end_date.dateTime()
        new_end_dt.setDate(new_end)
        new_end_dt.setTime(new_end_time)
        stats_window.end_date.setDateTime(new_end_dt)
        
        print(f"[TEST] Manual timeframe set: {new_start_dt.toString(Qt.DateFormat.ISODate)} to {new_end_dt.toString(Qt.DateFormat.ISODate)}")
    
    print("[TEST] Test complete! The statistics window should show:")
    print("  - Dataset selection dropdown")
    print("  - Timeframe range pickers (From/To)")
    print("  - Auto-adjusted timeframe when dataset is selected")
    print("  - Manual timeframe adjustment capability")
    
    return app, stats_window

if __name__ == "__main__":
    app, stats_window = test_statistics_timeframe()
    
    # Keep the application running
    print("\n[TEST] Press Ctrl+C to exit...")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[TEST] Exiting...")
        sys.exit(0) 