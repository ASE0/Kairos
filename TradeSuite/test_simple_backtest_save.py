#!/usr/bin/env python3
"""
Simple test for backtest save/load functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.main_hub import TradingStrategyHub
from gui.results_viewer_window import ResultsViewerWindow

def create_test_results():
    """Create test backtest results"""
    print("[TEST] Creating test backtest results...")
    
    # Create time index (1 day of 1-minute data)
    start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(days=1)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create OHLC data
    np.random.seed(42)
    base_price = 100.0
    prices = []
    current_price = base_price
    
    for i in range(len(time_index)):
        change = np.random.normal(0, 0.1) + 0.001
        current_price += change
        
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
    
    data = pd.DataFrame(prices, index=time_index)
    
    # Create test trades
    trades = [
        {
            'entry_time': time_index[10],
            'exit_time': time_index[15],
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'direction': 'long'
        },
        {
            'entry_time': time_index[30],
            'exit_time': time_index[35],
            'entry_price': 101.0,
            'exit_price': 100.8,
            'pnl': -0.2,
            'direction': 'short'
        }
    ]
    
    # Create test results
    results = {
        'strategy_name': 'Test SMA Strategy',
        'timeframe': '1min',
        'result_display_name': f'Test_SMA_Strategy_1min_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'initial_capital': 100000,
        'final_capital': 100700,
        'cumulative_pnl': 700,
        'total_return': 0.007,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.02,
        'win_rate': 0.5,
        'profit_factor': 1.8,
        'total_trades': 2,
        'equity_curve': [100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700],
        'trades': trades,
        'data': data,
        'action_details': {
            'SMA': pd.Series([False] * len(data), index=data.index)
        }
    }
    
    print(f"[TEST] Created test results with {len(trades)} trades")
    return results

def test_simple_backtest_save_load():
    """Test that backtest results are saved to disk and can be loaded"""
    print("[TEST] Testing Simple Backtest Save/Load Functionality...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Create test results
    test_results = create_test_results()
    
    # Save results to disk
    print("[TEST] Saving results to disk...")
    hub.strategy_manager.save_backtest_results(test_results)
    
    # Reload results from disk
    print("[TEST] Reloading results from disk...")
    hub.results = hub.strategy_manager.load_all_results()
    
    # Create results viewer window
    results_window = ResultsViewerWindow(hub)
    results_window.show()
    results_window.resize(1200, 800)
    
    print("[TEST] Results viewer created. Testing workflow:")
    print("1. Results viewer should be open")
    print("2. Results should be visible in the dropdown")
    print("3. Results should be loadable from disk")
    
    # Set up a timer to check the results
    def check_results():
        print("[TEST] Checking results...")
        
        # Check if results are visible
        result_count = results_window.results_combo.count()
        print(f"[TEST] Results in viewer: {result_count}")
        
        if result_count > 0:
            print("[TEST] ✅ SUCCESS: Results are visible in the viewer!")
            
            # Try to load the first result
            results_window.results_combo.setCurrentIndex(0)
            results_window._on_load_selected_result()
            
            print("[TEST] ✅ SUCCESS: Results loaded successfully!")
            QMessageBox.information(None, "Test Complete", 
                f"✅ SUCCESS!\n\n"
                f"Backtest results saved to disk and loaded in viewer\n"
                f"Results count: {result_count}\n"
                f"Strategy: {test_results['strategy_name']}\n"
                f"Trades: {test_results['total_trades']}")
        else:
            print("[TEST] ❌ FAILED: No results visible in viewer")
            QMessageBox.warning(None, "Test Failed", 
                "❌ FAILED: No results visible in viewer")
    
    # Check results after 1 second
    timer = QTimer()
    timer.singleShot(1000, check_results)
    
    return app, hub, results_window

if __name__ == "__main__":
    app, hub, results_window = test_simple_backtest_save_load()
    
    # Keep the application running
    print("\n[TEST] Press Ctrl+C to exit...")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[TEST] Exiting...")
        sys.exit(0) 