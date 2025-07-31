#!/usr/bin/env python3
"""
Test signal connection between BacktestWindow and MainHub
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.main_hub import TradingStrategyHub
from gui.backtest_window import BacktestWindow

def test_signal_connection():
    """Test that the signal connection is working"""
    print("[TEST] Testing signal connection...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Create backtest window
    backtest_window = BacktestWindow(hub)
    
    # Test signal emission manually
    print("[TEST] Testing manual signal emission...")
    test_results = {
        'strategy_name': 'Test Strategy',
        'timeframe': '1m',
        'total_trades': 5,
        'total_return': 0.05,
        'final_capital': 105000,
        'initial_capital': 100000,
        'trades': [],
        'equity_curve': []
    }
    
    # Emit signal manually
    backtest_window.backtest_complete.emit(test_results)
    
    print("[TEST] Signal emitted. Check if MainHub received it.")
    
    # Set up a timer to check the results
    def check_results():
        print("[TEST] Checking if results were saved...")
        
        # Check if results were saved to disk
        all_results = hub.strategy_manager.load_all_results()
        print(f"[TEST] Results loaded from disk: {len(all_results)}")
        
        if len(all_results) > 0:
            print("[TEST] ✅ SUCCESS: Results were saved and loaded!")
            print(f"[TEST] Result keys: {list(all_results.keys())}")
        else:
            print("[TEST] ❌ FAILED: No results found on disk")
    
    # Check results after 1 second
    timer = QTimer()
    timer.singleShot(1000, check_results)
    
    return app, hub, backtest_window

if __name__ == "__main__":
    app, hub, backtest_window = test_signal_connection()
    
    # Keep the application running
    print("\n[TEST] Press Ctrl+C to exit...")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[TEST] Exiting...")
        sys.exit(0) 