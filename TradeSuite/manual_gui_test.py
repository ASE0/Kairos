#!/usr/bin/env python3
"""
Manual GUI Test Script
Run this script to test the GUI manually
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gui_manually():
    """Test the GUI manually with automated interactions"""
    app = QApplication(sys.argv)
    
    # Import and create main window
    from gui.main_hub import TradingStrategyHub
    hub = TradingStrategyHub()
    hub.show()
    
    # Wait for window to appear
    QTest.qWait(1000)
    
    # Test 1: Check if window is visible
    if hub.isVisible():
        print("[PASS] Main window is visible")
    else:
        print("[FAIL] Main window is not visible")
    
    # Test 2: Check if datasets are loaded
    if hasattr(hub, 'datasets') and hub.datasets:
        print(f"[PASS] {len(hub.datasets)} datasets loaded")
    else:
        print("[WARN] No datasets loaded")
    
    # Test 3: Open backtest window
    try:
        from gui.backtest_window import BacktestWindow
        backtest_window = BacktestWindow(hub)
        backtest_window.show()
        QTest.qWait(1000)
        
        if backtest_window.isVisible():
            print("[PASS] Backtest window opened successfully")
        else:
            print("[FAIL] Backtest window failed to open")
            
    except Exception as e:
        print("[FAIL] Failed to open backtest window: {e}")
    
    # Keep the app running for manual inspection
    print("
[INFO] GUI is now open for manual testing...")
    print("Close the window when done testing.")
    
    return app.exec()

if __name__ == "__main__":
    test_gui_manually()
