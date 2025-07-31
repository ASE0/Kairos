#!/usr/bin/env python3
"""
Test script for backtest save/load functionality
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
from gui.backtest_window import BacktestWindow
from gui.results_viewer_window import ResultsViewerWindow
from strategies.strategy_builders import BacktestEngine

def create_test_data():
    """Create test dataset"""
    print("[TEST] Creating test dataset...")
    
    # Create time index (1 hour of 1-minute data)
    start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=1)
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
    print(f"[TEST] Created dataset: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    return data

def test_backtest_save_load():
    """Test that backtest results are saved to disk and can be loaded"""
    print("[TEST] Testing Backtest Save/Load Functionality...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Create test data
    test_data = create_test_data()
    
    # Add test data to hub
    hub.datasets['Test Dataset'] = {
        'data': test_data,
        'metadata': {'name': 'Test Dataset', 'source': 'test'}
    }
    
    # Create a simple test strategy
    from strategies.strategy_builders import SimpleMovingAverageStrategy
    test_strategy = SimpleMovingAverageStrategy(
        name="Test SMA Strategy",
        short_window=5,
        long_window=10
    )
    
    # Add strategy to hub
    hub.strategies['pattern']['test_sma'] = test_strategy
    
    # Create backtest window
    backtest_window = BacktestWindow(hub)
    backtest_window.show()
    backtest_window.resize(1200, 800)
    
    # Create results viewer window
    results_window = ResultsViewerWindow(hub)
    results_window.show()
    results_window.resize(1200, 800)
    
    print("[TEST] Windows created. Testing workflow:")
    print("1. Backtest window should be open")
    print("2. Results viewer should be open and show any existing results")
    print("3. Run a backtest in the backtest window")
    print("4. Check that results appear in the results viewer")
    print("5. Close and reopen results viewer to test disk loading")
    
    # Set up a timer to run the test automatically
    def run_test():
        print("[TEST] Running automatic test...")
        
        # Simulate running a backtest
        try:
            print("[TEST] Running backtest...")
            engine = BacktestEngine()
            results = engine.run_backtest(test_strategy, test_data.copy())
            
            # Add results to hub
            results['strategy_name'] = test_strategy.name
            results['timeframe'] = '1min'
            results['result_display_name'] = f"{test_strategy.name}_1min_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save results to disk
            hub.strategy_manager.save_backtest_results(results)
            
            # Reload results from disk
            hub.results = hub.strategy_manager.load_all_results()
            
            print(f"[TEST] Backtest completed: {results.get('total_trades', 0)} trades")
            print(f"[TEST] Results saved to disk")
            
            # Refresh results viewer
            results_window._refresh_results_from_disk()
            
            print("[TEST] Results viewer refreshed from disk")
            
            # Check if results are visible
            result_count = results_window.results_combo.count()
            print(f"[TEST] Results in viewer: {result_count}")
            
            if result_count > 0:
                print("[TEST] ✅ SUCCESS: Results are visible in the viewer!")
                QMessageBox.information(None, "Test Complete", 
                    f"✅ SUCCESS!\n\n"
                    f"Backtest completed with {results.get('total_trades', 0)} trades\n"
                    f"Results saved to disk and loaded in viewer\n"
                    f"Results count: {result_count}")
            else:
                print("[TEST] ❌ FAILED: No results visible in viewer")
                QMessageBox.warning(None, "Test Failed", 
                    "❌ FAILED: No results visible in viewer")
                
        except Exception as e:
            print(f"[TEST] ❌ ERROR: {e}")
            QMessageBox.critical(None, "Test Error", f"❌ ERROR: {e}")
    
    # Run test after 2 seconds
    timer = QTimer()
    timer.singleShot(2000, run_test)
    
    return app, hub, backtest_window, results_window

if __name__ == "__main__":
    app, hub, backtest_window, results_window = test_backtest_save_load()
    
    # Keep the application running
    print("\n[TEST] Press Ctrl+C to exit...")
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[TEST] Exiting...")
        sys.exit(0) 