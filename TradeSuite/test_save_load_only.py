#!/usr/bin/env python3
"""
Minimal test for backtest save/load functionality - no GUI, no BacktestEngine
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy_manager import StrategyManager

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

def test_save_load_only():
    """Test that backtest results are saved to disk and can be loaded"""
    print("[TEST] Testing Save/Load Functionality (No GUI)...")
    
    # Create strategy manager
    strategy_manager = StrategyManager()
    
    # Create test results
    test_results = create_test_results()
    
    # Save results to disk
    print("[TEST] Saving results to disk...")
    strategy_manager.save_backtest_results(test_results)
    
    # Reload results from disk
    print("[TEST] Reloading results from disk...")
    loaded_results = strategy_manager.load_all_results()
    
    # Check if results were loaded
    result_count = len(loaded_results)
    print(f"[TEST] Results loaded from disk: {result_count}")
    
    if result_count > 0:
        print("[TEST] ✅ SUCCESS: Results saved and loaded successfully!")
        
        # Check the first result
        first_result_id = list(loaded_results.keys())[0]
        first_result = loaded_results[first_result_id]
        
        print(f"[TEST] First result strategy: {first_result.get('strategy_name')}")
        print(f"[TEST] First result trades: {first_result.get('total_trades')}")
        print(f"[TEST] First result data shape: {first_result.get('data').shape if 'data' in first_result else 'No data'}")
        
        return True
    else:
        print("[TEST] ❌ FAILED: No results loaded from disk")
        return False

if __name__ == "__main__":
    success = test_save_load_only()
    
    if success:
        print("\n[TEST] ✅ SUCCESS: Backtest save/load functionality works!")
        sys.exit(0)
    else:
        print("\n[TEST] ❌ FAILED: Backtest save/load functionality failed!")
        sys.exit(1) 