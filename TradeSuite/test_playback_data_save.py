import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action, MultiTimeframeBacktestEngine
from patterns.enhanced_candlestick_patterns import FVGPattern
from core.data_structures import TimeRange

def test_playback_data_save():
    """Test that backtest engine saves execution data properly for playback"""
    print("Testing playback data save functionality...")
    
    # Create test data (1m bars for 1 day)
    start_date = datetime(2024, 1, 1, 9, 30, 0)
    end_date = datetime(2024, 1, 1, 16, 0, 0)
    dates = pd.date_range(start_date, end_date, freq='1min')
    
    # Create realistic OHLCV data (not diagonal)
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, len(dates))  # Small random changes
    prices = base_price + np.cumsum(price_changes)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.0005, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 0.001, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 0.001, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationships are correct
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    print(f"Created test data: {len(test_data)} bars, range: {test_data.index[0]} to {test_data.index[-1]}")
    print(f"Data columns: {list(test_data.columns)}")
    print(f"Sample data:\n{test_data.head()}")
    
    # Create a simple FVG strategy
    fvg_pattern = FVGPattern(timeframes=[TimeRange(1, 'm')], min_gap_size=0.001)
    action = Action(name="FVG", pattern=fvg_pattern)
    strategy = PatternStrategy(name="Test FVG", actions=[action])
    
    # Run backtest
    engine = MultiTimeframeBacktestEngine()
    print("\nRunning backtest...")
    results = engine.run_backtest(strategy, test_data.copy())
    
    # Check what was saved
    print(f"\nResults keys: {list(results.keys())}")
    print(f"'data' in results: {'data' in results}")
    print(f"'multi_tf_data' in results: {'multi_tf_data' in results}")
    
    if 'data' in results:
        saved_data = results['data']
        print(f"Saved 'data' shape: {saved_data.shape}")
        print(f"Saved 'data' columns: {list(saved_data.columns)}")
        print(f"Saved 'data' index range: {saved_data.index[0]} to {saved_data.index[-1]}")
        print(f"Sample saved data:\n{saved_data.head()}")
    
    if 'multi_tf_data' in results:
        multi_tf_data = results['multi_tf_data']
        print(f"Multi-timeframe data keys: {list(multi_tf_data.keys())}")
        if 'execution' in multi_tf_data:
            exec_data = multi_tf_data['execution']
            print(f"Execution data shape: {exec_data.shape}")
            print(f"Execution data columns: {list(exec_data.columns)}")
            print(f"Execution data index range: {exec_data.index[0]} to {exec_data.index[-1]}")
    
    # Test the results viewer's data loading logic
    print("\nTesting results viewer data loading logic...")
    
    # Simulate the _initialize_playback_data logic
    data = None
    multi_tf_data = results.get('multi_tf_data', {})
    
    if multi_tf_data:
        execution_data = multi_tf_data.get('execution')
        if execution_data is not None and isinstance(execution_data, pd.DataFrame):
            data = execution_data
            print(f"✓ Using multi-timeframe execution data with shape: {data.shape}")
        else:
            print(f"✗ Multi-timeframe data found but no execution data available")
    
    if data is None and 'data' in results and isinstance(results['data'], pd.DataFrame):
        data = results['data']
        print(f"✓ Using resampled data from backtest results with shape: {data.shape}")
    elif data is None:
        print(f"✗ No 'data' key in results or not a DataFrame")
    
    if data is None and 'equity_curve' in results and results['equity_curve']:
        print(f"⚠️ FALLING BACK TO SYNTHETIC DATA - This will show diagonal line!")
    
    if data is not None:
        print(f"✓ Playback data found: {len(data)} bars")
        print(f"✓ Data columns: {list(data.columns)}")
        print(f"✓ Index range: {data.index[0]} to {data.index[-1]}")
        
        # Check if data looks realistic (not diagonal)
        price_range = data['close'].max() - data['close'].min()
        print(f"✓ Price range: {price_range:.4f}")
        if price_range > 0.1:  # Should have some variation
            print("✓ Data appears realistic (not diagonal)")
        else:
            print("⚠️ Data may be too flat (diagonal-like)")
    else:
        print("✗ No valid playback data found")
    
    return results

if __name__ == "__main__":
    test_playback_data_save() 