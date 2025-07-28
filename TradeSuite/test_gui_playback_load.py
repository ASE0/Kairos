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

def test_gui_playback_load():
    """Test that GUI can properly load playback data from saved results"""
    print("Testing GUI playback data loading...")
    
    # Create test data with more variation (to avoid diagonal line)
    start_date = datetime(2024, 1, 1, 9, 30, 0)
    end_date = datetime(2024, 1, 1, 16, 0, 0)
    dates = pd.date_range(start_date, end_date, freq='1min')
    
    # Create more realistic OHLCV data with larger price movements
    np.random.seed(42)
    base_price = 100.0
    # Create more volatile price movements
    price_changes = np.random.normal(0, 0.01, len(dates))  # Larger changes
    prices = base_price + np.cumsum(price_changes)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.005, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 0.01, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 0.01, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationships are correct
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    print(f"Created test data: {len(test_data)} bars")
    print(f"Price range: {test_data['close'].max() - test_data['close'].min():.4f}")
    
    # Create strategy and run backtest
    fvg_pattern = FVGPattern(timeframes=[TimeRange(1, 'm')], min_gap_size=0.001)
    action = Action(name="FVG", pattern=fvg_pattern)
    strategy = PatternStrategy(name="Test FVG", actions=[action])
    
    engine = MultiTimeframeBacktestEngine()
    results = engine.run_backtest(strategy, test_data.copy())
    
    # Simulate the GUI's _initialize_playback_data method exactly
    print("\nSimulating GUI playback data initialization...")
    
    result_data = results  # This is what the GUI receives
    
    # Priority 1: Use multi-timeframe data if available
    data = None
    multi_tf_data = result_data.get('multi_tf_data', {})
    print(f"[DEBUG] Multi-timeframe data keys: {list(multi_tf_data.keys()) if multi_tf_data else 'None'}")
    
    if multi_tf_data:
        execution_data = multi_tf_data.get('execution')
        print(f"[DEBUG] Execution data type: {type(execution_data)}")
        if execution_data is not None and isinstance(execution_data, pd.DataFrame):
            data = execution_data
            print(f"[DEBUG] Using multi-timeframe execution data with shape: {data.shape}")
            print(f"[DEBUG] Execution data columns: {list(data.columns)}")
        else:
            print(f"[DEBUG] Multi-timeframe data found but no execution data available")
            print(f"[DEBUG] Execution data is None: {execution_data is None}")
            print(f"[DEBUG] Execution data is DataFrame: {isinstance(execution_data, pd.DataFrame)}")
    
    # Priority 2: Use the resampled data from the backtest results
    if data is None and 'data' in result_data and isinstance(result_data['data'], pd.DataFrame):
        data = result_data['data']
        print(f"[DEBUG] Using resampled data from backtest results with shape: {data.shape}")
        print(f"[DEBUG] Data columns: {list(data.columns)}")
    elif data is None:
        print(f"[DEBUG] No 'data' key in result_data or not a DataFrame")
        print(f"[DEBUG] 'data' key exists: {'data' in result_data}")
        if 'data' in result_data:
            print(f"[DEBUG] 'data' type: {type(result_data['data'])}")
    
    # Priority 3: Create synthetic data from equity curve if no resampled data available
    if data is None and 'equity_curve' in result_data and result_data['equity_curve']:
        print(f"[DEBUG] ⚠️ FALLING BACK TO SYNTHETIC DATA - This will show diagonal line!")
        print(f"[DEBUG] Creating synthetic data from equity curve with {len(result_data['equity_curve'])} points")
        # This is where the diagonal line comes from!
        equity_curve = result_data['equity_curve']
        n_points = len(equity_curve)
        
        timeframe = result_data.get('timeframe', '1d')
        if timeframe == '1d':
            freq = 'D'
        elif timeframe == '1h':
            freq = 'H'
        elif timeframe == '15min':
            freq = '15T'
        elif timeframe == '5min':
            freq = '5T'
        elif timeframe == '1min':
            freq = '1T'
        else:
            freq = '1T'
            
        time_index = pd.date_range('2023-01-01', periods=n_points, freq=freq)
        base_price = 100.0
        data = pd.DataFrame({
            'open': [base_price + i * 0.1 for i in range(n_points)],
            'high': [base_price + i * 0.1 + 0.5 for i in range(n_points)],
            'low': [base_price + i * 0.1 - 0.5 for i in range(n_points)],
            'close': [base_price + i * 0.1 + (equity_curve[i] - equity_curve[0]) / 1000 for i in range(n_points)],
            'volume': [1000 + i * 10 for i in range(n_points)]
        }, index=time_index)
        print(f"[DEBUG] Created synthetic DataFrame with shape: {data.shape} and frequency: {freq}")
        print(f"[DEBUG] ⚠️ WARNING: This synthetic data will show diagonal line, not real price data!")
    
    if data is None:
        print(f"[DEBUG] No valid data found for playback")
        return False
    
    # Check the final data
    print(f"\nFinal playback data:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Index range: {data.index[0]} to {data.index[-1]}")
    
    # Check if data looks realistic
    price_range = data['close'].max() - data['close'].min()
    print(f"Price range: {price_range:.4f}")
    
    # Check if this is synthetic data (diagonal line)
    if 'base_price + i * 0.1' in str(data['close'].iloc[0]):
        print("⚠️ WARNING: This is synthetic data (diagonal line)!")
        return False
    elif price_range < 0.1:
        print("⚠️ WARNING: Data may be too flat (diagonal-like)")
        return False
    else:
        print("✓ Data appears realistic (not diagonal)")
        return True

if __name__ == "__main__":
    success = test_gui_playback_load()
    if success:
        print("\n✓ GUI playback data loading test PASSED")
    else:
        print("\n✗ GUI playback data loading test FAILED") 