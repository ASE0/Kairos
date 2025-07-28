"""
Test Backtest Engine Timeframe Fix
==================================
Test to verify that the backtest engine uses action.time_range instead of pattern.timeframes
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import MultiTimeframeBacktestEngine, PatternStrategy, Action
from patterns.candlestick_patterns import CustomPattern
from core.data_structures import TimeRange, OHLCRatio

def create_test_data():
    """Create test data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    }, index=dates)
    return data

def test_backtest_engine_timeframe():
    """Test that backtest engine uses action.time_range for timeframe determination"""
    print("Testing backtest engine timeframe determination...")
    
    # Create test data
    data = create_test_data()
    
    # Create a breakout pattern with default 30-minute timeframe
    breakout_pattern = CustomPattern(
        name="Breakout",
        timeframes=[TimeRange(30, 'm')],  # Default 30-minute timeframe
        ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)]
    )
    
    # Create an action with 1-minute time_range (user setting)
    action = Action(
        name="Breakout 1m",
        pattern=breakout_pattern,
        time_range={"value": 1, "unit": "minutes"}  # User set to 1 minute
    )
    
    # Create strategy
    strategy = PatternStrategy(name="Breakout Test Strategy")
    strategy.add_action(action)
    
    # Create backtest engine
    engine = MultiTimeframeBacktestEngine()
    
    # Prepare multi-timeframe data
    multi_tf_data = engine.prepare_multi_timeframe_data(data, strategy)
    
    print(f"Available timeframes in multi_tf_data: {list(multi_tf_data.keys())}")
    
    # Should include 1min (from action.time_range) not 30min (from pattern.timeframes)
    if "1min" in multi_tf_data and "30min" not in multi_tf_data:
        print("✅ PASSED: Backtest engine uses correct timeframe (1min)")
        return True
    else:
        print("❌ FAILED: Backtest engine uses wrong timeframe")
        print(f"Expected: ['1min'], Got: {list(multi_tf_data.keys())}")
        return False

def test_evaluation_timeframe():
    """Test that evaluation uses the correct timeframe"""
    print("\nTesting evaluation timeframe...")
    
    # Create test data
    data = create_test_data()
    
    # Create a breakout pattern with default 30-minute timeframe
    breakout_pattern = CustomPattern(
        name="Breakout",
        timeframes=[TimeRange(30, 'm')],  # Default 30-minute timeframe
        ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)]
    )
    
    # Create an action with 1-minute time_range (user setting)
    action = Action(
        name="Breakout 1m",
        pattern=breakout_pattern,
        time_range={"value": 1, "unit": "minutes"}  # User set to 1 minute
    )
    
    # Create strategy
    strategy = PatternStrategy(name="Breakout Test Strategy")
    strategy.add_action(action)
    
    # Create backtest engine
    engine = MultiTimeframeBacktestEngine()
    
    # Prepare multi-timeframe data
    multi_tf_data = engine.prepare_multi_timeframe_data(data, strategy)
    
    # Evaluate strategy
    signals, action_signals, zones, patterns = engine.evaluate_strategy_multi_timeframe(strategy, multi_tf_data)
    
    print(f"Evaluation completed successfully")
    print(f"Signal count: {signals.sum()}")
    print(f"Action signals: {list(action_signals.keys())}")
    
    # Check that the action was evaluated on the correct timeframe
    if "Breakout 1m" in action_signals:
        print("✅ PASSED: Action evaluated successfully")
        return True
    else:
        print("❌ FAILED: Action evaluation failed")
        return False

if __name__ == "__main__":
    print("Testing Backtest Engine Timeframe Fix")
    print("=" * 40)
    
    # Test timeframe determination
    timeframe_ok = test_backtest_engine_timeframe()
    
    # Test evaluation
    evaluation_ok = test_evaluation_timeframe()
    
    if timeframe_ok and evaluation_ok:
        print("\n🎉 All backtest engine tests passed!")
        print("The backtest engine now uses action.time_range instead of pattern.timeframes.")
    else:
        print("\n❌ Some backtest engine tests failed!")
        print("The backtest engine may still use incorrect timeframes.") 