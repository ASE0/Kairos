"""
Test Backtest Timeframe Fix
===========================
Quick test to verify that the backtest window can handle both TimeRange objects 
and dictionaries when extracting timeframes from strategies.
"""

import sys
import os
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import QApplication

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.backtest_window import BacktestWindow
from strategies.strategy_builders import PatternStrategy, Action
from patterns.candlestick_patterns import HammerPattern
from core.data_structures import TimeRange

def test_timeframe_extraction():
    """Test that timeframe extraction works with both TimeRange objects and dictionaries"""
    print("Testing timeframe extraction from strategies...")
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create a backtest window instance
    backtest_window = BacktestWindow()
    
    # Test 1: Strategy with TimeRange objects
    print("\n1. Testing with TimeRange objects:")
    strategy1 = PatternStrategy(name="Test Strategy 1")
    
    # Add actions with TimeRange objects
    action1 = Action(
        name="Hammer 1m",
        pattern=HammerPattern(timeframes=[TimeRange(1, "minute")]),
        time_range=TimeRange(1, "minute")
    )
    action2 = Action(
        name="Hammer 5m", 
        pattern=HammerPattern(timeframes=[TimeRange(5, "minutes")]),
        time_range=TimeRange(5, "minutes")
    )
    strategy1.add_action(action1)
    strategy1.add_action(action2)
    
    timeframes1 = backtest_window._get_strategy_timeframes(strategy1)
    print(f"   Timeframes extracted: {timeframes1}")
    assert "1m" in timeframes1 and "5m" in timeframes1, "Failed to extract timeframes from TimeRange objects"
    print("   ✅ PASSED: TimeRange objects")
    
    # Test 2: Strategy with dictionary time_range (simulating loaded strategy)
    print("\n2. Testing with dictionary time_range:")
    strategy2 = PatternStrategy(name="Test Strategy 2")
    
    # Add actions with dictionary time_range
    action3 = Action(
        name="Hammer 15m",
        pattern=HammerPattern(timeframes=[TimeRange(15, "minutes")]),
        time_range={"value": 15, "unit": "minutes"}  # Dictionary format
    )
    action4 = Action(
        name="Hammer 30m",
        pattern=HammerPattern(timeframes=[TimeRange(30, "minutes")]), 
        time_range={"value": 30, "unit": "minutes"}  # Dictionary format
    )
    strategy2.add_action(action3)
    strategy2.add_action(action4)
    
    timeframes2 = backtest_window._get_strategy_timeframes(strategy2)
    print(f"   Timeframes extracted: {timeframes2}")
    assert "15m" in timeframes2 and "30m" in timeframes2, "Failed to extract timeframes from dictionaries"
    print("   ✅ PASSED: Dictionary time_range")
    
    # Test 3: Mixed strategy (some TimeRange, some dict)
    print("\n3. Testing with mixed time_range types:")
    strategy3 = PatternStrategy(name="Test Strategy 3")
    
    action5 = Action(
        name="Hammer 1h",
        pattern=HammerPattern(timeframes=[TimeRange(1, "hour")]),
        time_range=TimeRange(1, "hour")  # TimeRange object
    )
    action6 = Action(
        name="Hammer 4h", 
        pattern=HammerPattern(timeframes=[TimeRange(4, "hours")]),
        time_range={"value": 4, "unit": "hours"}  # Dictionary
    )
    strategy3.add_action(action5)
    strategy3.add_action(action6)
    
    timeframes3 = backtest_window._get_strategy_timeframes(strategy3)
    print(f"   Timeframes extracted: {timeframes3}")
    assert "1h" in timeframes3 and "4h" in timeframes3, "Failed to extract timeframes from mixed types"
    print("   ✅ PASSED: Mixed time_range types")
    
    # Test 4: Strategy with no time_range (should default to 1m)
    print("\n4. Testing with no time_range (default):")
    strategy4 = PatternStrategy(name="Test Strategy 4")
    
    action7 = Action(
        name="Hammer No Time",
        pattern=HammerPattern(timeframes=[TimeRange(1, "minute")]),
        # No time_range specified
    )
    strategy4.add_action(action7)
    
    timeframes4 = backtest_window._get_strategy_timeframes(strategy4)
    print(f"   Timeframes extracted: {timeframes4}")
    assert "1m" in timeframes4, "Failed to default to 1m when no time_range specified"
    print("   ✅ PASSED: Default timeframe")
    
    print("\n🎉 All timeframe extraction tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_timeframe_extraction()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 