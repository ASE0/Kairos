"""
Test Breakout Timeframe Fix
===========================
Test to verify that breakout pattern strategies show the correct timeframe
in the backtester (1 minute instead of 30 minutes)
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
from patterns.candlestick_patterns import CustomPattern
from core.data_structures import TimeRange, OHLCRatio

def test_breakout_timeframe_display():
    """Test that breakout pattern strategies show the correct timeframe"""
    print("Testing breakout pattern timeframe display...")
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create a backtest window instance
    backtest_window = BacktestWindow()
    
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
    
    # Test timeframe extraction
    timeframes = backtest_window._get_strategy_timeframes(strategy)
    print(f"Strategy timeframes extracted: {timeframes}")
    
    # Should show 1m (from action.time_range) not 30m (from pattern.timeframes)
    if "1m" in timeframes and "30m" not in timeframes:
        print("✅ PASSED: Breakout strategy shows correct timeframe (1m)")
        return True
    else:
        print("❌ FAILED: Breakout strategy shows wrong timeframe")
        print(f"Expected: ['1m'], Got: {timeframes}")
        return False

def test_fvg_timeframe_display():
    """Test that FVG pattern strategies show the correct timeframe"""
    print("\nTesting FVG pattern timeframe display...")
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create a backtest window instance
    backtest_window = BacktestWindow()
    
    # Create an FVG pattern with 1-minute timeframe
    fvg_pattern = CustomPattern(
        name="FVG",
        timeframes=[TimeRange(1, 'm')],  # 1-minute timeframe
        ohlc_ratios=[]
    )
    
    # Create an action with 1-minute time_range (matches pattern)
    action = Action(
        name="FVG 1m",
        pattern=fvg_pattern,
        time_range={"value": 1, "unit": "minutes"}  # Same as pattern
    )
    
    # Create strategy
    strategy = PatternStrategy(name="FVG Test Strategy")
    strategy.add_action(action)
    
    # Test timeframe extraction
    timeframes = backtest_window._get_strategy_timeframes(strategy)
    print(f"Strategy timeframes extracted: {timeframes}")
    
    # Should show 1m (from action.time_range)
    if "1m" in timeframes:
        print("✅ PASSED: FVG strategy shows correct timeframe (1m)")
        return True
    else:
        print("❌ FAILED: FVG strategy shows wrong timeframe")
        print(f"Expected: ['1m'], Got: {timeframes}")
        return False

if __name__ == "__main__":
    print("Testing Breakout Timeframe Fix")
    print("=" * 40)
    
    # Test breakout timeframe
    breakout_ok = test_breakout_timeframe_display()
    
    # Test FVG timeframe
    fvg_ok = test_fvg_timeframe_display()
    
    if breakout_ok and fvg_ok:
        print("\n🎉 All timeframe display tests passed!")
        print("The backtester will now show the correct timeframe for both breakout and FVG strategies.")
    else:
        print("\n❌ Some timeframe display tests failed!")
        print("The backtester may still show incorrect timeframes.") 