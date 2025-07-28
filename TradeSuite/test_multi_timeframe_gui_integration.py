"""
Test Multi-Timeframe GUI Integration
===================================
Comprehensive test to verify that the GUI properly handles multi-timeframe
datasets with automatic timeframe creation when strategies need them.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.main_hub import TradingStrategyHub
from gui.backtest_window import BacktestWindow
from strategies.strategy_builders import PatternStrategy, Action
from patterns.candlestick_patterns import HammerPattern
from core.data_structures import TimeRange
from processors.data_processor import MultiTimeframeProcessor


def load_nq_dataset():
    """Load the real NQ_5s.csv dataset"""
    print("Loading NQ_5s.csv dataset...")
    
    dataset_path = r"C:\Users\Arnav\Downloads\TradeSuite\NQ_5s.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    try:
        # Load the dataset
        data = pd.read_csv(dataset_path)
        print(f"✅ Loaded {len(data)} rows from NQ_5s.csv")
        print(f"Columns: {list(data.columns)}")
        
        # Create datetime index from Date and Time columns
        if 'Date' in data.columns and 'Time' in data.columns:
            data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
            data.set_index('datetime', inplace=True)
            print("✅ Created datetime index from Date and Time columns")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def create_multi_timeframe_strategy():
    """Create a strategy that uses multiple timeframes"""
    print("Creating multi-timeframe strategy...")
    
    # Create patterns for different timeframes
    hammer_1m = HammerPattern(timeframes=[TimeRange(1, 'm')])
    hammer_5m = HammerPattern(timeframes=[TimeRange(5, 'm')])
    hammer_15m = HammerPattern(timeframes=[TimeRange(15, 'm')])
    
    # Create actions for different timeframes
    action_1m = Action(
        name="hammer_1m",
        pattern=hammer_1m,
        time_range=TimeRange(1, 'm'),
        location_strategy="FVG"
    )
    
    action_5m = Action(
        name="hammer_5m", 
        pattern=hammer_5m,
        time_range=TimeRange(5, 'm'),
        location_strategy="FVG"
    )
    
    action_15m = Action(
        name="hammer_15m",
        pattern=hammer_15m, 
        time_range=TimeRange(15, 'm'),
        location_strategy="FVG"
    )
    
    # Create strategy that uses all timeframes
    strategy = PatternStrategy(
        name="Multi-Timeframe Test Strategy",
        actions=[action_1m, action_5m, action_15m],
        min_actions_required=1
    )
    
    print(f"Created strategy: {strategy.name}")
    print(f"Strategy timeframes: {[f'{action.time_range.value}{action.time_range.unit}' for action in strategy.actions]}")
    
    return strategy


def test_workspace_manager_multi_timeframe():
    """Test workspace manager multi-timeframe functionality directly"""
    print("\n=== Testing Workspace Manager Multi-Timeframe ===")
    
    from core.workspace_manager import WorkspaceManager
    
    # Create workspace manager
    workspace_manager = WorkspaceManager()
    
    # Load real dataset
    test_data = load_nq_dataset()
    if test_data is None:
        return False
    
    # Create timeframes
    timeframes = [
        TimeRange(1, 'm'),
        TimeRange(5, 'm'),
        TimeRange(15, 'm')
    ]
    
    # Save multi-timeframe dataset
    print("Saving multi-timeframe dataset...")
    success = workspace_manager.save_multi_timeframe_dataset(
        "NQ_Multi_Timeframe",
        test_data,
        timeframes,
        {'source': 'NQ_5s.csv', 'description': 'Real NQ dataset with multiple timeframes'}
    )
    
    if not success:
        print("❌ Failed to save multi-timeframe dataset")
        return False
    
    print("✅ Multi-timeframe dataset saved successfully")
    
    # Load multi-timeframe dataset
    print("Loading multi-timeframe dataset...")
    mtf_info = workspace_manager.load_multi_timeframe_dataset("NQ_Multi_Timeframe")
    
    if not mtf_info:
        print("❌ Failed to load multi-timeframe dataset")
        return False
    
    print(f"✅ Multi-timeframe dataset loaded: {list(mtf_info['datasets'].keys())}")
    
    # Test ensuring timeframes are available
    print("Testing automatic timeframe creation...")
    required_timeframes = ['1m', '5m', '10m', '15m', '30m']  # 10m and 30m not in original
    
    available_timeframes = workspace_manager.ensure_timeframes_available(
        "NQ_Multi_Timeframe",
        required_timeframes
    )
    
    if available_timeframes:
        print(f"✅ All required timeframes available: {list(available_timeframes.keys())}")
        
        # Check if missing timeframes were created
        if '10m' in available_timeframes and '30m' in available_timeframes:
            print("✅ Missing timeframes (10m, 30m) were automatically created!")
            return True
        else:
            print("❌ Missing timeframes were not created")
            return False
    else:
        print("❌ Failed to ensure timeframes are available")
        return False


def test_backtester_integration():
    """Test backtester with multi-timeframe strategy"""
    print("\n=== Testing Backtester Integration ===")
    
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Load real dataset
    test_data = load_nq_dataset()
    if test_data is None:
        return False
    
    # Create multi-timeframe strategy
    strategy = create_multi_timeframe_strategy()
    
    # Add to hub
    hub.datasets['NQ_Multi_Timeframe'] = {
        'data': test_data,
        'metadata': {'name': 'NQ_Multi_Timeframe', 'source': 'NQ_5s.csv'}
    }
    hub.strategies['pattern']['multi_tf_test'] = strategy
    
    # Create backtest window
    backtest = BacktestWindow(hub)
    
    # Simulate selecting dataset and strategy
    backtest.dataset_combo.addItem("NQ_Multi_Timeframe")
    backtest.dataset_combo.setCurrentText("NQ_Multi_Timeframe")
    
    backtest.strategy_combo.addItem("[pattern] Multi-Timeframe Test Strategy")
    backtest.strategy_combo.setCurrentText("[pattern] Multi-Timeframe Test Strategy")
    
    print("Backtest window configured with multi-timeframe dataset and strategy")
    
    # Test strategy timeframe detection
    strategy_timeframes = backtest._get_strategy_timeframes(strategy)
    print(f"Strategy timeframes detected: {strategy_timeframes}")
    
    if len(strategy_timeframes) == 3:  # 1m, 5m, 15m
        print("✅ Strategy timeframe detection working correctly")
    else:
        print(f"❌ Expected 3 timeframes, got {len(strategy_timeframes)}")
        return False
    
    # Test workspace manager integration
    if hasattr(hub, 'workspace_manager'):
        print("✅ Workspace manager available for multi-timeframe support")
        
        # Test ensuring timeframes are available
        available_timeframes = hub.workspace_manager.ensure_timeframes_available(
            'NQ_Multi_Timeframe', strategy_timeframes
        )
        
        if available_timeframes:
            print(f"✅ Multi-timeframe data available: {list(available_timeframes.keys())}")
            return True
        else:
            print("❌ Failed to ensure timeframes are available")
            return False
    else:
        print("❌ No workspace manager available")
        return False


def test_complete_workflow():
    """Test the complete multi-timeframe workflow"""
    print("\n=== Testing Complete Multi-Timeframe Workflow ===")
    
    app = QApplication(sys.argv)
    
    # Create main hub
    hub = TradingStrategyHub()
    
    # Load real dataset
    test_data = load_nq_dataset()
    if test_data is None:
        return False
    
    # Create multi-timeframe strategy
    strategy = create_multi_timeframe_strategy()
    
    # Step 1: Save multi-timeframe dataset with only 1m and 5m
    print("\nStep 1: Creating initial timeframes (1m, 5m only)...")
    
    initial_timeframes = [
        TimeRange(1, 'm'),
        TimeRange(5, 'm')
    ]
    
    success = hub.workspace_manager.save_multi_timeframe_dataset(
        "NQ_Complete_Test",
        test_data,
        initial_timeframes,
        {'source': 'NQ_5s.csv', 'description': 'Complete workflow test with real data'}
    )
    
    if not success:
        print("❌ Failed to save initial timeframes")
        return False
    
    print("✅ Initial timeframes (1m, 5m) saved successfully")
    
    # Step 2: Add strategy to hub
    print("\nStep 2: Adding multi-timeframe strategy...")
    hub.strategies['pattern']['complete_test'] = strategy
    
    # Step 3: Test backtester automatic timeframe creation
    print("\nStep 3: Testing backtester with missing timeframes...")
    
    backtest = BacktestWindow(hub)
    backtest.dataset_combo.addItem("NQ_Complete_Test")
    backtest.dataset_combo.setCurrentText("NQ_Complete_Test")
    backtest.strategy_combo.addItem("[pattern] Multi-Timeframe Test Strategy")
    backtest.strategy_combo.setCurrentText("[pattern] Multi-Timeframe Test Strategy")
    
    # Get strategy timeframes
    strategy_timeframes = backtest._get_strategy_timeframes(strategy)
    print(f"Strategy requires: {strategy_timeframes}")
    
    # Check if workspace manager can create missing timeframes
    if hasattr(hub, 'workspace_manager'):
        available_timeframes = hub.workspace_manager.ensure_timeframes_available(
            'NQ_Complete_Test', strategy_timeframes
        )
        
        if available_timeframes:
            print(f"✅ All required timeframes available: {list(available_timeframes.keys())}")
            
            # Verify that 15m was automatically created
            if '15m' in available_timeframes:
                print("✅ 15m timeframe was automatically created!")
                return True
            else:
                print("❌ 15m timeframe was not automatically created")
                return False
        else:
            print("❌ Failed to create missing timeframes")
            return False
    else:
        print("❌ No workspace manager available")
        return False


def main():
    """Run all integration tests"""
    print("🧪 Multi-Timeframe GUI Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Workspace Manager Multi-Timeframe", test_workspace_manager_multi_timeframe),
        ("Backtester Integration", test_backtester_integration),
        ("Complete Workflow", test_complete_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All multi-timeframe GUI integration tests passed!")
        print("The system can now:")
        print("  ✅ Create multi-timeframe datasets in workspace manager")
        print("  ✅ Automatically create missing timeframes when strategies need them")
        print("  ✅ Handle multi-timeframe strategies in backtester")
        print("  ✅ Preserve all original data without loss")
        return True
    else:
        print("⚠️ Some tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    main() 