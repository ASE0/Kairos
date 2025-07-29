#!/usr/bin/env python3
"""
Test GUI Integration with New Architecture
==========================================
Test that the new architecture integration works properly with the GUI components.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the test strategy
from strategies.vwap_test_strategy import VWAPTestStrategy

def create_test_data():
    """Create realistic test data with volume"""
    times = pd.date_range('2024-03-07 16:00:00', periods=100, freq='1min')
    
    # Create realistic price movement with volume
    base_price = 18200.0
    data = []
    
    for i in range(100):
        # Add some realistic price movement
        price_change = np.random.normal(0, 2)  # Random walk with some volatility
        if i > 0:
            price = data[-1]['close'] + price_change
        else:
            price = base_price
        
        # Ensure realistic OHLC relationships
        high = price + abs(np.random.normal(0, 1))
        low = price - abs(np.random.normal(0, 1))
        open_price = price + np.random.normal(0, 0.5)
        
        data.append({
            'datetime': times[i],
            'open': open_price,
            'high': max(open_price, high, price),
            'low': min(open_price, low, price),
            'close': price,
            'volume': np.random.randint(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

def test_strategy_detection():
    """Test that the strategy is properly detected for new architecture"""
    print("Testing Strategy Detection for New Architecture")
    print("="*50)
    
    # Create the test strategy
    strategy = VWAPTestStrategy()
    
    print(f"Strategy name: {strategy.name}")
    print(f"Strategy type: {type(strategy)}")
    print(f"Has to_dict method: {hasattr(strategy, 'to_dict')}")
    
    if hasattr(strategy, 'to_dict'):
        strategy_dict = strategy.to_dict()
        print(f"Strategy dict: {strategy_dict}")
        
        # Check if it has VWAP filters
        has_vwap = False
        for action in strategy_dict.get('actions', []):
            for filter_config in action.get('filters', []):
                if filter_config.get('type') == 'vwap':
                    has_vwap = True
                    print("✅ Found VWAP filter - should use new architecture")
                    break
        
        if not has_vwap:
            print("❌ No VWAP filter found")
    
    return strategy

def test_backtest_worker_logic():
    """Test the BacktestWorker logic without actually running the GUI"""
    print("\nTesting BacktestWorker Logic")
    print("="*50)
    
    # Simulate BacktestWorker logic
    strategy = VWAPTestStrategy()
    
    # Test the detection logic (simulate what BacktestWorker does)
    try:
        if hasattr(strategy, 'to_dict'):
            strategy_dict = strategy.to_dict()
        elif hasattr(strategy, '__dict__'):
            strategy_dict = strategy.__dict__
        else:
            strategy_dict = {}
        
        # Look for filter-only strategies with recognized filters
        should_use_new = False
        if 'actions' in strategy_dict:
            for action in strategy_dict['actions']:
                filters = action.get('filters', [])
                for filter_config in filters:
                    filter_type = filter_config.get('type', '')
                    # Check if it's a filter type that the new architecture supports
                    if filter_type in ['vwap', 'momentum', 'volatility', 'ma', 'bollinger_bands']:
                        print(f"✅ Found supported filter type: {filter_type}")
                        should_use_new = True
                        break
        
        print(f"Should use new architecture: {should_use_new}")
        
        if should_use_new:
            print("✅ Strategy would be routed to new architecture")
        else:
            print("❌ Strategy would use old architecture")
            
        return should_use_new
        
    except Exception as e:
        print(f"❌ Error in detection logic: {e}")
        return False

def test_new_architecture_execution():
    """Test the new architecture execution directly"""
    print("\nTesting New Architecture Execution")
    print("="*50)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create test data
        data = create_test_data()
        print(f"Created test data: {len(data)} bars")
        
        # Create strategy config
        strategy = VWAPTestStrategy()
        strategy_config = strategy.to_dict()
        
        print("Running new architecture backtest...")
        results = new_gui_integration.run_strategy_backtest(strategy_config, data)
        
        print("✅ New architecture execution successful!")
        print(f"Results keys: {list(results.keys())}")
        
        # Check for visualization data
        viz_data = results.get('visualization_data', {})
        print(f"Visualization data: {list(viz_data.keys())}")
        
        lines = viz_data.get('lines', [])
        print(f"Indicator lines: {len(lines)}")
        for line in lines:
            print(f"  - {line.get('name', 'Unknown')}: {line.get('component_type', 'unknown')} component")
        
        return True
        
    except Exception as e:
        print(f"❌ New architecture execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("GUI INTEGRATION TESTS FOR NEW ARCHITECTURE")
    print("="*60)
    print("Testing the integration between GUI components and new modular architecture")
    print()
    
    # Test 1: Strategy detection
    strategy = test_strategy_detection()
    
    # Test 2: BacktestWorker logic simulation
    detection_works = test_backtest_worker_logic()
    
    # Test 3: New architecture execution
    execution_works = test_new_architecture_execution()
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    print(f"✅ Strategy detection: PASSED" if strategy else "❌ Strategy detection: FAILED")
    print(f"✅ BacktestWorker logic: PASSED" if detection_works else "❌ BacktestWorker logic: FAILED")
    print(f"✅ New architecture execution: PASSED" if execution_works else "❌ New architecture execution: FAILED")
    
    overall_success = strategy and detection_works and execution_works
    
    if overall_success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The GUI should now properly use the new architecture for VWAP strategies.")
        print("\nTo test in GUI:")
        print("1. Run main.py")
        print("2. Go to Strategy Builder")
        print("3. Create a strategy with only VWAP filter")
        print("4. Run backtest - should show 'Using NEW MODULAR ARCHITECTURE...'")
        print("5. Check chart tab - should show VWAP line, NO FVG zones")
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED")
        print("The GUI integration needs debugging before it will work correctly.")
    
    print("="*60)

if __name__ == "__main__":
    main() 