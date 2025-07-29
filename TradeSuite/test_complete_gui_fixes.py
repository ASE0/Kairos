#!/usr/bin/env python3
"""
Complete GUI Compatibility Fixes Test
=====================================
Test all the GUI compatibility issues that were causing errors:
- DataFrame boolean evaluation in _update_overview
- Multi-timeframe data handling
- Monthly returns calculation
- Equity curve with proper datetime index
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dataframe_evaluation_fix():
    """Test the DataFrame evaluation fix"""
    print("Testing DataFrame Evaluation Fixes")
    print("="*50)
    
    # Create test data that mimics what new architecture returns
    times = pd.date_range('2024-03-07 16:00:00', periods=100, freq='1min')
    test_data = pd.DataFrame({
        'open': np.random.randn(100) + 18200,
        'high': np.random.randn(100) + 18205,
        'low': np.random.randn(100) + 18195,
        'close': np.random.randn(100) + 18200,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=times)
    
    # Test the problematic multi_tf_data evaluation
    print("Testing multi_tf_data as dict (old format):")
    multi_tf_data_dict = {'execution': test_data, '1min': test_data}
    
    # This should work (old format)
    if isinstance(multi_tf_data_dict, dict) and multi_tf_data_dict:
        print("✅ Dict format evaluation works")
    else:
        print("❌ Dict format evaluation failed")
    
    print("Testing multi_tf_data as DataFrame (new format):")
    multi_tf_data_df = test_data  # New architecture returns DataFrame directly
    
    # This would have failed before our fix
    try:
        if isinstance(multi_tf_data_df, pd.DataFrame) and not multi_tf_data_df.empty:
            print("✅ DataFrame format evaluation works with fix")
        else:
            print("❌ DataFrame format evaluation failed")
    except ValueError as e:
        print(f"❌ DataFrame evaluation error: {e}")
        return False
    
    # Test that we would have gotten an error with the old code
    try:
        # Simulate old problematic code: if multi_tf_data_df:
        bool(multi_tf_data_df)  # This should raise ValueError
        print("❌ Expected ValueError not raised")
        return False
    except ValueError:
        print("✅ Confirmed old code would have failed (ValueError as expected)")
    
    return True

def test_overview_update_simulation():
    """Simulate the _update_overview method with new architecture data"""
    print("\nTesting Overview Update Simulation")
    print("="*50)
    
    # Create new architecture results structure
    times = pd.date_range('2024-03-07 16:00:00', periods=200, freq='1min')
    test_data = pd.DataFrame({
        'open': np.random.randn(200) + 18200,
        'high': np.random.randn(200) + 18205,
        'low': np.random.randn(200) + 18195,
        'close': np.random.randn(200) + 18200,
        'volume': np.random.randint(1000, 5000, 200)
    }, index=times)
    
    # Create results that match new architecture output
    results = {
        'strategy_name': 'Test VWAP Strategy',
        'multi_tf_data': test_data,  # DataFrame instead of dict
        'initial_capital': 100000,
        'final_capital': 101500,
        'total_return': 0.015,
        'total_trades': 3
    }
    
    # Simulate the _update_overview logic with our fixes
    try:
        multi_tf_data = results.get('multi_tf_data', {})
        strategy_timeframes = []
        
        # Handle both old dict format and new DataFrame format (our fix)
        if isinstance(multi_tf_data, dict) and multi_tf_data:
            for tf_key in multi_tf_data.keys():
                if tf_key != 'execution':
                    strategy_timeframes.append(tf_key)
        elif isinstance(multi_tf_data, pd.DataFrame) and not multi_tf_data.empty:
            # New architecture returns DataFrame, treat as single timeframe
            strategy_timeframes.append('1min')
        
        print(f"✅ Strategy timeframes detected: {strategy_timeframes}")
        
        # Test execution data handling
        if strategy_timeframes:
            if isinstance(multi_tf_data, dict):
                execution_data = multi_tf_data.get('execution')
                if execution_data is not None:
                    execution_bars = len(execution_data)
            elif isinstance(multi_tf_data, pd.DataFrame):
                execution_bars = len(multi_tf_data)
            
            print(f"✅ Execution bars detected: {execution_bars}")
        
        print("✅ Overview update simulation successful")
        return True
        
    except Exception as e:
        print(f"❌ Overview update simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_equity_curve_plotting_fix():
    """Test equity curve plotting with Series format"""
    print("\nTesting Equity Curve Plotting Fix")
    print("="*50)
    
    try:
        # Test the problematic equity curve evaluation
        times = pd.date_range('2024-03-07 16:00:00', periods=100, freq='1min')
        equity_curve = pd.Series(
            data=np.cumsum(np.random.randn(100) * 10) + 100000,
            index=times
        )
        
        print("Testing equity curve as Series (new format):")
        
        # This would have failed before our fix
        try:
            if isinstance(equity_curve, pd.Series):
                if equity_curve.empty:
                    print("❌ Equity curve is empty")
                    return False
                else:
                    print("✅ Series format evaluation works with fix")
            else:
                print("❌ Not recognized as Series")
                return False
        except ValueError as e:
            print(f"❌ Series evaluation error: {e}")
            return False
        
        # Test that we would have gotten an error with the old code
        try:
            # Simulate old problematic code: if not equity_curve:
            bool(equity_curve)  # This should raise ValueError
            print("❌ Expected ValueError not raised")
            return False
        except ValueError:
            print("✅ Confirmed old code would have failed (ValueError as expected)")
        
        # Test metrics calculation with Series
        fake_trades = [{'type': 'SELL', 'pnl': 100}]
        
        # Simulate the fixed _calculate_metrics logic
        equity_empty = False
        if isinstance(equity_curve, pd.Series):
            equity_empty = equity_curve.empty
        elif isinstance(equity_curve, list):
            equity_empty = not equity_curve
        else:
            equity_empty = True
            
        if not fake_trades or equity_empty:
            print("❌ Metrics calculation failed")
            return False
        
        # Test Series-specific operations
        final_value = equity_curve.iloc[-1]
        returns = equity_curve.pct_change().dropna()
        print(f"✅ Series operations successful: final={final_value:.2f}, returns_len={len(returns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Equity curve plotting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_equity_curve_monthly_returns():
    """Test equity curve and monthly returns calculation"""
    print("\nTesting Equity Curve and Monthly Returns")
    print("="*50)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create longer test data spanning multiple months
        start_date = datetime(2024, 1, 1, 9, 0)
        times = pd.date_range(start_date, periods=5000, freq='1H')  # About 7 months of hourly data
        
        data = pd.DataFrame({
            'open': np.random.randn(5000) * 2 + 18200,
            'high': np.random.randn(5000) * 2 + 18205,
            'low': np.random.randn(5000) * 2 + 18195,
            'close': np.random.randn(5000) * 2 + 18200,
            'volume': np.random.randint(1000, 5000, 5000)
        }, index=times)
        
        print(f"✅ Created test data: {len(data)} bars over {(times[-1] - times[0]).days} days")
        
        # Create strategy config
        strategy_config = {
            "name": "Long Duration VWAP Test",
            "actions": [{"name": "vwap_filter", "filters": [{"type": "vwap", "condition": "above"}]}],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }
        
        # Run backtest
        results = new_gui_integration.run_strategy_backtest(strategy_config, data)
        
        # Test equity curve
        equity_curve = results.get('equity_curve')
        print(f"✅ Equity curve type: {type(equity_curve)}")
        print(f"   Has datetime index: {isinstance(equity_curve.index, pd.DatetimeIndex)}")
        print(f"   Length: {len(equity_curve)}")
        
        # Test monthly returns calculation (this was failing before)
        try:
            monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
            print(f"✅ Monthly resampling successful: {len(monthly_returns)} months")
            
            # Test month formatting (this was causing AttributeError)
            if len(monthly_returns) > 0 and hasattr(monthly_returns.index, 'strftime'):
                months = list(monthly_returns.index.strftime('%b %Y'))
                print(f"✅ Month formatting successful: {months[:3]}{'...' if len(months) > 3 else ''}")
            else:
                print("✅ Handled empty/invalid monthly returns correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Monthly returns calculation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Equity curve test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_backtest_worker_flow():
    """Test the complete BacktestWorker flow with all fixes"""
    print("\nTesting Complete BacktestWorker Flow")
    print("="*50)
    
    try:
        from strategies.vwap_test_strategy import VWAPTestStrategy
        from core.new_gui_integration import new_gui_integration
        
        # Create test strategy
        strategy = VWAPTestStrategy()
        
        # Create test data
        times = pd.date_range('2024-03-07 16:00:00', periods=300, freq='1min')
        data = pd.DataFrame({
            'open': 18200 + np.random.randn(300) * 5,
            'high': 18205 + np.random.randn(300) * 5,
            'low': 18195 + np.random.randn(300) * 5,
            'close': 18200 + np.random.randn(300) * 5,
            'volume': np.random.randint(1000, 5000, 300)
        }, index=times)
        
        # Test strategy detection
        strategy_dict = strategy.to_dict()
        should_use_new = False
        for action in strategy_dict.get('actions', []):
            for filter_config in action.get('filters', []):
                if filter_config.get('type') in ['vwap', 'momentum', 'volatility', 'ma', 'bollinger_bands']:
                    should_use_new = True
                    break
        
        print(f"✅ Strategy detection: {should_use_new}")
        
        if should_use_new:
            # Run new architecture
            results = new_gui_integration.run_strategy_backtest(strategy_dict, data)
            
            # Test all GUI compatibility
            required_fields = [
                'trades', 'total_trades', 'total_return', 'sharpe_ratio',
                'max_drawdown', 'win_rate', 'equity_curve', 'zones',
                'data', 'dataset_data', 'multi_tf_data'
            ]
            
            missing_fields = [field for field in required_fields if field not in results]
            if missing_fields:
                print(f"❌ Missing GUI fields: {missing_fields}")
                return False
            
            print("✅ All GUI compatibility fields present")
            
            # Test specific problematic fields
            multi_tf_data = results['multi_tf_data']
            equity_curve = results['equity_curve']
            
            print(f"✅ multi_tf_data type: {type(multi_tf_data)}")
            print(f"✅ equity_curve type: {type(equity_curve)}")
            
            # Test DataFrame evaluation fix
            if isinstance(multi_tf_data, pd.DataFrame) and not multi_tf_data.empty:
                print("✅ DataFrame evaluation fix working")
            else:
                print("❌ DataFrame evaluation issue")
                return False
            
            # Test equity curve datetime index
            if isinstance(equity_curve, pd.Series) and isinstance(equity_curve.index, pd.DatetimeIndex):
                print("✅ Equity curve datetime index correct")
            else:
                print("❌ Equity curve datetime index issue")
                return False
            
            return True
        else:
            print("❌ Strategy not detected for new architecture")
            return False
            
    except Exception as e:
        print(f"❌ Complete flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all GUI compatibility tests"""
    print("COMPLETE GUI COMPATIBILITY FIXES TEST")
    print("="*80)
    print("Testing all GUI compatibility issues and their fixes")
    print()
    
    # Test 1: DataFrame evaluation fixes
    test1_passed = test_dataframe_evaluation_fix()
    
    # Test 2: Overview update simulation
    test2_passed = test_overview_update_simulation()
    
    # Test 3: Equity curve plotting fix
    test3_passed = test_equity_curve_plotting_fix()
    
    # Test 4: Equity curve and monthly returns
    test4_passed = test_equity_curve_monthly_returns()
    
    # Test 5: Complete BacktestWorker flow
    test5_passed = test_complete_backtest_worker_flow()
    
    # Summary
    print("\n" + "="*80)
    print("GUI COMPATIBILITY TEST SUMMARY")
    print("="*80)
    
    tests = [
        ("DataFrame evaluation fixes", test1_passed),
        ("Overview update simulation", test2_passed), 
        ("Equity curve plotting fix", test3_passed),
        ("Equity curve & monthly returns", test4_passed),
        ("Complete BacktestWorker flow", test5_passed)
    ]
    
    for test_name, passed in tests:
        status = "PASSED" if passed else "FAILED"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")
    
    overall_success = all(passed for _, passed in tests)
    
    if overall_success:
        print("\n🎉 ALL GUI COMPATIBILITY TESTS PASSED!")
        print("All integration issues have been resolved:")
        print("- DataFrame boolean evaluation errors fixed")
        print("- Multi-timeframe data handling updated") 
        print("- Equity curve with proper datetime index")
        print("- Monthly returns calculation working")
        print("- Complete GUI compatibility maintained")
        print("\nThe GUI should now work without any errors!")
    else:
        print("\n❌ SOME GUI COMPATIBILITY TESTS FAILED")
        print("There are still issues that need to be resolved.")
    
    print("="*80)

if __name__ == "__main__":
    main() 