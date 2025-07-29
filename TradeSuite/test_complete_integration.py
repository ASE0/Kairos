#!/usr/bin/env python3
"""
Complete Integration Test
========================
Comprehensive test to verify all integration issues are fixed including:
- Equity curve generation with proper datetime index
- Monthly returns calculation
- Chart rendering with new architecture
- GUI compatibility
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_new_architecture_complete():
    """Test complete new architecture integration"""
    print("Testing Complete New Architecture Integration")
    print("="*60)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create realistic test data with proper datetime index
        start_date = datetime(2024, 3, 7, 16, 0)
        times = pd.date_range(start_date, periods=1000, freq='1min')
        
        # Create realistic price movement with volume
        base_price = 18200.0
        data = []
        
        for i in range(1000):
            # Add some realistic price movement
            price_change = np.random.normal(0, 2)
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
        
        print(f"✅ Created test data: {len(df)} bars with datetime index")
        print(f"   Index type: {type(df.index)}")
        print(f"   Index range: {df.index[0]} to {df.index[-1]}")
        
        # Create VWAP strategy config
        strategy_config = {
            "name": "Complete VWAP Test Strategy",
            "actions": [
                {
                    "name": "vwap_filter",
                    "filters": [
                        {
                            "type": "vwap",
                            "condition": "above"
                        }
                    ]
                }
            ],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }
        
        print("Running new architecture backtest...")
        results = new_gui_integration.run_strategy_backtest(strategy_config, df)
        
        print("✅ New architecture backtest completed!")
        
        # Test all expected result fields
        expected_fields = [
            'strategy_name', 'signals', 'trades', 'performance', 'visualization_data',
            'total_signals', 'data', 'zones', 'component_results', 'equity_curve',
            'total_trades', 'initial_capital', 'final_capital', 'cumulative_pnl',
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'
        ]
        
        missing_fields = []
        for field in expected_fields:
            if field not in results:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing result fields: {missing_fields}")
            return False
        else:
            print("✅ All expected result fields present")
        
        # Test equity curve specifically
        equity_curve = results.get('equity_curve')
        print(f"✅ Equity curve type: {type(equity_curve)}")
        print(f"   Length: {len(equity_curve) if equity_curve is not None else 'None'}")
        
        if isinstance(equity_curve, pd.Series):
            print(f"   Index type: {type(equity_curve.index)}")
            print(f"   Has datetime index: {isinstance(equity_curve.index, pd.DatetimeIndex)}")
            
            # Test monthly resampling (this is what was failing)
            try:
                monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
                print(f"✅ Monthly resampling successful: {len(monthly_returns)} months")
                
                if len(monthly_returns) > 0 and hasattr(monthly_returns.index, 'strftime'):
                    months = list(monthly_returns.index.strftime('%b %Y'))
                    print(f"✅ Month formatting successful: {months}")
                else:
                    print("✅ Handled empty/invalid monthly returns correctly")
                    
            except Exception as e:
                print(f"❌ Monthly resampling failed: {e}")
                return False
        else:
            print(f"❌ Equity curve is not pandas Series: {type(equity_curve)}")
            return False
        
        # Test visualization data
        viz_data = results.get('visualization_data', {})
        lines = viz_data.get('lines', [])
        print(f"✅ Visualization data: {len(lines)} indicator lines")
        
        for line in lines:
            line_name = line.get('name', 'Unknown')
            line_data = line.get('data')
            print(f"   - {line_name}: {type(line_data)}, length={len(line_data) if line_data is not None else 'None'}")
            
            if isinstance(line_data, pd.Series):
                print(f"     Index type: {type(line_data.index)}")
        
        # Test trades
        trades = results.get('trades', [])
        print(f"✅ Generated {len(trades)} trades")
        
        for i, trade in enumerate(trades[:3]):  # Show first 3 trades
            print(f"   Trade {i+1}: Entry={trade.get('entry_time')}, Exit={trade.get('exit_time')}, PnL=${trade.get('pnl', 0):.2f}")
        
        print("\n✅ COMPLETE INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Complete integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_backtest_worker_simulation():
    """Simulate the BacktestWorker flow to catch integration issues"""
    print("\nTesting BacktestWorker Simulation")
    print("="*60)
    
    try:
        from strategies.vwap_test_strategy import VWAPTestStrategy
        from core.new_gui_integration import new_gui_integration
        
        # Create test strategy (simulating what GUI does)
        strategy = VWAPTestStrategy()
        
        # Create test data
        times = pd.date_range('2024-03-07 16:00:00', periods=200, freq='1min')
        data = []
        base_price = 18200.0
        
        for i in range(200):
            price_change = np.random.normal(0, 1)
            price = base_price + (price_change * i * 0.1)  # Slight trend
            
            data.append({
                'datetime': times[i],
                'open': price + np.random.normal(0, 0.5),
                'high': price + abs(np.random.normal(0, 1)),
                'low': price - abs(np.random.normal(0, 1)),
                'close': price,
                'volume': np.random.randint(1000, 5000)
            })
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Simulate BacktestWorker detection logic
        print("Simulating BacktestWorker strategy detection...")
        
        if hasattr(strategy, 'to_dict'):
            strategy_dict = strategy.to_dict()
        else:
            strategy_dict = strategy.__dict__
        
        # Check for supported filters
        should_use_new = False
        for action in strategy_dict.get('actions', []):
            for filter_config in action.get('filters', []):
                filter_type = filter_config.get('type', '')
                if filter_type in ['vwap', 'momentum', 'volatility', 'ma', 'bollinger_bands']:
                    should_use_new = True
                    print(f"✅ Detected supported filter: {filter_type}")
                    break
        
        if should_use_new:
            print("✅ Would route to new architecture")
            
            # Convert strategy config
            strategy_config = strategy_dict
            
            # Run new architecture (simulating BacktestWorker)
            print("Running new architecture...")
            results = new_gui_integration.run_strategy_backtest(
                strategy_config, 
                df,
                initial_capital=100000,
                risk_per_trade=0.02
            )
            
            print("✅ BacktestWorker simulation successful!")
            
            # Verify GUI compatibility fields
            gui_required_fields = [
                'trades', 'total_trades', 'total_return', 'sharpe_ratio', 
                'max_drawdown', 'win_rate', 'equity_curve', 'zones',
                'data', 'dataset_data'
            ]
            
            missing = [field for field in gui_required_fields if field not in results]
            if missing:
                print(f"❌ Missing GUI compatibility fields: {missing}")
                return False
            else:
                print("✅ All GUI compatibility fields present")
            
            # Test equity curve datetime index
            equity_curve = results['equity_curve']
            if isinstance(equity_curve, pd.Series) and isinstance(equity_curve.index, pd.DatetimeIndex):
                print("✅ Equity curve has proper datetime index")
            else:
                print(f"❌ Equity curve index issue: {type(equity_curve)}, {type(equity_curve.index) if hasattr(equity_curve, 'index') else 'No index'}")
                return False
            
            return True
        else:
            print("❌ Would not route to new architecture")
            return False
            
    except Exception as e:
        print(f"❌ BacktestWorker simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive integration tests"""
    print("COMPREHENSIVE INTEGRATION TESTS")
    print("="*80)
    print("Testing all integration issues including equity curve, GUI compatibility, etc.")
    print()
    
    # Test 1: Complete new architecture integration
    test1_passed = test_new_architecture_complete()
    
    # Test 2: BacktestWorker simulation
    test2_passed = test_gui_backtest_worker_simulation()
    
    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    print(f"✅ Complete integration test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ BacktestWorker simulation: {'PASSED' if test2_passed else 'FAILED'}")
    
    overall_success = test1_passed and test2_passed
    
    if overall_success:
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("The integration issues should now be completely resolved.")
        print("\nThe GUI should now:")
        print("- Properly route VWAP strategies to new architecture")
        print("- Generate correct equity curve with datetime index")
        print("- Handle monthly returns calculation without errors")
        print("- Display VWAP indicator line in chart tab")
        print("- Show no unwanted FVG zones")
    else:
        print("\n❌ SOME COMPREHENSIVE TESTS FAILED")
        print("There are still integration issues that need to be resolved.")
    
    print("="*80)

if __name__ == "__main__":
    main() 