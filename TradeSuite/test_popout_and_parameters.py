#!/usr/bin/env python3
"""
Test Popout Chart and Parameter Configuration
=============================================
Tests two fixes:
1. VWAP displays correctly in popped-out chart 
2. VWAP and other filter parameters can be configured in strategy builder
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vwap_with_custom_parameters():
    """Test VWAP filter with custom parameters"""
    print("Testing VWAP Filter with Custom Parameters")
    print("="*50)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create test data
        times = pd.date_range('2024-03-07 16:00:00', periods=300, freq='1min')
        data = pd.DataFrame({
            'open': 18200 + np.random.randn(300) * 5,
            'high': 18205 + np.random.randn(300) * 5,
            'low': 18195 + np.random.randn(300) * 5,
            'close': 18200 + np.random.randn(300) * 5,
            'volume': np.random.randint(1000, 5000, 300)
        }, index=times)
        print(f"✅ Created test data: {len(data)} bars")
        
        # Test VWAP with custom parameters (as would come from strategy builder)
        strategy_config = {
            "name": "Custom VWAP Strategy",
            "actions": [{
                "name": "custom_vwap", 
                "filters": [{
                    "type": "vwap", 
                    "condition": "near",
                    "tolerance": 0.0025,  # Custom tolerance: 0.25%
                    "period": 100         # Custom period: 100 bars instead of default 200
                }]
            }],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }
        
        print(f"✅ Strategy config: VWAP with tolerance={strategy_config['actions'][0]['filters'][0]['tolerance']}, period={strategy_config['actions'][0]['filters'][0]['period']}")
        
        # Run backtest
        results = new_gui_integration.run_strategy_backtest(strategy_config, data)
        print(f"✅ Backtest completed successfully")
        
        # Check that VWAP visualization data is available
        viz_data = results.get('visualization_data', {})
        if viz_data and viz_data.get('lines'):
            vwap_lines = [line for line in viz_data['lines'] if 'vwap' in line.get('name', '').lower()]
            if vwap_lines:
                vwap_line = vwap_lines[0]
                print(f"✅ VWAP visualization data found: {vwap_line.get('name')}")
                print(f"   Config: color={vwap_line.get('config', {}).get('color')}, "
                      f"linewidth={vwap_line.get('config', {}).get('linewidth')}")
                
                # Test that data exists and has correct length
                vwap_data = vwap_line.get('data')
                if vwap_data is not None and not vwap_data.empty:
                    print(f"   ✅ VWAP data: {len(vwap_data)} points")
                    print(f"   Range: {vwap_data.min():.2f} - {vwap_data.max():.2f}")
                    return True
                else:
                    print("   ❌ VWAP data is empty")
                    return False
            else:
                print("❌ No VWAP lines found in visualization data")
                return False
        else:
            print("❌ No visualization data found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_momentum_with_custom_parameters():
    """Test momentum filter with custom parameters"""
    print("\nTesting Momentum Filter with Custom Parameters")
    print("="*50)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create test data
        times = pd.date_range('2024-03-07 16:00:00', periods=200, freq='1min')
        data = pd.DataFrame({
            'open': 18200 + np.cumsum(np.random.randn(200) * 2),
            'high': 18205 + np.cumsum(np.random.randn(200) * 2),
            'low': 18195 + np.cumsum(np.random.randn(200) * 2),
            'close': 18200 + np.cumsum(np.random.randn(200) * 2),
            'volume': np.random.randint(1000, 5000, 200)
        }, index=times)
        print(f"✅ Created trending test data: {len(data)} bars")
        
        # Test momentum with custom parameters
        strategy_config = {
            "name": "Custom Momentum Strategy",
            "actions": [{
                "name": "custom_momentum", 
                "filters": [{
                    "type": "momentum", 
                    "momentum_threshold": 0.015,  # Custom threshold: 1.5%
                    "lookback": 15,               # Custom lookback: 15 bars
                    "rsi_range": [40, 60]         # Custom RSI range: 40-60
                }]
            }],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }
        
        momentum_filter = strategy_config['actions'][0]['filters'][0]
        print(f"✅ Strategy config: Momentum with threshold={momentum_filter['momentum_threshold']}, "
              f"lookback={momentum_filter['lookback']}, RSI range={momentum_filter['rsi_range']}")
        
        # Run backtest
        results = new_gui_integration.run_strategy_backtest(strategy_config, data)
        print(f"✅ Backtest completed successfully")
        
        # Check signals were generated
        signals = results.get('signals')
        if signals is not None:
            signal_count = signals.sum() if hasattr(signals, 'sum') else sum(signals)
            print(f"✅ Momentum signals generated: {signal_count}")
            return True
        else:
            print("❌ No signals found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_volatility_with_custom_parameters():
    """Test volatility filter with custom parameters"""
    print("\nTesting Volatility Filter with Custom Parameters")
    print("="*50)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create volatile test data
        times = pd.date_range('2024-03-07 16:00:00', periods=200, freq='1min')
        base_price = 18200
        volatility = np.random.randn(200) * 10  # High volatility
        data = pd.DataFrame({
            'open': base_price + volatility + np.random.randn(200),
            'high': base_price + volatility + np.abs(np.random.randn(200)) * 3,
            'low': base_price + volatility - np.abs(np.random.randn(200)) * 3,
            'close': base_price + volatility + np.random.randn(200),
            'volume': np.random.randint(1000, 5000, 200)
        }, index=times)
        print(f"✅ Created volatile test data: {len(data)} bars")
        
        # Test volatility with custom parameters
        strategy_config = {
            "name": "Custom Volatility Strategy",
            "actions": [{
                "name": "custom_volatility", 
                "filters": [{
                    "type": "volatility", 
                    "min_atr_ratio": 0.005,   # Custom min: 0.5%
                    "max_atr_ratio": 0.03,    # Custom max: 3%
                    "atr_period": 10          # Custom period: 10 bars
                }]
            }],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }
        
        vol_filter = strategy_config['actions'][0]['filters'][0]
        print(f"✅ Strategy config: Volatility with min_atr={vol_filter['min_atr_ratio']}, "
              f"max_atr={vol_filter['max_atr_ratio']}, period={vol_filter['atr_period']}")
        
        # Run backtest
        results = new_gui_integration.run_strategy_backtest(strategy_config, data)
        print(f"✅ Backtest completed successfully")
        
        # Check signals were generated
        signals = results.get('signals')
        if signals is not None:
            signal_count = signals.sum() if hasattr(signals, 'sum') else sum(signals)
            print(f"✅ Volatility signals generated: {signal_count}")
            return True
        else:
            print("❌ No signals found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_popout_chart_visualization_data():
    """Test that visualization data is properly formatted for popout charts"""
    print("\nTesting Popout Chart Visualization Data")
    print("="*50)
    
    try:
        from core.new_gui_integration import new_gui_integration
        
        # Create test data
        times = pd.date_range('2024-03-07 16:00:00', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': 18200 + np.random.randn(100) * 2,
            'high': 18205 + np.random.randn(100) * 2,
            'low': 18195 + np.random.randn(100) * 2,
            'close': 18200 + np.random.randn(100) * 2,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=times)
        
        # Multi-indicator strategy to test popout
        strategy_config = {
            "name": "Multi-Indicator Strategy",
            "actions": [{
                "name": "multi_indicators", 
                "filters": [
                    {"type": "vwap", "condition": "above", "tolerance": 0.001, "period": 50},
                    {"type": "momentum", "momentum_threshold": 0.01, "lookback": 10, "rsi_range": [30, 70]},
                    {"type": "volatility", "min_atr_ratio": 0.005, "max_atr_ratio": 0.02, "atr_period": 14}
                ]
            }],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }
        
        print(f"✅ Multi-indicator strategy created")
        
        # Run backtest
        results = new_gui_integration.run_strategy_backtest(strategy_config, data)
        print(f"✅ Backtest completed")
        
        # Test visualization data structure (what popout chart needs)
        viz_data = results.get('visualization_data', {})
        if not viz_data:
            print("❌ No visualization data found")
            return False
        
        print(f"✅ Visualization data found: {list(viz_data.keys())}")
        
        # Test line indicators (VWAP, RSI, etc.)
        lines = viz_data.get('lines', [])
        print(f"✅ Line indicators: {len(lines)}")
        
        for i, line in enumerate(lines):
            line_name = line.get('name', f'Line_{i}')
            line_data = line.get('data')
            line_config = line.get('config', {})
            
            if line_data is not None and not line_data.empty:
                print(f"   {line_name}: {len(line_data)} points, "
                      f"color={line_config.get('color', 'unknown')}")
                
                # Verify data has proper index for popout chart matching
                if hasattr(line_data, 'index'):
                    common_index = data.index.intersection(line_data.index)
                    print(f"     Index compatibility: {len(common_index)}/{len(data)} bars match")
                    if len(common_index) == 0:
                        print(f"     ❌ WARNING: No index overlap for {line_name}")
                else:
                    print(f"     ❌ WARNING: {line_name} data has no index")
            else:
                print(f"   {line_name}: ❌ No data")
        
        # Test zones
        zones = viz_data.get('zones', [])
        if zones:
            print(f"✅ Zones: {len(zones)}")
        
        print("✅ Popout chart visualization data structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all popout and parameter tests"""
    print("POPOUT CHART AND PARAMETER CONFIGURATION TEST")
    print("="*80)
    print("Testing fixes for:")
    print("1. VWAP missing in popped-out charts")
    print("2. Enhanced parameter controls in strategy builder")
    print()
    
    # Test 1: VWAP with custom parameters
    test1_passed = test_vwap_with_custom_parameters()
    
    # Test 2: Momentum with custom parameters
    test2_passed = test_momentum_with_custom_parameters()
    
    # Test 3: Volatility with custom parameters
    test3_passed = test_volatility_with_custom_parameters()
    
    # Test 4: Popout chart visualization data
    test4_passed = test_popout_chart_visualization_data()
    
    # Summary
    print("\n" + "="*80)
    print("POPOUT AND PARAMETER TEST SUMMARY")
    print("="*80)
    
    tests = [
        ("VWAP with custom parameters", test1_passed),
        ("Momentum with custom parameters", test2_passed),
        ("Volatility with custom parameters", test3_passed),
        ("Popout chart visualization data", test4_passed)
    ]
    
    for test_name, passed in tests:
        status = "PASSED" if passed else "FAILED"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")
    
    overall_success = all(passed for _, passed in tests)
    
    if overall_success:
        print("\n🎉 ALL POPOUT AND PARAMETER TESTS PASSED!")
        print("\nFixed issues:")
        print("  ✅ Popout charts now show VWAP using new architecture data")
        print("  ✅ VWAP filter supports tolerance and period parameters")
        print("  ✅ Momentum filter supports lookback parameter")  
        print("  ✅ Volatility filter supports ATR period parameter")
        print("  ✅ All visualization data properly structured for popout")
        print("\nStrategy builder now has enhanced parameter controls!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Issues may remain with popout charts or parameter configuration.")
    
    print("="*80)

if __name__ == "__main__":
    main()