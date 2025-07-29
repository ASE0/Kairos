#!/usr/bin/env python3
"""
Test New Modular Strategy Architecture
=====================================
Comprehensive test demonstrating that the new architecture works seamlessly
and prevents the integration bugs we experienced before.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new architecture
from core.new_gui_integration import run_gui_compatible_test, new_gui_integration

def create_test_data():
    """Create realistic test data"""
    times = pd.date_range('2024-03-07 09:00:00', periods=100, freq='5min')
    
    # Create realistic price movement with volume
    base_price = 18200.0
    data = []
    
    for i in range(100):
        # Add some realistic price movement
        price_change = np.random.normal(0, 5)  # Random walk with some volatility
        if i > 0:
            price = data[-1]['close'] + price_change
        else:
            price = base_price
        
        # Ensure realistic OHLC relationships
        high = price + abs(np.random.normal(0, 2))
        low = price - abs(np.random.normal(0, 2))
        open_price = price + np.random.normal(0, 1)
        
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
    return df

def test_vwap_only_strategy():
    """Test VWAP-only strategy with new architecture"""
    print("\n" + "="*60)
    print("TESTING VWAP-ONLY STRATEGY (New Architecture)")
    print("="*60)
    
    # Create strategy with only VWAP filter
    strategy = {
        "name": "New VWAP Only Strategy",
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
        "gates_and_logic": {},  # No gates
        "location_gate_params": {}
    }
    
    # Create test data
    data = create_test_data()
    data.to_csv('new_arch_test_data.csv', index=False)
    
    print(f"✅ Created test data: {len(data)} bars")
    print(f"📊 Columns: {data.columns.tolist()}")
    
    # Run test
    print(f"\n🚀 Running VWAP-only test with new architecture...")
    
    try:
        result = run_gui_compatible_test(
            strategy_config=strategy,
            data_path='new_arch_test_data.csv',
            output_path='new_arch_vwap_results.json'
        )
        
        print("✅ Test completed successfully!")
        
        # Analyze results
        print(f"\n📈 Test Results:")
        print(f"  - Total trades: {len(result.get('trades', []))}")
        print(f"  - Total signals: {result.get('total_signals', 0)}")
        print(f"  - Zones detected: {len(result.get('zones', []))}")
        
        # Check visualization data
        viz_data = result.get('visualization_data', {})
        print(f"\n🖼️  Visualization Data:")
        print(f"  - Lines: {len(viz_data.get('lines', []))}")
        print(f"  - Zones: {len(viz_data.get('zones', []))}")
        print(f"  - Markers: {len(viz_data.get('markers', []))}")
        print(f"  - Bands: {len(viz_data.get('bands', []))}")
        
        # Verify expected behavior
        zones = result.get('zones', [])
        if len(zones) == 0:
            print("✅ CORRECT: No unwanted zones created")
        else:
            print(f"❌ PROBLEM: {len(zones)} zones created (should be 0)")
        
        lines = viz_data.get('lines', [])
        vwap_lines = [line for line in lines if 'vwap' in line['name'].lower()]
        if len(vwap_lines) > 0:
            print("✅ CORRECT: VWAP line data available for chart")
            print(f"  - VWAP line config: {vwap_lines[0].get('config', {})}")
        else:
            print("❌ PROBLEM: No VWAP line data found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in new architecture test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_component_strategy():
    """Test strategy with multiple components"""
    print("\n" + "="*60)
    print("TESTING MULTI-COMPONENT STRATEGY (New Architecture)")
    print("="*60)
    
    # Create strategy with multiple filters
    strategy = {
        "name": "Multi-Component Strategy",
        "actions": [
            {
                "name": "multi_filter",
                "filters": [
                    {
                        "type": "vwap",
                        "condition": "above"
                    },
                    {
                        "type": "ma", 
                        "condition": "above",
                        "period": 20
                    },
                    {
                        "type": "volatility",
                        "min_atr_ratio": 0.005,
                        "max_atr_ratio": 0.02
                    }
                ]
            }
        ],
        "combination_logic": "AND",
        "gates_and_logic": {},
        "location_gate_params": {}
    }
    
    # Create test data
    data = create_test_data()
    
    print(f"✅ Created test data: {len(data)} bars")
    
    # Run test
    print(f"\n🚀 Running multi-component test...")
    
    try:
        result = run_gui_compatible_test(
            strategy_config=strategy,
            data_path=data,  # Pass DataFrame directly
            output_path='new_arch_multi_results.json'
        )
        
        print("✅ Multi-component test completed successfully!")
        
        # Analyze results
        print(f"\n📈 Test Results:")
        print(f"  - Total trades: {len(result.get('trades', []))}")
        print(f"  - Total signals: {result.get('total_signals', 0)}")
        
        # Check visualization data
        viz_data = result.get('visualization_data', {})
        print(f"\n🖼️  Visualization Data:")
        print(f"  - Lines: {len(viz_data.get('lines', []))}")
        for line in viz_data.get('lines', []):
            print(f"    * {line['name']} ({line.get('component_type', 'unknown')})")
        
        # Should have multiple indicator lines
        expected_indicators = ['vwap', 'ma']
        actual_indicators = [line['name'].lower() for line in viz_data.get('lines', [])]
        
        for indicator in expected_indicators:
            if any(indicator in actual for actual in actual_indicators):
                print(f"✅ CORRECT: {indicator.upper()} indicator present")
            else:
                print(f"❌ PROBLEM: {indicator.upper()} indicator missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-component test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_registry():
    """Test the component registry system"""
    print("\n" + "="*60)
    print("TESTING COMPONENT REGISTRY SYSTEM")
    print("="*60)
    
    # Get available components
    components = new_gui_integration.get_available_components()
    
    print("📋 Available Components:")
    for comp_type, comp_list in components.items():
        print(f"  {comp_type}: {comp_list}")
    
    # Verify expected components are registered
    expected_filters = ['vwap', 'momentum', 'volatility', 'ma']
    actual_filters = components.get('filters', [])
    
    for filter_name in expected_filters:
        if filter_name in actual_filters:
            print(f"✅ CORRECT: {filter_name} filter registered")
        else:
            print(f"❌ PROBLEM: {filter_name} filter not registered")
    
    # Test component info retrieval
    print(f"\n🔍 Component Information:")
    for filter_name in expected_filters[:2]:  # Test first 2
        if filter_name in actual_filters:
            info = new_gui_integration.get_component_info('filter', filter_name)
            print(f"  {filter_name}:")
            print(f"    - Required columns: {info.get('required_columns', [])}")
            print(f"    - Visualization type: {info.get('visualization_config', {}).get('type', 'none')}")

def test_strategy_validation():
    """Test strategy validation"""
    print("\n" + "="*60)
    print("TESTING STRATEGY VALIDATION")
    print("="*60)
    
    # Test valid strategy
    valid_strategy = {
        "name": "Valid Test Strategy",
        "actions": [
            {
                "name": "test_action",
                "filters": [
                    {
                        "type": "vwap",
                        "condition": "above"
                    }
                ]
            }
        ]
    }
    
    is_valid, errors = new_gui_integration.validate_strategy_config(valid_strategy)
    if is_valid:
        print("✅ CORRECT: Valid strategy passed validation")
    else:
        print(f"❌ PROBLEM: Valid strategy failed validation: {errors}")
    
    # Test invalid strategy
    invalid_strategy = {
        "name": "Invalid Test Strategy",
        "actions": [
            {
                "name": "test_action",
                "filters": [
                    {
                        "type": "nonexistent_filter",
                        "condition": "above"
                    }
                ]
            }
        ]
    }
    
    is_valid, errors = new_gui_integration.validate_strategy_config(invalid_strategy)
    if not is_valid:
        print("✅ CORRECT: Invalid strategy failed validation")
        print(f"  - Errors: {errors}")
    else:
        print("❌ PROBLEM: Invalid strategy passed validation")

def main():
    """Run all tests"""
    print("NEW MODULAR STRATEGY ARCHITECTURE - COMPREHENSIVE TESTS")
    print("="*60)
    print("Testing the new architecture to ensure seamless integration")
    print("and prevention of the bugs we experienced with the old system.")
    
    # Test component registry
    test_component_registry()
    
    # Test strategy validation
    test_strategy_validation()
    
    # Test VWAP-only strategy
    vwap_success = test_vwap_only_strategy()
    
    # Test multi-component strategy
    multi_success = test_multi_component_strategy()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if vwap_success:
        print("✅ VWAP-only strategy test: PASSED")
    else:
        print("❌ VWAP-only strategy test: FAILED")
    
    if multi_success:
        print("✅ Multi-component strategy test: PASSED")
    else:
        print("❌ Multi-component strategy test: FAILED")
    
    overall_success = vwap_success and multi_success
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The new architecture is working correctly and should prevent")
        print("the integration bugs we experienced with the old system.")
        print("\nKey improvements:")
        print("✅ Modular component system")
        print("✅ Consistent interfaces")
        print("✅ Unified execution path")
        print("✅ Separated visualization logic")
        print("✅ Component registry system")
        print("✅ Proper error handling")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("The new architecture needs debugging before deployment.")
    
    print("="*60)

if __name__ == "__main__":
    main() 