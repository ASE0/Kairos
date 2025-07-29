#!/usr/bin/env python3
"""
Test VWAP Filter in Complete Isolation
======================================
This test creates a strategy with ONLY the VWAP filter enabled
(no location strategy, no other filters) to see what actually happens.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_test_simulator import run_gui_compatible_test

def create_vwap_only_strategy():
    """Create a strategy with ONLY VWAP filter enabled"""
    strategy = {
        "name": "VWAP Only Strategy",
        "actions": [
            {
                "name": "vwap_only",
                "filters": [
                    {
                        "type": "vwap",
                        "condition": "above"
                    }
                ]
            }
        ],
        "combination_logic": "AND",
        "gates_and_logic": {},  # No location gates for pure filter strategy
        "location_gate_params": {}
    }
    return strategy

def create_test_data():
    """Create test data with clear VWAP patterns"""
    times = pd.date_range('2024-03-07 09:00:00', periods=50, freq='1min')
    
    # Create data with clear VWAP patterns
    data = []
    base_price = 100.0
    
    for i in range(50):
        if i < 20:
            # Price below VWAP
            price = base_price - 2 + i * 0.1
        elif i < 35:
            # Price above VWAP
            price = base_price + 2 + i * 0.1
        else:
            # Price below VWAP again
            price = base_price - 1 + i * 0.1
        
        data.append({
            'datetime': times[i],
            'open': price - 0.5,
            'high': price + 0.5,
            'low': price - 0.5,
            'close': price,
            'volume': 1000 + np.random.randint(-100, 100)
        })
    
    return pd.DataFrame(data)

def test_vwap_standalone():
    """Test VWAP filter in complete isolation"""
    print("TESTING VWAP FILTER IN COMPLETE ISOLATION")
    print("="*60)
    print("This test creates a strategy with ONLY the VWAP filter enabled.")
    print("No location strategy, no other filters.")
    print()
    
    # Create strategy with only VWAP filter
    strategy = create_vwap_only_strategy()
    
    # Create test data
    data = create_test_data()
    
    # Save test files
    with open('vwap_only_strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)
    
    data.to_csv('vwap_test_data.csv', index=False)
    
    print("✅ Test files created:")
    print("  - vwap_only_strategy.json")
    print("  - vwap_test_data.csv")
    
    # Show strategy configuration
    print(f"\n📋 Strategy Configuration:")
    print(f"  - Actions: {len(strategy['actions'])}")
    print(f"  - Action 1: {strategy['actions'][0]['name']}")
    print(f"  - Filters: {len(strategy['actions'][0]['filters'])}")
    print(f"  - Filter 1: {strategy['actions'][0]['filters'][0]}")
    print(f"  - Location Strategy: {'None' if 'location_strategy' not in strategy['actions'][0] else strategy['actions'][0]['location_strategy']}")
    
    # Run test
    print(f"\n🚀 Running VWAP-only test...")
    
    try:
        result = run_gui_compatible_test(
            strategy_config=strategy,
            data_path='vwap_test_data.csv',
            output_path='vwap_only_results.json'
        )
        
        print("✅ Test completed successfully!")
        print(f"📊 Results saved to: vwap_only_results.json")
        
        # Analyze results
        if result:
            print(f"\n📈 Test Results:")
            print(f"  - Total trades: {len(result.get('trades', []))}")
            print(f"  - Total signals: {result.get('total_signals', 0)}")
            print(f"  - Zones detected: {len(result.get('zones', []))}")
            
            if result.get('trades'):
                print(f"  - First trade: {result['trades'][0]}")
            
            # Check if zones were created (should be 0 for VWAP-only)
            zones = result.get('zones', [])
            if zones:
                print(f"⚠️  WARNING: {len(zones)} zones detected (should be 0 for VWAP-only)")
                print(f"   Zone types: {[z.get('type', 'unknown') for z in zones]}")
            else:
                print(f"✅ No zones detected (correct for VWAP-only)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in VWAP-only test: {e}")
        return False

def main():
    """Main test function"""
    print("VWAP FILTER STANDALONE TEST")
    print("="*60)
    print("This test verifies that the VWAP filter works independently.")
    print("Expected behavior:")
    print("  - No FVG zones should appear")
    print("  - VWAP line should be visible")
    print("  - Signals should be based purely on price vs VWAP")
    print("  - Trades should be generated based on VWAP conditions")
    print()
    
    success = test_vwap_standalone()
    
    print("\n" + "="*60)
    if success:
        print("🎉 VWAP-only test completed!")
        print("📊 Check the results to see if VWAP filter works independently.")
        print("🔍 Look for:")
        print("  - No FVG zones in chart")
        print("  - VWAP line visible")
        print("  - Signals based on price vs VWAP")
    else:
        print("❌ VWAP-only test failed")
    
    print("="*60)

if __name__ == "__main__":
    main() 