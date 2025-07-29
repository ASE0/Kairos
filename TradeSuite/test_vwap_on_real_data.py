#!/usr/bin/env python3
"""
Test VWAP Filter on Real NQ Dataset
===================================
This test creates a VWAP-only strategy and runs it on the actual NQ_5s_1m dataset
to verify what appears in the chart tab.
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
        "name": "VWAP Only Test Strategy",
        "actions": [
            {
                "name": "vwap_filter_only",
                "filters": [
                    {
                        "type": "vwap",
                        "condition": "above"
                    }
                ]
                # NO location_strategy - this should be pure filter-only
            }
        ],
        "combination_logic": "AND",
        "gates_and_logic": {},  # No location gates for pure filter strategy
        "location_gate_params": {}
    }
    return strategy

def test_vwap_on_real_data():
    """Test VWAP filter on real NQ dataset"""
    print("TESTING VWAP FILTER ON REAL NQ DATASET")
    print("="*60)
    print("Dataset: NQ_5s_1m_2024-03-07.csv")
    print("Expected: VWAP line should appear in chart tab, NO FVG zones")
    print()
    
    # Load the real NQ dataset
    dataset_file = 'NQ_5s_1m_2024-03-07.csv'
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return False
    
    data = pd.read_csv(dataset_file)
    print(f"✅ Loaded dataset: {len(data)} bars")
    print(f"📊 Columns: {data.columns.tolist()}")
    print(f"📅 Date range: {data['datetime'].iloc[0]} to {data['datetime'].iloc[-1]}")
    print(f"📈 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    print(f"📊 Volume range: {data['volume'].min():,} - {data['volume'].max():,}")
    
    # Use only 1 day of data as requested
    data_subset = data.head(288)  # Approximately 1 day (288 x 5min = 1440min = 24hrs)
    print(f"📅 Using subset: {len(data_subset)} bars (approximately 1 day)")
    
    # Save subset for testing
    data_subset.to_csv('nq_1day_test.csv', index=False)
    
    # Create strategy with only VWAP filter
    strategy = create_vwap_only_strategy()
    
    # Save strategy
    with open('vwap_real_test_strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)
    
    print(f"\n📋 Strategy Configuration:")
    print(f"  - Actions: {len(strategy['actions'])}")
    print(f"  - Action 1: {strategy['actions'][0]['name']}")
    print(f"  - Filters: {len(strategy['actions'][0]['filters'])}")
    print(f"  - Filter 1: {strategy['actions'][0]['filters'][0]}")
    print(f"  - Location Strategy: {'None' if 'location_strategy' not in strategy['actions'][0] else strategy['actions'][0]['location_strategy']}")
    print(f"  - Gates: {strategy['gates_and_logic']}")
    
    # Calculate VWAP manually to verify expected behavior
    vwap = (data_subset['close'] * data_subset['volume']).cumsum() / data_subset['volume'].cumsum()
    price_above_vwap = (data_subset['close'] > vwap).sum()
    total_bars = len(data_subset)
    
    print(f"\n📊 Manual VWAP Analysis:")
    print(f"  - Total bars: {total_bars}")
    print(f"  - Bars where price > VWAP: {price_above_vwap}")
    print(f"  - Percentage above VWAP: {(price_above_vwap/total_bars)*100:.1f}%")
    print(f"  - VWAP range: ${vwap.min():.2f} - ${vwap.max():.2f}")
    
    # Run test
    print(f"\n🚀 Running VWAP test on real NQ data...")
    
    try:
        result = run_gui_compatible_test(
            strategy_config=strategy,
            data_path='nq_1day_test.csv',
            output_path='vwap_real_test_results.json'
        )
        
        print("✅ Test completed successfully!")
        print(f"📊 Results saved to: vwap_real_test_results.json")
        
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
                print(f"   Zone types: {[z.get('type', 'unknown') for z in zones[:5]]}")
                if len(zones) > 5:
                    print(f"   ... and {len(zones)-5} more zones")
            else:
                print(f"✅ No zones detected (correct for VWAP-only)")
            
            # Check what chart tab would show
            print(f"\n🖼️  Chart Tab Analysis:")
            print(f"  - Dataset has volume column: {'volume' in data_subset.columns}")
            print(f"  - Dataset length > 20: {len(data_subset) > 20}")
            print(f"  - VWAP line should appear: {('volume' in data_subset.columns) and (len(data_subset) > 20)}")
            print(f"  - Zones to plot: {len(zones)}")
            
            if len(zones) > 0:
                print(f"  ❌ PROBLEM: Chart will show zones instead of pure VWAP!")
            else:
                print(f"  ✅ GOOD: Chart should show VWAP line only!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in VWAP test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("VWAP FILTER TEST ON REAL NQ DATASET")
    print("="*60)
    print("This test verifies VWAP filter behavior on actual market data.")
    print("Expected behavior:")
    print("  - No FVG zones should be created")
    print("  - VWAP line should be visible in chart tab")
    print("  - Signals should be based purely on price vs VWAP")
    print("  - Strategy should work as pure filter-only")
    print()
    
    success = test_vwap_on_real_data()
    
    print("\n" + "="*60)
    if success:
        print("🎉 VWAP real data test completed!")
        print("📊 Now check the chart tab in the GUI to verify:")
        print("  1. VWAP line is visible (purple line)")
        print("  2. NO FVG zones are shown")
        print("  3. Signals are based on price vs VWAP")
    else:
        print("❌ VWAP real data test failed")
    
    print("="*60)

if __name__ == "__main__":
    main() 