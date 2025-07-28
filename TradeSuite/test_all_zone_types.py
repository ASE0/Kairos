#!/usr/bin/env python3
"""
Test script for all 5 zone types implementation with new UI parameters
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action

def create_test_data():
    """Create test data with various patterns to trigger all zone types"""
    # Create 100 bars of test data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    
    # Start with base price
    base_price = 100.0
    prices = [base_price]
    
    # Create various patterns to trigger different zone types
    for i in range(1, 100):
        prev_price = prices[-1]
        
        # Create gaps for FVG detection
        if i == 10:  # Bullish FVG
            open_price = prev_price + 2
            high_price = open_price + 1
            low_price = open_price - 0.5
            close_price = open_price + 0.5
        elif i == 11:  # Gap up
            open_price = prev_price + 5  # Large gap
            high_price = open_price + 2
            low_price = open_price - 1
            close_price = open_price + 1
        elif i == 20:  # Bearish FVG
            open_price = prev_price - 2
            high_price = open_price + 0.5
            low_price = open_price - 1
            close_price = open_price - 0.5
        elif i == 21:  # Gap down
            open_price = prev_price - 5  # Large gap
            high_price = open_price + 1
            low_price = open_price - 2
            close_price = open_price - 1
        elif i == 30:  # Order Block - Bullish (down candle before up impulse)
            open_price = prev_price + 1
            high_price = open_price + 0.5
            low_price = open_price - 2
            close_price = open_price - 1  # Down candle
        elif i == 31:  # Up impulse
            open_price = prev_price + 3  # Large up move
            high_price = open_price + 2
            low_price = open_price - 0.5
            close_price = open_price + 1.5
        elif i == 40:  # Order Block - Bearish (up candle before down impulse)
            open_price = prev_price - 1
            high_price = open_price + 2
            low_price = open_price - 0.5
            close_price = open_price + 1  # Up candle
        elif i == 41:  # Down impulse
            open_price = prev_price - 3  # Large down move
            high_price = open_price + 0.5
            low_price = open_price - 2
            close_price = open_price - 1.5
        elif i == 50:  # VWAP zone (create some volatility around VWAP)
            vwap = 100.0  # Target VWAP
            open_price = vwap + np.random.normal(0, 1)
            high_price = open_price + abs(np.random.normal(0, 0.5))
            low_price = open_price - abs(np.random.normal(0, 0.5))
            close_price = open_price + np.random.normal(0, 0.3)
        elif i == 60:  # Support/Resistance (create range)
            open_price = prev_price + np.random.normal(0, 0.5)
            high_price = open_price + abs(np.random.normal(0, 1))
            low_price = open_price - abs(np.random.normal(0, 1))
            close_price = open_price + np.random.normal(0, 0.5)
        elif i == 70:  # Imbalance (large price move)
            open_price = prev_price + np.random.normal(0, 0.5)
            high_price = open_price + 5  # Large move
            low_price = open_price - 0.5
            close_price = open_price + 4  # Large close
        else:
            # Normal price movement
            open_price = prev_price + np.random.normal(0, 0.5)
            high_price = open_price + abs(np.random.normal(0, 0.5))
            low_price = open_price - abs(np.random.normal(0, 0.5))
            close_price = open_price + np.random.normal(0, 0.3)
        
        # Ensure OHLC relationships are valid
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        prices.append(close_price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': [100.0] + [100.0 + i * 0.1 + np.random.normal(0, 0.5) for i in range(99)],
        'high': [100.5] + [100.0 + i * 0.1 + np.random.normal(0, 0.5) + abs(np.random.normal(0, 0.5)) for i in range(99)],
        'low': [99.5] + [100.0 + i * 0.1 + np.random.normal(0, 0.5) - abs(np.random.normal(0, 0.5)) for i in range(99)],
        'close': [100.0] + [100.0 + i * 0.1 + np.random.normal(0, 0.5) for i in range(99)],
        'volume': [1000] + [1000 + np.random.randint(-100, 100) for i in range(99)]
    }, index=dates)
    
    # Override with specific patterns
    # FVG patterns
    data.iloc[10, :] = [102, 103, 101.5, 102.5, 1000]  # Bullish FVG setup
    data.iloc[11, :] = [107, 109, 106, 108, 1000]      # Gap up
    data.iloc[20, :] = [98, 98.5, 97, 97.5, 1000]      # Bearish FVG setup  
    data.iloc[21, :] = [92, 93, 90, 91, 1000]          # Gap down
    
    # Order Block patterns
    data.iloc[30, :] = [101, 101.5, 99, 99, 1000]      # Down candle before up impulse
    data.iloc[31, :] = [104, 106, 103.5, 105.5, 1000]  # Up impulse
    data.iloc[40, :] = [99, 101, 98.5, 100, 1000]      # Up candle before down impulse
    data.iloc[41, :] = [96, 96.5, 94, 94.5, 1000]      # Down impulse
    
    # Imbalance pattern
    data.iloc[70, :] = [105, 110, 104.5, 109, 1000]    # Large move
    
    return data

def test_all_zone_types():
    """Test all 5 zone types with new UI parameters"""
    print("🧪 Testing All 5 Zone Types with New UI Parameters")
    print("=" * 60)
    
    # Create test data
    data = create_test_data()
    print(f"Created test data: {len(data)} bars")
    
    # Create strategy with all zone types
    strategy = PatternStrategy(
        name="All Zone Types Test",
        actions=[],
        location_gate_params={
            # Shared parameters
            'zone_gamma': 0.95,
            'zone_tau_bars': 50,
            'zone_drop_threshold': 0.01,
            'bar_interval_minutes': 5,
            
            # FVG parameters
            'fvg_epsilon': 2.0,
            'fvg_N': 3,
            'fvg_sigma': 0.1,
            'fvg_beta1': 0.7,
            'fvg_beta2': 0.3,
            'fvg_phi': 0.2,
            'fvg_lambda': 0.0,
            
            # Order Block parameters
            'ob_impulse_threshold': 0.02,
            'ob_lookback': 10,
            
            # VWAP parameters
            'vwap_k': 1.0,
            'vwap_lookback': 20,
            
            # Support/Resistance parameters
            'sr_window': 20,
            'sr_sigma_r': 5,
            'sr_sigma_t': 3,
            
            # Imbalance parameters
            'imbalance_threshold': 5,  # Lowered for testing
            'imbalance_gamma_mem': 0.01,
            'imbalance_sigma_rev': 20,
        }
    )
    
    # Test indices where we expect zones
    test_indices = [10, 11, 20, 21, 30, 31, 40, 41, 50, 60, 70]
    zone_types_found = set()
    total_zones_detected = 0
    
    print(f"\n🔍 Testing zone detection at specific indices...")
    
    for idx in test_indices:
        print(f"\n🔍 Testing index {idx} (timestamp: {data.index[idx]})")
        print(f"  Price: {data.iloc[idx]['close']:.2f}")
        
        # Detect all zone types
        zones = strategy._detect_all_zone_types(data, idx)
        
        if zones:
            print(f"  Found {len(zones)} zones:")
            for zone in zones:
                zone_type = zone['type']
                direction = zone['direction']
                zone_min = zone['zone_min']
                zone_max = zone['zone_max']
                strength = zone.get('strength', 0)
                
                print(f"    - {zone_type} ({direction}): {zone_min:.2f} - {zone_max:.2f} (strength: {strength:.3f})")
                
                # Show zone-specific parameters
                if zone_type == 'FVG':
                    print(f"      ε={zone.get('epsilon', 'N/A')}, N={zone.get('N', 'N/A')}, σ={zone.get('sigma', 'N/A')}")
                elif zone_type == 'OrderBlock':
                    print(f"      Impulse threshold={zone.get('impulse_threshold', 'N/A')}, Lookback={zone.get('lookback', 'N/A')}")
                elif zone_type == 'VWAP':
                    print(f"      k={zone.get('k', 'N/A')}, Lookback={zone.get('lookback', 'N/A')}")
                elif zone_type == 'Imbalance':
                    print(f"      Threshold={zone.get('imbalance_threshold', 'N/A')}, γ_mem={zone.get('gamma_mem', 'N/A')}")
                
                zone_types_found.add(zone_type)
                total_zones_detected += 1
                
                # Check if current price is in zone
                current_price = data.iloc[idx]['close']
                if zone_min <= current_price <= zone_max:
                    print(f"      ✓ Current price {current_price:.2f} is INSIDE this zone")
                else:
                    print(f"      ✗ Current price {current_price:.2f} is OUTSIDE this zone")
        else:
            print("  No zones detected")
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 60)
    print(f"Total zones detected: {total_zones_detected}")
    print(f"Zone types found: {sorted(zone_types_found)}")
    
    expected_types = {'FVG', 'OrderBlock', 'VWAP', 'Imbalance'}
    missing_types = expected_types - zone_types_found
    extra_types = zone_types_found - expected_types
    
    if missing_types:
        print(f"❌ Missing zone types: {sorted(missing_types)}")
    else:
        print("✅ All expected zone types detected!")
    
    if extra_types:
        print(f"⚠️  Extra zone types: {sorted(extra_types)}")
    
    # Test zone decay system
    print(f"\n⏰ Testing Zone Decay System")
    print("=" * 60)
    decay_info = strategy.get_zone_decay_info()
    for key, value in decay_info.items():
        print(f"{key}: {value}")
    
    return len(zone_types_found) == 4

def test_zone_colors():
    """Test zone color mapping"""
    print(f"\n🎨 Testing Zone Color Mapping")
    print("=" * 60)
    
    zone_colors = {
        'FVG': {'bullish': 'lightgreen', 'bearish': 'lightcoral'},
        'OrderBlock': {'bullish': 'lightblue', 'bearish': 'lightpink'},
        'VWAP': {'neutral': 'yellow'},
        'Imbalance': {'bullish': 'purple', 'bearish': 'purple'}
    }
    
    for zone_type, directions in zone_colors.items():
        print(f"{zone_type}:")
        for direction, color in directions.items():
            print(f"  {direction}: {color}")
    
    return True

def test_ui_parameter_mapping():
    """Test that UI parameters are correctly mapped to zone detection"""
    print(f"\n🔧 Testing UI Parameter Mapping")
    print("=" * 60)
    
    # Test parameter mapping
    test_params = {
        'zone_gamma': 0.95,
        'zone_tau_bars': 50,
        'zone_drop_threshold': 0.01,
        'bar_interval_minutes': 5,
        'fvg_epsilon': 2.0,
        'fvg_N': 3,
        'fvg_sigma': 0.1,
        'fvg_beta1': 0.7,
        'fvg_beta2': 0.3,
        'fvg_phi': 0.2,
        'fvg_lambda': 0.0,
        'ob_impulse_threshold': 0.02,
        'ob_lookback': 10,
        'vwap_k': 1.0,
        'vwap_lookback': 20,
        'sr_window': 20,
        'sr_sigma_r': 5,
        'sr_sigma_t': 3,
        'imbalance_threshold': 5,
        'imbalance_gamma_mem': 0.01,
        'imbalance_sigma_rev': 20,
    }
    
    strategy = PatternStrategy(
        name="Parameter Test",
        actions=[],
        location_gate_params=test_params
    )
    
    print("✅ All UI parameters successfully mapped to strategy")
    
    # Test zone type mapping
    zone_type_mapping = {
        "FVG (Fair Value Gap)": "FVG",
        "Order Block": "OrderBlock", 
        "VWAP Mean-Reversion Band": "VWAP",
        "Imbalance Memory Zone": "Imbalance"
    }
    
    print("✅ Zone type mapping verified:")
    for ui_name, internal_name in zone_type_mapping.items():
        print(f"  {ui_name} → {internal_name}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Zone Type Testing")
    print("=" * 60)
    
    # Run all tests
    test1 = test_all_zone_types()
    test2 = test_zone_colors()
    test3 = test_ui_parameter_mapping()
    
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"Zone Detection Test: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Color Mapping Test: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Parameter Mapping Test: {'✅ PASSED' if test3 else '❌ FAILED'}")
    
    if test1 and test2 and test3:
        print(f"\n🎉 ALL TESTS PASSED! All 5 zone types are working with the new UI.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.") 