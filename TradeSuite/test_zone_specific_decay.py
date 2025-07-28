#!/usr/bin/env python3
"""
Test script for zone-specific decay parameters
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
        if i == 20:  # Bullish FVG
            prices.append(prev_price + 5)  # Gap up
        elif i == 21:
            prices.append(prev_price + 2)  # Continue up
        elif i == 40:  # Bearish FVG
            prices.append(prev_price - 5)  # Gap down
        elif i == 41:
            prices.append(prev_price - 2)  # Continue down
        # Create impulse moves for Order Block detection
        elif i == 60:  # Sharp down impulse
            prices.append(prev_price - 3)  # Sharp down
        elif i == 61:
            prices.append(prev_price - 2)  # Continue down
        elif i == 80:  # Sharp up impulse
            prices.append(prev_price + 3)  # Sharp up
        elif i == 81:
            prices.append(prev_price + 2)  # Continue up
        else:
            # Random walk
            change = np.random.normal(0, 0.5)
            prices.append(max(50, prev_price + change))
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Create realistic OHLC from close price
        high = close + abs(np.random.normal(0, 0.3))
        low = close - abs(np.random.normal(0, 0.3))
        open_price = np.random.uniform(low, high)
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_zone_specific_decay():
    """Test that zone-specific decay parameters are working correctly"""
    print("Testing zone-specific decay parameters...")
    
    # Create test data
    data = create_test_data()
    print(f"Created test data with {len(data)} bars")
    
    # Create strategy with zone-specific parameters
    strategy = PatternStrategy(
        name="Zone Decay Test",
        actions=[],
        location_gate_params={
            # FVG parameters
            'fvg_gamma': 0.90,  # Faster decay
            'fvg_tau_bars': 30,
            'fvg_drop_threshold': 0.02,
            
            # Order Block parameters
            'ob_gamma': 0.95,  # Slower decay
            'ob_tau_bars': 100,
            'ob_drop_threshold': 0.01,
            
            # VWAP parameters
            'vwap_gamma': 0.85,  # Very fast decay
            'vwap_tau_bars': 10,
            'vwap_drop_threshold': 0.05,
            
            # Imbalance parameters
            'imbalance_gamma': 0.98,  # Very slow decay
            'imbalance_tau_bars': 150,
            'imbalance_drop_threshold': 0.005,
            
            # Global settings
            'bar_interval_minutes': 5,
        }
    )
    
    # Test zone decay calculations for each zone type
    zone_types = ['FVG', 'OrderBlock', 'VWAP', 'Imbalance']
    
    print("\nTesting zone decay calculations:")
    print("-" * 60)
    
    for zone_type in zone_types:
        print(f"\n{zone_type} Zone Decay:")
        
        # Test zone_is_active method
        for bars in [5, 10, 20, 50, 100]:
            is_active = strategy.zone_is_active(bars, zone_type)
            strength = strategy.calculate_zone_strength(bars, 1.0, zone_type)
            print(f"  Bars {bars:3d}: Active={is_active}, Strength={strength:.4f}")
    
    # Test zone detection and creation
    print("\nTesting zone detection and creation:")
    print("-" * 60)
    
    # Enable location gate
    strategy.gates_and_logic = {'location_gate': True}
    
    # Evaluate strategy to create zones
    signals, action_details = strategy.evaluate(data)
    
    print(f"Created {len(strategy.simple_zones)} zones")
    
    # Check that zones have zone-specific decay parameters
    for i, zone in enumerate(strategy.simple_zones[:5]):  # Show first 5 zones
        zone_type = zone.get('zone_type', 'Unknown')
        gamma = zone.get('gamma')
        tau_bars = zone.get('tau_bars')
        drop_threshold = zone.get('drop_threshold')
        
        print(f"Zone {i+1}: {zone_type}")
        print(f"  γ={gamma:.3f}, τ={tau_bars}, drop_threshold={drop_threshold:.3f}")
        
        # Verify parameters match zone type
        if zone_type == 'FVG':
            assert gamma == 0.90, f"FVG gamma should be 0.90, got {gamma}"
            assert tau_bars == 30, f"FVG tau should be 30, got {tau_bars}"
        elif zone_type == 'OrderBlock':
            assert gamma == 0.95, f"OrderBlock gamma should be 0.95, got {gamma}"
            assert tau_bars == 100, f"OrderBlock tau should be 100, got {tau_bars}"
        elif zone_type == 'VWAP':
            assert gamma == 0.85, f"VWAP gamma should be 0.85, got {gamma}"
            assert tau_bars == 10, f"VWAP tau should be 10, got {tau_bars}"
        elif zone_type == 'Imbalance':
            assert gamma == 0.98, f"Imbalance gamma should be 0.98, got {gamma}"
            assert tau_bars == 150, f"Imbalance tau should be 150, got {tau_bars}"
    
    print("\n✅ All zone-specific decay parameter tests passed!")
    
    # Test decay over time
    print("\nTesting decay over time:")
    print("-" * 60)
    
    for zone_type in zone_types:
        print(f"\n{zone_type} decay over time:")
        initial_strength = 1.0
        
        for bars in range(0, 51, 5):
            strength = strategy.calculate_zone_strength(bars, initial_strength, zone_type)
            is_active = strategy.zone_is_active(bars, zone_type)
            print(f"  Bars {bars:2d}: Strength={strength:.4f}, Active={is_active}")
    
    return True

if __name__ == "__main__":
    try:
        test_zone_specific_decay()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 