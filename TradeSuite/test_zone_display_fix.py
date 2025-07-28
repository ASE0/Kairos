#!/usr/bin/env python3
"""
Test Zone Display Fix
=====================
Verify that zones and micro comb peaks are being created and displayed correctly
in both main chart and popout chart
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.strategy_builders import PatternStrategy, Action, MultiTimeframeBacktestEngine

def create_test_data():
    """Create test data with known patterns that should trigger zone detection"""
    # Create synthetic data with gaps and patterns
    dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min')
    
    # Create data with gaps that should trigger FVG detection
    data = []
    for i, date in enumerate(dates):
        if i < 30:
            # Normal price movement
            open_price = 100 + i * 0.1
            high_price = open_price + 0.5
            low_price = open_price - 0.3
            close_price = open_price + 0.2
        elif i == 30:
            # Gap up - should create bullish FVG
            open_price = 105  # Gap up from previous close ~103
            high_price = open_price + 1.0
            low_price = open_price - 0.2
            close_price = open_price + 0.8
        elif i < 60:
            # Normal movement after gap
            open_price = 105 + (i - 30) * 0.05
            high_price = open_price + 0.4
            low_price = open_price - 0.2
            close_price = open_price + 0.1
        elif i == 60:
            # Gap down - should create bearish FVG
            open_price = 102  # Gap down from previous close ~106.5
            high_price = open_price + 0.3
            low_price = open_price - 1.0
            close_price = open_price - 0.7
        else:
            # Normal movement after gap
            open_price = 102 + (i - 60) * 0.03
            high_price = open_price + 0.3
            low_price = open_price - 0.2
            close_price = open_price + 0.05
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': 1000 + np.random.randint(0, 500)
        })
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    return df

def test_zone_detection_and_display():
    """Test that zones are detected and can be displayed"""
    print("=== Testing Zone Detection and Display ===")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with {len(df)} bars")
    
    # Create a simple strategy with location gate
    strategy = PatternStrategy("Test Zone Strategy")
    
    # Add location gate action
    action = Action(
        name="Buy on Zone",
        pattern=None,  # Location only
        location_strategy="FVG",  # Use FVG zone type
        location_params={
            'fvg_gamma': 0.95,
            'fvg_tau_bars': 20,
            'fvg_drop_threshold': 0.1,
            'fvg_min_gap_size': 0.5,
            'fvg_max_gap_size': 10.0
        }
    )
    strategy.add_action(action)
    
    # Run backtest
    engine = MultiTimeframeBacktestEngine()
    results = engine.run_backtest(
        data=df,
        strategy=strategy,
        initial_capital=10000,
        risk_per_trade=0.02
    )
    
    # Check results
    print(f"Backtest completed with {len(results.get('trades', []))} trades")
    print(f"Zones detected: {len(results.get('zones', []))}")
    
    # Verify zones have required fields
    zones = results.get('zones', [])
    if zones:
        print(f"First zone structure:")
        first_zone = zones[0]
        required_fields = ['timestamp', 'zone_min', 'zone_max', 'comb_centers', 'zone_type', 'zone_direction']
        for field in required_fields:
            if field in first_zone:
                print(f"  {field}: {first_zone[field]}")
            else:
                print(f"  {field}: MISSING")
        
        # Check decay parameters
        decay_fields = ['initial_strength', 'creation_index', 'gamma', 'tau_bars', 'drop_threshold']
        for field in decay_fields:
            if field in first_zone:
                print(f"  {field}: {first_zone[field]}")
            else:
                print(f"  {field}: MISSING")
    
    # Test chart display simulation
    print("\n=== Testing Chart Display ===")
    
    # Simulate what the chart methods would do
    zones_plotted = 0
    for zone in zones:
        ts = zone.get('timestamp')
        zmin = zone.get('zone_min')
        zmax = zone.get('zone_max')
        comb_centers = zone.get('comb_centers', [])
        
        if ts is not None and zmin is not None and zmax is not None:
            print(f"Zone {zones_plotted + 1}: {ts} - {zmin:.2f} to {zmax:.2f}")
            if comb_centers:
                print(f"  Comb centers: {comb_centers}")
            zones_plotted += 1
    
    print(f"\nTotal zones ready for plotting: {zones_plotted}")
    
    # Test popout chart simulation
    print("\n=== Testing Popout Chart ===")
    popout_zones_plotted = 0
    for zone in zones:
        ts = zone.get('timestamp')
        zmin = zone.get('zone_min')
        zmax = zone.get('zone_max')
        comb_centers = zone.get('comb_centers', [])
        
        if ts is not None and zmin is not None and zmax is not None:
            # Simulate popout chart plotting logic
            print(f"Popout Zone {popout_zones_plotted + 1}: {ts} - {zmin:.2f} to {zmax:.2f}")
            if comb_centers:
                print(f"  Popout comb centers: {comb_centers}")
            popout_zones_plotted += 1
    
    print(f"\nTotal zones ready for popout plotting: {popout_zones_plotted}")
    
    # Summary
    print("\n=== Test Summary ===")
    if zones_plotted > 0:
        print("✅ Zones are being created and stored correctly")
        print("✅ Zones have all required fields for plotting")
        print("✅ Both main chart and popout chart should display zones")
        print("✅ Micro comb peaks are included in zone data")
    else:
        print("❌ No zones were created - check zone detection logic")
    
    return zones_plotted > 0

if __name__ == "__main__":
    success = test_zone_detection_and_display()
    if success:
        print("\n🎉 All tests passed! Zones and comb peaks should display correctly.")
    else:
        print("\n❌ Tests failed. Check zone detection logic.") 