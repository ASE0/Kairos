#!/usr/bin/env python3
"""
Test script for zone decay system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action

def test_zone_decay_system():
    """Test the zone decay system"""
    print("🧪 Testing Zone Decay System")
    print("=" * 50)
    
    # Create a test strategy
    strategy = PatternStrategy(
        name="Test Zone Decay",
        actions=[],
        gates_and_logic={'location_gate': True}
    )
    
    # Test default parameters
    print("\n📊 Default Zone Decay Parameters:")
    decay_info = strategy.get_zone_decay_info()
    for key, value in decay_info.items():
        print(f"  {key}: {value}")
    
    # Test zone_is_active function
    print("\n🔍 Testing zone_is_active function:")
    for bars in [0, 10, 25, 50, 75, 100]:
        is_active = strategy.zone_is_active(bars)
        strength = strategy.calculate_zone_strength(bars)
        print(f"  Bars since creation: {bars:3d} | Active: {str(is_active):5s} | Strength: {strength:.4f}")
    
    # Test different gamma values
    print("\n🔄 Testing different gamma values:")
    for gamma in [0.90, 0.95, 0.97, 0.99]:
        print(f"\n  Gamma = {gamma}:")
        for bars in [0, 10, 25, 50]:
            strength = strategy.calculate_zone_strength(bars, 1.0, gamma)
            print(f"    Bars {bars:2d}: Strength = {strength:.4f}")
    
    # Test different tau values
    print("\n⏰ Testing different tau values:")
    for tau in [10, 25, 50, 100]:
        print(f"\n  Tau = {tau} bars:")
        for bars in [0, 10, 25, 50, 75, 100]:
            is_active = strategy.zone_is_active(bars, tau=tau)
            print(f"    Bars {bars:3d}: Active = {str(is_active)}")
    
    # Test zone validity in days
    print("\n📅 Testing zone validity in days:")
    for bar_interval in [1, 5, 15, 60, 1440]:  # 1min, 5min, 15min, 1hour, 1day
        for tau in [10, 25, 50, 100]:
            zone_days = strategy.calculate_zone_days_valid(tau, bar_interval)
            print(f"  {bar_interval:4d}min bars, τ={tau:3d}: {zone_days:.2f} days")
    
    # Test with custom parameters
    print("\n🎛️ Testing with custom parameters:")
    strategy.location_gate_params.update({
        'zone_gamma': 0.90,
        'zone_tau_bars': 25,
        'zone_drop_threshold': 0.05,
        'bar_interval_minutes': 5
    })
    
    custom_decay_info = strategy.get_zone_decay_info()
    print("  Custom parameters:")
    for key, value in custom_decay_info.items():
        print(f"    {key}: {value}")
    
    print("\n✅ Zone decay system test completed!")

def test_zone_creation_with_decay():
    """Test zone creation with decay information"""
    print("\n🧪 Testing Zone Creation with Decay")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create strategy with location gate enabled
    strategy = PatternStrategy(
        name="Test Zone Creation",
        actions=[],
        gates_and_logic={'location_gate': True}
    )
    
    # Set custom decay parameters
    strategy.location_gate_params.update({
        'zone_gamma': 0.95,
        'zone_tau_bars': 30,
        'zone_drop_threshold': 0.01,
        'bar_interval_minutes': 5,
        'gate_threshold': 0.0  # Allow all zones for testing
    })
    
    print(f"Strategy parameters: {strategy.location_gate_params}")
    
    # Simulate zone creation at different indices
    test_indices = [10, 25, 50, 75]
    
    for idx in test_indices:
        print(f"\n📍 Testing zone creation at index {idx}:")
        
        # Simulate location gate check (this would normally create zones)
        # For testing, we'll manually create a zone
        zone_data = {
            'timestamp': data.index[idx],
            'zone_min': 100.0,
            'zone_max': 110.0,
            'comb_centers': [102.5, 107.5],
            'initial_strength': 0.8,
            'creation_index': idx,
            'gamma': strategy.location_gate_params['zone_gamma'],
            'tau_bars': strategy.location_gate_params['zone_tau_bars'],
            'drop_threshold': strategy.location_gate_params['zone_drop_threshold'],
            'bar_interval_minutes': strategy.location_gate_params['bar_interval_minutes'],
            'zone_days_valid': strategy.calculate_zone_days_valid(
                strategy.location_gate_params['zone_tau_bars'],
                strategy.location_gate_params['bar_interval_minutes']
            )
        }
        
        strategy.simple_zones.append(zone_data)
        
        print(f"  Created zone: {zone_data}")
        
        # Test zone activity at different future bars
        for future_bars in [0, 5, 15, 30, 45]:
            future_idx = idx + future_bars
            if future_idx < len(data):
                bars_since_creation = future_bars
                is_active = strategy.zone_is_active(
                    bars_since_creation,
                    zone_data['gamma'],
                    zone_data['tau_bars'],
                    zone_data['drop_threshold']
                )
                current_strength = strategy.calculate_zone_strength(
                    bars_since_creation,
                    zone_data['initial_strength'],
                    zone_data['gamma']
                )
                print(f"    Future bar {future_bars:2d}: Active={str(is_active):5s}, Strength={current_strength:.4f}")
    
    print(f"\n📊 Total zones created: {len(strategy.simple_zones)}")
    print("✅ Zone creation with decay test completed!")

if __name__ == "__main__":
    test_zone_decay_system()
    test_zone_creation_with_decay() 