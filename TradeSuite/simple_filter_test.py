#!/usr/bin/env python3
"""
Simple Filter Test - Verify Filters Are Working
==============================================
This script creates a simple test to verify that the filters are working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import Action

def test_vwap_filter():
    """Test VWAP filter implementation"""
    print("\n" + "="*50)
    print("TESTING VWAP FILTER")
    print("="*50)
    
    # Create test data
    times = pd.date_range('2024-03-07 09:00:00', periods=20, freq='1min')
    data = pd.DataFrame({
        'open': [100 + i for i in range(20)],
        'high': [101 + i for i in range(20)],
        'low': [99 + i for i in range(20)],
        'close': [100.5 + i for i in range(20)],
        'volume': [1000] * 20
    }, index=times)
    
    # Create action with VWAP filter
    action = Action(
        name="test_vwap",
        filters=[{
            'type': 'vwap',
            'condition': 'above'
        }]
    )
    
    # Apply filter
    filter_signals = action._apply_filter(data, {'type': 'vwap', 'condition': 'above'})
    
    # Calculate expected VWAP manually
    vwap_expected = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    expected_signals = data['close'] > vwap_expected
    
    print(f"VWAP Filter Test Results:")
    print(f"  - Total bars: {len(data)}")
    print(f"  - Filter signals: {filter_signals.sum()}")
    print(f"  - Expected signals: {expected_signals.sum()}")
    print(f"  - Match: {filter_signals.equals(expected_signals)}")
    
    # Show first few values
    print(f"\nFirst 5 bars comparison:")
    for i in range(5):
        print(f"  Bar {i}: Close={data['close'].iloc[i]:.2f}, VWAP={vwap_expected.iloc[i]:.2f}, "
              f"Filter={filter_signals.iloc[i]}, Expected={expected_signals.iloc[i]}")
    
    return filter_signals.equals(expected_signals)

def test_momentum_filter():
    """Test momentum filter implementation"""
    print("\n" + "="*50)
    print("TESTING MOMENTUM FILTER")
    print("="*50)
    
    # Create test data with momentum
    times = pd.date_range('2024-03-07 09:00:00', periods=20, freq='1min')
    data = pd.DataFrame({
        'open': [100 + i * 0.5 for i in range(20)],
        'high': [101 + i * 0.5 for i in range(20)],
        'low': [99 + i * 0.5 for i in range(20)],
        'close': [100.5 + i * 0.5 for i in range(20)],
        'volume': [1000] * 20
    }, index=times)
    
    # Create action with momentum filter
    action = Action(
        name="test_momentum",
        filters=[{
            'type': 'momentum',
            'momentum_threshold': 0.001,  # Lower threshold for testing
            'rsi_range': [0, 100]  # Full range
        }]
    )
    
    # Apply filter
    filter_signals = action._apply_filter(data, {
        'type': 'momentum',
        'momentum_threshold': 0.001,  # Lower threshold for testing
        'rsi_range': [0, 100]  # Full range
    })
    
    print(f"Momentum Filter Test Results:")
    print(f"  - Total bars: {len(data)}")
    print(f"  - Filter signals: {filter_signals.sum()}")
    print(f"  - Signal rate: {filter_signals.sum() / len(data):.2%}")
    
    return filter_signals.sum() > 0

def test_volatility_filter():
    """Test volatility filter implementation"""
    print("\n" + "="*50)
    print("TESTING VOLATILITY FILTER")
    print("="*50)
    
    # Create test data with volatility
    times = pd.date_range('2024-03-07 09:00:00', periods=20, freq='1min')
    data = pd.DataFrame({
        'open': [100 + np.random.normal(0, 1) for _ in range(20)],
        'high': [101 + np.random.normal(0, 1) for _ in range(20)],
        'low': [99 + np.random.normal(0, 1) for _ in range(20)],
        'close': [100.5 + np.random.normal(0, 1) for _ in range(20)],
        'volume': [1000] * 20
    }, index=times)
    
    # Create action with volatility filter
    action = Action(
        name="test_volatility",
        filters=[{
            'type': 'volatility',
            'min_atr_ratio': 0.01,
            'max_atr_ratio': 0.05
        }]
    )
    
    # Apply filter
    filter_signals = action._apply_filter(data, {
        'type': 'volatility',
        'min_atr_ratio': 0.01,
        'max_atr_ratio': 0.05
    })
    
    print(f"Volatility Filter Test Results:")
    print(f"  - Total bars: {len(data)}")
    print(f"  - Filter signals: {filter_signals.sum()}")
    print(f"  - Signal rate: {filter_signals.sum() / len(data):.2%}")
    
    return filter_signals.sum() > 0

def main():
    """Run all filter tests"""
    print("SIMPLE FILTER TESTING")
    print("="*50)
    
    results = {}
    
    # Test each filter
    try:
        results['vwap'] = test_vwap_filter()
        print(f"✅ VWAP Filter: {'PASS' if results['vwap'] else 'FAIL'}")
    except Exception as e:
        print(f"❌ VWAP Filter: ERROR - {e}")
        results['vwap'] = False
    
    try:
        results['momentum'] = test_momentum_filter()
        print(f"✅ Momentum Filter: {'PASS' if results['momentum'] else 'FAIL'}")
    except Exception as e:
        print(f"❌ Momentum Filter: ERROR - {e}")
        results['momentum'] = False
    
    try:
        results['volatility'] = test_volatility_filter()
        print(f"✅ Volatility Filter: {'PASS' if results['volatility'] else 'FAIL'}")
    except Exception as e:
        print(f"❌ Volatility Filter: ERROR - {e}")
        results['volatility'] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for filter_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{'✅' if result else '❌'} {filter_name.upper()}: {status}")
    
    print(f"\nOverall: {passed}/{total} filters working correctly")
    
    if passed == total:
        print("🎉 All filters are working correctly!")
    else:
        print("⚠️ Some filters need attention.")

if __name__ == "__main__":
    main() 