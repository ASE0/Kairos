#!/usr/bin/env python3
"""
Debug Filter Test
================
Simple test to understand why filters are not generating signals when combined.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import Action, PatternStrategy

def create_test_data():
    """Create simple test data"""
    np.random.seed(42)
    n_bars = 100
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.01, n_bars)  # 1% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = price * (1 + np.random.normal(0, 0.002))
        close_price = price
        volume = np.random.randint(100, 1000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01', periods=len(df), freq='1min')
    return df

def test_single_filter():
    """Test a single filter to see if it generates signals"""
    print("=== Testing Single Filter ===")
    
    data = create_test_data()
    print(f"Test data created: {len(data)} bars")
    
    # Test VWAP filter only
    action = Action(
        name="VWAP_Test",
        filters=[
            {
                'type': 'vwap',
                'tolerance': 0.01,  # 1% tolerance
                'condition': 'near'
            }
        ]
    )
    
    signals = action.apply(data)
    print(f"VWAP filter signals: {signals.sum()}/{len(signals)} ({signals.sum()/len(signals)*100:.2f}%)")
    
    # Test momentum filter only
    action2 = Action(
        name="Momentum_Test",
        filters=[
            {
                'type': 'momentum',
                'momentum_threshold': 0.0001,  # Very low threshold
                'lookback': 5,
                'rsi_range': [20, 80]  # Very wide range
            }
        ]
    )
    
    signals2 = action2.apply(data)
    print(f"Momentum filter signals: {signals2.sum()}/{len(signals2)} ({signals2.sum()/len(signals2)*100:.2f}%)")
    
    return signals.sum() > 0 or signals2.sum() > 0

def test_filter_combination():
    """Test combining filters to see where the issue is"""
    print("\n=== Testing Filter Combination ===")
    
    data = create_test_data()
    
    # Test VWAP + Momentum with OR logic
    action = Action(
        name="VWAP_Momentum_OR",
        filters=[
            {
                'type': 'vwap',
                'tolerance': 0.01,
                'condition': 'near'
            },
            {
                'type': 'momentum',
                'momentum_threshold': 0.0001,
                'lookback': 5,
                'rsi_range': [20, 80]
            }
        ]
    )
    
    # Create a strategy with OR logic
    strategy = PatternStrategy(
        name="OR_Test",
        actions=[action],
        combination_logic='OR'
    )
    
    signals, action_details = strategy.evaluate(data)
    print(f"OR combination signals: {signals.sum()}/{len(signals)} ({signals.sum()/len(signals)*100:.2f}%)")
    
    return signals.sum() > 0

if __name__ == "__main__":
    print("DEBUG FILTER TEST")
    print("=" * 50)
    
    # Test single filters
    single_success = test_single_filter()
    
    # Test filter combination
    combination_success = test_filter_combination()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Single filters working: {'YES' if single_success else 'NO'}")
    print(f"Filter combination working: {'YES' if combination_success else 'NO'}")
    
    if not single_success:
        print("❌ Single filters are not generating signals")
    elif not combination_success:
        print("❌ Filter combination is not working")
    else:
        print("✅ Filters are working correctly")