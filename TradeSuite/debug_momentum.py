#!/usr/bin/env python3
"""
Debug Momentum Filter
====================
This script helps debug why the momentum filter is not working.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import Action

def debug_momentum():
    """Debug the momentum filter implementation"""
    print("DEBUGGING MOMENTUM FILTER")
    print("="*50)
    
    # Create test data with clear momentum
    times = pd.date_range('2024-03-07 09:00:00', periods=20, freq='1min')
    data = pd.DataFrame({
        'open': [100 + i * 0.5 for i in range(20)],
        'high': [101 + i * 0.5 for i in range(20)],
        'low': [99 + i * 0.5 for i in range(20)],
        'close': [100.5 + i * 0.5 for i in range(20)],
        'volume': [1000] * 20
    }, index=times)
    
    print("Test Data:")
    print(data.head())
    print(f"Data shape: {data.shape}")
    
    # Calculate returns manually
    returns = data['close'].pct_change()
    print(f"\nReturns (first 10):")
    print(returns.head(10))
    
    # Test momentum calculation manually
    lookback = 10
    momentum_threshold = 0.02
    
    print(f"\nManual Momentum Calculation:")
    print(f"Lookback: {lookback}")
    print(f"Threshold: {momentum_threshold}")
    
    for i in range(lookback, min(lookback + 5, len(data))):
        recent_returns = returns.iloc[i-lookback:i]
        momentum = np.mean(recent_returns)
        print(f"Bar {i}: momentum = {momentum:.6f}, abs = {abs(momentum):.6f}, threshold = {momentum_threshold}")
        print(f"  Recent returns: {recent_returns.values}")
    
    # Test the filter
    action = Action(
        name="test_momentum",
        filters=[{
            'type': 'momentum',
            'momentum_threshold': 0.02,
            'rsi_range': [30, 70]
        }]
    )
    
    print(f"\nTesting Filter:")
    filter_signals = action._apply_filter(data, {
        'type': 'momentum',
        'momentum_threshold': 0.02,
        'rsi_range': [30, 70]
    })
    
    print(f"Filter signals: {filter_signals.sum()}")
    print(f"Signal indices: {filter_signals[filter_signals].index.tolist()}")
    
    # Calculate RSI manually to see the values
    print(f"\nManual RSI Calculation:")
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    print(f"RSI values (all): {rsi.values}")
    print(f"RSI range: [{rsi.min():.2f}, {rsi.max():.2f}]")
    print(f"RSI in [30,70] range: {((rsi >= 30) & (rsi <= 70)).sum()}")
    
    # Try with wider RSI range
    print(f"\nTesting with wider RSI range [0, 100]:")
    filter_signals_wide = action._apply_filter(data, {
        'type': 'momentum',
        'momentum_threshold': 0.001,
        'rsi_range': [0, 100]
    })
    
    print(f"Filter signals (wide RSI): {filter_signals_wide.sum()}")
    print(f"Signal indices: {filter_signals_wide[filter_signals_wide].index.tolist()}")

if __name__ == "__main__":
    debug_momentum() 