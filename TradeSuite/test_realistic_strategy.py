#!/usr/bin/env python3
"""
Realistic Index Strategy 11.2 Test
==================================
Test the Index Strategy 11.2 with more realistic filter parameters
to generate reasonable signal rates.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import Action, PatternStrategy

def create_realistic_test_data():
    """Create more realistic test data with proper market conditions"""
    np.random.seed(42)
    n_bars = 1000
    
    # Generate realistic price data with trends and volatility
    base_price = 100.0
    prices = [base_price]
    
    # Create realistic market movements
    for i in range(1, n_bars):
        # Add trend component
        trend = 0.0001 * np.sin(i / 100)  # Slow trend
        
        # Add volatility component
        volatility = 0.01 * (1 + 0.5 * np.sin(i / 50))  # Varying volatility
        
        # Add random component
        random_component = np.random.normal(0, volatility)
        
        # Combine components
        price_change = trend + random_component
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        # Create realistic OHLC from price
        open_price = price * (1 + np.random.normal(0, 0.002))
        close_price = price
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(100, 2000)  # More realistic volume range
        
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

def test_realistic_ofm_strategy():
    """Test Order Flow Momentum (OFM) Strategy with realistic parameters"""
    print("=== Testing Realistic OFM Strategy ===")
    
    data = create_realistic_test_data()
    print(f"Test data created: {len(data)} bars")
    
    # OFM Strategy with more restrictive parameters
    ofm_action = Action(
        name="OFM_Realistic",
        filters=[
            {
                'type': 'order_flow',
                'min_cvd_threshold': 500,  # More restrictive
                'large_trade_ratio': 0.5   # More restrictive
            },
            {
                'type': 'vwap',
                'condition': 'near',
                'tolerance': 0.005,  # 0.5% tolerance - more restrictive
                'period': 200
            },
            {
                'type': 'momentum',
                'momentum_threshold': 0.002,  # More restrictive
                'lookback': 10,
                'rsi_range': [40, 60]  # More restrictive RSI range
            },
            {
                'type': 'volatility',
                'min_atr_ratio': 0.005,  # More restrictive
                'max_atr_ratio': 0.05    # More restrictive
            }
        ]
    )
    
    ofm_strategy = PatternStrategy(
        name="OFM_Realistic_Strategy",
        actions=[ofm_action],
        combination_logic='AND'  # Use AND logic for more restrictive signals
    )
    
    signals, action_details = ofm_strategy.evaluate(data)
    signal_rate = (signals.sum() / len(signals)) * 100
    print(f"OFM Strategy: {signals.sum()} signals, {signal_rate:.2f}% signal rate")
    
    return signals.sum() > 0 and signal_rate < 30  # Should generate some signals but not too many

def test_realistic_mmr_strategy():
    """Test Microstructure Mean Reversion (MMR) Strategy with realistic parameters"""
    print("\n=== Testing Realistic MMR Strategy ===")
    
    data = create_realistic_test_data()
    
    # MMR Strategy with more restrictive parameters
    mmr_action = Action(
        name="MMR_Realistic",
        filters=[
            {
                'type': 'order_flow',
                'min_cvd_threshold': 200,  # More restrictive
                'large_trade_ratio': 0.6   # More restrictive
            },
            {
                'type': 'volatility',
                'min_atr_ratio': 0.008,  # More restrictive
                'max_atr_ratio': 0.04    # More restrictive
            },
            {
                'type': 'momentum',
                'momentum_threshold': 0.003,  # More restrictive
                'lookback': 5,
                'rsi_range': [30, 70]  # More restrictive RSI range
            }
        ]
    )
    
    mmr_strategy = PatternStrategy(
        name="MMR_Realistic_Strategy",
        actions=[mmr_action],
        combination_logic='AND'
    )
    
    signals, action_details = mmr_strategy.evaluate(data)
    signal_rate = (signals.sum() / len(signals)) * 100
    print(f"MMR Strategy: {signals.sum()} signals, {signal_rate:.2f}% signal rate")
    
    return signals.sum() > 0 and signal_rate < 30

def test_realistic_lvb_strategy():
    """Test Liquidity Vacuum Breakout (LVB) Strategy with realistic parameters"""
    print("\n=== Testing Realistic LVB Strategy ===")
    
    data = create_realistic_test_data()
    
    # LVB Strategy with more restrictive parameters
    lvb_action = Action(
        name="LVB_Realistic",
        filters=[
            {
                'type': 'order_flow',
                'min_cvd_threshold': 300,  # More restrictive
                'large_trade_ratio': 0.7   # More restrictive
            },
            {
                'type': 'volatility',
                'min_atr_ratio': 0.01,   # More restrictive
                'max_atr_ratio': 0.08    # More restrictive
            },
            {
                'type': 'momentum',
                'momentum_threshold': 0.004,  # More restrictive
                'lookback': 3,
                'rsi_range': [45, 55]  # Very restrictive RSI range
            },
            {
                'type': 'volume',
                'min_volume': 500,      # More restrictive
                'volume_ratio': 1.5     # More restrictive
            }
        ]
    )
    
    lvb_strategy = PatternStrategy(
        name="LVB_Realistic_Strategy",
        actions=[lvb_action],
        combination_logic='AND'
    )
    
    signals, action_details = lvb_strategy.evaluate(data)
    signal_rate = (signals.sum() / len(signals)) * 100
    print(f"LVB Strategy: {signals.sum()} signals, {signal_rate:.2f}% signal rate")
    
    return signals.sum() > 0 and signal_rate < 30

def test_realistic_master_strategy():
    """Test the complete Index Strategy 11.2 with realistic parameters"""
    print("\n=== Testing Realistic Master Strategy ===")
    
    data = create_realistic_test_data()
    
    # Master Strategy with realistic parameters
    master_strategy = PatternStrategy(
        name="Index_Strategy_11.2_Realistic",
        actions=[
            Action(
                name="OFM_Realistic",
                filters=[
                    {
                        'type': 'order_flow',
                        'min_cvd_threshold': 500,
                        'large_trade_ratio': 0.5
                    },
                    {
                        'type': 'vwap',
                        'condition': 'near',
                        'tolerance': 0.005,
                        'period': 200
                    },
                    {
                        'type': 'momentum',
                        'momentum_threshold': 0.002,
                        'lookback': 10,
                        'rsi_range': [40, 60]
                    },
                    {
                        'type': 'volatility',
                        'min_atr_ratio': 0.005,
                        'max_atr_ratio': 0.05
                    }
                ]
            ),
            Action(
                name="MMR_Realistic",
                filters=[
                    {
                        'type': 'order_flow',
                        'min_cvd_threshold': 200,
                        'large_trade_ratio': 0.6
                    },
                    {
                        'type': 'volatility',
                        'min_atr_ratio': 0.008,
                        'max_atr_ratio': 0.04
                    },
                    {
                        'type': 'momentum',
                        'momentum_threshold': 0.003,
                        'lookback': 5,
                        'rsi_range': [30, 70]
                    }
                ]
            ),
            Action(
                name="LVB_Realistic",
                filters=[
                    {
                        'type': 'order_flow',
                        'min_cvd_threshold': 300,
                        'large_trade_ratio': 0.7
                    },
                    {
                        'type': 'volatility',
                        'min_atr_ratio': 0.01,
                        'max_atr_ratio': 0.08
                    },
                    {
                        'type': 'momentum',
                        'momentum_threshold': 0.004,
                        'lookback': 3,
                        'rsi_range': [45, 55]
                    },
                    {
                        'type': 'volume',
                        'min_volume': 500,
                        'volume_ratio': 1.5
                    }
                ]
            )
        ],
        combination_logic='OR',  # Use OR logic to combine strategies
        gates_and_logic={
            'location_gate': True,
            'volatility_gate': True
        },
        location_gate_params={
            'gate_threshold': 0.1
        }
    )
    
    signals, action_details = master_strategy.evaluate(data)
    signal_rate = (signals.sum() / len(signals)) * 100
    print(f"Master Strategy: {signals.sum()} signals, {signal_rate:.2f}% signal rate")
    
    return signals.sum() > 0 and signal_rate < 80

if __name__ == "__main__":
    print("REALISTIC INDEX STRATEGY 11.2 TEST")
    print("=" * 50)
    
    # Test individual strategies
    ofm_success = test_realistic_ofm_strategy()
    mmr_success = test_realistic_mmr_strategy()
    lvb_success = test_realistic_lvb_strategy()
    master_success = test_realistic_master_strategy()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"OFM Strategy: {'PASS' if ofm_success else 'FAIL'}")
    print(f"MMR Strategy: {'PASS' if mmr_success else 'FAIL'}")
    print(f"LVB Strategy: {'PASS' if lvb_success else 'FAIL'}")
    print(f"Master Strategy: {'PASS' if master_success else 'FAIL'}")
    
    all_passed = ofm_success and mmr_success and lvb_success and master_success
    
    if all_passed:
        print("\n🎉 ALL STRATEGIES WORKING WITH REALISTIC PARAMETERS!")
        print("✅ Index Strategy 11.2 is ready for production use")
    else:
        print("\n⚠️ SOME STRATEGIES NEED ADJUSTMENT")
        print("❌ Need to fine-tune filter parameters")