#!/usr/bin/env python3
"""
Test script to verify equity curves and zones work correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action, MultiTimeframeBacktestEngine
from core.data_structures import TimeRange

def test_equity_and_zones():
    """Test that equity curves and zones work correctly"""
    print("🧪 Testing Equity Curves and Zones")
    print("=" * 50)
    
    # Load the NQ_5s dataset
    csv_path = r"C:\Users\Arnav\Downloads\TradeSuite\NQ_5s.csv"
    print(f"📁 Loading dataset: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Fix the index to use real dates
    if 'Date' in df.columns:
        df.index = pd.to_datetime(df['Date'])
        print(f"   Set index to 'Date' column")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
    else:
        print("❌ No 'Date' column found")
        return False
    
    # Map columns to standard OHLCV names
    col_map = {
        'open': None, 'high': None, 'low': None, 'close': None, 'volume': None
    }
    
    # Map the actual column names from NQ_5s.csv
    for col in df.columns:
        col_clean = col.strip()  # Remove leading/trailing spaces
        if col_clean == 'Open':
            col_map['open'] = col
        elif col_clean == 'High':
            col_map['high'] = col
        elif col_clean == 'Low':
            col_map['low'] = col
        elif col_clean == 'Last':
            col_map['close'] = col
        elif col_clean == 'Volume':
            col_map['volume'] = col
    
    print(f"   Column mapping: {col_map}")
    
    # Only keep and rename if all required columns are present
    if all(col_map.values()):
        df = df[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        print(f"   Mapped columns successfully")
    else:
        print(f"❌ Missing required columns. Found: {col_map}")
        return False
    
    # Print data characteristics for diagnostics
    print(f"\n📊 Data Characteristics:")
    print(f"   High range: {df['high'].min():.4f} to {df['high'].max():.4f}")
    print(f"   Low range: {df['low'].min():.4f} to {df['low'].max():.4f}")
    print(f"   First 20 highs: {list(df['high'].head(20))}")
    print(f"   First 20 lows: {list(df['low'].head(20))}")
    print(f"   High std dev: {df['high'].std():.6f}")
    print(f"   Low std dev: {df['low'].std():.6f}")
    
    # Use a subset for testing (rows 100 to 2000)
    test_data = df.iloc[100:2000].copy()
    print(f"   Using subset: {len(test_data)} bars for testing (indices 100 to 2000)")
    print(f"   First 10 indices: {list(test_data.index[:10])}")
    print(f"   Unique indices: {test_data.index.nunique()} / {len(test_data)}")
    if test_data.index.nunique() < len(test_data):
        print("⚠️  Index is not unique. Creating synthetic DatetimeIndex for test subset.")
        test_data.index = pd.date_range('2024-03-07', periods=len(test_data), freq='T')
        print(f"   New index: {list(test_data.index[:10])}")
    
    # Debug: Print data characteristics
    print(f"\n📊 Data Characteristics:")
    print(f"   High range: {test_data['high'].min():.4f} to {test_data['high'].max():.4f}")
    print(f"   Low range: {test_data['low'].min():.4f} to {test_data['low'].max():.4f}")
    print(f"   First 20 highs: {list(test_data['high'].head(20))}")
    print(f"   First 20 lows: {list(test_data['low'].head(20))}")
    print(f"   High std dev: {test_data['high'].std():.6f}")
    print(f"   Low std dev: {test_data['low'].std():.6f}")
    
    # Create a simple definitive zone strategy
    print("\n🔧 Creating definitive zone strategy")
    strategy = PatternStrategy(
        name="Test Definitive Zone",
        actions=[Action(name="Definitive Zone")],
        gates_and_logic={'location_gate': True},
        location_gate_params={'gate_threshold': 0.0, 'sr_threshold': 0.001, 'sr_window': 5}
    )
    
    # Run backtest
    print("\n🚀 Running backtest")
    engine = MultiTimeframeBacktestEngine()
    results = engine.run_backtest(strategy, test_data, initial_capital=100000)
    
    # Check results
    print(f"\n📈 Backtest Results:")
    print(f"   Total trades: {results.get('total_trades', 0)}")
    print(f"   Total return: {results.get('total_return', 0):.2%}")
    print(f"   Final capital: ${results.get('final_capital', 0):,.2f}")
    
    # Check equity curve
    equity_curve = results.get('equity_curve', [])
    print(f"   Equity curve length: {len(equity_curve)}")
    if equity_curve:
        print(f"   Equity curve range: ${min(equity_curve):,.2f} to ${max(equity_curve):,.2f}")
        print(f"   ✅ Equity curve generated successfully")
    else:
        print(f"   ❌ No equity curve generated")
        return False
    
    # Check zones
    zones = results.get('zones', [])
    print(f"   Zones found: {len(zones)}")
    if zones:
        for i, zone in enumerate(zones[:3]):  # Show first 3 zones
            idx = zone.get('index')
            zone_min = zone.get('zone_min')
            zone_max = zone.get('zone_max')
            comb_centers = zone.get('comb_centers', [])
            print(f"     Zone {i}: index={idx}, range=[{zone_min:.2f}, {zone_max:.2f}], combs={len(comb_centers)}")
        print(f"   ✅ Zones generated successfully")
    else:
        print(f"   ❌ No zones generated")
        return False
    
    # Check data
    data = results.get('data')
    print(f"   Result data length: {len(data) if data is not None else 0}")
    if data is not None:
        print(f"   ✅ Data passed correctly")
    else:
        print(f"   ❌ No data in results")
        return False
    
    print(f"\n✅ All tests passed! Equity curves and zones are working correctly.")
    return True

if __name__ == "__main__":
    success = test_equity_and_zones()
    sys.exit(0 if success else 1) 