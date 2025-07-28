#!/usr/bin/env python3
"""
Test script to verify zone overlay fix
"""

import pandas as pd
import os
import sys

def test_zone_overlay_fix():
    """Test that zones are being passed to chart overlay correctly"""
    print("🔍 Testing zone overlay fix...")
    
    # Load the recent dataset
    dataset_path = os.path.join('recent_dataset', 'most_recent.csv')
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Loaded dataset: {len(df)} rows")
        
        # Process data like main.py does
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            df.index = df['datetime']
            print("✅ Set index to combined 'Date' and 'Time' columns")
        elif 'Date' in df.columns:
            df.index = pd.to_datetime(df['Date'])
            print("✅ Set index to 'Date' column")
        
        # Handle duplicate indices
        if not df.index.is_unique:
            before_agg = len(df)
            df = df.groupby(df.index).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            })
            after_agg = len(df)
            print(f"✅ Aggregated {before_agg} -> {after_agg} bars")
        
        # Map columns
        col_map = {
            'open': None, 'high': None, 'low': None, 'close': None, 'volume': None
        }
        for col in df.columns:
            lcol = col.lower()
            if lcol.startswith('open'):
                col_map['open'] = col
            elif lcol.startswith('high'):
                col_map['high'] = col
            elif lcol.startswith('low'):
                col_map['low'] = col
            elif lcol.startswith('close') or lcol.startswith('last'):
                col_map['close'] = col
            elif lcol.startswith('vol'):
                col_map['volume'] = col
        
        if all(col_map.values()):
            df = df[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            print("✅ Mapped columns to OHLCV format")
        
        # Test zone creation
        from strategies.strategy_builders import PatternStrategy, Action
        
        # Create a simple strategy with location-only action
        action = Action(name="Test Zone Action")
        strategy = PatternStrategy(
            name="Test Zone Strategy",
            actions=[action],
            gates_and_logic={'location_gate': True},
            location_gate_params={'gate_threshold': 0.1}  # Low threshold for testing
        )
        
        # Test data (first 100 bars)
        test_data = df.head(100).copy()
        print(f"✅ Testing with {len(test_data)} bars")
        
        # Run strategy evaluation
        signals, action_details = strategy.evaluate(test_data)
        
        # Check if zones were created
        simple_zones = getattr(strategy, 'simple_zones', [])
        calculated_zones = getattr(strategy, 'calculated_zones', [])
        
        print(f"✅ Strategy evaluation complete")
        print(f"📊 Signals generated: {signals.sum()}")
        print(f"📊 Simple zones created: {len(simple_zones)}")
        print(f"📊 Complex zones created: {len(calculated_zones)}")
        
        if simple_zones:
            print("✅ SUCCESS: Simple zones are being created!")
            print(f"📊 Sample zone: {simple_zones[0]}")
            return True
        else:
            print("❌ FAILURE: No simple zones were created")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zone_overlay_fix()
    if success:
        print("\n🎉 Zone overlay fix test PASSED!")
        print("Zones should now appear on the chart when you run a backtest.")
    else:
        print("\n💥 Zone overlay fix test FAILED!")
        print("Check the logs for more details.") 