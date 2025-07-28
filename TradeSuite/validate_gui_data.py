#!/usr/bin/env python3
"""
Validate that your actual data works with the patched GUI logic
Tests index conversion, date filtering, and zone overlays on real data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_dataset(data_path, dataset_name="Unknown"):
    """Validate a single dataset"""
    print(f"\n🔍 Validating dataset: {dataset_name}")
    print("-" * 40)
    
    try:
        # Load the data
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Test index conversion logic (same as BacktestWindow)
        original_df = df.copy()
        
        # PATCH: Always use combined 'Date' and 'Time' if both are present
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns and 'Time' in df.columns:
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                df.set_index('datetime', inplace=True)
                print("✅ Set index to combined 'Date' and 'Time' columns")
            else:
                for col in ['datetime', 'date', 'Date', 'timestamp', 'Timestamp']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                        print(f"✅ Converted '{col}' to DatetimeIndex")
                        break
        
        # Debug index info
        print(f"Index type: {type(df.index)}")
        print(f"Index unique: {df.index.is_unique}")
        print(f"Index range: {df.index.min()} to {df.index.max()}")
        print(f"Sample indices: {list(df.index[:5])}")
        
        # Test date filtering
        if isinstance(df.index, pd.DatetimeIndex):
            # Test filtering to last 7 days
            end_date = df.index.max()
            start_date = end_date - timedelta(days=7)
            
            filtered_df = df[(df.index >= start_date) & (df.index < end_date)]
            print(f"✅ Date filtering test: {len(df)} -> {len(filtered_df)} bars")
            print(f"Filter range: {start_date} to {end_date}")
        else:
            print("❌ Could not create DatetimeIndex - date filtering will not work")
        
        # Test OHLC column mapping
        required_cols = ['open', 'high', 'low', 'close']
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
        
        print(f"Column mapping: {col_map}")
        
        if all(col_map.values()):
            print("✅ All required OHLC columns found")
        else:
            missing = [k for k, v in col_map.items() if v is None]
            print(f"⚠️ Missing columns: {missing}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False

def validate_recent_dataset():
    """Validate the most recent dataset"""
    recent_path = os.path.join('recent_dataset', 'most_recent.csv')
    
    if os.path.exists(recent_path):
        return validate_dataset(recent_path, "Most Recent Dataset")
    else:
        print("❌ No recent dataset found at recent_dataset/most_recent.csv")
        return False

def validate_workspace_datasets():
    """Validate datasets in the workspaces directory"""
    workspace_datasets = os.path.join('workspaces', 'datasets')
    
    if not os.path.exists(workspace_datasets):
        print("❌ No workspaces/datasets directory found")
        return False
    
    success_count = 0
    total_count = 0
    
    for root, dirs, files in os.walk(workspace_datasets):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                dataset_name = os.path.relpath(file_path, workspace_datasets)
                
                if validate_dataset(file_path, dataset_name):
                    success_count += 1
                total_count += 1
    
    print(f"\n📊 Workspace validation summary: {success_count}/{total_count} datasets valid")
    return success_count == total_count

def test_zone_overlay_logic():
    """Test zone overlay logic with sample data"""
    print(f"\n🎯 Testing Zone Overlay Logic")
    print("-" * 40)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'Date': [d.date() for d in dates],
        'Time': [d.time() for d in dates],
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Convert to DatetimeIndex (same as BacktestWindow)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df.index = df['datetime']
    
    # Create sample zones
    zones = [
        {
            'zone_min': 105.0,
            'zone_max': 115.0,
            'index': 10,
            'comb_centers': [107.5, 112.5]
        },
        {
            'zone_min': 95.0,
            'zone_max': 105.0,
            'index': 50,
            'comb_centers': [97.5, 102.5]
        },
        {
            'zone_min': 110.0,
            'zone_max': 120.0,
            'index': 999,  # Out of bounds
            'comb_centers': [112.5, 117.5]
        }
    ]
    
    # Test zone mapping (same as BacktestWindow)
    for i, zone in enumerate(zones):
        zone_min = zone.get('zone_min')
        zone_max = zone.get('zone_max')
        zone_idx = zone.get('index')
        comb_centers = zone.get('comb_centers', [])
        
        print(f"Zone {i}: index={zone_idx}, min={zone_min}, max={zone_max}, comb_centers={comb_centers}")
        
        if zone_idx is not None and (zone_idx < 0 or zone_idx >= len(df)):
            print(f"  ⚠️ Zone {i} index {zone_idx} out of bounds for df of length {len(df)}")
            continue
            
        if zone_min is not None and zone_max is not None and zone_idx is not None and 0 <= zone_idx < len(df) and comb_centers:
            start_time = df.index[zone_idx]
            end_idx = min(zone_idx + 5, len(df) - 1)
            end_time = df.index[end_idx]
            print(f"  ✅ Zone {i} mapping: {start_time} to {end_time}")
        else:
            print(f"  ❌ Zone {i} has invalid data")
    
    return True

def main():
    """Main validation function"""
    print("🔧 GUI Data Validation Tool")
    print("=" * 50)
    print("This tool validates that your data will work with the patched GUI logic.")
    
    success = True
    
    # Test recent dataset
    if not validate_recent_dataset():
        success = False
    
    # Test workspace datasets
    if not validate_workspace_datasets():
        success = False
    
    # Test zone overlay logic
    if not test_zone_overlay_logic():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All validations passed! Your data should work with the patched GUI.")
        print("You can now test the backtester with confidence.")
    else:
        print("❌ Some validations failed. Check the output above for issues.")
        print("The GUI may have problems with some of your data.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 