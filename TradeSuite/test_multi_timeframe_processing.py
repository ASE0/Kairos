#!/usr/bin/env python3
"""
Test Multi-Timeframe Processing
===============================
Tests the new MultiTimeframeProcessor with the NQ_5s.csv dataset
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.data_processor import MultiTimeframeProcessor
from core.data_structures import TimeRange

def test_multi_timeframe_processing():
    """Test the multi-timeframe processing with NQ_5s.csv"""
    print("=== Multi-Timeframe Processing Test ===")
    
    # Load the dataset
    dataset_path = r"C:\Users\Arnav\Downloads\TradeSuite\NQ_5s.csv"
    print(f"Loading dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Strip whitespace from all column names
        df.columns = [col.strip() for col in df.columns]
        # Create datetime index
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.set_index('datetime', inplace=True)
        # Map columns to standard OHLCV format
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Last': 'close',
            'Volume': 'volume'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"\nProcessed dataset shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Timeframe: {df.index[1] - df.index[0]}")
        
        # Create MultiTimeframeProcessor
        mtf_processor = MultiTimeframeProcessor()
        
        # Define timeframes to test
        timeframes = [
            TimeRange(1, 'm'),   # 1 minute
            TimeRange(5, 'm'),   # 5 minutes
            TimeRange(15, 'm'),  # 15 minutes
            TimeRange(30, 'm'),  # 30 minutes
            TimeRange(1, 'h'),   # 1 hour
        ]
        
        print(f"\nCreating timeframe datasets for: {[f'{tf.value}{tf.unit}' for tf in timeframes]}")
        
        # Create timeframe datasets
        timeframe_datasets = mtf_processor.create_timeframe_datasets(df, timeframes)
        
        print(f"\n=== Results ===")
        print(f"Original data rows: {len(df)}")
        print(f"Original data preserved: {len(mtf_processor.get_original_data())}")
        
        for tf_str, tf_df in timeframe_datasets.items():
            print(f"{tf_str}: {len(tf_df)} rows")
            print(f"  Date range: {tf_df.index.min()} to {tf_df.index.max()}")
            print(f"  Sample data:")
            print(f"    {tf_df.head(3)}")
            print()
        
        # Test automatic timeframe creation
        print("=== Testing Automatic Timeframe Creation ===")
        
        # Simulate strategy requesting additional timeframes
        strategy_timeframes = ['1m', '5m', '10m', '15m', '30m', '1h']
        missing_timeframes = []
        
        for tf_str in strategy_timeframes:
            if tf_str not in timeframe_datasets:
                missing_timeframes.append(tf_str)
        
        if missing_timeframes:
            print(f"Strategy requires additional timeframes: {missing_timeframes}")
            print("Creating missing timeframes...")
            
            # Create missing timeframes
            additional_timeframes = []
            for tf_str in missing_timeframes:
                # Parse timeframe string
                import re
                match = re.match(r'(\d+)([smhd])', tf_str)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    additional_timeframes.append(TimeRange(value, unit))
            
            # Create additional datasets
            additional_datasets = mtf_processor.create_timeframe_datasets(
                mtf_processor.get_original_data(), 
                additional_timeframes
            )
            
            # Merge with existing datasets
            timeframe_datasets.update(additional_datasets)
            
            print("Additional timeframes created:")
            for tf_str, tf_df in additional_datasets.items():
                print(f"  {tf_str}: {len(tf_df)} rows")
        
        print(f"\n=== Final Results ===")
        print(f"Total timeframe datasets: {len(timeframe_datasets)}")
        for tf_str in sorted(timeframe_datasets.keys()):
            tf_df = timeframe_datasets[tf_str]
            print(f"{tf_str}: {len(tf_df)} rows")
        
        # Test signal alignment
        print(f"\n=== Testing Signal Alignment ===")
        
        # Create sample signals for different timeframes
        sample_signals = {}
        for tf_str in ['1m', '5m', '15m']:
            if tf_str in timeframe_datasets:
                tf_df = timeframe_datasets[tf_str]
                # Create a simple signal (e.g., price above SMA)
                if len(tf_df) > 20:
                    sma = tf_df['close'].rolling(20).mean()
                    signal = tf_df['close'] > sma
                    sample_signals[tf_str] = signal
                    print(f"Created signal for {tf_str}: {signal.sum()} signals out of {len(signal)} bars")
        
        # Align signals
        if sample_signals:
            aligned_signals = mtf_processor.align_signals_across_timeframes(sample_signals)
            print(f"Aligned signals shape: {aligned_signals.shape}")
            print(f"Aligned signals columns: {list(aligned_signals.columns)}")
        
        print("\n=== Test Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_timeframe_processing()
    if success:
        print("\n✅ Multi-timeframe processing test PASSED")
    else:
        print("\n❌ Multi-timeframe processing test FAILED")
        sys.exit(1) 