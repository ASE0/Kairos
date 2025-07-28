"""
Test DataStripper Fix
=====================
Quick test to verify that the DataStripper can load NQ_5s.csv without errors.
"""

import sys
import os
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.data_processor import DataStripper

def test_data_stripper_load():
    """Test that DataStripper can load NQ_5s.csv"""
    print("Testing DataStripper.load_data() with NQ_5s.csv")
    
    dataset_path = r"C:\Users\Arnav\Downloads\TradeSuite\NQ_5s.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    try:
        # Create DataStripper instance
        stripper = DataStripper()
        
        # Load data (this should work now with the fixed method signature)
        data = stripper.load_data(dataset_path)
        
        print(f"✅ Successfully loaded {len(data)} rows")
        print(f"Columns: {list(data.columns)}")
        print(f"Index type: {type(data.index)}")
        
        # Check if required columns are present
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(data.columns)}")
            return False
        else:
            print("✅ All required columns present")
            return True
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_stripper_load()
    if success:
        print("\n🎉 DataStripper fix successful!")
    else:
        print("\n❌ DataStripper fix failed!") 