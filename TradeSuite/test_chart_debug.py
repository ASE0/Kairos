#!/usr/bin/env python3
"""
Test script to debug chart update process
"""

import pandas as pd
import os
from datetime import datetime, timedelta

def test_chart_data_flow():
    """Test the data flow from backtest to chart"""
    print("🔍 Testing chart data flow...")
    
    # Load the recent dataset
    dataset_path = os.path.join('recent_dataset', 'most_recent.csv')
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Loaded dataset: {len(df)} rows, columns: {list(df.columns)}")
        
        # Process data like main.py does
        if 'Date' in df.columns:
            df.index = pd.to_datetime(df['Date'])
            print("✅ Set index to 'Date' column")
        
        # Map columns to standard OHLCV names
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
            print("✅ Mapped to standard OHLC columns")
        
        # Simulate date filtering (like backtest window)
        current_date = datetime.now()
        start_date = current_date - timedelta(days=180)  # 6 months ago
        end_date = current_date
        
        print(f"📅 Filtering date range: {start_date} to {end_date}")
        print(f"📅 Dataset range: {df.index.min()} to {df.index.max()}")
        
        # Check if ranges overlap
        if start_date > df.index.max() or end_date < df.index.min():
            print("❌ Selected date range is outside dataset range!")
            start_date = df.index.min()
            end_date = df.index.max()
            print(f"🔄 Using full dataset range instead: {start_date} to {end_date}")
        
        # Filter data
        before_filter = len(df)
        filtered_df = df[(df.index >= start_date) & (df.index < end_date)]
        after_filter = len(filtered_df)
        
        print(f"📊 Filtering results: {before_filter} -> {after_filter} bars")
        
        # Simulate what the backtest engine would return
        results = {
            'data': filtered_df,
            'total_trades': 0,
            'total_return': 0,
            'equity_curve': [],
            'trades': [],
            'zones': []
        }
        
        print(f"📋 Results keys: {list(results.keys())}")
        print(f"📊 Results['data'] type: {type(results.get('data'))}")
        print(f"📊 Results['data'] shape: {results.get('data').shape if results.get('data') is not None else 'None'}")
        
        # Simulate chart update logic
        print("\n🔍 Simulating chart update logic...")
        
        if not results or results.get('data') is None:
            print("❌ No results or data is None")
            return
        
        display_data = results.get('data')
        print(f"📊 Display data: {len(display_data) if display_data is not None else 'None'} bars")
        
        if display_data is None or not isinstance(display_data, pd.DataFrame):
            print(f"❌ Display data is invalid: type={type(display_data)}")
            return
        
        # Prepare data for plotting
        df_chart = display_data.copy()
        print(f"📊 Chart df shape: {df_chart.shape}, columns: {list(df_chart.columns)}")
        
        if not isinstance(df_chart.index, pd.DatetimeIndex):
            print("❌ Chart df index is not DatetimeIndex")
            return
        
        df_chart = df_chart[~df_chart.index.duplicated(keep='first')]
        df_chart = df_chart.sort_index()
        print(f"📊 After deduplication: {len(df_chart)} bars")
        
        # Check OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df_chart.columns for col in required_cols):
            print(f"❌ Missing OHLC columns. Available: {list(df_chart.columns)}")
            return
        
        df_chart = df_chart[required_cols]
        df_chart.columns = [col.capitalize() for col in required_cols]
        
        print(f"📊 Final chart df length: {len(df_chart)}")
        
        if len(df_chart) < 10:
            print("❌ Not enough bars for charting (< 10)")
            print(f"   This is why you see 'not enough candlesticks'")
            print(f"   Bars: {len(df_chart)}")
            print(f"   Index min: {df_chart.index.min()}")
            print(f"   Index max: {df_chart.index.max()}")
            print(f"   First 10: {list(df_chart.index[:10])}")
        else:
            print("✅ Enough bars for charting")
            print(f"   Should be able to plot {len(df_chart)} candlesticks")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chart_data_flow() 