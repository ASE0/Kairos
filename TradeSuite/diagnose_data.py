import pandas as pd
import os

# Path to the most recent dataset (adjust if needed)
RECENT_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'recent_dataset', 'most_recent.csv')

print(f"Loading dataset: {RECENT_DATASET_PATH}")
if not os.path.exists(RECENT_DATASET_PATH):
    print("ERROR: Dataset file does not exist.")
    exit(1)

data = pd.read_csv(RECENT_DATASET_PATH)
print("\n--- RAW DATA ---")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(data.head())

# Robust DatetimeIndex handling
if 'datetime' in data.columns:
    data.index = pd.to_datetime(data['datetime'])
    print("Set index to pd.to_datetime('datetime') column.")
elif 'Date' in data.columns and 'Time' in data.columns:
    data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    data.index = data['datetime']
    print("Set index to combined 'Date' and 'Time' columns.")
elif 'Date' in data.columns:
    data.index = pd.to_datetime(data['Date'])
    print("Set index to pd.to_datetime('Date') column (WARNING: day-level only).")
else:
    data.index = pd.date_range(start='2000-01-01', periods=len(data), freq='T')
    print("No datetime column found, using synthetic date range index.")

print("\n--- INDEX INFO ---")
print(f"Index type: {type(data.index)}")
print(f"Num rows: {len(data)}")
print(f"Num unique index: {data.index.nunique()}")
print(f"Index min: {data.index.min()}")
print(f"Index max: {data.index.max()}")

# Try resampling to 15min
if isinstance(data.index, pd.DatetimeIndex):
    ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    # Map your columns to expected names if needed
    col_map = {}
    for col in ohlc_dict:
        for c in data.columns:
            if c.lower().startswith(col):
                col_map[col] = c
                break
    if len(col_map) < len(ohlc_dict):
        print(f"ERROR: Missing columns for resampling: {[col for col in ohlc_dict if col not in col_map]}")
    else:
        resampled = data[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})
        resampled = resampled.resample('15min').agg(ohlc_dict).dropna()
        print("\n--- RESAMPLED DATA (15min) ---")
        print(f"Shape: {resampled.shape}")
        print(resampled.head(10))
        if len(resampled) < 2:
            print("WARNING: Resampled data has less than 2 bars. Your dataset may be too short for 15min intervals.")
else:
    print("ERROR: Index is not a DatetimeIndex, cannot resample.") 