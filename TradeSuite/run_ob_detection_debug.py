import pandas as pd
from core.order_block_gate import detect_order_blocks, Bar

# Load the dataset
csv_path = 'workspaces/datasets/NQ_5s_1m.csv'
df = pd.read_csv(csv_path)

# If there is a datetime column, parse it; else, combine Date and Time if present
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
elif 'Date' in df.columns and 'Time' in df.columns:
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
else:
    raise ValueError('No datetime or Date+Time columns found!')

# Filter for 2024-03-07
mask = (df['datetime'] >= '2024-03-07') & (df['datetime'] < '2024-03-08')
df_day = df[mask].reset_index(drop=True)
print(f"Loaded {len(df_day)} bars for 2024-03-07")

bars = [
    Bar(
        dt=str(row['datetime']),
        open=float(row['open']),
        high=float(row['high']),
        low=float(row['low']),
        close=float(row['close']),
        volume=float(row['volume'])
    )
    for _, row in df_day.iterrows()
]

print("\n[Default OB parameters, epsilon_pts=0.1]")
default_zones = detect_order_blocks(bars, epsilon_pts=0.1)
print(f"Detected {len(default_zones)} OB zones (default params)")
for i, z in enumerate(default_zones):
    print(f"Zone {i+1}: idx={z['bar_index']}, dir={z['zone_direction']}, min={z['zone_min']}, max={z['zone_max']}, impulse={z['impulse']:.3f}, ts={z['timestamp']}")

print("\n[Permissive OB parameters]")
permissive_zones = detect_order_blocks(
    bars,
    buffer_pts=1.0,
    max_block_lookback=20,
    gamma=0.9,
    tau=100,
    min_impulse=0.01
)
print(f"Detected {len(permissive_zones)} OB zones (permissive params)")
for i, z in enumerate(permissive_zones):
    print(f"Zone {i+1}: idx={z['bar_index']}, dir={z['zone_direction']}, min={z['zone_min']}, max={z['zone_max']}, impulse={z['impulse']:.3f}, ts={z['timestamp']}") 