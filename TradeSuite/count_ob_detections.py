import pandas as pd
import json

df = pd.read_csv('workspaces/datasets/NQ_5s_1m.csv')
with open('output.json') as f:
    out = json.load(f)
zones = out['zones']
date_idx = pd.to_datetime(df['datetime'])
count = 0
for z in zones:
    idx = z.get('bar_index')
    if idx is not None and 0 <= idx < len(date_idx):
        d = date_idx.iloc[idx]
        if str(d.date()) == '2024-03-07':
            count += 1
print('OB detections on 2024-03-07:', count) 