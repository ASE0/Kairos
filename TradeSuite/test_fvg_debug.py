#!/usr/bin/env python3
"""
Debug FVG detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Create test data
data = []
base_time = datetime(2024, 3, 7, 16, 15, 0)

# Pre-FVG bars (3 bars)
for i in range(3):
    data.append({
        'datetime': base_time + timedelta(minutes=i),
        'open': 100.0 + i * 0.5,
        'high': 100.5 + i * 0.5,
        'low': 99.5 + i * 0.5,
        'close': 100.2 + i * 0.5,
        'volume': 1000 + i * 100
    })

# FVG creation bar (index 3) - create a gap up
data.append({
    'datetime': base_time + timedelta(minutes=3),
    'open': 102.0,  # Open above previous high
    'high': 102.5,  # High above previous high
    'low': 101.8,   # Low above previous high (gap)
    'close': 102.2,
    'volume': 1500
})

# Post-FVG bars (3 bars)
for i in range(3):
    data.append({
        'datetime': base_time + timedelta(minutes=4+i),
        'open': 101.5 + i * 0.3,
        'high': 102.0 + i * 0.3,
        'low': 101.0 + i * 0.3,
        'close': 101.8 + i * 0.3,
        'volume': 1200 + i * 100
    })

df = pd.DataFrame(data)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print("Test data:")
print(df)

# Test FVG detection
from patterns.enhanced_candlestick_patterns import FVGIndicator

indicator = FVGIndicator(min_gap_size=0.001, max_touches=3)
result = indicator.calculate(df)

print(f"\nFVG detection result:")
print(f"Active FVGs: {len(result['active'])}")
for fvg in result['active']:
    print(f"  FVG at {fvg.timestamp}: {fvg.low} - {fvg.high}, direction: {fvg.direction}")

# Test FVGPattern
from patterns.enhanced_candlestick_patterns import FVGPattern
from core.data_structures import TimeRange

pattern = FVGPattern(timeframes=[TimeRange(1, 'm')], min_gap_size=0.001, max_touches=3)
signals = pattern.detect(df)
zones = pattern.detect_zones(df)

print(f"\nFVGPattern signals:")
print(f"Signals: {signals.tolist()}")
print(f"Zones: {len(zones)}")
for zone in zones:
    print(f"  Zone: {zone}")

# Check what should be detected
print(f"\nExpected FVG at index 3:")
print(f"Bar 2 (index 2): high={df.iloc[2]['high']}, close={df.iloc[2]['close']}")
print(f"Bar 3 (index 3): low={df.iloc[3]['low']}, high={df.iloc[3]['high']}")
print(f"Bar 4 (index 4): low={df.iloc[4]['low']}, high={df.iloc[4]['high']}")

# Check for gaps
gap_up = df.iloc[3]['low'] > df.iloc[2]['high']
gap_down = df.iloc[2]['low'] > df.iloc[3]['high']

print(f"Gap up (bar 2 high < bar 3 low): {gap_up}")
print(f"Gap down (bar 2 low > bar 3 high): {gap_down}")

if gap_up:
    gap_size = (df.iloc[3]['low'] - df.iloc[2]['high']) / df.iloc[2]['high']
    print(f"Gap size: {gap_size:.4f}")
elif gap_down:
    gap_size = (df.iloc[2]['low'] - df.iloc[3]['high']) / df.iloc[3]['high']
    print(f"Gap size: {gap_size:.4f}")
else:
    print("No gap detected") 