#!/usr/bin/env python3
"""
Debug script for Hammer pattern detection
"""

import pandas as pd
from patterns.candlestick_patterns import HammerPattern
from core.data_structures import TimeRange
from tests.data_factory import create_dataset

# Regenerate the test data
data_path = create_dataset('hammer', {'pattern_type': 'both'})
print(f"Generated data at: {data_path}")

# Load the test data
df = pd.read_csv(data_path)
print("Data shape:", df.shape)
print("First 10 rows:")
print(df.head(10))

# Create pattern and detect
pattern = HammerPattern([TimeRange(1, 'min')], pattern_type='both')
signals = pattern.detect(df)

print("\nSignals:", signals.tolist())
print("True signals at indices:", [i for i, s in enumerate(signals) if s])

# Print all bars for manual inspection
print("\nAll bars:")
print(df) 