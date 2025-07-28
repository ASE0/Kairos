#!/usr/bin/env python3
"""
Check Order Block test data
"""

from tests.data_factory import create_dataset
import pandas as pd

# Create order block test data
data_path = create_dataset('order_block_zone')
df = pd.read_csv(data_path)

print('Order Block Test Data:')
print(df)

print('\nChecking Order Block pattern:')
print(f'Bar 8 (OB bar): open={df.iloc[8]["open"]:.2f}, high={df.iloc[8]["high"]:.2f}, low={df.iloc[8]["low"]:.2f}, close={df.iloc[8]["close"]:.2f}')
print(f'Bar 9 (impulse move): open={df.iloc[9]["open"]:.2f}, high={df.iloc[9]["high"]:.2f}, low={df.iloc[9]["low"]:.2f}, close={df.iloc[9]["close"]:.2f}')

# Calculate impulse move
impulse_move = (df.iloc[8]["close"] - df.iloc[9]["close"]) / df.iloc[8]["close"]
print(f'Impulse move: {impulse_move:.3f}')

# Check if bar 8 is bullish (close > open)
is_bullish = df.iloc[8]["close"] > df.iloc[8]["open"]
print(f'Bar 8 is bullish: {is_bullish}')

# Check if this should trigger order block detection
impulse_threshold = 0.02
should_trigger = impulse_move > impulse_threshold and is_bullish
print(f'Should trigger OB detection: {should_trigger} (threshold: {impulse_threshold})') 