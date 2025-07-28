#!/usr/bin/env python3
"""
Check FVG test data
"""

from tests.data_factory import create_dataset
import pandas as pd

# Create FVG test data
data_path = create_dataset('fvg')
df = pd.read_csv(data_path)

print('FVG Test Data:')
print(df.head(10))

print('\nChecking bars 1, 2, 3:')
print(f'Bar 1 (H): {df.iloc[1]["high"]:.2f}')
print(f'Bar 3 (L): {df.iloc[3]["low"]:.2f}')
print(f'Gap condition: {df.iloc[1]["high"]:.2f} < {df.iloc[3]["low"]:.2f} = {df.iloc[1]["high"] < df.iloc[3]["low"]}')

print('\nChecking bars 0, 1, 2 (what FVG detector sees at i=2):')
print(f'Bar 0 (H): {df.iloc[0]["high"]:.2f}')
print(f'Bar 2 (L): {df.iloc[2]["low"]:.2f}')
print(f'Gap condition: {df.iloc[0]["high"]:.2f} < {df.iloc[2]["low"]:.2f} = {df.iloc[0]["high"] < df.iloc[2]["low"]}')

print('\nChecking bars 5, 6, 7 (what FVG detector sees at i=7):')
print(f'Bar 5 (H): {df.iloc[5]["high"]:.2f}')
print(f'Bar 7 (L): {df.iloc[7]["low"]:.2f}')
print(f'Gap condition: {df.iloc[5]["high"]:.2f} < {df.iloc[7]["low"]:.2f} = {df.iloc[5]["high"] < df.iloc[7]["low"]}') 