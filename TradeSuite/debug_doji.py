import pandas as pd

# Load the data
df = pd.read_csv('tests/test_doji_standard_data.csv')
bar = df.iloc[5]

print('Bar at index 5:')
print(f'Open: {bar["open"]}, Close: {bar["close"]}, High: {bar["high"]}, Low: {bar["low"]}')

# Calculate metrics
body = abs(bar["close"] - bar["open"])
total_range = bar["high"] - bar["low"]
body_ratio = body / total_range
upper_wick = bar["high"] - max(bar["open"], bar["close"])
lower_wick = min(bar["open"], bar["close"]) - bar["low"]
wick_diff = abs(upper_wick - lower_wick) / total_range

print(f'Body: {body}, Total range: {total_range}, Body ratio: {body_ratio:.4f}')
print(f'Upper wick: {upper_wick}, Lower wick: {lower_wick}')
print(f'Wick difference ratio: {wick_diff:.4f}')
print(f'Max body ratio allowed: 0.1, Wick symmetry tolerance: 0.3')

# Check criteria
print(f'\nCriteria check:')
print(f'Body ratio <= 0.1: {body_ratio <= 0.1} ({body_ratio:.4f} <= 0.1)')
print(f'Wick symmetry <= 0.3: {wick_diff <= 0.3} ({wick_diff:.4f} <= 0.3)')
print(f'Both criteria met: {body_ratio <= 0.1 and wick_diff <= 0.3}') 