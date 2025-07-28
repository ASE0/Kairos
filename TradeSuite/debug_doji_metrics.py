import pandas as pd
from patterns.enhanced_candlestick_patterns import get_candle_metrics, PredefinedPatterns, CustomParametricPattern

df = pd.read_csv('test_strong_body_data.csv')
metrics = get_candle_metrics(df)

print('Metrics at index 5:')
print(f'Body ratio: {metrics["body_ratio"].iloc[5]:.4f}')
print(f'Direction: {metrics["direction"].iloc[5]}')

params = PredefinedPatterns.long_body()
print(f'\nStrong body parameters:')
print(f'Min body ratio: {params.min_body_ratio}')
print(f'Allowed directions: {[d.value for d in params.allowed_directions]}')

pattern = CustomParametricPattern('StrongBody', params)
signals = pattern.detect(df)
print(f'\nSignals: {signals.tolist()[:10]}')
print(f'True signals at indices: {[i for i, s in enumerate(signals) if s]}') 