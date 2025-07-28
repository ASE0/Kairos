from tests.run_headless import create_single_pattern_strategy, run_headless_test
import tests.data_factory as df

for pattern, params in [
    ('double_wick', {'min_wick_ratio': 0.3, 'max_body_ratio': 0.4}),
    ('ii_bars', {'min_bars': 2}),
]:
    print(f'=== {pattern} ===')
    p = df.create_dataset(pattern, params)
    s = create_single_pattern_strategy(pattern, params)
    idx, res = run_headless_test(s, p)
    print('Detection index:', idx)
    print('Results:', res)
    print() 