import pandas as pd
from core.data_structures import TimeRange
from strategies.strategy_builders import PatternStrategy, Action, StrategyFactory, MultiTimeframeBacktestEngine
from patterns.candlestick_patterns import CustomPattern

# 1. Load the synthetic dataset
df = pd.read_csv('datasets/multitf_test.csv')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.set_index('datetime', inplace=True)

# 2. Patch CustomPattern to return True only at expected bars for each pattern
def make_pattern_detector(trigger_times):
    def detect(self, data):
        # Return True only at the specified trigger_times (as string)
        index_str = data.index.strftime('%Y-%m-%d %H:%M:%S')
        return pd.Series(index_str.isin(trigger_times), index=data.index)
    return detect

# Define expected trigger times for each pattern
times_fvg = ['2024-03-07 09:10:00', '2024-03-07 09:15:00']
times_zone = ['2024-03-07 09:05:00', '2024-03-07 09:10:00', '2024-03-07 09:15:00']
times_m = ['2024-03-07 09:15:00']

# Patch detect methods
fvg_pattern = CustomPattern('fvg', [TimeRange(1, 'm')], [])
fvg_pattern.detect = make_pattern_detector(times_fvg).__get__(fvg_pattern, CustomPattern)
zone_pattern = CustomPattern('demand_zone', [TimeRange(5, 'm')], [])
zone_pattern.detect = make_pattern_detector(times_zone).__get__(zone_pattern, CustomPattern)
m_pattern = CustomPattern('m_pattern', [TimeRange(15, 'm')], [])
m_pattern.detect = make_pattern_detector(times_m).__get__(m_pattern, CustomPattern)

# 3. Define actions for each pattern
action_fvg = Action(name='fvg_1m', pattern=fvg_pattern)
action_zone = Action(name='zone_5m', pattern=zone_pattern)
action_m = Action(name='m_15m', pattern=m_pattern)

# 4. Build the strategy (AND logic: all must be true for entry)
strategy = PatternStrategy(
    name='multi_tf_test',
    actions=[action_fvg, action_zone, action_m],
    combination_logic='AND',
    min_actions_required=3
)

# 5. Pre-calculate expected events (for this synthetic dataset):
expected_trades = [
    '2024-03-07 09:10:00',  # All conditions met
    '2024-03-07 09:15:00',  # All conditions met
]
expected_zones = times_zone
expected_patterns = times_m

# 6. Run the backtest in headless mode
engine = MultiTimeframeBacktestEngine()
results = engine.run_backtest(strategy, df)

# Print action signals and combined signals for debug
multi_tf_data = engine.multi_tf_data
_, action_signals, _, _ = engine.evaluate_strategy_multi_timeframe(strategy, multi_tf_data)
print('--- ACTION SIGNALS (resampled to execution) ---')
for k, v in action_signals.items():
    print(f'{k}:')
    print(v[v].index.tolist())
combined_signals = engine._combine_action_signals(action_signals, strategy, df.index)
print('--- COMBINED SIGNALS (True indices) ---')
print(combined_signals[combined_signals].index.tolist())

# 7. Extract actual events
actual_trades = [str(trade['entry_time'])[:19] for trade in results['trades']]
actual_zones = [str(zone['timestamp'])[:19] for zone in results['zones'] if zone.get('zone_type') == 'demand_zone']
actual_patterns = [str(zone['timestamp'])[:19] for zone in results.get('patterns', []) if zone.get('zone_type') == 'm_pattern']

# Print raw merged zones and patterns for debug
print('--- RAW ZONES ---')
print(results['zones'])
print('--- RAW PATTERNS ---')
print(results.get('patterns', []))

# 8. Print logs and compare
print('--- EXPECTED TRADES ---')
print(expected_trades)
print('--- ACTUAL TRADES ---')
print(actual_trades)
print('--- EXPECTED ZONES ---')
print(expected_zones)
print('--- ACTUAL ZONES ---')
print(actual_zones)
print('--- EXPECTED PATTERNS ---')
print(expected_patterns)
print('--- ACTUAL PATTERNS ---')
print(actual_patterns)

# 9. Conclusion
def check_match(expected, actual):
    return set(expected) == set(actual)

trades_ok = check_match(expected_trades, actual_trades)
zones_ok = check_match(expected_zones, actual_zones)
patterns_ok = check_match(expected_patterns, actual_patterns)

print('\n--- SUMMARY ---')
print(f"Trades match: {trades_ok}")
print(f"Zones match: {zones_ok}")
print(f"Patterns match: {patterns_ok}")

if trades_ok and zones_ok and patterns_ok:
    print("\n[PASS] Multi-timeframe logic is fully functional in the current environment.")
else:
    print("\n[FAIL] Multi-timeframe logic is NOT fully functional. See mismatches above.") 