import pandas as pd
import matplotlib.pyplot as plt
from patterns.candlestick_patterns import HammerPattern
from core.data_structures import TimeRange
from strategies.strategy_builders import BacktestEngine, Action, PatternStrategy

# Load the provided dataset
csv_path = r'C:/Users/Arnav/Downloads/TradeSuite/NQ_5s.csv'
data = pd.read_csv(csv_path)

# Ensure columns are correct and parse datetime if needed
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

# Create HammerPattern action and strategy
pattern = HammerPattern([TimeRange(5, 'm')])
action = Action(name='hammer_5m', pattern=pattern)
strategy = PatternStrategy(actions=[action])

# Run backtest
engine = BacktestEngine()
results = engine.run_backtest(strategy, data, initial_capital=100000, risk_per_trade=0.02)

# Print results
print(f"Initial Capital: {results['initial_capital']}")
print(f"Final Capital: {results['final_capital']}")
print(f"Total Return: {results['total_return']:.4%}")
print(f"Total Trades: {results['total_trades']}")
print(f"Win Rate: {results['win_rate']:.2%}")

# Plot equity curve
import matplotlib.pyplot as plt
plt.plot(results['equity_curve'])
plt.title('Equity Curve')
plt.xlabel('Bar')
plt.ylabel('Equity')
plt.show()

# Check for mismatch
expected_return = (results['final_capital'] - results['initial_capital']) / results['initial_capital']
if abs(results['total_return'] - expected_return) > 1e-6 or results['total_return'] == 0:
    print(f"[WARNING] Total return mismatch or zero: total_return={results['total_return']}, expected={expected_return}")
else:
    print("[OK] Total return matches equity curve.") 