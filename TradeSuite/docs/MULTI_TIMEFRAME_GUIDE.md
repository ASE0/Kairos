# Multi-Timeframe Backtest Engine Guide

## Overview

The TradeSuite now includes a **Multi-Timeframe Backtest Engine** that preserves strategy timeframes while allowing visual timeframe changes for chart display. This addresses the previous limitation where changing the backtest timeframe would resample all strategy components to that timeframe.

## Key Features

### 1. Strategy Timeframe Preservation
- **Strategy components maintain their original timeframes** (e.g., 1-day gate, 1-hour pattern)
- **No more incorrect signal generation** from evaluating patterns on wrong timeframes
- **True multi-timeframe strategy behavior** is preserved

### 2. Visual Timeframe Flexibility
- **Chart display can be changed** to any timeframe without affecting strategy logic
- **Real-time chart updates** when changing display timeframe
- **Strategy evaluation remains unchanged** regardless of chart display

### 3. Intelligent Signal Combination
- **Signals from different timeframes are properly combined**
- **Forward-filling of signals** to execution timeframe
- **Maintains strategy logic** (AND, OR, WEIGHTED combinations)

## How It Works

### Multi-Timeframe Data Preparation

```python
# The engine automatically detects required timeframes from strategy actions
required_timeframes = set()
for action in strategy.actions:
    if action.pattern and hasattr(action.pattern, 'timeframes'):
        for tf in action.pattern.timeframes:
            tf_str = f"{tf.value}{tf.unit}"
            required_timeframes.add(tf_str)

# Data is resampled to all required timeframes
multi_tf_data = {}
for tf in required_timeframes:
    resampled = original_data.resample(tf).agg(ohlc_dict).dropna()
    multi_tf_data[tf] = resampled
```

### Strategy Evaluation Process

1. **Each action is evaluated on its appropriate timeframe**
2. **Signals are resampled to execution timeframe**
3. **Signals are combined according to strategy logic**
4. **Backtest runs on execution timeframe**

### Visual Display

- **Chart shows execution timeframe data by default**
- **User can change display timeframe** without re-running backtest
- **Strategy timeframes are shown in chart title**
- **Performance metrics remain accurate**

## Usage Examples

### Example 1: Multi-Timeframe Strategy

```python
# Strategy with 1-day gate + 1-hour pattern + 15-minute pattern
daily_gate = CustomPattern(
    name="Daily_Gate",
    timeframes=[TimeRange(1, 'd')],
    ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)]
)

hourly_pattern = CustomPattern(
    name="Hourly_Pattern", 
    timeframes=[TimeRange(1, 'h')],
    ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.25, lower_wick_ratio=0.25)]
)

fifteen_min_pattern = CustomPattern(
    name="15min_Pattern",
    timeframes=[TimeRange(15, 'm')],
    ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)]
)

# Create strategy with AND logic (all timeframes must agree)
strategy = PatternStrategy(
    name="Multi_Timeframe_Strategy",
    actions=[daily_action, hourly_action, fifteen_min_action],
    combination_logic='AND',
    min_actions_required=3
)
```

### Example 2: Running Multi-Timeframe Backtest

```python
# Create engine and run backtest
engine = MultiTimeframeBacktestEngine()
results = engine.run_backtest(
    strategy=strategy,
    data=original_data,  # 1-minute data
    initial_capital=100000,
    risk_per_trade=0.02
)

# Results include multi-timeframe information
print(f"Strategy timeframes: {list(engine.multi_tf_data.keys())}")
print(f"Execution timeframe: {len(results['data'])} bars")
```

## Backtest Window Changes

### Timeframe Selection Behavior

**Before (Old Behavior):**
- Changing timeframe resampled ALL data to that timeframe
- Strategy components evaluated on wrong timeframes
- Incorrect signal generation and performance

**After (New Behavior):**
- Changing timeframe only affects chart display
- Strategy timeframes are preserved
- Accurate multi-timeframe evaluation

### UI Updates

1. **Chart Title Shows Timeframe Information:**
   ```
   Strategy Name
   Strategy TFs: 1d, 1h, 15m | Execution: 10080 bars | Display: 30min
   ```

2. **Overview Tab Shows Multi-Timeframe Info:**
   ```
   Backtest Results (Display: 30min)
   
   Strategy: Multi_Timeframe_Strategy
   Strategy Timeframes: 1d, 1h, 15m
   Execution Timeframe: 10080 bars
   
   Initial Capital: $100,000.00
   Final Capital: $102,500.00
   Total Return: 2.50%
   Total Trades: 15
   ```

3. **Log Messages:**
   ```
   ✅ Using Multi-Timeframe Backtest Engine - strategy timeframes will be preserved
   Resampled to 1d: 7 bars
   Resampled to 1h: 168 bars  
   Resampled to 15m: 672 bars
   Chart resampled to 30min for display: 336 bars
   Chart display updated to 30min timeframe (strategy timeframes preserved)
   ```

## Technical Implementation

### MultiTimeframeBacktestEngine Class

```python
@dataclass
class MultiTimeframeBacktestEngine:
    def prepare_multi_timeframe_data(self, original_data, strategy):
        # Detect required timeframes from strategy actions
        # Resample data to all required timeframes
        # Return multi-timeframe data dictionary
    
    def evaluate_strategy_multi_timeframe(self, strategy, multi_tf_data):
        # Evaluate each action on its appropriate timeframe
        # Resample signals to execution timeframe
        # Combine signals according to strategy logic
    
    def run_backtest(self, strategy, data, initial_capital, risk_per_trade):
        # Prepare multi-timeframe data
        # Evaluate strategy on multi-timeframe data
        # Run backtest on execution timeframe
        # Return results with multi-timeframe information
```

### Signal Resampling

```python
def _resample_signals_to_execution(self, signals, execution_index):
    """Resample signals from their timeframe to execution timeframe"""
    resampled = pd.Series(False, index=execution_index)
    
    for i, timestamp in enumerate(execution_index):
        if timestamp in signals.index:
            resampled.iloc[i] = signals[timestamp]
        else:
            # Find the last signal before this timestamp
            prev_signals = signals[signals.index <= timestamp]
            if not prev_signals.empty:
                resampled.iloc[i] = prev_signals.iloc[-1]
    
    return resampled
```

## Benefits

### 1. Accurate Strategy Evaluation
- **Patterns evaluated on correct timeframes**
- **Proper signal generation** without false positives/negatives
- **Realistic backtest results**

### 2. Flexible Visualization
- **View results at any timeframe** without re-running backtest
- **Compare different chart views** quickly
- **Maintain strategy accuracy** while exploring visual options

### 3. Performance
- **Single backtest run** provides all timeframe data
- **Efficient signal combination** without redundant calculations
- **Fast chart updates** when changing display timeframe

## Testing

Run the test script to verify functionality:

```bash
python test_multi_timeframe_backtest.py
```

This will:
1. Create a multi-timeframe strategy
2. Run a backtest with 1-minute data
3. Verify that strategy timeframes are preserved
4. Test visual timeframe changes

## Migration from Old System

### For Existing Strategies
- **No changes required** to strategy definitions
- **Automatic multi-timeframe detection** from existing timeframes
- **Backward compatible** with single-timeframe strategies

### For New Strategies
- **Define timeframes in pattern objects** as before
- **Use combination logic** to specify how signals should be combined
- **Multi-timeframe engine handles the rest**

## Limitations and Considerations

### 1. Data Requirements
- **Original data should be high-frequency** (1-minute or lower)
- **Sufficient data points** for all required timeframes
- **Proper datetime index** required

### 2. Performance
- **Memory usage increases** with multiple timeframes
- **Processing time scales** with number of timeframes
- **Large datasets may require optimization**

### 3. Signal Timing
- **Forward-filling may introduce delays** in signal propagation
- **Consider signal timing** when designing strategies
- **Test with realistic data** to verify timing behavior

## Future Enhancements

### Planned Features
1. **Advanced signal timing** with custom propagation rules
2. **Timeframe-specific filters** and conditions
3. **Dynamic timeframe selection** based on market conditions
4. **Performance optimization** for large multi-timeframe datasets

### Integration Opportunities
1. **Real-time multi-timeframe analysis**
2. **Portfolio-level multi-timeframe strategies**
3. **Advanced risk management** across timeframes
4. **Machine learning** for timeframe selection

## Conclusion

The Multi-Timeframe Backtest Engine provides a robust foundation for accurate strategy evaluation while maintaining the flexibility to visualize results at different timeframes. This addresses a critical limitation in the previous system and enables the development of sophisticated multi-timeframe trading strategies. 