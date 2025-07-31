# Heatmap Feature Documentation

## Overview

The heatmap feature in the Results Viewer provides a visual representation of when each individual sub-strategy (building block) of a full strategy triggers entry and exit signals. This allows you to see how each sub-strategy behaves in relation to one another and identify patterns in signal timing.

## How It Works

### 1. Building Block Extraction

The heatmap automatically extracts building blocks from your strategy results by analyzing:

- **action_details**: The main source of signal data, containing pandas Series representations of when each sub-strategy fired
- **strategy_params**: Filters, location strategies, and enabled components
- **gates_enabled**: Strategy gates that were active during the backtest
- **component_summary**: Additional strategy components and patterns

### 2. Signal Processing

For each building block, the system:

1. **Parses pandas Series strings**: Converts the string representation back into time series data
2. **Aligns time indices**: Ensures all signals use the same time base
3. **Resamples data**: Aggregates signals into time bins (1min, 5min, 15min, etc.)
4. **Creates signal matrix**: Builds a 2D matrix where:
   - Rows = Building blocks (sub-strategies)
   - Columns = Time periods
   - Values = Signal intensity (0 = no signal, 1+ = signal strength)

### 3. Visualization

The heatmap displays:

- **Y-axis**: Individual building blocks (fvg, vwap, order_block, etc.)
- **X-axis**: Time periods (HH:MM format)
- **Color intensity**: 
  - Darker colors = More sub-strategies firing simultaneously
  - Lighter colors = Fewer sub-strategies firing
  - White = No signals

## Usage

### Accessing the Heatmap

1. Open the Results Viewer
2. Load a strategy result
3. Navigate to the "Heatmap" tab

### Controls

#### Time Binning
- **1 minute**: Shows signals at 1-minute intervals
- **5 minutes**: Aggregates signals into 5-minute bins
- **15 minutes**: Aggregates signals into 15-minute bins
- **1 hour**: Aggregates signals into 1-hour bins
- **4 hours**: Aggregates signals into 4-hour bins
- **1 day**: Aggregates signals into daily bins

#### Color Schemes
- **Viridis**: Default blue-green-yellow scheme
- **Plasma**: Purple-orange-yellow scheme
- **Inferno**: Black-red-yellow scheme
- **Magma**: Black-purple-white scheme
- **RdBu**: Red-white-blue diverging scheme
- **Spectral**: Rainbow color scheme

#### Legend
- **Show Legend**: Toggle color intensity information
- **Hide Legend**: Show only basic heatmap

## Interpreting the Heatmap

### Signal Patterns

1. **Individual Signals**: Each row shows when a specific building block fired
2. **Simultaneous Signals**: Darker areas indicate multiple sub-strategies firing together
3. **Signal Density**: Higher values = more intense signal activity

### Common Patterns

- **Clustered Signals**: Multiple building blocks firing together may indicate strong market conditions
- **Sequential Signals**: One building block firing after another may show signal confirmation
- **Sparse Signals**: Light areas indicate periods of low signal activity

### Building Block Types

#### Core Strategies
- **fvg**: Fair Value Gap detection signals
- **vwap**: Volume Weighted Average Price signals
- **order_block**: Order Block detection signals

#### Filters
- **momentum**: Momentum filter signals
- **volume**: Volume filter signals
- **volatility**: Volatility filter signals

#### Gates
- **location_gate**: Location-based entry gates
- **volatility_gate**: Volatility-based gates
- **regime_gate**: Market regime gates

#### Events
- **Trades**: Trade entry and exit points
- **Zones**: Zone activation events

## Technical Details

### Data Format

The heatmap expects `action_details` in this format:
```json
{
  "fvg": "datetime\n2024-06-02 18:00:00     True\n2024-06-02 18:05:00    False\n...",
  "vwap": "datetime\n2024-06-02 18:00:00    False\n2024-06-02 18:05:00     True\n...",
  "order_block": "datetime\n2024-06-02 18:00:00     True\n2024-06-02 18:05:00    False\n..."
}
```

### Matrix Creation

1. **Time Series Creation**: Each building block gets a pandas Series with boolean signals
2. **Resampling**: Signals are aggregated into time bins using pandas resample()
3. **Matrix Assembly**: 2D numpy array with building blocks as rows and time periods as columns

### Color Mapping

- **Normalization**: Data is normalized to [0, 1] range
- **Color Mapping**: matplotlib colormaps applied to normalized data
- **Intensity**: Higher values = darker colors

## Troubleshooting

### Common Issues

1. **No Heatmap Displayed**
   - Check that `action_details` contains valid pandas Series strings
   - Verify building blocks are properly extracted
   - Ensure time series data is available

2. **Empty Heatmap**
   - No signals found in action_details
   - Time series parsing failed
   - Check debug output for parsing errors

3. **Incorrect Signal Patterns**
   - Verify action_details format matches expected structure
   - Check time index alignment
   - Review signal parsing logic

### Debug Information

The system provides debug output showing:
- Extracted building blocks
- Signal parsing results
- Matrix creation details
- Color mapping information

## Examples

### Basic Multi-Strategy
```
Building Blocks: [fvg, vwap, order_block, Trades, Zones]
Time Periods: 09:00, 09:15, 09:30, 09:45, 10:00, ...
Signal Matrix: 
[[1, 0, 1, 0, 1, ...],  # fvg signals
 [0, 1, 0, 1, 0, ...],  # vwap signals  
 [1, 0, 0, 1, 1, ...],  # order_block signals
 [0, 0, 1, 0, 0, ...],  # trade entries
 [1, 0, 0, 0, 1, ...]]  # zone activations
```

### Signal Interpretation
- **09:00**: fvg + order_block + zone = Strong bullish setup
- **09:15**: vwap only = Moderate signal
- **09:30**: fvg + trade = Entry signal
- **09:45**: vwap + order_block = Confirmation signals
- **10:00**: fvg + order_block + zone = Strong setup again

## Future Enhancements

1. **Entry/Exit Separation**: Separate heatmaps for entry vs exit signals
2. **Signal Strength**: Weighted signals based on confidence levels
3. **Interactive Features**: Click to view detailed signal information
4. **Comparison Mode**: Compare heatmaps across different strategies
5. **Export Options**: Save heatmap as image or data
6. **Custom Color Schemes**: User-defined color mappings
7. **Signal Filtering**: Show/hide specific building blocks
8. **Time Range Selection**: Focus on specific time periods 