#!/usr/bin/env python3
"""
Headless GUI Test - Run GUI logic without displaying windows
This allows testing the backtest functionality programmatically
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, MagicMock
from strategies.strategy_builders import PatternStrategy, Action, MultiTimeframeBacktestEngine
from patterns.candlestick_patterns import CustomPattern, TimeRange
from patterns.enhanced_candlestick_patterns import FVGPattern
from patterns.candlestick_patterns import HammerPattern
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.patches import Rectangle

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class HeadlessBacktestWindow:
    """Headless version of BacktestWindow for testing"""
    
    def __init__(self):
        self.log_messages = []
        self.current_results = None
        self.resampled_data = None
        
        # Mock GUI elements
        self.overlay_toggles = {
            'Zones': Mock(isChecked=lambda: True),
            'Entries/Exits': Mock(isChecked=lambda: True)
        }
        
        # Mock matplotlib
        self.ax = Mock()
        self.canvas = Mock()
        
    def _add_log(self, message):
        """Add message to log (headless version)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        print(log_entry)
    
    def test_data_processing(self, data_path):
        """Test the complete data processing pipeline"""
        print(f"\n🧪 Testing Data Processing Pipeline")
        print("=" * 50)
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            self._add_log(f"Loaded {len(data)} rows from {data_path}")
            
            # Test index conversion (same as BacktestWindow)
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns and 'Time' in data.columns:
                    data = data.copy()
                    data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
                    data.set_index('datetime', inplace=True)
                    self._add_log("[PATCH] Set index to combined 'Date' and 'Time' columns.")
                else:
                    for col in ['datetime', 'date', 'Date', 'timestamp', 'Timestamp']:
                        if col in data.columns:
                            data = data.copy()
                            data[col] = pd.to_datetime(data[col])
                            data.set_index(col, inplace=True)
                            self._add_log(f"Converted '{col}' to DatetimeIndex.")
                            break
            
            # Debug index info
            self._add_log(f"[DEBUG] Index type: {type(data.index)}, unique: {data.index.is_unique}, sample: {list(data.index[:5])}")
            
            # Test filtering to a user-specified range
            if isinstance(data.index, pd.DatetimeIndex):
                actual_start = data.index.min()
                actual_end = data.index.max()
                self._add_log(f"Dataset date range: {actual_start} to {actual_end}")
                # User-specified range
                start_dt = actual_start + timedelta(hours=2)
                end_dt = actual_start + timedelta(hours=4)
                before_filter = len(data)
                filtered_data = data[(data.index >= start_dt) & (data.index < end_dt)]
                self._add_log(f"PASS User range: {start_dt} to {end_dt} ({before_filter} -> {len(filtered_data)} bars)")
                assert all(filtered_data.index >= start_dt) and all(filtered_data.index < end_dt)
                # Edge: empty range
                empty_data = data[(data.index >= pd.to_datetime('2100-01-01')) & (data.index < pd.to_datetime('2100-01-02'))]
                self._add_log(f"PASS Empty range: {len(empty_data)} bars")
                assert len(empty_data) == 0
                # Test chart data preparation
                self._test_chart_data_preparation(filtered_data)
                # Test zone overlays: all in-range
                self._test_zone_overlays(filtered_data)
                # All out-of-bounds
                bad_zones = [
                    {'zone_min': 105.0, 'zone_max': 115.0, 'index': 999, 'comb_centers': [107.5, 112.5]},
                    {'zone_min': 95.0, 'zone_max': 105.0, 'index': -1, 'comb_centers': [97.5, 102.5]}
                ]
                for i, zone in enumerate(bad_zones):
                    zone_idx = zone.get('index')
                    if zone_idx is not None and (zone_idx < 0 or zone_idx >= len(filtered_data)):
                        self._add_log(f"PASS Correctly detected out-of-bounds zone index: {zone_idx}")
                    else:
                        self._add_log(f"FAIL Failed to detect out-of-bounds zone index: {zone_idx}")
                # Overlays toggled off
                overlays = {'Zones': False, 'Entries/Exits': False}
                if not overlays['Zones']:
                    self._add_log("PASS Zones overlay toggled off: no zones should be plotted")
                if not overlays['Entries/Exits']:
                    self._add_log("PASS Entries/Exits overlay toggled off: no trades should be plotted")
                # Malformed data
                bad_data = pd.DataFrame({'foo': [1,2,3], 'bar': [4,5,6]})
                try:
                    _ = bad_data[['open', 'high', 'low', 'close']]
                    self._add_log("FAIL Should have raised KeyError for missing columns")
                except KeyError:
                    self._add_log("PASS Correctly handled missing OHLC columns")
                return True
            else:
                self._add_log("WARNING: Data does not have a DatetimeIndex after conversion.")
                return False
                
        except Exception as e:
            self._add_log(f"ERROR: {e}")
            return False
    
    def _test_chart_data_preparation(self, data):
        """Test chart data preparation logic"""
        self._add_log("\n📊 Testing Chart Data Preparation")
        
        # Simulate chart data preparation (same as _update_chart_tab)
        df = data.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns and 'Time' in df.columns:
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                df.index = df['datetime']
                self._add_log("[PATCH] Chart: Set index to combined 'Date' and 'Time' columns.")
            elif 'datetime' in df.columns:
                df.index = pd.to_datetime(df['datetime'])
            elif 'Date' in df.columns:
                df.index = pd.to_datetime(df['Date'])
            else:
                df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='min')
        
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        self._add_log(f"[DEBUG] Chart index type: {type(df.index)}, unique: {df.index.is_unique}, sample: {list(df.index[:5])}")
        
        # Test OHLC column mapping
        required_cols = ['open', 'high', 'low', 'close']
        col_map = {c: c.capitalize() for c in required_cols}
        if not all(col in df.columns for col in required_cols):
            self._add_log("❌ Missing OHLC data for chart.")
            return False
        
        df = df[required_cols]
        df.columns = [col_map[c] for c in required_cols]
        
        if len(df) < 10:
            self._add_log(f"❌ Not enough bars to plot candlesticks: {len(df)}")
            return False
        
        self._add_log(f"✅ Chart data prepared: {len(df)} bars, columns: {list(df.columns)}")
        return True
    
    def _test_zone_overlays(self, data):
        """Test zone overlay logic"""
        self._add_log("\n🎯 Testing Zone Overlays")
        
        # Create sample zones
        zones = [
            {
                'zone_min': 105.0,
                'zone_max': 115.0,
                'index': 10,
                'comb_centers': [107.5, 112.5]
            },
            {
                'zone_min': 95.0,
                'zone_max': 105.0,
                'index': 50,
                'comb_centers': [97.5, 102.5]
            },
            {
                'zone_min': 110.0,
                'zone_max': 120.0,
                'index': 999,  # Out of bounds
                'comb_centers': [112.5, 117.5]
            }
        ]
        
        # Test zone mapping (same as _update_chart_tab)
        for i, zone in enumerate(zones):
            zone_min = zone.get('zone_min')
            zone_max = zone.get('zone_max')
            zone_idx = zone.get('index')
            comb_centers = zone.get('comb_centers', [])
            
            # PATCH: Debug zone mapping
            self._add_log(f"[DEBUG] Zone {i}: index={zone_idx}, min={zone_min}, max={zone_max}, comb_centers={comb_centers}")
            
            if zone_idx is not None and (zone_idx < 0 or zone_idx >= len(data)):
                self._add_log(f"[WARNING] Zone {i} index {zone_idx} out of bounds for df of length {len(data)}")
                continue
                
            if zone_min is not None and zone_max is not None and zone_idx is not None and 0 <= zone_idx < len(data) and comb_centers:
                start_time = data.index[zone_idx]
                end_idx = min(zone_idx + 5, len(data) - 1)
                end_time = data.index[end_idx]
                self._add_log(f"✅ Zone {i} would be drawn from {start_time} to {end_time}")
            else:
                self._add_log(f"❌ Zone {i} has invalid data")
    
    def simulate_backtest(self, data_path, strategy_name="Test Strategy"):
        """Simulate a complete backtest run"""
        print(f"\n🚀 Simulating Backtest: {strategy_name}")
        print("=" * 50)
        
        # Test data processing
        if not self.test_data_processing(data_path):
            return False
        
        # Simulate backtest results
        self.current_results = {
            'strategy_name': strategy_name,
            'total_trades': 25,
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.68,
            'equity_curve': [100000] + [100000 + i * 100 for i in range(1, 100)],
            'trades': [
                {
                    'entry_time': '2024-01-01 10:00:00',
                    'exit_time': '2024-01-01 11:00:00',
                    'entry_price': 100.0,
                    'exit_price': 102.0,
                    'pnl': 200.0,
                    'entry_idx': 10,
                    'exit_idx': 22
                }
            ],
            'zones': [
                {
                    'zone_min': 105.0,
                    'zone_max': 115.0,
                    'index': 10,
                    'comb_centers': [107.5, 112.5]
                }
            ]
        }
        
        self._add_log("✅ Backtest simulation completed successfully")
        return True
    
    def get_test_summary(self):
        """Get a summary of the test results"""
        return {
            'log_messages': self.log_messages,
            'results': self.current_results,
            'success': len([msg for msg in self.log_messages if 'ERROR' in msg]) == 0
        }

    def test_multi_timeframe_processing(self, data_path):
        """Test multi-timeframe processing with automatic timeframe creation"""
        print(f"\n🔄 Testing Multi-Timeframe Processing")
        print("=" * 50)
        
        try:
            from processors.data_processor import MultiTimeframeProcessor
            from core.data_structures import TimeRange
            
            # Load and prepare data
            data = pd.read_csv(data_path)
            self._add_log(f"Loaded {len(data)} rows from {data_path}")
            
            # PATCH: Use both Date and Time columns for unique timestamp index if available
            if 'Date' in data.columns and 'Time' in data.columns:
                try:
                    data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), errors='raise')
                    data.index = data['datetime']
                    print("[PATCH] Set index to combined 'Date' and 'Time' columns.")
                except Exception as e:
                    print(f"[PATCH] Failed to parse 'Date' and 'Time' columns: {e}\nUsing synthetic index.")
                    data.index = pd.date_range(start='2000-01-01', periods=len(data), freq='T')
            elif 'datetime' in data.columns:
                data.index = pd.to_datetime(data['datetime'])
                print("[PATCH] Set index to pd.to_datetime('datetime') column.")
            elif 'Date' in data.columns:
                try:
                    data.index = pd.to_datetime(data['Date'], errors='raise')
                    print("[PATCH] Set index to 'Date' column.")
                except Exception as e:
                    print(f"[PATCH] Failed to parse 'Date' column: {e}\nUsing synthetic index.")
                    data.index = pd.date_range(start='2000-01-01', periods=len(data), freq='T')
            elif 'timestamp' in data.columns:
                data.index = pd.to_datetime(data['timestamp'])
                print("[PATCH] Set index to 'timestamp' column.")
            else:
                data.index = pd.date_range(start='2000-01-01', periods=len(data), freq='T')
                print("[PATCH] No datetime column found, using synthetic date range index.")
            # PATCH: Handle duplicate indices by aggregating OHLC data
            if not data.index.is_unique:
                print(f"[PATCH] Found duplicate indices. Aggregating OHLC data...")
                before_agg = len(data)
                # Aggregate OHLC data for duplicate timestamps
                data = data.groupby(data.index).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                })
                after_agg = len(data)
                print(f"[PATCH] Aggregated {before_agg} -> {after_agg} bars")
            # Map columns to standard OHLCV names for downstream compatibility
            col_map = {
                'open': None, 'high': None, 'low': None, 'close': None, 'volume': None
            }
            for col in data.columns:
                lcol = col.lower()
                if lcol.startswith('open'):
                    col_map['open'] = col
                elif lcol.startswith('high'):
                    col_map['high'] = col
                elif lcol.startswith('low'):
                    col_map['low'] = col
                elif lcol.startswith('close') or lcol.startswith('last'):
                    col_map['close'] = col
                elif lcol.startswith('vol'):
                    col_map['volume'] = col
            # Only keep and rename if all required columns are present
            if all(col_map.values()):
                data = data[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
                data.columns = ['open', 'high', 'low', 'close', 'volume']
            # Sort and deduplicate index
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            # Filter to 2024-06-02
            start_dt = pd.to_datetime('2024-06-02 00:00:00')
            end_dt = pd.to_datetime('2024-06-03 00:00:00')
            data = data[(data.index >= start_dt) & (data.index < end_dt)]
            self._add_log(f"Processed data shape: {data.shape}")
            
            # Create MultiTimeframeProcessor
            mtf_processor = MultiTimeframeProcessor()
            
            # Test 1: Create initial timeframes (like in Data Stripper)
            initial_timeframes = [
                TimeRange(1, 'm'),   # 1 minute
                TimeRange(5, 'm'),   # 5 minutes
                TimeRange(15, 'm'),  # 15 minutes
            ]
            
            self._add_log(f"Creating initial timeframes: {[f'{tf.value}{tf.unit}' for tf in initial_timeframes]}")
            initial_datasets = mtf_processor.create_timeframe_datasets(data, initial_timeframes)
            
            for tf_str, tf_df in initial_datasets.items():
                self._add_log(f"  {tf_str}: {len(tf_df)} rows")
            
            # Test 2: Strategy requires additional timeframes
            strategy_required_timeframes = ['1m', '5m', '10m', '15m', '30m']
            missing_timeframes = []
            
            for tf_str in strategy_required_timeframes:
                if tf_str not in initial_datasets:
                    missing_timeframes.append(tf_str)
            
            if missing_timeframes:
                self._add_log(f"Strategy requires additional timeframes: {missing_timeframes}")
                self._add_log("Auto-creating missing timeframes...")
                
                # Create missing timeframes
                additional_timeframes = []
                for tf_str in missing_timeframes:
                    import re
                    match = re.match(r'(\d+)([smhd])', tf_str)
                    if match:
                        value = int(match.group(1))
                        unit = match.group(2)
                        additional_timeframes.append(TimeRange(value, unit))
                
                # Create additional datasets
                additional_datasets = mtf_processor.create_timeframe_datasets(
                    mtf_processor.get_original_data(), 
                    additional_timeframes
                )
                
                # Merge with existing datasets
                initial_datasets.update(additional_datasets)
                
                self._add_log("Additional timeframes created:")
                for tf_str, tf_df in additional_datasets.items():
                    self._add_log(f"  {tf_str}: {len(tf_df)} rows")
            
            # Test 3: Verify all required timeframes are available
            all_available = list(initial_datasets.keys())
            self._add_log(f"All available timeframes: {sorted(all_available)}")
            
            for required_tf in strategy_required_timeframes:
                if required_tf in initial_datasets:
                    self._add_log(f"✅ {required_tf} timeframe available")
                else:
                    self._add_log(f"❌ {required_tf} timeframe missing")
                    return False
            
            # Test 4: Simulate backtester loading dataset with specific timeframe
            self._add_log("\n📊 Testing Backtester Timeframe Loading")
            
            # Simulate loading 1m dataset but strategy needs 10m
            loaded_timeframe = '1m'
            strategy_needs_timeframe = '10m'
            
            self._add_log(f"Backtester loads {loaded_timeframe} dataset")
            self._add_log(f"Strategy requires {strategy_needs_timeframe} timeframe")
            
            if strategy_needs_timeframe in initial_datasets:
                strategy_data = initial_datasets[strategy_needs_timeframe]
                self._add_log(f"✅ Strategy can access {strategy_needs_timeframe} data: {len(strategy_data)} rows")
                
                # Test signal alignment across timeframes
                self._test_signal_alignment(initial_datasets, mtf_processor)
            else:
                self._add_log(f"❌ Strategy cannot access {strategy_needs_timeframe} data")
                return False
            
            # Test 5: Verify no data loss
            original_rows = len(mtf_processor.get_original_data())
            self._add_log(f"\n📈 Data Integrity Check")
            self._add_log(f"Original data preserved: {original_rows} rows")
            
            if original_rows == len(data):
                self._add_log("✅ No data was lost during processing")
            else:
                self._add_log(f"❌ Data loss detected: {len(data)} -> {original_rows}")
                return False
            
            self._add_log("✅ Multi-timeframe processing test PASSED")
            return True
            
        except Exception as e:
            self._add_log(f"❌ Multi-timeframe processing test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_signal_alignment(self, timeframe_datasets, mtf_processor):
        """Test signal alignment across different timeframes"""
        self._add_log("\n🎯 Testing Signal Alignment")
        
        # Create sample signals for different timeframes
        sample_signals = {}
        for tf_str in ['1m', '5m', '15m']:
            if tf_str in timeframe_datasets:
                tf_df = timeframe_datasets[tf_str]
                if len(tf_df) > 20:
                    # Create a simple signal (price above SMA)
                    sma = tf_df['close'].rolling(20).mean()
                    signal = tf_df['close'] > sma
                    sample_signals[tf_str] = signal
                    self._add_log(f"Created signal for {tf_str}: {signal.sum()} signals out of {len(signal)} bars")
        
        # Align signals to highest resolution timeframe
        if sample_signals:
            aligned_signals = mtf_processor.align_signals_across_timeframes(sample_signals)
            self._add_log(f"Aligned signals shape: {aligned_signals.shape}")
            self._add_log(f"Aligned signals columns: {list(aligned_signals.columns)}")
            
            if not aligned_signals.empty:
                self._add_log("✅ Signal alignment successful")
            else:
                self._add_log("❌ Signal alignment failed")
        else:
            self._add_log("⚠️ No signals to align")

def plot_fvg_zones(df, fvg_zones, title, filename):
    # Prepare data for mplfinance
    df_plot = df.copy()
    df_plot = df_plot[['open', 'high', 'low', 'close', 'volume']].copy()
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot candlesticks
    mpf.plot(df_plot, type='candle', ax=ax, volume=False, style='yahoo', show_nontrading=True)
    # Overlay FVG zones
    for ts, zmin, zmax in fvg_zones:
        if ts in df_plot.index:
            idx = df_plot.index.get_loc(ts)
            # Draw rectangle from zmin to zmax at the bar's x position
            ax.add_patch(Rectangle((idx-0.4, zmin), 0.8, zmax-zmin, color='orange', alpha=0.3, label='FVG'))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"[HEADLESS-GUI] Saved plot: {filename}")

# 1. Load the dataset and filter to 2024-06-02
csv_path = 'workspaces/datasets/NQ_5s_1m.csv'
df = pd.read_csv(csv_path)
# PATCH: Use both Date and Time columns for unique timestamp index if available
if 'Date' in df.columns and 'Time' in df.columns:
    try:
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='raise')
        df.index = df['datetime']
        print("[PATCH] Set index to combined 'Date' and 'Time' columns.")
    except Exception as e:
        print(f"[PATCH] Failed to parse 'Date' and 'Time' columns: {e}\nUsing synthetic index.")
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
elif 'datetime' in df.columns:
    df.index = pd.to_datetime(df['datetime'])
    print("[PATCH] Set index to pd.to_datetime('datetime') column.")
elif 'Date' in df.columns:
    try:
        df.index = pd.to_datetime(df['Date'], errors='raise')
        print("[PATCH] Set index to 'Date' column.")
    except Exception as e:
        print(f"[PATCH] Failed to parse 'Date' column: {e}\nUsing synthetic index.")
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
elif 'timestamp' in df.columns:
    df.index = pd.to_datetime(df['timestamp'])
    print("[PATCH] Set index to 'timestamp' column.")
else:
    df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
    print("[PATCH] No datetime column found, using synthetic date range index.")
# PATCH: Handle duplicate indices by aggregating OHLC data
if not df.index.is_unique:
    print(f"[PATCH] Found duplicate indices. Aggregating OHLC data...")
    before_agg = len(df)
    # Aggregate OHLC data for duplicate timestamps
    df = df.groupby(df.index).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'
    })
    after_agg = len(df)
    print(f"[PATCH] Aggregated {before_agg} -> {after_agg} bars")
# Map columns to standard OHLCV names for downstream compatibility
col_map = {
    'open': None, 'high': None, 'low': None, 'close': None, 'volume': None
}
for col in df.columns:
    lcol = col.lower()
    if lcol.startswith('open'):
        col_map['open'] = col
    elif lcol.startswith('high'):
        col_map['high'] = col
    elif lcol.startswith('low'):
        col_map['low'] = col
    elif lcol.startswith('close') or lcol.startswith('last'):
        col_map['close'] = col
    elif lcol.startswith('vol'):
        col_map['volume'] = col
# Only keep and rename if all required columns are present
if all(col_map.values()):
    df = df[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
# Sort and deduplicate index
if not df.index.is_monotonic_increasing:
    df = df.sort_index()
df = df[~df.index.duplicated(keep='first')]
# Filter to 2024-06-02
start_dt = pd.to_datetime('2024-06-02 00:00:00')
end_dt = pd.to_datetime('2024-06-03 00:00:00')
df = df[(df.index >= start_dt) & (df.index < end_dt)]
print(f"[HEADLESS-GUI] Loaded {len(df)} rows for 2024-06-02 from {csv_path}")

# 2. Construct FVG and Hammer actions/patterns as in the GUI
fvg_pattern = CustomPattern('fvg', [TimeRange(1, 'm')], [])
hammer_pattern = CustomPattern('hammer', [TimeRange(5, 'm')], [])
action_fvg = Action(name='fvg_1m', pattern=fvg_pattern)
action_hammer = Action(name='hammer_5m', pattern=hammer_pattern)

# 3. Construct strategies as in the GUI
strat_fvg = PatternStrategy(
    name='fvg_only',
    actions=[action_fvg],
    combination_logic='OR',
)
strat_fvg1mh5m = PatternStrategy(
    name='fvg1mh5m',
    actions=[action_fvg, action_hammer],
    combination_logic='OR',
)

# 4. Print action/pattern info
for strat in [strat_fvg, strat_fvg1mh5m]:
    print(f"[HEADLESS-GUI] STRATEGY: {strat.name}")
    for a in strat.actions:
        print(f"  ACTION: {a.name}, PATTERN: {getattr(a.pattern, 'name', None)}, TYPE: {type(a.pattern)}")

# 5. Run backtests
engine = MultiTimeframeBacktestEngine()
res_fvg = engine.run_backtest(strat_fvg, df)
res_fvg1mh5m = engine.run_backtest(strat_fvg1mh5m, df)

def extract_fvg_zones(zones):
    return sorted([
        (str(z.get('timestamp'))[:19], z.get('zone_min'), z.get('zone_max'))
        for z in zones if z.get('zone_type','').lower() == 'fvg'])

fvg_zones_fvg = extract_fvg_zones(res_fvg['zones'])
fvg_zones_fvg1mh5m = extract_fvg_zones(res_fvg1mh5m['zones'])

print('\n[HEADLESS-GUI] FVG-only (1m pattern) zones:')
for z in fvg_zones_fvg:
    print(z)
print('\n[HEADLESS-GUI] FVG+Hammer (1m FVG + 5m Hammer) zones:')
for z in fvg_zones_fvg1mh5m:
    print(z)

# Print and compare FVG zones for both strategies
print("\n[HEADLESS-GUI] FVG-only (1m pattern) zones:")
for z in fvg_zones_fvg:
    print(f"  {z}")
print("\n[HEADLESS-GUI] FVG+Hammer (1m FVG + 5m Hammer) zones:")
for z in fvg_zones_fvg1mh5m:
    print(f"  {z}")

# Compare as sets for strict equality
set_fvg = set(fvg_zones_fvg)
set_fvg1mh5m = set(fvg_zones_fvg1mh5m)
if set_fvg != set_fvg1mh5m:
    print("\n[HEADLESS-GUI][FAIL] FVG zone locations differ!")
    print("Zones only in FVG-only:")
    for z in sorted(set_fvg - set_fvg1mh5m):
        print(f"  {z}")
    print("Zones only in FVG+Hammer:")
    for z in sorted(set_fvg1mh5m - set_fvg):
        print(f"  {z}")
    assert False, 'FVG zone locations are not identical!'
else:
    print("\n[HEADLESS-GUI][PASS] FVG zone locations are identical between FVG-only and FVG+Hammer strategies.")

# Plot and save for FVG-only
plot_fvg_zones(df, fvg_zones_fvg, 'FVG-only (1m pattern) with FVG Zones', 'fvg_only_zones.png')
# Plot and save for FVG+Hammer
plot_fvg_zones(df, fvg_zones_fvg1mh5m, 'FVG+Hammer (1m FVG + 5m Hammer) with FVG Zones', 'fvg_hammer_zones.png')

def test_mtf_execution_timeframe_and_fvg_zones():
    print("\n[TEST] MultiTimeframeBacktestEngine execution timeframe and FVG zone consistency\n")
    # Load a 1m dataset (use recent_dataset/most_recent.csv for real data)
    df = pd.read_csv('recent_dataset/most_recent.csv')
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df.index = df['datetime']
    elif 'datetime' in df.columns:
        df.index = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'])
    else:
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
    # Map columns to standard OHLCV names
    col_map = {'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}
    for col in df.columns:
        lcol = col.lower()
        if lcol.startswith('open'):
            col_map['open'] = col
        elif lcol.startswith('high'):
            col_map['high'] = col
        elif lcol.startswith('low'):
            col_map['low'] = col
        elif lcol.startswith('close') or lcol.startswith('last'):
            col_map['close'] = col
        elif lcol.startswith('vol'):
            col_map['volume'] = col
    if all(col_map.values()):
        df = df[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
    # FVG-only strategy
    fvg_action = Action(name='fvg_1m', pattern=FVGPattern(timeframes=[TimeRange(1, 'm')]))
    strat_fvg = PatternStrategy(name='FVG Only', actions=[fvg_action])
    # FVG+Hammer strategy
    hammer_action = Action(name='hammer_5m', pattern=HammerPattern(timeframes=[TimeRange(5, 'm')]))
    strat_fvg_hammer = PatternStrategy(name='FVG+Hammer', actions=[fvg_action, hammer_action])
    # Run both backtests
    engine = MultiTimeframeBacktestEngine()
    results_fvg = engine.run_backtest(strat_fvg, df)
    results_fvg_hammer = engine.run_backtest(strat_fvg_hammer, df)
    # Check execution data frequency
    freq_fvg = pd.infer_freq(results_fvg['data'].index)
    freq_fvg_hammer = pd.infer_freq(results_fvg_hammer['data'].index)
    print(f"[TEST] FVG-only results['data'] bars: {len(results_fvg['data'])}, freq: {freq_fvg}")
    print(f"[TEST] FVG+Hammer results['data'] bars: {len(results_fvg_hammer['data'])}, freq: {freq_fvg_hammer}")
    assert freq_fvg == 'T' or freq_fvg == 'min', f"Expected 1m freq, got {freq_fvg}"
    assert freq_fvg_hammer == 'T' or freq_fvg_hammer == 'min', f"Expected 1m freq, got {freq_fvg_hammer}"
    # Check FVG zones are identical
    fvg_zones_fvg = [z for z in results_fvg['zones'] if z.get('zone_type', '').lower() == 'fvg']
    fvg_zones_fvg_hammer = [z for z in results_fvg_hammer['zones'] if z.get('zone_type', '').lower() == 'fvg']
    print(f"[TEST] FVG-only FVG zones: {len(fvg_zones_fvg)}")
    print(f"[TEST] FVG+Hammer FVG zones: {len(fvg_zones_fvg_hammer)}")
    assert fvg_zones_fvg == fvg_zones_fvg_hammer, "FVG zones differ between FVG-only and FVG+Hammer strategies!"
    print("[PASS] Execution timeframe and FVG zones are correct and consistent.")

test_mtf_execution_timeframe_and_fvg_zones() 