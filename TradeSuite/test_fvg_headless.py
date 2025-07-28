#!/usr/bin/env python3
"""
Headless FVG Test - No GUI Required
===================================
Comprehensive test script to verify FVG detection and plotting logic
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action, MultiTimeframeBacktestEngine
from patterns.candlestick_patterns import CustomPattern, TimeRange

def create_test_fvg_strategy():
    """Create a test strategy with only FVG location gate"""
    print("Creating test FVG strategy...")
    
    # Create a simple action with only FVG location gate
    action = Action(
        name="FVG Test",
        location_strategy="FVG",
        location_params={
            'fvg_epsilon': 0.0,  # No buffer
            'fvg_N': 3,
            'fvg_sigma': 0.1,
            'fvg_beta1': 0.7,
            'fvg_beta2': 0.3,
            'fvg_phi': 0.2,
            'fvg_lambda': 0.0,
            'fvg_gamma': 0.95,
            'fvg_tau_bars': 15,
            'fvg_drop_threshold': 0.01,
        }
    )
    
    # Create strategy with only FVG configuration
    strategy = PatternStrategy(
        name="FVG Test Strategy",
        actions=[action],
        gates_and_logic={
            'location_gate': True  # Enable location gate
        },
        location_gate_params={
            # FVG-specific parameters
            'fvg_epsilon': 1.0,
            'fvg_N': 3,
            'fvg_sigma': 0.1,
            'fvg_beta1': 0.7,
            'fvg_beta2': 0.3,
            'fvg_phi': 0.2,
            'fvg_lambda': 0.0,
            'fvg_gamma': 0.95,
            'fvg_tau_bars': 50,
            'fvg_drop_threshold': 0.01,
            'fvg_min_gap_size': 0.1,  # Lower threshold for testing
        }
    )
    
    print(f"Strategy created with gates_and_logic: {strategy.gates_and_logic}")
    print(f"Strategy location_gate_params keys: {list(strategy.location_gate_params.keys())}")
    
    return strategy

def load_nq_dataset():
    """Load the NQ dataset"""
    print("Loading NQ dataset...")
    
    # Try to load the dataset
    csv_path = 'NQ_5s_1m.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} rows, columns: {list(df.columns)}")
    
    # Ensure we have the right columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns. Available: {list(df.columns)}")
        # Try to map common column names
        col_map = {}
        for col in df.columns:
            lcol = col.lower()
            if lcol.startswith('open'):
                col_map['open'] = col
            elif lcol.startswith('high'):
                col_map['high'] = col
            elif lcol.startswith('low'):
                col_map['low'] = col
            elif lcol.startswith('close'):
                col_map['close'] = col
            elif lcol.startswith('vol'):
                col_map['volume'] = col
        
        if len(col_map) == 5:
            df = df[list(col_map.values())]
            df.columns = required_cols
            print("Successfully mapped column names")
        else:
            print(f"Could not map columns. Found: {col_map}")
            return None
    
    # Set index
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df.index = df['datetime']
        df = df.drop(['Date', 'Time', 'datetime'], axis=1, errors='ignore')
    elif 'datetime' in df.columns:
        df.index = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1, errors='ignore')
    else:
        df.index = pd.date_range('2024-01-01', periods=len(df), freq='1min')
    
    print(f"Dataset index: {df.index.min()} to {df.index.max()}")
    print(f"Sample data:\n{df.head()}")
    
    return df

def test_fvg_detection_directly(strategy, data):
    """Test FVG detection directly without backtester"""
    print("\n=== Testing FVG Detection Directly ===")
    
    # Test FVG detection at different indices
    test_indices = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    total_fvg_zones = 0
    fvg_details = []
    
    for idx in test_indices:
        if idx >= len(data):
            continue
            
        print(f"\nTesting FVG detection at index {idx}...")
        
        # Call the FVG detection method directly
        fvg_zones = strategy._detect_fvg_zones(data, idx)
        
        print(f"Found {len(fvg_zones)} FVG zones at index {idx}")
        
        for i, zone in enumerate(fvg_zones):
            zone_info = {
                'index': idx,
                'direction': zone['direction'],
                'zone_min': zone['zone_min'],
                'zone_max': zone['zone_max'],
                'gap_size': zone.get('gap_size', 0),
                'strength': zone.get('strength', 0),
                'comb_centers': zone.get('comb_centers', [])
            }
            fvg_details.append(zone_info)
            total_fvg_zones += 1
            
            print(f"  FVG {i+1}: {zone['direction']} - min={zone['zone_min']:.2f}, max={zone['zone_max']:.2f}, gap={zone.get('gap_size', 0):.2f}")
    
    # Test the zone configuration logic
    print("\n=== Testing Zone Configuration ===")
    all_zones = strategy._detect_all_zone_types(data, 20)
    
    zone_types = {}
    for zone in all_zones:
        zone_type = zone.get('type', 'Unknown')
        zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
    
    print(f"Total zones detected: {len(all_zones)}")
    for zone_type, count in zone_types.items():
        print(f"  {zone_type}: {count}")
    
    return total_fvg_zones, fvg_details

def test_strategy_evaluation(strategy, data):
    """Test strategy evaluation to generate zones"""
    print("\n=== Testing Strategy Evaluation ===")
    
    # Clear any existing zones
    strategy.simple_zones = []
    
    # Run strategy evaluation
    signals, action_details = strategy.evaluate(data)
    
    print(f"Strategy evaluation complete:")
    print(f"  - Generated {len(strategy.simple_zones)} zones")
    print(f"  - Generated {signals.sum()} signals")
    
    # Analyze the zones
    zone_types = {}
    zone_directions = {}
    
    for zone in strategy.simple_zones:
        zone_type = zone.get('zone_type', 'Unknown')
        zone_direction = zone.get('zone_direction', 'neutral')
        
        zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
        zone_directions[zone_direction] = zone_directions.get(zone_direction, 0) + 1
    
    print(f"Zone analysis:")
    for zone_type, count in zone_types.items():
        print(f"  {zone_type}: {count}")
    
    print(f"Direction analysis:")
    for direction, count in zone_directions.items():
        print(f"  {direction}: {count}")
    
    return strategy.simple_zones, signals

def create_visualization(data, zones, output_file="fvg_test_results.png"):
    """Create a visualization of the FVG zones"""
    print(f"\n=== Creating Visualization: {output_file} ===")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot candlesticks
    for i in range(len(data)):
        bar = data.iloc[i]
        color = 'green' if bar['close'] >= bar['open'] else 'red'
        
        # Body
        body_height = bar['close'] - bar['open']
        body_bottom = min(bar['open'], bar['close'])
        
        rect = patches.Rectangle((i, body_bottom), 0.8, abs(body_height), 
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Wicks
        ax.plot([i+0.4, i+0.4], [bar['low'], bar['high']], color='black', linewidth=1)
    
    # Plot FVG zones
    fvg_zones = [z for z in zones if z.get('type') == 'FVG']
    
    for zone in fvg_zones:
        zone_min = zone['zone_min']
        zone_max = zone['zone_max']
        direction = zone.get('direction', 'neutral')
        
        # Color based on direction
        if direction == 'bullish':
            color = 'lightgreen'
            alpha = 0.3
        elif direction == 'bearish':
            color = 'lightcoral'
            alpha = 0.3
        else:
            color = 'lightblue'
            alpha = 0.3
        
        # Create zone rectangle
        zone_rect = patches.Rectangle((0, zone_min), len(data), zone_max - zone_min,
                                    facecolor=color, edgecolor='darkgreen' if direction == 'bullish' else 'darkred',
                                    alpha=alpha, linewidth=2)
        ax.add_patch(zone_rect)
        
        # Add label
        ax.text(len(data) + 1, (zone_min + zone_max) / 2, 
               f"{direction.upper()} FVG\n{zone_min:.1f} - {zone_max:.1f}",
               verticalalignment='center', fontsize=8, fontweight='bold')
    
    # Plot comb centers
    for zone in fvg_zones:
        comb_centers = zone.get('comb_centers', [])
        for center in comb_centers:
            ax.plot([0, len(data)], [center, center], '--', color='purple', alpha=0.5, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('Price')
    ax.set_title('FVG Zone Detection Test Results')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightgreen', alpha=0.3, label='Bullish FVG'),
        patches.Patch(color='lightcoral', alpha=0.3, label='Bearish FVG'),
        plt.Line2D([0], [0], color='purple', linestyle='--', label='Comb Centers')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    return output_file

def validate_fvg_logic(data, zones):
    """Validate FVG detection logic"""
    print("\n=== Validating FVG Logic ===")
    
    fvg_zones = [z for z in zones if z.get('type') == 'FVG']
    
    if not fvg_zones:
        print("❌ No FVG zones detected!")
        return False
    
    print(f"✅ Found {len(fvg_zones)} FVG zones")
    
    # Validate each FVG zone
    valid_zones = 0
    
    for zone in fvg_zones:
        zone_min = zone['zone_min']
        zone_max = zone['zone_max']
        direction = zone.get('direction', 'neutral')
        
        # Basic validation
        if zone_min >= zone_max:
            print(f"❌ Invalid zone: min ({zone_min}) >= max ({zone_max})")
            continue
        
        if direction not in ['bullish', 'bearish']:
            print(f"❌ Invalid direction: {direction}")
            continue
        
        # Check if zone bounds are reasonable
        price_range = data['high'].max() - data['low'].min()
        zone_size = zone_max - zone_min
        
        if zone_size > price_range * 0.5:  # Zone shouldn't be more than 50% of price range
            print(f"⚠️  Large zone: {zone_size:.2f} ({(zone_size/price_range)*100:.1f}% of price range)")
        
        valid_zones += 1
        print(f"✅ Valid {direction} FVG: {zone_min:.2f} - {zone_max:.2f}")
    
    print(f"\nValidation Summary:")
    print(f"  - Total FVG zones: {len(fvg_zones)}")
    print(f"  - Valid zones: {valid_zones}")
    print(f"  - Success rate: {(valid_zones/len(fvg_zones))*100:.1f}%")
    
    return valid_zones > 0

def run_comprehensive_test():
    """Run comprehensive FVG test"""
    print("=== FVG Headless Test Suite ===")
    print("=" * 50)
    
    # Load dataset
    data = load_nq_dataset()
    if data is None:
        print("❌ Failed to load dataset. Exiting.")
        return False
    
    # Create strategy
    strategy = create_test_fvg_strategy()
    
    # Test 1: Direct FVG detection
    print("\n" + "="*50)
    print("TEST 1: Direct FVG Detection")
    print("="*50)
    
    total_fvg_zones, fvg_details = test_fvg_detection_directly(strategy, data)
    
    if total_fvg_zones == 0:
        print("❌ No FVG zones detected in direct testing!")
        return False
    
    print(f"✅ Direct FVG detection: {total_fvg_zones} zones found")
    
    # Test 2: Strategy evaluation
    print("\n" + "="*50)
    print("TEST 2: Strategy Evaluation")
    print("="*50)
    
    zones, signals = test_strategy_evaluation(strategy, data)
    
    if len(zones) == 0:
        print("❌ No zones generated by strategy evaluation!")
        return False
    
    print(f"✅ Strategy evaluation: {len(zones)} zones generated")
    
    # Test 3: FVG logic validation
    print("\n" + "="*50)
    print("TEST 3: FVG Logic Validation")
    print("="*50)
    
    logic_valid = validate_fvg_logic(data, zones)
    
    if not logic_valid:
        print("❌ FVG logic validation failed!")
        return False
    
    # Test 4: Create visualization
    print("\n" + "="*50)
    print("TEST 4: Visualization")
    print("="*50)
    
    try:
        output_file = create_visualization(data, zones)
        print(f"✅ Visualization created: {output_file}")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    fvg_zones = [z for z in zones if z.get('type') == 'FVG']
    other_zones = [z for z in zones if z.get('type') != 'FVG']
    
    print(f"✅ FVG zones detected: {len(fvg_zones)}")
    print(f"⚠️  Other zones detected: {len(other_zones)}")
    print(f"✅ Total signals generated: {signals.sum()}")
    
    if len(fvg_zones) > 0 and len(other_zones) == 0:
        print("🎉 SUCCESS: Only FVG zones detected as expected!")
        return True
    elif len(fvg_zones) > 0:
        print("⚠️  PARTIAL SUCCESS: FVG zones detected, but other zones also present")
        return True
    else:
        print("❌ FAILURE: No FVG zones detected")
        return False

def run_fvg_detection_on_custom_dataset():
    print("\n=== FVG Detection Test: Handcrafted Dataset ===")
    df = pd.read_csv('custom_fvg_test_dataset.csv', parse_dates=['datetime'])
    print('Loaded df shape:', df.shape)
    print(df.head())
    df.index = pd.to_datetime(df['datetime'])
    from strategies.strategy_builders import PatternStrategy, Action
    # Create a minimal FVG-only strategy
    action = Action(
        name="FVG Test",
        location_strategy="FVG",
        location_params={
            'fvg_epsilon': 0.0,  # No buffer
            'fvg_N': 3,
            'fvg_sigma': 0.1,
            'fvg_beta1': 0.7,
            'fvg_beta2': 0.3,
            'fvg_phi': 0.2,
            'fvg_lambda': 0.0,
            'fvg_gamma': 0.95,
            'fvg_tau_bars': 15,
            'fvg_drop_threshold': 0.01,
        }
    )
    strat = PatternStrategy(actions=[action], gates_and_logic={'location_gate': True})
    detected_fvgs = []
    for i in range(2, len(df)-1):
        zones = strat._detect_fvg_zones(df, i)
        for z in zones:
            detected_fvgs.append({
                'index': i,
                'datetime': df.index[i],
                'zone_type': z['type'],
                'zone_direction': z['direction'],
                'zone_min': z['zone_min'],
                'zone_max': z['zone_max']
            })
    print("Detected FVGs:")
    for fvg in detected_fvgs:
        print(fvg)
    print("\nExpected:")
    print("Bullish FVG at 2024-01-03 (index 2): 106.0 to 108.0")
    print("Bearish FVG at 2024-01-05 (index 4): 97.0 to 108.0")
    print("\nCompare the above output to expected FVGs.")

def make_pattern_detector(trigger_times):
    def detect(self, data):
        idx = data.index.astype(str)
        return pd.Series(idx.isin(trigger_times), index=data.index)
    return detect

def test_fvg_vs_fvg_or_hammer():
    # 1. Create synthetic 1m dataset
    times = pd.date_range('2024-03-07 09:00:00', periods=20, freq='1min')
    df = pd.DataFrame({
        'open': [100 + i for i in range(20)],
        'high': [101 + i for i in range(20)],
        'low': [99 + i for i in range(20)],
        'close': [100 + i for i in range(20)],
        'volume': 1000,
    }, index=times)

    # 2. Define FVG pattern triggers
    times_fvg = ['2024-03-07 09:05:00', '2024-03-07 09:10:00', '2024-03-07 09:15:00']
    fvg_pattern = CustomPattern('fvg', [TimeRange(1, 'm')], [])
    fvg_pattern.detect = make_pattern_detector(times_fvg).__get__(fvg_pattern, CustomPattern)
    action_fvg = Action(name='fvg_1m', pattern=fvg_pattern)

    # 3. Define Hammer pattern triggers (5m, but triggers at different times)
    times_hammer = ['2024-03-07 09:12:00']
    hammer_pattern = CustomPattern('hammer', [TimeRange(5, 'm')], [])
    hammer_pattern.detect = make_pattern_detector(times_hammer).__get__(hammer_pattern, CustomPattern)
    action_hammer = Action(name='hammer_5m', pattern=hammer_pattern)

    # 4. FVG-only strategy
    strat_fvg = PatternStrategy(
        name='fvg_only',
        actions=[action_fvg],
        combination_logic='OR',
    )
    # 5. FVG+Hammer (OR) strategy
    strat_or = PatternStrategy(
        name='fvg_or_hammer',
        actions=[action_fvg, action_hammer],
        combination_logic='OR',
    )
    # 6. Run backtests
    engine = MultiTimeframeBacktestEngine()
    res_fvg = engine.run_backtest(strat_fvg, df)
    res_or = engine.run_backtest(strat_or, df)
    # 7. Extract FVG zones from both
    def extract_fvg_zones(zones):
        return sorted([str(z['timestamp'])[:19] for z in zones if z.get('zone_type', '').lower() == 'fvg'])
    fvg_zones_fvg = extract_fvg_zones(res_fvg['zones'])
    fvg_zones_or = extract_fvg_zones(res_or['zones'])
    # 8. Compare
    missing = [z for z in fvg_zones_fvg if z not in fvg_zones_or]
    print('FVG-only zones:', fvg_zones_fvg)
    print('FVG+Hammer(OR) zones:', fvg_zones_or)
    if not missing:
        print('[PASS] All FVG zones are present in the OR strategy.')
    else:
        print('[FAIL] Missing FVG zones in OR strategy:', missing)

def test_real_nq_fvg_vs_fvg_or_hammer():
    # 1. Load the real dataset
    df = pd.read_csv('workspaces/datasets/NQ_5s_1m.csv', parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    # 2. Filter to 6/2/2024 (assuming US format MM/DD/YYYY)
    df = df[(df.index >= '2024-06-02 00:00:00') & (df.index < '2024-06-03 00:00:00')]
    print(f"Filtered to {len(df)} rows for 2024-06-02")
    # 3. Build FVG and Hammer patterns using the real detection logic
    fvg_pattern = CustomPattern('fvg', [TimeRange(1, 'm')], [])
    hammer_pattern = CustomPattern('hammer', [TimeRange(5, 'm')], [])
    action_fvg = Action(name='fvg_1m', pattern=fvg_pattern)
    action_hammer = Action(name='hammer_5m', pattern=hammer_pattern)
    # 4. FVG-only strategy
    strat_fvg = PatternStrategy(
        name='fvg_only',
        actions=[action_fvg],
        combination_logic='OR',
    )
    # 5. FVG+Hammer (OR) strategy
    strat_or = PatternStrategy(
        name='fvg_or_hammer',
        actions=[action_fvg, action_hammer],
        combination_logic='OR',
    )
    # 6. Run backtests
    engine = MultiTimeframeBacktestEngine()
    res_fvg = engine.run_backtest(strat_fvg, df)
    res_or = engine.run_backtest(strat_or, df)
    # 7. Extract FVG zones from both
    def extract_fvg_zones(zones):
        return sorted([(str(z['timestamp'])[:19], z.get('zone_min'), z.get('zone_max')) for z in zones if z.get('zone_type', '').lower() == 'fvg'])
    fvg_zones_fvg = extract_fvg_zones(res_fvg['zones'])
    fvg_zones_or = extract_fvg_zones(res_or['zones'])
    # 8. Compare
    missing = [z for z in fvg_zones_fvg if z not in fvg_zones_or]
    print('FVG-only zones:', fvg_zones_fvg)
    print('FVG+Hammer(OR) zones:', fvg_zones_or)
    if not missing:
        print('[PASS] All FVG zones are present in the OR strategy.')
    else:
        print('[FAIL] Missing FVG zones in OR strategy:', missing)

def generate_custom_fvg_test_dataset(filename='custom_fvg_test_dataset.csv'):
    """Generate a small dataset with known FVGs for robust testing."""
    # Create 10 bars, 1-minute interval
    base_time = datetime(2024, 3, 7, 9, 0, 0)
    rows = []
    for i in range(10):
        dt = base_time + timedelta(minutes=i)
        # Default: no FVG
        open_ = 100 + i
        high = 101 + i
        low = 99 + i
        close = 100 + i
        volume = 1000
        # Insert a bullish FVG at index 4 (bars 2,3,4)
        if i == 2:
            high = 105
        if i == 4:
            low = 108
        # Insert a bearish FVG at index 7 (bars 5,6,7)
        if i == 5:
            low = 97
        if i == 7:
            high = 96
        rows.append([dt.strftime('%Y-%m-%d %H:%M:%S'), open_, high, low, close, volume])
    df = pd.DataFrame(rows, columns=['datetime','open','high','low','close','volume'])
    df.to_csv(filename, index=False)
    print(f"Custom FVG test dataset written to {filename}")

def test_fvg_location_gate_vs_fvg_location_gate_and_hammer_pattern_real_nq():
    """Test that FVG zones are identical between FVG-only (1m FVG pattern) and FVG+Hammer (1m FVG + 5m Hammer pattern) strategies on real NQ data for 6/2/2024, using the exact same logic as the GUI."""
    df = pd.read_csv('workspaces/datasets/NQ_5s_1m.csv', parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    # Splice to 6/2/2024 only
    df = df[(df.index >= '2024-06-02 00:00:00') & (df.index < '2024-06-03 00:00:00')]
    print(f"Filtered to {len(df)} rows for 2024-06-02")
    from strategies.strategy_builders import PatternStrategy, Action, MultiTimeframeBacktestEngine
    from patterns.candlestick_patterns import CustomPattern, TimeRange

    # --- PATCH: Mirror GUI logic for FVG and Hammer actions ---
    # FVG as 1m pattern (same as GUI)
    fvg_pattern = CustomPattern('fvg', [TimeRange(1, 'm')], [])
    action_fvg = Action(name='fvg_1m', pattern=fvg_pattern)
    # Hammer as 5m pattern (same as GUI)
    hammer_pattern = CustomPattern('hammer', [TimeRange(5, 'm')], [])
    action_hammer = Action(name='hammer_5m', pattern=hammer_pattern)
    # FVG-only strategy (same as GUI)
    strat_fvg = PatternStrategy(
        name='fvg_only',
        actions=[action_fvg],
        combination_logic='OR',
    )
    # FVG+Hammer strategy (same as GUI)
    strat_fvg1mh5m = PatternStrategy(
        name='fvg1mh5m',
        actions=[action_fvg, action_hammer],
        combination_logic='OR',
    )
    engine = MultiTimeframeBacktestEngine()
    res_fvg = engine.run_backtest(strat_fvg, df)
    res_fvg1mh5m = engine.run_backtest(strat_fvg1mh5m, df)
    def extract_fvg_zones(zones):
        return sorted([
            (str(z.get('timestamp'))[:19], z.get('zone_min'), z.get('zone_max'))
            for z in zones if z.get('zone_type','').lower() == 'fvg'])
    fvg_zones_fvg = extract_fvg_zones(res_fvg['zones'])
    fvg_zones_fvg1mh5m = extract_fvg_zones(res_fvg1mh5m['zones'])
    print('\nFVG-only (1m pattern) zones:')
    for z in fvg_zones_fvg:
        print(z)
    print('\nFVG+Hammer (1m FVG + 5m Hammer) zones:')
    for z in fvg_zones_fvg1mh5m:
        print(z)
    if fvg_zones_fvg == fvg_zones_fvg1mh5m:
        print('[PASS] FVG zones are identical between FVG-only and FVG+Hammer strategies.')
    else:
        print('[FAIL] FVG zones differ!')
        print('Missing in FVG+Hammer:', [z for z in fvg_zones_fvg if z not in fvg_zones_fvg1mh5m])
        print('Extra in FVG+Hammer:', [z for z in fvg_zones_fvg1mh5m if z not in fvg_zones_fvg])
    assert fvg_zones_fvg == fvg_zones_fvg1mh5m, 'FVG zones must be identical!'

def debug_print_fvg_zones_on_real_nq():
    """Directly print all FVG zones detected by the location gate logic on the real NQ dataset for 6/2/2024."""
    df = pd.read_csv('workspaces/datasets/NQ_5s_1m.csv', parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[(df.index >= '2024-06-02 00:00:00') & (df.index < '2024-06-03 00:00:00')]
    print(f"[DEBUG] NQ 6/2/2024 rows: {len(df)}")
    from strategies.strategy_builders import PatternStrategy
    fvg_location_gate_params = {
        'fvg_epsilon': 0.0,
        'fvg_N': 3,
        'fvg_sigma': 0.1,
        'fvg_beta1': 0.7,
        'fvg_beta2': 0.3,
        'fvg_phi': 0.2,
        'fvg_lambda': 0.0,
        'fvg_gamma': 0.95,
        'fvg_tau_bars': 15,
        'fvg_drop_threshold': 0.01,
    }
    strat = PatternStrategy(
        name='fvg_debug',
        actions=[],
        gates_and_logic={'location_gate': True},
        location_gate_params=fvg_location_gate_params.copy(),
    )
    detected_fvgs = []
    for i in range(len(df)):
        zones = strat._detect_fvg_zones(df, i)
        for z in zones:
            detected_fvgs.append({
                'timestamp': df.index[i],
                'zone_min': z['zone_min'],
                'zone_max': z['zone_max'],
                'direction': z['direction']
            })
    print(f"[DEBUG] Detected {len(detected_fvgs)} FVG zones:")
    for z in detected_fvgs:
        print(f"  {z['timestamp']} | {z['direction']} | {z['zone_min']:.2f} - {z['zone_max']:.2f}")
    if not detected_fvgs:
        print('[DEBUG] No FVG zones detected by direct logic!')

if __name__ == "__main__":
    test_fvg_location_gate_vs_fvg_location_gate_and_hammer_pattern_real_nq() 