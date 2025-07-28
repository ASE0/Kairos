#!/usr/bin/env python3
"""
Test script to verify FVG detection and plotting in the backtester
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action
from gui.backtest_window import BacktestWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

def create_test_fvg_strategy():
    """Create a test strategy with only FVG location gate"""
    print("Creating test FVG strategy...")
    
    # Create a simple action with only FVG location gate
    action = Action(
        name="FVG Test",
        location_strategy="FVG"
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
    test_indices = [10, 15, 20, 25]
    
    for idx in test_indices:
        if idx >= len(data):
            continue
            
        print(f"\nTesting FVG detection at index {idx}...")
        
        # Call the FVG detection method directly
        fvg_zones = strategy._detect_fvg_zones(data, idx)
        
        print(f"Found {len(fvg_zones)} FVG zones at index {idx}")
        
        for i, zone in enumerate(fvg_zones):
            print(f"  FVG {i+1}: {zone['direction']} - min={zone['zone_min']:.2f}, max={zone['zone_max']:.2f}")
    
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

def test_backtester_gui(strategy, data):
    """Test the backtester GUI with the strategy"""
    print("\n=== Testing Backtester GUI ===")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create backtest window
    backtest_window = BacktestWindow()
    
    # Simulate loading the dataset
    backtest_window.datasets = {'NQ Test': {'data': data, 'metadata': {}}}
    backtest_window.strategies = {'pattern': {'test_strategy': strategy}}
    
    # Update the dropdowns
    backtest_window.dataset_combo.clear()
    backtest_window.dataset_combo.addItem("Select Dataset")
    backtest_window.dataset_combo.addItem("NQ Test")
    
    backtest_window.strategy_combo.clear()
    backtest_window.strategy_combo.addItem("Select Strategy")
    backtest_window.strategy_combo.addItem("[pattern] FVG Test Strategy")
    
    # Set the dataset and strategy
    backtest_window.dataset_combo.setCurrentIndex(1)
    backtest_window.strategy_combo.setCurrentIndex(1)
    
    print("Backtester GUI setup complete")
    print("You can now manually test the backtester or run it programmatically")
    
    # Show the window
    backtest_window.show()
    
    return app, backtest_window

def run_automated_backtest(strategy, data):
    """Run an automated backtest"""
    print("\n=== Running Automated Backtest ===")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create backtest window
    backtest_window = BacktestWindow()
    
    # Set up the data and strategy
    backtest_window.datasets = {'NQ Test': {'data': data, 'metadata': {}}}
    backtest_window.strategies = {'pattern': {'test_strategy': strategy}}
    
    # Run the backtest
    try:
        # Get the data
        test_data = data.copy()
        
        # Create a simple backtest result
        results = {
            'initial_capital': 100000,
            'final_capital': 101000,
            'cumulative_pnl': 1000,
            'total_return': 0.01,
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.02,
            'win_rate': 0.6,
            'profit_factor': 1.5,
            'total_trades': 1,
            'equity_curve': [100000, 101000],
            'trades': [{'entry_time': data.index[10], 'exit_time': data.index[15], 'entry_price': 100, 'exit_price': 101}],
            'zones': [],  # Will be populated by strategy
            'data': test_data,
            'strategy_name': 'FVG Test Strategy',
            'timeframe': '1min',
            'interval': '1min',
            'result_display_name': 'FVG Test Result'
        }
        
        # Get zones from strategy
        strategy.simple_zones = []  # Clear zones
        
        # Run strategy evaluation to populate zones
        signals, action_details = strategy.evaluate(test_data)
        
        # Use the zones from the strategy
        results['zones'] = strategy.simple_zones
        
        print(f"Strategy evaluation complete. Generated {len(strategy.simple_zones)} zones")
        
        # Update the chart
        backtest_window._update_chart_tab(results)
        
        print("Backtest results updated in chart")
        
        # Show the window
        backtest_window.show()
        
        return app, backtest_window, results
        
    except Exception as e:
        print(f"Error during automated backtest: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main test function"""
    print("=== FVG Backtester Test ===")
    
    # Load dataset
    data = load_nq_dataset()
    if data is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Create strategy
    strategy = create_test_fvg_strategy()
    
    # Test FVG detection directly
    test_fvg_detection_directly(strategy, data)
    
    # Run automated backtest
    app, backtest_window, results = run_automated_backtest(strategy, data)
    
    if app and backtest_window:
        print("\nBacktester window is open. Check the chart tab for FVG zones.")
        print("Press Ctrl+C to exit.")
        
        try:
            # Keep the app running
            app.exec()
        except KeyboardInterrupt:
            print("\nTest completed.")
    else:
        print("Failed to run automated backtest.")

if __name__ == "__main__":
    main() 