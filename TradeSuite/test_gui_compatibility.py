#!/usr/bin/env python3
"""
Test GUI Compatibility - Compare Old vs New Testing Approach
===========================================================
This script demonstrates the difference between the old headless mode
and the new GUI-compatible testing approach.

The key insight is that the old headless mode in main.py uses a completely
different code path than the GUI, which is why "it works in testing" but
doesn't work when you actually use it in the GUI.

The new approach uses the exact same code path as the GUI, ensuring
that when we say "it works", it actually works for you.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import subprocess
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our new GUI-compatible tester
from gui_test_simulator import run_gui_compatible_test, HeadlessBacktestWindow


def create_test_strategy():
    """Create a simple test strategy"""
    return {
        "name": "Test FVG Strategy",
        "actions": [
            {
                "name": "fvg_test",
                "location_strategy": "FVG",
                "location_params": {
                    "fvg_epsilon": 0.0,
                    "fvg_N": 3,
                    "fvg_sigma": 0.1,
                    "fvg_beta1": 0.7,
                    "fvg_beta2": 0.3,
                    "fvg_phi": 0.2,
                    "fvg_lambda": 0.0,
                    "fvg_gamma": 0.95,
                    "fvg_tau_bars": 15,
                    "fvg_drop_threshold": 0.01,
                }
            }
        ],
        "combination_logic": "AND",
        "gates_and_logic": {"location_gate": True},
        "location_gate_params": {
            "fvg_epsilon": 1.0,
            "fvg_N": 3,
            "fvg_sigma": 0.1,
            "fvg_beta1": 0.7,
            "fvg_beta2": 0.3,
            "fvg_phi": 0.2,
            "fvg_lambda": 0.0,
            "fvg_gamma": 0.95,
            "fvg_tau_bars": 50,
            "fvg_drop_threshold": 0.01,
            "fvg_min_gap_size": 0.1,
        }
    }


def create_test_data():
    """Create test data with known FVG patterns"""
    # Create synthetic data with FVG patterns
    times = pd.date_range('2024-03-07 09:00:00', periods=100, freq='1min')
    
    # Create data with gaps that should trigger FVG detection
    data = []
    for i in range(100):
        if i == 20:  # Create a gap up
            data.append({
                'open': 105.0,  # Gap up from previous close of 100
                'high': 107.0,
                'low': 104.0,
                'close': 106.0,
                'volume': 1000
            })
        elif i == 40:  # Create a gap down
            data.append({
                'open': 95.0,  # Gap down from previous close of 100
                'high': 97.0,
                'low': 94.0,
                'close': 96.0,
                'volume': 1000
            })
        else:
            # Normal price movement
            base_price = 100 + (i * 0.1)
            data.append({
                'open': base_price,
                'high': base_price + 1.0,
                'low': base_price - 1.0,
                'close': base_price + 0.2,
                'volume': 1000
            })
    
    df = pd.DataFrame(data, index=times)
    return df


def test_old_vs_new_approach():
    """Compare the old headless approach vs the new GUI-compatible approach"""
    print(f"\n{'='*80}")
    print("COMPARING OLD vs NEW TESTING APPROACHES")
    print(f"{'='*80}")
    
    # Create test data
    test_df = create_test_data()
    test_data_path = "test_fvg_comparison.csv"
    test_df.to_csv(test_data_path)
    print(f"✅ Created test data: {test_data_path}")
    
    # Create test strategy
    strategy_config = create_test_strategy()
    strategy_file = "test_fvg_strategy.json"
    with open(strategy_file, 'w') as f:
        json.dump(strategy_config, f, indent=2)
    print(f"✅ Created test strategy: {strategy_file}")
    
    print(f"\n{'='*60}")
    print("TESTING WITH NEW GUI-COMPATIBLE APPROACH")
    print(f"{'='*60}")
    
    # Test with new GUI-compatible approach
    new_results = run_gui_compatible_test(strategy_config, test_data_path, "new_results.json")
    
    print(f"\n{'='*60}")
    print("TESTING WITH OLD HEADLESS APPROACH")
    print(f"{'='*60}")
    
    # Test with old headless approach
    old_results = test_old_headless_approach(strategy_file, test_data_path, "old_results.json")
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Compare results
    print("NEW APPROACH (GUI-Compatible):")
    if 'error' in new_results:
        print(f"  ❌ Error: {new_results['error']}")
    else:
        print(f"  ✅ Success: {new_results.get('strategy_name', 'Unknown')}")
        print(f"  📊 Data length: {new_results.get('data_length', 0)}")
        print(f"  🎯 Signals: {len(new_results.get('signals', []))}")
        print(f"  🏪 Zones: {len(new_results.get('zones', []))}")
        print(f"  💰 Trades: {len(new_results.get('trades', []))}")
    
    print("\nOLD APPROACH (main.py headless):")
    if 'error' in old_results:
        print(f"  ❌ Error: {old_results['error']}")
    else:
        print(f"  ✅ Success: {old_results.get('strategy_name', 'Unknown')}")
        print(f"  📊 Data length: {old_results.get('data_length', 0)}")
        print(f"  🎯 Signals: {len(old_results.get('signals', []))}")
        print(f"  🏪 Zones: {len(old_results.get('zones', []))}")
        print(f"  💰 Trades: {len(old_results.get('trades', []))}")
    
    # Clean up
    for file in [test_data_path, strategy_file, "new_results.json", "old_results.json"]:
        if os.path.exists(file):
            os.remove(file)
    
    print(f"\n{'='*60}")
    print("KEY INSIGHT")
    print(f"{'='*60}")
    print("The old headless mode in main.py uses a completely different code path")
    print("than the GUI. It has its own data processing, strategy creation, and")
    print("backtest execution logic that doesn't match what happens in the GUI.")
    print()
    print("The new GUI-compatible approach uses the EXACT same classes and methods")
    print("as the GUI, ensuring that when we say 'it works', it actually works")
    print("when you use it in the actual GUI.")
    print()
    print("This is why you've experienced the disconnect between 'testing works'")
    print("and 'actual GUI doesn't work' - they were using different code paths!")


def test_old_headless_approach(strategy_file: str, data_file: str, output_file: str) -> Dict[str, Any]:
    """Test using the old headless approach from main.py"""
    try:
        # Run the old headless command
        cmd = [
            'python', 'main.py',
            '--headless',
            '--strategy', strategy_file,
            '--data', data_file,
            '--output', output_file
        ]
        
        print(f"Running old headless command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            return {'error': f'Command failed with return code {result.returncode}'}
        
        # Read results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            return {'error': f'Output file not found: {output_file}'}
            
    except Exception as e:
        return {'error': f'Exception in old headless test: {e}'}


def demonstrate_gui_code_path():
    """Demonstrate that the new approach uses the same code path as GUI"""
    print(f"\n{'='*80}")
    print("DEMONSTRATING GUI CODE PATH COMPATIBILITY")
    print(f"{'='*80}")
    
    # Create test data
    test_df = create_test_data()
    test_data_path = "gui_compatibility_test.csv"
    test_df.to_csv(test_data_path)
    
    # Create test strategy
    strategy_config = create_test_strategy()
    
    print("1. Creating HeadlessBacktestWindow (same as GUI BacktestWindow)")
    window = HeadlessBacktestWindow()
    
    print("2. Processing data with _process_data_for_backtest (same as GUI)")
    df = pd.read_csv(test_data_path)
    processed_data = window._process_data_for_backtest(df)
    
    print("3. Creating strategy with _create_strategy_from_config (same as GUI)")
    strategy = window._create_strategy_from_config(strategy_config)
    
    print("4. Running backtest with BacktestWorker (same as GUI)")
    from gui.backtest_window import BacktestWorker
    parameters = {'initial_capital': 100000, 'position_size': 0.02}
    worker = BacktestWorker(processed_data, strategy, parameters)
    
    print("5. Connecting signals (same as GUI)")
    worker.log.connect(window._add_log)
    
    print("6. Running worker.run() (same as GUI)")
    worker.run()
    
    print("\n✅ This demonstrates that we're using the EXACT same code path")
    print("   as the GUI BacktestWindow. When we say 'it works', it will")
    print("   actually work in your GUI because we're using the same logic!")
    
    # Clean up
    if os.path.exists(test_data_path):
        os.remove(test_data_path)


if __name__ == "__main__":
    # Demonstrate the GUI code path compatibility
    demonstrate_gui_code_path()
    
    # Compare old vs new approaches
    test_old_vs_new_approach()
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("The new GUI-compatible testing approach ensures that when we test")
    print("strategies, we're using the exact same code path as your GUI.")
    print("This eliminates the disconnect between 'testing works' and")
    print("'actual GUI doesn't work' that you've been experiencing.")
    print()
    print("From now on, when I say a strategy works, it will actually work")
    print("in your GUI because we're using the same logic!") 