#!/usr/bin/env python3
"""
Debug script to see what building blocks are actually in your real strategy
"""

import sys
import os
import json
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_most_recent_result():
    """Load your most recent strategy result"""
    
    # Look for result files
    results_dirs = ["workspaces/results", "../workspaces/results"]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            result_files = []
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith('.json') and 'result_' in file:
                        result_files.append(os.path.join(root, file))
            
            if result_files:
                # Sort by modification time (most recent first)
                result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # Try to load the most recent ones
                for result_file in result_files[:5]:
                    try:
                        print(f"Trying: {result_file}")
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        print(f"✅ Successfully loaded: {result_file}")
                        return result_data
                        
                    except Exception as e:
                        print(f"❌ Failed to load {result_file}: {e}")
                        continue
    
    print("❌ No valid result files found")
    return None

def analyze_building_blocks(result_data):
    """Analyze what building blocks are actually in your strategy"""
    
    print("\n=== ANALYZING YOUR REAL STRATEGY ===")
    print(f"Strategy name: {result_data.get('strategy_name', 'Unknown')}")
    print(f"Total trades: {result_data.get('total_trades', 0)}")
    
    # Check action_details
    action_details = result_data.get('action_details', {})
    print(f"\n📊 ACTION_DETAILS: {list(action_details.keys())}")
    
    for name, data in action_details.items():
        if isinstance(data, str):
            # Count signals in string representation
            signal_count = data.count('True')
            print(f"  - {name}: {signal_count} signals (string format)")
        elif hasattr(data, 'sum'):
            # Pandas series
            signal_count = data.sum()
            print(f"  - {name}: {signal_count} signals (series format)")
        else:
            print(f"  - {name}: {type(data)} (unknown format)")
    
    # Check strategy_params
    strategy_params = result_data.get('strategy_params', {})
    print(f"\n⚙️ STRATEGY_PARAMS: {len(strategy_params)} parameters")
    
    # Look for enabled components
    enabled_components = []
    for key, value in strategy_params.items():
        if key.endswith('_enabled') and value:
            component = key.replace('_enabled', '')
            enabled_components.append(component)
    
    if enabled_components:
        print(f"  Enabled components: {enabled_components}")
    
    # Check filters
    filters = strategy_params.get('filters', [])
    if filters:
        print(f"  Filters: {filters}")
    
    # Check location strategies
    location_strategies = strategy_params.get('location_strategies', [])
    if location_strategies:
        print(f"  Location strategies: {location_strategies}")
    
    # Check gates
    gates_enabled = result_data.get('gates_enabled', {})
    active_gates = [name for name, enabled in gates_enabled.items() if enabled]
    if active_gates:
        print(f"  Active gates: {active_gates}")
    
    # Check trades
    trades = result_data.get('trades', [])
    if trades:
        print(f"\n📈 TRADES: {len(trades)} trades found")
        if len(trades) > 0:
            first_trade = trades[0]
            print(f"  First trade keys: {list(first_trade.keys())}")
    
    return action_details

if __name__ == "__main__":
    result_data = load_most_recent_result()
    if result_data:
        action_details = analyze_building_blocks(result_data)
        
        print("\n=== SUMMARY ===")
        print("For the heatmap to work, we need:")
        print("1. Building blocks with actual signal data")
        print("2. Each building block should show when IT triggers (not when trades happen)")
        print("3. Multiple building blocks if your strategy uses multiple components")
        
        if action_details:
            print(f"\nYour strategy has {len(action_details)} building blocks with signal data")
        else:
            print("\n❌ No action_details found - this is the problem!")