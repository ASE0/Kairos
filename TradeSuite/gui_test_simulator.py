#!/usr/bin/env python3
"""
GUI Test Simulator - Run GUI Logic Headlessly
=============================================
This module simulates the exact same code path as the GUI backtest window
but runs in a headless environment. This ensures that when we test strategies,
we're using the same logic that you use in the actual GUI.

Key differences from the old headless mode:
1. Uses the same BacktestWorker class as the GUI
2. Uses the same data processing logic as the GUI
3. Uses the same strategy loading logic as the GUI
4. Uses the same MultiTimeframeBacktestEngine as the GUI
5. Processes results the same way as the GUI

This ensures that when we say "it works", it actually works in your GUI.
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

# Import the same classes used by the GUI
from gui.backtest_window import BacktestWorker
from strategies.strategy_builders import MultiTimeframeBacktestEngine, PatternStrategy, Action
from core.data_structures import BaseStrategy


class HeadlessBacktestWindow:
    """Headless version of BacktestWindow that uses the same code path"""
    
    def __init__(self):
        self.log_messages = []
        self.current_results = None
        self.engine = MultiTimeframeBacktestEngine()
        
    def _add_log(self, message: str):
        """Add message to log (headless version)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        print(log_entry)
    
    def _process_data_for_backtest(self, data):
        """Process data for backtest with proper datetime handling and filtering
        This is the EXACT same logic as the GUI BacktestWindow._process_data_for_backtest method"""
        try:
            # --- 1. Convert to DatetimeIndex if needed ---
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
            
            self._add_log(f"[DEBUG] Index type: {type(data.index)}, unique: {data.index.is_unique}, sample: {list(data.index[:5])}")
            
            # --- PATCH: Check for required columns ---
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self._add_log(f"❌ ERROR: Dataset is missing required columns: {missing_cols}")
                self._add_log(f"Columns found: {list(data.columns)}")
                return None
            
            # --- 2. Filter to selected date range (use full range for testing) ---
            if isinstance(data.index, pd.DatetimeIndex):
                actual_start = data.index.min()
                actual_end = data.index.max()
                self._add_log(f"Dataset date range: {actual_start} to {actual_end}")
                
                # For testing, use the full dataset range
                start_dt = actual_start
                end_dt = actual_end + pd.Timedelta(days=1)
                
                before_filter = len(data)
                filtered_data = data[(data.index >= start_dt) & (data.index < end_dt)]
                self._add_log(f"[DEBUG] Filtered to date range: {start_dt} to {end_dt} ({before_filter} -> {len(filtered_data)} bars)")
                
                # PATCH: Show warning if filter is empty
                if len(filtered_data) == 0:
                    self._add_log(f"❌ ERROR: No data available for the selected date range: {start_dt} to {end_dt}")
                    return None
                
                data = filtered_data
            else:
                self._add_log("WARNING: Data does not have a DatetimeIndex after conversion. Cannot filter by date range.")
            
            # --- 3. Multi-timeframe engine will handle resampling internally ---
            self._add_log("✅ Using Multi-Timeframe Backtest Engine - strategy timeframes will be preserved")
            self._add_log(f"Original data: {len(data)} bars, timeframe: {data.index.freq if hasattr(data.index, 'freq') else 'Unknown'}")
            
            # --- 4. Drop rows with missing OHLCV data ---
            data = data.dropna()
            self._add_log(f"Final data for backtest: {len(data)} bars, columns: {list(data.columns)}")
            
            if len(data) == 0:
                self._add_log("❌ ERROR: No data available after dropping missing values.")
                return None
            
            return data
            
        except Exception as e:
            self._add_log(f"❌ Error processing data: {e}")
            return None
    
    def _create_strategy_from_config(self, strategy_config: Dict[str, Any]) -> PatternStrategy:
        """Create a strategy object from configuration (same logic as GUI)"""
        try:
            strategy_name = strategy_config.get('name', 'TestStrategy')
            actions = strategy_config.get('actions', [])
            
            # Convert actions to Action objects
            action_objs = []
            for action_config in actions:
                action = Action(
                    name=action_config.get('name', ''),
                    pattern=None,  # Will be set by location_strategy
                    time_range=action_config.get('time_range'),
                    location_strategy=action_config.get('location_strategy'),
                    location_params=action_config.get('location_params', {}),
                    filters=action_config.get('filters', [])
                )
                action_objs.append(action)
            
            # Create strategy
            strategy = PatternStrategy(
                name=strategy_name,
                actions=action_objs,
                combination_logic=strategy_config.get('combination_logic', 'AND'),
                weights=strategy_config.get('weights'),
                min_actions_required=strategy_config.get('min_actions_required', 1),
                gates_and_logic=strategy_config.get('gates_and_logic', {'location_gate': True}),
                location_gate_params=strategy_config.get('location_gate_params', {})
            )
            
            self._add_log(f"✅ Created strategy: {strategy_name} with {len(action_objs)} actions")
            return strategy
            
        except Exception as e:
            self._add_log(f"❌ Error creating strategy: {e}")
            return None
    
    def run_backtest(self, strategy_config: Dict[str, Any], data_path: str, 
                    initial_capital: float = 100000, risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """Run backtest using the same code path as the GUI"""
        try:
            self._add_log("🚀 Starting GUI-compatible backtest...")
            
            # 1. Load and process data (same as GUI)
            self._add_log(f"📊 Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            
            # Process data using the same logic as GUI
            processed_data = self._process_data_for_backtest(df)
            if processed_data is None:
                return {'error': 'Failed to process data'}
            
            # 2. Create strategy (same as GUI)
            strategy = self._create_strategy_from_config(strategy_config)
            if strategy is None:
                return {'error': 'Failed to create strategy'}
            
            # 3. Create BacktestWorker (same as GUI)
            parameters = {
                'initial_capital': initial_capital,
                'position_size': risk_per_trade
            }
            
            worker = BacktestWorker(processed_data, strategy, parameters)
            
            # 4. Run backtest (same as GUI)
            self._add_log("🔄 Running backtest with MultiTimeframeBacktestEngine...")
            
            # Connect to worker signals for logging
            worker.log.connect(self._add_log)
            
            # Run the worker (this calls the same run() method as GUI)
            worker.run()
            
            # Get results
            if hasattr(worker, 'results'):
                results = worker.results
            else:
                # Fallback: run the engine directly
                results = self.engine.run_backtest(strategy, processed_data, 
                                                initial_capital=initial_capital,
                                                risk_per_trade=risk_per_trade)
            
            self._add_log("✅ Backtest completed successfully")
            self.current_results = results
            return results
            
        except Exception as e:
            error_msg = f"❌ Error in backtest: {e}"
            self._add_log(error_msg)
            import traceback
            self._add_log(f"Traceback: {traceback.format_exc()}")
            return {'error': error_msg}
    
    def get_test_summary(self):
        """Get a summary of the test results"""
        return {
            'log_messages': self.log_messages,
            'results': self.current_results,
            'success': len([msg for msg in self.log_messages if '❌ ERROR' in msg]) == 0
        }


def run_gui_compatible_test(strategy_config: Dict[str, Any], data_path: str, 
                          output_path: str = None) -> Dict[str, Any]:
    """
    Run a test using the exact same code path as the GUI.
    
    Args:
        strategy_config: Strategy configuration dictionary
        data_path: Path to the data CSV file
        output_path: Optional path to save results
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("GUI-COMPATIBLE STRATEGY TEST")
    print(f"{'='*60}")
    
    # Create headless backtest window
    window = HeadlessBacktestWindow()
    
    # Run backtest
    results = window.run_backtest(strategy_config, data_path)
    
    # Get summary
    summary = window.get_test_summary()
    
    # Save results if output path provided
    if output_path and results:
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✅ Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {summary['success']}")
    print(f"Log messages: {len(summary['log_messages'])}")
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"✅ Strategy: {results.get('strategy_name', 'Unknown')}")
        print(f"📊 Data length: {results.get('data_length', 0)}")
        print(f"🎯 Signals: {len(results.get('signals', []))}")
        print(f"🏪 Zones: {len(results.get('zones', []))}")
        print(f"💰 Trades: {len(results.get('trades', []))}")
    
    return results


def test_strategy_from_files(strategy_file: str, data_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Test a strategy using files (compatible with command line usage)
    
    Args:
        strategy_file: Path to strategy JSON file
        data_file: Path to data CSV file
        output_file: Optional path for output JSON
        
    Returns:
        Dictionary with test results
    """
    # Load strategy config
    with open(strategy_file, 'r') as f:
        strategy_config = json.load(f)
    
    # Run test
    return run_gui_compatible_test(strategy_config, data_file, output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GUI-Compatible Strategy Tester')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy configuration file')
    parser.add_argument('--data', type=str, required=True, help='Data file path')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    results = test_strategy_from_files(args.strategy, args.data, args.output)
    
    # Exit with appropriate code
    if 'error' in results:
        sys.exit(1)
    else:
        sys.exit(0) 