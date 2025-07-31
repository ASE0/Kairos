#!/usr/bin/env python3
"""
Isolated test for backtest save/load functionality - minimal imports
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what we need
import os
import json
import pandas as pd
from datetime import datetime

class StrategyManager:
    """Handles saving, loading, and management of strategies and backtest results."""
    def __init__(self, workspace_path='workspaces'):
        self.base_path = workspace_path
        self.strategies_path = os.path.join(self.base_path, 'strategies')
        self.patterns_path = os.path.join(self.base_path, 'patterns')
        self.results_path = os.path.join(self.base_path, 'results')
        self._create_dirs()

    def _create_dirs(self):
        """Create necessary workspace directories."""
        os.makedirs(self.strategies_path, exist_ok=True)
        os.makedirs(self.patterns_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

    def save_backtest_results(self, results):
        """Save backtest results, linked to a strategy."""
        strategy_name = results.get('strategy_name')
        timeframe = results.get('timeframe', 'UnknownTF')
        if not strategy_name:
            print("Cannot save results without a strategy name.")
            return
        # Sanitize for filesystem
        safe_strategy = str(strategy_name).replace(' ', '_').replace('/', '_')
        safe_timeframe = str(timeframe).replace(' ', '_').replace('/', '_')
        strategy_folder_name = safe_strategy
        results_dir = os.path.join(self.results_path, strategy_folder_name)
        os.makedirs(results_dir, exist_ok=True)
        result_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{safe_strategy}_{safe_timeframe}_{result_id}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Handle non-serializable objects like pandas DataFrames
        serializable_results = results.copy()
        
        # Convert DataFrames to JSON format for proper serialization
        if 'data' in serializable_results and isinstance(serializable_results['data'], pd.DataFrame):
            serializable_results['data'] = serializable_results['data'].to_json(orient='split')
        if 'dataset_data' in serializable_results and isinstance(serializable_results['dataset_data'], pd.DataFrame):
            serializable_results['dataset_data'] = serializable_results['dataset_data'].to_json(orient='split')
        if 'multi_tf_data' in serializable_results and isinstance(serializable_results['multi_tf_data'], dict):
            # Handle multi-timeframe data dictionary
            mtf_data = serializable_results['multi_tf_data']
            for tf_key, tf_data in mtf_data.items():
                if isinstance(tf_data, pd.DataFrame):
                    mtf_data[tf_key] = tf_data.to_json(orient='split')
            serializable_results['multi_tf_data'] = mtf_data
        if 'equity_curve' in serializable_results and isinstance(serializable_results['equity_curve'], pd.Series):
            serializable_results['equity_curve'] = serializable_results['equity_curve'].to_json()
        if 'trades' in serializable_results and isinstance(serializable_results['trades'], pd.DataFrame):
             serializable_results['trades'] = serializable_results['trades'].to_json(orient='split')

        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Results for '{strategy_name}' saved to {filepath}")
        except Exception as e:
            print(f"Error saving results for {strategy_name}: {e}")

    def load_all_results(self):
        """Load all backtest results from the workspace."""
        all_results = {}
        if not os.path.exists(self.results_path):
            return all_results
            
        for strategy_folder in os.listdir(self.results_path):
            strategy_results_path = os.path.join(self.results_path, strategy_folder)
            if os.path.isdir(strategy_results_path):
                for filename in os.listdir(strategy_results_path):
                    if filename.endswith('.json'):
                        filepath = os.path.join(strategy_results_path, filename)
                        try:
                            with open(filepath, 'r') as f:
                                result = json.load(f)
                            result_id = f"{strategy_folder}_{filename.replace('.json', '')}"
                            
                            # Deserialize pandas objects
                            if 'data' in result and isinstance(result['data'], str):
                                result['data'] = pd.read_json(result['data'], orient='split')
                            if 'dataset_data' in result and isinstance(result['dataset_data'], str):
                                result['dataset_data'] = pd.read_json(result['dataset_data'], orient='split')
                            if 'multi_tf_data' in result and isinstance(result['multi_tf_data'], dict):
                                # Handle multi-timeframe data dictionary
                                mtf_data = result['multi_tf_data']
                                for tf_key, tf_data in mtf_data.items():
                                    if isinstance(tf_data, str):
                                        mtf_data[tf_key] = pd.read_json(tf_data, orient='split')
                                result['multi_tf_data'] = mtf_data
                            if 'equity_curve' in result and isinstance(result['equity_curve'], str):
                                result['equity_curve'] = pd.read_json(result['equity_curve'], typ='series')
                            if 'trades' in result and isinstance(result['trades'], str):
                                result['trades'] = pd.read_json(result['trades'], orient='split')

                            all_results[result_id] = result
                        except Exception as e:
                            print(f"Error loading result from {filename}: {e}")
        return all_results

def create_test_results():
    """Create test backtest results"""
    print("[TEST] Creating test backtest results...")
    
    # Create time index (1 day of 1-minute data)
    start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(days=1)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create OHLC data
    np.random.seed(42)
    base_price = 100.0
    prices = []
    current_price = base_price
    
    for i in range(len(time_index)):
        change = np.random.normal(0, 0.1) + 0.001
        current_price += change
        
        high = current_price + abs(np.random.normal(0, 0.05))
        low = current_price - abs(np.random.normal(0, 0.05))
        close = current_price + np.random.normal(0, 0.02)
        
        prices.append({
            'open': current_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 5000)
        })
        current_price = close
    
    data = pd.DataFrame(prices, index=time_index)
    
    # Create test trades
    trades = [
        {
            'entry_time': time_index[10],
            'exit_time': time_index[15],
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'direction': 'long'
        },
        {
            'entry_time': time_index[30],
            'exit_time': time_index[35],
            'entry_price': 101.0,
            'exit_price': 100.8,
            'pnl': -0.2,
            'direction': 'short'
        }
    ]
    
    # Create test results
    results = {
        'strategy_name': 'Test SMA Strategy',
        'timeframe': '1min',
        'result_display_name': f'Test_SMA_Strategy_1min_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'initial_capital': 100000,
        'final_capital': 100700,
        'cumulative_pnl': 700,
        'total_return': 0.007,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.02,
        'win_rate': 0.5,
        'profit_factor': 1.8,
        'total_trades': 2,
        'equity_curve': [100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700],
        'trades': trades,
        'data': data,
        'action_details': {
            'SMA': pd.Series([False] * len(data), index=data.index)
        }
    }
    
    print(f"[TEST] Created test results with {len(trades)} trades")
    return results

def test_isolated_save_load():
    """Test that backtest results are saved to disk and can be loaded"""
    print("[TEST] Testing Isolated Save/Load Functionality...")
    
    # Create strategy manager
    strategy_manager = StrategyManager()
    
    # Create test results
    test_results = create_test_results()
    
    # Save results to disk
    print("[TEST] Saving results to disk...")
    strategy_manager.save_backtest_results(test_results)
    
    # Reload results from disk
    print("[TEST] Reloading results from disk...")
    loaded_results = strategy_manager.load_all_results()
    
    # Check if results were loaded
    result_count = len(loaded_results)
    print(f"[TEST] Results loaded from disk: {result_count}")
    
    if result_count > 0:
        print("[TEST] ✅ SUCCESS: Results saved and loaded successfully!")
        
        # Check the first result
        first_result_id = list(loaded_results.keys())[0]
        first_result = loaded_results[first_result_id]
        
        print(f"[TEST] First result strategy: {first_result.get('strategy_name')}")
        print(f"[TEST] First result trades: {first_result.get('total_trades')}")
        print(f"[TEST] First result data shape: {first_result.get('data').shape if 'data' in first_result else 'No data'}")
        
        return True
    else:
        print("[TEST] ❌ FAILED: No results loaded from disk")
        return False

if __name__ == "__main__":
    success = test_isolated_save_load()
    
    if success:
        print("\n[TEST] ✅ SUCCESS: Backtest save/load functionality works!")
        sys.exit(0)
    else:
        print("\n[TEST] ❌ FAILED: Backtest save/load functionality failed!")
        sys.exit(1) 