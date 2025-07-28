"""
core/workspace_manager.py
=========================
Manages workspace saving, loading, and organization.
"""

import os
import json
import dill
from typing import Dict, Any
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

    def save_strategy(self, strategy: Any):
        """Serialize and save a strategy object."""
        strategy_name = strategy.name.replace(' ', '_')
        filepath = os.path.join(self.strategies_path, f"{strategy_name}.dill")
        try:
            with open(filepath, 'wb') as f:
                dill.dump(strategy, f)
            print(f"Strategy '{strategy.name}' saved to {filepath}")
        except Exception as e:
            print(f"Error saving strategy {strategy.name}: {e}")

    def load_strategies(self) -> Dict[str, Any]:
        """Load all strategies from the workspace."""
        strategies = {}
        if not os.path.exists(self.strategies_path):
            return strategies
        for filename in os.listdir(self.strategies_path):
            if filename.endswith('.dill'):
                filepath = os.path.join(self.strategies_path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        strategy = dill.load(f)
                    # Only set fallback name if missing or blank, never overwrite a valid name
                    if not hasattr(strategy, 'name') or not strategy.name or not strategy.name.strip():
                        strategy.name = filename.replace('.dill', '').replace('_', ' ')
                    if hasattr(strategy, 'id'):
                        strategies[strategy.id] = strategy
                    else:
                        print(f"Warning: Loaded object {filename} is not a valid strategy.")
                except Exception as e:
                    print(f"Error loading strategy from {filename}: {e}")
        return strategies

    def save_pattern(self, pattern: Any):
        """Serialize and save a pattern object."""
        pattern_name = pattern.name.replace(' ', '_')
        filepath = os.path.join(self.patterns_path, f"{pattern_name}.dill")
        try:
            with open(filepath, 'wb') as f:
                dill.dump(pattern, f)
            print(f"Pattern '{pattern.name}' saved to {filepath}")
        except Exception as e:
            print(f"Error saving pattern {pattern.name}: {e}")

    def load_patterns(self) -> Dict[str, Any]:
        """Load all patterns from the workspace."""
        patterns = {}
        if not os.path.exists(self.patterns_path):
            return patterns
            
        for filename in os.listdir(self.patterns_path):
            if filename.endswith('.dill'):
                filepath = os.path.join(self.patterns_path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        pattern = dill.load(f)
                    # Assuming custom patterns have a unique name that can be used as a key
                    if hasattr(pattern, 'name'):
                        patterns[pattern.name] = pattern
                    else:
                        print(f"Warning: Loaded object {filename} is not a valid pattern.")
                except Exception as e:
                    print(f"Error loading pattern from {filename}: {e}")
        return patterns

    def delete_pattern(self, pattern_name: str) -> bool:
        """Delete a pattern file."""
        filename = pattern_name.replace(' ', '_')
        filepath = os.path.join(self.patterns_path, f"{filename}.dill")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted pattern: {pattern_name}")
                return True
            except Exception as e:
                print(f"Error deleting pattern file {filepath}: {e}")
                return False
        return False

    def delete_strategy(self, strategy_name: str) -> bool:
        """Delete a strategy and its associated results."""
        filename = strategy_name.replace(' ', '_')
        strategy_filepath = os.path.join(self.strategies_path, f"{filename}.dill")
        
        # Delete strategy file
        if os.path.exists(strategy_filepath):
            try:
                os.remove(strategy_filepath)
                print(f"Deleted strategy: {strategy_name}")
            except Exception as e:
                print(f"Error deleting strategy file {strategy_filepath}: {e}")
                return False
        
        # Delete associated results
        results_dir = os.path.join(self.results_path, filename)
        if os.path.exists(results_dir):
            try:
                for result_file in os.listdir(results_dir):
                    os.remove(os.path.join(results_dir, result_file))
                os.rmdir(results_dir)
                print(f"Deleted associated results for: {strategy_name}")
            except Exception as e:
                print(f"Error deleting results directory {results_dir}: {e}")

        return True

    def save_backtest_results(self, results: Dict[str, Any]):
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

    def load_all_results(self) -> Dict[str, Any]:
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