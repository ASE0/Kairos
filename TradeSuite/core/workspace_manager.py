"""
core/workspace_manager.py
=========================
Comprehensive workspace management for saving/loading all components
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

from patterns.candlestick_patterns import CandlestickPattern, CustomPattern, PatternFactory
from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action
from core.data_structures import TimeRange, OHLCRatio
from processors.data_processor import MultiTimeframeProcessor


class WorkspaceManager:
    """Manages saving and loading of complete trading workspace"""
    
    def __init__(self, workspace_dir: str = "workspaces"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.patterns_dir = self.workspace_dir / "patterns"
        self.strategies_dir = self.workspace_dir / "strategies"
        self.datasets_dir = self.workspace_dir / "datasets"
        self.configs_dir = self.workspace_dir / "configs"
        
        # Create subdirectories
        for dir_path in [self.patterns_dir, self.strategies_dir, self.datasets_dir, self.configs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Multi-timeframe processor for automatic timeframe creation
        self.mtf_processor = MultiTimeframeProcessor()
    
    def save_pattern(self, pattern: CandlestickPattern, name: str) -> bool:
        """Save a pattern to workspace"""
        try:
            pattern_data = self._pattern_to_dict(pattern)
            filepath = self.patterns_dir / f"{name}.json"
            
            with open(filepath, 'w') as f:
                json.dump(pattern_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving pattern: {e}")
            return False
    
    def load_pattern(self, name: str) -> Optional[CandlestickPattern]:
        """Load a pattern from workspace"""
        try:
            filepath = self.patterns_dir / f"{name}.json"
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                pattern_data = json.load(f)
            
            return self._dict_to_pattern(pattern_data)
        except Exception as e:
            print(f"Error loading pattern: {e}")
            return None
    
    def save_strategy(self, strategy: PatternStrategy, name: str) -> bool:
        """Save a strategy to workspace"""
        try:
            strategy_data = self._strategy_to_dict(strategy)
            filepath = self.strategies_dir / f"{name}.json"
            
            with open(filepath, 'w') as f:
                json.dump(strategy_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving strategy: {e}")
            return False
    
    def load_strategy(self, name: str) -> Optional[PatternStrategy]:
        """Load a strategy from workspace"""
        try:
            filepath = self.strategies_dir / f"{name}.json"
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                strategy_data = json.load(f)
            
            return self._dict_to_strategy(strategy_data)
        except Exception as e:
            print(f"Error loading strategy: {e}")
            return None
    
    def save_dataset(self, data: pd.DataFrame, name: str, metadata: Dict[str, Any] = None) -> bool:
        """Save a dataset to workspace"""
        try:
            # Save data as CSV
            data_filepath = self.datasets_dir / f"{name}.csv"
            data.to_csv(data_filepath)
            
            # Save metadata
            metadata = metadata or {}
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['rows'] = len(data)
            metadata['columns'] = list(data.columns)
            
            metadata_filepath = self.datasets_dir / f"{name}_metadata.json"
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving dataset: {e}")
            return False
    
    def load_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a dataset from workspace"""
        try:
            data_filepath = self.datasets_dir / f"{name}.csv"
            metadata_filepath = self.datasets_dir / f"{name}_metadata.json"
            
            if not data_filepath.exists():
                return None
            
            # Load data
            data = pd.read_csv(data_filepath, index_col=0, parse_dates=True)
            
            # Load metadata
            metadata = {}
            if metadata_filepath.exists():
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
            
            return {
                'data': data,
                'metadata': metadata
            }
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def save_workspace_config(self, config: Dict[str, Any], name: str) -> bool:
        """Save workspace configuration"""
        try:
            config['saved_at'] = datetime.now().isoformat()
            filepath = self.configs_dir / f"{name}.json"
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def load_workspace_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load workspace configuration"""
        try:
            filepath = self.configs_dir / f"{name}.json"
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
    
    def list_patterns(self) -> List[str]:
        """List all saved patterns"""
        return [f.stem for f in self.patterns_dir.glob("*.json")]
    
    def list_strategies(self) -> List[str]:
        """List all saved strategies"""
        return [f.stem for f in self.strategies_dir.glob("*.json")]
    
    def list_datasets(self) -> List[str]:
        """List all saved datasets"""
        return [f.stem for f in self.datasets_dir.glob("*.csv")]
    
    def list_configs(self) -> List[str]:
        """List all saved configurations"""
        return [f.stem for f in self.configs_dir.glob("*.json")]
    
    def delete_pattern(self, name: str) -> bool:
        """Delete a pattern"""
        try:
            filepath = self.patterns_dir / f"{name}.json"
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting pattern: {e}")
            return False
    
    def delete_strategy(self, name: str) -> bool:
        """Delete a strategy"""
        try:
            filepath = self.strategies_dir / f"{name}.json"
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting strategy: {e}")
            return False
    
    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset"""
        try:
            data_filepath = self.datasets_dir / f"{name}.csv"
            metadata_filepath = self.datasets_dir / f"{name}_metadata.json"
            
            deleted = False
            if data_filepath.exists():
                data_filepath.unlink()
                deleted = True
            
            if metadata_filepath.exists():
                metadata_filepath.unlink()
            
            return deleted
        except Exception as e:
            print(f"Error deleting dataset: {e}")
            return False
    
    def _pattern_to_dict(self, pattern: CandlestickPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization"""
        pattern_dict = {
            'name': pattern.name,
            'type': type(pattern).__name__,
            'timeframes': [{'value': tf.value, 'unit': tf.unit} for tf in pattern.timeframes],
            'required_bars': pattern.required_bars
        }
        
        if isinstance(pattern, CustomPattern):
            pattern_dict.update({
                'ohlc_ratios': [
                    {
                        'body_ratio': ratio.body_ratio,
                        'upper_wick_ratio': ratio.upper_wick_ratio,
                        'lower_wick_ratio': ratio.lower_wick_ratio
                    }
                    for ratio in pattern.ohlc_ratios
                ],
                'custom_formula': pattern.custom_formula,
                'advanced_features': pattern.advanced_features
            })
        else:
            # Add pattern-specific parameters
            if hasattr(pattern, 'min_bars'):
                pattern_dict['min_bars'] = pattern.min_bars
            if hasattr(pattern, 'min_wick_ratio'):
                pattern_dict['min_wick_ratio'] = pattern.min_wick_ratio
            if hasattr(pattern, 'max_body_ratio'):
                pattern_dict['max_body_ratio'] = pattern.max_body_ratio
            if hasattr(pattern, 'pattern_type'):
                pattern_dict['pattern_type'] = pattern.pattern_type
        
        return pattern_dict
    
    def _dict_to_pattern(self, pattern_dict: Dict[str, Any]) -> CandlestickPattern:
        """Convert dictionary back to pattern"""
        pattern_type = pattern_dict['type']
        timeframes = [TimeRange(tf['value'], tf['unit']) for tf in pattern_dict['timeframes']]
        
        if pattern_type == 'CustomPattern':
            ohlc_ratios = [
                OHLCRatio(
                    body_ratio=ratio.get('body_ratio'),
                    upper_wick_ratio=ratio.get('upper_wick_ratio'),
                    lower_wick_ratio=ratio.get('lower_wick_ratio')
                )
                for ratio in pattern_dict.get('ohlc_ratios', [])
            ]
            
            return CustomPattern(
                name=pattern_dict['name'],
                timeframes=timeframes,
                ohlc_ratios=ohlc_ratios,
                custom_formula=pattern_dict.get('custom_formula'),
                required_bars=pattern_dict.get('required_bars', 1),
                advanced_features=pattern_dict.get('advanced_features', {})
            )
        else:
            # Create using pattern factory
            kwargs = {'timeframes': timeframes}
            
            if pattern_type == 'IIBarsPattern':
                kwargs['min_bars'] = pattern_dict.get('min_bars', 2)
            elif pattern_type == 'DoubleWickPattern':
                kwargs['min_wick_ratio'] = pattern_dict.get('min_wick_ratio', 0.3)
                kwargs['max_body_ratio'] = pattern_dict.get('max_body_ratio', 0.4)
            elif pattern_type == 'HammerPattern':
                kwargs['min_lower_wick_ratio'] = pattern_dict.get('min_lower_wick_ratio', 0.6)
                kwargs['max_upper_wick_ratio'] = pattern_dict.get('max_upper_wick_ratio', 0.1)
            elif pattern_type == 'EngulfingPattern':
                kwargs['pattern_type'] = pattern_dict.get('pattern_type', 'both')
            
            pattern = PatternFactory.create_pattern(pattern_type.lower().replace('pattern', ''), **kwargs)
            pattern.name = pattern_dict['name']
            return pattern
    
    def _strategy_to_dict(self, strategy: PatternStrategy) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization"""
        strategy_dict = {
            'name': strategy.name,
            'type': strategy.type,
            'combination_logic': strategy.combination_logic,
            'min_actions_required': strategy.min_actions_required,
            'gates_and_logic': strategy.gates_and_logic,
            'actions': []
        }
        
        for action in strategy.actions:
            action_dict = {
                'name': action.name,
                'location_strategy': action.location_strategy,
                'location_params': action.location_params,
                'filters': action.filters
            }
            
            if action.pattern:
                action_dict['pattern'] = self._pattern_to_dict(action.pattern)
            
            if action.time_range:
                action_dict['time_range'] = {
                    'value': action.time_range.value,
                    'unit': action.time_range.unit
                }
            
            strategy_dict['actions'].append(action_dict)
        
        return strategy_dict
    
    def _dict_to_strategy(self, strategy_dict: Dict[str, Any]) -> PatternStrategy:
        """Convert dictionary back to strategy"""
        strategy = PatternStrategy(
            name=strategy_dict['name'],
            combination_logic=strategy_dict.get('combination_logic', 'AND'),
            min_actions_required=strategy_dict.get('min_actions_required', 1),
            gates_and_logic=strategy_dict.get('gates_and_logic', {})
        )
        
        for action_dict in strategy_dict.get('actions', []):
            action = Action(
                name=action_dict['name'],
                location_strategy=action_dict.get('location_strategy'),
                location_params=action_dict.get('location_params', {}),
                filters=action_dict.get('filters', [])
            )
            
            if 'pattern' in action_dict:
                action.pattern = self._dict_to_pattern(action_dict['pattern'])
            
            if 'time_range' in action_dict:
                tr = action_dict['time_range']
                action.time_range = TimeRange(tr['value'], tr['unit'])
            
            strategy.add_action(action)
        
        return strategy
    
    def save_multi_timeframe_dataset(self, 
                                   name: str, 
                                   original_data: pd.DataFrame,
                                   timeframes: List[TimeRange],
                                   metadata: Dict[str, Any] = None) -> bool:
        """Save a multi-timeframe dataset with automatic timeframe creation"""
        try:
            # Create multi-timeframe datasets
            mtf_datasets = self.mtf_processor.create_timeframe_datasets(original_data, timeframes)
            
            # Save each timeframe dataset
            saved_timeframes = []
            for tf_str, tf_df in mtf_datasets.items():
                tf_name = f"{name}_{tf_str}"
                if self.save_dataset(tf_df, tf_name, metadata):
                    saved_timeframes.append(tf_str)
            
            # Save original data
            original_name = f"{name}_original"
            if self.save_dataset(original_data, original_name, metadata):
                saved_timeframes.append("original")
            
            # Save multi-timeframe metadata
            mtf_metadata = {
                'name': name,
                'timeframes': saved_timeframes,
                'original_timeframes': [f"{tf.value}{tf.unit}" for tf in timeframes],
                'created_at': datetime.now().isoformat(),
                'original_rows': len(original_data),
                'timeframe_rows': {tf_str: len(tf_df) for tf_str, tf_df in mtf_datasets.items()}
            }
            
            if metadata:
                mtf_metadata.update(metadata)
            
            mtf_config_path = self.configs_dir / f"{name}_mtf_config.json"
            with open(mtf_config_path, 'w') as f:
                json.dump(mtf_metadata, f, indent=2, default=str)
            
            print(f"Saved multi-timeframe dataset '{name}' with timeframes: {saved_timeframes}")
            return True
            
        except Exception as e:
            print(f"Error saving multi-timeframe dataset: {e}")
            return False
    
    def load_multi_timeframe_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a multi-timeframe dataset with all available timeframes"""
        try:
            # Load multi-timeframe configuration
            mtf_config_path = self.configs_dir / f"{name}_mtf_config.json"
            if not mtf_config_path.exists():
                return None
            
            with open(mtf_config_path, 'r') as f:
                mtf_config = json.load(f)
            
            # Load all timeframe datasets
            datasets = {}
            for tf_str in mtf_config.get('timeframes', []):
                if tf_str == 'original':
                    dataset_name = f"{name}_original"
                else:
                    dataset_name = f"{name}_{tf_str}"
                
                dataset_info = self.load_dataset(dataset_name)
                if dataset_info:
                    datasets[tf_str] = dataset_info
            
            return {
                'config': mtf_config,
                'datasets': datasets
            }
            
        except Exception as e:
            print(f"Error loading multi-timeframe dataset: {e}")
            return None
    
    def ensure_timeframes_available(self, 
                                  dataset_name: str, 
                                  required_timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Ensure all required timeframes are available for a dataset, creating missing ones"""
        try:
            # Load the multi-timeframe dataset
            mtf_info = self.load_multi_timeframe_dataset(dataset_name)
            if not mtf_info:
                print(f"Multi-timeframe dataset '{dataset_name}' not found")
                return {}
            
            available_timeframes = list(mtf_info['datasets'].keys())
            missing_timeframes = []
            
            for tf_str in required_timeframes:
                if tf_str not in available_timeframes:
                    missing_timeframes.append(tf_str)
            
            if missing_timeframes:
                print(f"Creating missing timeframes for '{dataset_name}': {missing_timeframes}")
                
                # Get original data
                original_data = mtf_info['datasets'].get('original', {}).get('data')
                if original_data is None:
                    print("Original data not found, cannot create missing timeframes")
                    return {}
                
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
                additional_datasets = self.mtf_processor.create_timeframe_datasets(
                    original_data, 
                    additional_timeframes
                )
                
                # Save additional datasets
                for tf_str, tf_df in additional_datasets.items():
                    tf_name = f"{dataset_name}_{tf_str}"
                    self.save_dataset(tf_df, tf_name, mtf_info['config'])
                    mtf_info['datasets'][tf_str] = {'data': tf_df, 'metadata': mtf_info['config']}
                
                # Update configuration
                mtf_info['config']['timeframes'].extend(missing_timeframes)
                mtf_info['config']['timeframe_rows'].update(
                    {tf_str: len(tf_df) for tf_str, tf_df in additional_datasets.items()}
                )
                
                # Save updated configuration
                mtf_config_path = self.configs_dir / f"{dataset_name}_mtf_config.json"
                with open(mtf_config_path, 'w') as f:
                    json.dump(mtf_info['config'], f, indent=2, default=str)
                
                print(f"Created and saved {len(missing_timeframes)} missing timeframes")
            
            # Return all available datasets
            return {
                tf_str: dataset_info['data'] 
                for tf_str, dataset_info in mtf_info['datasets'].items()
                if tf_str in required_timeframes
            }
            
        except Exception as e:
            print(f"Error ensuring timeframes available: {e}")
            return {}
    
    def get_strategy_timeframes(self, strategy: PatternStrategy) -> List[str]:
        """Extract required timeframes from a strategy"""
        timeframes = set()
        unit_map = {'minute': 'm', 'minutes': 'm', 'hour': 'h', 'hours': 'h', 'day': 'd', 'days': 'd', 'second': 's', 'seconds': 's'}
        
        if hasattr(strategy, 'actions'):
            for action in strategy.actions:
                if action.time_range:
                    # Handle both TimeRange objects and dictionaries
                    if hasattr(action.time_range, 'value') and hasattr(action.time_range, 'unit'):
                        value = action.time_range.value
                        unit = action.time_range.unit
                    elif isinstance(action.time_range, dict):
                        value = action.time_range.get('value')
                        unit = action.time_range.get('unit')
                    else:
                        value = None
                        unit = None
                    if value is not None and unit is not None:
                        abbr_unit = unit_map.get(str(unit).lower(), str(unit)[0])
                        timeframes.add(f"{value}{abbr_unit}")
        
        return list(timeframes) 