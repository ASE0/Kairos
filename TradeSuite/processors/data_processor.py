"""
processors/data_processor.py
============================
Data processing, stripping, and transformation classes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from core.data_structures import TimeRange, DatasetMetadata, VolatilityProfile

logger = logging.getLogger(__name__)


class DataStripper:
    """Handles data stripping and column management"""
    
    # Default column mapping for Sierra Chart data
    DEFAULT_COLUMN_MAP = {
        'Date': 'date',
        'Time': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Last': 'close',
        'Volume': 'volume',
        '# of Trades': 'trades',
        'OHLC Avg': 'ohlc_avg',
        'HLC Avg': 'hlc_avg',
        'HL Avg': 'hl_avg',
        'Bid Volume': 'bid_volume',
        'Ask Volume': 'ask_volume',
        'Bid': 'bid',
        'Ask': 'ask',
        'IBH': 'ibh',
        'IBL': 'ibl'
    }
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.metadata = None
        
    def load_data(self, filepath: str, column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Load data from CSV file with column mapping"""
        try:
            # Load data
            self.original_data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.original_data)} rows from {filepath}")
            
            # Apply column mapping
            mapping = column_mapping or self.DEFAULT_COLUMN_MAP
            self.original_data.rename(columns=mapping, inplace=True)
            
            # Combine date and time columns if they exist
            if 'date' in self.original_data.columns and 'time' in self.original_data.columns:
                self.original_data['datetime'] = pd.to_datetime(
                    self.original_data['date'] + ' ' + self.original_data['time']
                )
                self.original_data.set_index('datetime', inplace=True)
                self.original_data.drop(['date', 'time'], axis=1, inplace=True)
            
            # Initialize metadata
            self.metadata = DatasetMetadata(
                name=filepath.split('/')[-1].split('.')[0],
                source_file=filepath,
                rows_original=len(self.original_data),
                columns_kept=list(self.original_data.columns)
            )
            
            self.processed_data = self.original_data.copy()
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def define_columns(self, column_definitions: str) -> Dict[str, str]:
        """
        Define columns from comma-separated string
        Format: "column1:type1,column2:type2,..."
        """
        mapping = {}
        
        for definition in column_definitions.split(','):
            if ':' in definition:
                col_name, col_type = definition.strip().split(':')
                mapping[col_name.strip()] = col_type.strip()
                
        return mapping
    
    def strip_columns(self, columns_to_keep: List[str]) -> pd.DataFrame:
        """Keep only specified columns"""
        if self.processed_data is None:
            raise ValueError("No data loaded")
            
        # Track removed columns
        removed_columns = [col for col in self.processed_data.columns 
                          if col not in columns_to_keep]
        
        # Strip columns
        self.processed_data = self.processed_data[columns_to_keep]
        
        # Update metadata
        self.metadata.columns_kept = columns_to_keep
        self.metadata.columns_removed.extend(removed_columns)
        self.metadata.filters_applied.append({
            'type': 'column_strip',
            'kept': columns_to_keep,
            'removed': removed_columns
        })
        
        logger.info(f"Kept {len(columns_to_keep)} columns, removed {len(removed_columns)}")
        return self.processed_data
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns for trading analysis"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def validate_required_columns(self) -> bool:
        """Validate that all required columns are present"""
        if self.processed_data is None:
            return False
            
        required = self.get_required_columns()
        return all(col in self.processed_data.columns for col in required)


class LCMDataFilter:
    """Handles LCM-based data filtering and timeframe alignment"""
    
    def __init__(self, base_lcm: int = 1):
        self.base_lcm = base_lcm  # Base LCM in seconds
        
    def calculate_lcm_points(self, timeframes: List[TimeRange], 
                           lcm_multiples: int = 2) -> List[int]:
        """Calculate which data points to keep based on LCM logic"""
        # Convert timeframes to seconds
        timeframe_seconds = [tf.to_seconds() for tf in timeframes]
        
        # Find LCM of all timeframes
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        overall_lcm = reduce(lcm, timeframe_seconds)
        
        # Calculate points to keep
        keep_points = set()
        
        for tf_seconds in timeframe_seconds:
            # Keep points at timeframe boundaries +/- lcm_multiples
            for multiple in range(-lcm_multiples, lcm_multiples + 1):
                point = tf_seconds + (multiple * self.base_lcm)
                if point >= 0:
                    keep_points.add(point)
                    
        return sorted(list(keep_points))
    
    def filter_data(self, data: pd.DataFrame, timeframes: List[TimeRange],
                   lcm_multiples: int = 2) -> pd.DataFrame:
        """Filter data based on LCM logic"""
        if data.empty:
            return data
            
        # Ensure we have a proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index is not a DatetimeIndex, attempting to convert...")
            try:
                data = data.copy()
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"Failed to convert index to datetime: {e}")
                return data
            
        # Get points to keep
        keep_points = self.calculate_lcm_points(timeframes, lcm_multiples)
        
        # Calculate seconds from start for each row
        start_time = data.index[0]
        seconds_from_start = (data.index - start_time).total_seconds()
        
        # Ensure seconds_from_start is a pandas Series
        if not isinstance(seconds_from_start, pd.Series):
            seconds_from_start = pd.Series(seconds_from_start, index=data.index)
        
        # Filter based on LCM points
        mask = pd.Series(False, index=data.index)
        
        for point in keep_points:
            # Find closest data points to each LCM point
            distances = (seconds_from_start - point).abs()  # Keep as pandas Series
            if not distances.empty:
                closest_idx = distances.idxmin()
                mask.loc[closest_idx] = True
            
        filtered_data = data[mask]
        logger.info(f"Filtered data from {len(data)} to {len(filtered_data)} rows")
        
        return filtered_data
    
    def align_multi_timeframe_data(self, data: pd.DataFrame, 
                                  timeframes: List[TimeRange]) -> Dict[str, pd.DataFrame]:
        """Create aligned data for multiple timeframes"""
        aligned_data = {}
        
        for tf in timeframes:
            # Resample data to timeframe
            tf_str = f"{tf.value}{tf.unit}"
            
            # Pandas resample string
            resample_map = {'s': 'S', 'm': 'T', 'h': 'H', 'd': 'D'}
            resample_str = f"{tf.value}{resample_map.get(tf.unit, tf.unit)}"
            
            # Resample with proper aggregation
            resampled = data.resample(resample_str).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            aligned_data[tf_str] = resampled
            
        return aligned_data


class PatternFilter:
    """Filters data based on candlestick patterns"""
    
    def __init__(self):
        self.applied_filters = []
        
    def filter_by_pattern(self, data: pd.DataFrame, 
                         pattern: 'CandlestickPattern',
                         keep_pattern: bool = True) -> pd.DataFrame:
        """Filter data to keep/remove pattern occurrences"""
        # Detect pattern
        pattern_signals = pattern.detect(data)
        
        # Apply filter
        if keep_pattern:
            filtered_data = data[pattern_signals]
        else:
            filtered_data = data[~pattern_signals]
            
        # Track filter
        self.applied_filters.append({
            'pattern': pattern.name,
            'keep': keep_pattern,
            'rows_before': len(data),
            'rows_after': len(filtered_data)
        })
        
        logger.info(f"Pattern filter '{pattern.name}' applied: "
                   f"{len(data)} -> {len(filtered_data)} rows")
        
        return filtered_data
    
    def filter_by_multiple_patterns(self, data: pd.DataFrame,
                                   patterns: List['CandlestickPattern'],
                                   logic: str = 'AND') -> pd.DataFrame:
        """Filter by multiple patterns with AND/OR logic"""
        if not patterns:
            return data
            
        # Get signals for each pattern
        all_signals = pd.DataFrame(index=data.index)
        
        for pattern in patterns:
            all_signals[pattern.name] = pattern.detect(data)
            
        # Apply logic
        if logic == 'AND':
            combined_signal = all_signals.all(axis=1)
        elif logic == 'OR':
            combined_signal = all_signals.any(axis=1)
        else:
            raise ValueError(f"Unknown logic: {logic}")
            
        # Filter data
        filtered_data = data[combined_signal]
        
        # Track filter
        self.applied_filters.append({
            'patterns': [p.name for p in patterns],
            'logic': logic,
            'rows_before': len(data),
            'rows_after': len(filtered_data)
        })
        
        return filtered_data


class VolatilityCalculator:
    """Calculates and categorizes market volatility"""
    
    def __init__(self):
        self.methods = {
            'atr': self.calculate_atr_volatility,
            'std': self.calculate_std_volatility,
            'parkinson': self.calculate_parkinson_volatility,
            'garman_klass': self.calculate_garman_klass_volatility
        }
        
    def calculate_volatility(self, data: pd.DataFrame, 
                           method: str = 'atr',
                           period: int = 20) -> VolatilityProfile:
        """Calculate volatility using specified method"""
        if method not in self.methods:
            raise ValueError(f"Unknown volatility method: {method}")
        
        # Check for required OHLC columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            # Try to find alternative column names
            column_mapping = {}
            for col in missing_columns:
                alternatives = [c for c in data.columns if col in c.lower()]
                if alternatives:
                    column_mapping[col] = alternatives[0]
                    logger.info(f"Using '{alternatives[0]}' for '{col}' column")
                else:
                    # Try more flexible matching
                    flexible_alternatives = []
                    for available_col in data.columns:
                        available_col_lower = available_col.lower().strip()
                        if col == 'close' and ('close' in available_col_lower or 'last' in available_col_lower):
                            flexible_alternatives.append(available_col)
                        elif col == 'open' and 'open' in available_col_lower:
                            flexible_alternatives.append(available_col)
                        elif col == 'high' and 'high' in available_col_lower:
                            flexible_alternatives.append(available_col)
                        elif col == 'low' and 'low' in available_col_lower:
                            flexible_alternatives.append(available_col)
                    
                    if flexible_alternatives:
                        column_mapping[col] = flexible_alternatives[0]
                        logger.info(f"Using flexible alternative '{flexible_alternatives[0]}' for '{col}' column")
                    else:
                        raise ValueError(f"Missing required column '{col}' for volatility calculation. Available columns: {list(data.columns)}")
            
            # Create a copy with proper column names
            data = data.copy()
            for standard_name, actual_name in column_mapping.items():
                data[standard_name] = data[actual_name]
            
        # Calculate raw volatility
        volatility_series = self.methods[method](data, period)
        
        # Normalize to 0-100 scale
        min_vol = volatility_series.quantile(0.05)
        max_vol = volatility_series.quantile(0.95)
        
        if max_vol > min_vol:
            normalized = ((volatility_series.mean() - min_vol) / (max_vol - min_vol)) * 100
            volatility_value = int(np.clip(normalized, 0, 100))
        else:
            volatility_value = 50
            
        # Create profile
        profile = VolatilityProfile(
            value=volatility_value,
            category=VolatilityProfile.categorize(volatility_value),
            calculated_metrics={
                'method': method,
                'raw_value': float(volatility_series.mean()),
                'period': period
            },
            user_defined=False
        )
        
        return profile
    
    def calculate_atr_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range volatility"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        
        # Normalize by price
        return atr / close
    
    def calculate_std_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate standard deviation volatility"""
        returns = data['close'].pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    
    def calculate_parkinson_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        return log_hl.rolling(window=period).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2)))
        ) * np.sqrt(252)
    
    def calculate_garman_klass_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        
        rs = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
        
        return rs.rolling(window=period).apply(
            lambda x: np.sqrt(np.sum(x) / len(x))
        ) * np.sqrt(252)


class DatasetProcessor:
    """Main processor that combines all data processing capabilities"""
    
    def __init__(self):
        self.stripper = DataStripper()
        self.lcm_filter = LCMDataFilter()
        self.pattern_filter = PatternFilter()
        self.volatility_calc = VolatilityCalculator()
        self.processed_datasets = {}
        
    def process_dataset(self, filepath: str, 
                       processing_config: Dict[str, Any]) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Process a dataset with full configuration"""
        
        # Load data
        data = self.stripper.load_data(filepath, 
                                      processing_config.get('column_mapping'))
        
        # Strip columns if specified
        if 'columns_to_keep' in processing_config:
            data = self.stripper.strip_columns(processing_config['columns_to_keep'])
            
        # Apply LCM filter if timeframes specified
        if 'timeframes' in processing_config:
            timeframes = [TimeRange(**tf) for tf in processing_config['timeframes']]
            data = self.lcm_filter.filter_data(
                data, 
                timeframes,
                processing_config.get('lcm_multiples', 2)
            )
            
        # Apply pattern filters if specified
        if 'pattern_filters' in processing_config:
            for pf_config in processing_config['pattern_filters']:
                # Create pattern instance
                from patterns.candlestick_patterns import PatternFactory
                pattern = PatternFactory.create_pattern(**pf_config['pattern'])
                
                # Apply filter
                data = self.pattern_filter.filter_by_pattern(
                    data,
                    pattern,
                    pf_config.get('keep', True)
                )
                
        # Calculate volatility
        volatility_profile = self.volatility_calc.calculate_volatility(
            data,
            processing_config.get('volatility_method', 'atr'),
            processing_config.get('volatility_period', 20)
        )
        
        # Update metadata
        self.stripper.metadata.rows_processed = len(data)
        self.stripper.metadata.volatility = volatility_profile
        
        # Store processed dataset
        dataset_name = processing_config.get('name', f"dataset_{len(self.processed_datasets)}")
        self.processed_datasets[dataset_name] = {
            'data': data,
            'metadata': self.stripper.metadata
        }
        
        return data, self.stripper.metadata
