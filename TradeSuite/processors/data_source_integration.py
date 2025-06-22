"""
processors/data_source_integration.py
=====================================
Integration module for multiple data sources (Zorro, Sierra Chart, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataSourceDetector:
    """Automatically detects data source format"""
    
    @staticmethod
    def detect_source(file_path: str, sample_rows: int = 10) -> str:
        """Detect the source of the data file"""
        try:
            # Read first few rows
            df = pd.read_csv(file_path, nrows=sample_rows)
            columns = df.columns.tolist()
            
            # Check for Zorro format
            if DataSourceDetector._is_zorro_format(columns, df):
                return "zorro"
            
            # Check for Sierra Chart format
            elif DataSourceDetector._is_sierra_chart_format(columns):
                return "sierra_chart"
            
            # Check for MetaTrader format
            elif DataSourceDetector._is_metatrader_format(columns):
                return "metatrader"
            
            # Check for NinjaTrader format
            elif DataSourceDetector._is_ninjatrader_format(columns):
                return "ninjatrader"
            
            # Check for generic OHLC format
            elif DataSourceDetector._is_generic_ohlc_format(columns):
                return "generic"
            
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error detecting data source: {e}")
            return "unknown"
    
    @staticmethod
    def _is_zorro_format(columns: List[str], df: pd.DataFrame) -> bool:
        """Check if data is in Zorro format"""
        # Zorro typically exports with specific column names
        zorro_indicators = [
            # Zorro often uses Date format like "2020-01-01 12:00:00"
            any('date' in col.lower() for col in columns),
            # Check for typical Zorro column patterns
            any(col in ['Open', 'High', 'Low', 'Close', 'Volume'] for col in columns),
            # Zorro might include additional columns like Spread
            any('spread' in col.lower() for col in columns)
        ]
        
        # Check date format
        date_col = next((col for col in columns if 'date' in col.lower()), None)
        if date_col and len(df) > 0:
            try:
                # Zorro uses specific date format
                date_str = str(df[date_col].iloc[0])
                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date_str):
                    zorro_indicators.append(True)
            except:
                pass
                
        return sum(zorro_indicators) >= 2
    
    @staticmethod
    def _is_sierra_chart_format(columns: List[str]) -> bool:
        """Check if data is in Sierra Chart format"""
        sierra_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Last', 'Volume', 
                         '# of Trades', 'OHLC Avg', 'HLC Avg', 'HL Avg', 
                         'Bid Volume', 'Ask Volume', 'Bid', 'Ask', 'IBH', 'IBL']
        
        # Check for exact match or subset
        matching = sum(1 for col in sierra_columns if col in columns)
        return matching >= 6  # At least 6 Sierra-specific columns
    
    @staticmethod
    def _is_metatrader_format(columns: List[str]) -> bool:
        """Check if data is in MetaTrader format"""
        mt_indicators = [
            any('time' in col.lower() for col in columns),
            any('open' in col.lower() for col in columns),
            any('high' in col.lower() for col in columns),
            any('low' in col.lower() for col in columns),
            any('close' in col.lower() for col in columns),
            any('tick' in col.lower() for col in columns),
        ]
        return sum(mt_indicators) >= 4
    
    @staticmethod
    def _is_ninjatrader_format(columns: List[str]) -> bool:
        """Check if data is in NinjaTrader format"""
        # NinjaTrader typically uses semicolon separators and specific format
        nt_indicators = [
            'Time' in columns,
            'Open' in columns,
            'High' in columns,
            'Low' in columns,
            'Close' in columns,
            'Volume' in columns
        ]
        return sum(nt_indicators) >= 5
    
    @staticmethod
    def _is_generic_ohlc_format(columns: List[str]) -> bool:
        """Check if data is in generic OHLC format"""
        required = ['open', 'high', 'low', 'close']
        columns_lower = [col.lower() for col in columns]
        return all(req in columns_lower for req in required)


class DataSourceMapper:
    """Maps columns from different sources to standard format"""
    
    # Standard column names used internally
    STANDARD_COLUMNS = {
        'datetime': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'trades': 'trades',
        'bid': 'bid',
        'ask': 'ask',
        'bid_volume': 'bid_volume',
        'ask_volume': 'ask_volume'
    }
    
    # Column mappings for different sources
    SOURCE_MAPPINGS = {
        'zorro': {
            'Date': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Spread': 'spread'
        },
        'sierra_chart': {
            'Date': 'date',
            'Time': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'close',  # Note: Sierra uses 'Last' instead of 'Close'
            'Volume': 'volume',
            '# of Trades': 'trades',
            'Bid': 'bid',
            'Ask': 'ask',
            'Bid Volume': 'bid_volume',
            'Ask Volume': 'ask_volume'
        },
        'metatrader': {
            'Time': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            ' Last': 'close',  # Handle MetaTrader's ' Last' column
            'Volume': 'volume',
            'Spread': 'spread'
        },
        'ninjatrader': {
            'Time': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
    }
    
    @classmethod
    def get_mapping(cls, source: str) -> Dict[str, str]:
        """Get column mapping for a specific source"""
        return cls.SOURCE_MAPPINGS.get(source, {})
    
    @classmethod
    def map_columns(cls, df: pd.DataFrame, source: str, 
                   custom_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Map columns from source format to standard format"""
        # Get base mapping for source
        mapping = cls.get_mapping(source)
        
        logger.info(f"Mapping columns for source: {source}")
        logger.info(f"Original columns: {list(df.columns)}")
        logger.info(f"Base mapping: {mapping}")
        
        # Override with custom mapping if provided
        if custom_mapping:
            mapping.update(custom_mapping)
            logger.info(f"Updated mapping with custom: {mapping}")
        
        # Create a copy of the dataframe to work with
        df_mapped = df.copy()
        
        # Apply mapping with fallback for missing columns
        for expected_col, target_col in mapping.items():
            if expected_col in df_mapped.columns:
                # Direct match
                df_mapped = df_mapped.rename(columns={expected_col: target_col})
            else:
                # Try to find column with leading/trailing spaces
                stripped_cols = {col.strip(): col for col in df_mapped.columns}
                if expected_col.strip() in stripped_cols:
                    actual_col = stripped_cols[expected_col.strip()]
                    df_mapped = df_mapped.rename(columns={actual_col: target_col})
                    logger.info(f"Mapped '{actual_col}' to '{target_col}'")
        
        logger.info(f"Columns after mapping: {list(df_mapped.columns)}")
        
        # Handle special cases
        if source == 'sierra_chart':
            # Combine date and time columns
            if 'date' in df_mapped.columns and 'time' in df_mapped.columns:
                df_mapped['datetime'] = pd.to_datetime(
                    df_mapped['date'] + ' ' + df_mapped['time']
                )
                df_mapped.drop(['date', 'time'], axis=1, inplace=True)
        
        return df_mapped


class ZorroDataLoader:
    """Specialized loader for Zorro data files"""
    
    def __init__(self):
        self.pattern_columns = []
        self.indicator_columns = []
        
    def load_zorro_export(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load Zorro export file with metadata"""
        # Detect delimiter (Zorro can use comma or semicolon)
        with open(file_path, 'r') as f:
            first_line = f.readline()
            delimiter = ';' if ';' in first_line else ','
        
        # Load data
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Parse metadata from column names
        metadata = self._extract_metadata(df)
        
        # Standardize columns
        df = self._standardize_zorro_data(df)
        
        return df, metadata
    
    def _extract_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract Zorro-specific metadata"""
        metadata = {
            'source': 'zorro',
            'patterns': [],
            'indicators': [],
            'timeframe': None,
            'asset': None
        }
        
        # Look for pattern columns (Zorro might export pattern signals)
        for col in df.columns:
            if 'pattern' in col.lower():
                metadata['patterns'].append(col)
            elif any(ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd']):
                metadata['indicators'].append(col)
        
        # Try to detect timeframe from data
        if 'datetime' in df.columns or 'Date' in df.columns:
            date_col = 'datetime' if 'datetime' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            if len(df) > 1:
                time_diff = (df[date_col].iloc[1] - df[date_col].iloc[0]).total_seconds()
                metadata['timeframe'] = self._seconds_to_timeframe(time_diff)
        
        return metadata
    
    def _standardize_zorro_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Zorro data format"""
        # Map columns
        df = DataSourceMapper.map_columns(df, 'zorro')
        
        # Ensure datetime index
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
            df.set_index('datetime', inplace=True)
            df.drop('Date', axis=1, inplace=True)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume' and 'Volume' in df.columns:
                    df[col] = df['Volume']
                elif col == 'close' and 'Close' not in df.columns and 'Last' in df.columns:
                    df[col] = df['Last']
                else:
                    logger.warning(f"Missing required column: {col}")
        
        return df
    
    def _seconds_to_timeframe(self, seconds: float) -> str:
        """Convert seconds to timeframe string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h"
        else:
            return f"{int(seconds/86400)}d"
    
    def load_zorro_pattern_results(self, file_path: str) -> pd.DataFrame:
        """Load Zorro pattern detection results"""
        df = pd.read_csv(file_path)
        
        # Expected columns: Date, Pattern, Confidence, Direction, etc.
        pattern_data = {
            'datetime': pd.to_datetime(df['Date']),
            'pattern': df.get('Pattern', 'Unknown'),
            'confidence': df.get('Confidence', 1.0),
            'direction': df.get('Direction', 'Long'),
            'entry': df.get('Entry', df.get('Close', 0)),
            'stop': df.get('Stop', 0),
            'target': df.get('Target', 0)
        }
        
        return pd.DataFrame(pattern_data)


class UniversalDataLoader:
    """Universal loader that handles all data sources"""
    
    def __init__(self):
        self.detector = DataSourceDetector()
        self.zorro_loader = ZorroDataLoader()
        
    def load_data(self, file_path: str, 
                 source: Optional[str] = None,
                 custom_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from any supported source
        
        Args:
            file_path: Path to data file
            source: Optional source override ('zorro', 'sierra_chart', etc.)
            custom_mapping: Optional custom column mapping
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        # Detect source if not specified
        if source is None:
            source = self.detector.detect_source(file_path)
            logger.info(f"Detected data source: {source}")
        
        # Load based on source
        if source == 'zorro':
            df, metadata = self.zorro_loader.load_zorro_export(file_path)
        else:
            # Generic loading
            df = pd.read_csv(file_path)
            metadata = {'source': source}
            
            # Apply column mapping
            df = DataSourceMapper.map_columns(df, source, custom_mapping)
            
            # Ensure datetime index
            self._ensure_datetime_index(df, source)
        
        # Validate data
        self._validate_data(df)
        
        # Add source to metadata
        metadata['source'] = source
        metadata['file_path'] = file_path
        metadata['rows'] = len(df)
        metadata['columns'] = df.columns.tolist()
        
        return df, metadata
    
    def _ensure_datetime_index(self, df: pd.DataFrame, source: str):
        """Ensure DataFrame has datetime index"""
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        elif df.index.name != 'datetime' and not isinstance(df.index, pd.DatetimeIndex):
            # Try to find a datetime column
            date_columns = [col for col in df.columns 
                          if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                df.index = pd.to_datetime(df[date_columns[0]])
                df.index.name = 'datetime'
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate that data has required columns"""
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Try to find alternative column names
            for col in missing:
                alternatives = [c for c in df.columns if col in c.lower()]
                if alternatives:
                    logger.info(f"Found alternative for {col}: {alternatives[0]}")
                    df[col] = df[alternatives[0]]
                else:
                    # Try more flexible matching
                    flexible_alternatives = []
                    for available_col in df.columns:
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
                        logger.info(f"Found flexible alternative for {col}: {flexible_alternatives[0]}")
                        df[col] = df[flexible_alternatives[0]]
                    else:
                        logger.error(f"No alternative found for required column: {col}")
                        logger.error(f"Available columns: {list(df.columns)}")


# Integration with existing DataStripper
class EnhancedDataStripper:
    """Enhanced data stripper with multi-source support"""
    
    def __init__(self):
        self.universal_loader = UniversalDataLoader()
        self.metadata = None
        self.source = None
        
    def load_data(self, filepath: str, 
                 source: Optional[str] = None,
                 column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Load data from any supported source"""
        # Use universal loader
        df, metadata = self.universal_loader.load_data(filepath, source, column_mapping)
        
        self.metadata = metadata
        self.source = metadata['source']
        
        # Log source-specific information
        logger.info(f"Loaded {self.source} data: {len(df)} rows")
        
        if self.source == 'zorro' and metadata.get('patterns'):
            logger.info(f"Found Zorro patterns: {metadata['patterns']}")
        
        return df
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the loaded data source"""
        return {
            'source': self.source,
            'metadata': self.metadata,
            'column_mapping': DataSourceMapper.get_mapping(self.source)
        }


# Example Zorro integration script
ZORRO_EXPORT_SCRIPT = """
// Zorro script to export pattern data for Python integration
// Save this as ExportPatterns.c in Zorro/Strategy folder

function run()
{
    BarPeriod = 5; // 5-minute bars
    StartDate = 20230101;
    EndDate = 20231231;
    
    asset("EUR/USD"); // or your asset
    
    // Define file handle
    string filename = "Data\\\\PythonExport.csv";
    
    if(is(INITRUN)) {
        // Write header
        file_delete(filename);
        file_append(filename, 
            "Date,Open,High,Low,Close,Volume,Pattern,PatternStrength,VWAP,RSI\\n");
    }
    
    // Calculate indicators
    vars Price = series(price());
    var vwap = VWAP(Price, 20);
    var rsi = RSI(Price, 14);
    
    // Detect patterns
    int pat = pattern(0); // Detect all patterns
    var patStrength = 0;
    
    if(pat & PATTERN_HAMMER) patStrength = 0.8;
    else if(pat & PATTERN_DOJI) patStrength = 0.6;
    else if(pat & PATTERN_ENGULFING) patStrength = 0.9;
    
    // Export data
    if(!is(LOOKBACK)) {
        file_append(filename, strf("%s,%f,%f,%f,%f,%f,%d,%f,%f,%f\\n",
            strdate("%Y-%m-%d %H:%M:%S", 0),
            priceOpen(0), priceHigh(0), priceLow(0), priceClose(0),
            marketVol(0), pat, patStrength, vwap, rsi));
    }
}
"""