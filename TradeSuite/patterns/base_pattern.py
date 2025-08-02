"""
Base pattern class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

from core.data_structures import TimeRange

class CandlestickPattern(ABC):
    """Abstract base class for all candlestick patterns"""
    
    def __init__(self, name: str, timeframes: List[TimeRange]):
        self.name = name
        self.timeframes = timeframes
        self.required_bars = self.get_required_bars()
        
    @abstractmethod
    def get_required_bars(self) -> int:
        """Return number of bars needed to identify this pattern"""
        pass
    
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect pattern in data
        Returns: Series of boolean values indicating pattern presence
        """
        pass
    
    @abstractmethod
    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate pattern strength (0-1)
        Returns: Series of float values indicating pattern strength
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)