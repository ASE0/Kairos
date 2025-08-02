"""
Fair Value Gap (FVG) pattern implementation
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from core.data_structures import TimeRange
from patterns.base_pattern import CandlestickPattern
from core.feature_quantification import detect_fvg

class FVGPattern(CandlestickPattern):
    """Fair Value Gap pattern"""
    
    def __init__(self, timeframes: List[TimeRange], min_gap_size: float = 0.001):
        super().__init__("FVG", timeframes)
        self.min_gap_size = min_gap_size
    
    def get_required_bars(self) -> int:
        return 3  # Need 3 bars to detect a gap
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect FVG patterns in data"""
        if not self.validate_data(data):
            return pd.Series(False, index=data.index)
        
        # Get FVG zones
        fvgs = detect_fvg(data['high'].values, data['low'].values, data['close'].values, self.min_gap_size)
        
        # Create signal series
        signals = pd.Series(False, index=data.index)
        
        # Mark signal at the start of each FVG
        for fvg in fvgs:
            start_idx = fvg['start_idx']
            if start_idx < len(signals):
                signals.iloc[start_idx] = True
        
        return signals
    
    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate pattern strength"""
        signals = self.detect(data)
        
        # Get FVG zones
        fvgs = detect_fvg(data['high'].values, data['low'].values, data['close'].values, self.min_gap_size)
        
        # Create strength series
        strength = pd.Series(0.0, index=data.index)
        
        # Set strength at each FVG start
        for fvg in fvgs:
            start_idx = fvg['start_idx']
            if start_idx < len(strength):
                strength.iloc[start_idx] = fvg['strength']
        
        return strength.where(signals, 0.0).fillna(0.0)