"""
patterns/candlestick_patterns.py
=================================
Candlestick pattern definitions and detection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from core.data_structures import OHLCRatio, TimeRange, BaseStrategy
from core.feature_quantification import (
    body_size, upper_wick, lower_wick, wick_ratios, doji_ness,
    two_bar_strength, dual_layer_location, momentum_boost,
    realized_vol, atr, bayesian_update
)


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


class IIBarsPattern(CandlestickPattern):
    """Inside-Inside bars pattern"""
    
    def __init__(self, timeframes: List[TimeRange], min_bars: int = 2):
        self.min_bars = min_bars
        super().__init__("II_Bars", timeframes)
        
    def get_required_bars(self) -> int:
        return self.min_bars + 1
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect II bars pattern using vectorized operations."""
        if not self.validate_data(data):
            return pd.Series(False, index=data.index)
            
        # The core condition for an inside bar is that the current bar's range
        # is contained within the previous bar's range.
        is_inside = (data['high'] <= data['high'].shift(1)) & \
                    (data['low'] >= data['low'].shift(1))
        
        # An II-Bars pattern is a series of consecutive inside bars.
        # We use a rolling window to check if all bars in the window are inside bars.
        # The `all()` method on the rolling object returns True if every value in the window is True.
        signals = is_inside.rolling(window=self.min_bars).all()
        
        # Rolling operations produce NaNs for the initial periods; fill them with False.
        return signals.fillna(False)
    
    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate II pattern strength based on volatility compression."""
        signals = self.detect(data)
        strength = pd.Series(0.0, index=data.index)
        
        # Only calculate strength where the pattern is detected
        if signals.any():
            true_range = data['high'] - data['low']
            # Look at the range of the bar that started the pattern sequence
            initial_range = true_range.shift(self.min_bars)
            
            # Compression is the reduction in range size.
            # We avoid division by zero.
            compression = (1 - (true_range / initial_range)).where(initial_range > 0, 0)
            
            # Strength is the compression, clipped between 0 and 1.
            # We only apply this strength to the bars where the signal is True.
            strength = compression.clip(0, 1).where(signals, 0.0)
            
        return strength


class DoubleWickPattern(CandlestickPattern):
    """Double wick pattern (Spinning Top) with customizable ratios"""
    
    def __init__(self, timeframes: List[TimeRange], 
                 min_wick_ratio: float = 0.3,
                 max_body_ratio: float = 0.4):
        self.min_wick_ratio = min_wick_ratio
        self.max_body_ratio = max_body_ratio
        super().__init__("Double_Wick", timeframes)
        
    def get_required_bars(self) -> int:
        return 1
    
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect double wick pattern using vectorized operations."""
        if not self.validate_data(data):
            return pd.Series(False, index=data.index)

        total_range = (data['high'] - data['low']).replace(0, np.nan)
        body = body_size(data['open'], data['close'])
        upper = upper_wick(data['open'], data['close'], data['high'])
        lower = lower_wick(data['open'], data['close'], data['low'])
        
        body_ratio = body / total_range
        upper_wick_ratio = upper / total_range
        lower_wick_ratio = lower / total_range
        
        signals = (body_ratio <= self.max_body_ratio) & \
                  (upper_wick_ratio >= self.min_wick_ratio) & \
                  (lower_wick_ratio >= self.min_wick_ratio)
                  
        return signals.fillna(False)
    
    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate strength based on wick symmetry and body size."""
        signals = self.detect(data)
        strength = pd.Series(0.0, index=data.index)

        if signals.any():
            total_range = (data['high'] - data['low']).replace(0, np.nan)
            body = body_size(data['open'], data['close'])
            upper = upper_wick(data['open'], data['close'], data['high'])
            lower = lower_wick(data['open'], data['close'], data['low'])

            # Strength is a combination of how small the body is and how symmetrical the wicks are.
            wick_symmetry = 1 - (abs(upper - lower) / total_range)
            body_weakness = 1 - (body / total_range)
            
            combined_strength = ((wick_symmetry + body_weakness) / 2).clip(0, 1)
            strength = combined_strength.where(signals, 0.0)
            
        return strength.fillna(0.0)


class EngulfingPattern(CandlestickPattern):
    """Bullish/Bearish engulfing pattern"""
    
    def __init__(self, timeframes: List[TimeRange], 
                 pattern_type: str = 'both'):  # 'bullish', 'bearish', 'both'
        if pattern_type not in ['bullish', 'bearish', 'both']:
            raise ValueError("pattern_type must be 'bullish', 'bearish', or 'both'")
        self.pattern_type = pattern_type
        super().__init__("Engulfing", timeframes)
        
    def get_required_bars(self) -> int:
        return 2

    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detects engulfing patterns using vectorized operations."""
        if not self.validate_data(data):
            return pd.Series(False, index=data.index)

        # Get properties of the current and previous bars
        body = body_size(data['open'], data['close'])
        prev_body = body.shift(1)
        
        is_current_bullish = data['close'] > data['open']
        is_current_bearish = data['close'] < data['open']
        is_prev_bullish = data['close'].shift(1) > data['open'].shift(1)
        is_prev_bearish = data['close'].shift(1) < data['open'].shift(1)

        # Define engulfing conditions
        body_engulfs_prev_body = body > prev_body
        
        # Bullish Engulfing: previous is bearish, current is bullish, and current body engulfs previous body
        is_bullish_engulfing = (is_prev_bearish & is_current_bullish & body_engulfs_prev_body)
        
        # Bearish Engulfing: previous is bullish, current is bearish, and current body engulfs previous body
        is_bearish_engulfing = (is_prev_bullish & is_current_bearish & body_engulfs_prev_body)

        # Return signals based on the selected pattern type
        if self.pattern_type == 'bullish':
            signals = is_bullish_engulfing
        elif self.pattern_type == 'bearish':
            signals = is_bearish_engulfing
        else: # 'both'
            signals = is_bullish_engulfing | is_bearish_engulfing
            
        return signals.fillna(False)

    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate engulfing strength based on the ratio of the two bodies."""
        signals = self.detect(data)
        strength = pd.Series(0.0, index=data.index)

        if signals.any():
            body = body_size(data['open'], data['close'])
            prev_body = body.shift(1).replace(0, np.nan)
            
            # Strength is the ratio of the current body to the previous body.
            # A larger ratio means a more powerful engulfing candle.
            strength_ratio = (body / prev_body).clip(0, 3) / 3 # Normalize to 0-1 range, capping at 3x
            strength = strength_ratio.where(signals, 0.0)
            
        return strength.fillna(0.0)


class CustomPattern(CandlestickPattern):
    """
    A highly configurable pattern based on custom rules, ratios, and formulas.
    WARNING: The `custom_formula` feature uses `eval()` and is not safe to use with untrusted input.
    A secure parsing engine would be required for production use.
    """
    
    def __init__(self, name: str, timeframes: List[TimeRange],
                 ohlc_ratios: List[OHLCRatio],
                 custom_formula: Optional[str] = None,
                 required_bars: int = 1,
                 advanced_features: Optional[Dict[str, Any]] = None):
        
        self.ohlc_ratios = ohlc_ratios
        self.custom_formula = custom_formula
        self._required_bars = required_bars
        self.advanced_features = advanced_features or {}
        
        super().__init__(name, timeframes)
        
    def get_required_bars(self) -> int:
        return self._required_bars

    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detects custom patterns using a combination of vectorized and specific logic."""
        if not self.validate_data(data) or len(data) < self.get_required_bars():
            return pd.Series(False, index=data.index)
        
        # Start with all signals as False
        final_signals = pd.Series(False, index=data.index)

        # 1. Vectorized Basic Ratio Checks
        # We start with a base of True signals that we will progressively filter down.
        if self.ohlc_ratios:
            base_signals = pd.Series(True, index=data.index)
            for ratio in self.ohlc_ratios:
                # This check remains vectorized
                base_signals &= self._check_ratio_vectorized(data, ratio)
        else:
            base_signals = pd.Series(True, index=data.index)
            
        # Get the indices where the base signals are true to avoid looping over the whole dataframe
        possible_indices = base_signals[base_signals].index
        
        # 2. Loop for multi-bar formulas and advanced features
        # This part is harder to vectorize due to the multi-bar nature and custom formulas.
        for i in possible_indices:
            pos = data.index.get_loc(i)
            if pos < self.get_required_bars() - 1:
                continue

            # Get the slice of data for the current pattern check
            bars = data.iloc[pos - self.get_required_bars() + 1 : pos + 1]
            
            formula_ok = True
            if self.custom_formula:
                formula_ok = self._evaluate_formula(bars, self.custom_formula)
            
            advanced_ok = True
            if self.advanced_features:
                advanced_ok = self._apply_advanced_features(data, pos, bars)
            
            if formula_ok and advanced_ok:
                final_signals.at[i] = True
                
        return final_signals

    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate strength for custom pattern."""
        signals = self.detect(data)
        strength = pd.Series(0.0, index=data.index)
        
        # Loop only over the detected signals for efficiency
        for i in signals[signals].index:
            strength.at[i] = self._calculate_advanced_strength(data, i)
            
        return strength

    def _check_ratio_vectorized(self, data: pd.DataFrame, ratio: OHLCRatio) -> pd.Series:
        """Perform a single ratio check in a vectorized manner."""
        results = pd.Series(False, index=data.index)
        if ratio.body_ratio is not None:
            val = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
            results = val > ratio.body_ratio
        elif ratio.upper_wick_ratio is not None:
            val = (data['high'] - data[['open', 'close']].max(axis=1)) / (data['high'] - data['low']).replace(0, np.nan)
            results = val > ratio.upper_wick_ratio
        elif ratio.lower_wick_ratio is not None:
            val = (data[['open', 'close']].min(axis=1) - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
            results = val > ratio.lower_wick_ratio
        elif ratio.custom_formula is not None:
            # Evaluate custom formula if provided
            try:
                results = data.eval(ratio.custom_formula)
            except Exception:
                results = pd.Series(False, index=data.index)
        return results

    def _apply_advanced_features(self, data: pd.DataFrame, index: int, bars: pd.DataFrame) -> bool:
        """
        Apply advanced, non-vectorizable features. This part remains a loop.
        (Original logic is kept as it is complex and strategy-specific)
        """
        if 'vol_increase' in self.advanced_features:
            vol_factor = self.advanced_features['vol_increase']
            avg_vol = data['volume'].iloc[max(0, index - 20):index].mean()
            if bars.iloc[-1]['volume'] < avg_vol * vol_factor:
                return False
        
        if 'atr_factor' in self.advanced_features:
            atr_val = atr(
                data['high'].iloc[max(0, index - 14):index + 1].values,
                data['low'].iloc[max(0, index - 14):index + 1].values,
                data['close'].iloc[max(0, index - 14):index + 1].values
            )
            if (bars.iloc[-1]['high'] - bars.iloc[-1]['low']) < atr_val * self.advanced_features['atr_factor']:
                return False

        if 'min_body_size_vs_atr' in self.advanced_features:
            atr_val = atr(
                data['high'].iloc[max(0, index-14):index+1].values,
                data['low'].iloc[max(0, index-14):index+1].values,
                data['close'].iloc[max(0, index-14):index+1].values
            )
            min_body_factor = self.advanced_features['min_body_size_vs_atr']
            if body_size(bars.iloc[-1]['open'], bars.iloc[-1]['close']) < atr_val * min_body_factor:
                return False

        if 'prev_bar_opposite_color' in self.advanced_features and self.get_required_bars() > 1:
            curr_is_bull = bars.iloc[-1]['close'] > bars.iloc[-1]['open']
            prev_is_bull = bars.iloc[-2]['close'] > bars.iloc[-2]['open']
            if curr_is_bull == prev_is_bull:
                return False
                
        return True

    def _calculate_advanced_strength(self, data: pd.DataFrame, index: int) -> float:
        """(Original logic is kept as it is complex and strategy-specific)"""
        strength_score = 0.5  # Base score
        
        try:
            # Momentum boost
            if 'momentum' in self.advanced_features.get('strength_components', []):
                strength_score += momentum_boost(data, index, lookback=10) * 0.2

            # Two-bar strength
            if 'two_bar' in self.advanced_features.get('strength_components', []):
                if index > 0:
                    strength_score += two_bar_strength(
                        data['open'].iloc[index-1:index+1],
                        data['close'].iloc[index-1:index+1]
                    ) * 0.3

            # Location score
            if 'location' in self.advanced_features.get('strength_components', []):
                strength_score += dual_layer_location(data, index) * 0.5
        except IndexError:
            # Not enough data for strength calculation
            return 0.5
            
        return max(0, min(1, strength_score))

    def _check_basic_ratios(self, bars: pd.DataFrame) -> bool:
        # This method is now deprecated in favor of the vectorized version,
        # but kept for reference or if a non-vectorizable check is ever needed.
        for ratio in self.ohlc_ratios:
            if not self._check_ratio(bars.iloc[-1], ratio):
                return False
        return True

    def _check_ratio(self, bar: pd.Series, ratio: OHLCRatio) -> bool:
        # Use the correct attributes
        if ratio.body_ratio is not None:
            val = abs(bar['close'] - bar['open']) / (bar['high'] - bar['low']) if (bar['high'] - bar['low']) != 0 else 0
            return val > ratio.body_ratio
        elif ratio.upper_wick_ratio is not None:
            val = (bar['high'] - max(bar['open'], bar['close'])) / (bar['high'] - bar['low']) if (bar['high'] - bar['low']) != 0 else 0
            return val > ratio.upper_wick_ratio
        elif ratio.lower_wick_ratio is not None:
            val = (min(bar['open'], bar['close']) - bar['low']) / (bar['high'] - bar['low']) if (bar['high'] - bar['low']) != 0 else 0
            return val > ratio.lower_wick_ratio
        elif ratio.custom_formula is not None:
            # Not vectorized, so skip for now
            return False
        return False

    def _evaluate_formula(self, bars: pd.DataFrame, formula: str) -> bool:
        """
        Evaluate a custom formula.
        WARNING: Uses eval(), not safe for untrusted input.
        """
        try:
            # Create a context for eval
            c = bars['close'].values
            o = bars['open'].values
            h = bars['high'].values
            l = bars['low'].values
            v = bars['volume'].values
            
            # Allow access to numpy functions
            # In a real scenario, this should be a heavily sandboxed environment.
            return eval(formula, {"__builtins__": None}, {'np': np, 'c': c, 'o': o, 'h': h, 'l': l, 'v': v})
        except Exception:
            return False


class HammerPattern(CandlestickPattern):
    """Hammer and Hanging Man pattern (robust, EngulfingPattern-style)"""
    def __init__(self, timeframes: List[TimeRange], pattern_type: str = 'both'):
        if pattern_type not in ['hammer', 'hanging_man', 'both']:
            raise ValueError("pattern_type must be 'hammer', 'hanging_man', or 'both'")
        self.pattern_type = pattern_type
        super().__init__("Hammer", timeframes)

    def get_required_bars(self) -> int:
        return 1

    def detect(self, data: pd.DataFrame) -> pd.Series:
        if not self.validate_data(data):
            return pd.Series(False, index=data.index)
        total_range = (data['high'] - data['low']).replace(0, np.nan)
        body = (data['close'] - data['open']).abs()
        upper = data['high'] - data[['open', 'close']].max(axis=1)
        lower = data[['open', 'close']].min(axis=1) - data['low']
        # Use sensitivity-controlled thresholds if present
        min_lower_wick_to_body = getattr(self, '_min_lower_wick_to_body', 2.0)
        max_upper_wick_to_body = getattr(self, '_max_upper_wick_to_body', 0.2)
        max_body_to_range = getattr(self, '_max_body_to_range', 0.35)
        # Criteria: long lower wick, small upper wick, small body
        is_hammer = (
            (lower >= min_lower_wick_to_body * body) &
            (upper <= max_upper_wick_to_body * body) &
            (body / total_range <= max_body_to_range) &
            (data['close'] > data['open'])
        )
        is_hanging_man = (
            (lower >= min_lower_wick_to_body * body) &
            (upper <= max_upper_wick_to_body * body) &
            (body / total_range <= max_body_to_range) &
            (data['close'] < data['open'])
        )
        if self.pattern_type == 'hammer':
            signals = is_hammer
        elif self.pattern_type == 'hanging_man':
            signals = is_hanging_man
        else:
            signals = is_hammer | is_hanging_man
        return signals.fillna(False)

    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        signals = self.detect(data)
        total_range = (data['high'] - data['low']).replace(0, np.nan)
        lower = data[['open', 'close']].min(axis=1) - data['low']
        body = (data['close'] - data['open']).abs()
        # Strength: ratio of lower wick to body, normalized
        with np.errstate(divide='ignore', invalid='ignore'):
            strength = (lower / (body + 1e-6)).clip(0, 5) / 5
        return strength.where(signals, 0.0).fillna(0.0)


# Pattern factory
class PatternFactory:
    """Factory for creating candlestick patterns"""
    
    @staticmethod
    def create_pattern(pattern_type: str, **kwargs) -> CandlestickPattern:
        """Create a pattern instance based on type"""
        patterns = {
            'ii_bars': IIBarsPattern,
            'double_wick': DoubleWickPattern,
            'hammer': HammerPattern,
            'engulfing': EngulfingPattern,
            'custom': CustomPattern
        }
        
        pattern_class = patterns.get(pattern_type.lower())
        if pattern_class:
            return pattern_class(**kwargs)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
