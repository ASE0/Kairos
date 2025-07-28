"""
patterns/enhanced_candlestick_patterns.py
=========================================
Enhanced candlestick patterns with custom parameters and FVG/Breaker indicators
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from enum import Enum

from core.data_structures import TimeRange
from patterns.candlestick_patterns import CandlestickPattern

# Add at the top of the file (after imports)
body_size = lambda open_, close_: np.abs(close_ - open_)
upper_wick = lambda open_, close_, high: high - np.maximum(open_, close_)
lower_wick = lambda open_, close_, low: np.minimum(open_, close_) - low

class PatternDirection(Enum):
    """Pattern direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BOTH = "both"


def get_candle_metrics(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculates candle metrics for an entire DataFrame in a vectorized way.
    """
    total_range = (data['high'] - data['low']).replace(0, np.nan)
    body_size_s = body_size(data['open'], data['close'])
    
    # Fix direction classification to match single-candle version
    # Use numpy.where for more explicit control
    direction_values = np.where(data['close'] > data['open'], PatternDirection.BULLISH, PatternDirection.BEARISH)
    # Override with NEUTRAL when body is very small (less than 5% of total range)
    direction_values = np.where(body_size_s < total_range * 0.05, PatternDirection.NEUTRAL, direction_values)
    direction = pd.Series(direction_values, index=data.index)

    return {
        'body_size': body_size_s,
        'body_ratio': (body_size_s / total_range).fillna(0),
        'upper_wick': upper_wick(data['open'], data['close'], data['high']),
        'lower_wick': lower_wick(data['open'], data['close'], data['low']),
        'upper_wick_ratio': (upper_wick(data['open'], data['close'], data['high']) / total_range).fillna(0),
        'lower_wick_ratio': (lower_wick(data['open'], data['close'], data['low']) / total_range).fillna(0),
        'total_range': total_range.fillna(0),
        'direction': direction
    }


@dataclass
class CandleMetrics:
    """Metrics for a single candle"""
    body_size: float
    body_ratio: float
    upper_wick: float
    lower_wick: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    total_range: float
    direction: PatternDirection
    
    @classmethod
    def from_ohlc(cls, open_price: float, high: float, low: float, close: float) -> 'CandleMetrics':
        """Calculate metrics from OHLC"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return cls(
                body_size=0, body_ratio=0, upper_wick=0, lower_wick=0,
                upper_wick_ratio=0, lower_wick_ratio=0, total_range=0,
                direction=PatternDirection.NEUTRAL
            )
            
        body_ratio = body_size / total_range
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        direction = PatternDirection.BULLISH if close > open_price else PatternDirection.BEARISH
        if body_size < total_range * 0.05:  # Very small body
            direction = PatternDirection.NEUTRAL
            
        return cls(
            body_size=body_size,
            body_ratio=body_ratio,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            total_range=total_range,
            direction=direction
        )


@dataclass
class PatternParameters:
    """Parameters for custom pattern definition"""
    # Body parameters
    min_body_ratio: Optional[float] = None
    max_body_ratio: Optional[float] = None
    body_to_wick_ratio: Optional[Tuple[float, float]] = None  # (min, max)
    
    # Wick parameters
    upper_wick_to_body_ratio: Optional[Tuple[float, float]] = None
    lower_wick_to_body_ratio: Optional[Tuple[float, float]] = None
    wick_symmetry_tolerance: Optional[float] = None  # 0-1, how similar wicks should be
    
    # Direction
    allowed_directions: List[PatternDirection] = field(default_factory=lambda: [PatternDirection.BOTH])
    
    # Multi-bar parameters
    consecutive_bars: int = 1
    trend_requirement: Optional[str] = None  # 'higher_highs', 'lower_lows', etc.
    
    # Price change parameters
    min_price_change: Optional[float] = None  # Minimum % change after pattern
    max_price_change: Optional[float] = None  # Maximum % change after pattern
    measure_bars: int = 5  # How many bars to measure price change
    
    def get_match_signals(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """
        Check if candle metrics match parameters in a vectorized way.
        Returns a boolean Series of matches.
        """
        signals = pd.Series(True, index=metrics['body_size'].index)

        # Vectorized checks
        if self.min_body_ratio is not None:
            signals &= metrics['body_ratio'] >= self.min_body_ratio
        if self.max_body_ratio is not None:
            signals &= metrics['body_ratio'] <= self.max_body_ratio
            
        if self.body_to_wick_ratio:
            total_wick = metrics['upper_wick'] + metrics['lower_wick']
            ratio = (metrics['body_size'] / total_wick.replace(0, np.nan)).fillna(0)
            signals &= (ratio >= self.body_to_wick_ratio[0]) & (ratio <= self.body_to_wick_ratio[1])
                    
        if self.upper_wick_to_body_ratio:
            ratio = (metrics['upper_wick'] / metrics['body_size'].replace(0, np.nan)).fillna(0)
            signals &= (ratio >= self.upper_wick_to_body_ratio[0]) & (ratio <= self.upper_wick_to_body_ratio[1])
                
        if self.lower_wick_to_body_ratio:
            ratio = (metrics['lower_wick'] / metrics['body_size'].replace(0, np.nan)).fillna(0)
            signals &= (ratio >= self.lower_wick_to_body_ratio[0]) & (ratio <= self.lower_wick_to_body_ratio[1])
                
        if self.wick_symmetry_tolerance is not None:
            wick_diff = abs(metrics['upper_wick'] - metrics['lower_wick']) / metrics['total_range'].replace(0, np.nan)
            signals &= wick_diff.fillna(0) <= self.wick_symmetry_tolerance
                    
        if PatternDirection.BOTH not in self.allowed_directions:
            signals &= metrics['direction'].isin(self.allowed_directions)
                
        return signals


class CustomParametricPattern:
    """Custom pattern with parameters"""
    
    def __init__(self, name: str, parameters: PatternParameters):
        self.name = name
        self.parameters = parameters
        self.pattern_occurrences = []
        self.price_changes = []
        
    def detect(self, data: pd.DataFrame) -> pd.Series:
        """Detect pattern in data using vectorized operations."""
        # Always use RangeIndex for all index math
        df = data.reset_index(drop=True)
        
        # 1. Calculate all candle metrics at once
        metrics = get_candle_metrics(df)
        
        # 2. Get initial matches based on single-candle parameters
        signals = self.parameters.get_match_signals(metrics)
        
        # 3. Apply multi-bar conditions
        if self.parameters.consecutive_bars > 1:
            signals = signals.rolling(window=self.parameters.consecutive_bars).all()

        if self.parameters.trend_requirement:
            trend_signals = self._check_trend_vectorized(df)
            signals &= trend_signals
            
        final_signals = signals.fillna(False)

        # 4. Track occurrences and price changes (this part remains a loop but only on positive signals)
        self.pattern_occurrences = final_signals[final_signals].index.tolist()
        
        if self.parameters.measure_bars > 0 and final_signals.any():
            entry_prices = df['close'][final_signals]
            future_indices = entry_prices.index.to_series() + self.parameters.measure_bars
            valid_future_indices = future_indices[future_indices < len(df)]
            if not valid_future_indices.empty:
                future_prices = df['close'].iloc[valid_future_indices.values]
                # Align indices for calculation using iloc
                entry_prices_aligned = entry_prices.iloc[valid_future_indices.index - self.parameters.measure_bars]
                
                price_changes_s = (future_prices.values - entry_prices_aligned.values) / entry_prices_aligned.values
                self.price_changes = price_changes_s.tolist()

        return final_signals
        
    def _check_trend_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Check trend requirement using vectorized rolling operations."""
        window = self.parameters.consecutive_bars
        if self.parameters.trend_requirement == 'higher_highs':
            return (data['high'] > data['high'].shift(1)).rolling(window=window).all()
        elif self.parameters.trend_requirement == 'lower_lows':
            return (data['low'] < data['low'].shift(1)).rolling(window=window).all()
        elif self.parameters.trend_requirement == 'ascending':
            return (data['close'] > data['close'].shift(1)).rolling(window=window).all()
        elif self.parameters.trend_requirement == 'descending':
            return (data['close'] < data['close'].shift(1)).rolling(window=window).all()
        return pd.Series(True, index=data.index)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        if not self.price_changes:
            return {
                'occurrences': len(self.pattern_occurrences),
                'avg_price_change': 0,
                'std_price_change': 0,
                'win_rate': 0,
                'avg_positive_change': 0,
                'avg_negative_change': 0
            }
        stats = {
            'occurrences': len(self.pattern_occurrences),
            'avg_price_change': np.mean(self.price_changes),
            'std_price_change': np.std(self.price_changes),
            'win_rate': len([pc for pc in self.price_changes if pc > 0]) / len(self.price_changes),
            'avg_positive_change': np.mean([pc for pc in self.price_changes if pc > 0]),
            'avg_negative_change': np.mean([pc for pc in self.price_changes if pc < 0])
        }
        return stats


# Predefined patterns with parameters
class PredefinedPatterns:
    """Collection of predefined patterns"""
    
    @staticmethod
    def hammer() -> PatternParameters:
        """Hammer pattern parameters"""
        return PatternParameters(
            min_body_ratio=0.1,
            max_body_ratio=0.35,
            lower_wick_to_body_ratio=(2.0, 10.0),
            upper_wick_to_body_ratio=(0.0, 0.5),
            allowed_directions=[PatternDirection.BULLISH, PatternDirection.BEARISH]
        )
        
    @staticmethod
    def shooting_star() -> PatternParameters:
        """Shooting star pattern parameters"""
        return PatternParameters(
            min_body_ratio=0.1,
            max_body_ratio=0.35,
            upper_wick_to_body_ratio=(2.0, 10.0),
            lower_wick_to_body_ratio=(0.0, 0.5),
            allowed_directions=[PatternDirection.BULLISH, PatternDirection.BEARISH]
        )
        
    @staticmethod
    def doji() -> PatternParameters:
        """Doji pattern parameters"""
        return PatternParameters(
            max_body_ratio=0.1,
            wick_symmetry_tolerance=0.3,
            allowed_directions=[PatternDirection.NEUTRAL]
        )
        
    @staticmethod
    def marubozu() -> PatternParameters:
        """Marubozu (no wick) pattern parameters"""
        return PatternParameters(
            min_body_ratio=0.95,
            upper_wick_to_body_ratio=(0.0, 0.05),
            lower_wick_to_body_ratio=(0.0, 0.05),
            allowed_directions=[PatternDirection.BULLISH, PatternDirection.BEARISH]
        )
        
    @staticmethod
    def spinning_top() -> PatternParameters:
        """Spinning top pattern parameters"""
        return PatternParameters(
            min_body_ratio=0.1,
            max_body_ratio=0.35,
            wick_symmetry_tolerance=0.3
        )
        
    @staticmethod
    def long_body() -> PatternParameters:
        """Long body candle parameters"""
        return PatternParameters(
            min_body_ratio=0.7,
            allowed_directions=[PatternDirection.BULLISH, PatternDirection.BEARISH]
        )


@dataclass
class FVGLevel:
    """Fair Value Gap level"""
    timestamp: pd.Timestamp
    high: float
    low: float
    midpoint: float
    direction: PatternDirection
    touches: int = 0
    respected: bool = True
    broken: bool = False
    strength: float = 1.0  # Weakens with touches
    

class FVGIndicator:
    """Fair Value Gap indicator"""
    
    def __init__(self, min_gap_size: float = 0.001, max_touches: int = 3):
        self.min_gap_size = min_gap_size
        self.max_touches = max_touches
        self.active_fvgs: List[FVGLevel] = []
        self.historical_fvgs: List[FVGLevel] = []
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, List[FVGLevel]]:
        """Calculate FVG levels"""
        self.active_fvgs = []
        self.historical_fvgs = []
        
        for i in range(2, len(data)):
            # Get three consecutive candles
            candle1 = data.iloc[i-2]
            candle2 = data.iloc[i-1]
            candle3 = data.iloc[i]
            gap_size = (candle3['low'] - candle1['high']) / candle1['high']
            print(f"[FVG GAP CHECK] Bar {i}: candle1_high={candle1['high']}, candle3_low={candle3['low']}, gap_size={gap_size:.6f}")
            # DEBUG: Print candle info for synthetic test data
            if len(data) <= 20:  # Only for small test datasets
                print(f"[FVG DEBUG] Bar {i}: Candle1 (i-2): H={candle1['high']:.2f}, L={candle1['low']:.2f}")
                print(f"[FVG DEBUG] Bar {i}: Candle2 (i-1): H={candle2['high']:.2f}, L={candle2['low']:.2f}")
                print(f"[FVG DEBUG] Bar {i}: Candle3 (i): H={candle3['high']:.2f}, L={candle3['low']:.2f}")
            
            # Check for bullish FVG (gap up)
            if candle1['high'] < candle3['low']:
                gap_size = (candle3['low'] - candle1['high']) / candle1['high']
                if gap_size >= self.min_gap_size:
                    print(f"[FVG DETECTED] Bullish FVG at bar {i}: gap_size={gap_size:.4f}, high[i-2]={candle1['high']}, low[i]={candle3['low']}")
                    fvg = FVGLevel(
                        timestamp=i,  # PATCH: use integer index for timestamp
                        high=candle3['low'],
                        low=candle1['high'],
                        midpoint=(candle3['low'] + candle1['high']) / 2,
                        direction=PatternDirection.BULLISH
                    )
                    self.active_fvgs.append(fvg)
                    print(f"[FVG DETECTED] Appended: {fvg}")
            
            # Check for bearish FVG (gap down)
            elif candle1['low'] > candle3['high']:
                gap_size = (candle1['low'] - candle3['high']) / candle3['high']
                if gap_size >= self.min_gap_size:
                    if len(data) <= 20:  # Only for small test datasets
                        print(f"[FVG DEBUG] Bearish FVG detected at bar {i}: gap_size={gap_size:.4f}")
                    fvg = FVGLevel(
                        timestamp=i,  # PATCH: use integer index for timestamp
                        high=candle1['low'],
                        low=candle3['high'],
                        midpoint=(candle1['low'] + candle3['high']) / 2,
                        direction=PatternDirection.BEARISH
                    )
                    self.active_fvgs.append(fvg)
                    if len(data) <= 20:  # Only for small test datasets
                        print(f"[FVG DEBUG] Added bearish FVG: {fvg}")
            elif len(data) <= 20:  # Only for small test datasets
                print(f"[FVG DEBUG] No FVG at bar {i}: candle1_high={candle1['high']:.2f}, candle3_low={candle3['low']:.2f}, candle1_low={candle1['low']:.2f}, candle3_high={candle3['high']:.2f}")
            
        # Process FVGs with price action
        self._process_fvg_touches(data)
        
        return {
            'active': self.active_fvgs,
            'historical': self.historical_fvgs
        }
        
    def _process_fvg_touches(self, data: pd.DataFrame):
        """Process FVG touches and breaks"""
        for fvg in self.active_fvgs[:]:  # Copy list to modify during iteration
            fvg_idx = data.index.get_loc(fvg.timestamp)
            
            for i in range(fvg_idx + 1, len(data)):
                bar = data.iloc[i]
                
                # Check if price touches FVG
                if self._touches_fvg(bar, fvg):
                    fvg.touches += 1
                    fvg.strength *= 0.8  # Weaken with each touch
                    
                    # Check if closed within FVG
                    if bar['close'] > fvg.low and bar['close'] < fvg.high:
                        # Price closed inside FVG
                        if abs(bar['close'] - fvg.midpoint) / (fvg.high - fvg.low) < 0.2:
                            # Strong rejection at midpoint
                            fvg.respected = True
                            
                # Check if FVG is broken
                if self._breaks_fvg(bar, fvg):
                    fvg.broken = True
                    fvg.respected = False
                    self.active_fvgs.remove(fvg)
                    self.historical_fvgs.append(fvg)
                    break
                    
                # Remove if touched too many times
                if fvg.touches >= self.max_touches and not fvg.broken:
                    fvg.respected = False
                    self.active_fvgs.remove(fvg)
                    self.historical_fvgs.append(fvg)
                    break
                    
    def _touches_fvg(self, bar: pd.Series, fvg: FVGLevel) -> bool:
        """Check if bar touches FVG"""
        return (bar['high'] >= fvg.low and bar['low'] <= fvg.high)
        
    def _breaks_fvg(self, bar: pd.Series, fvg: FVGLevel) -> bool:
        """Check if bar breaks through FVG"""
        if fvg.direction == PatternDirection.BULLISH:
            # Bullish FVG broken if close below it
            return bar['close'] < fvg.low
        else:
            # Bearish FVG broken if close above it
            return bar['close'] > fvg.high
            
    def get_nearest_fvg(self, price: float, direction: Optional[PatternDirection] = None) -> Optional[FVGLevel]:
        """Get nearest active FVG to price"""
        candidates = self.active_fvgs
        if direction:
            candidates = [fvg for fvg in candidates if fvg.direction == direction]
            
        if not candidates:
            return None
            
        # Find nearest by midpoint
        nearest = min(candidates, key=lambda fvg: abs(fvg.midpoint - price))
        return nearest


@dataclass
class BreakerBlock:
    """Breaker block level"""
    timestamp: pd.Timestamp
    high: float
    low: float
    direction: PatternDirection  # Direction of the break
    origin_type: str  # 'swing_high', 'swing_low', 'order_block'
    strength: float = 1.0
    confirmed: bool = False
    broken: bool = False
    

class BreakerIndicator:
    """Breaker block indicator"""
    
    def __init__(self, lookback: int = 20, min_swing_strength: float = 0.001):
        self.lookback = lookback
        self.min_swing_strength = min_swing_strength
        self.active_breakers: List[BreakerBlock] = []
        self.historical_breakers: List[BreakerBlock] = []
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, List[BreakerBlock]]:
        """Calculate breaker blocks"""
        self.active_breakers = []
        self.historical_breakers = []
        
        # Find swing points
        swings = self._find_swing_points(data)
        
        # Process each swing point
        for i, (swing_idx, swing_type) in enumerate(swings):
            if swing_idx + self.lookback >= len(data):
                continue
                
            # Look for break of structure
            if self._check_break_of_structure(data, swing_idx, swing_type):
                # Create breaker block
                swing_bar = data.iloc[swing_idx]
                
                if swing_type == 'high':
                    # Bearish breaker (broken swing high)
                    breaker = BreakerBlock(
                        timestamp=data.index[swing_idx],
                        high=swing_bar['high'],
                        low=swing_bar['low'],
                        direction=PatternDirection.BEARISH,
                        origin_type='swing_high'
                    )
                else:
                    # Bullish breaker (broken swing low)
                    breaker = BreakerBlock(
                        timestamp=data.index[swing_idx],
                        high=swing_bar['high'],
                        low=swing_bar['low'],
                        direction=PatternDirection.BULLISH,
                        origin_type='swing_low'
                    )
                    
                self.active_breakers.append(breaker)
                
        # Process breaker confirmations and breaks
        self._process_breaker_reactions(data)
        
        return {
            'active': self.active_breakers,
            'historical': self.historical_breakers
        }
        
    def _find_swing_points(self, data: pd.DataFrame) -> List[Tuple[int, str]]:
        """Find swing highs and lows"""
        swings = []
        
        for i in range(self.lookback, len(data) - self.lookback):
            # Check for swing high
            if self._is_swing_high(data, i):
                swings.append((i, 'high'))
                
            # Check for swing low
            if self._is_swing_low(data, i):
                swings.append((i, 'low'))
                
        return swings
        
    def _is_swing_high(self, data: pd.DataFrame, idx: int) -> bool:
        """Check if index is a swing high"""
        high = data.iloc[idx]['high']
        
        # Check left side
        for i in range(idx - self.lookback, idx):
            if data.iloc[i]['high'] >= high:
                return False
                
        # Check right side
        for i in range(idx + 1, min(idx + self.lookback + 1, len(data))):
            if data.iloc[i]['high'] >= high:
                return False
                
        return True
        
    def _is_swing_low(self, data: pd.DataFrame, idx: int) -> bool:
        """Check if index is a swing low"""
        low = data.iloc[idx]['low']
        
        # Check left side
        for i in range(idx - self.lookback, idx):
            if data.iloc[i]['low'] <= low:
                return False
                
        # Check right side
        for i in range(idx + 1, min(idx + self.lookback + 1, len(data))):
            if data.iloc[i]['low'] <= low:
                return False
                
        return True
        
    def _check_break_of_structure(self, data: pd.DataFrame, swing_idx: int, swing_type: str) -> bool:
        """Check if structure is broken after swing point"""
        swing_price = data.iloc[swing_idx]['high' if swing_type == 'high' else 'low']
        
        for i in range(swing_idx + 1, min(swing_idx + self.lookback * 2, len(data))):
            if swing_type == 'high' and data.iloc[i]['close'] > swing_price:
                return True
            elif swing_type == 'low' and data.iloc[i]['close'] < swing_price:
                return True
                
        return False
        
    def _process_breaker_reactions(self, data: pd.DataFrame):
        """Process breaker confirmations and breaks"""
        for breaker in self.active_breakers[:]:
            breaker_idx = data.index.get_loc(breaker.timestamp)
            
            for i in range(breaker_idx + 1, len(data)):
                bar = data.iloc[i]
                
                # Check if price returns to breaker
                if self._touches_breaker(bar, breaker):
                    # Check for confirmation (rejection)
                    if breaker.direction == PatternDirection.BULLISH:
                        if bar['close'] > breaker.high:
                            breaker.confirmed = True
                            breaker.strength = 1.5  # Strengthen on confirmation
                    else:
                        if bar['close'] < breaker.low:
                            breaker.confirmed = True
                            breaker.strength = 1.5
                            
                # Check if breaker is broken
                if self._breaks_breaker(bar, breaker):
                    breaker.broken = True
                    self.active_breakers.remove(breaker)
                    self.historical_breakers.append(breaker)
                    break
                    
    def _touches_breaker(self, bar: pd.Series, breaker: BreakerBlock) -> bool:
        """Check if bar touches breaker block"""
        return (bar['high'] >= breaker.low and bar['low'] <= breaker.high)
        
    def _breaks_breaker(self, bar: pd.Series, breaker: BreakerBlock) -> bool:
        """Check if bar breaks breaker block"""
        if breaker.direction == PatternDirection.BULLISH:
            # Bullish breaker broken if close below it
            return bar['close'] < breaker.low * 0.999  # Small buffer
        else:
            # Bearish breaker broken if close above it
            return bar['close'] > breaker.high * 1.001  # Small buffer


class FVGPattern(CandlestickPattern):
    """Fair Value Gap pattern for use in GUI and backtest engine"""
    def __init__(self, timeframes=None, min_gap_size=0.001, max_touches=3):
        if timeframes is None:
            timeframes = [TimeRange(1, 'm')]
        super().__init__("FVG", timeframes)
        self.indicator = FVGIndicator(min_gap_size=min_gap_size, max_touches=max_touches)
        self.name = "FVG"

    def get_required_bars(self):
        return 3

    def detect(self, data: pd.DataFrame) -> pd.Series:
        # Always use RangeIndex for all index math
        df = data.reset_index(drop=True)
        # Mark True at the timestamp of each detected FVG
        fvg_dict = self.indicator.calculate(df)
        active = fvg_dict.get('active', [])
        signals = pd.Series(False, index=df.index)
        for fvg in active:
            if fvg.timestamp in signals.index:
                signals.loc[fvg.timestamp] = True
        return signals

    def detect_zones(self, data: pd.DataFrame) -> list:
        # Always use RangeIndex for all index math
        df = data.reset_index(drop=True)
        # Use the full DataFrame to detect all FVGs
        fvg_dict = self.indicator.calculate(df)
        active = fvg_dict.get('active', [])
        detected_zones = []
        for fvg in active:
            if hasattr(fvg, 'timestamp'):
                i = fvg.timestamp if isinstance(fvg.timestamp, int) else df.index.get_loc(fvg.timestamp)
                zone = {
                    'timestamp': fvg.timestamp,
                    'bar_index': i,
                    'zone_min': min(fvg.low, fvg.high),
                    'zone_max': max(fvg.low, fvg.high),
                    'zone_type': 'FVG',
                    'zone_direction': getattr(fvg, 'direction', None).value if hasattr(fvg, 'direction') else None
                }
                detected_zones.append(zone)
        return detected_zones

    def get_strength(self, data: pd.DataFrame) -> pd.Series:
        # Return the strength for each detected FVG
        df = data.reset_index(drop=True)
        fvg_dict = self.indicator.calculate(df)
        active = fvg_dict.get('active', [])
        strength = pd.Series(0.0, index=df.index)
        for fvg in active:
            if fvg.timestamp in strength.index:
                strength.loc[fvg.timestamp] = fvg.strength
        return strength
