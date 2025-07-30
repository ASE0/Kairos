#!/usr/bin/env python3
"""
Data Factory for Pattern Testing
===============================
Generates synthetic OHLCV datasets that trigger specific patterns exactly once.
Based on the mathematical framework from the comprehensive trading strategy docs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import os


def create_dataset(pattern_name: str, params: Dict[str, Any] = None) -> str:
    """
    Create a synthetic dataset that MUST trigger the given pattern/gate/filter exactly once.
    For support_resistance_band: first strict new high at index 60.
    For vwap_mean_reversion_band: first close outside VWAP ± k·sigma at index 20.
    For all other patterns, use the previous pattern-specific generator logic.
    """
    if params is None:
        params = {}
    base_price = 18250.0
    base_time = datetime(2024, 3, 7, 16, 15, 0)
    bars = []
    if pattern_name == 'support_resistance_band':
        # 60 bars with same high/low, then bar 60 is a new high
        for i in range(60):
            dt = base_time + pd.Timedelta(minutes=i)
            bars.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': base_price,
                'high': base_price + 10,
                'low': base_price - 10,
                'close': base_price,
                'volume': 1000
            })
        # Bar 60: new high
        dt = base_time + pd.Timedelta(minutes=60)
        bars.append({
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'open': base_price,
            'high': base_price + 20,  # new high
            'low': base_price - 10,
            'close': base_price,
            'volume': 1000
        })
        # Filler bars
        for i in range(61, 70):
            dt = base_time + pd.Timedelta(minutes=i)
            bars.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': base_price,
                'high': base_price + 10,
                'low': base_price - 10,
                'close': base_price,
                'volume': 1000
            })
    elif pattern_name == 'vwap_mean_reversion_band':
        # 20 bars at VWAP, then bar 20 is a deviation
        vwap = base_price
        for i in range(20):
            dt = base_time + pd.Timedelta(minutes=i)
            bars.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': vwap,
                'high': vwap + 5,
                'low': vwap - 5,
                'close': vwap,
                'volume': 1000
            })
        # Bar 20: close outside VWAP band
        dt = base_time + pd.Timedelta(minutes=20)
        bars.append({
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'open': vwap,
            'high': vwap + 5,
            'low': vwap - 5,
            'close': vwap + 20,  # deviation
            'volume': 1000
        })
        # Filler bars
        for i in range(21, 30):
            dt = base_time + pd.Timedelta(minutes=i)
            bars.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': vwap,
                'high': vwap + 5,
                'low': vwap - 5,
                'close': vwap,
                'volume': 1000
            })
    elif pattern_name == 'imbalance_memory_zone':
        bars.extend(_generate_imbalance_memory_zone_bars(base_price, params))
    elif pattern_name == 'fvg':
        bars.extend(_generate_fvg_bars(base_price, params))
    elif pattern_name == 'engulfing':
        bars.extend(_generate_engulfing_bars(base_price, params))
    elif pattern_name == 'hammer':
        bars.extend(_generate_hammer_bars(base_price, params))
    elif pattern_name == 'double_wick':
        bars.extend(_generate_double_wick_bars(base_price, params))
    elif pattern_name == 'ii_bars':
        bars.extend(_generate_ii_bars_bars(base_price, params))
    elif pattern_name == 'doji':
        bars.extend(_generate_doji_bars(base_price, params))
    elif pattern_name == 'marubozu':
        bars.extend(_generate_marubozu_bars(base_price, params))
    elif pattern_name == 'spinning_top':
        bars.extend(_generate_spinning_top_bars(base_price, params))
    elif pattern_name == 'weak_body':
        bars.extend(_generate_weak_body_bars(base_price, params))
    elif pattern_name == 'strong_body':
        bars.extend(_generate_strong_body_bars(base_price, params))
    elif pattern_name == 'engulfing_bullish':
        bars.extend(_generate_engulfing_bars(base_price, {'pattern_type': 'bullish'}))
    elif pattern_name == 'engulfing_bearish':
        bars.extend(_generate_engulfing_bars(base_price, {'pattern_type': 'bearish'}))
    elif pattern_name == 'doji_standard':
        bars.extend(_generate_doji_bars(base_price, {'standard': True}))
    elif pattern_name == 'momentum_breakout':
        bars.extend(_generate_momentum_breakout_bars(base_price, params))
    elif pattern_name == 'momentum_reversal':
        bars.extend(_generate_momentum_reversal_bars(base_price, params))
    elif pattern_name == 'high_volatility':
        bars.extend(_generate_high_volatility_bars(base_price, params))
    elif pattern_name == 'low_volatility':
        bars.extend(_generate_low_volatility_bars(base_price, params))
    elif pattern_name == 'support_bounce':
        bars.extend(_generate_support_bounce_bars(base_price, params))
    elif pattern_name == 'resistance_rejection':
        bars.extend(_generate_resistance_rejection_bars(base_price, params))
    elif pattern_name == 'three_white_soldiers':
        bars.extend(_generate_three_white_soldiers_bars(base_price, params))
    elif pattern_name == 'three_black_crows':
        bars.extend(_generate_three_black_crows_bars(base_price, params))
    elif pattern_name == 'four_price_doji':
        bars.extend(_generate_four_price_doji_bars(base_price, params))
    elif pattern_name == 'dragonfly_doji':
        bars.extend(_generate_dragonfly_doji_bars(base_price, params))
    elif pattern_name == 'gravestone_doji':
        bars.extend(_generate_gravestone_doji_bars(base_price, params))
    elif pattern_name == 'volatility_expansion':
        bars.extend(_generate_volatility_expansion_bars(base_price, params))
    elif pattern_name == 'volatility_contraction':
        bars.extend(_generate_volatility_contraction_bars(base_price, params))
    elif pattern_name == 'trend_continuation':
        bars.extend(_generate_trend_continuation_bars(base_price, params))
    elif pattern_name == 'trend_reversal':
        bars.extend(_generate_trend_reversal_bars(base_price, params))
    elif pattern_name == 'gap_up':
        bars.extend(_generate_gap_up_bars(base_price, params))
    elif pattern_name == 'gap_down':
        bars.extend(_generate_gap_down_bars(base_price, params))
    elif pattern_name == 'consolidation':
        bars.extend(_generate_consolidation_bars(base_price, params))
    elif pattern_name == 'breakout':
        bars.extend(_generate_breakout_bars(base_price, params))
    elif pattern_name == 'exhaustion':
        bars.extend(_generate_exhaustion_bars(base_price, params))
    elif pattern_name == 'accumulation':
        bars.extend(_generate_accumulation_bars(base_price, params))
    elif pattern_name == 'distribution':
        bars.extend(_generate_distribution_bars(base_price, params))
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    df = pd.DataFrame(bars)
    filename = f"test_{pattern_name}_data.csv"
    df.to_csv(filename, index=False)
    return filename


def create_gate_dataset(gate_name: str, params: Dict[str, Any] = None) -> str:
    """
    Create a synthetic dataset for testing a specific gate.
    
    Args:
        gate_name: Name of the gate to test
        params: Gate parameters
        
    Returns:
        Path to the created CSV file
    """
    if params is None:
        params = {}
    
    base_price = 18250.0
    bars = []
    base_time = datetime(2024, 3, 7, 16, 15, 0)
    
    # Generate filler bars before the gate condition (5 bars)
    bars.extend(_generate_filler_bars(base_price, 5, gate_name, base_time))
    
    # Generate the gate condition at index 5 (6th bar, 0-based)
    supported_gates = [
        'location_gate', 'volatility_gate', 'regime_gate', 'bayesian_gate',
        'fvg_gate', 'momentum_gate', 'volume_gate',
        'time_gate', 'correlation_gate'
    ]
    
    if gate_name not in supported_gates:
        raise NotImplementedError(f"No dataset generator for gate: {gate_name}")

    # Generate gate-specific bars
    if gate_name == 'location_gate':
        bars.extend(_generate_location_gate_bars(base_price, params))
    elif gate_name == 'volatility_gate':
        bars.extend(_generate_volatility_gate_bars(base_price, params))
    elif gate_name == 'regime_gate':
        bars.extend(_generate_regime_gate_bars(base_price, params))
    elif gate_name == 'bayesian_gate':
        bars.extend(_generate_bayesian_gate_bars(base_price, params))
    elif gate_name == 'fvg_gate':
        bars.extend(_generate_fvg_gate_bars(base_price, params))
    elif gate_name == 'momentum_gate':
        bars.extend(_generate_momentum_gate_bars(base_price, params))
    elif gate_name == 'volume_gate':
        bars.extend(_generate_volume_gate_bars(base_price, params))
    elif gate_name == 'time_gate':
        bars.extend(_generate_time_gate_bars(base_price, params))
    elif gate_name == 'correlation_gate':
        bars.extend(_generate_correlation_gate_bars(base_price, params))
    else:
        raise ValueError(f"Unknown gate: {gate_name}")
    
    # Generate filler bars after the gate condition (5 bars)
    after_time = base_time + timedelta(minutes=5) + timedelta(minutes=1)
    bars.extend(_generate_filler_bars(base_price + 10, 5, gate_name, after_time))
    
    # Create DataFrame
    df = pd.DataFrame(bars)
    
    # Save to CSV
    filename = f"test_{gate_name}_data.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath, index=False)
    
    return filepath


def create_filter_dataset(filter_name: str, params: Dict[str, Any] = None) -> str:
    """
    Create a synthetic dataset for testing a specific filter.
    
    Args:
        filter_name: Name of the filter to test
        params: Filter parameters
        
    Returns:
        Path to the created CSV file
    """
    if params is None:
        params = {}
    
    base_price = 18250.0
    bars = []
    base_time = datetime(2024, 3, 7, 16, 15, 0)
    
    # Generate filler bars before the filter condition (5 bars)
    bars.extend(_generate_filler_bars(base_price, 5, filter_name, base_time))
    
    # Generate the filter condition at index 5 (6th bar, 0-based)
    supported_filters = [
        'volume', 'time', 'volatility', 'momentum', 'price', 'regime', 'advanced'
    ]
    
    if filter_name not in supported_filters:
        raise NotImplementedError(f"No dataset generator for filter: {filter_name}")

    # Generate filter-specific bars
    if filter_name == 'volume':
        bars.extend(_generate_volume_filter_bars(base_price, params))
    elif filter_name == 'time':
        bars.extend(_generate_time_filter_bars(base_price, params))
    elif filter_name == 'volatility':
        bars.extend(_generate_volatility_filter_bars(base_price, params))
    elif filter_name == 'momentum':
        bars.extend(_generate_momentum_filter_bars(base_price, params))
    elif filter_name == 'price':
        bars.extend(_generate_price_filter_bars(base_price, params))
    elif filter_name == 'regime':
        bars.extend(_generate_regime_filter_bars(base_price, params))
    elif filter_name == 'advanced':
        bars.extend(_generate_advanced_filter_bars(base_price, params))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    # Generate filler bars after the filter condition (5 bars)
    after_time = base_time + timedelta(minutes=5) + timedelta(minutes=1)
    bars.extend(_generate_filler_bars(base_price + 10, 5, filter_name, after_time))
    
    # Create DataFrame
    df = pd.DataFrame(bars)
    
    # Save to CSV
    filename = f"test_{filter_name}_filter_data.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath, index=False)
    
    return filepath


def create_negative_dataset(pattern_name: str, params: Dict[str, Any] = None) -> str:
    """
    Create a synthetic dataset that should NOT trigger the given pattern/gate/filter.
    For most, this is just a sequence of filler bars that never satisfy the condition.
    For support_resistance_band: all highs/lows are equal (no new high/low possible).
    For vwap_mean_reversion_band: all closes are at VWAP (no deviation).
    """
    if params is None:
        params = {}
    base_price = 18250.0
    base_time = datetime(2024, 3, 7, 16, 15, 0)
    bars = []
    # Support/Resistance band removed
    if pattern_name == 'vwap_mean_reversion_band':
        # All closes at VWAP, no deviation
        vwap = base_price
        for i in range(30):
            dt = base_time + pd.Timedelta(minutes=i)
            bars.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': vwap,
                'high': vwap + 5,
                'low': vwap - 5,
                'close': vwap,  # exactly at VWAP
                'volume': 1000
            })
    elif pattern_name == 'imbalance_memory_zone':
        # All closes are the same, no large price move
        for i in range(30):
            dt = base_time + pd.Timedelta(minutes=i)
            bars.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': base_price,
                'high': base_price + 2,
                'low': base_price - 2,
                'close': base_price,
                'volume': 1000
            })
    else:
        bars = _generate_filler_bars(base_price, 30, pattern_name, base_time)
    df = pd.DataFrame(bars)
    filename = f"test_{pattern_name}_negative_data.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath, index=False)
    return filepath


def _generate_fvg_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Fair Value Gap (FVG) pattern.
    Ensures H_{t-1} < L_{t+1} for a valid bullish FVG.
    The FVG will be detected at index 6 (7th bar, 0-based).
    """
    bars = []
    # 5 filler bars
    for i in range(5):
        bars.append({
            'datetime': datetime(2024, 3, 7, 16, 15, 0) + pd.Timedelta(minutes=i),
            'open': base_price,
            'high': base_price + 1.0,
            'low': base_price - 1.0,
            'close': base_price + 0.5,
            'volume': 1000
        })
    # Bar 5: H_{t-1} (gap reference) - High = base_price + 1.0
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.0,  # H_{t-1} = 18251.0
        'low': base_price + 0.8,
        'close': base_price + 1.2,
        'volume': 1500
    })
    # Bar 6: Filler (gap bar) with low = base_price + 25.0
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 21, 0),
        'open': base_price + 1.2,
        'high': base_price + 1.4,
        'low': base_price + 25.0,  # L_{t+1} = 18275.0 > H_{t-1} = 18251.0
        'close': base_price + 25.2,
        'volume': 1500
    })
    # Bar 7: Filler
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 22, 0),
        'open': base_price + 2.0,
        'high': base_price + 2.5,
        'low': base_price + 1.7,
        'close': base_price + 2.2,
        'volume': 1500
    })
    return bars


def _generate_engulfing_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create an Engulfing pattern.
    
    Mathematical rules from docs:
    - Bullish Engulfing: B₂ > B₁ and C₂ > O₂ and O₂ < C₁ and C₂ > O₁
    - Bearish Engulfing: B₂ > B₁ and C₂ < O₂ and O₂ > C₁ and C₂ < O₁
    - Two-bar reversal pattern strength: A₂bar = β_eng ⋅ (B₂ / B₁)
    """
    bars = []
    pattern_type = params.get('pattern_type', 'both')
    
    # Bar 5: First bar (smaller body)
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.5,
        'low': base_price + 0.5,
        'close': base_price + 1.2,  # Small body
        'volume': 1500
    })
    
    # Bar 6: Second bar (larger body that engulfs)
    if pattern_type == 'bullish':
        # Bullish engulfing
    bars.append({
            'datetime': datetime(2024, 3, 7, 16, 21, 0),
            'open': base_price + 0.8,  # Open below previous close
            'high': base_price + 3.0,  # High above previous open
            'low': base_price + 0.5,   # Low below previous open
            'close': base_price + 2.8,  # Close above previous open
            'volume': 1500
        })
    else:
        # Bearish engulfing
    bars.append({
            'datetime': datetime(2024, 3, 7, 16, 21, 0),
            'open': base_price + 2.5,  # Open above previous close
            'high': base_price + 2.8,  # High above previous open
            'low': base_price + 0.2,   # Low below previous open
            'close': base_price + 0.5,  # Close below previous open
            'volume': 1500
    })
    
    return bars


def _generate_hammer_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Hammer pattern.
    
    Mathematical rules from docs:
    - Body ratio: Bₜ / (Hₜ - Lₜ) ≤ 0.4
    - Lower wick ratio: Wˡₜ / (Hₜ - Lₜ) ≥ 0.6
    - Upper wick ratio: Wᵘₜ / (Hₜ - Lₜ) ≤ 0.1
    """
    bars = []
    
    # Bar 5: Hammer pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.1,  # Small upper wick
        'low': base_price + 0.2,   # Long lower wick
        'close': base_price + 1.0,  # Small body
        'volume': 1500
    })
    
    return bars


def _generate_double_wick_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Double Wick (Spinning Top) pattern.
    
    Mathematical rules from docs:
    - Body ratio: Bₜ / (Hₜ - Lₜ) ≤ 0.4
    - Upper wick ratio: Wᵘₜ / (Hₜ - Lₜ) ≥ 0.3
    - Lower wick ratio: Wˡₜ / (Hₜ - Lₜ) ≥ 0.3
    """
    bars = []
    
    # Bar 5: Spinning top pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.8,  # Long upper wick
        'low': base_price + 0.2,   # Long lower wick
        'close': base_price + 1.0,  # Small body
        'volume': 1500
    })
    
    return bars


def _generate_ii_bars_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create an Inside-Inside (II) bars pattern.
    
    Mathematical rules from docs:
    - Each bar's range is contained within the previous bar's range
    - Hᵢ ≤ H_{i-1} and Lᵢ ≥ L_{i-1}
    """
    bars = []
    
    # Bar 5: First inside bar
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,  # Contained within previous
        'low': base_price + 0.5,   # Contained within previous
        'close': base_price + 1.5,
        'volume': 1500
    })
    
    # Bar 6: Second inside bar
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 21, 0),
        'open': base_price + 1.2,
        'high': base_price + 1.8,  # Further contained
        'low': base_price + 0.7,   # Further contained
        'close': base_price + 1.3,
        'volume': 1500
    })
    
    return bars


def _generate_doji_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Doji pattern.
    
    Mathematical rules from docs:
    - Doji-ness score: Dₜ = exp[−(Bₜ/(Hₜ − Lₜ))²/(2σ_b²)] × exp[−(Ẇᵘₜ − Ẇˡₜ)²/(2σ_w²)]
    - σ_b = 0.05 (body size sensitivity)
    - σ_w = 0.1 (wick symmetry requirement)
    """
    bars = []
    
    # Bar 5: Doji pattern (very small body, symmetrical wicks)
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.5,  # Equal upper wick
        'low': base_price + 0.5,   # Equal lower wick
        'close': base_price + 1.0,  # Very small body
        'volume': 1500
    })
    
    return bars


def _generate_marubozu_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Marubozu pattern.
    Mathematical rules from docs:
    - Large body: Bₜ / (Hₜ - Lₜ) ≥ 0.8
    - No wicks: Wᵘₜ ≈ 0 and Wˡₜ ≈ 0
    """
    bars = []
    # Bar 5: Marubozu pattern (large body, no wicks)
        bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,  # No upper wick
        'low': base_price + 1.0,   # No lower wick
        'close': base_price + 2.0,  # Large body
            'volume': 1500
        })
    return bars


def _generate_spinning_top_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Spinning Top pattern.
    
    Mathematical rules from docs:
    - Small body: Bₜ / (Hₜ - Lₜ) ≤ 0.4
    - Long wicks: Wᵘₜ / (Hₜ - Lₜ) ≥ 0.3 and Wˡₜ / (Hₜ - Lₜ) ≥ 0.3
    """
    bars = []
    
    # Bar 5: Spinning top pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.8,  # Long upper wick
        'low': base_price + 0.2,   # Long lower wick
        'close': base_price + 1.0,  # Small body
        'volume': 1500
    })
    
    return bars


def _generate_weak_body_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Weak Body pattern.
    
    Mathematical rules from docs:
    - Small body: Bₜ / (Hₜ - Lₜ) ≤ 0.3
    """
    bars = []
    
    # Bar 5: Weak body pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.2,
        'low': base_price + 0.8,
        'close': base_price + 1.05,  # Very small body
        'volume': 1500
    })
    
    return bars


def _generate_strong_body_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Strong Body pattern.
    Mathematical rules from docs:
    - Large body: Bₜ / (Hₜ - Lₜ) ≥ 0.7
    """
    bars = []
    # Bar 5: Strong body pattern
        bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 1.0,
        'close': base_price + 1.8,  # Large body
        'volume': 1500
        })
    return bars 


def _generate_breakout_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Breakout pattern.
    
    Mathematical rules from docs:
    - Price breaks above resistance or below support
    - Increased volume and momentum
    """
    bars = []
    
    # Bar 5: Breakout pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 3.0,  # Strong breakout
        'low': base_price + 0.8,
        'close': base_price + 2.8,  # Close near high
        'volume': 2500  # High volume
    })
    
    return bars


def _generate_exhaustion_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create an Exhaustion pattern.
    
    Mathematical rules from docs:
    - Small body with long wicks
    - Indicates trend exhaustion
    """
    bars = []
    
    # Bar 5: Exhaustion pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.9,  # Long upper wick
        'low': base_price + 0.1,   # Long lower wick
        'close': base_price + 1.0,  # Small body
        'volume': 1500
    })
    
    return bars


def _generate_accumulation_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create an Accumulation pattern.
    
    Mathematical rules from docs:
    - Small range bars near support
    - Low volume
    """
    bars = []
    
    # Bar 5: Accumulation pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 0.8,
        'high': base_price + 1.2,  # Small range
        'low': base_price + 0.6,
        'close': base_price + 1.0,
        'volume': 800  # Low volume
    })
    
    return bars


def _generate_distribution_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Distribution pattern.
    
    Mathematical rules from docs:
    - Small range bars near resistance
    - Low volume
    """
    bars = []
    
    # Bar 5: Distribution pattern
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.8,
        'high': base_price + 2.2,  # Small range
        'low': base_price + 1.6,
        'close': base_price + 2.0,
        'volume': 800  # Low volume
    })
    
    return bars


def get_expected_detection_index(pattern_name: str, params: Dict[str, Any] = None) -> int:
    """
    Get the expected detection index for a pattern.
    
    Args:
        pattern_name: Name of the pattern
        params: Pattern parameters
        
    Returns:
        Expected detection index (5 for most patterns, as they're at bar 5)
    """
    if params is None:
        params = {}
    
    # Most patterns are detected at bar 5 (index 5)
    # Multi-bar patterns may have different detection points
    multi_bar_patterns = {
        'engulfing': 6,  # Second bar of engulfing
        'engulfing_bullish': 6,
        'engulfing_bearish': 6,
        'ii_bars': 6,  # Second bar of II pattern
        'three_white_soldiers': 7,  # Third bar of three-bar pattern
        'three_black_crows': 7,  # Third bar of three-bar pattern
        'fvg': 7,  # Gap bar (L_{t+1}) - after 5 filler bars + 2 pattern bars
        'vwap_mean_reversion_band': 20,
        # Support/Resistance band removed
        'imbalance_memory_zone': 15,
    }
    
    return multi_bar_patterns.get(pattern_name, 5)


def _generate_filler_bars(base_price: float, num_bars: int, pattern_name: str, start_time: datetime) -> list:
    """Generate filler bars that don't trigger the pattern"""
    bars = []
    
    for i in range(num_bars):
    bars.append({
            'datetime': start_time + timedelta(minutes=i),
            'open': base_price + i * 0.1,
            'high': base_price + i * 0.1 + 0.5,
            'low': base_price + i * 0.1 - 0.5,
            'close': base_price + i * 0.1 + 0.2,
            'volume': 1000 + i * 100
        })
    
    return bars 


def _generate_dragonfly_doji_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Dragonfly Doji pattern.
    
    Mathematical rules from docs:
    - Dₜ = exp[−(Bₜ/(Hₜ − Lₜ))²/(2σ_b²)] × exp[−(Ẇᵘₜ − Ẇˡₜ)²/(2σ_w²)]
    - Dragonfly: Very small body, long lower wick, no upper wick
    """
    bars = []
    # Bar 5: Dragonfly Doji
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.0,  # Same as open/close (no upper wick)
        'low': base_price - 2.0,   # Long lower wick
        'close': base_price + 1.0,  # Same as open (very small body)
        'volume': 1500
    })
    return bars


def _generate_gravestone_doji_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Gravestone Doji pattern.
    
    Mathematical rules from docs:
    - Dₜ = exp[−(Bₜ/(Hₜ − Lₜ))²/(2σ_b²)] × exp[−(Ẇᵘₜ − Ẇˡₜ)²/(2σ_w²)]
    - Gravestone: Very small body, long upper wick, no lower wick
    """
    bars = []
    # Bar 5: Gravestone Doji
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 3.0,  # Long upper wick
        'low': base_price + 1.0,   # Same as open/close (no lower wick)
        'close': base_price + 1.0,  # Same as open (very small body)
        'volume': 1500
    })
    return bars


def _generate_four_price_doji_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Four Price Doji pattern.
    
    Mathematical rules from docs:
    - Dₜ = exp[−(Bₜ/(Hₜ − Lₜ))²/(2σ_b²)] × exp[−(Ẇᵘₜ − Ẇˡₜ)²/(2σ_w²)]
    - Four Price: Open = High = Low = Close (perfect doji)
    """
    bars = []
    # Bar 5: Four Price Doji
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.0,  # All same price
        'low': base_price + 1.0,
        'close': base_price + 1.0,
        'volume': 1500
    })
    return bars


def _generate_three_white_soldiers_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Three White Soldiers pattern.
    
    Mathematical rules from docs:
    - Three consecutive bullish candles with higher highs and higher lows
    - Each candle opens within the previous candle's body
    """
    bars = []
    # Bar 5: First white soldier
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    # Bar 6: Second white soldier
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 21, 0),
        'open': base_price + 1.5,  # Opens within previous body
        'high': base_price + 3.0,  # Higher high
        'low': base_price + 1.2,   # Higher low
        'close': base_price + 2.8,
        'volume': 1500
    })
    # Bar 7: Third white soldier
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 22, 0),
        'open': base_price + 2.2,  # Opens within previous body
        'high': base_price + 4.0,  # Higher high
        'low': base_price + 2.0,   # Higher low
        'close': base_price + 3.8,
        'volume': 1500
    })
    return bars


def _generate_three_black_crows_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Three Black Crows pattern.
    
    Mathematical rules from docs:
    - Three consecutive bearish candles with lower highs and lower lows
    - Each candle opens near the previous candle's high
    """
    bars = []
    # Bar 5: First black crow
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 3.0,
        'high': base_price + 3.2,
        'low': base_price + 1.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    # Bar 6: Second black crow
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 21, 0),
        'open': base_price + 2.8,  # Opens near previous high
        'high': base_price + 2.9,  # Lower high
        'low': base_price + 0.8,   # Lower low
        'close': base_price + 1.0,
        'volume': 1500
    })
    # Bar 7: Third black crow
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 22, 0),
        'open': base_price + 2.5,  # Opens near previous high
        'high': base_price + 2.6,  # Lower high
        'low': base_price + 0.2,   # Lower low
        'close': base_price + 0.5,
        'volume': 1500
    })
    return bars


def _generate_momentum_breakout_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Momentum Breakout pattern.
    
    Mathematical rules from docs:
    - Strong directional move with increasing volume
    - Price breaks above resistance or below support
    """
    bars = []
    # Bar 5: Momentum breakout
            bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 5.0,  # Strong breakout
        'low': base_price + 0.8,
        'close': base_price + 4.8,
        'volume': 3000  # High volume
    })
    return bars


def _generate_momentum_reversal_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Momentum Reversal pattern.
    
    Mathematical rules from docs:
    - Strong move in one direction followed by reversal
    - Volume spike at reversal point
    """
    bars = []
    # Bar 5: Momentum reversal
            bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 3.0,
        'high': base_price + 3.5,
        'low': base_price - 2.0,  # Strong reversal
        'close': base_price - 1.8,
        'volume': 3000  # High volume
    })
    return bars


def _generate_high_volatility_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a High Volatility pattern.
    
    Mathematical rules from docs:
    - Large price range relative to average
    - σₜ = √[(1/N) ∑(rᵢ − ȓ)²]
    """
    bars = []
    # Bar 5: High volatility
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 8.0,  # Very large range
        'low': base_price - 5.0,
        'close': base_price + 2.0,
        'volume': 2000
    })
    return bars


def _generate_low_volatility_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Low Volatility pattern.
    
    Mathematical rules from docs:
    - Small price range relative to average
    - σₜ = √[(1/N) ∑(rᵢ − ȓ)²]
    """
    bars = []
    # Bar 5: Low volatility
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.2,  # Very small range
        'low': base_price + 0.9,
        'close': base_price + 1.1,
        'volume': 800
    })
    return bars


def _generate_support_bounce_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Support Bounce pattern.
    
    Mathematical rules from docs:
    - Price touches support level and bounces up
    - Mₜ = exp[−((Lₜ−R_infₜ)²)/(2σ_r²)]
    """
    bars = []
    # Bar 5: Support bounce
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 0.5,
        'high': base_price + 2.0,
        'low': base_price + 0.1,  # Touches support
        'close': base_price + 1.8,  # Bounces up
        'volume': 1500
    })
    return bars


def _generate_resistance_rejection_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Resistance Rejection pattern.
    
    Mathematical rules from docs:
    - Price touches resistance level and bounces down
    - Mₜ = exp[−((Hₜ−R_supₜ)²)/(2σ_r²)]
    """
    bars = []
    # Bar 5: Resistance rejection
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 2.0,
        'high': base_price + 2.9,  # Touches resistance
        'low': base_price + 0.5,
        'close': base_price + 0.8,  # Rejects down
        'volume': 1500
    })
    return bars


def _generate_volatility_expansion_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Volatility Expansion pattern.
    
    Mathematical rules from docs:
    - Increasing price range over multiple bars
    - ATRₜ = (1/n) ∑TRᵢ
    """
    bars = []
    # Bar 5: Volatility expansion
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 6.0,  # Expanding range
        'low': base_price - 3.0,
        'close': base_price + 2.0,
        'volume': 1800
    })
    return bars


def _generate_volatility_contraction_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Volatility Contraction pattern.
    
    Mathematical rules from docs:
    - Decreasing price range over multiple bars
    - ATRₜ = (1/n) ∑TRᵢ
    """
    bars = []
    # Bar 5: Volatility contraction
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.1,  # Contracting range
        'low': base_price + 0.95,
        'close': base_price + 1.05,
        'volume': 600
    })
    return bars


def _generate_trend_continuation_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Trend Continuation pattern.
    
    Mathematical rules from docs:
    - Price continues in established direction
    - M(t,y) = (1/n) ∑_{i=1}^n |rᵢ| ⋅ sign(rᵢ)
    """
    bars = []
    # Bar 5: Trend continuation
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 2.0,
        'high': base_price + 4.0,  # Continues upward
        'low': base_price + 1.8,
        'close': base_price + 3.8,
        'volume': 1500
    })
    return bars


def _generate_trend_reversal_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Trend Reversal pattern.
    
    Mathematical rules from docs:
    - Price reverses established direction
    - M(t,y) = (1/n) ∑_{i=1}^n |rᵢ| ⋅ sign(rᵢ)
    """
    bars = []
    # Bar 5: Trend reversal
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 3.0,
        'high': base_price + 3.2,
        'low': base_price - 1.0,  # Reverses down
        'close': base_price - 0.8,
        'volume': 2000
    })
    return bars


def _generate_gap_up_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Gap Up pattern.
    
    Mathematical rules from docs:
    - Current bar's low > previous bar's high
    """
    bars = []
    # Bar 5: Gap up
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 5.0,  # Gap up from previous high
        'high': base_price + 7.0,
        'low': base_price + 4.8,   # Low > previous high
        'close': base_price + 6.5,
        'volume': 1500
    })
    return bars


def _generate_gap_down_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Gap Down pattern.
    
    Mathematical rules from docs:
    - Current bar's high < previous bar's low
    """
    bars = []
    # Bar 5: Gap down
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price - 3.0,  # Gap down from previous low
        'high': base_price - 1.0,  # High < previous low
        'low': base_price - 4.0,
        'close': base_price - 2.5,
        'volume': 1500
    })
    return bars


def _generate_consolidation_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a Consolidation pattern.
    
    Mathematical rules from docs:
    - Price moves sideways in a narrow range
    - Low volatility, small body candles
    """
    bars = []
    # Bar 5: Consolidation
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 1.3,  # Small range
        'low': base_price + 0.8,
        'close': base_price + 1.1,  # Small body
        'volume': 800
    })
    return bars 


def _generate_location_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger location gate conditions"""
    bars = []
    # Bar 5: Location gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars


def _generate_volatility_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger volatility gate conditions"""
    bars = []
    # Bar 5: High volatility condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 8.0,  # High volatility
        'low': base_price - 5.0,
        'close': base_price + 2.0,
        'volume': 2000
    })
    return bars


def _generate_regime_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger regime gate conditions"""
    bars = []
    # Bar 5: Trending regime condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 4.0,  # Strong trend
        'low': base_price + 0.8,
        'close': base_price + 3.8,
        'volume': 1500
    })
    return bars


def _generate_bayesian_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger bayesian gate conditions"""
    bars = []
    # Bar 5: Bayesian gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 3.0,
        'low': base_price + 0.5,
        'close': base_price + 2.8,
        'volume': 1500
    })
    return bars


def _generate_fvg_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger FVG gate conditions"""
    bars = []
    # Bar 5: FVG gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars


# Support/Resistance gate function removed


def _generate_momentum_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger momentum gate conditions"""
    bars = []
    # Bar 5: Momentum gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 4.0,  # Strong momentum
        'low': base_price + 0.8,
        'close': base_price + 3.8,
        'volume': 1500
    })
    return bars


def _generate_volume_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger volume gate conditions"""
    bars = []
    # Bar 5: Volume gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 3000  # High volume
    })
    return bars


def _generate_time_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger time gate conditions"""
    bars = []
    # Bar 5: Time gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars


def _generate_correlation_gate_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger correlation gate conditions"""
    bars = []
    # Bar 5: Correlation gate condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars


def _generate_volume_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger volume filter conditions"""
    bars = []
    # Bar 5: Volume filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 3000  # High volume
    })
    return bars


def _generate_time_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger time filter conditions"""
    bars = []
    # Bar 5: Time filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars


def _generate_volatility_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger volatility filter conditions"""
    bars = []
    # Bar 5: Volatility filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 6.0,  # High volatility
        'low': base_price - 3.0,
        'close': base_price + 2.0,
        'volume': 1500
    })
    return bars


def _generate_momentum_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger momentum filter conditions"""
    bars = []
    # Bar 5: Momentum filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 4.0,  # Strong momentum
        'low': base_price + 0.8,
        'close': base_price + 3.8,
        'volume': 1500
    })
    return bars


def _generate_price_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger price filter conditions"""
    bars = []
    # Bar 5: Price filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars


def _generate_regime_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger regime filter conditions"""
    bars = []
    # Bar 5: Regime filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 3.0,
        'low': base_price + 0.5,
        'close': base_price + 2.8,
        'volume': 1500
    })
    return bars


def _generate_advanced_filter_bars(base_price: float, params: Dict[str, Any]) -> list:
    """Generate bars that trigger advanced filter conditions"""
    bars = []
    # Bar 5: Advanced filter condition
    bars.append({
        'datetime': datetime(2024, 3, 7, 16, 20, 0),
        'open': base_price + 1.0,
        'high': base_price + 2.0,
        'low': base_price + 0.5,
        'close': base_price + 1.8,
        'volume': 1500
    })
    return bars 


def _generate_vwap_mean_reversion_band_bars(base_price: float, params: dict) -> list:
    """
    Generate bars that create a valid VWAP mean-reversion band trigger.
    VWAP: Price crosses above/below a calculated VWAP band.
    """
    bars = []
    # 20 filler bars
    for i in range(20):
        bars.append({'datetime': f'2024-03-07 16:{15+i:02d}:00', 'open': base_price, 'high': base_price+1, 'low': base_price-1, 'close': base_price+0.5, 'volume': 1000})
    # VWAP band trigger: price jumps above VWAP at index 20
    bars.append({'datetime': '2024-03-07 16:35:00', 'open': base_price+0.5, 'high': base_price+5, 'low': base_price+0.5, 'close': base_price+5, 'volume': 1000})
    return bars 


# Support/Resistance band function removed 


def _generate_imbalance_memory_zone_bars(base_price: float, params: Dict[str, Any]) -> list:
    """
    Generate bars that create a valid Imbalance Memory Zone trigger.
    At a known index (e.g., 15), create a large price move to trigger the imbalance.
    """
    bars = []
    base_time = datetime(2024, 3, 7, 16, 15, 0)
    imbalance_threshold = params.get('imbalance_threshold', 100)
    # Filler bars
    for i in range(15):
        dt = base_time + pd.Timedelta(minutes=i)
        bars.append({
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'open': base_price,
            'high': base_price + 2,
            'low': base_price - 2,
            'close': base_price,
            'volume': 1000
        })
    # Bar 15: Large price move (imbalance)
    dt = base_time + pd.Timedelta(minutes=15)
    bars.append({
        'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'open': base_price,
        'high': base_price + 2,
        'low': base_price - 2,
        'close': base_price + imbalance_threshold + 10,  # Large jump
        'volume': 1000
    })
    # Filler bars after (increase to 15 bars after, for a total of 31 bars)
    for i in range(16, 31):
        dt = base_time + pd.Timedelta(minutes=i)
        bars.append({
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'open': base_price,
            'high': base_price + 2,
            'low': base_price - 2,
            'close': base_price,
            'volume': 1000
        })
    return bars 


def order_block_dataset(direction: str) -> str:
    """Create a minimal dataset with a single bullish or bearish order block."""
    import pandas as pd
    base = 100.0
    bars = []
    # 3 filler bars
    for i in range(3):
        bars.append({'open': base, 'high': base+2, 'low': base-2, 'close': base+1, 'volume': 1000})
    if direction == 'bearish':
        # Up-bar (OB)
        bars.append({'open': base, 'high': base+3, 'low': base-1, 'close': base+2, 'volume': 1200})
        # Down impulse
        bars.append({'open': base+2, 'high': base+2.5, 'low': base-5, 'close': base-4, 'volume': 1500})
    else:
        # Down-bar (OB)
        bars.append({'open': base+2, 'high': base+3, 'low': base-1, 'close': base, 'volume': 1200})
        # Up impulse
        bars.append({'open': base, 'high': base+6, 'low': base-1, 'close': base+5, 'volume': 1500})
    # 3 more fillers
    for i in range(3):
        bars.append({'open': base, 'high': base+2, 'low': base-2, 'close': base+1, 'volume': 1000})
    df = pd.DataFrame(bars)
    fname = f'test_order_block_{direction}_data.csv'
    df.to_csv(fname, index=False)
    return fname 