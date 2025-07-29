#!/usr/bin/env python3
"""
Filter Components for New Strategy Architecture
==============================================
All filter implementations using the new modular architecture.
Each filter is self-contained and implements the BaseFilter interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ..strategy_architecture import (
    BaseFilter, ComponentResult, VisualizationType, 
    component_registry
)

# =============================================================================
# VWAP FILTER
# =============================================================================

class VWAPFilter(BaseFilter):
    """
    Volume Weighted Average Price (VWAP) Filter
    
    Generates signals based on price relationship to VWAP.
    Always displays VWAP line on chart regardless of condition.
    """
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute VWAP and generate signals"""
        if not self.validate_data(data) or 'volume' not in data.columns:
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                values=None,
                metadata={'error': 'Missing required columns'}
            )
        
        # Get VWAP calculation period
        period = self.config.get('period', 200)  # Default 200 bars
        
        # Calculate VWAP according to mathematical framework: VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        if period > 0:
            # Rolling VWAP with specified period
            price_volume = typical_price * data['volume']
            vwap = price_volume.rolling(window=period, min_periods=1).sum() / data['volume'].rolling(window=period, min_periods=1).sum()
        else:
            # Cumulative VWAP (original behavior)
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Generate signals based on condition
        condition = self.get_filter_condition()
        tolerance = self.config.get('tolerance', 0.001)  # 0.1% default
        
        if condition == 'above':
            signals = data['close'] > vwap
        elif condition == 'below':
            signals = data['close'] < vwap
        elif condition == 'near':
            signals = (abs(data['close'] - vwap) / vwap) <= tolerance
        else:
            self.logger.warning(f"Unknown VWAP condition: {condition}, defaulting to 'above'")
            signals = data['close'] > vwap
        
        self.logger.info(f"VWAP filter ({condition}): {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            values=vwap,  # VWAP values for chart display
            metadata={
                'condition': condition,
                'tolerance': tolerance,
                'period': period,
                'vwap_range': [vwap.min(), vwap.max()]
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Configuration for VWAP line display"""
        return {
            'type': VisualizationType.LINE.value,
            'color': 'purple',
            'linewidth': 1.5,
            'alpha': 0.8,
            'linestyle': '-',
            'label': f'VWAP ({self.get_filter_condition()})',
            'zorder': 5
        }

# =============================================================================
# MOMENTUM FILTER  
# =============================================================================

class MomentumFilter(BaseFilter):
    """
    Momentum Filter with RSI
    
    Based on mathematical framework: M(t,y) = (1/n) Σ |r_i|·sign(r_i)
    """
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute momentum and RSI signals"""
        if not self.validate_data(data):
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing required columns'}
            )
        
        lookback = self.config.get('lookback', 10)
        momentum_threshold = self.config.get('momentum_threshold', 0.001)
        rsi_range = self.config.get('rsi_range', [0, 100])  # Full range by default
        
        # Calculate momentum per mathematical framework
        returns = data['close'].pct_change()
        momentum_signals = pd.Series(False, index=data.index)
        momentum_values = pd.Series(0.0, index=data.index)
        
        for i in range(lookback, len(data)):
            recent_returns = returns.iloc[i-lookback:i]
            # Simplified momentum calculation for directional movement
            momentum = np.mean(recent_returns)
            momentum_values.iloc[i] = momentum
            momentum_signals.iloc[i] = abs(momentum) > momentum_threshold
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Combine momentum and RSI
        rsi_min, rsi_max = rsi_range
        rsi_signals = (rsi >= rsi_min) & (rsi <= rsi_max)
        
        # Only apply RSI filter if we have valid RSI values
        valid_rsi = rsi.notna()
        if valid_rsi.sum() > 0:
            final_signals = momentum_signals & rsi_signals
        else:
            final_signals = momentum_signals
        
        self.logger.info(f"Momentum filter: {final_signals.sum()}/{len(final_signals)} signals generated")
        
        return ComponentResult(
            signals=final_signals,
            values=rsi,  # Display RSI line
            metadata={
                'momentum_threshold': momentum_threshold,
                'rsi_range': rsi_range,
                'momentum_signals': momentum_signals.sum(),
                'rsi_signals': rsi_signals.sum()
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Configuration for RSI line display"""
        return {
            'type': VisualizationType.LINE.value,
            'color': 'orange',
            'linewidth': 1,
            'alpha': 0.7,
            'linestyle': '-',
            'label': 'RSI (Momentum)',
            'zorder': 4
        }

# =============================================================================
# VOLATILITY FILTER
# =============================================================================

class VolatilityFilter(BaseFilter):
    """
    Volatility Filter using ATR
    
    Based on mathematical framework: ATR = Average True Range
    """
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute ATR-based volatility signals"""
        if not self.validate_data(data):
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing required columns'}
            )
        
        min_atr_ratio = self.config.get('min_atr_ratio', 0.01)
        max_atr_ratio = self.config.get('max_atr_ratio', 0.05)
        atr_period = self.config.get('atr_period', 14)
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=atr_period).mean()
        
        # Calculate ATR ratio
        avg_price = data['close'].rolling(window=atr_period).mean()
        atr_ratio = atr / avg_price
        
        # Generate signals based on ATR ratio bounds
        signals = (atr_ratio >= min_atr_ratio) & (atr_ratio <= max_atr_ratio)
        
        self.logger.info(f"Volatility filter: {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            values=atr_ratio,  # Display ATR ratio
            metadata={
                'min_atr_ratio': min_atr_ratio,
                'max_atr_ratio': max_atr_ratio,
                'atr_ratio_range': [atr_ratio.min(), atr_ratio.max()]
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Configuration for ATR ratio line display"""
        return {
            'type': VisualizationType.LINE.value,
            'color': 'red',
            'linewidth': 1,
            'alpha': 0.6,
            'linestyle': '--',
            'label': 'ATR Ratio',
            'zorder': 3
        }

# =============================================================================
# MOVING AVERAGE FILTER
# =============================================================================

class MovingAverageFilter(BaseFilter):
    """Simple Moving Average Filter"""
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute moving average signals"""
        if not self.validate_data(data):
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing required columns'}
            )
        
        period = self.config.get('period', 20)
        tolerance = self.config.get('tolerance', 0.001)
        condition = self.get_filter_condition()
        
        # Calculate moving average
        ma = data['close'].rolling(window=period).mean()
        
        # Generate signals based on condition
        if condition == 'above':
            signals = data['close'] > ma
        elif condition == 'below':
            signals = data['close'] < ma
        elif condition == 'near':
            signals = (abs(data['close'] - ma) / ma) <= tolerance
        else:
            signals = data['close'] > ma
        
        self.logger.info(f"MA{period} filter ({condition}): {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            values=ma,
            metadata={
                'period': period,
                'condition': condition,
                'tolerance': tolerance
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Configuration for MA line display"""
        period = self.config.get('period', 20)
        return {
            'type': VisualizationType.LINE.value,
            'color': 'blue',
            'linewidth': 1,
            'alpha': 0.8,
            'linestyle': '-',
            'label': f'MA{period}',
            'zorder': 4
        }

# =============================================================================
# BOLLINGER BANDS FILTER
# =============================================================================

class BollingerBandsFilter(BaseFilter):
    """Bollinger Bands Filter"""
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute Bollinger Bands signals"""
        if not self.validate_data(data):
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing required columns'}
            )
        
        period = self.config.get('period', 20)
        std_dev = self.config.get('std_dev', 2)
        condition = self.get_filter_condition()
        
        # Calculate Bollinger Bands
        ma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        # Generate signals based on condition
        if condition == 'touching_upper':
            signals = data['high'] >= upper_band
        elif condition == 'touching_lower':
            signals = data['low'] <= lower_band
        elif condition == 'outside_bands':
            signals = (data['close'] > upper_band) | (data['close'] < lower_band)
        elif condition == 'inside_bands':
            signals = (data['close'] <= upper_band) & (data['close'] >= lower_band)
        else:
            signals = data['close'] > upper_band
        
        self.logger.info(f"Bollinger Bands filter ({condition}): {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            values=ma,  # Middle line
            visualization_data={
                'upper': upper_band,
                'lower': lower_band
            },
            metadata={
                'period': period,
                'std_dev': std_dev,
                'condition': condition
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Configuration for Bollinger Bands display"""
        period = self.config.get('period', 20)
        return {
            'type': VisualizationType.BAND.value,
            'color': 'gray',
            'alpha': 0.2,
            'label': f'BB({period})',
            'zorder': 2
        }

# =============================================================================
# MICROSTRUCTURE FILTERS (Simplified for now)
# =============================================================================

class TickFrequencyFilter(BaseFilter):
    """Tick Frequency Filter (microstructure)"""
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute tick frequency signals using volume as proxy"""
        if not self.validate_data(data) or 'volume' not in data.columns:
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing volume data'}
            )
        
        max_ticks_per_second = self.config.get('max_ticks_per_second', 50)
        min_book_depth = self.config.get('min_book_depth', 100)
        
        # Use volume as proxy for tick frequency
        avg_volume = data['volume'].rolling(window=20).mean()
        signals = (data['volume'] <= max_ticks_per_second * 1000) & (avg_volume >= min_book_depth)
        
        self.logger.info(f"Tick frequency filter: {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            metadata={
                'max_ticks_per_second': max_ticks_per_second,
                'min_book_depth': min_book_depth
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """No specific visualization for tick frequency"""
        return {}

class SpreadFilter(BaseFilter):
    """Spread Filter (microstructure)"""
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute spread signals using price volatility as proxy"""
        if not self.validate_data(data):
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing required columns'}
            )
        
        max_spread_ticks = self.config.get('max_spread_ticks', 2)
        
        # Use price volatility as proxy for spread
        price_volatility = data['close'].rolling(window=20).std()
        avg_price = data['close'].rolling(window=20).mean()
        spread_ratio = price_volatility / avg_price
        
        signals = spread_ratio <= (max_spread_ticks * 0.001)
        
        self.logger.info(f"Spread filter: {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            metadata={
                'max_spread_ticks': max_spread_ticks
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """No specific visualization for spread"""
        return {}

class OrderFlowFilter(BaseFilter):
    """Order Flow Filter (microstructure)"""
    
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """Compute order flow signals using volume as proxy"""
        if not self.validate_data(data) or 'volume' not in data.columns:
            return ComponentResult(
                signals=pd.Series(False, index=data.index),
                metadata={'error': 'Missing volume data'}
            )
        
        min_cvd_threshold = self.config.get('min_cvd_threshold', 1000)
        large_trade_ratio = self.config.get('large_trade_ratio', 0.35)
        
        # Use volume as proxy for order flow
        avg_volume = data['volume'].rolling(window=20).mean()
        large_trades = data['volume'] > (avg_volume * large_trade_ratio)
        large_trade_ratio_actual = large_trades.rolling(window=20).mean()
        
        signals = (data['volume'] >= min_cvd_threshold) & (large_trade_ratio_actual >= large_trade_ratio)
        
        self.logger.info(f"Order flow filter: {signals.sum()}/{len(signals)} signals generated")
        
        return ComponentResult(
            signals=signals,
            metadata={
                'min_cvd_threshold': min_cvd_threshold,
                'large_trade_ratio': large_trade_ratio
            }
        )
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """No specific visualization for order flow"""
        return {}

# =============================================================================
# REGISTER ALL FILTERS
# =============================================================================

def register_all_filters():
    """Register all filter components with the global registry"""
    component_registry.register_filter('vwap', VWAPFilter)
    component_registry.register_filter('momentum', MomentumFilter)
    component_registry.register_filter('volatility', VolatilityFilter)
    component_registry.register_filter('ma', MovingAverageFilter)
    component_registry.register_filter('bollinger_bands', BollingerBandsFilter)
    component_registry.register_filter('tick_frequency', TickFrequencyFilter)
    component_registry.register_filter('spread', SpreadFilter)
    component_registry.register_filter('order_flow', OrderFlowFilter)
    
    print("Registered all filter components!")

# Auto-register when imported
register_all_filters() 