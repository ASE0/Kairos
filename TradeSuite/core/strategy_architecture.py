#!/usr/bin/env python3
"""
New Strategy Architecture Foundation
===================================
A complete rewrite of the strategy system with proper separation of concerns,
modular components, and consistent interfaces.

Key Principles:
1. Component Isolation: Each component (filter, gate, pattern) is self-contained
2. Consistent Interfaces: All components implement the same base interface
3. Unified Execution: Single execution path regardless of component combination
4. Visual Separation: Chart rendering is completely separate from component logic
5. Registry System: Central component registry for discovery and validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import logging

# =============================================================================
# COMPONENT TYPE DEFINITIONS
# =============================================================================

class ComponentType(Enum):
    """Types of strategy components"""
    FILTER = "filter"
    GATE = "gate" 
    PATTERN = "pattern"
    LOCATION = "location"

class SignalType(Enum):
    """Types of signals components can generate"""
    BOOLEAN = "boolean"          # True/False signals
    NUMERIC = "numeric"          # Numerical values
    CATEGORICAL = "categorical"  # Category labels

class VisualizationType(Enum):
    """Types of chart visualizations"""
    LINE = "line"               # Line overlay (e.g., VWAP, MA)
    ZONE = "zone"               # Shaded areas (e.g., support/resistance)
    MARKER = "marker"           # Points/arrows (e.g., pattern signals)
    BAND = "band"               # Upper/lower bands (e.g., Bollinger)
    HISTOGRAM = "histogram"     # Volume-style bars

# =============================================================================
# BASE COMPONENT INTERFACE
# =============================================================================

@dataclass
class ComponentResult:
    """Standardized result from any strategy component"""
    signals: pd.Series              # Boolean signals (when to act)
    values: Optional[pd.Series] = None      # Numerical values (for indicators)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info
    visualization_data: Dict[str, Any] = field(default_factory=dict)  # Chart data

class BaseComponent(ABC):
    """Base interface that ALL strategy components must implement"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.component_type = self._get_component_type()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    @abstractmethod
    def _get_component_type(self) -> ComponentType:
        """Return the type of this component"""
        pass
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> ComponentResult:
        """
        Main computation method - must be implemented by all components
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            
        Returns:
            ComponentResult with signals, values, and visualization data
        """
        pass
    
    @abstractmethod
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Return configuration for chart visualization
        
        Returns:
            Dict with visualization type, colors, labels, etc.
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required_cols = ['open', 'high', 'low', 'close']
        return all(col in data.columns for col in required_cols)
    
    def get_required_columns(self) -> List[str]:
        """Return list of required data columns"""
        return ['open', 'high', 'low', 'close']

# =============================================================================
# SPECIALIZED BASE CLASSES
# =============================================================================

class BaseFilter(BaseComponent):
    """Base class for all filters (VWAP, momentum, volatility, etc.)"""
    
    def _get_component_type(self) -> ComponentType:
        return ComponentType.FILTER
    
    def get_filter_condition(self) -> str:
        """Return the filter condition (above, below, near, etc.)"""
        return self.config.get('condition', 'above')

class BaseGate(BaseComponent):
    """Base class for all gates (location, volatility, regime, etc.)"""
    
    def _get_component_type(self) -> ComponentType:
        return ComponentType.GATE
    
    def get_gate_threshold(self) -> float:
        """Return the gate threshold value"""
        return self.config.get('threshold', 0.5)

class BasePattern(BaseComponent):
    """Base class for all patterns (candlestick, chart patterns, etc.)"""
    
    def _get_component_type(self) -> ComponentType:
        return ComponentType.PATTERN

class BaseLocation(BaseComponent):
    """Base class for all location strategies (FVG, order blocks, etc.)"""
    
    def _get_component_type(self) -> ComponentType:
        return ComponentType.LOCATION

# =============================================================================
# COMPONENT REGISTRY SYSTEM
# =============================================================================

class ComponentRegistry:
    """Central registry for all strategy components"""
    
    def __init__(self):
        self._filters: Dict[str, type] = {}
        self._gates: Dict[str, type] = {}
        self._patterns: Dict[str, type] = {}
        self._locations: Dict[str, type] = {}
        self.logger = logging.getLogger("ComponentRegistry")
    
    def register_filter(self, name: str, filter_class: type):
        """Register a new filter component"""
        if not issubclass(filter_class, BaseFilter):
            raise ValueError(f"Filter {name} must inherit from BaseFilter")
        self._filters[name] = filter_class
        self.logger.info(f"Registered filter: {name}")
    
    def register_gate(self, name: str, gate_class: type):
        """Register a new gate component"""
        if not issubclass(gate_class, BaseGate):
            raise ValueError(f"Gate {name} must inherit from BaseGate")
        self._gates[name] = gate_class
        self.logger.info(f"Registered gate: {name}")
    
    def register_pattern(self, name: str, pattern_class: type):
        """Register a new pattern component"""
        if not issubclass(pattern_class, BasePattern):
            raise ValueError(f"Pattern {name} must inherit from BasePattern")
        self._patterns[name] = pattern_class
        self.logger.info(f"Registered pattern: {name}")
    
    def register_location(self, name: str, location_class: type):
        """Register a new location strategy"""
        if not issubclass(location_class, BaseLocation):
            raise ValueError(f"Location {name} must inherit from BaseLocation")
        self._locations[name] = location_class
        self.logger.info(f"Registered location: {name}")
    
    def create_component(self, component_type: str, name: str, config: Dict[str, Any]) -> BaseComponent:
        """Create a component instance"""
        registries = {
            'filter': self._filters,
            'gate': self._gates,
            'pattern': self._patterns,
            'location': self._locations
        }
        
        if component_type not in registries:
            raise ValueError(f"Unknown component type: {component_type}")
        
        registry = registries[component_type]
        if name not in registry:
            raise ValueError(f"Unknown {component_type}: {name}")
        
        component_class = registry[name]
        return component_class(name=name, config=config)
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get list of all available components by type"""
        return {
            'filters': list(self._filters.keys()),
            'gates': list(self._gates.keys()),
            'patterns': list(self._patterns.keys()),
            'locations': list(self._locations.keys())
        }

# Global registry instance
component_registry = ComponentRegistry()

# =============================================================================
# STRATEGY EXECUTION ENGINE
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a complete strategy"""
    name: str
    filters: List[Dict[str, Any]] = field(default_factory=list)
    gates: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    locations: List[Dict[str, Any]] = field(default_factory=list)
    combination_logic: str = "AND"  # How to combine signals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'filters': self.filters,
            'gates': self.gates,
            'patterns': self.patterns,
            'locations': self.locations,
            'combination_logic': self.combination_logic
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary"""
        return cls(**data)

class StrategyEngine:
    """Unified execution engine for all strategies"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.logger = logging.getLogger("StrategyEngine")
    
    def execute_strategy(self, config: StrategyConfig, data: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Execute a complete strategy with consistent behavior
        
        Args:
            config: Strategy configuration
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (final_signals, visualization_data)
        """
        self.logger.info(f"Executing strategy: {config.name}")
        
        # Handle empty data edge case
        if data.empty:
            self.logger.warning("Empty data provided to strategy execution")
            empty_signals = pd.Series(dtype=bool, name='signals')
            empty_viz = {
                'lines': [], 'zones': [], 'markers': [], 'bands': [], 'histograms': []
            }
            return empty_signals, empty_viz
        
        # Initialize results
        all_signals = []
        visualization_data = {
            'lines': [],      # Line overlays (VWAP, MA, etc.)
            'zones': [],      # Shaded areas
            'markers': [],    # Point markers
            'bands': [],      # Upper/lower bands
            'histograms': []  # Volume-style bars
        }
        
        # Execute all component types in order
        component_results = {}
        
        # 1. Execute filters
        for filter_config in config.filters:
            filter_name = filter_config['type']
            component = self.registry.create_component('filter', filter_name, filter_config)
            result = component.compute(data)
            component_results[f"filter_{filter_name}"] = result
            all_signals.append(result.signals)
            self._add_visualization_data(component, result, visualization_data)
        
        # 2. Execute patterns  
        for pattern_config in config.patterns:
            pattern_name = pattern_config['type']
            component = self.registry.create_component('pattern', pattern_name, pattern_config)
            result = component.compute(data)
            component_results[f"pattern_{pattern_name}"] = result
            all_signals.append(result.signals)
            self._add_visualization_data(component, result, visualization_data)
        
        # 3. Execute locations
        for location_config in config.locations:
            location_name = location_config['type']
            component = self.registry.create_component('location', location_name, location_config)
            result = component.compute(data)
            component_results[f"location_{location_name}"] = result
            all_signals.append(result.signals)
            self._add_visualization_data(component, result, visualization_data)
        
        # 4. Combine all signals
        if not all_signals:
            # No components - return all False
            final_signals = pd.Series(False, index=data.index)
        elif config.combination_logic.upper() == "AND":
            # All components must be True
            final_signals = pd.Series(True, index=data.index)
            for signals in all_signals:
                final_signals = final_signals & signals
        elif config.combination_logic.upper() == "OR":
            # Any component can be True
            final_signals = pd.Series(False, index=data.index)
            for signals in all_signals:
                final_signals = final_signals | signals
        else:
            raise ValueError(f"Unknown combination logic: {config.combination_logic}")
        
        # 5. Apply gates (if any)
        for gate_config in config.gates:
            gate_name = gate_config['type']
            component = self.registry.create_component('gate', gate_name, gate_config)
            result = component.compute(data)
            # Gates filter the final signals
            final_signals = final_signals & result.signals
            self._add_visualization_data(component, result, visualization_data)
        
        self.logger.info(f"Strategy execution complete. Signals: {final_signals.sum()}/{len(final_signals)}")
        
        return final_signals, visualization_data
    
    def _add_visualization_data(self, component: BaseComponent, result: ComponentResult, viz_data: Dict[str, Any]):
        """Add component's visualization data to the overall visualization"""
        viz_config = component.get_visualization_config()
        
        if not viz_config:
            return
        
        viz_type = viz_config.get('type')
        
        if viz_type == VisualizationType.LINE.value:
            viz_data['lines'].append({
                'name': component.name,
                'data': result.values,
                'config': viz_config,
                'component_type': component.component_type.value
            })
        elif viz_type == VisualizationType.ZONE.value:
            viz_data['zones'].extend(result.visualization_data.get('zones', []))
        elif viz_type == VisualizationType.MARKER.value:
            viz_data['markers'].extend(result.visualization_data.get('markers', []))
        elif viz_type == VisualizationType.BAND.value:
            viz_data['bands'].append({
                'name': component.name,
                'data': result.visualization_data,
                'config': viz_config,
                'component_type': component.component_type.value
            })
        elif viz_type == VisualizationType.HISTOGRAM.value:
            viz_data['histograms'].append({
                'name': component.name,
                'data': result.values,
                'config': viz_config,
                'component_type': component.component_type.value
            })

# =============================================================================
# CHART VISUALIZATION SYSTEM
# =============================================================================

class ChartRenderer:
    """Unified chart rendering system that handles all component types"""
    
    def __init__(self):
        self.logger = logging.getLogger("ChartRenderer")
    
    def render_strategy_chart(self, data: pd.DataFrame, signals: pd.Series, 
                            visualization_data: Dict[str, Any], ax=None) -> None:
        """
        Render a complete strategy chart with all components
        
        Args:
            data: OHLCV DataFrame
            signals: Final strategy signals  
            visualization_data: All visualization data from components
            ax: Matplotlib axis (if None, will create new figure)
        """
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # 1. Draw candlesticks
        self._draw_candlesticks(data, ax)
        
        # 2. Draw line overlays (VWAP, MA, etc.)
        for line_data in visualization_data.get('lines', []):
            self._draw_line_overlay(line_data, ax)
        
        # 3. Draw zones (support/resistance, FVG, etc.)
        for zone_data in visualization_data.get('zones', []):
            self._draw_zone(zone_data, data, ax)
        
        # 4. Draw bands (Bollinger, etc.)
        for band_data in visualization_data.get('bands', []):
            self._draw_band(band_data, ax)
        
        # 5. Draw markers (pattern signals, etc.)
        for marker_data in visualization_data.get('markers', []):
            self._draw_marker(marker_data, data, ax)
        
        # 6. Draw signal indicators
        self._draw_signals(signals, data, ax)
        
        # 7. Format chart
        self._format_chart(ax, data)
        
        self.logger.info("Chart rendering complete")
    
    def _draw_candlesticks(self, data: pd.DataFrame, ax):
        """Draw OHLC candlesticks"""
        # Use mplfinance for proper candlesticks
        plot_data = data[['open', 'high', 'low', 'close']].copy()
        plot_data.columns = ['Open', 'High', 'Low', 'Close']
        
        try:
            mpf.plot(plot_data, type='candle', ax=ax, style='charles', 
                    show_nontrading=True, returnfig=False)
        except Exception as e:
            # Fallback to simple line chart
            self.logger.warning(f"Candlestick drawing failed: {e}, using line chart")
            ax.plot(data.index, data['close'], color='black', linewidth=1)
    
    def _draw_line_overlay(self, line_data: Dict[str, Any], ax):
        """Draw line overlays (VWAP, moving averages, etc.)"""
        config = line_data['config']
        values = line_data['data']
        
        if values is None or values.empty:
            return
        
        ax.plot(values.index, values.values, 
               color=config.get('color', 'purple'),
               linewidth=config.get('linewidth', 1),
               alpha=config.get('alpha', 0.8),
               linestyle=config.get('linestyle', '-'),
               label=config.get('label', line_data['name']),
               zorder=config.get('zorder', 5))
    
    def _draw_zone(self, zone_data: Dict[str, Any], data: pd.DataFrame, ax):
        """Draw shaded zones"""
        start_idx = zone_data.get('start_idx', 0)
        end_idx = zone_data.get('end_idx', len(data) - 1)
        min_price = zone_data.get('min_price')
        max_price = zone_data.get('max_price')
        color = zone_data.get('color', 'blue')
        alpha = zone_data.get('alpha', 0.3)
        
        if min_price is None or max_price is None:
            return
        
        x_range = data.index[start_idx:end_idx+1]
        ax.fill_between(x_range, min_price, max_price, 
                       color=color, alpha=alpha, zorder=10)
    
    def _draw_band(self, band_data: Dict[str, Any], ax):
        """Draw upper/lower bands"""
        config = band_data['config']
        data = band_data['data']
        
        upper = data.get('upper')
        lower = data.get('lower')
        
        if upper is not None and lower is not None:
            ax.fill_between(upper.index, upper.values, lower.values,
                           color=config.get('color', 'gray'),
                           alpha=config.get('alpha', 0.2),
                           zorder=config.get('zorder', 3))
    
    def _draw_marker(self, marker_data: Dict[str, Any], data: pd.DataFrame, ax):
        """Draw point markers"""
        idx = marker_data.get('index')
        price = marker_data.get('price')
        marker_type = marker_data.get('marker', '^')
        color = marker_data.get('color', 'green')
        size = marker_data.get('size', 100)
        
        if idx is not None and price is not None:
            ax.scatter(data.index[idx], price, marker=marker_type, 
                      color=color, s=size, zorder=12)
    
    def _draw_signals(self, signals: pd.Series, data: pd.DataFrame, ax):
        """Draw strategy signals"""
        signal_indices = signals[signals].index
        signal_prices = data.loc[signal_indices, 'close']
        
        if not signal_indices.empty:
            ax.scatter(signal_indices, signal_prices, 
                      marker='^', color='lime', s=80, 
                      label='Strategy Signals', zorder=15)
    
    def _format_chart(self, ax, data: pd.DataFrame):
        """Format the chart with labels, legend, etc."""
        ax.set_title('Strategy Chart')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis based on timeframe
        self._format_time_axis(ax, data)
    
    def _format_time_axis(self, ax, data: pd.DataFrame):
        """Format time axis based on data timeframe"""
        import matplotlib.dates as mdates
        from datetime import timedelta
        
        if len(data) < 2:
            return
        
        time_diff = data.index[1] - data.index[0]
        
        if time_diff <= timedelta(minutes=1):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(data)//20)))
        elif time_diff <= timedelta(hours=1):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(data)//12)))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_architecture():
    """Initialize the new strategy architecture"""
    logger = logging.getLogger("StrategyArchitecture")
    logger.info("Initializing new strategy architecture...")
    
    # The component registry is already created as a global instance
    # Individual components will register themselves when imported
    
    logger.info("Strategy architecture initialized successfully")

if __name__ == "__main__":
    # Example usage
    initialize_architecture()
    print("New Strategy Architecture loaded successfully!")
    print("\nAvailable components:", component_registry.get_available_components()) 