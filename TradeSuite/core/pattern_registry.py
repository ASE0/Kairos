"""Pattern Registry System
Automatically discovers and registers new patterns, strategies, filters, and gates
"""

import os
import importlib
import inspect
from typing import Dict, List, Any, Type, Optional
from dataclasses import dataclass
from pathlib import Path
from core.data_structures import TimeRange


@dataclass
class PatternInfo:
    """Information about a registered pattern"""
    name: str
    class_type: Type
    module: str
    parameters: Dict[str, Any]
    description: str = ""


@dataclass
class FilterInfo:
    """Information about a registered filter"""
    name: str
    class_type: Type
    module: str
    parameters: Dict[str, Any]
    description: str = ""


@dataclass
class GateInfo:
    """Information about a registered gate"""
    name: str
    class_type: Type
    module: str
    parameters: Dict[str, Any]
    description: str = ""


@dataclass
class StrategyInfo:
    """Information about a registered strategy"""
    name: str
    class_type: Type
    module: str
    parameters: Dict[str, Any]
    description: str = ""


class PatternRegistry:
    """Central registry for all trading components"""
    
    def __init__(self):
        self.patterns: Dict[str, PatternInfo] = {}
        self.filters: Dict[str, FilterInfo] = {}
        self.gates: Dict[str, GateInfo] = {}
        self.strategies: Dict[str, StrategyInfo] = {}
        
        # Auto-discover components
        self._discover_components()
    
    def _discover_components(self):
        """Automatically discover all available components"""
        # Discover patterns
        self._discover_patterns()
        
        # Discover filters
        self._discover_filters()
        
        # Discover gates
        self._discover_gates()
        
        # Discover strategies
        self._discover_strategies()
        
        print(f"Registry: Discovered {len(self.patterns)} patterns, {len(self.filters)} filters, "
              f"{len(self.gates)} gates, {len(self.strategies)} strategies")
    
    def _discover_patterns(self):
        """Discover all available patterns"""
        # Basic candlestick patterns
        try:
            from patterns.candlestick_patterns import (
                EngulfingPattern, IIBarsPattern, 
                DoubleWickPattern, CustomPattern
            )
            
            basic_patterns = [
                (EngulfingPattern, 'engulfing'),
                (IIBarsPattern, 'ii_bars'),
                (DoubleWickPattern, 'double_wick'),
                (CustomPattern, 'custom')
            ]
            
            for pattern_class, name in basic_patterns:
                self._register_pattern(pattern_class, name, 'patterns.candlestick_patterns')
        except ImportError as e:
            print(f"Warning: Could not import basic patterns: {e}")
        
        # Discover patterns from main hub if available
        self._discover_main_hub_patterns()
        
        # Custom patterns from workspace
        self._discover_custom_patterns()
    
    def _discover_filters(self):
        """Discover all available filters"""
        filter_types = [
            # Volume filters
            ('volume', 'Volume Filter', {
                'min_volume': 1000,
                'volume_ratio': 1.5,
                'vwap_distance': 0.01
            }),
            
            # Time filters
            ('time', 'Time Filter', {
                'start_time': '09:30',
                'end_time': '16:00',
                'session': 'regular',
                'day_of_week': [1, 2, 3, 4, 5]
            }),
            
            # Volatility filters
            ('volatility', 'Volatility Filter', {
                'min_atr': 0.01,
                'max_atr': 0.05,
                'atr_percentile': 50
            }),
            
            # Momentum filters
            ('momentum', 'Momentum Filter', {
                'min_momentum': 0.001,
                'rsi_range': [30, 70],
                'macd_signal': 'bullish'
            }),
            
            # Price filters
            ('price', 'Price Filter', {
                'above_ma': 20,
                'below_ma': 50,
                'support_resistance': 0.02
            }),
            
            # Market regime filters
            ('regime', 'Market Regime Filter', {
                'trending': True,
                'ranging': True,
                'volatile': True
            }),
            
            # Advanced filters
            ('advanced', 'Advanced Filter', {
                'fvg_fill': True,
                'order_block': True,
                'liquidity_grab': True
            })
        ]
        
        for filter_type, name, params in filter_types:
            self._register_filter(filter_type, name, params)
    
    def _discover_gates(self):
        """Discover all available gates"""
        gate_types = [
            # Basic gates
            ('location_gate', 'Location Gate', {
                'fvg_tolerance': 0.01,
                'sr_tolerance': 0.02
            }),
            ('volatility_gate', 'Volatility Gate', {
                'min_atr_ratio': 0.01,
                'max_atr_ratio': 0.05
            }),
            ('regime_gate', 'Regime Gate', {
                'momentum_threshold': 0.02,
                'trend_probability': 0.3
            }),
            ('bayesian_gate', 'Bayesian Gate', {
                'min_state_probability': 0.3,
                'confidence_threshold': 0.6
            }),
            
            # Advanced gates
            ('fvg_gate', 'FVG Gate', {
                'fvg_tolerance': 0.01,
                'fill_threshold': 0.5
            }),
            ('support_resistance_gate', 'Support/Resistance Gate', {
                'sr_tolerance': 0.02,
                'breakout_threshold': 0.01
            }),
            ('momentum_gate', 'Momentum Gate', {
                'momentum_threshold': 0.02,
                'rsi_range': [30, 70]
            }),
            ('volume_gate', 'Volume Gate', {
                'volume_ratio': 1.5,
                'vwap_distance': 0.01
            }),
            ('time_gate', 'Time Gate', {
                'start_time': '09:30',
                'end_time': '16:00'
            }),
            ('correlation_gate', 'Correlation Gate', {
                'correlation_threshold': 0.7,
                'lookback_period': 20
            })
        ]
        
        for gate_type, name, params in gate_types:
            self._register_gate(gate_type, name, params)
    
    def _discover_strategies(self):
        """Discover all available strategies"""
        try:
            from strategies.strategy_builders import PatternStrategy, StrategyFactory
            
            strategy_types = [
                ('pattern_strategy', 'Pattern Strategy', {
                    'combination_logic': 'AND',
                    'min_actions_required': 1
                }),
                ('weighted_strategy', 'Weighted Strategy', {
                    'combination_logic': 'WEIGHTED',
                    'min_actions_required': 1
                }),
                ('conditional_strategy', 'Conditional Strategy', {
                    'combination_logic': 'OR',
                    'min_actions_required': 1
                })
            ]
            
            for strategy_type, name, params in strategy_types:
                self._register_strategy(strategy_type, name, params)
        except ImportError as e:
            print(f"Warning: Could not import strategies: {e}")
    
    def _discover_custom_patterns(self):
        """Discover custom patterns from workspace"""
        custom_patterns_dir = Path("workspaces/patterns")
        if custom_patterns_dir.exists():
            for pattern_file in custom_patterns_dir.glob("*.py"):
                try:
                    # Import the custom pattern module
                    module_name = f"workspaces.patterns.{pattern_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Look for pattern classes in the module
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and hasattr(obj, 'get_strength'):
                            # This looks like a pattern class
                            self._register_pattern(obj, name.lower(), module_name)
                except Exception as e:
                    print(f"Warning: Could not load custom pattern {pattern_file}: {e}")
    
    def _discover_main_hub_patterns(self):
        """Discover patterns from main hub if available"""
        try:
            # Import the main hub patterns directly
            from gui.main_hub import TradingStrategyHub
            
            # Create a temporary main hub instance to get the patterns
            # We need to create a QApplication first
            import sys
            from PyQt6.QtWidgets import QApplication
            
            # Create QApplication if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            # Create main hub and load defaults
            main_hub = TradingStrategyHub()
            
            # Register all patterns from main hub
            for pattern_name, pattern_obj in main_hub.patterns.items():
                if pattern_name not in self.patterns:  # Avoid duplicates
                    self._register_main_hub_pattern(pattern_name, pattern_obj)
            
            print(f"Discovered {len(main_hub.patterns)} patterns from main hub")
            
            # Clean up
            main_hub.close()
            
        except Exception as e:
            print(f"Warning: Could not discover main hub patterns: {e}")
            # Fallback: try to discover patterns without QApplication
            self._discover_main_hub_patterns_fallback()
    
    def _discover_main_hub_patterns_fallback(self):
        """Fallback method to discover main hub patterns without QApplication"""
        try:
            # Import the pattern classes directly
            from patterns.candlestick_patterns import (
                EngulfingPattern, IIBarsPattern, 
                DoubleWickPattern, CustomPattern
            )
            from core.data_structures import TimeRange, OHLCRatio
            
            # Create the same patterns as in main hub
            main_hub_patterns = {
                # Basic patterns
                'ii_bars': IIBarsPattern(timeframes=[TimeRange(5, 'm')]),
                'double_wick': DoubleWickPattern(timeframes=[TimeRange(15, 'm')]),
                'engulfing_bullish': EngulfingPattern(timeframes=[TimeRange(1, 'h')], pattern_type='bullish'),
                'engulfing_bearish': EngulfingPattern(timeframes=[TimeRange(1, 'h')], pattern_type='bearish'),
                
                # Advanced patterns (simplified versions)
                'doji_standard': CustomPattern(
                    name="Doji_Standard",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.1, upper_wick_ratio=0.4, lower_wick_ratio=0.4)]
                ),
                'strong_body': CustomPattern(
                    name="Strong_Body",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.7, upper_wick_ratio=0.15, lower_wick_ratio=0.15)]
                ),
                'weak_body': CustomPattern(
                    name="Weak_Body",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.2, upper_wick_ratio=0.4, lower_wick_ratio=0.4)]
                ),
                'momentum_breakout': CustomPattern(
                    name="Momentum_Breakout",
                    timeframes=[TimeRange(5, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)]
                ),
                'momentum_reversal': CustomPattern(
                    name="Momentum_Reversal",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.25, lower_wick_ratio=0.25)]
                ),
                'high_volatility': CustomPattern(
                    name="High_Volatility",
                    timeframes=[TimeRange(5, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)]
                ),
                'low_volatility': CustomPattern(
                    name="Low_Volatility",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.3, upper_wick_ratio=0.35, lower_wick_ratio=0.35)]
                ),
                'support_bounce': CustomPattern(
                    name="Support_Bounce",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.2, lower_wick_ratio=0.3)]
                ),
                'resistance_rejection': CustomPattern(
                    name="Resistance_Rejection",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.3, lower_wick_ratio=0.2)]
                ),
                'three_white_soldiers': CustomPattern(
                    name="Three_White_Soldiers",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[
                        OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                        OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                        OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)
                    ],
                    required_bars=3
                ),
                'three_black_crows': CustomPattern(
                    name="Three_Black_Crows",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[
                        OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                        OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                        OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)
                    ],
                    required_bars=3
                ),
                'four_price_doji': CustomPattern(
                    name="Four_Price_Doji",
                    timeframes=[TimeRange(30, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.01, upper_wick_ratio=0.495, lower_wick_ratio=0.495)]
                ),
                'dragonfly_doji': CustomPattern(
                    name="Dragonfly_Doji",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.1, upper_wick_ratio=0.0, lower_wick_ratio=0.8)]
                ),
                'gravestone_doji': CustomPattern(
                    name="Gravestone_Doji",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.1, upper_wick_ratio=0.8, lower_wick_ratio=0.0)]
                ),
                'volatility_expansion': CustomPattern(
                    name="Volatility_Expansion",
                    timeframes=[TimeRange(5, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)]
                ),
                'volatility_contraction': CustomPattern(
                    name="Volatility_Contraction",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.2, upper_wick_ratio=0.4, lower_wick_ratio=0.4)]
                ),
                'trend_continuation': CustomPattern(
                    name="Trend_Continuation",
                    timeframes=[TimeRange(30, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)]
                ),
                'trend_reversal': CustomPattern(
                    name="Trend_Reversal",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.25, lower_wick_ratio=0.25)]
                ),
                'gap_up': CustomPattern(
                    name="Gap_Up",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.7, upper_wick_ratio=0.15, lower_wick_ratio=0.15)]
                ),
                'gap_down': CustomPattern(
                    name="Gap_Down",
                    timeframes=[TimeRange(15, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.7, upper_wick_ratio=0.15, lower_wick_ratio=0.15)]
                ),
                'consolidation': CustomPattern(
                    name="Consolidation",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.3, upper_wick_ratio=0.35, lower_wick_ratio=0.35)]
                ),
                'breakout': CustomPattern(
                    name="Breakout",
                    timeframes=[TimeRange(30, 'm')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)]
                ),
                'exhaustion': CustomPattern(
                    name="Exhaustion",
                    timeframes=[TimeRange(1, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)]
                ),
                'accumulation': CustomPattern(
                    name="Accumulation",
                    timeframes=[TimeRange(4, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)]
                ),
                'distribution': CustomPattern(
                    name="Distribution",
                    timeframes=[TimeRange(4, 'h')],
                    ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)]
                )
            }
            
            # Register all patterns
            for pattern_name, pattern_obj in main_hub_patterns.items():
                if pattern_name not in self.patterns:  # Avoid duplicates
                    self._register_main_hub_pattern(pattern_name, pattern_obj)
            
            print(f"Discovered {len(main_hub_patterns)} patterns from main hub fallback")
            
        except Exception as e:
            print(f"Warning: Could not discover main hub patterns (fallback): {e}")
    
    def _register_main_hub_pattern(self, pattern_name: str, pattern_obj):
        """Register a pattern from the main hub"""
        self.patterns[pattern_name] = PatternInfo(
            name=pattern_name,
            class_type=type(pattern_obj),
            module='gui.main_hub',
            parameters=self._extract_pattern_parameters(pattern_obj),
            description=f"Main hub pattern: {pattern_name}"
        )
    
    def _extract_pattern_parameters(self, pattern_obj) -> Dict[str, Any]:
        """Extract parameters from a pattern object"""
        params = {}
        
        # Extract common pattern attributes
        if hasattr(pattern_obj, 'timeframes'):
            params['timeframes'] = str(pattern_obj.timeframes)
        
        if hasattr(pattern_obj, 'name'):
            params['name'] = pattern_obj.name
        
        if hasattr(pattern_obj, 'required_bars'):
            params['required_bars'] = pattern_obj.required_bars
        
        # Extract custom pattern parameters
        if hasattr(pattern_obj, 'ohlc_ratios'):
            params['ohlc_ratios'] = str(pattern_obj.ohlc_ratios)
        
        if hasattr(pattern_obj, 'custom_formula'):
            params['custom_formula'] = pattern_obj.custom_formula
        
        if hasattr(pattern_obj, 'advanced_features'):
            params['advanced_features'] = str(pattern_obj.advanced_features)
        
        # Extract specific pattern parameters
        if hasattr(pattern_obj, 'min_bars'):
            params['min_bars'] = pattern_obj.min_bars
        
        if hasattr(pattern_obj, 'min_wick_ratio'):
            params['min_wick_ratio'] = pattern_obj.min_wick_ratio
        
        if hasattr(pattern_obj, 'max_body_ratio'):
            params['max_body_ratio'] = pattern_obj.max_body_ratio
        
        if hasattr(pattern_obj, 'min_lower_wick_ratio'):
            params['min_lower_wick_ratio'] = pattern_obj.min_lower_wick_ratio
        
        if hasattr(pattern_obj, 'max_upper_wick_ratio'):
            params['max_upper_wick_ratio'] = pattern_obj.max_upper_wick_ratio
        
        if hasattr(pattern_obj, 'pattern_type'):
            params['pattern_type'] = pattern_obj.pattern_type
        
        return params
    
    def _register_pattern(self, pattern_class: Type, name: str, module: str):
        """Register a pattern"""
        # Extract parameters from class constructor
        params = self._extract_class_parameters(pattern_class)
        
        self.patterns[name] = PatternInfo(
            name=name,
            class_type=pattern_class,
            module=module,
            parameters=params,
            description=getattr(pattern_class, '__doc__', '')
        )
    
    def _register_filter(self, filter_type: str, name: str, parameters: Dict):
        """Register a filter"""
        self.filters[filter_type] = FilterInfo(
            name=name,
            class_type=None,  # Will be created dynamically
            module='core.filters',
            parameters=parameters,
            description=f"Filter for {name}"
        )
    
    def _register_gate(self, gate_type: str, name: str, parameters: Dict):
        """Register a gate"""
        self.gates[gate_type] = GateInfo(
            name=name,
            class_type=None,  # Will be created dynamically
            module='core.gates',
            parameters=parameters,
            description=f"Gate for {name}"
        )
    
    def _register_strategy(self, strategy_type: str, name: str, parameters: Dict):
        """Register a strategy"""
        self.strategies[strategy_type] = StrategyInfo(
            name=name,
            class_type=None,  # Will be created dynamically
            module='strategies.strategy_builders',
            parameters=parameters,
            description=f"Strategy: {name}"
        )
    
    def _extract_class_parameters(self, cls: Type) -> Dict[str, Any]:
        """Extract parameters from a class constructor"""
        try:
            sig = inspect.signature(cls.__init__)
            params = {}
            for name, param in sig.parameters.items():
                if name != 'self' and param.default != inspect.Parameter.empty:
                    params[name] = param.default
            return params
        except:
            return {}
    
    def get_pattern_names(self) -> List[str]:
        """Get all registered pattern names"""
        return list(self.patterns.keys())
    
    def get_filter_types(self) -> List[str]:
        """Get all registered filter types"""
        return list(self.filters.keys())
    
    def get_gate_types(self) -> List[str]:
        """Get all registered gate types"""
        return list(self.gates.keys())
    
    def get_strategy_types(self) -> List[str]:
        """Get all registered strategy types"""
        return list(self.strategies.keys())
    
    def get_pattern_info(self, name: str) -> Optional[PatternInfo]:
        """Get information about a pattern"""
        return self.patterns.get(name)
    
    def get_filter_info(self, filter_type: str) -> Optional[FilterInfo]:
        """Get information about a filter"""
        return self.filters.get(filter_type)
    
    def get_gate_info(self, gate_type: str) -> Optional[GateInfo]:
        """Get information about a gate"""
        return self.gates.get(gate_type)
    
    def get_strategy_info(self, strategy_type: str) -> Optional[StrategyInfo]:
        """Get information about a registered strategy"""
        return self.strategies.get(strategy_type)
    
    def create_pattern(self, pattern_name: str, **kwargs) -> Any:
        """Create a pattern instance with parameters"""
        if pattern_name not in self.patterns:
            # If pattern is not in the registry, create a default CustomPattern
            from patterns.candlestick_patterns import CustomPattern
            timeframes = kwargs.get('timeframes', [TimeRange(15, 'm')])
            return CustomPattern(name=pattern_name, timeframes=timeframes, ohlc_ratios=[])
        
        pattern_info = self.patterns[pattern_name]
        
        # Handle timeframes parameter
        timeframes = kwargs.pop('timeframes', [TimeRange(15, 'm')])
        if isinstance(timeframes, list) and timeframes and isinstance(timeframes[0], str):
            # Convert string timeframes to TimeRange objects
            tf_objects = []
            for tf_str in timeframes:
                import re
                match = re.match(r'(\d+)([smhd])', tf_str)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    tf_objects.append(TimeRange(value, unit))
                else:
                    tf_objects.append(TimeRange(15, 'm'))
            timeframes = tf_objects
        
        try:
            if pattern_info.class_type:
                # For specific registered classes, instantiate them
                if pattern_info.class_type.__name__ == 'CustomPattern':
                    return pattern_info.class_type(name=pattern_name, timeframes=timeframes, ohlc_ratios=kwargs.get('ohlc_ratios', []))
                else:
                    return pattern_info.class_type(timeframes=timeframes, **kwargs)
            else:
                # Fallback for patterns defined only by module/type
                from patterns.candlestick_patterns import CustomPattern
                return CustomPattern(name=pattern_name, timeframes=timeframes, ohlc_ratios=[])
                
        except Exception as e:
            print(f"Error creating pattern '{pattern_name}': {e}")
            from patterns.candlestick_patterns import CustomPattern
            return CustomPattern(name=pattern_name, timeframes=[TimeRange(15, 'm')], ohlc_ratios=[])
    
    def refresh(self):
        """Re-discover all components"""
        self.patterns.clear()
        self.filters.clear()
        self.gates.clear()
        self.strategies.clear()
        self._discover_components()


# Global registry instance
registry = PatternRegistry() 