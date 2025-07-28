"""
Pattern Registry
===============
Central registry for all trading components including microstructure strategies
"""

import inspect
from dataclasses import dataclass
from typing import Dict, List, Type, Any, Optional
import importlib


@dataclass 
class PatternInfo:
    """Information about a pattern"""
    name: str
    class_type: Type
    module: str
    parameters: Dict[str, Any]
    description: str = ""


@dataclass
class FilterInfo:
    """Information about a filter"""
    name: str
    class_type: Optional[Type]
    module: str
    parameters: Dict[str, Any]
    description: str = ""


@dataclass
class GateInfo:
    """Information about a gate"""
    name: str
    class_type: Optional[Type]
    module: str
    parameters: Dict[str, Any]
    description: str = ""


@dataclass
class StrategyInfo:
    """Information about a strategy"""
    name: str
    class_type: Optional[Type]
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
        
        # Microstructure strategies (patterns in the GUI context)
        try:
            from strategies.microstructure_strategies import (
                OrderFlowMomentumStrategy, MicrostructureMeanReversionStrategy,
                LiquidityVacuumBreakoutStrategy
            )
            
            microstructure_patterns = [
                (OrderFlowMomentumStrategy, 'order_flow_momentum'),
                (MicrostructureMeanReversionStrategy, 'microstructure_mean_reversion'),
                (LiquidityVacuumBreakoutStrategy, 'liquidity_vacuum_breakout')
            ]
            
            for pattern_class, name in microstructure_patterns:
                self._register_pattern(pattern_class, name, 'strategies.microstructure_strategies')
        except ImportError as e:
            print(f"Warning: Could not import microstructure strategies: {e}")
    
    def _discover_main_hub_patterns(self):
        """Discover patterns from main hub"""
        try:
            # This would be populated by the main hub when patterns are loaded
            pass
        except Exception as e:
            print(f"Warning: Could not discover main hub patterns: {e}")
    
    def _discover_custom_patterns(self):
        """Discover custom patterns from workspace"""
        try:
            # This would scan the workspaces directory for custom patterns
            pass
        except Exception as e:
            print(f"Warning: Could not discover custom patterns: {e}")
    
    def _discover_filters(self):
        """Discover all available filters"""
        filter_types = [
            # Basic filters
            ('volume_filter', 'Volume Filter', {
                'min_volume': 1000,
                'volume_ratio': 1.5
            }),
            ('time_filter', 'Time Filter', {
                'start_time': '09:30',
                'end_time': '16:00'
            }),
            ('volatility_filter', 'Volatility Filter', {
                'min_atr_ratio': 0.01,
                'max_atr_ratio': 0.05
            }),
            ('momentum_filter', 'Momentum Filter', {
                'momentum_threshold': 0.02,
                'rsi_range': [30, 70]
            }),
            ('price_filter', 'Price Filter', {
                'min_price': 1.0,
                'max_price': 1000.0
            }),
            
            # Advanced filters
            ('regime_filter', 'Regime Filter', {
                'allowed_regimes': ['trending', 'ranging'],
                'confidence_threshold': 0.7
            }),
            ('correlation_filter', 'Correlation Filter', {
                'correlation_threshold': 0.7,
                'lookback_period': 20
            }),
            
            # Microstructure filters
            ('tick_frequency_filter', 'Tick Frequency Filter', {
                'max_ticks_per_second': 50,
                'min_book_depth': 100
            }),
            ('spread_filter', 'Spread Filter', {
                'max_spread_ticks': 2,
                'normal_spread_multiple': 5
            }),
            ('order_flow_filter', 'Order Flow Filter', {
                'min_cvd_threshold': 1000,
                'large_trade_ratio': 0.35
            })
        ]
        
        for filter_type, name, params in filter_types:
            self._register_filter(filter_type, name, params)
    
    def _discover_gates(self):
        """Discover all available gates"""
        gate_types = [
            # Basic gates
            ('location_gate', 'Location Gate', {
                'fvg_tolerance': 0.01
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
            }),
            # Order Block gate (docs-compliant)
            ('order_block_gate', 'Order Block', {
                'ob_impulse_threshold': 0.02,   # Minimum impulse move [0.01, 0.1]
                'ob_lookback': 10,              # Lookback for impulse detection [5, 50]
                'ob_gamma': 0.95,               # Exponential decay per bar [0.8, 0.99]
                'ob_tau_bars': 50,              # Hard purge after tau bars [5, 200]
                'ob_buffer_points': 0.1         # Buffer points [0.01, 1.0], user-tunable
            }),
            
            # Microstructure gates
            ('market_environment_gate', 'Market Environment Gate', {
                'allowed_states': ['TRENDING', 'RANGING'],
                'min_confidence': 0.7,
                'tick_window': 5000
            }),
            ('news_time_gate', 'News Time Gate', {
                'avoid_major_news': True,
                'avoid_secondary_news': False,
                'exit_before_seconds': 30
            }),
            ('tick_validation_gate', 'Tick Validation Gate', {
                'max_spread_multiple': 5,
                'min_book_depth': 100,
                'max_stale_time_ms': 1000
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
                }),
                
                # Microstructure strategies
                ('order_flow_momentum', 'Order Flow Momentum (OFM)', {
                    'cvd_period': 1000,
                    'imbalance_threshold': 1500,
                    'large_trade_size': 10,
                    'absorption_ratio': 400,
                    'trail_ticks': 3
                }),
                ('microstructure_mean_reversion', 'Microstructure Mean Reversion (MMR)', {
                    'sweep_threshold': 75,
                    'book_imbalance': 3.0,
                    'quiet_period': 200,
                    'reversion_percent': 0.6,
                    'max_heat': 4
                }),
                ('liquidity_vacuum_breakout', 'Liquidity Vacuum Breakout (LVB)', {
                    'consolidation_ticks': 500,
                    'volume_reduction': 0.3,
                    'range_ticks': 5,
                    'breakout_volume': 100,
                    'target_multiple': 2.5
                }),
                ('master_control_layer', 'Master Control Layer', {
                    'max_ticks_per_second': 50,
                    'min_book_depth': 100,
                    'max_spread': 2,
                    'account_value': 100000
                })
            ]
            
            for strategy_type, name, params in strategy_types:
                self._register_strategy(strategy_type, name, params)
        except ImportError as e:
            print(f"Warning: Could not import strategies: {e}")
    
    def get_pattern_names(self) -> List[str]:
        """Get list of all pattern names"""
        return list(self.patterns.keys())
    
    def get_filter_types(self) -> List[str]:
        """Get list of all filter types"""
        return list(self.filters.keys())
    
    def get_gate_types(self) -> List[str]:
        """Get list of all gate types"""
        return list(self.gates.keys())
    
    def get_strategy_types(self) -> List[str]:
        """Get list of all strategy types"""
        return list(self.strategies.keys())
    
    def get_pattern_info(self, name: str) -> Optional[PatternInfo]:
        """Get pattern information by name"""
        return self.patterns.get(name)
    
    def get_filter_info(self, filter_type: str) -> Optional[FilterInfo]:
        """Get filter information by type"""
        return self.filters.get(filter_type)
    
    def get_gate_info(self, gate_type: str) -> Optional[GateInfo]:
        """Get gate information by type"""
        return self.gates.get(gate_type)
    
    def get_strategy_info(self, strategy_type: str) -> Optional[StrategyInfo]:
        """Get strategy information by type"""
        return self.strategies.get(strategy_type)
    
    def create_pattern(self, name: str, **kwargs):
        """Create a pattern instance by name"""
        pattern_info = self.get_pattern_info(name)
        if pattern_info and pattern_info.class_type:
            return pattern_info.class_type(**kwargs)
        return None
    
    def create_strategy(self, strategy_type: str, **kwargs):
        """Create a strategy instance by type"""
        strategy_info = self.get_strategy_info(strategy_type)
        if strategy_info and strategy_info.class_type:
            return strategy_info.class_type(**kwargs)
        return None
    
    def get_pattern_parameters(self, name: str) -> Dict[str, Any]:
        """Get default parameters for a pattern"""
        pattern_info = self.get_pattern_info(name)
        return pattern_info.parameters if pattern_info else {}
    
    def get_filter_parameters(self, filter_type: str) -> Dict[str, Any]:
        """Get default parameters for a filter"""
        filter_info = self.get_filter_info(filter_type)
        return filter_info.parameters if filter_info else {}
    
    def get_gate_parameters(self, gate_type: str) -> Dict[str, Any]:
        """Get default parameters for a gate"""
        gate_info = self.get_gate_info(gate_type)
        return gate_info.parameters if gate_info else {}
    
    def get_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        strategy_info = self.get_strategy_info(strategy_type)
        return strategy_info.parameters if strategy_info else {}
    
    def _extract_class_parameters(self, class_type: Type) -> Dict[str, Any]:
        """Extract parameters from class constructor"""
        try:
            signature = inspect.signature(class_type.__init__)
            parameters = {}
            
            for name, param in signature.parameters.items():
                if name == 'self':
                    continue
                    
                # Get default value if available
                if param.default != inspect.Parameter.empty:
                    parameters[name] = param.default
                else:
                    # Try to infer type and provide reasonable default
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            parameters[name] = 0
                        elif param.annotation == float:
                            parameters[name] = 0.0
                        elif param.annotation == str:
                            parameters[name] = ""
                        elif param.annotation == bool:
                            parameters[name] = False
                        else:
                            parameters[name] = None
                    else:
                        parameters[name] = None
            
            return parameters
        except Exception as e:
            print(f"Warning: Could not extract parameters for {class_type}: {e}")
            return {}
    
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
    
    def refresh_components(self):
        """Refresh component discovery"""
        self.patterns.clear()
        self.filters.clear()
        self.gates.clear()
        self.strategies.clear()
        self._discover_components()


# Global registry instance
registry = PatternRegistry() 