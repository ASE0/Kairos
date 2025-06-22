"""
Comprehensive Strategy Factory
Creates strategies with all advanced parameters that the AI optimizer can configure
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.data_structures import TimeRange
from strategies.strategy_builders import PatternStrategy, Action, StrategyFactory
from patterns.candlestick_patterns import PatternFactory


@dataclass
class AdvancedStrategyConfig:
    """Configuration for advanced strategy creation"""
    # Pattern configuration
    patterns: List[str]
    pattern_params: Dict[str, Dict]
    pattern_timeframes: Dict[str, List[str]]  # Per-pattern timeframes
    
    # Filter configuration
    filters: List[Dict]
    filter_params: Dict[str, Dict]
    
    # Strategy combination
    combination_logic: str
    weights: List[float]
    min_actions_required: int
    
    # Gates and execution logic
    gates: Dict
    gate_params: Dict[str, Dict]
    execution_logic: Dict
    
    # Risk management
    risk_params: Dict
    position_sizing: Dict
    
    # Advanced features
    timeframes: List[str]  # Global timeframes (legacy)
    volatility_profile: Dict
    market_regime: Dict


class ComprehensiveStrategyFactory:
    """Factory for creating comprehensive strategies with all parameters"""
    
    def __init__(self):
        self.registry = None
        try:
            from core.pattern_registry import registry
            self.registry = registry
        except ImportError:
            pass
    
    def create_strategy(self, config: AdvancedStrategyConfig) -> PatternStrategy:
        """Create a comprehensive strategy from configuration"""
        # Create actions for each pattern
        actions = []
        for pattern_name in config.patterns:
            # Get pattern-specific timeframes or use global timeframes as fallback
            pattern_timeframes = config.pattern_timeframes.get(pattern_name, config.timeframes)
            pattern = self._create_pattern(pattern_name, config.pattern_params.get(pattern_name, {}), pattern_timeframes)
            action = self._create_action(pattern, config.filters, config.filter_params)
            actions.append(action)
        
        # Create gates and logic
        gates_and_logic = self._create_gates_and_logic(config)
        
        # Create strategy
        strategy = StrategyFactory.create_pattern_strategy(
            name=f"comprehensive_strategy_{self._generate_id()}",
            actions=actions,
            combination_logic=config.combination_logic,
            weights=config.weights,
            min_actions_required=config.min_actions_required,
            gates_and_logic=gates_and_logic
        )
        
        # Apply advanced features
        self._apply_advanced_features(strategy, config)
        
        return strategy
    
    def _create_pattern(self, pattern_name: str, params: Dict, timeframes: List[str]) -> Any:
        """Create a pattern instance with parameters"""
        try:
            if self.registry:
                return self.registry.create_pattern(pattern_name, **params)
        except Exception as e:
            print(f"Warning: Could not create pattern '{pattern_name}' using registry: {e}")
        
        # Fallback to manual creation
        return self._create_pattern_manual(pattern_name, params, timeframes)
    
    def _create_pattern_manual(self, pattern_name: str, params: Dict, timeframes: List[str]) -> Any:
        """Create pattern manually with fallback"""
        from patterns.candlestick_patterns import (
            EngulfingPattern, IIBarsPattern, 
            DoubleWickPattern, CustomPattern
        )
        
        # Convert timeframes to TimeRange objects
        tf_objects = []
        for tf_str in timeframes:
            if isinstance(tf_str, str):
                # Parse timeframe string like "15m", "1h", "1d"
                import re
                match = re.match(r'(\d+)([smhd])', tf_str)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    tf_objects.append(TimeRange(value, unit))
                else:
                    tf_objects.append(TimeRange(15, 'm'))  # Default fallback
            else:
                tf_objects.append(tf_str)
        
        if not tf_objects:
            tf_objects = [TimeRange(15, 'm')]  # Default fallback
        
        if pattern_name == 'engulfing':
            pattern_type = params.get('pattern_type', 'both')
            return EngulfingPattern(tf_objects, pattern_type)
            
        elif pattern_name == 'ii_bars':
            min_bars = params.get('min_bars', 2)
            return IIBarsPattern(tf_objects, min_bars)
            
        elif pattern_name == 'double_wick':
            min_wick_ratio = params.get('min_wick_ratio', 0.3)
            max_body_ratio = params.get('max_body_ratio', 0.4)
            return DoubleWickPattern(tf_objects, min_wick_ratio, max_body_ratio)
            
        elif pattern_name == 'doji':
            max_body_ratio = params.get('max_body_ratio', 0.1)
            ohlc_ratios = [{'body_ratio': (0, max_body_ratio)}]
            return CustomPattern("doji", tf_objects, ohlc_ratios)
            
        else:
            return PatternFactory.create_pattern(pattern_name)
    
    def _create_action(self, pattern: Any, filters: List[Dict], filter_params: Dict) -> Action:
        """Create an action with filters"""
        # Apply filter parameters to filters
        enhanced_filters = []
        for filter_config in filters:
            filter_type = filter_config['type']
            if filter_type in filter_params:
                # Merge default filter config with parameters
                enhanced_filter = filter_config.copy()
                enhanced_filter.update(filter_params[filter_type])
                enhanced_filters.append(enhanced_filter)
            else:
                enhanced_filters.append(filter_config)
        
        return Action(
            name=f"action_{pattern.__class__.__name__.lower()}",
            pattern=pattern,
            filters=enhanced_filters
        )
    
    def _create_gates_and_logic(self, config: AdvancedStrategyConfig) -> Dict:
        """Create gates and logic configuration"""
        gates_and_logic = {}
        
        # Add gates
        for gate_name, enabled in config.gates.items():
            if enabled:
                gates_and_logic[gate_name] = True
        
        # Add execution logic
        execution_logic = config.execution_logic
        if execution_logic.get('master_equation'):
            gates_and_logic['master_equation'] = True
        if execution_logic.get('volatility_veto'):
            gates_and_logic['volatility_veto'] = True
        if execution_logic.get('alignment'):
            gates_and_logic['alignment'] = True
        
        # Add thresholds
        gates_and_logic['exec_threshold'] = execution_logic.get('exec_threshold', 0.5)
        gates_and_logic['mmrs_threshold'] = execution_logic.get('mmrs_threshold', 0.5)
        gates_and_logic['min_state_probability'] = execution_logic.get('min_state_probability', 0.3)
        
        # Add gate-specific parameters
        for gate_name, params in config.gate_params.items():
            if gate_name in config.gates and config.gates[gate_name]:
                for param_name, value in params.items():
                    gates_and_logic[f"{gate_name}_{param_name}"] = value
        
        return gates_and_logic
    
    def _apply_advanced_features(self, strategy: PatternStrategy, config: AdvancedStrategyConfig):
        """Apply advanced features to strategy"""
        # Apply timeframes
        if config.timeframes:
            strategy.timeframes = config.timeframes
        
        # Apply volatility profile
        if config.volatility_profile:
            strategy.volatility_profile = config.volatility_profile
        
        # Apply market regime
        if config.market_regime:
            strategy.market_regime = config.market_regime
        
        # Apply risk parameters
        if config.risk_params:
            strategy.risk_params = config.risk_params
        
        # Apply position sizing
        if config.position_sizing:
            strategy.position_sizing = config.position_sizing
    
    def _generate_id(self) -> str:
        """Generate unique ID for strategy"""
        from datetime import datetime
        return datetime.now().strftime('%H%M%S%f')[:-3]
    
    def validate_config(self, config: AdvancedStrategyConfig) -> List[str]:
        """Validate strategy configuration and return any errors"""
        errors = []
        
        # Validate patterns
        if not config.patterns:
            errors.append("At least one pattern must be specified")
        
        # Validate combination logic
        if config.combination_logic not in ['AND', 'OR', 'WEIGHTED']:
            errors.append("Combination logic must be AND, OR, or WEIGHTED")
        
        # Validate weights for weighted combination
        if config.combination_logic == 'WEIGHTED':
            if len(config.weights) != len(config.patterns):
                errors.append("Number of weights must match number of patterns for weighted combination")
            if abs(sum(config.weights) - 1.0) > 0.01:
                errors.append("Weights must sum to 1.0 for weighted combination")
        
        # Validate minimum actions required
        if config.min_actions_required < 1 or config.min_actions_required > len(config.patterns):
            errors.append("Minimum actions required must be between 1 and number of patterns")
        
        # Validate risk parameters
        if config.risk_params:
            if 'stop_loss_pct' in config.risk_params:
                sl = config.risk_params['stop_loss_pct']
                if sl <= 0 or sl > 0.5:
                    errors.append("Stop loss percentage must be between 0 and 0.5")
        
        return errors


# Global factory instance
comprehensive_factory = ComprehensiveStrategyFactory() 