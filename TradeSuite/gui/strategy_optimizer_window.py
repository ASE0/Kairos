"""
gui/strategy_optimizer_window.py
==============================
Window for AI-powered strategy optimization using genetic algorithms
"""

import logging
logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pyqtgraph as pg
from datetime import datetime
import json
import random
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import time

from strategies.strategy_builders import StrategyFactory, PatternStrategy, RiskStrategy
from patterns.candlestick_patterns import PatternFactory
from core.feature_quantification import *
from core.pattern_registry import registry


@dataclass
class StrategyGenome:
    """Represents a comprehensive strategy configuration for genetic algorithm"""
    # Pattern configuration
    patterns: List[str] = field(default_factory=list)
    pattern_params: Dict[str, Dict] = field(default_factory=dict)  # Pattern-specific parameters
    pattern_timeframes: Dict[str, List[str]] = field(default_factory=dict)  # Per-pattern timeframes
    
    # Filter configuration
    filters: List[Dict] = field(default_factory=list)
    filter_params: Dict[str, Dict] = field(default_factory=dict)  # Filter-specific parameters
    
    # Strategy combination
    combination_logic: str = 'AND'
    weights: List[float] = field(default_factory=list)
    min_actions_required: int = 1
    
    # Gates and execution logic
    gates: Dict = field(default_factory=dict)
    gate_params: Dict[str, Dict] = field(default_factory=dict)  # Gate-specific parameters
    execution_logic: Dict = field(default_factory=dict)  # Master equation, thresholds, etc.
    
    # Risk management
    risk_params: Dict = field(default_factory=dict)
    position_sizing: Dict = field(default_factory=dict)  # Kelly, fixed, dynamic
    
    # Advanced features
    timeframes: List[str] = field(default_factory=list)  # Global timeframes (legacy)
    volatility_profile: Dict = field(default_factory=dict)
    market_regime: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'patterns': self.patterns,
            'pattern_params': self.pattern_params,
            'pattern_timeframes': self.pattern_timeframes,
            'filters': self.filters,
            'filter_params': self.filter_params,
            'combination_logic': self.combination_logic,
            'weights': self.weights,
            'min_actions_required': self.min_actions_required,
            'gates': self.gates,
            'gate_params': self.gate_params,
            'execution_logic': self.execution_logic,
            'risk_params': self.risk_params,
            'position_sizing': self.position_sizing,
            'timeframes': self.timeframes,
            'volatility_profile': self.volatility_profile,
            'market_regime': self.market_regime
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class StrategyOptimizer:
    """Enhanced genetic algorithm for comprehensive strategy optimization"""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_strategies = []
        self.fitness_history = []
        
        # Auto-discover available building blocks
        self._discover_available_components()
    
    def _discover_available_components(self):
        """Automatically discover all available patterns, filters, gates, etc."""
        # Use the pattern registry for comprehensive discovery
        
        # Discover patterns
        self.available_patterns = self._discover_patterns()
        
        # Discover filters
        self.available_filters = self._discover_filters()
        
        # Discover gates
        self.available_gates = self._discover_gates()
        
        # Discover timeframes
        self.available_timeframes = self._discover_timeframes()
        
        # Discover risk strategies
        self.available_risk_strategies = self._discover_risk_strategies()
        
        # Discover position sizing methods
        self.available_position_sizing = self._discover_position_sizing()
        
        logger.info(f"AI Optimizer: Discovered {len(self.available_patterns)} patterns, {len(self.available_filters)} filters, {len(self.available_gates)} gates")
        logger.info(f"Available patterns: {self.available_patterns}")
        logger.info(f"Available gates: {self.available_gates}")
    
    def _discover_patterns(self) -> List[str]:
        """Discover all available patterns from main hub"""
        try:
            # Get patterns from pattern registry
            from core.pattern_registry import registry
            registry.refresh()
            
            # Get all pattern names from registry
            all_patterns = list(registry.patterns.keys())
            
            # Filter to only include main hub patterns (31 patterns)
            # These are the patterns created by the main hub
            main_hub_patterns = [
                'hammer', 'engulfing', 'ii_bars', 'double_wick', 'custom',
                'engulfing_bullish', 'engulfing_bearish', 'doji_standard', 
                'strong_body', 'weak_body', 'momentum_breakout', 'momentum_reversal',
                'high_volatility', 'low_volatility', 'support_bounce', 'resistance_rejection',
                'three_white_soldiers', 'three_black_crows', 'four_price_doji',
                'dragonfly_doji', 'gravestone_doji', 'volatility_expansion',
                'volatility_contraction', 'trend_continuation', 'trend_reversal',
                'gap_up', 'gap_down', 'consolidation', 'breakout', 'exhaustion',
                'accumulation', 'distribution'
            ]
            
            # Only use patterns that exist in the registry and are main hub patterns
            available_patterns = [p for p in main_hub_patterns if p in all_patterns]
            
            logger.info(f"AI Optimizer: Discovered {len(available_patterns)} main hub patterns")
            return available_patterns
            
        except Exception as e:
            logger.info(f"Error discovering patterns: {e}")
            # Fallback to basic patterns
            return ['hammer', 'engulfing', 'ii_bars', 'double_wick', 'custom']
    
    def _discover_filters(self) -> List[Dict]:
        """Discover all available filter types using registry"""
        filters = []
        for filter_type in registry.get_filter_types():
            filter_info = registry.get_filter_info(filter_type)
            if filter_info:
                filters.append({
                    'type': filter_type,
                    'name': filter_info.name,
                    'parameters': filter_info.parameters
                })
        return filters
    
    def _discover_gates(self) -> List[str]:
        """Discover all available gate types using registry"""
        return registry.get_gate_types()
    
    def _discover_timeframes(self) -> List[str]:
        """Discover available timeframes"""
        timeframes = [
            '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
        ]
        
        return timeframes
    
    def _discover_risk_strategies(self) -> List[str]:
        """Discover available risk strategy types"""
        risk_strategies = [
            'fixed_stop', 'atr_stop', 'kelly_stop', 'trailing_stop',
            'pattern_stop', 'volatility_stop', 'dynamic_stop'
        ]
        
        return risk_strategies
    
    def _discover_position_sizing(self) -> List[str]:
        """Discover available position sizing methods"""
        sizing_methods = [
            'fixed_percent', 'kelly_criterion', 'risk_per_trade',
            'volatility_adjusted', 'dynamic_sizing', 'portfolio_heat'
        ]
        
        return sizing_methods
    
    def _get_custom_patterns(self) -> List[str]:
        """Get any custom patterns that have been created"""
        # This would check the pattern registry or workspace
        # For now, return empty list
        return []
    
    def create_initial_population(self):
        """Create initial random population with comprehensive parameters"""
        self.population = []
        
        for _ in range(self.population_size):
            genome = self._create_random_genome()
            self.population.append(genome)
    
    def _create_random_genome(self) -> StrategyGenome:
        """Create a random strategy genome"""
        # Random number of patterns (1-5)
        num_patterns = random.randint(1, min(5, len(self.available_patterns)))
        selected_patterns = random.sample(self.available_patterns, num_patterns)
        
        # Generate pattern parameters and timeframes
        pattern_params = {}
        pattern_timeframes = {}
        
        for pattern in selected_patterns:
            pattern_params[pattern] = self._generate_random_pattern_params(pattern)
            # Generate per-pattern timeframes (1-3 timeframes per pattern)
            num_tf = random.randint(1, min(3, len(self.available_timeframes)))
            pattern_timeframes[pattern] = random.sample(self.available_timeframes, num_tf)
        
        # Random filters (0-3)
        num_filters = random.randint(0, min(3, len(self.available_filters)))
        filters = []
        filter_params = {}
        if num_filters > 0:
            selected_filters = random.sample(self.available_filters, num_filters)
            for filter_info in selected_filters:
                filters.append(filter_info)
                filter_params[filter_info['name']] = self._generate_random_filter_params(filter_info['type'])
        
        # Random combination logic
        combination_logic = random.choice(['AND', 'OR', 'WEIGHTED'])
        
        # Random weights for weighted combination
        weights = []
        if combination_logic == 'WEIGHTED':
            weights = [random.uniform(0.1, 1.0) for _ in selected_patterns]
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # Random minimum actions required
        min_actions_required = random.randint(1, len(selected_patterns))
        
        # Random gates (0-3)
        num_gates = random.randint(0, min(3, len(self.available_gates)))
        gates = {}
        gate_params = {}
        if num_gates > 0:
            selected_gates = random.sample(self.available_gates, num_gates)
            for gate in selected_gates:
                gates[gate] = True
                gate_params[gate] = self._generate_random_gate_params(gate)
        
        # Random execution logic
        execution_logic = self._generate_random_execution_logic()
        
        # Random risk parameters
        risk_params = self._generate_random_risk_params()
        
        # Random position sizing
        position_sizing = self._generate_random_position_sizing()
        
        # Random global timeframes (legacy support)
        num_timeframes = random.randint(1, min(3, len(self.available_timeframes)))
        timeframes = random.sample(self.available_timeframes, num_timeframes)
        
        # Random volatility profile
        volatility_profile = self._generate_random_volatility_profile()
        
        # Random market regime
        market_regime = self._generate_random_market_regime()
        
        return StrategyGenome(
            patterns=selected_patterns,
            pattern_params=pattern_params,
            pattern_timeframes=pattern_timeframes,
            filters=filters,
            filter_params=filter_params,
            combination_logic=combination_logic,
            weights=weights,
            min_actions_required=min_actions_required,
            gates=gates,
            gate_params=gate_params,
            execution_logic=execution_logic,
            risk_params=risk_params,
            position_sizing=position_sizing,
            timeframes=timeframes,
            volatility_profile=volatility_profile,
            market_regime=market_regime
        )
    
    def _generate_random_pattern_params(self, pattern_name: str) -> Dict:
        """Generate random parameters for a specific pattern"""
        params = {}
        
        if pattern_name == 'engulfing':
            params = {
                'pattern_type': random.choice(['bullish', 'bearish', 'both'])
            }
        elif pattern_name == 'ii_bars':
            params = {
                'min_bars': random.randint(2, 5)
            }
        elif pattern_name == 'double_wick':
            params = {
                'min_wick_ratio': random.uniform(0.2, 0.5),
                'max_body_ratio': random.uniform(0.3, 0.6)
            }
        elif pattern_name == 'doji':
            params = {
                'max_body_ratio': random.uniform(0.05, 0.15)
            }
        
        return params
    
    def _generate_random_filter_params(self, filter_type: str) -> Dict:
        """Generate random parameters for a specific filter type"""
        params = {}
        
        if filter_type == 'volume':
            params = {
                'min_volume': random.randint(500, 5000),
                'volume_ratio': random.uniform(1.2, 2.0),
                'vwap_distance': random.uniform(0.005, 0.02)
            }
        elif filter_type == 'time':
            params = {
                'start_time': f"{random.randint(9, 10):02d}:{random.randint(0, 59):02d}",
                'end_time': f"{random.randint(15, 16):02d}:{random.randint(0, 59):02d}",
                'session': random.choice(['regular', 'pre_market', 'after_hours'])
            }
        elif filter_type == 'volatility':
            params = {
                'min_atr': random.uniform(0.005, 0.02),
                'max_atr': random.uniform(0.03, 0.08),
                'atr_percentile': random.randint(25, 75)
            }
        elif filter_type == 'momentum':
            params = {
                'min_momentum': random.uniform(0.0005, 0.002),
                'rsi_range': [random.randint(20, 40), random.randint(60, 80)],
                'macd_signal': random.choice(['bullish', 'bearish', 'both'])
            }
        
        return params
    
    def _generate_random_gate_params(self, gate_name: str) -> Dict:
        """Generate random parameters for a specific gate"""
        params = {}
        
        if gate_name == 'location_gate':
            params = {
                'fvg_tolerance': random.uniform(0.005, 0.02),
                'sr_tolerance': random.uniform(0.01, 0.03)
            }
        elif gate_name == 'volatility_gate':
            params = {
                'min_atr_ratio': random.uniform(0.005, 0.02),
                'max_atr_ratio': random.uniform(0.03, 0.08)
            }
        elif gate_name == 'regime_gate':
            params = {
                'momentum_threshold': random.uniform(0.01, 0.03),
                'trend_probability': random.uniform(0.2, 0.5)
            }
        elif gate_name == 'bayesian_gate':
            params = {
                'min_state_probability': random.uniform(0.2, 0.6),
                'confidence_threshold': random.uniform(0.5, 0.8)
            }
        
        return params
    
    def _generate_random_execution_logic(self) -> Dict:
        """Generate random execution logic parameters"""
        return {
            'master_equation': random.choice([True, False]),
            'exec_threshold': random.uniform(0.4, 0.8),
            'mmrs_threshold': random.uniform(0.3, 0.7),
            'volatility_veto': random.choice([True, False]),
            'alignment': random.choice([True, False])
        }
    
    def _generate_random_risk_params(self) -> Dict:
        """Generate random risk parameters"""
        return {
            'stop_loss_pct': random.uniform(0.01, 0.05),
            'risk_reward_ratio': random.uniform(1.5, 3.0),
            'position_size_pct': random.uniform(0.01, 0.05),
            'max_risk_per_trade': random.uniform(0.01, 0.03),
            'trailing_stop_pct': random.uniform(0.005, 0.02)
        }
    
    def _generate_random_position_sizing(self) -> Dict:
        """Generate random position sizing parameters"""
        method = random.choice(self.available_position_sizing)
        params = {'method': method}
        
        if method == 'kelly_criterion':
            params.update({
                'win_probability': random.uniform(0.4, 0.7),
                'win_loss_ratio': random.uniform(1.5, 3.0)
            })
        elif method == 'risk_per_trade':
            params.update({
                'risk_percent': random.uniform(0.01, 0.03)
            })
        elif method == 'volatility_adjusted':
            params.update({
                'volatility_multiplier': random.uniform(0.5, 2.0)
            })
        
        return params
    
    def _generate_random_volatility_profile(self) -> Dict:
        """Generate random volatility profile parameters"""
        return {
            'profile_type': random.choice(['normal', 'high', 'low']),
            'adjustment_factor': random.uniform(0.5, 1.5)
        }
    
    def _generate_random_market_regime(self) -> Dict:
        """Generate random market regime parameters"""
        return {
            'regime_type': random.choice(['trending', 'ranging', 'volatile']),
            'detection_method': random.choice(['momentum', 'volatility', 'correlation'])
        }
    
    def evaluate_fitness(self, genome: StrategyGenome, data: pd.DataFrame) -> float:
        """Evaluate fitness of a strategy genome"""
        try:
            # Create strategy from genome
            strategy = self._create_strategy_from_genome(genome)
            
            # Run backtest
            from strategies.strategy_builders import BacktestEngine
            engine = BacktestEngine()
            results = engine.run_backtest(strategy, data)
            
            # Calculate fitness score
            fitness = self._calculate_fitness_score(results)
            return fitness
            
        except Exception as e:
            logger.info(f"Error evaluating genome: {e}")
            return 0.0
    
    def _create_strategy_from_genome(self, genome: StrategyGenome) -> PatternStrategy:
        """Create a strategy from genome using comprehensive factory"""
        try:
            from strategies.comprehensive_strategy_factory import ComprehensiveStrategyFactory, AdvancedStrategyConfig
            
            factory = ComprehensiveStrategyFactory()
            
            # Create advanced config from genome
            config = AdvancedStrategyConfig(
                patterns=genome.patterns,
                pattern_params=genome.pattern_params,
                pattern_timeframes=genome.pattern_timeframes,
                filters=genome.filters,
                filter_params=genome.filter_params,
                combination_logic=genome.combination_logic,
                weights=genome.weights,
                min_actions_required=genome.min_actions_required,
                gates=genome.gates,
                gate_params=genome.gate_params,
                execution_logic=genome.execution_logic,
                risk_params=genome.risk_params,
                position_sizing=genome.position_sizing,
                timeframes=genome.timeframes,
                volatility_profile=genome.volatility_profile,
                market_regime=genome.market_regime
            )
            
            # Create strategy using factory
            strategy = factory.create_strategy(config)
            return strategy
            
        except Exception as e:
            logger.info(f"Warning: Could not create strategy using comprehensive factory: {e}")
            # Fallback to simplified strategy creation
            return self._create_simplified_strategy(genome)
    
    def _create_simplified_strategy(self, genome: StrategyGenome) -> PatternStrategy:
        """Create a simplified strategy as fallback"""
        # Create actions for each pattern with parameters
        actions = []
        for pattern_name in genome.patterns:
            # Create pattern instance with parameters
            pattern = self._create_pattern_with_params(pattern_name, genome.pattern_params.get(pattern_name, {}))
            
            # Create action with filters
            from strategies.strategy_builders import Action
            action = Action(
                name=f"action_{pattern_name}",
                pattern=pattern,
                filters=genome.filters
            )
            actions.append(action)
        
        # Create strategy with all parameters
        strategy = StrategyFactory.create_pattern_strategy(
            name=f"optimized_strategy_{datetime.now().strftime('%H%M%S')}",
            actions=actions,
            combination_logic=genome.combination_logic,
            weights=genome.weights,
            min_actions_required=genome.min_actions_required,
            gates_and_logic=self._create_gates_and_logic(genome)
        )
        
        # Apply advanced features
        self._apply_advanced_features(strategy, genome)
        
        return strategy
    
    def _create_pattern_with_params(self, pattern_name: str, params: Dict) -> Any:
        """Create a pattern instance with specific parameters using registry"""
        try:
            # Try to create pattern using registry first
            pattern = registry.create_pattern(pattern_name, **params)
            return pattern
        except Exception as e:
            logger.info(f"Warning: Could not create pattern '{pattern_name}' using registry: {e}")
            
            # Fallback to manual pattern creation
            from patterns.candlestick_patterns import (
                EngulfingPattern, IIBarsPattern, 
                DoubleWickPattern, CustomPattern
            )
            from core.data_structures import TimeRange
            
            # Default timeframes
            timeframes = [TimeRange(15, 'm')]
            
            if pattern_name == 'engulfing':
                pattern_type = params.get('pattern_type', 'both')
                return EngulfingPattern(timeframes, pattern_type)
                
            elif pattern_name == 'ii_bars':
                min_bars = params.get('min_bars', 2)
                return IIBarsPattern(timeframes, min_bars)
                
            elif pattern_name == 'double_wick':
                min_wick_ratio = params.get('min_wick_ratio', 0.3)
                max_body_ratio = params.get('max_body_ratio', 0.4)
                return DoubleWickPattern(timeframes, min_wick_ratio, max_body_ratio)
                
            elif pattern_name == 'doji':
                max_body_ratio = params.get('max_body_ratio', 0.1)
                # Create custom doji pattern
                ohlc_ratios = [
                    {'body_ratio': (0, max_body_ratio)}
                ]
                return CustomPattern("doji", timeframes, ohlc_ratios)
                
            else:
                # Default pattern creation
                return PatternFactory.create_pattern(pattern_name)
    
    def _create_gates_and_logic(self, genome: StrategyGenome) -> Dict:
        """Create gates and logic configuration from genome"""
        gates_and_logic = {}
        
        # Add gates
        for gate_name, enabled in genome.gates.items():
            if enabled:
                gates_and_logic[gate_name] = True
        
        # Add execution logic
        execution_logic = genome.execution_logic
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
        
        return gates_and_logic
    
    def _apply_advanced_features(self, strategy: PatternStrategy, genome: StrategyGenome):
        """Apply advanced features to strategy"""
        # Apply timeframes
        if genome.timeframes:
            # This would set the strategy to use multiple timeframes
            pass
        
        # Apply volatility profile
        if genome.volatility_profile:
            # This would set volatility-based adjustments
            pass
        
        # Apply market regime
        if genome.market_regime:
            # This would set regime-specific behavior
            pass
        
        # Apply risk parameters
        if genome.risk_params:
            # This would set risk management parameters
            pass
        
        # Apply position sizing
        if genome.position_sizing:
            # This would set position sizing method
            pass
    
    def _calculate_fitness_score(self, results: Dict) -> float:
        """Calculate fitness score from backtest results"""
        sharpe = results.get('sharpe_ratio', 0)
        profit_factor = results.get('profit_factor', 0)
        max_dd = results.get('max_drawdown', 1.0)
        total_return = results.get('total_return', 0)
        total_trades = results.get('total_trades', 0)
        
        # Penalize strategies with too few trades
        if total_trades < 10:
            return 0.0
        
        # Multi-objective fitness
        fitness = (
            max(0, sharpe) * 0.3 +
            min(profit_factor, 5.0) * 0.2 +
            (1 - max_dd) * 0.2 +
            max(0, total_return) * 0.3
        )
        
        return fitness
    
    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Crossover two parent genomes with comprehensive parameters"""
        # Crossover patterns
        if random.random() < 0.5:
            patterns = parent1.patterns.copy()
            pattern_params = parent1.pattern_params.copy()
            pattern_timeframes = parent1.pattern_timeframes.copy()
        else:
            patterns = parent2.patterns.copy()
            pattern_params = parent2.pattern_params.copy()
            pattern_timeframes = parent2.pattern_timeframes.copy()
        
        # Crossover pattern parameters
        for pattern in patterns:
            if pattern in parent1.pattern_params and pattern in parent2.pattern_params:
                if random.random() < 0.5:
                    pattern_params[pattern] = parent1.pattern_params[pattern].copy()
                else:
                    pattern_params[pattern] = parent2.pattern_params[pattern].copy()
        
        # Crossover filters
        if random.random() < 0.5:
            filters = parent1.filters.copy()
            filter_params = parent1.filter_params.copy()
        else:
            filters = parent2.filters.copy()
            filter_params = parent2.filter_params.copy()
        
        # Crossover filter parameters
        for filter_config in filters:
            filter_type = filter_config['type']
            if filter_type in parent1.filter_params and filter_type in parent2.filter_params:
                if random.random() < 0.5:
                    filter_params[filter_type] = parent1.filter_params[filter_type].copy()
                else:
                    filter_params[filter_type] = parent2.filter_params[filter_type].copy()
        
        # Crossover combination logic
        combination_logic = random.choice([parent1.combination_logic, parent2.combination_logic])
        
        # Crossover weights
        if combination_logic == 'WEIGHTED':
            weights = []
            max_len = max(len(parent1.weights), len(parent2.weights))
            for i in range(max_len):
                if i < len(parent1.weights) and i < len(parent2.weights):
                    weights.append(random.choice([parent1.weights[i], parent2.weights[i]]))
                elif i < len(parent1.weights):
                    weights.append(parent1.weights[i])
                else:
                    weights.append(parent2.weights[i])
        else:
            weights = []
        
        # Crossover minimum actions required
        min_actions_required = random.choice([parent1.min_actions_required, parent2.min_actions_required])
        
        # Crossover gates and gate parameters
        gates = {}
        gate_params = {}
        all_gates = set(parent1.gates.keys()) | set(parent2.gates.keys())
        for gate in all_gates:
            if random.random() < 0.5:
                gates[gate] = parent1.gates.get(gate, False)
                if gate in parent1.gate_params:
                    gate_params[gate] = parent1.gate_params[gate].copy()
            else:
                gates[gate] = parent2.gates.get(gate, False)
                if gate in parent2.gate_params:
                    gate_params[gate] = parent2.gate_params[gate].copy()
        
        # Crossover execution logic
        execution_logic = {}
        for key in set(parent1.execution_logic.keys()) | set(parent2.execution_logic.keys()):
            if random.random() < 0.5:
                execution_logic[key] = parent1.execution_logic.get(key)
            else:
                execution_logic[key] = parent2.execution_logic.get(key)
        
        # Crossover risk parameters
        risk_params = {}
        for key in set(parent1.risk_params.keys()) | set(parent2.risk_params.keys()):
            if random.random() < 0.5:
                risk_params[key] = parent1.risk_params.get(key)
            else:
                risk_params[key] = parent2.risk_params.get(key)
        
        # Crossover position sizing
        if random.random() < 0.5:
            position_sizing = parent1.position_sizing.copy()
        else:
            position_sizing = parent2.position_sizing.copy()
        
        # Crossover timeframes
        if random.random() < 0.5:
            timeframes = parent1.timeframes.copy()
        else:
            timeframes = parent2.timeframes.copy()
        
        # Crossover volatility profile
        if random.random() < 0.5:
            volatility_profile = parent1.volatility_profile.copy()
        else:
            volatility_profile = parent2.volatility_profile.copy()
        
        # Crossover market regime
        if random.random() < 0.5:
            market_regime = parent1.market_regime.copy()
        else:
            market_regime = parent2.market_regime.copy()
        
        return StrategyGenome(
            patterns=patterns,
            pattern_params=pattern_params,
            pattern_timeframes=pattern_timeframes,
            filters=filters,
            filter_params=filter_params,
            combination_logic=combination_logic,
            weights=weights,
            min_actions_required=min_actions_required,
            gates=gates,
            gate_params=gate_params,
            execution_logic=execution_logic,
            risk_params=risk_params,
            position_sizing=position_sizing,
            timeframes=timeframes,
            volatility_profile=volatility_profile,
            market_regime=market_regime
        )
    
    def mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Mutate a genome with comprehensive parameters"""
        mutated = StrategyGenome(
            patterns=genome.patterns.copy(),
            pattern_params=genome.pattern_params.copy(),
            pattern_timeframes=genome.pattern_timeframes.copy(),
            filters=genome.filters.copy(),
            filter_params=genome.filter_params.copy(),
            combination_logic=genome.combination_logic,
            weights=genome.weights.copy(),
            min_actions_required=genome.min_actions_required,
            gates=genome.gates.copy(),
            gate_params=genome.gate_params.copy(),
            execution_logic=genome.execution_logic.copy(),
            risk_params=genome.risk_params.copy(),
            position_sizing=genome.position_sizing.copy(),
            timeframes=genome.timeframes.copy(),
            volatility_profile=genome.volatility_profile.copy(),
            market_regime=genome.market_regime.copy()
        )
        
        # Mutate patterns
        if random.random() < self.mutation_rate:
            if len(mutated.patterns) > 1 and random.random() < 0.5:
                # Remove a pattern
                removed_pattern = mutated.patterns.pop(random.randint(0, len(mutated.patterns)-1))
                if removed_pattern in mutated.pattern_params:
                    del mutated.pattern_params[removed_pattern]
            else:
                # Add a pattern
                new_pattern = random.choice(self.available_patterns)
                if new_pattern not in mutated.patterns:
                    mutated.patterns.append(new_pattern)
                    mutated.pattern_params[new_pattern] = self._generate_random_pattern_params(new_pattern)
        
        # Mutate pattern parameters
        for pattern in mutated.patterns:
            if random.random() < self.mutation_rate:
                mutated.pattern_params[pattern] = self._generate_random_pattern_params(pattern)
        
        # Mutate filters
        if random.random() < self.mutation_rate:
            if len(mutated.filters) > 0 and random.random() < 0.5:
                # Remove a filter
                removed_filter = mutated.filters.pop(random.randint(0, len(mutated.filters)-1))
                filter_type = removed_filter['type']
                if filter_type in mutated.filter_params:
                    del mutated.filter_params[filter_type]
            else:
                # Add a filter
                new_filter = random.choice(self.available_filters)
                if new_filter not in mutated.filters:
                    mutated.filters.append(new_filter)
                    filter_type = new_filter['type']
                    mutated.filter_params[filter_type] = self._generate_random_filter_params(filter_type)
        
        # Mutate filter parameters
        for filter_config in mutated.filters:
            filter_type = filter_config['type']
            if random.random() < self.mutation_rate:
                mutated.filter_params[filter_type] = self._generate_random_filter_params(filter_type)
        
        # Mutate combination logic
        if random.random() < self.mutation_rate:
            mutated.combination_logic = random.choice(['AND', 'OR', 'WEIGHTED'])
        
        # Mutate minimum actions required
        if random.random() < self.mutation_rate:
            mutated.min_actions_required = random.randint(1, len(mutated.patterns))
        
        # Mutate gates
        if random.random() < self.mutation_rate:
            # Toggle random gates
            for gate in self.available_gates:
                if random.random() < 0.3:
                    if gate in mutated.gates:
                        del mutated.gates[gate]
                        if gate in mutated.gate_params:
                            del mutated.gate_params[gate]
                    else:
                        mutated.gates[gate] = True
                        mutated.gate_params[gate] = self._generate_random_gate_params(gate)
        
        # Mutate gate parameters
        for gate in mutated.gates:
            if random.random() < self.mutation_rate:
                mutated.gate_params[gate] = self._generate_random_gate_params(gate)
        
        # Mutate execution logic
        if random.random() < self.mutation_rate:
            mutated.execution_logic = self._generate_random_execution_logic()
        
        # Mutate risk parameters
        for key in mutated.risk_params:
            if random.random() < self.mutation_rate:
                mutated.risk_params = self._generate_random_risk_params()
                break
        
        # Mutate position sizing
        if random.random() < self.mutation_rate:
            mutated.position_sizing = self._generate_random_position_sizing()
        
        # Mutate timeframes
        if random.random() < self.mutation_rate:
            num_timeframes = random.randint(1, min(3, len(self.available_timeframes)))
            mutated.timeframes = random.sample(self.available_timeframes, num_timeframes)
        
        # Mutate volatility profile
        if random.random() < self.mutation_rate:
            mutated.volatility_profile = self._generate_random_volatility_profile()
        
        # Mutate market regime
        if random.random() < self.mutation_rate:
            mutated.market_regime = self._generate_random_market_regime()
        
        return mutated
    
    def evolve(self, data: pd.DataFrame, progress_callback=None) -> List[StrategyGenome]:
        """Run genetic algorithm evolution"""
        # Create initial population
        self.create_initial_population()
        
        for generation in range(self.generations):
            # Evaluate fitness for all genomes
            fitness_scores = []
            for i, genome in enumerate(self.population):
                fitness = self.evaluate_fitness(genome, data)
                fitness_scores.append((fitness, genome))
                
                # Update progress
                if progress_callback:
                    progress = (generation * len(self.population) + i + 1) / (self.generations * len(self.population))
                    progress_callback(progress, generation, i, fitness)
            
            # Sort by fitness
            fitness_scores.sort(reverse=True)
            
            # Keep best strategies
            self.best_strategies = [genome for _, genome in fitness_scores[:10]]
            
            # Record best fitness
            best_fitness = fitness_scores[0][0]
            self.fitness_history.append(best_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism: keep top 20%
            elite_count = max(1, self.population_size // 5)
            new_population.extend([genome for _, genome in fitness_scores[:elite_count]])
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        return self.best_strategies
    
    def _tournament_selection(self, fitness_scores, tournament_size=3):
        """Tournament selection for parent selection"""
        tournament = random.sample(fitness_scores, tournament_size)
        return max(tournament, key=lambda x: x[0])[1]

    def refresh_components(self):
        """Refresh the component discovery to include any new patterns, filters, gates, etc."""
        logger.info("Refreshing AI Optimizer components...")
        
        # Refresh the optimizer components
        self._discover_available_components()
        
        logger.info(f"AI Optimizer refreshed: {len(self.available_patterns)} patterns, {len(self.available_filters)} filters, {len(self.available_gates)} gates")
    
    def update_components_display(self):
        """Update the components display with current discoveries"""
        summary = self.get_available_components_summary()
        
        display_text = f"""📊 Discovered Components:
        
🎯 Patterns ({len(summary['patterns'])}):
{', '.join(summary['patterns'][:10])}{'...' if len(summary['patterns']) > 10 else ''}

🔍 Filters ({len(summary['filters'])}):
{', '.join(summary['filters'][:10])}{'...' if len(summary['filters']) > 10 else ''}

🚪 Gates ({len(summary['gates'])}):
{', '.join(summary['gates'][:10])}{'...' if len(summary['gates']) > 10 else ''}

⏰ Timeframes ({len(summary['timeframes'])}):
{', '.join(summary['timeframes'])}

💰 Risk Strategies ({len(summary['risk_strategies'])}):
{', '.join(summary['risk_strategies'])}

📈 Position Sizing ({len(summary['position_sizing'])}):
{', '.join(summary['position_sizing'])}"""
        
        self.components_text.setPlainText(display_text)
    
    def get_available_components_summary(self) -> Dict[str, List[str]]:
        """Get a summary of all available components for the UI"""
        return {
            'patterns': self.available_patterns,
            'filters': [f['type'] for f in self.available_filters],
            'gates': self.available_gates,
            'timeframes': self.available_timeframes,
            'risk_strategies': self.available_risk_strategies,
            'position_sizing': self.available_position_sizing
        }


class StrategyOptimizerWindow(QMainWindow):
    """Window for AI strategy optimization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("AI Strategy Optimizer")
        self.setGeometry(400, 200, 1000, 800)
        
        # Optimization state
        self.optimizer = None
        self.is_optimizing = False
        self.current_data = None
        
        # Setup UI
        self._setup_ui()
        
        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QPushButton {
                background-color: #cccccc;
                border: 1px solid #888;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #b0b0b0;
            }
            QPushButton:disabled {
                background-color: #eeeeee;
                color: #888888;
            }
        """)
        
        # Initialize optimizer
        self.optimizer = StrategyOptimizer()
        
        # Update components display
        self.update_components_display()
    
    def _setup_ui(self):
        """Setup UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("AI Strategy Optimizer")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Available components section
        components_group = QGroupBox("Available Components")
        components_layout = QVBoxLayout()
        
        # Components summary
        self.components_text = QTextEdit()
        self.components_text.setMaximumHeight(150)
        self.components_text.setReadOnly(True)
        components_layout.addWidget(QLabel("Discovered Components:"))
        components_layout.addWidget(self.components_text)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Components")
        refresh_btn.clicked.connect(self.refresh_components)
        components_layout.addWidget(refresh_btn)
        
        components_group.setLayout(components_layout)
        main_layout.addWidget(components_group)
        
        # Optimization parameters
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QGridLayout()
        
        # Population size
        params_layout.addWidget(QLabel("Population Size:"), 0, 0)
        self.population_size_spin = QSpinBox()
        self.population_size_spin.setRange(10, 200)
        self.population_size_spin.setValue(50)
        params_layout.addWidget(self.population_size_spin, 0, 1)
        
        # Generations
        params_layout.addWidget(QLabel("Generations:"), 1, 0)
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(10, 500)
        self.generations_spin.setValue(100)
        params_layout.addWidget(self.generations_spin, 1, 1)
        
        # Mutation rate
        params_layout.addWidget(QLabel("Mutation Rate:"), 2, 0)
        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.01, 0.5)
        self.mutation_rate_spin.setValue(0.1)
        self.mutation_rate_spin.setSingleStep(0.01)
        params_layout.addWidget(self.mutation_rate_spin, 2, 1)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Configuration section
        config_group = self._create_config_section()
        main_layout.addWidget(config_group)
        
        # Progress section
        progress_group = self._create_progress_section()
        main_layout.addWidget(progress_group)
        
        # Results section
        results_group = self._create_results_section()
        main_layout.addWidget(results_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self._start_optimization)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_optimization)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("Save Best Strategy")
        self.save_btn.clicked.connect(self._save_best_strategy)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        layout.addLayout(main_layout)
    
    def _create_config_section(self) -> QGroupBox:
        """Create configuration section"""
        group = QGroupBox("Optimization Configuration")
        layout = QGridLayout()
        
        # Population size
        layout.addWidget(QLabel("Population Size:"), 0, 0)
        self.population_size = QSpinBox()
        self.population_size.setRange(10, 200)
        self.population_size.setValue(50)
        layout.addWidget(self.population_size, 0, 1)
        
        # Generations
        layout.addWidget(QLabel("Generations:"), 0, 2)
        self.generations = QSpinBox()
        self.generations.setRange(10, 500)
        self.generations.setValue(100)
        layout.addWidget(self.generations, 0, 3)
        
        # Mutation rate
        layout.addWidget(QLabel("Mutation Rate:"), 1, 0)
        self.mutation_rate = QDoubleSpinBox()
        self.mutation_rate.setRange(0.01, 0.5)
        self.mutation_rate.setValue(0.1)
        self.mutation_rate.setSingleStep(0.01)
        layout.addWidget(self.mutation_rate, 1, 1)
        
        # Dataset selection
        layout.addWidget(QLabel("Dataset:"), 1, 2)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Select Dataset...")
        layout.addWidget(self.dataset_combo, 1, 3)
        
        # Available patterns
        layout.addWidget(QLabel("Available Patterns:"), 2, 0)
        self.patterns_list = QListWidget()
        self.patterns_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        patterns = ['engulfing', 'ii_bars', 'double_wick', 'doji']
        for pattern in patterns:
            item = QListWidgetItem(pattern)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.patterns_list.addItem(item)
        layout.addWidget(self.patterns_list, 2, 1, 1, 3)
        
        group.setLayout(layout)
        return group
    
    def _create_progress_section(self) -> QGroupBox:
        """Create progress section"""
        group = QGroupBox("Optimization Progress")
        layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status labels
        status_layout = QHBoxLayout()
        
        self.generation_label = QLabel("Generation: 0/0")
        status_layout.addWidget(self.generation_label)
        
        self.fitness_label = QLabel("Best Fitness: 0.0")
        status_layout.addWidget(self.fitness_label)
        
        self.strategies_label = QLabel("Strategies Evaluated: 0")
        status_layout.addWidget(self.strategies_label)
        
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # Fitness history plot
        self.fitness_plot = pg.PlotWidget()
        self.fitness_plot.setLabel('left', 'Best Fitness')
        self.fitness_plot.setLabel('bottom', 'Generation')
        layout.addWidget(self.fitness_plot)
        
        group.setLayout(layout)
        return group
    
    def _create_results_section(self) -> QGroupBox:
        """Create results section"""
        group = QGroupBox("Best Strategies")
        layout = QVBoxLayout()
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            'Rank', 'Fitness', 'Patterns', 'Filters', 'Logic', 'Risk Params', 'Gates'
        ])
        layout.addWidget(self.results_table)
        
        group.setLayout(layout)
        return group
    
    def _start_optimization(self):
        """Start the optimization process"""
        if self.is_optimizing:
            return
        
        # Get configuration
        population_size = self.population_size_spin.value()
        generations = self.generations_spin.value()
        mutation_rate = self.mutation_rate_spin.value()
        
        # Get selected dataset
        if self.dataset_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select a dataset")
            return
        
        # Get selected patterns
        selected_patterns = []
        for i in range(self.patterns_list.count()):
            item = self.patterns_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_patterns.append(item.text())
        
        if not selected_patterns:
            QMessageBox.warning(self, "Warning", "Please select at least one pattern")
            return
        
        # Get dataset from parent window
        if self.parent_window and hasattr(self.parent_window, 'datasets'):
            dataset_name = self.dataset_combo.currentText()
            if dataset_name in self.parent_window.datasets:
                self.current_data = self.parent_window.datasets[dataset_name]['data']
            else:
                QMessageBox.warning(self, "Warning", f"Dataset '{dataset_name}' not found")
                return
        else:
            QMessageBox.warning(self, "Warning", "No datasets available")
            return
        
        # Create optimizer
        self.optimizer = StrategyOptimizer(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        
        # Update available patterns
        self.optimizer.available_patterns = selected_patterns
        
        # Start optimization in background thread
        self.is_optimizing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.fitness_plot.clear()
        self.results_table.setRowCount(0)
        
        # Start optimization thread
        self.optimization_thread = OptimizationThread(
            self.optimizer, self.current_data
        )
        self.optimization_thread.progress_updated.connect(self._update_progress)
        self.optimization_thread.optimization_complete.connect(self._optimization_complete)
        self.optimization_thread.start()
    
    def _stop_optimization(self):
        """Stop the optimization process"""
        if hasattr(self, 'optimization_thread'):
            self.optimization_thread.stop()
        
        self.is_optimizing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _update_progress(self, progress, generation, strategy_index, fitness):
        """Update progress display"""
        self.progress_bar.setValue(int(progress * 100))
        self.generation_label.setText(f"Generation: {generation + 1}/{self.generations_spin.value()}")
        self.fitness_label.setText(f"Best Fitness: {fitness:.3f}")
        self.strategies_label.setText(f"Strategies Evaluated: {strategy_index + 1}")
        
        # Update fitness plot
        if hasattr(self.optimizer, 'fitness_history') and self.optimizer.fitness_history:
            self.fitness_plot.clear()
            x = list(range(len(self.optimizer.fitness_history)))
            self.fitness_plot.plot(x, self.optimizer.fitness_history, pen='b')
    
    def _optimization_complete(self, best_strategies):
        """Handle optimization completion"""
        self.is_optimizing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        # Update results table
        self._update_results_table(best_strategies)
        
        QMessageBox.information(self, "Optimization Complete", 
                              f"Found {len(best_strategies)} best strategies!")
    
    def _update_results_table(self, strategies):
        """Update results table with best strategies"""
        self.results_table.setRowCount(len(strategies))
        
        for i, strategy in enumerate(strategies):
            # Rank
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            # Fitness (placeholder - would need to recalculate)
            self.results_table.setItem(i, 1, QTableWidgetItem("N/A"))
            
            # Patterns
            patterns_text = ", ".join(strategy.patterns)
            self.results_table.setItem(i, 2, QTableWidgetItem(patterns_text))
            
            # Filters
            filters_text = ", ".join([f.get('type', 'unknown') for f in strategy.filters])
            self.results_table.setItem(i, 3, QTableWidgetItem(filters_text))
            
            # Logic
            self.results_table.setItem(i, 4, QTableWidgetItem(strategy.combination_logic))
            
            # Risk params
            risk_text = f"SL:{strategy.risk_params.get('stop_loss_pct', 0):.2%}"
            self.results_table.setItem(i, 5, QTableWidgetItem(risk_text))
            
            # Gates
            gates_text = ", ".join(strategy.gates.keys())
            self.results_table.setItem(i, 6, QTableWidgetItem(gates_text))
    
    def _save_best_strategy(self):
        """Save the best strategy"""
        if not self.optimizer or not self.optimizer.best_strategies:
            return
        
        best_strategy = self.optimizer.best_strategies[0]
        
        # Create actual strategy object
        strategy = self.optimizer._create_strategy_from_genome(best_strategy)
        
        # Add to parent window
        if self.parent_window and hasattr(self.parent_window, 'on_strategy_created'):
            self.parent_window.on_strategy_created(strategy)
        
        QMessageBox.information(self, "Strategy Saved", 
                              f"Best strategy '{strategy.name}' has been saved!")

    def refresh_components(self):
        """Refresh the component discovery to include any new patterns, filters, gates, etc."""
        logger.info("Refreshing AI Optimizer components...")
        
        # Refresh the optimizer components
        self.optimizer.refresh_components()
        
        # Update the display
        self.update_components_display()
        
        logger.info(f"AI Optimizer refreshed: {len(self.optimizer.available_patterns)} patterns, {len(self.optimizer.available_filters)} filters, {len(self.optimizer.available_gates)} gates")
    
    def update_components_display(self):
        """Update the components display with current discoveries"""
        summary = self.optimizer.get_available_components_summary()
        
        display_text = f"""📊 Discovered Components:
        
🎯 Patterns ({len(summary['patterns'])}):
{', '.join(summary['patterns'][:10])}{'...' if len(summary['patterns']) > 10 else ''}

🔍 Filters ({len(summary['filters'])}):
{', '.join(summary['filters'][:10])}{'...' if len(summary['filters']) > 10 else ''}

🚪 Gates ({len(summary['gates'])}):
{', '.join(summary['gates'][:10])}{'...' if len(summary['gates']) > 10 else ''}

⏰ Timeframes ({len(summary['timeframes'])}):
{', '.join(summary['timeframes'])}

💰 Risk Strategies ({len(summary['risk_strategies'])}):
{', '.join(summary['risk_strategies'])}

📈 Position Sizing ({len(summary['position_sizing'])}):
{', '.join(summary['position_sizing'])}"""
        
        self.components_text.setPlainText(display_text)


class OptimizationThread(QThread):
    """Background thread for optimization"""
    
    progress_updated = pyqtSignal(float, int, int, float)
    optimization_complete = pyqtSignal(list)
    
    def __init__(self, optimizer, data):
        super().__init__()
        self.optimizer = optimizer
        self.data = data
        self.should_stop = False
    
    def run(self):
        """Run optimization"""
        def progress_callback(progress, generation, strategy_index, fitness):
            if not self.should_stop:
                self.progress_updated.emit(progress, generation, strategy_index, fitness)
        
        best_strategies = self.optimizer.evolve(self.data, progress_callback)
        
        if not self.should_stop:
            self.optimization_complete.emit(best_strategies)
    
    def stop(self):
        """Stop optimization"""
        self.should_stop = True 