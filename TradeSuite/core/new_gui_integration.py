#!/usr/bin/env python3
"""
New GUI Integration for Modular Strategy Architecture
====================================================
Seamless integration between the new modular architecture and the GUI components.
Ensures that any new component added works immediately in all parts of the system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from dataclasses import asdict

from .strategy_architecture import (
    StrategyConfig, StrategyEngine, ChartRenderer, component_registry,
    initialize_architecture
)

# Import all component types to auto-register
from .components import filters  # This will auto-register all filters

class NewBacktestEngine:
    """
    New backtest engine that works consistently with the modular architecture
    """
    
    def __init__(self):
        self.strategy_engine = StrategyEngine(component_registry)
        self.chart_renderer = ChartRenderer()
        self.logger = logging.getLogger("NewBacktestEngine")
        
        # Initialize architecture
        initialize_architecture()
    
    def run_backtest(self, strategy_config: Dict[str, Any], data: pd.DataFrame,
                    initial_capital: float = 100000,
                    risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """
        Run a backtest using the new modular architecture
        
        Args:
            strategy_config: Strategy configuration dictionary
            data: OHLCV DataFrame with DatetimeIndex
            initial_capital: Starting capital
            risk_per_trade: Risk per trade (0.02 = 2%)
            
        Returns:
            Backtest results with trades, signals, and visualization data
        """
        self.logger.info(f"Starting new backtest: {strategy_config.get('name', 'Unnamed')}")
        
        # Convert strategy config to new format
        config = self._convert_strategy_config(strategy_config)
        
        # Execute strategy to get signals and visualization data
        signals, visualization_data = self.strategy_engine.execute_strategy(config, data)
        
        # Generate trades from signals
        trades = self._generate_trades(signals, data, initial_capital, risk_per_trade)
        
        # Calculate performance metrics and equity curve
        performance, equity_curve = self._calculate_performance(trades, initial_capital, data)
        
        # Prepare results in old format for GUI compatibility
        results = {
            'strategy_name': config.name,
            'signals': signals,
            'trades': trades,
            'performance': performance,
            'visualization_data': visualization_data,
            'total_signals': signals.sum(),
            'data': data,  # Include data for chart rendering
            'zones': [],   # No zones in new architecture unless explicitly created
            'component_results': self._get_component_summary(visualization_data),
            
            # Add old format compatibility fields
            'equity_curve': equity_curve,
            'total_trades': len(trades),
            'initial_capital': initial_capital,
            'final_capital': initial_capital + sum(trade['pnl'] for trade in trades),
            'cumulative_pnl': sum(trade['pnl'] for trade in trades),
            'total_return': performance['total_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown': performance['max_drawdown'],
            'win_rate': performance['win_rate'],
            'profit_factor': self._calculate_profit_factor(trades),
            'dataset_data': data,  # For chart compatibility
            'multi_tf_data': data,  # For compatibility
            'action_details': [],
            'S_adj_scores': [],
            'S_net_scores': [],
            'per_zone_strengths': [],
            'momentum_scores': [],
            'volatility_scores': [],
            'imbalance_scores': [],
            'enhanced_momentum': [],
            'strategy_params': config.to_dict() if hasattr(config, 'to_dict') else {},
            'gates_enabled': config.gates,
            'timeframe': '1min',  # Default
            'interval': '1min',   # Default
            'result_display_name': config.name,
            'patterns': []  # No patterns in new architecture
        }
        
        self.logger.info(f"Backtest complete: {len(trades)} trades, {signals.sum()} signals")
        
        return results
    
    def _convert_strategy_config(self, old_config: Dict[str, Any]) -> StrategyConfig:
        """Convert old strategy config format to new modular format"""
        
        # Extract filters from actions
        filters = []
        patterns = []
        locations = []
        
        actions = old_config.get('actions', [])
        for action in actions:
            # Extract filters
            action_filters = action.get('filters', [])
            filters.extend(action_filters)
            
            # Extract patterns
            if action.get('pattern'):
                patterns.append({
                    'type': action['pattern'],
                    'config': action.get('pattern_config', {})
                })
            
            # Extract location strategies
            if action.get('location_strategy'):
                locations.append({
                    'type': action['location_strategy'],
                    'config': action.get('location_config', {})
                })
        
        # Extract gates
        gates = []
        gates_and_logic = old_config.get('gates_and_logic', {})
        for gate_name, enabled in gates_and_logic.items():
            if enabled:
                gates.append({
                    'type': gate_name.replace('_gate', ''),  # Remove '_gate' suffix
                    'config': old_config.get('location_gate_params', {})
                })
        
        return StrategyConfig(
            name=old_config.get('name', 'Converted Strategy'),
            filters=filters,
            gates=gates,
            patterns=patterns,
            locations=locations,
            combination_logic=old_config.get('combination_logic', 'AND')
        )
    
    def _generate_trades(self, signals: pd.Series, data: pd.DataFrame,
                        initial_capital: float, risk_per_trade: float) -> List[Dict[str, Any]]:
        """Generate trades from signals using simple logic"""
        trades = []
        in_trade = False
        entry_price = 0
        entry_time = None
        entry_idx = None
        
        for i, (timestamp, signal) in enumerate(signals.items()):
            if signal and not in_trade:
                # Enter trade
                entry_price = data.loc[timestamp, 'close']
                entry_time = timestamp
                entry_idx = i
                in_trade = True
                
            elif not signal and in_trade:
                # Exit trade
                exit_price = data.loc[timestamp, 'close']
                exit_time = timestamp
                
                # Calculate trade metrics
                size = (initial_capital * risk_per_trade) / entry_price
                pnl = (exit_price - entry_price) * size
                
                # Calculate MAE and MFE (simplified)
                trade_data = data.loc[entry_time:exit_time]
                mae = (entry_price - trade_data['low'].min()) * size
                mfe = (trade_data['high'].max() - entry_price) * size
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': size,
                    'pnl': pnl,
                    'mae': mae,
                    'mfe': mfe,
                    'exit_reason': 'Signal Exit',
                    'entry_idx': entry_idx,
                    'exit_idx': i
                })
                
                in_trade = False
        
        # Close any open trade at the end
        if in_trade:
            exit_price = data.iloc[-1]['close']
            exit_time = data.index[-1]
            size = (initial_capital * risk_per_trade) / entry_price
            pnl = (exit_price - entry_price) * size
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'mae': 0,
                'mfe': 0,
                'exit_reason': 'End of Data',
                'entry_idx': entry_idx,
                'exit_idx': len(data) - 1
            })
        
        return trades
    
    def _calculate_performance(self, trades: List[Dict[str, Any]], 
                             initial_capital: float, data: pd.DataFrame) -> Tuple[Dict[str, Any], List[float]]:
        """Calculate performance metrics"""
        if not trades:
            # Generate empty equity curve with proper datetime index
            empty_equity_curve = self._generate_equity_curve([], initial_capital, data)
            
            return {
                'total_return': 0.0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_trade': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'num_trades': 0
            }, empty_equity_curve
        
        pnls = [trade['pnl'] for trade in trades]
        total_pnl = sum(pnls)
        total_return = total_pnl / initial_capital
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        
        avg_trade = total_pnl / len(trades) if trades else 0
        
        # Simplified drawdown calculation
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) / initial_capital if len(drawdowns) > 0 else 0
        
        # Simplified Sharpe ratio
        if len(pnls) > 1:
            avg_return = np.mean(pnls) / initial_capital
            std_return = np.std(pnls) / initial_capital
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Generate equity curve with proper datetime index
        equity_curve = self._generate_equity_curve(trades, initial_capital, data)
        
        performance = {
            'total_return': total_return,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades)
        }
        
        return performance, equity_curve
    
    def _generate_equity_curve(self, trades: List[Dict[str, Any]], 
                              initial_capital: float, data: pd.DataFrame) -> pd.Series:
        """Generate equity curve aligned with data timeline with proper datetime index"""
        equity_values = [initial_capital] * len(data)
        
        # Apply trade PnL to equity curve
        for trade in trades:
            exit_idx = trade.get('exit_idx')
            pnl = trade.get('pnl', 0)
            
            if exit_idx is not None and exit_idx < len(equity_values):
                # Add PnL to all points after trade exit
                for i in range(exit_idx, len(equity_values)):
                    equity_values[i] += pnl
        
        # Create pandas Series with proper datetime index
        equity_curve = pd.Series(equity_values, index=data.index)
        return equity_curve
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _get_component_summary(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of components used"""
        summary = {
            'filters': [],
            'patterns': [],
            'locations': [],
            'gates': [],
            'visualization_types': {
                'lines': len(visualization_data.get('lines', [])),
                'zones': len(visualization_data.get('zones', [])),
                'markers': len(visualization_data.get('markers', [])),
                'bands': len(visualization_data.get('bands', [])),
                'histograms': len(visualization_data.get('histograms', []))
            }
        }
        
        # Extract component names from visualization data
        for line_data in visualization_data.get('lines', []):
            component_type = line_data.get('component_type', 'unknown')
            summary.setdefault(f"{component_type}s", []).append(line_data['name'])
        
        return summary

class NewGUIIntegration:
    """
    GUI integration layer that provides seamless connection to the new architecture
    """
    
    def __init__(self):
        self.backtest_engine = NewBacktestEngine()
        self.chart_renderer = ChartRenderer()
        self.logger = logging.getLogger("NewGUIIntegration")
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get all available components for the strategy builder"""
        return component_registry.get_available_components()
    
    def validate_strategy_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a strategy configuration"""
        errors = []
        
        if not config.get('name'):
            errors.append("Strategy name is required")
        
        # Check if at least one component is specified
        has_components = (
            config.get('actions', []) or 
            config.get('filters', []) or 
            config.get('patterns', []) or 
            config.get('locations', [])
        )
        
        if not has_components:
            errors.append("Strategy must have at least one component (filter, pattern, or location)")
        
        # Validate individual components
        try:
            strategy_config = self.backtest_engine._convert_strategy_config(config)
            
            # Try to create each component to validate
            for filter_config in strategy_config.filters:
                try:
                    component_registry.create_component('filter', filter_config['type'], filter_config)
                except Exception as e:
                    errors.append(f"Invalid filter '{filter_config['type']}': {e}")
            
            for pattern_config in strategy_config.patterns:
                try:
                    component_registry.create_component('pattern', pattern_config['type'], pattern_config)
                except Exception as e:
                    errors.append(f"Invalid pattern '{pattern_config['type']}': {e}")
            
            for location_config in strategy_config.locations:
                try:
                    component_registry.create_component('location', location_config['type'], location_config)
                except Exception as e:
                    errors.append(f"Invalid location '{location_config['type']}': {e}")
                    
        except Exception as e:
            errors.append(f"Strategy configuration error: {e}")
        
        return len(errors) == 0, errors
    
    def run_strategy_backtest(self, strategy_config: Dict[str, Any], 
                            data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run a backtest with the new architecture"""
        return self.backtest_engine.run_backtest(strategy_config, data, **kwargs)
    
    def render_strategy_chart(self, backtest_results: Dict[str, Any], ax=None):
        """Render strategy chart using the new unified renderer"""
        data = backtest_results.get('data')
        signals = backtest_results.get('signals')
        visualization_data = backtest_results.get('visualization_data', {})
        
        if data is None or signals is None:
            self.logger.error("Missing data or signals for chart rendering")
            return
        
        self.chart_renderer.render_strategy_chart(data, signals, visualization_data, ax)
    
    def create_strategy_config(self, name: str, components: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create a strategy config from component specifications"""
        config = StrategyConfig(
            name=name,
            filters=components.get('filters', []),
            gates=components.get('gates', []),
            patterns=components.get('patterns', []),
            locations=components.get('locations', []),
            combination_logic=components.get('combination_logic', 'AND')
        )
        
        return config.to_dict()
    
    def get_component_info(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific component"""
        try:
            # Create a dummy component to get its info
            dummy_config = {'type': component_name}
            component = component_registry.create_component(component_type, component_name, dummy_config)
            
            return {
                'name': component_name,
                'type': component_type,
                'required_columns': component.get_required_columns(),
                'visualization_config': component.get_visualization_config(),
                'description': component.__class__.__doc__ or "No description available"
            }
        except Exception as e:
            return {
                'name': component_name,
                'type': component_type,
                'error': str(e)
            }

# =============================================================================
# COMPATIBILITY LAYER
# =============================================================================

class CompatibilityWrapper:
    """
    Wrapper to maintain compatibility with existing GUI code while 
    transitioning to the new architecture
    """
    
    def __init__(self):
        self.new_integration = NewGUIIntegration()
        self.logger = logging.getLogger("CompatibilityWrapper")
    
    def run_gui_compatible_test(self, strategy_config: Dict[str, Any], 
                              data_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Compatibility method for existing test scripts
        """
        # Load data
        if isinstance(data_path, str):
            data = pd.read_csv(data_path)
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
        else:
            data = data_path  # Assume it's already a DataFrame
        
        # Run backtest
        results = self.new_integration.run_strategy_backtest(strategy_config, data)
        
        # Convert results to old format for compatibility
        compatible_results = self._convert_results_to_old_format(results)
        
        # Save results if output path provided
        if output_path:
            self._save_results(compatible_results, output_path)
        
        return compatible_results
    
    def _convert_results_to_old_format(self, new_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new results format to old format for compatibility"""
        
        # Convert trades to old format
        trades = []
        for trade in new_results.get('trades', []):
            trades.append({
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'entry_price': float(trade['entry_price']),
                'exit_price': float(trade['exit_price']),
                'size': float(trade['size']),
                'pnl': float(trade['pnl']),
                'mae': float(trade.get('mae', 0)),
                'mfe': float(trade.get('mfe', 0)),
                'exit_reason': trade.get('exit_reason', 'Unknown')
            })
        
        return {
            'trades': trades,
            'total_signals': int(new_results.get('total_signals', 0)),
            'zones': new_results.get('zones', []),  # Should be empty in new architecture
            'performance': new_results.get('performance', {}),
            'strategy_name': new_results.get('strategy_name', 'Unknown'),
            'component_summary': new_results.get('component_results', {}),
            'data': new_results.get('data'),
            'visualization_data': new_results.get('visualization_data', {})
        }
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        # Convert non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key == 'data':
                continue  # Skip large DataFrame
            elif isinstance(value, pd.Series):
                serializable_results[key] = value.tolist()
            elif hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Create global instances for easy access
new_gui_integration = NewGUIIntegration()
compatibility_wrapper = CompatibilityWrapper()

# Convenience function for existing code
def run_gui_compatible_test(strategy_config: Dict[str, Any], 
                          data_path: str, output_path: str = None) -> Dict[str, Any]:
    """Global convenience function for compatibility"""
    return compatibility_wrapper.run_gui_compatible_test(strategy_config, data_path, output_path)

if __name__ == "__main__":
    # Test the new integration
    print("New GUI Integration loaded successfully!")
    print("\nAvailable components:")
    components = new_gui_integration.get_available_components()
    for comp_type, comp_list in components.items():
        print(f"  {comp_type}: {comp_list}") 