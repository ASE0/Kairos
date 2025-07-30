"""
Building Block Strategy Builder
==============================
Creates specific building block strategies for testing.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action, StrategyFactory
from patterns.candlestick_patterns import PatternFactory

class BuildingBlockStrategyBuilder:
    """Builder for creating building block strategies"""
    
    def __init__(self):
        self.strategy_manager = None
        
    def create_vwap_strategy(self) -> PatternStrategy:
        """Create a VWAP mean reversion strategy"""
        print("Creating VWAP strategy...")
        
        # Create VWAP action with location strategy
        vwap_action = Action(
            name="VWAP Mean Reversion",
            pattern=None,  # No pattern, just location
            location_strategy="VWAP Mean-Reversion Band",
            location_params={
                'vwap_k': 1.0,
                'vwap_lookback': 20,
                'vwap_gamma': 0.95,
                'vwap_tau_bars': 15,
                'vwap_drop_threshold': 0.01
            },
            filters=[
                {
                    'type': 'vwap',
                    'condition': 'near',
                    'tolerance': 0.001
                }
            ]
        )
        
        # Create strategy
        strategy = PatternStrategy(
            name="VWAP_Mean_Reversion",
            actions=[vwap_action],
            combination_logic='AND',
            gates_and_logic={
                'location_gate': True,
                'momentum_gate': False,
                'volatility_gate': False
            },
            location_gate_params={
                'bar_interval_minutes': 1,
                'vwap_k': 1.0,
                'vwap_lookback': 20,
                'vwap_gamma': 0.95,
                'vwap_tau_bars': 15,
                'vwap_drop_threshold': 0.01
            }
        )
        
        return strategy
        
    def create_order_block_strategy(self) -> PatternStrategy:
        """Create an Order Block strategy"""
        print("Creating Order Block strategy...")
        
        # Create Order Block action
        ob_action = Action(
            name="Order Block Detection",
            pattern=None,  # No pattern, just location
            location_strategy="Order Block",
            location_params={
                'ob_impulse_threshold': 0.02,
                'ob_lookback_period': 10,
                'ob_gamma': 0.95,
                'ob_tau_bars': 80,
                'ob_drop_threshold': 0.01
            },
            filters=[
                {
                    'type': 'volume',
                    'volume_ratio': 1.5,
                    'min_volume': 100
                }
            ]
        )
        
        # Create strategy
        strategy = PatternStrategy(
            name="Order_Block_Strategy",
            actions=[ob_action],
            combination_logic='AND',
            gates_and_logic={
                'location_gate': True,
                'momentum_gate': False,
                'volatility_gate': False
            },
            location_gate_params={
                'bar_interval_minutes': 1,
                'ob_impulse_threshold': 0.02,
                'ob_lookback_period': 10,
                'ob_gamma': 0.95,
                'ob_tau_bars': 80,
                'ob_drop_threshold': 0.01
            }
        )
        
        return strategy
        
    def create_fvg_strategy(self) -> PatternStrategy:
        """Create a Fair Value Gap strategy"""
        print("Creating FVG strategy...")
        
        # Create FVG action
        fvg_action = Action(
            name="Fair Value Gap Detection",
            pattern=None,  # No pattern, just location
            location_strategy="FVG (Fair Value Gap)",
            location_params={
                'fvg_epsilon': 2,
                'fvg_N': 3,
                'fvg_sigma': 0.1,
                'fvg_beta1': 0.7,
                'fvg_beta2': 0.3,
                'fvg_phi': 0.2,
                'fvg_lambda': 0.0,
                'fvg_gamma': 0.95,
                'fvg_tau_bars': 50,
                'fvg_drop_threshold': 0.01
            },
            filters=[
                {
                    'type': 'momentum',
                    'lookback': 10,
                    'momentum_threshold': 0.01
                }
            ]
        )
        
        # Create strategy
        strategy = PatternStrategy(
            name="FVG_Strategy",
            actions=[fvg_action],
            combination_logic='AND',
            gates_and_logic={
                'location_gate': True,
                'momentum_gate': False,
                'volatility_gate': False
            },
            location_gate_params={
                'bar_interval_minutes': 1,
                'fvg_epsilon': 2,
                'fvg_N': 3,
                'fvg_sigma': 0.1,
                'fvg_beta1': 0.7,
                'fvg_beta2': 0.3,
                'fvg_phi': 0.2,
                'fvg_lambda': 0.0,
                'fvg_gamma': 0.95,
                'fvg_tau_bars': 50,
                'fvg_drop_threshold': 0.01
            }
        )
        
        return strategy
        
    def create_support_resistance_strategy(self) -> PatternStrategy:
        """Create a Support/Resistance strategy"""
        print("Creating Support/Resistance strategy...")
        
        # Create Support/Resistance action
        sr_action = Action(
            name="Support/Resistance Detection",
            pattern=None,  # No pattern, just location
            location_strategy="Support/Resistance Band",
            location_params={
                'sr_window': 20,
                'sr_buffer_pts': 2,
                'sr_sigma_r': 0.1,
                'sr_sigma_t': 0.1,
                'sr_gamma': 0.95,
                'sr_tau_bars': 60,
                'sr_drop_threshold': 0.01
            },
            filters=[
                {
                    'type': 'volatility',
                    'min_atr_ratio': 0.01,
                    'max_atr_ratio': 0.05
                }
            ]
        )
        
        # Create strategy
        strategy = PatternStrategy(
            name="Support_Resistance_Strategy",
            actions=[sr_action],
            combination_logic='AND',
            gates_and_logic={
                'location_gate': True,
                'momentum_gate': False,
                'volatility_gate': False
            },
            location_gate_params={
                'bar_interval_minutes': 1,
                'sr_window': 20,
                'sr_buffer_pts': 2,
                'sr_sigma_r': 0.1,
                'sr_sigma_t': 0.1,
                'sr_gamma': 0.95,
                'sr_tau_bars': 60,
                'sr_drop_threshold': 0.01
            }
        )
        
        return strategy
        
    def create_strategy(self, strategy_name: str) -> Optional[PatternStrategy]:
        """Create a strategy by name"""
        if strategy_name == "vwap":
            return self.create_vwap_strategy()
        elif strategy_name == "order_block":
            return self.create_order_block_strategy()
        elif strategy_name == "fvg":
            return self.create_fvg_strategy()
        elif strategy_name == "support_resistance":
            return self.create_support_resistance_strategy()
        else:
            print(f"Unknown strategy: {strategy_name}")
            return None
            
    def save_strategy(self, strategy: PatternStrategy, filename: str = None):
        """Save strategy to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy.name}_{timestamp}.dill"
            
        try:
            import dill
            with open(filename, 'wb') as f:
                dill.dump(strategy, f)
            print(f"Strategy saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving strategy: {e}")
            return None
            
    def load_strategy(self, filename: str) -> Optional[PatternStrategy]:
        """Load strategy from file"""
        try:
            import dill
            with open(filename, 'rb') as f:
                strategy = dill.load(f)
            print(f"Strategy loaded from: {filename}")
            return strategy
        except Exception as e:
            print(f"Error loading strategy: {e}")
            return None

def main():
    """Test the strategy builder"""
    builder = BuildingBlockStrategyBuilder()
    
    # Test creating all strategies
    strategies = ["vwap", "order_block", "fvg", "support_resistance"]
    
    for strategy_name in strategies:
        print(f"\n=== Testing {strategy_name} strategy creation ===")
        strategy = builder.create_strategy(strategy_name)
        if strategy:
            print(f"Successfully created {strategy_name} strategy")
            print(f"Strategy name: {strategy.name}")
            print(f"Actions: {len(strategy.actions)}")
            for action in strategy.actions:
                print(f"  - {action.name}: {action.location_strategy}")
        else:
            print(f"Failed to create {strategy_name} strategy")

if __name__ == "__main__":
    main() 