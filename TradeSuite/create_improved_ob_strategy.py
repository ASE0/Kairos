#!/usr/bin/env python3
"""
Create Improved Order Block Strategy
===================================
Creates an OB strategy with better parameters for detecting more significant zones
"""

import dill
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import Action, PatternStrategy
from core.data_structures import TimeRange

def create_improved_ob_strategy():
    """Create an improved OB strategy with better parameters"""
    
    # Create improved OB strategy with better parameters
    actions = [
        Action(
            name="OrderBlock",
            pattern=None,
            time_range=TimeRange(1, 'min'),
            location_strategy="OrderBlock",
            location_params={
                # Improved Order Block parameters for better detection
                'epsilon_pts': 2.0,              # Increased buffer for wider zones
                'max_impulse_bars': 5,           # Look further for impulse moves
                'min_impulse_score': 0.5,        # Lower threshold for more detection
                'min_impulse_body_mult': 1.2,    # Lower body multiple requirement
                'max_block_lookback': 5,         # Look further back for block candles
                'min_block_body_frac': 0.15,     # Lower body fraction requirement
                'gamma_imp': 1.5,                # Lower gamma for less aggressive scoring
                'delta_imp': 1.0,                # Lower delta for less wick sensitivity
                'gamma_decay': 0.98,             # Slower decay for longer-lasting zones
                'tau_bars': 100                  # Longer zone lifetime
            },
            filters=[]
        )
    ]

    # Create the strategy
    strategy = PatternStrategy(
        name="Improved_OrderBlock",
        actions=actions,
        combination_logic="AND",
        weights=None,
        min_actions_required=1,
        gates_and_logic={'location_gate': True},
        location_gate_params={
            # Additional location gate parameters
            'gate_threshold': 0.1,              # Lower threshold for more signals
            'lookback': 100,                    # Longer lookback
            'zone_gamma': 0.98,                 # Slower zone decay
            'zone_tau_bars': 100,               # Longer zone lifetime
            'zone_drop_threshold': 0.005        # Lower drop threshold
        }
    )

    # Save the improved strategy
    with open('workspaces/strategies/ob_improved.dill', 'wb') as f:
        dill.dump(strategy, f)

    print("✅ Created improved OB strategy with better parameters")
    print("📁 Saved as: workspaces/strategies/ob_improved.dill")
    print("\n🔧 Key Improvements:")
    print("   • epsilon_pts: 2.0 (was 0.1) - Wider zones")
    print("   • min_impulse_score: 0.5 (was 1.0) - More detection")
    print("   • min_impulse_body_mult: 1.2 (was 1.5) - Less restrictive")
    print("   • max_block_lookback: 5 (was 3) - Look further back")
    print("   • gamma_decay: 0.98 (was 0.95) - Slower decay")
    print("   • tau_bars: 100 (was 50) - Longer lifetime")
    
    return strategy

if __name__ == "__main__":
    create_improved_ob_strategy() 