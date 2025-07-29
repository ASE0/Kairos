#!/usr/bin/env python3
"""
VWAP Test Strategy for New Architecture
======================================
A simple VWAP-only strategy that will be detected by the new modular architecture.
"""

from strategies.strategy_builders import PatternStrategy, Action
from typing import Dict, Any

class VWAPTestStrategy(PatternStrategy):
    """A simple VWAP test strategy that uses the new architecture"""
    
    def __init__(self):
        # Create the VWAP filter action
        vwap_action = Action(
            name="vwap_filter",
            filters=[{
                "type": "vwap",
                "condition": "above"
            }]
        )
        
        super().__init__(
            name="VWAP Test Strategy (New Architecture)",
            actions=[vwap_action],
            combination_logic="AND",
            gates_and_logic={},  # No gates for pure filter strategy
            location_gate_params={}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary format for new architecture"""
        return {
            "name": self.name,
            "actions": [
                {
                    "name": "vwap_filter",
                    "filters": [
                        {
                            "type": "vwap",
                            "condition": "above"
                        }
                    ]
                }
            ],
            "combination_logic": "AND",
            "gates_and_logic": {},
            "location_gate_params": {}
        }

# Create an instance for the strategy registry
vwap_test_strategy = VWAPTestStrategy() 