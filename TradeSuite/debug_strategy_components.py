#!/usr/bin/env python3
"""
Debug script to understand what building blocks your strategy actually has
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_strategy_from_main():
    """Analyze strategy components from the main trading system"""
    
    print("=== ANALYZING YOUR STRATEGY COMPONENTS ===")
    
    # Look at strategy builder files
    try:
        from strategies.strategy_builders import StrategyBuilder
        print("✅ Found StrategyBuilder")
        
        # Create a sample strategy to see what components it has
        builder = StrategyBuilder()
        print(f"StrategyBuilder methods: {[method for method in dir(builder) if not method.startswith('_')]}")
        
    except Exception as e:
        print(f"❌ Error importing StrategyBuilder: {e}")
    
    # Look at core components
    try:
        from core.components import filters
        print("✅ Found filters module")
        print(f"Filter classes: {[name for name in dir(filters) if not name.startswith('_')]}")
        
    except Exception as e:
        print(f"❌ Error importing filters: {e}")
    
    # Look at patterns
    try:
        from patterns import candlestick_patterns
        print("✅ Found candlestick_patterns")
        
    except Exception as e:
        print(f"❌ Error importing candlestick_patterns: {e}")
    
    # Common building blocks in trading strategies
    common_building_blocks = [
        "fvg",           # Fair Value Gap
        "vwap",          # Volume Weighted Average Price
        "order_block",   # Order Block
        "support_resistance", # Support/Resistance
        "momentum",      # Momentum indicators
        "volatility",    # Volatility measures
        "imbalance",     # Order flow imbalance
        "tick_frequency", # Tick frequency analysis
        "zones",         # Supply/Demand zones
        "gates",         # Entry/exit gates
        "filters",       # Signal filters
        "mmr",           # Market Microstructure Regime
    ]
    
    print(f"\n📊 COMMON BUILDING BLOCKS IN TRADING STRATEGIES:")
    for block in common_building_blocks:
        print(f"  - {block}")
    
    print(f"\n💡 YOUR STRATEGY SHOULD HAVE MULTIPLE BUILDING BLOCKS")
    print(f"   Each building block should show when it triggers signals")
    print(f"   The heatmap should show ALL of them, not just 'mmr'")

if __name__ == "__main__":
    analyze_strategy_from_main()