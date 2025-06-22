#!/usr/bin/env python3
"""
Test script to verify AI optimizer changes:
1. Per-pattern timeframes
2. Correct pattern count (31 patterns)
3. Pattern builder consistency
"""

from gui.strategy_optimizer_window import StrategyOptimizer
from gui.strategy_optimizer_window import StrategyGenome

def test_optimizer():
    """Test AI optimizer functionality"""
    print("Testing AI Optimizer Changes...")
    
    # Create optimizer
    optimizer = StrategyOptimizer()
    optimizer.refresh_components()
    
    print(f"\n1. Pattern Count:")
    print(f"   Available patterns: {len(optimizer.available_patterns)}")
    print(f"   Expected: 31 patterns")
    print(f"   Actual patterns: {optimizer.available_patterns}")
    
    print(f"\n2. Per-Pattern Timeframes:")
    # Create a test genome
    genome = optimizer._create_random_genome()
    print(f"   Genome patterns: {genome.patterns}")
    print(f"   Pattern timeframes: {genome.pattern_timeframes}")
    
    # Check that each pattern has its own timeframes
    for pattern in genome.patterns:
        if pattern in genome.pattern_timeframes:
            print(f"   {pattern}: {genome.pattern_timeframes[pattern]}")
        else:
            print(f"   {pattern}: No timeframes specified")
    
    print(f"\n3. Strategy Creation:")
    try:
        strategy = optimizer._create_strategy_from_genome(genome)
        print(f"   Strategy created successfully: {strategy.name}")
        print(f"   Strategy actions: {len(strategy.actions)}")
    except Exception as e:
        print(f"   Error creating strategy: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_optimizer() 