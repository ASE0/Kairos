#!/usr/bin/env python3
"""
Comprehensive strategy validation test based on the provided documents
Tests Order Flow Momentum (OFM), Microstructure Mean Reversion (MMR), and Liquidity Vacuum Breakout (LVB)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.strategy_manager import StrategyManager
from strategies.strategy_builders import BacktestEngine
from core.data_structures import *

def create_test_data_with_microstructure():
    """Create realistic test data with microstructure features"""
    # Create 2 hours of 1-minute data (120 bars)
    start_time = datetime(2024, 1, 15, 9, 30, 0)
    end_time = datetime(2024, 1, 15, 11, 30, 0)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    np.random.seed(42)  # For reproducible results
    
    # Start price around 100
    base_price = 100.0
    prices = [base_price]
    
    # Create microstructure features
    microstructure_events = [
        (20, 25, 'sweep_up', 0.8),      # Bar 20-25: Upward sweep
        (45, 50, 'sweep_down', -0.6),    # Bar 45-50: Downward sweep  
        (70, 75, 'consolidation', 0.0),  # Bar 70-75: Consolidation
        (90, 95, 'breakout_up', 1.2),    # Bar 90-95: Upward breakout
        (110, 115, 'institutional_flow', 0.4)  # Bar 110-115: Institutional flow
    ]
    
    for i in range(1, len(timestamps)):
        # Check for microstructure events
        event_impact = 0
        for start_bar, end_bar, event_type, impact in microstructure_events:
            if start_bar <= i <= end_bar:
                event_impact = impact
                break
        
        # Random walk with microstructure impact
        change = np.random.normal(0, 0.03) + event_impact * 0.1
        if i % 30 == 0:  # Every 30 minutes, add some trend
            change += np.random.normal(0, 0.1)
        
        new_price = prices[-1] + change
        prices.append(max(0.01, new_price))
    
    # Create OHLCV data with realistic microstructure
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.02))
        high = close + volatility
        low = close - volatility
        open_price = close + np.random.normal(0, 0.01)
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume with microstructure patterns
        base_volume = np.random.randint(1000, 5000)
        
        # Add volume spikes for microstructure events
        for start_bar, end_bar, event_type, impact in microstructure_events:
            if start_bar <= i <= end_bar:
                if event_type in ['sweep_up', 'sweep_down']:
                    base_volume *= 3  # High volume for sweeps
                elif event_type == 'institutional_flow':
                    base_volume *= 2.5  # High volume for institutional flow
                elif event_type == 'consolidation':
                    base_volume *= 0.5  # Low volume for consolidation
                elif event_type == 'breakout_up':
                    base_volume *= 2  # Medium-high volume for breakouts
                break
        
        volume = max(100, int(base_volume))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df

def create_ofm_strategy():
    """Create Order Flow Momentum strategy based on document specifications"""
    from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action
    
    # Create OFM pattern strategy
    ofm_action = Action(
        name="OFM",
        location_strategy="VWAP",  # Use VWAP as proxy for order flow
        location_params={
            'vwap_period': 20,
            'volume_threshold': 2000,  # Large trade threshold
            'imbalance_threshold': 0.6,  # Bid/ask imbalance
            'absorption_ratio': 400,  # Volume per tick movement
            'trail_ticks': 3
        }
    )
    
    pattern_strategy = PatternStrategy(
        name="OFM_Pattern",
        actions=[ofm_action],
        combination_logic='AND',
        min_actions_required=1,
        location_gate_params={
            'gate_threshold': 0.3,  # Order flow efficiency threshold
            'lookback': 50
        }
    )
    
    # Create risk strategy for OFM
    risk_strategy = RiskStrategy(
        name="OFM_Risk",
        entry_method='market',
        stop_method='fixed',
        exit_method='trailing',
        stop_loss_pct=0.015,  # 1.5% stop loss
        trailing_stop_pct=0.005,  # 0.5% trailing stop
        risk_reward_ratio=2.0
    )
    
    # Create combined strategy
    combined_strategy = CombinedStrategy(
        name="OFM_Strategy",
        pattern_strategy=pattern_strategy,
        risk_strategy=risk_strategy
    )
    
    return combined_strategy

def create_mmr_strategy():
    """Create Microstructure Mean Reversion strategy based on document specifications"""
    from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action
    
    # Create MMR pattern strategy
    mmr_action = Action(
        name="MMR",
        location_strategy="OrderBlock",  # Use order blocks for sweep detection
        location_params={
            'sweep_threshold': 75,  # Contracts for sweep detection
            'book_imbalance': 3.0,  # Bid/ask ratio
            'quiet_period': 200,  # Ticks for quiet period
            'reversion_percent': 0.6,  # Reversion percentage
            'max_heat': 4  # Max adverse excursion
        }
    )
    
    pattern_strategy = PatternStrategy(
        name="MMR_Pattern",
        actions=[mmr_action],
        combination_logic='AND',
        min_actions_required=1,
        location_gate_params={
            'gate_threshold': 0.2,  # Lower threshold for mean reversion
            'lookback': 100
        }
    )
    
    # Create risk strategy for MMR
    risk_strategy = RiskStrategy(
        name="MMR_Risk",
        entry_method='limit',
        stop_method='fixed',
        exit_method='fixed_rr',
        stop_loss_pct=0.02,  # 2% stop loss
        risk_reward_ratio=1.5  # Lower RR for mean reversion
    )
    
    # Create combined strategy
    combined_strategy = CombinedStrategy(
        name="MMR_Strategy",
        pattern_strategy=pattern_strategy,
        risk_strategy=risk_strategy
    )
    
    return combined_strategy

def create_lvb_strategy():
    """Create Liquidity Vacuum Breakout strategy based on document specifications"""
    from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action
    
    # Create LVB pattern strategy
    lvb_action = Action(
        name="LVB",
        location_strategy="SupportResistance",  # Use S/R for consolidation detection
        location_params={
            'consolidation_ticks': 500,  # Ticks for consolidation
            'volume_reduction': 0.3,  # Volume reduction threshold
            'range_ticks': 5,  # Max range during consolidation
            'breakout_volume': 100,  # Contracts for breakout
            'target_multiple': 2.5  # Risk:reward multiple
        }
    )
    
    pattern_strategy = PatternStrategy(
        name="LVB_Pattern",
        actions=[lvb_action],
        combination_logic='AND',
        min_actions_required=1,
        location_gate_params={
            'gate_threshold': 0.4,  # Higher threshold for breakouts
            'lookback': 200
        }
    )
    
    # Create risk strategy for LVB
    risk_strategy = RiskStrategy(
        name="LVB_Risk",
        entry_method='stop',
        stop_method='fixed',
        exit_method='fixed_rr',
        stop_loss_pct=0.025,  # 2.5% stop loss
        risk_reward_ratio=2.5  # Higher RR for breakouts
    )
    
    # Create combined strategy
    combined_strategy = CombinedStrategy(
        name="LVB_Strategy",
        pattern_strategy=pattern_strategy,
        risk_strategy=risk_strategy
    )
    
    return combined_strategy

def validate_strategy_logic(strategy, data, strategy_name):
    """Validate that the strategy logic matches the document specifications"""
    print(f"\n{'='*80}")
    print(f"VALIDATING {strategy_name} STRATEGY LOGIC")
    print(f"{'='*80}")
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run_backtest(strategy, data, initial_capital=100000, risk_per_trade=0.02)
    
    print(f"\nBacktest Results:")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    
    # Analyze trades for logic validation
    trades = results.get('trades', [])
    print(f"\nTrade Logic Analysis:")
    print(f"{'Trade':<4} {'Entry Time':<20} {'Exit Time':<20} {'Entry':<8} {'Exit':<8} {'Stop':<8} {'Target':<8} {'Size':<8} {'PnL':<10} {'Exit Reason':<15}")
    print("-" * 120)
    
    logic_issues = []
    
    for i, trade in enumerate(trades, 1):
        entry_time = trade.get('entry_time', 'N/A')
        exit_time = trade.get('exit_time', 'N/A')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        size = trade.get('size', 0)
        pnl = trade.get('pnl', 0)
        exit_reason = trade.get('exit_reason', 'N/A')
        
        print(f"{i:<4} {str(entry_time):<20} {str(exit_time):<20} {entry_price:<8.2f} {exit_price:<8.2f} {stop_loss:<8.2f} {take_profit:<8.2f} {size:<8.2f} {pnl:<10.2f} {exit_reason:<15}")
        
        # Validate strategy-specific logic
        if strategy_name == "OFM":
            # OFM should have trailing stops and institutional flow characteristics
            if exit_reason == 'Time Exit':
                logic_issues.append(f"⚠️  Trade {i}: OFM should use trailing stops, not time exits")
            if size < 500:  # Assuming large position sizes for institutional flow
                logic_issues.append(f"⚠️  Trade {i}: OFM should have larger position sizes for institutional flow")
                
        elif strategy_name == "MMR":
            # MMR should have fixed stops and mean reversion characteristics
            if stop_loss == 0:
                logic_issues.append(f"❌ Trade {i}: MMR must have fixed stop losses")
            if abs(entry_price - exit_price) > 0.05:  # Large moves not typical for mean reversion
                logic_issues.append(f"⚠️  Trade {i}: MMR should have smaller price movements")
                
        elif strategy_name == "LVB":
            # LVB should have breakout characteristics
            if exit_reason == 'Time Exit':
                logic_issues.append(f"⚠️  Trade {i}: LVB should exit on breakout failure or targets")
            if take_profit == 0:
                logic_issues.append(f"❌ Trade {i}: LVB must have take profit targets")
    
    # Check strategy components
    print(f"\nStrategy Component Validation:")
    if hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy:
        print(f"✅ Pattern Strategy: {strategy.pattern_strategy.name}")
        print(f"   Actions: {len(strategy.pattern_strategy.actions)}")
        for action in strategy.pattern_strategy.actions:
            print(f"   - {action.name}: {action.location_strategy}")
            if hasattr(action, 'location_params'):
                print(f"     Params: {action.location_params}")
    else:
        print("❌ No pattern strategy found")
        logic_issues.append("❌ Missing pattern strategy component")
    
    if hasattr(strategy, 'risk_strategy') and strategy.risk_strategy:
        print(f"✅ Risk Strategy: {strategy.risk_strategy.name}")
        print(f"   Entry Method: {strategy.risk_strategy.entry_method}")
        print(f"   Stop Method: {strategy.risk_strategy.stop_method}")
        print(f"   Exit Method: {strategy.risk_strategy.exit_method}")
        print(f"   Stop Loss %: {getattr(strategy.risk_strategy, 'stop_loss_pct', 'N/A')}")
        print(f"   Risk/Reward: {getattr(strategy.risk_strategy, 'risk_reward_ratio', 'N/A')}")
    else:
        print("❌ No risk strategy found")
        logic_issues.append("❌ Missing risk strategy component")
    
    # Print logic issues
    if logic_issues:
        print(f"\nLogic Issues Found:")
        for issue in logic_issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ {strategy_name} strategy logic appears correct!")
    
    return len(logic_issues) == 0

def main():
    """Main validation function"""
    print("Starting Strategy Validation Test")
    print("="*80)
    
    # Create test data
    print("Creating test data with microstructure features...")
    data = create_test_data_with_microstructure()
    print(f"Created {len(data)} bars of data from {data.index[0]} to {data.index[-1]}")
    
    # Test each strategy
    strategies = [
        ("OFM", create_ofm_strategy()),
        ("MMR", create_mmr_strategy()),
        ("LVB", create_lvb_strategy())
    ]
    
    results = {}
    
    for strategy_name, strategy in strategies:
        print(f"\nTesting {strategy_name} strategy...")
        success = validate_strategy_logic(strategy, data, strategy_name)
        results[strategy_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    for strategy_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{strategy_name}: {status}")
    
    total_passed = sum(results.values())
    total_strategies = len(results)
    
    print(f"\nOverall Result: {total_passed}/{total_strategies} strategies passed validation")
    
    if total_passed == total_strategies:
        print("🎉 All strategies passed validation!")
    else:
        print("⚠️  Some strategies need attention")

if __name__ == "__main__":
    main() 