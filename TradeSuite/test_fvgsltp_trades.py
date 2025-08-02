#!/usr/bin/env python3
"""
Test script to analyze FVGSLTP strategy trades on one day of data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import dill

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.strategy_manager import StrategyManager
from strategies.strategy_builders import BacktestEngine
from core.data_structures import *

def load_fvgsltp_strategy():
    """Load the FVGSLTP strategy from workspace"""
    try:
        # Try to load from workspace
        workspace_dir = "workspaces/strategies"
        strategy_files = [f for f in os.listdir(workspace_dir) if f.endswith('.dill')]
        
        print(f"Found strategy files: {strategy_files}")
        
        # Look for FVGSLTP or related strategies
        for filename in strategy_files:
            filepath = os.path.join(workspace_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    strategy = dill.load(f)
                    print(f"Loaded strategy from {filename}: {type(strategy).__name__}")
                    if hasattr(strategy, 'name'):
                        print(f"Strategy name: {strategy.name}")
                    if hasattr(strategy, 'type'):
                        print(f"Strategy type: {strategy.type}")
                    return strategy
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        print("No FVGSLTP strategy found in workspace, creating a test strategy...")
        return create_test_fvgsltp_strategy()
        
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return create_test_fvgsltp_strategy()

def create_test_fvgsltp_strategy():
    """Create a test FVGSLTP strategy for analysis"""
    from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy
    
    # Create FVG pattern strategy
    fvg_action = Action(
        name="FVG",
        location_strategy="FVG",
        location_params={
            'fvg_epsilon': 0.001,
            'fvg_N': 3,
            'fvg_sigma': 0.1,
            'fvg_beta1': 0.4,
            'fvg_beta2': 0.6,
            'fvg_phi': 0.2,
            'fvg_lambda': 0.5,
            'fvg_gamma': 0.95,
            'fvg_tau_bars': 50,
            'fvg_drop_threshold': 0.01
        }
    )
    
    pattern_strategy = PatternStrategy(
        name="FVG_Pattern",
        actions=[fvg_action],
        combination_logic='AND',
        min_actions_required=1
    )
    
    # Create SLTP risk strategy
    risk_strategy = RiskStrategy(
        name="SLTP_Risk",
        entry_method='market',
        stop_method='fixed',
        exit_method='fixed_rr',
        stop_loss_pips=50,
        take_profit_pips=100,
        risk_per_trade=0.02,
        max_positions=1
    )
    
    # Create combined strategy
    combined_strategy = CombinedStrategy(
        name="FVGSLTP",
        pattern_strategy=pattern_strategy,
        risk_strategy=risk_strategy
    )
    
    return combined_strategy

def create_one_day_test_data():
    """Create one day of test data"""
    # Create one day of 1-minute data
    start_time = datetime(2024, 1, 15, 9, 30, 0)  # Market open
    end_time = datetime(2024, 1, 15, 16, 0, 0)    # Market close
    
    # Generate 390 minutes (6.5 hours of trading)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create realistic OHLCV data with some volatility
    np.random.seed(42)  # For reproducible results
    
    # Start price around 100
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, len(timestamps)):
        # Random walk with some trend
        change = np.random.normal(0, 0.1)  # Small random change
        if i % 60 == 0:  # Every hour, add some trend
            change += np.random.normal(0, 0.5)
        
        new_price = prices[-1] + change
        prices.append(max(0.01, new_price))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.05))
        high = close + volatility
        low = close - volatility
        open_price = close + np.random.normal(0, 0.02)
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df

def analyze_trades(strategy, data):
    """Run backtest and analyze trades"""
    print("\n" + "="*60)
    print("FVGSLTP STRATEGY TRADE ANALYSIS")
    print("="*60)
    
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
    
    # Analyze each trade
    trades = results.get('trades', [])
    print(f"\nDetailed Trade Analysis:")
    print(f"{'Trade':<6} {'Entry Time':<20} {'Exit Time':<20} {'Entry':<8} {'Exit':<8} {'Size':<8} {'PnL':<10} {'Exit Reason':<15}")
    print("-" * 100)
    
    for i, trade in enumerate(trades, 1):
        entry_time = trade.get('entry_time', 'N/A')
        exit_time = trade.get('exit_time', 'N/A')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        size = trade.get('size', 0)
        pnl = trade.get('pnl', 0)
        exit_reason = trade.get('exit_reason', 'N/A')
        
        print(f"{i:<6} {str(entry_time):<20} {str(exit_time):<20} {entry_price:<8.2f} {exit_price:<8.2f} {size:<8.2f} {pnl:<10.2f} {exit_reason:<15}")
    
    # Check for potential issues
    print(f"\nTrade Analysis Summary:")
    
    if not trades:
        print("❌ No trades generated - check strategy configuration")
        return
    
    # Check trade timing
    for i, trade in enumerate(trades, 1):
        entry_time = trade.get('entry_time')
        exit_time = trade.get('exit_time')
        
        if entry_time and exit_time:
            if entry_time >= exit_time:
                print(f"❌ Trade {i}: Entry time ({entry_time}) >= Exit time ({exit_time})")
            else:
                duration = exit_time - entry_time
                if duration < timedelta(minutes=1):
                    print(f"⚠️  Trade {i}: Very short duration ({duration})")
                elif duration > timedelta(hours=8):
                    print(f"⚠️  Trade {i}: Very long duration ({duration})")
    
    # Check price levels
    for i, trade in enumerate(trades, 1):
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        
        if entry_price <= 0 or exit_price <= 0:
            print(f"❌ Trade {i}: Invalid prices - Entry: {entry_price}, Exit: {exit_price}")
        
        if stop_loss > 0 and take_profit > 0:
            if entry_price <= stop_loss and entry_price >= take_profit:
                print(f"❌ Trade {i}: Entry price between stop loss and take profit")
    
    # Check position sizing
    for i, trade in enumerate(trades, 1):
        size = trade.get('size', 0)
        if size <= 0:
            print(f"❌ Trade {i}: Invalid position size: {size}")
    
    print(f"\n✅ Trade analysis complete!")

def main():
    """Main function"""
    print("Loading FVGSLTP strategy...")
    strategy = load_fvgsltp_strategy()
    
    print("Creating one day of test data...")
    data = create_one_day_test_data()
    print(f"Created {len(data)} bars of data from {data.index[0]} to {data.index[-1]}")
    
    # Analyze the strategy
    analyze_trades(strategy, data)
    
    # Also check the strategy components
    print(f"\nStrategy Component Analysis:")
    if hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy:
        print(f"✅ Pattern Strategy: {strategy.pattern_strategy.name}")
        print(f"   Actions: {len(strategy.pattern_strategy.actions)}")
        for action in strategy.pattern_strategy.actions:
            print(f"   - {action.name}: {action.location_strategy}")
    else:
        print("❌ No pattern strategy found")
    
    if hasattr(strategy, 'risk_strategy') and strategy.risk_strategy:
        print(f"✅ Risk Strategy: {strategy.risk_strategy.name}")
        print(f"   Entry Method: {strategy.risk_strategy.entry_method}")
        print(f"   Stop Method: {strategy.risk_strategy.stop_method}")
        print(f"   Exit Method: {strategy.risk_strategy.exit_method}")
    else:
        print("❌ No risk strategy found")

if __name__ == "__main__":
    main() 