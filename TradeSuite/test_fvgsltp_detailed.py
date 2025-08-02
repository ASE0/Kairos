#!/usr/bin/env python3
"""
Detailed test script to analyze FVGSLTP strategy trades on one day of data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dill

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.strategy_manager import StrategyManager
from strategies.strategy_builders import BacktestEngine
from core.data_structures import *

def create_fvgsltp_strategy():
    """Create a proper FVGSLTP strategy for analysis"""
    from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action
    
    # Create FVG pattern strategy with proper parameters
    fvg_action = Action(
        name="FVG",
        location_strategy="FVG",
        location_params={
            'fvg_epsilon': 0.1,        # Gap threshold (10 ticks)
            'fvg_N': 3,                # Number of peaks for validation
            'fvg_sigma': 0.2,          # Peak width for noise filtering
            'fvg_beta1': 0.7,          # Base weight for volume confirmation
            'fvg_beta2': 0.3,          # Comb weight for momentum
            'fvg_phi': 0.3,            # Expansion factor for zone growth
            'fvg_lambda': 0.0,         # No skew
            'fvg_gamma': 0.98,         # Slow decay rate
            'fvg_tau_bars': 20,        # Zone lifetime (shorter for tick data)
            'fvg_drop_threshold': 0.1,  # Minimum strength (10 ticks)
            'min_gap_size': 0.1,       # Minimum gap size (10 ticks)
            'max_gap_size': 0.5,       # Maximum gap size (50 ticks)
            'volume_filter': True,      # Enable volume confirmation
            'min_volume': 100,         # Minimum volume for validation
            'signal_generation': {
                'mode': 'zone_only',    # Only generate signals at zone boundaries
                'entry_side': 'fill',   # Enter when price fills the zone
                'validation': {
                    'volume': True,     # Require volume confirmation
                    'momentum': True,   # Require momentum confirmation
                    'spread': True      # Require normal spread
                }
            }
        },
        filters=[
            {
                'type': 'volume',
                'lookback': 20,
                'threshold': 100,
                'condition': 'above_average'
            },
            {
                'type': 'volatility',
                'lookback': 50,
                'threshold': 0.02,
                'condition': 'below_max'
            },
            {
                'type': 'spread',
                'max_ticks': 2,
                'condition': 'normal'
            }
        ]
    )
    
    pattern_strategy = PatternStrategy(
        name="FVG_Pattern",
        actions=[fvg_action],
        combination_logic='AND',
        min_actions_required=1
    )
    
    # Set entry and exit conditions
    pattern_strategy.entry_conditions = {
        'long': [
            'price_crosses_zone_low',  # Enter long when price crosses lower FVG boundary
            'volume_above_average',    # Require above average volume
            'zone_strength > 0.5'      # Strong zone rating
        ],
        'short': [
            'price_crosses_zone_high', # Enter short when price crosses upper FVG boundary
            'volume_above_average',    # Require above average volume
            'zone_strength > 0.5'      # Strong zone rating
        ]
    }
    
    pattern_strategy.exit_conditions = [
        'zone_filled',              # Exit when FVG is filled
        'zone_invalidated',         # Exit if zone becomes invalid
        'adverse_excursion > 5',    # Exit on excessive adverse movement
        'time_in_trade > 50'        # Time-based exit (50 bars)
    ]
    
    pattern_strategy.location_gate_params = {
        'gate_threshold': 0.3,      # Higher threshold for better quality
        'lookback': 50,             # Shorter lookback for faster response
        'min_zone_size': 0.05,      # Minimum 5 tick zones
        'max_zone_size': 0.5,       # Maximum 50 tick zones
        'volume_threshold': 100,    # Minimum volume requirement
        'strength_threshold': 0.5   # Minimum zone strength
    }
    
    # Create SLTP risk strategy with proper parameters
    risk_strategy = RiskStrategy(
        name="SLTP_Risk"
    )
    
    # Set core parameters
    risk_strategy.entry_method = 'market'
    risk_strategy.stop_method = 'dynamic'
    risk_strategy.exit_method = 'multi_target'
    risk_strategy.stop_loss_pct = 0.005
    risk_strategy.risk_reward_ratio = 2.0
    
    # Set risk parameters
    risk_strategy.parameters = {
        'position_sizing': {
            'method': 'kelly',
            'fraction': 0.5,
            'max_risk_per_trade': 0.02,
            'min_size': 100,
            'max_size': 1000
        },
        'stop_loss': {
            'method': 'dynamic',
            'initial_pct': 0.005,
            'min_distance': 5,
            'max_distance': 20,
            'adjustment': {
                'method': 'zone_based',
                'reference': 'zone_boundary',
                'buffer': 2
            }
        },
        'take_profit': {
            'method': 'multi_target',
            'targets': [
                {'size': 0.3, 'ratio': 1.5},
                {'size': 0.3, 'ratio': 2.0},
                {'size': 0.4, 'ratio': 2.5}
            ],
            'adjustment': {
                'method': 'zone_based',
                'reference': 'zone_boundary',
                'buffer': 2
            }
        }
    }
    
    # Set entry rules
    risk_strategy.entry_rules = {
        'execution': {
            'method': 'market',
            'max_slippage': 2,
            'timeout': 100,
            'retry': {
                'attempts': 3,
                'delay': 50
            }
        },
        'filters': {
            'volume': {
                'min': 100,
                'above_average': True,
                'lookback': 20
            },
            'spread': {
                'max': 2,
                'normal_range': True
            },
            'distance': {
                'min': 5,
                'max': 20,
                'reference': 'zone_boundary'
            }
        },
        'timing': {
            'min_bars_since_signal': 1,
            'max_bars_since_signal': 5
        }
    }
    
    # Set exit rules
    risk_strategy.exit_rules = {
        'stop_loss': {
            'initial': {
                'method': 'zone_based',
                'reference': 'zone_boundary',
                'buffer': 2
            },
            'trailing': {
                'activation': 1.0,
                'step': 0.2,
                'lock': 0.5,
                'reference': 'entry'
            },
            'breakeven': {
                'activation': 0.8,
                'buffer': 0.2,
                'reference': 'entry'
            }
        },
        'take_profit': {
            'method': 'scaled',
            'levels': [
                {'size': 0.3, 'ratio': 1.5, 'type': 'limit'},
                {'size': 0.3, 'ratio': 2.0, 'type': 'limit'},
                {'size': 0.4, 'ratio': 2.5, 'type': 'trailing'}
            ],
            'reference': 'zone_boundary'
        },
        'time_stop': {
            'max_bars': 20,
            'conditions': {
                'in_loss': True,
                'no_progress': {
                    'bars': 10,
                    'threshold': 0.2
                }
            },
            'logic': 'AND'
        }
    }
    
    # Create combined strategy
    combined_strategy = CombinedStrategy(
        name="FVGSLTP",
        pattern_strategy=pattern_strategy,
        risk_strategy=risk_strategy
    )
    
    # Set combination method
    combined_strategy.combination_method = 'sequential'  # Pattern then risk
    
    # Set combined strategy parameters
    combined_strategy.parameters = {
        'max_positions': 1,
        'max_risk': 0.02,
        'portfolio_stop': 0.05,
        'daily_stop': 0.02
    }
    
    return combined_strategy

def create_realistic_test_data():
    """Create one day of realistic test data with clear FVG opportunities"""
    # Create one day of 1-minute data
    start_time = datetime(2024, 1, 15, 9, 30, 0)  # Market open
    end_time = datetime(2024, 1, 15, 16, 0, 0)    # Market close
    
    # Generate 390 minutes (6.5 hours of trading)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create data with clear FVG patterns
    np.random.seed(42)  # For reproducible results
    
    # Start price around 100
    base_price = 100.0
    prices = [base_price]
    
    # Create some clear FVG opportunities
    fvg_opportunities = [
        (30, 35, 0.5),   # Bar 30-35: Gap up
        (80, 85, -0.3),  # Bar 80-85: Gap down
        (150, 155, 0.4), # Bar 150-155: Gap up
        (220, 225, -0.2), # Bar 220-225: Gap down
        (300, 305, 0.6), # Bar 300-305: Gap up
    ]
    
    for i in range(1, len(timestamps)):
        # Check if this is an FVG opportunity
        fvg_gap = 0
        for start_bar, end_bar, gap_size in fvg_opportunities:
            if start_bar <= i <= end_bar:
                fvg_gap = gap_size
                break
        
        # Random walk with FVG gaps
        change = np.random.normal(0, 0.05) + fvg_gap  # Small random change + FVG gap
        if i % 60 == 0:  # Every hour, add some trend
            change += np.random.normal(0, 0.2)
        
        new_price = prices[-1] + change
        prices.append(max(0.01, new_price))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.03))
        high = close + volatility
        low = close - volatility
        open_price = close + np.random.normal(0, 0.01)
        
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

def analyze_trades_detailed(strategy, data):
    """Run backtest and analyze trades in detail"""
    print("\n" + "="*80)
    print("FVGSLTP STRATEGY DETAILED TRADE ANALYSIS")
    print("="*80)
    
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
    
    # Analyze each trade in detail
    trades = results.get('trades', [])
    print(f"\nDetailed Trade Analysis:")
    print(f"{'Trade':<4} {'Entry Time':<20} {'Exit Time':<20} {'Entry':<8} {'Exit':<8} {'Stop':<8} {'Target':<8} {'Size':<8} {'PnL':<10} {'Exit Reason':<15}")
    print("-" * 120)
    
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
    
    # Detailed analysis
    print(f"\nTrade Analysis Summary:")
    
    if not trades:
        print("❌ No trades generated - check strategy configuration")
        return
    
    # Check trade timing and logic
    issues_found = []
    
    for i, trade in enumerate(trades, 1):
        entry_time = trade.get('entry_time')
        exit_time = trade.get('exit_time')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        size = trade.get('size', 0)
        exit_reason = trade.get('exit_reason', 'N/A')
        
        # Check timing
        if entry_time and exit_time:
            if entry_time >= exit_time:
                issues_found.append(f"❌ Trade {i}: Entry time ({entry_time}) >= Exit time ({exit_time})")
            else:
                duration = exit_time - entry_time
                if duration < timedelta(minutes=1):
                    issues_found.append(f"⚠️  Trade {i}: Very short duration ({duration})")
                elif duration > timedelta(hours=8):
                    issues_found.append(f"⚠️  Trade {i}: Very long duration ({duration})")
        
        # Check price levels
        if entry_price <= 0 or exit_price <= 0:
            issues_found.append(f"❌ Trade {i}: Invalid prices - Entry: {entry_price}, Exit: {exit_price}")
        
        # Check stop loss and take profit logic
        if stop_loss > 0 and take_profit > 0:
            if entry_price <= stop_loss and entry_price >= take_profit:
                issues_found.append(f"❌ Trade {i}: Entry price between stop loss and take profit")
            
            # Check if exit price makes sense for the exit reason
            if exit_reason == 'Stop Loss' and abs(exit_price - stop_loss) > 0.01:
                issues_found.append(f"⚠️  Trade {i}: Exit price ({exit_price}) doesn't match stop loss ({stop_loss})")
            
            if exit_reason == 'Take Profit' and abs(exit_price - take_profit) > 0.01:
                issues_found.append(f"⚠️  Trade {i}: Exit price ({exit_price}) doesn't match take profit ({take_profit})")
        
        # Check position sizing
        if size <= 0:
            issues_found.append(f"❌ Trade {i}: Invalid position size: {size}")
        
        # Check PnL calculation
        expected_pnl = (exit_price - entry_price) * size
        if abs(expected_pnl - trade.get('pnl', 0)) > 0.01:
            issues_found.append(f"⚠️  Trade {i}: PnL calculation mismatch - Expected: {expected_pnl:.2f}, Actual: {trade.get('pnl', 0):.2f}")
    
    # Print all issues
    if issues_found:
        print("\nIssues Found:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("✅ No major issues found in trade execution!")
    
    # Check strategy components
    print(f"\nStrategy Component Analysis:")
    if hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy:
        print(f"✅ Pattern Strategy: {strategy.pattern_strategy.name}")
        print(f"   Actions: {len(strategy.pattern_strategy.actions)}")
        for action in strategy.pattern_strategy.actions:
            print(f"   - {action.name}: {action.location_strategy}")
            if hasattr(action, 'location_params'):
                print(f"     Params: {action.location_params}")
    else:
        print("❌ No pattern strategy found")
    
    if hasattr(strategy, 'risk_strategy') and strategy.risk_strategy:
        print(f"✅ Risk Strategy: {strategy.risk_strategy.name}")
        print(f"   Entry Method: {strategy.risk_strategy.entry_method}")
        print(f"   Stop Method: {strategy.risk_strategy.stop_method}")
        print(f"   Exit Method: {strategy.risk_strategy.exit_method}")
        print(f"   Stop Loss Pips: {getattr(strategy.risk_strategy, 'stop_loss_pips', 'N/A')}")
        print(f"   Take Profit Pips: {getattr(strategy.risk_strategy, 'take_profit_pips', 'N/A')}")
        print(f"   Risk Per Trade: {getattr(strategy.risk_strategy, 'risk_per_trade', 'N/A')}")
    else:
        print("❌ No risk strategy found")
    
    print(f"\n✅ Detailed analysis complete!")

def main():
    """Main function"""
    print("Creating FVGSLTP strategy...")
    strategy = create_fvgsltp_strategy()
    
    print("Creating one day of realistic test data...")
    data = create_realistic_test_data()
    print(f"Created {len(data)} bars of data from {data.index[0]} to {data.index[-1]}")
    
    # Analyze the strategy
    analyze_trades_detailed(strategy, data)

if __name__ == "__main__":
    main() 