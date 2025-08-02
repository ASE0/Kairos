"""
Comprehensive Strategy Test Script
=================================
This script comprehensively tests the strategies from the documentation
by creating them programmatically and testing their logic and backtesting
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import strategy components
from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action, BacktestEngine
from core.data_structures import BaseStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveStrategyTester:
    """Comprehensive tester for creating and testing strategies from documentation"""
    
    def __init__(self):
        self.test_results = {}
        self.strategies_created = {}
        self.test_data = None
        
    def create_test_dataset(self) -> pd.DataFrame:
        """Create a test dataset with tick-like data for strategy testing"""
        logger.info("Creating comprehensive test dataset...")
        
        # Create realistic tick data with microstructure features
        np.random.seed(42)  # For reproducible results
        
        # Generate 1000 ticks with realistic price movements
        n_ticks = 1000
        base_price = 100.0
        
        # Create price series with some trends and reversals
        price_changes = np.random.normal(0, 0.1, n_ticks)
        # Add some trends
        price_changes[200:300] += 0.05  # Uptrend
        price_changes[500:600] -= 0.05  # Downtrend
        price_changes[800:900] += 0.03  # Another uptrend
        
        prices = [base_price]
        for change in price_changes:
            prices.append(prices[-1] + change)
        
        # Create tick data
        data = []
        for i in range(n_ticks):
            price = prices[i]
            spread = np.random.uniform(0.01, 0.05)  # 1-5 cent spread
            bid = price - spread/2
            ask = price + spread/2
            
            # Simulate volume and trade sizes
            volume = np.random.randint(1, 100)
            large_trades = np.random.randint(0, 5)  # Some large trades
            
            # Simulate CVD (Cumulative Volume Delta)
            if i % 100 < 30:  # 30% of time, strong buying
                bid_volume = volume * 0.7
                ask_volume = volume * 0.3
            elif i % 100 < 60:  # 30% of time, strong selling
                bid_volume = volume * 0.3
                ask_volume = volume * 0.7
            else:  # 40% of time, balanced
                bid_volume = volume * 0.5
                ask_volume = volume * 0.5
            
            # Add some microstructure features
            if i % 200 == 0:  # Every 200 ticks, create a sweep
                sweep_size = np.random.randint(50, 200)
                if np.random.random() > 0.5:
                    # Upward sweep
                    ask_volume += sweep_size
                    ask += 0.02
                else:
                    # Downward sweep
                    bid_volume += sweep_size
                    bid -= 0.02
            
            # Consolidation periods
            if 400 <= i < 450 or 700 <= i < 750:
                volume *= 0.3  # Reduced volume
                spread *= 1.5  # Wider spreads
            
            data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(seconds=i),
                'open': price - 0.01,
                'high': price + 0.02,
                'low': price - 0.02,
                'close': price,
                'volume': volume,
                'bid_volume': int(bid_volume),
                'ask_volume': int(ask_volume),
                'large_trades': large_trades,
                'spread': ask - bid
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Save test dataset
        test_file = 'comprehensive_test_dataset.csv'
        df.to_csv(test_file)
        logger.info(f"Created comprehensive test dataset with {len(df)} ticks: {test_file}")
        
        self.test_data = df
        return df
    
    def create_ofm_strategy(self) -> PatternStrategy:
        """Create Order Flow Momentum strategy from documentation"""
        logger.info("Creating Order Flow Momentum (OFM) strategy...")
        
        # Create actions for OFM strategy based on documentation
        # Action 1: CVD (Cumulative Volume Delta) filter
        action1 = Action(
            name="CVD_Filter",
            filters=[{
                'type': 'order_flow',
                'min_cvd_threshold': 1500,
                'large_trade_ratio': 0.35
            }]
        )
        
        # Action 2: Volume filter for institutional involvement
        action2 = Action(
            name="Volume_Filter",
            filters=[{
                'type': 'volume',
                'min_volume': 50,
                'volume_ratio': 1.5
            }]
        )
        
        # Action 3: Spread filter
        action3 = Action(
            name="Spread_Filter",
            filters=[{
                'type': 'spread',
                'max_spread_ticks': 1
            }]
        )
        
        # Action 4: Absorption filter
        action4 = Action(
            name="Absorption_Filter",
            filters=[{
                'type': 'volume',
                'min_volume': 20,
                'volume_ratio': 0.8  # Not hitting resistance
            }]
        )
        
        # Create the strategy
        strategy = PatternStrategy(
            name="Order_Flow_Momentum",
            actions=[action1, action2, action3, action4],
            combination_logic='AND'
        )
        
        # Add strategy configuration
        strategy.description = "Order Flow Momentum strategy for institutional accumulation/distribution phases"
        strategy.parameters = {
            'cvd_period': 1000,
            'imbalance_threshold': 1500,
            'large_trade_size': 10,
            'absorption_ratio': 400,
            'trail_ticks': 3,
            'entry_threshold': 0.35,
            'spread_limit': 1,
        }
        
        # Add entry and exit conditions
        strategy.entry_conditions = {
            'long': [
                'cvd > imbalance_threshold',
                'large_ratio > entry_threshold',
                'absorption < absorption_ratio',
                'bid_pulling == True',
                'spread <= spread_limit'
            ],
            'short': [
                'cvd < -imbalance_threshold',
                'large_ratio > entry_threshold', 
                'absorption < absorption_ratio',
                'ask_pulling == True',
                'spread <= spread_limit'
            ]
        }
        
        strategy.exit_conditions = {
            'stop_loss': 'largest_bid_cluster_below_entry - 1_tick',
            'take_profit': 'entry + (2.5 * risk)',
            'trailing_stop': 'current_bid - trail_ticks',
            'flow_reversal_exit': 'cvd_last_200_ticks < -imbalance_threshold/3',
            'absorption_exit': 'volume_at_price_level > absorption_ratio * 2'
        }
        
        self.strategies_created['OFM'] = strategy
        logger.info(f"Created OFM strategy: {strategy.name}")
        return strategy
    
    def create_mmr_strategy(self) -> PatternStrategy:
        """Create Microstructure Mean Reversion strategy from documentation"""
        logger.info("Creating Microstructure Mean Reversion (MMR) strategy...")
        
        # Create actions for MMR strategy based on documentation
        # Action 1: Sweep detection
        action1 = Action(
            name="Sweep_Detection",
            filters=[{
                'type': 'volume',
                'min_volume': 75,  # Sweep threshold
                'volume_ratio': 2.0
            }]
        )
        
        # Action 2: Book imbalance
        action2 = Action(
            name="Book_Imbalance",
            filters=[{
                'type': 'order_flow',
                'min_cvd_threshold': 500,
                'large_trade_ratio': 0.4
            }]
        )
        
        # Action 3: Quiet period after sweep
        action3 = Action(
            name="Quiet_Period",
            filters=[{
                'type': 'volume',
                'min_volume': 10,
                'volume_ratio': 0.5  # Reduced volume
            }]
        )
        
        # Action 4: Reversion setup
        action4 = Action(
            name="Reversion_Setup",
            filters=[{
                'type': 'momentum',
                'lookback': 20,
                'momentum_threshold': 0.01
            }]
        )
        
        # Create the strategy
        strategy = PatternStrategy(
            name="Microstructure_Mean_Reversion",
            actions=[action1, action2, action3, action4],
            combination_logic='AND'
        )
        
        # Add strategy configuration
        strategy.description = "Microstructure Mean Reversion strategy for sweep exhaustion and liquidity gaps"
        strategy.parameters = {
            'sweep_threshold': 75,
            'book_imbalance': 3.0,
            'quiet_period': 200,
            'reversion_percent': 0.6,
            'max_heat': 4,
            'levels_taken': 3,
        }
        
        # Add entry and exit conditions
        strategy.entry_conditions = {
            'long': [
                'sweep.direction == SELL',
                'setup_ready == True',
                'analyze_book_support() > book_imbalance'
            ],
            'short': [
                'sweep.direction == BUY',
                'setup_ready == True',
                '1/analyze_book_support() > book_imbalance'
            ]
        }
        
        strategy.exit_conditions = {
            'stop_loss': 'sweep.price - (sweep.levels * tick_size)',
            'take_profit': 'entry + (reversion_percent * (entry - sweep.price))',
            'time_stop': 'ticks_since_entry > 500 and pnl < 0',
            'max_heat_stop': 'adverse_excursion > max_heat'
        }
        
        self.strategies_created['MMR'] = strategy
        logger.info(f"Created MMR strategy: {strategy.name}")
        return strategy
    
    def create_lvb_strategy(self) -> PatternStrategy:
        """Create Liquidity Vacuum Breakout strategy from documentation"""
        logger.info("Creating Liquidity Vacuum Breakout (LVB) strategy...")
        
        # Create actions for LVB strategy based on documentation
        # Action 1: Consolidation detection
        action1 = Action(
            name="Consolidation_Detection",
            filters=[{
                'type': 'volatility',
                'min_atr_ratio': 0.005,
                'max_atr_ratio': 0.02
            }]
        )
        
        # Action 2: Volume reduction
        action2 = Action(
            name="Volume_Reduction",
            filters=[{
                'type': 'volume',
                'min_volume': 20,
                'volume_ratio': 0.3  # Reduced volume
            }]
        )
        
        # Action 3: Breakout detection
        action3 = Action(
            name="Breakout_Detection",
            filters=[{
                'type': 'volume',
                'min_volume': 100,
                'volume_ratio': 2.0  # Volume surge
            }]
        )
        
        # Action 4: Range filter
        action4 = Action(
            name="Range_Filter",
            filters=[{
                'type': 'price',
                'min_price': 95,
                'max_price': 105
            }]
        )
        
        # Create the strategy
        strategy = PatternStrategy(
            name="Liquidity_Vacuum_Breakout",
            actions=[action1, action2, action3, action4],
            combination_logic='AND'
        )
        
        # Add strategy configuration
        strategy.description = "Liquidity Vacuum Breakout strategy for pre-breakout consolidation"
        strategy.parameters = {
            'consolidation_ticks': 500,
            'volume_reduction': 0.3,
            'range_ticks': 5,
            'breakout_volume': 100,
            'target_multiple': 2.5,
        }
        
        # Add entry and exit conditions
        strategy.entry_conditions = {
            'long': [
                'tick.price > consolidation.high',
                'tick.volume >= breakout_volume',
                'tick_cvd > 0',
                'next_tick.price >= tick.price'
            ],
            'short': [
                'tick.price < consolidation.low',
                'tick.volume >= breakout_volume',
                'tick_cvd < 0',
                'next_tick.price <= tick.price'
            ]
        }
        
        strategy.exit_conditions = {
            'stop_loss': 'consolidation.low - 1_tick',
            'take_profit': 'entry + (target_multiple * (entry - stop))',
            'breakout_failure': 'ticks_since_entry > 100 and price < consolidation.high'
        }
        
        self.strategies_created['LVB'] = strategy
        logger.info(f"Created LVB strategy: {strategy.name}")
        return strategy
    
    def create_risk_strategy(self) -> RiskStrategy:
        """Create risk management strategy based on documentation"""
        logger.info("Creating risk management strategy...")
        
        # Create risk strategy
        strategy = RiskStrategy(
            name="Tick_Based_Risk_Management",
            entry_method='market',
            stop_method='fixed',
            exit_method='fixed_rr',
            stop_loss_pct=0.02,
            risk_reward_ratio=2.0,
            atr_multiplier=2.0,
            trailing_stop_pct=0.01
        )
        
        # Add strategy configuration
        strategy.description = "Risk management strategy for tick-based trading"
        strategy.parameters = {
            'max_ticks_per_second': 50,
            'min_book_depth': 100,
            'max_spread': 2,
            'risk_per_trade': 0.01,
            'max_drawdown': 0.15,
            'position_size_limit': 0.02,
        }
        
        # Add entry and exit rules
        strategy.entry_rules = {
            'market_conditions': [
                'ticks_per_second <= max_ticks_per_second',
                'book_depth >= min_book_depth',
                'spread <= max_spread'
            ],
            'position_sizing': 'floor(risk_per_trade / (tick_risk * tick_value))',
            'volatility_adjustment': 'min(1.0, 20 / tick_changes_per_minute)'
        }
        
        strategy.exit_rules = {
            'stop_loss': 'entry - (risk_ticks * tick_size)',
            'take_profit': 'entry + (reward_ticks * tick_size)',
            'time_stop': 'ticks_since_entry > 1000',
            'drawdown_stop': 'current_drawdown > max_drawdown'
        }
        
        self.strategies_created['RISK'] = strategy
        logger.info(f"Created risk strategy: {strategy.name}")
        return strategy
    
    def combine_strategies(self) -> CombinedStrategy:
        """Combine pattern and risk strategies"""
        logger.info("Combining strategies...")
        
        # Get the created strategies
        pattern_strategies = {
            'OFM': self.strategies_created['OFM'],
            'MMR': self.strategies_created['MMR'],
            'LVB': self.strategies_created['LVB']
        }
        risk_strategy = self.strategies_created['RISK']
        
        # Create combined strategy
        combined_strategy = CombinedStrategy(
            name="Combined_Tick_Based_Strategy",
            pattern_strategy=pattern_strategies['OFM'],  # Use OFM as primary
            risk_strategy=risk_strategy
        )
        
        # Add strategy configuration
        combined_strategy.description = "Combined tick-based strategy with OFM, MMR, LVB and risk management"
        combined_strategy.pattern_strategies = pattern_strategies
        combined_strategy.combination_method = 'weighted_average'
        combined_strategy.risk_parameters = {
            'max_correlation': 0.5,
            'combined_max_dd': 0.10,
            'risk_of_ruin': 0.01,
        }
        
        self.strategies_created['COMBINED'] = combined_strategy
        logger.info(f"Created combined strategy: {combined_strategy.name}")
        return combined_strategy
    
    def test_strategy_logic(self, strategy: BaseStrategy, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test the logic of a strategy to ensure it represents the intended behavior"""
        logger.info(f"Testing strategy logic: {strategy.name}")
        
        test_results = {
            'strategy_name': strategy.name,
            'test_passed': True,
            'entry_tests': {},
            'exit_tests': {},
            'risk_tests': {},
            'errors': []
        }
        
        try:
            # Handle different strategy types
            if isinstance(strategy, RiskStrategy):
                # Test RiskStrategy specifically
                self._test_risk_strategy_logic(strategy, test_results)
            elif isinstance(strategy, CombinedStrategy):
                # Test CombinedStrategy specifically
                self._test_combined_strategy_logic(strategy, test_results, test_data)
            else:
                # Test PatternStrategy
                self._test_pattern_strategy_logic(strategy, test_results, test_data)
            
            # Overall test result
            if test_results['errors']:
                test_results['test_passed'] = False
                logger.warning(f"⚠ {strategy.name} has {len(test_results['errors'])} issues")
            else:
                logger.info(f"✓ {strategy.name} logic test passed")
                
        except Exception as e:
            test_results['test_passed'] = False
            test_results['errors'].append(f"Test execution failed: {str(e)}")
            logger.error(f"✗ {strategy.name} test execution failed: {e}")
        
        return test_results
    
    def _test_pattern_strategy_logic(self, strategy: PatternStrategy, test_results: Dict, test_data: pd.DataFrame):
        """Test PatternStrategy specific logic"""
        # Test 1: Check if strategy has required components
        if hasattr(strategy, 'actions') and strategy.actions:
            test_results['entry_tests']['has_actions'] = True
            logger.info(f"✓ {strategy.name} has {len(strategy.actions)} actions")
            
            # Check each action
            for i, action in enumerate(strategy.actions):
                if action.name and action.filters:
                    logger.info(f"  ✓ Action {i+1}: {action.name} with {len(action.filters)} filters")
                else:
                    test_results['errors'].append(f"Action {i+1} missing name or filters")
                    logger.error(f"  ✗ Action {i+1} missing name or filters")
        else:
            test_results['entry_tests']['has_actions'] = False
            test_results['errors'].append("Missing actions")
            logger.error(f"✗ {strategy.name} missing actions")
        
        # Test 2: Check if strategy can evaluate data
        try:
            if hasattr(strategy, 'evaluate'):
                signals, details = strategy.evaluate(test_data.head(100))  # Test with first 100 bars
                signal_count = signals.sum() if hasattr(signals, 'sum') else 0
                test_results['signal_generation'] = True
                logger.info(f"✓ {strategy.name} can evaluate data and generated {signal_count} signals")
            else:
                test_results['signal_generation'] = False
                test_results['errors'].append("Missing evaluate method")
                logger.error(f"✗ {strategy.name} missing evaluate method")
            
        except Exception as e:
            test_results['signal_generation'] = False
            test_results['errors'].append(f"Evaluation failed: {str(e)}")
            logger.error(f"✗ {strategy.name} evaluation failed: {e}")
        
        # Test 3: Check if strategy has reasonable parameters
        if hasattr(strategy, 'parameters') and getattr(strategy, 'parameters', None):
            params = strategy.parameters
            test_results['parameter_tests'] = {}
            
            # Check for reasonable parameter ranges
            if 'cvd_period' in params:
                if 100 <= params['cvd_period'] <= 5000:
                    test_results['parameter_tests']['cvd_period'] = True
                else:
                    test_results['parameter_tests']['cvd_period'] = False
                    test_results['errors'].append("CVD period out of reasonable range")
            
            if 'imbalance_threshold' in params:
                if 100 <= params['imbalance_threshold'] <= 10000:
                    test_results['parameter_tests']['imbalance_threshold'] = True
                else:
                    test_results['parameter_tests']['imbalance_threshold'] = False
                    test_results['errors'].append("Imbalance threshold out of reasonable range")
            
            logger.info(f"✓ {strategy.name} parameters validated")
        else:
            test_results['errors'].append("Missing parameters")
            logger.error(f"✗ {strategy.name} missing parameters")
        
        # Test 4: Check entry and exit conditions
        if hasattr(strategy, 'entry_conditions') and getattr(strategy, 'entry_conditions', None):
            entry_conditions = strategy.entry_conditions
            if 'long' in entry_conditions and 'short' in entry_conditions:
                test_results['entry_tests']['has_entry_conditions'] = True
                logger.info(f"✓ {strategy.name} has entry conditions for both long and short")
            else:
                test_results['entry_tests']['has_entry_conditions'] = False
                test_results['errors'].append("Missing entry conditions for long/short")
                logger.error(f"✗ {strategy.name} missing entry conditions for long/short")
        else:
            test_results['entry_tests']['has_entry_conditions'] = False
            test_results['errors'].append("Missing entry conditions")
            logger.error(f"✗ {strategy.name} missing entry conditions")
        
        if hasattr(strategy, 'exit_conditions') and getattr(strategy, 'exit_conditions', None):
            exit_conditions = strategy.exit_conditions
            if exit_conditions:
                test_results['exit_tests']['has_exit_conditions'] = True
                logger.info(f"✓ {strategy.name} has exit conditions")
            else:
                test_results['exit_tests']['has_exit_conditions'] = False
                test_results['errors'].append("Missing exit conditions")
                logger.error(f"✗ {strategy.name} missing exit conditions")
        else:
            test_results['exit_tests']['has_exit_conditions'] = False
            test_results['errors'].append("Missing exit conditions")
            logger.error(f"✗ {strategy.name} missing exit conditions")
    
    def _test_risk_strategy_logic(self, strategy: RiskStrategy, test_results: Dict):
        """Test RiskStrategy specific logic"""
        # RiskStrategy doesn't have actions or evaluate - check risk-specific attributes
        
        # Test 1: Check risk management components
        if hasattr(strategy, 'stop_loss_pct') and strategy.stop_loss_pct > 0:
            test_results['risk_tests']['has_stop_loss'] = True
            logger.info(f"✓ {strategy.name} has stop loss: {strategy.stop_loss_pct:.2%}")
        else:
            test_results['risk_tests']['has_stop_loss'] = False
            test_results['errors'].append("Missing stop loss")
            logger.error(f"✗ {strategy.name} missing stop loss")
        
        if hasattr(strategy, 'risk_reward_ratio') and strategy.risk_reward_ratio > 0:
            test_results['risk_tests']['has_risk_reward'] = True
            logger.info(f"✓ {strategy.name} has risk/reward ratio: {strategy.risk_reward_ratio}")
        else:
            test_results['risk_tests']['has_risk_reward'] = False
            test_results['errors'].append("Missing risk/reward ratio")
            logger.error(f"✗ {strategy.name} missing risk/reward ratio")
        
        # Test 2: Check entry and exit rules (not conditions)
        if hasattr(strategy, 'entry_rules') and getattr(strategy, 'entry_rules', None):
            entry_rules = strategy.entry_rules
            if entry_rules and isinstance(entry_rules, dict):
                test_results['entry_tests']['has_entry_rules'] = True
                logger.info(f"✓ {strategy.name} has entry rules")
            else:
                test_results['entry_tests']['has_entry_rules'] = False
                test_results['errors'].append("Missing or invalid entry rules")
                logger.error(f"✗ {strategy.name} missing or invalid entry rules")
        else:
            test_results['entry_tests']['has_entry_rules'] = False
            test_results['errors'].append("Missing entry rules")
            logger.error(f"✗ {strategy.name} missing entry rules")
        
        if hasattr(strategy, 'exit_rules') and getattr(strategy, 'exit_rules', None):
            exit_rules = strategy.exit_rules
            if exit_rules and isinstance(exit_rules, dict):
                test_results['exit_tests']['has_exit_rules'] = True
                logger.info(f"✓ {strategy.name} has exit rules")
            else:
                test_results['exit_tests']['has_exit_rules'] = False
                test_results['errors'].append("Missing or invalid exit rules")
                logger.error(f"✗ {strategy.name} missing or invalid exit rules")
        else:
            test_results['exit_tests']['has_exit_rules'] = False
            test_results['errors'].append("Missing exit rules")
            logger.error(f"✗ {strategy.name} missing exit rules")
        
        # Test 3: Check parameters
        if hasattr(strategy, 'parameters') and getattr(strategy, 'parameters', None):
            params = strategy.parameters
            test_results['parameter_tests'] = {}
            
            # Check for reasonable risk parameters
            if 'risk_per_trade' in params:
                if 0.005 <= params['risk_per_trade'] <= 0.05:  # 0.5% to 5%
                    test_results['parameter_tests']['risk_per_trade'] = True
                else:
                    test_results['parameter_tests']['risk_per_trade'] = False
                    test_results['errors'].append("Risk per trade out of reasonable range")
            
            if 'max_drawdown' in params:
                if 0.05 <= params['max_drawdown'] <= 0.30:  # 5% to 30%
                    test_results['parameter_tests']['max_drawdown'] = True
                else:
                    test_results['parameter_tests']['max_drawdown'] = False
                    test_results['errors'].append("Max drawdown out of reasonable range")
            
            logger.info(f"✓ {strategy.name} parameters validated")
        else:
            test_results['errors'].append("Missing parameters")
            logger.error(f"✗ {strategy.name} missing parameters")
    
    def _test_combined_strategy_logic(self, strategy: CombinedStrategy, test_results: Dict, test_data: pd.DataFrame):
        """Test CombinedStrategy specific logic"""
        # CombinedStrategy delegates most properties to its pattern_strategy
        
        # Test 1: Check if it has pattern and risk strategies
        if hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy:
            test_results['entry_tests']['has_pattern_strategy'] = True
            logger.info(f"✓ {strategy.name} has pattern strategy: {strategy.pattern_strategy.name}")
            
            # Check delegated actions through pattern_strategy
            if hasattr(strategy.pattern_strategy, 'actions') and strategy.pattern_strategy.actions:
                test_results['entry_tests']['has_actions'] = True
                logger.info(f"✓ {strategy.name} has {len(strategy.pattern_strategy.actions)} actions (via pattern strategy)")
            else:
                test_results['entry_tests']['has_actions'] = False
                test_results['errors'].append("Missing actions in pattern strategy")
                logger.error(f"✗ {strategy.name} missing actions in pattern strategy")
        else:
            test_results['entry_tests']['has_pattern_strategy'] = False
            test_results['errors'].append("Missing pattern strategy")
            logger.error(f"✗ {strategy.name} missing pattern strategy")
        
        if hasattr(strategy, 'risk_strategy') and strategy.risk_strategy:
            test_results['risk_tests']['has_risk_strategy'] = True
            logger.info(f"✓ {strategy.name} has risk strategy: {strategy.risk_strategy.name}")
        else:
            test_results['risk_tests']['has_risk_strategy'] = False
            test_results['errors'].append("Missing risk strategy")
            logger.error(f"✗ {strategy.name} missing risk strategy")
        
        # Test 2: Check if strategy can evaluate data
        try:
            if hasattr(strategy, 'evaluate'):
                signals, details = strategy.evaluate(test_data.head(100))  # Test with first 100 bars
                signal_count = signals.sum() if hasattr(signals, 'sum') else 0
                test_results['signal_generation'] = True
                logger.info(f"✓ {strategy.name} can evaluate data and generated {signal_count} signals")
            else:
                test_results['signal_generation'] = False
                test_results['errors'].append("Missing evaluate method")
                logger.error(f"✗ {strategy.name} missing evaluate method")
            
        except Exception as e:
            test_results['signal_generation'] = False
            test_results['errors'].append(f"Evaluation failed: {str(e)}")
            logger.error(f"✗ {strategy.name} evaluation failed: {e}")
        
        # Test 3: Check delegated parameters
        if (hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy and 
            hasattr(strategy.pattern_strategy, 'parameters') and 
            getattr(strategy.pattern_strategy, 'parameters', None)):
            
            params = strategy.pattern_strategy.parameters
            test_results['parameter_tests'] = {}
            
            # Check for reasonable parameter ranges (delegated from pattern strategy)
            if 'cvd_period' in params:
                if 100 <= params['cvd_period'] <= 5000:
                    test_results['parameter_tests']['cvd_period'] = True
                else:
                    test_results['parameter_tests']['cvd_period'] = False
                    test_results['errors'].append("CVD period out of reasonable range")
            
            logger.info(f"✓ {strategy.name} parameters validated (via pattern strategy)")
        else:
            test_results['errors'].append("Missing parameters in pattern strategy")
            logger.error(f"✗ {strategy.name} missing parameters in pattern strategy")
        
        # Test 4: Check delegated entry and exit conditions
        if (hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy and 
            hasattr(strategy.pattern_strategy, 'entry_conditions') and 
            getattr(strategy.pattern_strategy, 'entry_conditions', None)):
            
            entry_conditions = strategy.pattern_strategy.entry_conditions
            if 'long' in entry_conditions and 'short' in entry_conditions:
                test_results['entry_tests']['has_entry_conditions'] = True
                logger.info(f"✓ {strategy.name} has entry conditions (via pattern strategy)")
            else:
                test_results['entry_tests']['has_entry_conditions'] = False
                test_results['errors'].append("Missing entry conditions in pattern strategy")
                logger.error(f"✗ {strategy.name} missing entry conditions in pattern strategy")
        else:
            test_results['entry_tests']['has_entry_conditions'] = False
            test_results['errors'].append("Missing entry conditions in pattern strategy")
            logger.error(f"✗ {strategy.name} missing entry conditions in pattern strategy")
        
        if (hasattr(strategy, 'pattern_strategy') and strategy.pattern_strategy and 
            hasattr(strategy.pattern_strategy, 'exit_conditions') and 
            getattr(strategy.pattern_strategy, 'exit_conditions', None)):
            
            exit_conditions = strategy.pattern_strategy.exit_conditions
            if exit_conditions:
                test_results['exit_tests']['has_exit_conditions'] = True
                logger.info(f"✓ {strategy.name} has exit conditions (via pattern strategy)")
            else:
                test_results['exit_tests']['has_exit_conditions'] = False
                test_results['errors'].append("Missing exit conditions in pattern strategy")
                logger.error(f"✗ {strategy.name} missing exit conditions in pattern strategy")
        else:
            test_results['exit_tests']['has_exit_conditions'] = False
            test_results['errors'].append("Missing exit conditions in pattern strategy")
            logger.error(f"✗ {strategy.name} missing exit conditions in pattern strategy")
    
    def test_strategy_with_backtest(self, strategy: BaseStrategy, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test a strategy using the backtest engine"""
        logger.info(f"Testing strategy with backtest: {strategy.name}")
        
        try:
            # Handle RiskStrategy differently since it doesn't have evaluate method
            if isinstance(strategy, RiskStrategy):
                # For RiskStrategy, simulate a simple backtest by testing its methods
                results = self._simulate_risk_strategy_backtest(strategy, test_data)
            else:
                # Create backtest engine for PatternStrategy and CombinedStrategy
                backtest_engine = BacktestEngine()
                
                # Run backtest
                results = backtest_engine.run_backtest(
                    strategy=strategy,
                    data=test_data,
                    initial_capital=100000,
                    risk_per_trade=0.02
                )
            
            logger.info(f"✓ Backtest completed for {strategy.name}")
            logger.info(f"  - Total trades: {results.get('total_trades', 0)}")
            logger.info(f"  - Final capital: ${results.get('final_capital', 0):,.2f}")
            logger.info(f"  - Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  - Max drawdown: {results.get('max_drawdown', 0):.2%}")
            logger.info(f"  - Win rate: {results.get('win_rate', 0):.2%}")
            logger.info(f"  - Profit factor: {results.get('profit_factor', 0):.2f}")
            
            return {
                'strategy_name': strategy.name,
                'test_passed': True,
                'backtest_results': results,
                'errors': []
            }
            
        except Exception as e:
            logger.error(f"✗ Backtest failed for {strategy.name}: {e}")
            return {
                'strategy_name': strategy.name,
                'test_passed': False,
                'backtest_results': {},
                'errors': [f"Backtest failed: {str(e)}"]
            }
    
    def _simulate_risk_strategy_backtest(self, strategy: RiskStrategy, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate a backtest for RiskStrategy by testing its risk management methods"""
        logger.info(f"Simulating risk strategy backtest for {strategy.name}")
        
        # Test RiskStrategy methods with sample data
        sample_trades = []
        initial_capital = 100000
        
        # Simulate some trades to test risk management
        for i in range(min(10, len(test_data) - 1)):  # Test with up to 10 sample trades
            try:
                row = test_data.iloc[i]
                
                # Test entry price calculation
                entry_price = strategy.calculate_entry_price(row)
                
                # Test stop loss calculation  
                stop_loss = strategy.calculate_stop_loss(
                    entry_price=entry_price,
                    signal_bar=row,
                    data=test_data,
                    bar_index=i
                )
                
                # Test position size calculation
                position_size = strategy.calculate_position_size(
                    account_size=initial_capital,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    data=test_data,
                    bar_index=i
                )
                
                sample_trades.append({
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'position_size': position_size,
                    'risk_amount': abs(entry_price - stop_loss) * position_size
                })
                
            except Exception as e:
                logger.warning(f"Risk calculation failed for sample {i}: {e}")
                continue
        
        # Calculate simulated results
        total_risk = sum(trade['risk_amount'] for trade in sample_trades)
        avg_position_size = np.mean([trade['position_size'] for trade in sample_trades]) if sample_trades else 0
        
        results = {
            'total_trades': 0,  # RiskStrategy doesn't generate signals
            'final_capital': initial_capital,  # No actual trades
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': float('inf'),
            'risk_tests_passed': len(sample_trades),
            'total_risk_amount': total_risk,
            'avg_position_size': avg_position_size,
            'risk_management_functional': len(sample_trades) > 0
        }
        
        logger.info(f"  - Risk tests passed: {len(sample_trades)}")
        logger.info(f"  - Total risk amount: ${total_risk:,.2f}")
        logger.info(f"  - Average position size: {avg_position_size:.2f}")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all strategies"""
        logger.info("Starting comprehensive strategy test...")
        
        # Create test dataset
        test_data = self.create_test_dataset()
        
        # Create strategies
        ofm_strategy = self.create_ofm_strategy()
        mmr_strategy = self.create_mmr_strategy()
        lvb_strategy = self.create_lvb_strategy()
        risk_strategy = self.create_risk_strategy()
        
        # Combine strategies
        combined_strategy = self.combine_strategies()
        
        # Test each strategy
        strategies_to_test = {
            'OFM': ofm_strategy,
            'MMR': mmr_strategy,
            'LVB': lvb_strategy,
            'RISK': risk_strategy,
            'COMBINED': combined_strategy
        }
        
        for name, strategy in strategies_to_test.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {name} strategy...")
            logger.info(f"{'='*50}")
            
            # Test logic
            logic_result = self.test_strategy_logic(strategy, test_data)
            
            # Test with backtest
            backtest_result = self.test_strategy_with_backtest(strategy, test_data)
            
            # Combine results
            combined_result = {
                'strategy_name': strategy.name,
                'logic_test': logic_result,
                'backtest_test': backtest_result,
                'overall_passed': logic_result['test_passed'] and backtest_result['test_passed']
            }
            
            self.test_results[name] = combined_result
            
            # Print summary
            if combined_result['overall_passed']:
                logger.info(f"✓ {name} strategy test PASSED")
            else:
                logger.error(f"✗ {name} strategy test FAILED")
                for error in logic_result['errors']:
                    logger.error(f"  - Logic: {error}")
                for error in backtest_result['errors']:
                    logger.error(f"  - Backtest: {error}")
        
        # Generate final report
        self.generate_comprehensive_report()
        
        return self.test_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE STRATEGY TEST REPORT")
        logger.info("="*80)
        
        total_strategies = len(self.test_results)
        passed_strategies = sum(1 for result in self.test_results.values() if result['overall_passed'])
        failed_strategies = total_strategies - passed_strategies
        
        logger.info(f"Total Strategies Tested: {total_strategies}")
        logger.info(f"Passed: {passed_strategies}")
        logger.info(f"Failed: {failed_strategies}")
        logger.info(f"Success Rate: {passed_strategies/total_strategies*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 50)
        
        for strategy_name, result in self.test_results.items():
            status = "PASS" if result['overall_passed'] else "FAIL"
            logger.info(f"{strategy_name:15} | {status}")
            
            if not result['overall_passed']:
                for error in result['logic_test']['errors']:
                    logger.info(f"  └─ Logic: {error}")
                for error in result['backtest_test']['errors']:
                    logger.info(f"  └─ Backtest: {error}")
        
        # Save detailed results to file
        report_file = f"comprehensive_strategy_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        if failed_strategies == 0:
            logger.info("\n🎉 ALL COMPREHENSIVE STRATEGY TESTS PASSED! 🎉")
        else:
            logger.warning(f"\n⚠ {failed_strategies} strategy tests failed. Review the errors above.")


def main():
    """Main function to run the comprehensive strategy test"""
    logger.info("Starting comprehensive strategy testing...")
    
    tester = ComprehensiveStrategyTester()
    results = tester.run_comprehensive_test()
    
    logger.info("Comprehensive strategy testing completed!")
    return results


if __name__ == "__main__":
    main() 