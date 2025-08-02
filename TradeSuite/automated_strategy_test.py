"""
Automated Strategy Test Script
==============================
This script automates the creation and testing of strategies from the documentation:
- Order Flow Momentum (OFM)
- Microstructure Mean Reversion (MMR) 
- Liquidity Vacuum Breakout (LVB)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import GUI components
from gui.strategy_builder_window import StrategyBuilderWindow
from gui.risk_manager_window import RiskManagerWindow
from gui.strategy_combiner_window import StrategyCombinerWindow
from gui.backtest_window import BacktestWindow

# Import strategy components
from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy
from core.data_structures import BaseStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedStrategyTester:
    """Automated tester for creating and testing strategies from documentation"""
    
    def __init__(self):
        self.test_results = {}
        self.strategies_created = {}
        
    def create_test_dataset(self) -> pd.DataFrame:
        """Create a test dataset with tick-like data for strategy testing"""
        logger.info("Creating test dataset...")
        
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
                'price': price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'bid_volume': int(bid_volume),
                'ask_volume': int(ask_volume),
                'large_trades': large_trades,
                'spread': ask - bid
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Save test dataset
        test_file = 'test_strategy_dataset.csv'
        df.to_csv(test_file)
        logger.info(f"Created test dataset with {len(df)} ticks: {test_file}")
        
        return df
    
    def create_ofm_strategy(self) -> PatternStrategy:
        """Create Order Flow Momentum strategy from documentation"""
        logger.info("Creating Order Flow Momentum (OFM) strategy...")
        
        # Create strategy configuration based on documentation
        ofm_config = {
            'name': 'Order_Flow_Momentum',
            'description': 'Order Flow Momentum strategy for institutional accumulation/distribution phases',
            'parameters': {
                'cvd_period': 1000,  # CVD calculation period
                'imbalance_threshold': 1500,  # Net delta threshold
                'large_trade_size': 10,  # Minimum large trade size
                'absorption_ratio': 400,  # Volume per tick movement
                'trail_ticks': 3,  # Trailing stop in ticks
                'entry_threshold': 0.35,  # Large trade ratio threshold
                'spread_limit': 1,  # Maximum spread in ticks
            },
            'entry_conditions': {
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
            },
            'exit_conditions': {
                'stop_loss': 'largest_bid_cluster_below_entry - 1_tick',  # For long
                'take_profit': 'entry + (2.5 * risk)',  # 2.5:1 reward:risk
                'trailing_stop': 'current_bid - trail_ticks',  # For long
                'flow_reversal_exit': 'cvd_last_200_ticks < -imbalance_threshold/3',
                'absorption_exit': 'volume_at_price_level > absorption_ratio * 2'
            }
        }
        
        # Create the strategy object
        strategy = PatternStrategy(
            name=ofm_config['name'],
            actions=[]  # Will be populated with actions based on conditions
        )
        
        # Store configuration for later use
        strategy.description = ofm_config['description']
        strategy.parameters = ofm_config['parameters']
        strategy.entry_conditions = ofm_config['entry_conditions']
        strategy.exit_conditions = ofm_config['exit_conditions']
        
        self.strategies_created['OFM'] = strategy
        logger.info(f"Created OFM strategy: {strategy.name}")
        return strategy
    
    def create_mmr_strategy(self) -> PatternStrategy:
        """Create Microstructure Mean Reversion strategy from documentation"""
        logger.info("Creating Microstructure Mean Reversion (MMR) strategy...")
        
        # Create strategy configuration based on documentation
        mmr_config = {
            'name': 'Microstructure_Mean_Reversion',
            'description': 'Microstructure Mean Reversion strategy for sweep exhaustion and liquidity gaps',
            'parameters': {
                'sweep_threshold': 75,  # Single aggressive order size
                'book_imbalance': 3.0,  # Bid/ask ratio
                'quiet_period': 200,  # Ticks after sweep
                'reversion_percent': 0.6,  # Reversion target
                'max_heat': 4,  # Maximum adverse excursion
                'levels_taken': 3,  # Minimum levels swept
            },
            'entry_conditions': {
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
            },
            'exit_conditions': {
                'stop_loss': 'sweep.price - (sweep.levels * tick_size)',  # For long
                'take_profit': 'entry + (reversion_percent * (entry - sweep.price))',  # For long
                'time_stop': 'ticks_since_entry > 500 and pnl < 0',
                'max_heat_stop': 'adverse_excursion > max_heat'
            }
        }
        
        # Create the strategy object
        strategy = PatternStrategy(
            name=mmr_config['name'],
            actions=[]  # Will be populated with actions based on conditions
        )
        
        # Store configuration for later use
        strategy.description = mmr_config['description']
        strategy.parameters = mmr_config['parameters']
        strategy.entry_conditions = mmr_config['entry_conditions']
        strategy.exit_conditions = mmr_config['exit_conditions']
        
        self.strategies_created['MMR'] = strategy
        logger.info(f"Created MMR strategy: {strategy.name}")
        return strategy
    
    def create_lvb_strategy(self) -> PatternStrategy:
        """Create Liquidity Vacuum Breakout strategy from documentation"""
        logger.info("Creating Liquidity Vacuum Breakout (LVB) strategy...")
        
        # Create strategy configuration based on documentation
        lvb_config = {
            'name': 'Liquidity_Vacuum_Breakout',
            'description': 'Liquidity Vacuum Breakout strategy for pre-breakout consolidation',
            'parameters': {
                'consolidation_ticks': 500,  # Consolidation period
                'volume_reduction': 0.3,  # Volume reduction vs average
                'range_ticks': 5,  # Maximum range during consolidation
                'breakout_volume': 100,  # Minimum breakout volume
                'target_multiple': 2.5,  # Risk:reward ratio
            },
            'entry_conditions': {
                'long': [
                    'tick.price > consolidation.high',
                    'tick.volume >= breakout_volume',
                    'tick_cvd > 0',  # Buying pressure
                    'next_tick.price >= tick.price'  # Confirmation
                ],
                'short': [
                    'tick.price < consolidation.low',
                    'tick.volume >= breakout_volume',
                    'tick_cvd < 0',  # Selling pressure
                    'next_tick.price <= tick.price'  # Confirmation
                ]
            },
            'exit_conditions': {
                'stop_loss': 'consolidation.low - 1_tick',  # For long
                'take_profit': 'entry + (target_multiple * (entry - stop))',  # For long
                'breakout_failure': 'ticks_since_entry > 100 and price < consolidation.high'  # For long
            }
        }
        
        # Create the strategy object
        strategy = PatternStrategy(
            name=lvb_config['name'],
            actions=[]  # Will be populated with actions based on conditions
        )
        
        # Store configuration for later use
        strategy.description = lvb_config['description']
        strategy.parameters = lvb_config['parameters']
        strategy.entry_conditions = lvb_config['entry_conditions']
        strategy.exit_conditions = lvb_config['exit_conditions']
        
        self.strategies_created['LVB'] = strategy
        logger.info(f"Created LVB strategy: {strategy.name}")
        return strategy
    
    def create_risk_strategy(self) -> RiskStrategy:
        """Create risk management strategy based on documentation"""
        logger.info("Creating risk management strategy...")
        
        # Create risk strategy configuration
        risk_config = {
            'name': 'Tick_Based_Risk_Management',
            'description': 'Risk management strategy for tick-based trading',
            'parameters': {
                'max_ticks_per_second': 50,  # Maximum tick frequency
                'min_book_depth': 100,  # Minimum contracts each side
                'max_spread': 2,  # Maximum spread in ticks
                'risk_per_trade': 0.01,  # 1% risk per trade
                'max_drawdown': 0.15,  # 15% maximum drawdown
                'position_size_limit': 0.02,  # 2% position size
            },
            'entry_rules': {
                'market_conditions': [
                    'ticks_per_second <= max_ticks_per_second',
                    'book_depth >= min_book_depth',
                    'spread <= max_spread'
                ],
                'position_sizing': 'floor(risk_per_trade / (tick_risk * tick_value))',
                'volatility_adjustment': 'min(1.0, 20 / tick_changes_per_minute)'
            },
            'exit_rules': {
                'stop_loss': 'entry - (risk_ticks * tick_size)',
                'take_profit': 'entry + (reward_ticks * tick_size)',
                'time_stop': 'ticks_since_entry > 1000',
                'drawdown_stop': 'current_drawdown > max_drawdown'
            }
        }
        
        # Create the risk strategy object
        strategy = RiskStrategy(
            name=risk_config['name']
        )
        
        # Store configuration for later use
        strategy.description = risk_config['description']
        strategy.parameters = risk_config['parameters']
        strategy.entry_rules = risk_config['entry_rules']
        strategy.exit_rules = risk_config['exit_rules']
        
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
        combined_config = {
            'name': 'Combined_Tick_Based_Strategy',
            'description': 'Combined tick-based strategy with OFM, MMR, LVB and risk management',
            'pattern_strategies': pattern_strategies,
            'risk_strategy': risk_strategy,
            'combination_method': 'weighted_average',  # or 'voting', 'hierarchical'
            'weights': {
                'OFM': 0.4,
                'MMR': 0.3,
                'LVB': 0.3
            },
            'risk_parameters': {
                'max_correlation': 0.5,  # Maximum correlation between strategies
                'combined_max_dd': 0.10,  # 10% maximum drawdown
                'risk_of_ruin': 0.01,  # 1% probability of ruin
            }
        }
        
        # Create the combined strategy object
        combined_strategy = CombinedStrategy(
            name=combined_config['name'],
            pattern_strategy=pattern_strategies['OFM'],  # Use OFM as primary pattern strategy
            risk_strategy=risk_strategy
        )
        
        # Store configuration for later use (only non-property attributes)
        combined_strategy.description = combined_config['description']
        combined_strategy.pattern_strategies = pattern_strategies
        combined_strategy.combination_method = combined_config['combination_method']
        combined_strategy.risk_parameters = combined_config['risk_parameters']
        
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
            # Test 1: Check if strategy has required components
            if hasattr(strategy, 'entry_conditions') and getattr(strategy, 'entry_conditions', None):
                test_results['entry_tests']['has_entry_conditions'] = True
                logger.info(f"✓ {strategy.name} has entry conditions")
            else:
                test_results['entry_tests']['has_entry_conditions'] = False
                test_results['errors'].append("Missing entry conditions")
                logger.error(f"✗ {strategy.name} missing entry conditions")
            
            if hasattr(strategy, 'exit_conditions') and getattr(strategy, 'exit_conditions', None):
                test_results['exit_tests']['has_exit_conditions'] = True
                logger.info(f"✓ {strategy.name} has exit conditions")
            else:
                test_results['exit_tests']['has_exit_conditions'] = False
                test_results['errors'].append("Missing exit conditions")
                logger.error(f"✗ {strategy.name} missing exit conditions")
            
            # Test 2: Check if parameters are reasonable
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
            
            # Test 3: Check if strategy can process test data
            try:
                # Simulate strategy processing
                sample_data = test_data.head(100)  # Use first 100 ticks
                
                # Check if strategy has required methods
                if hasattr(strategy, 'generate_signals'):
                    signals = strategy.generate_signals(sample_data)
                    test_results['signal_generation'] = True
                    logger.info(f"✓ {strategy.name} can generate signals")
                else:
                    test_results['signal_generation'] = False
                    test_results['errors'].append("Missing signal generation method")
                    logger.error(f"✗ {strategy.name} missing signal generation")
                
            except Exception as e:
                test_results['signal_generation'] = False
                test_results['errors'].append(f"Signal generation failed: {str(e)}")
                logger.error(f"✗ {strategy.name} signal generation failed: {e}")
            
            # Test 4: Check risk management components
            if hasattr(strategy, 'risk_parameters') and getattr(strategy, 'risk_parameters', None):
                risk_params = strategy.risk_parameters
                test_results['risk_tests']['has_risk_parameters'] = True
                
                # Check risk parameters
                if 'max_drawdown' in risk_params:
                    if 0.05 <= risk_params['max_drawdown'] <= 0.25:
                        test_results['risk_tests']['max_drawdown'] = True
                    else:
                        test_results['risk_tests']['max_drawdown'] = False
                        test_results['errors'].append("Max drawdown out of reasonable range")
                
                logger.info(f"✓ {strategy.name} risk parameters validated")
            else:
                test_results['risk_tests']['has_risk_parameters'] = False
                test_results['errors'].append("Missing risk parameters")
                logger.error(f"✗ {strategy.name} missing risk parameters")
            
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
            
            test_result = self.test_strategy_logic(strategy, test_data)
            self.test_results[name] = test_result
            
            # Print summary
            if test_result['test_passed']:
                logger.info(f"✓ {name} strategy test PASSED")
            else:
                logger.error(f"✗ {name} strategy test FAILED")
                for error in test_result['errors']:
                    logger.error(f"  - {error}")
        
        # Generate final report
        self.generate_test_report()
        
        return self.test_results
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE STRATEGY TEST REPORT")
        logger.info("="*80)
        
        total_strategies = len(self.test_results)
        passed_strategies = sum(1 for result in self.test_results.values() if result['test_passed'])
        failed_strategies = total_strategies - passed_strategies
        
        logger.info(f"Total Strategies Tested: {total_strategies}")
        logger.info(f"Passed: {passed_strategies}")
        logger.info(f"Failed: {failed_strategies}")
        logger.info(f"Success Rate: {passed_strategies/total_strategies*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        logger.info("-" * 50)
        
        for strategy_name, result in self.test_results.items():
            status = "PASS" if result['test_passed'] else "FAIL"
            logger.info(f"{strategy_name:15} | {status}")
            
            if not result['test_passed']:
                for error in result['errors']:
                    logger.info(f"  └─ {error}")
        
        # Save detailed results to file
        report_file = f"strategy_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        if failed_strategies == 0:
            logger.info("\n🎉 ALL STRATEGY TESTS PASSED! 🎉")
        else:
            logger.warning(f"\n⚠ {failed_strategies} strategy tests failed. Review the errors above.")


def main():
    """Main function to run the automated strategy test"""
    logger.info("Starting automated strategy testing...")
    
    tester = AutomatedStrategyTester()
    results = tester.run_comprehensive_test()
    
    logger.info("Automated strategy testing completed!")
    return results


if __name__ == "__main__":
    main() 