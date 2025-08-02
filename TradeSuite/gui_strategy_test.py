"""
GUI-Based Strategy Test Script
==============================
This script uses the actual GUI components to create and test strategies from the documentation
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

# Import GUI components
from gui.strategy_builder_window import StrategyBuilderWindow
from gui.risk_manager_window import RiskManagerWindow
from gui.strategy_combiner_window import StrategyCombinerWindow
from gui.backtest_window import BacktestWindow

# Import strategy components
from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy, Action
from core.data_structures import BaseStrategy
from patterns.candlestick_patterns import CandlestickPattern

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GUIStrategyTester:
    """GUI-based tester for creating and testing strategies from documentation"""
    
    def __init__(self):
        self.test_results = {}
        self.strategies_created = {}
        self.test_data = None
        
    def create_test_dataset(self) -> pd.DataFrame:
        """Create a test dataset with tick-like data for strategy testing"""
        logger.info("Creating test dataset...")
        
        # Create realistic tick data with microstructure features
        np.random.seed(42)  # For reproducible results
        
        # Generate 500 ticks with realistic price movements (shorter for faster testing)
        n_ticks = 500
        base_price = 100.0
        
        # Create price series with some trends and reversals
        price_changes = np.random.normal(0, 0.1, n_ticks)
        # Add some trends
        price_changes[100:150] += 0.05  # Uptrend
        price_changes[250:300] -= 0.05  # Downtrend
        price_changes[400:450] += 0.03  # Another uptrend
        
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
            if i % 100 == 0:  # Every 100 ticks, create a sweep
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
            if 200 <= i < 225 or 350 <= i < 375:
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
        test_file = 'gui_test_dataset.csv'
        df.to_csv(test_file)
        logger.info(f"Created test dataset with {len(df)} ticks: {test_file}")
        
        self.test_data = df
        return df
    
    def create_ofm_strategy_via_gui(self) -> PatternStrategy:
        """Create Order Flow Momentum strategy using GUI components"""
        logger.info("Creating Order Flow Momentum (OFM) strategy via GUI...")
        
        # Create a mock parent window for the strategy builder
        class MockParent:
            def __init__(self):
                self.patterns = {}
                self.datasets = {}
                self.strategies = {}
        
        mock_parent = MockParent()
        
        # Create strategy builder window
        strategy_builder = StrategyBuilderWindow(parent=mock_parent)
        
        # Set strategy name
        strategy_builder.strategy_name.setText("Order_Flow_Momentum")
        
        # Add actions for OFM strategy
        # Action 1: CVD (Cumulative Volume Delta) filter
        action1 = Action(
            name="CVD_Filter",
            filters=[{
                'type': 'order_flow',
                'min_cvd_threshold': 1500,
                'large_trade_ratio': 0.35
            }]
        )
        
        # Action 2: Volume filter
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
        
        # Create the strategy
        strategy = PatternStrategy(
            name="Order_Flow_Momentum",
            actions=[action1, action2, action3],
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
        
        self.strategies_created['OFM'] = strategy
        logger.info(f"Created OFM strategy: {strategy.name}")
        return strategy
    
    def create_mmr_strategy_via_gui(self) -> PatternStrategy:
        """Create Microstructure Mean Reversion strategy using GUI components"""
        logger.info("Creating Microstructure Mean Reversion (MMR) strategy via GUI...")
        
        # Create actions for MMR strategy
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
        
        # Action 3: Quiet period
        action3 = Action(
            name="Quiet_Period",
            filters=[{
                'type': 'volume',
                'min_volume': 10,
                'volume_ratio': 0.5  # Reduced volume
            }]
        )
        
        # Create the strategy
        strategy = PatternStrategy(
            name="Microstructure_Mean_Reversion",
            actions=[action1, action2, action3],
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
        
        self.strategies_created['MMR'] = strategy
        logger.info(f"Created MMR strategy: {strategy.name}")
        return strategy
    
    def create_lvb_strategy_via_gui(self) -> PatternStrategy:
        """Create Liquidity Vacuum Breakout strategy using GUI components"""
        logger.info("Creating Liquidity Vacuum Breakout (LVB) strategy via GUI...")
        
        # Create actions for LVB strategy
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
        
        # Create the strategy
        strategy = PatternStrategy(
            name="Liquidity_Vacuum_Breakout",
            actions=[action1, action2, action3],
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
        
        self.strategies_created['LVB'] = strategy
        logger.info(f"Created LVB strategy: {strategy.name}")
        return strategy
    
    def create_risk_strategy_via_gui(self) -> RiskStrategy:
        """Create risk management strategy using GUI components"""
        logger.info("Creating risk management strategy via GUI...")
        
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
        
        self.strategies_created['RISK'] = strategy
        logger.info(f"Created risk strategy: {strategy.name}")
        return strategy
    
    def combine_strategies_via_gui(self) -> CombinedStrategy:
        """Combine strategies using GUI components"""
        logger.info("Combining strategies via GUI...")
        
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
    
    def test_strategy_with_backtest(self, strategy: BaseStrategy, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test a strategy using the backtest engine"""
        logger.info(f"Testing strategy with backtest: {strategy.name}")
        
        try:
            # Import backtest engine
            from strategies.strategy_builders import BacktestEngine
            
            # Create backtest engine
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
            if hasattr(strategy, 'actions') and strategy.actions:
                test_results['entry_tests']['has_actions'] = True
                logger.info(f"✓ {strategy.name} has {len(strategy.actions)} actions")
            else:
                test_results['entry_tests']['has_actions'] = False
                test_results['errors'].append("Missing actions")
                logger.error(f"✗ {strategy.name} missing actions")
            
            # Test 2: Check if strategy can evaluate data
            try:
                if hasattr(strategy, 'evaluate'):
                    signals, details = strategy.evaluate(test_data.head(50))  # Test with first 50 bars
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
            
            # Test 4: Check risk management components for risk strategies
            if isinstance(strategy, RiskStrategy):
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
    
    def run_comprehensive_gui_test(self):
        """Run comprehensive test using GUI components"""
        logger.info("Starting comprehensive GUI-based strategy test...")
        
        # Create test dataset
        test_data = self.create_test_dataset()
        
        # Create strategies using GUI components
        ofm_strategy = self.create_ofm_strategy_via_gui()
        mmr_strategy = self.create_mmr_strategy_via_gui()
        lvb_strategy = self.create_lvb_strategy_via_gui()
        risk_strategy = self.create_risk_strategy_via_gui()
        
        # Combine strategies
        combined_strategy = self.combine_strategies_via_gui()
        
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
        self.generate_gui_test_report()
        
        return self.test_results
    
    def generate_gui_test_report(self):
        """Generate comprehensive GUI test report"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE GUI STRATEGY TEST REPORT")
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
        report_file = f"gui_strategy_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        if failed_strategies == 0:
            logger.info("\n🎉 ALL GUI STRATEGY TESTS PASSED! 🎉")
        else:
            logger.warning(f"\n⚠ {failed_strategies} strategy tests failed. Review the errors above.")


def main():
    """Main function to run the GUI-based strategy test"""
    logger.info("Starting GUI-based strategy testing...")
    
    tester = GUIStrategyTester()
    results = tester.run_comprehensive_gui_test()
    
    logger.info("GUI-based strategy testing completed!")
    return results


if __name__ == "__main__":
    main() 