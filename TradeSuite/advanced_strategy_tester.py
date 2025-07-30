"""
Advanced Strategy Tester
=======================
Advanced automated script that creates strategies and runs actual backtests.
Tests the complete workflow from strategy creation to backtest execution.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class AdvancedStrategyTester:
    def __init__(self):
        self.test_results = {}
        self.strategies_to_test = [
            "vwap",
            "order_block", 
            "fvg",
            "support_resistance",
            "bollinger_bands",
            "ma",
            "volatility",
            "momentum"
        ]
        
    def start_gui_application(self):
        """Start the GUI application programmatically"""
        print("Starting GUI application...")
        
        try:
            # Import PyQt6
            from PyQt6.QtWidgets import QApplication
            from gui.main_hub import TradingStrategyHub
            
            # Create QApplication
            self.app = QApplication(sys.argv)
            
            # Create main hub
            self.hub = TradingStrategyHub()
            self.hub.show()
            
            print("GUI application started successfully!")
            return True
            
        except Exception as e:
            print(f"Error starting GUI: {e}")
            return False
            
    def create_strategy_programmatically(self, strategy_name: str):
        """Create a strategy programmatically using the strategy builder"""
        print(f"Creating strategy programmatically: {strategy_name}")
        
        try:
            # Import strategy builders
            from strategies.strategy_builders import *
            
            # Create strategy based on name
            strategy = None
            
            if strategy_name == "vwap":
                strategy = VWAPStrategy()
            elif strategy_name == "order_block":
                strategy = OrderBlockStrategy()
            elif strategy_name == "fvg":
                strategy = FVGStrategy()
            elif strategy_name == "support_resistance":
                strategy = SupportResistanceStrategy()
            elif strategy_name == "bollinger_bands":
                strategy = BollingerBandsStrategy()
            elif strategy_name == "ma":
                strategy = MovingAverageStrategy()
            elif strategy_name == "volatility":
                strategy = VolatilityStrategy()
            elif strategy_name == "momentum":
                strategy = MomentumStrategy()
            else:
                print(f"Unknown strategy: {strategy_name}")
                return None
                
            # Save strategy
            if hasattr(self.hub, 'strategy_manager'):
                self.hub.strategy_manager.save_strategy(strategy)
                print(f"Strategy {strategy_name} created and saved successfully!")
                
            return strategy
            
        except Exception as e:
            print(f"Error creating strategy {strategy_name}: {e}")
            return None
            
    def run_backtest_programmatically(self, strategy_name: str):
        """Run a backtest programmatically"""
        print(f"Running backtest programmatically: {strategy_name}")
        
        try:
            # Get the strategy
            strategy = None
            if hasattr(self.hub, 'strategy_manager'):
                strategy = self.hub.strategy_manager.get_strategy(strategy_name)
                
            if not strategy:
                print(f"Strategy {strategy_name} not found!")
                return None
                
            # Get dataset
            dataset = None
            if hasattr(self.hub, 'workspace_manager'):
                datasets = self.hub.workspace_manager.get_datasets()
                if datasets:
                    dataset = datasets[0]  # Use first available dataset
                    
            if not dataset:
                print("No dataset available!")
                return None
                
            # Run backtest
            from core.backtest_engine import BacktestEngine
            
            engine = BacktestEngine()
            results = engine.run_backtest(strategy, dataset)
            
            print(f"Backtest completed for {strategy_name}!")
            return results
            
        except Exception as e:
            print(f"Error running backtest for {strategy_name}: {e}")
            return None
            
    def test_complete_workflow(self, strategy_name: str):
        """Test the complete workflow for a strategy"""
        print(f"\n=== Testing Complete Workflow: {strategy_name} ===")
        
        workflow_result = {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        # Step 1: Create strategy
        print(f"Step 1: Creating strategy {strategy_name}")
        strategy = self.create_strategy_programmatically(strategy_name)
        
        if strategy:
            workflow_result["steps"]["creation"] = {
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            print(f"✓ Strategy {strategy_name} created successfully")
        else:
            workflow_result["steps"]["creation"] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            print(f"✗ Strategy {strategy_name} creation failed")
            return workflow_result
            
        # Step 2: Run backtest
        print(f"Step 2: Running backtest for {strategy_name}")
        backtest_results = self.run_backtest_programmatically(strategy_name)
        
        if backtest_results:
            workflow_result["steps"]["backtest"] = {
                "status": "success",
                "results": backtest_results,
                "timestamp": datetime.now().isoformat()
            }
            print(f"✓ Backtest for {strategy_name} completed successfully")
        else:
            workflow_result["steps"]["backtest"] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            print(f"✗ Backtest for {strategy_name} failed")
            
        return workflow_result
        
    def test_all_strategies(self):
        """Test all strategies with complete workflow"""
        print("=== Advanced Strategy Tester ===")
        print("Testing complete workflow for all strategies...")
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "strategies": {}
        }
        
        for strategy in self.strategies_to_test:
            workflow_result = self.test_complete_workflow(strategy)
            results["strategies"][strategy] = workflow_result
            
            # Brief pause between strategies
            time.sleep(1)
            
        return results
        
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
        # Create summary
        successful_creations = len([s for s in results["strategies"].values() 
                                  if s["steps"]["creation"]["status"] == "success"])
        successful_backtests = len([s for s in results["strategies"].values() 
                                 if s["steps"]["backtest"]["status"] == "success"])
        
        summary = {
            "test_timestamp": timestamp,
            "total_strategies": len(self.strategies_to_test),
            "successful_creations": successful_creations,
            "successful_backtests": successful_backtests,
            "strategies_tested": list(results["strategies"].keys())
        }
        
        summary_filename = f"advanced_test_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_filename}")
        
        # Print summary
        print("\n=== Test Summary ===")
        print(f"Strategies tested: {len(self.strategies_to_test)}")
        print(f"Successful creations: {successful_creations}")
        print(f"Successful backtests: {successful_backtests}")
        
    def run_complete_test(self):
        """Run the complete advanced test"""
        try:
            # Start GUI application
            if not self.start_gui_application():
                print("Failed to start GUI application!")
                return False
                
            # Wait for GUI to fully load
            time.sleep(5)
            
            # Run all tests
            results = self.test_all_strategies()
            
            # Save results
            self.save_results(results)
            
            return True
            
        except Exception as e:
            print(f"Error during advanced testing: {e}")
            return False
        finally:
            # Clean up
            if hasattr(self, 'app'):
                self.app.quit()
                print("GUI application closed.")

def main():
    """Main function"""
    tester = AdvancedStrategyTester()
    success = tester.run_complete_test()
    
    if success:
        print("Advanced testing completed successfully!")
    else:
        print("Advanced testing failed!")

if __name__ == "__main__":
    main() 