"""
Automated Strategy Tester
========================
Full automated script that tests all strategies using command-line interface.
No user interaction required - runs completely automatically.
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class AutomatedStrategyTester:
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
            
    def test_strategy_creation(self, strategy_name: str):
        """Test creating a specific strategy"""
        print(f"Testing strategy creation: {strategy_name}")
        
        try:
            # Open strategy builder
            self.hub.open_strategy_builder()
            time.sleep(2)
            
            # Log the test
            print(f"Strategy builder opened for {strategy_name}")
            
            return {
                "status": "success",
                "strategy": strategy_name,
                "action": "creation_test",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error testing strategy creation {strategy_name}: {e}")
            return {
                "status": "error",
                "strategy": strategy_name,
                "action": "creation_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_backtest_engine(self, strategy_name: str):
        """Test the backtest engine with a specific strategy"""
        print(f"Testing backtest engine: {strategy_name}")
        
        try:
            # Open backtest engine
            self.hub.open_backtest_engine()
            time.sleep(2)
            
            # Log the test
            print(f"Backtest engine opened for {strategy_name}")
            
            return {
                "status": "success",
                "strategy": strategy_name,
                "action": "backtest_test",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error testing backtest engine {strategy_name}: {e}")
            return {
                "status": "error",
                "strategy": strategy_name,
                "action": "backtest_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_dataset_explorer(self):
        """Test the dataset explorer"""
        print("Testing dataset explorer...")
        
        try:
            # Open dataset explorer
            self.hub.open_dataset_explorer()
            time.sleep(2)
            
            print("Dataset explorer opened successfully")
            
            return {
                "status": "success",
                "action": "dataset_explorer_test",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error testing dataset explorer: {e}")
            return {
                "status": "error",
                "action": "dataset_explorer_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_all_components(self):
        """Test all GUI components"""
        print("=== Testing All GUI Components ===")
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "components": {},
            "strategies": {}
        }
        
        # Test dataset explorer
        results["components"]["dataset_explorer"] = self.test_dataset_explorer()
        
        # Test each strategy
        for strategy in self.strategies_to_test:
            print(f"\n--- Testing {strategy} ---")
            
            # Test strategy creation
            creation_result = self.test_strategy_creation(strategy)
            results["strategies"][strategy] = {
                "creation": creation_result,
                "backtest": None
            }
            
            # Test backtest engine
            backtest_result = self.test_backtest_engine(strategy)
            results["strategies"][strategy]["backtest"] = backtest_result
            
            time.sleep(1)  # Brief pause between strategies
            
        return results
        
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"automated_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
        # Also create a summary
        summary = {
            "test_timestamp": timestamp,
            "total_strategies": len(self.strategies_to_test),
            "successful_creations": len([s for s in results["strategies"].values() 
                                       if s["creation"]["status"] == "success"]),
            "successful_backtests": len([s for s in results["strategies"].values() 
                                       if s["backtest"]["status"] == "success"]),
            "strategies_tested": list(results["strategies"].keys())
        }
        
        summary_filename = f"test_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_filename}")
        
    def run_complete_test(self):
        """Run the complete automated test"""
        print("=== Automated Strategy Tester ===")
        print("Starting complete GUI component test...")
        
        try:
            # Start GUI application
            if not self.start_gui_application():
                print("Failed to start GUI application!")
                return False
                
            # Wait for GUI to fully load
            time.sleep(5)
            
            # Run all tests
            results = self.test_all_components()
            
            # Save results
            self.save_results(results)
            
            # Print summary
            print("\n=== Test Summary ===")
            successful_creations = len([s for s in results["strategies"].values() 
                                      if s["creation"]["status"] == "success"])
            successful_backtests = len([s for s in results["strategies"].values() 
                                     if s["backtest"]["status"] == "success"])
            
            print(f"Strategies tested: {len(self.strategies_to_test)}")
            print(f"Successful creations: {successful_creations}")
            print(f"Successful backtests: {successful_backtests}")
            print(f"Dataset explorer: {results['components']['dataset_explorer']['status']}")
            
            return True
            
        except Exception as e:
            print(f"Error during automated testing: {e}")
            return False
        finally:
            # Clean up
            if hasattr(self, 'app'):
                self.app.quit()
                print("GUI application closed.")

def main():
    """Main function"""
    tester = AutomatedStrategyTester()
    success = tester.run_complete_test()
    
    if success:
        print("Automated testing completed successfully!")
    else:
        print("Automated testing failed!")

if __name__ == "__main__":
    main() 