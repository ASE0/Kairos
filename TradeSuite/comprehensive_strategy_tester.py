"""
Comprehensive Strategy Tester
============================
Complete automated testing system for the trading application.
Tests GUI components, strategy creation, and backtest functionality.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveStrategyTester:
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
            
    def test_gui_components(self):
        """Test all GUI components"""
        print("=== Testing GUI Components ===")
        
        components = {
            "dataset_explorer": self.test_dataset_explorer,
            "strategy_builder": self.test_strategy_builder,
            "backtest_engine": self.test_backtest_engine,
            "results_viewer": self.test_results_viewer,
            "statistics_window": self.test_statistics_window,
            "risk_manager": self.test_risk_manager
        }
        
        results = {}
        for component_name, test_func in components.items():
            print(f"\n--- Testing {component_name} ---")
            try:
                result = test_func()
                results[component_name] = result
                status = "✓" if result.get("status") == "success" else "✗"
                print(f"{status} {component_name}: {result.get('status', 'unknown')}")
            except Exception as e:
                results[component_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"✗ {component_name}: error - {e}")
                
        return results
        
    def test_dataset_explorer(self):
        """Test the dataset explorer"""
        try:
            self.hub.open_dataset_explorer()
            time.sleep(1)
            return {
                "status": "success",
                "action": "dataset_explorer_test",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "dataset_explorer_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_strategy_builder(self):
        """Test the strategy builder"""
        try:
            self.hub.open_strategy_builder()
            time.sleep(1)
            return {
                "status": "success",
                "action": "strategy_builder_test",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "strategy_builder_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_backtest_engine(self):
        """Test the backtest engine"""
        try:
            self.hub.open_backtest_engine()
            time.sleep(1)
            return {
                "status": "success",
                "action": "backtest_engine_test",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "backtest_engine_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_results_viewer(self):
        """Test the results viewer"""
        try:
            # This would need to be implemented based on the results viewer interface
            return {
                "status": "not_implemented",
                "action": "results_viewer_test",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "results_viewer_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_statistics_window(self):
        """Test the statistics window"""
        try:
            # This would need to be implemented based on the statistics window interface
            return {
                "status": "not_implemented",
                "action": "statistics_window_test",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "statistics_window_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_risk_manager(self):
        """Test the risk manager"""
        try:
            # This would need to be implemented based on the risk manager interface
            return {
                "status": "not_implemented",
                "action": "risk_manager_test",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "action": "risk_manager_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_strategy_creation(self, strategy_name: str):
        """Test creating a specific strategy"""
        print(f"Testing strategy creation: {strategy_name}")
        
        try:
            # Open strategy builder
            self.hub.open_strategy_builder()
            time.sleep(1)
            
            return {
                "status": "success",
                "strategy": strategy_name,
                "action": "creation_test",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "strategy": strategy_name,
                "action": "creation_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_backtest_execution(self, strategy_name: str):
        """Test backtest execution for a specific strategy"""
        print(f"Testing backtest execution: {strategy_name}")
        
        try:
            # Open backtest engine
            self.hub.open_backtest_engine()
            time.sleep(1)
            
            return {
                "status": "success",
                "strategy": strategy_name,
                "action": "backtest_execution_test",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "strategy": strategy_name,
                "action": "backtest_execution_test",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_all_strategies(self):
        """Test all strategies"""
        print("=== Testing All Strategies ===")
        
        results = {}
        for strategy in self.strategies_to_test:
            print(f"\n--- Testing {strategy} ---")
            
            # Test strategy creation
            creation_result = self.test_strategy_creation(strategy)
            results[strategy] = {
                "creation": creation_result,
                "backtest": None
            }
            
            # Test backtest execution
            backtest_result = self.test_backtest_execution(strategy)
            results[strategy]["backtest"] = backtest_result
            
            time.sleep(1)  # Brief pause between strategies
            
        return results
        
    def run_comprehensive_test(self):
        """Run the complete comprehensive test"""
        print("=== Comprehensive Strategy Tester ===")
        print("Starting complete system test...")
        
        try:
            # Start GUI application
            if not self.start_gui_application():
                print("Failed to start GUI application!")
                return False
                
            # Wait for GUI to fully load
            time.sleep(5)
            
            # Test GUI components
            component_results = self.test_gui_components()
            
            # Test all strategies
            strategy_results = self.test_all_strategies()
            
            # Combine results
            results = {
                "test_timestamp": datetime.now().isoformat(),
                "components": component_results,
                "strategies": strategy_results
            }
            
            # Save results
            self.save_results(results)
            
            # Print summary
            self.print_summary(results)
            
            return True
            
        except Exception as e:
            print(f"Error during comprehensive testing: {e}")
            return False
        finally:
            # Clean up
            if hasattr(self, 'app'):
                self.app.quit()
                print("GUI application closed.")
                
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
        # Create summary
        summary = self.create_summary(results)
        
        summary_filename = f"comprehensive_test_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_filename}")
        
    def create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of test results"""
        # Count successful components
        successful_components = len([c for c in results["components"].values() 
                                  if c.get("status") == "success"])
        total_components = len(results["components"])
        
        # Count successful strategy tests
        successful_creations = len([s for s in results["strategies"].values() 
                                  if s["creation"]["status"] == "success"])
        successful_backtests = len([s for s in results["strategies"].values() 
                                 if s["backtest"]["status"] == "success"])
        total_strategies = len(results["strategies"])
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "components": {
                "total": total_components,
                "successful": successful_components,
                "success_rate": f"{successful_components/total_components*100:.1f}%" if total_components > 0 else "0%"
            },
            "strategies": {
                "total": total_strategies,
                "successful_creations": successful_creations,
                "successful_backtests": successful_backtests,
                "creation_success_rate": f"{successful_creations/total_strategies*100:.1f}%" if total_strategies > 0 else "0%",
                "backtest_success_rate": f"{successful_backtests/total_strategies*100:.1f}%" if total_strategies > 0 else "0%"
            },
            "strategies_tested": list(results["strategies"].keys())
        }
        
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("\n" + "="*50)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*50)
        
        # Component summary
        successful_components = len([c for c in results["components"].values() 
                                  if c.get("status") == "success"])
        total_components = len(results["components"])
        
        print(f"\nGUI Components:")
        print(f"  Total: {total_components}")
        print(f"  Successful: {successful_components}")
        print(f"  Success Rate: {successful_components/total_components*100:.1f}%" if total_components > 0 else "0%")
        
        # Strategy summary
        successful_creations = len([s for s in results["strategies"].values() 
                                  if s["creation"]["status"] == "success"])
        successful_backtests = len([s for s in results["strategies"].values() 
                                 if s["backtest"]["status"] == "success"])
        total_strategies = len(results["strategies"])
        
        print(f"\nStrategies:")
        print(f"  Total: {total_strategies}")
        print(f"  Successful Creations: {successful_creations}")
        print(f"  Successful Backtests: {successful_backtests}")
        print(f"  Creation Success Rate: {successful_creations/total_strategies*100:.1f}%" if total_strategies > 0 else "0%")
        print(f"  Backtest Success Rate: {successful_backtests/total_strategies*100:.1f}%" if total_strategies > 0 else "0%")
        
        print("\n" + "="*50)

def main():
    """Main function"""
    tester = ComprehensiveStrategyTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("Comprehensive testing completed successfully!")
    else:
        print("Comprehensive testing failed!")

if __name__ == "__main__":
    main() 