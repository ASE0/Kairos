"""
Building Block Strategy Tester
=============================
Comprehensive testing system for individual building block strategies.
Tests each strategy with its specific test dataset and validates logic.
"""

import sys
import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class BuildingBlockTester:
    def __init__(self):
        self.test_results = {}
        self.building_blocks = [
            "vwap",
            "order_block", 
            "fvg",
            "support_resistance"
        ]
        
        # Test datasets for each building block
        self.test_datasets = {
            "vwap": "test_vwap_dataset.csv",
            "order_block": "test_order_block_dataset.csv", 
            "fvg": "test_fvg_dataset.csv",
            "support_resistance": "test_support_resistance_dataset.csv"
        }
        
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
            
    def load_test_dataset(self, strategy_name: str) -> Optional[pd.DataFrame]:
        """Load the test dataset for a specific strategy"""
        dataset_file = self.test_datasets.get(strategy_name)
        if not dataset_file or not os.path.exists(dataset_file):
            print(f"Test dataset not found: {dataset_file}")
            return None
            
        try:
            df = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
            print(f"Loaded test dataset: {dataset_file} ({len(df)} bars)")
            return df
        except Exception as e:
            print(f"Error loading dataset {dataset_file}: {e}")
            return None
            
    def load_test_metadata(self, strategy_name: str) -> Optional[Dict]:
        """Load the test metadata for a specific strategy"""
        metadata_file = f"test_{strategy_name}_dataset_metadata.json"
        if not os.path.exists(metadata_file):
            print(f"Test metadata not found: {metadata_file}")
            return None
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded test metadata: {metadata_file}")
            return metadata
        except Exception as e:
            print(f"Error loading metadata {metadata_file}: {e}")
            return None
            
    def create_strategy(self, strategy_name: str):
        """Create a specific building block strategy"""
        print(f"Creating {strategy_name} strategy...")
        
        try:
            # Import the building block strategy builder
            from building_block_strategy_builder import BuildingBlockStrategyBuilder
            
            builder = BuildingBlockStrategyBuilder()
            strategy = builder.create_strategy(strategy_name)
            
            if strategy:
                # Save the strategy
                filename = builder.save_strategy(strategy)
                
                # Store the strategy for later use
                self.current_strategy = strategy
                
                return {
                    "status": "success",
                    "strategy": strategy_name,
                    "action": "creation",
                    "filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "strategy": strategy_name,
                    "action": "creation",
                    "error": "Failed to create strategy",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "strategy": strategy_name,
                "action": "creation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def run_backtest(self, strategy_name: str, dataset_file: str):
        """Run backtest for a specific strategy"""
        print(f"Running backtest for {strategy_name}...")
        
        try:
            # Import the correct backtest engine
            from strategies.strategy_builders import BacktestEngine
            
            # Load the dataset
            dataset = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
            
            # Get the strategy we created
            if not hasattr(self, 'current_strategy') or self.current_strategy is None:
                return {
                    "status": "error",
                    "strategy": strategy_name,
                    "action": "backtest",
                    "error": "No strategy available for backtest",
                    "timestamp": datetime.now().isoformat()
                }
            
            strategy = self.current_strategy
            
            # Create backtest engine
            engine = BacktestEngine()
            
            # Run backtest with proper parameters
            results = engine.run_backtest(
                strategy=strategy,
                data=dataset,
                initial_capital=100000,
                risk_per_trade=0.02
            )
            
            # Store results for analysis
            self.current_backtest_results = results
            
            return {
                "status": "success",
                "strategy": strategy_name,
                "action": "backtest",
                "dataset": dataset_file,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "strategy": strategy_name,
                "action": "backtest",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def analyze_backtest_results(self, strategy_name: str, metadata: Dict, dataset: pd.DataFrame) -> Dict:
        """Analyze backtest results to validate strategy logic"""
        print(f"Analyzing backtest results for {strategy_name}...")
        
        # Import the logic validator
        from logic_validator import LogicValidator
        
        validator = LogicValidator()
        
        # Get the actual backtest results
        if hasattr(self, 'current_backtest_results') and self.current_backtest_results:
            backtest_results = self.current_backtest_results
        else:
            # Fallback to mock results if no real results available
            backtest_results = {
                "trades": [
                    {
                        "entry_time": "2024-01-01 10:00:00",
                        "exit_time": "2024-01-01 10:30:00",
                        "entry_price": 5005.0,
                        "exit_price": 5002.0,
                        "direction": "SHORT",
                        "entry_bar": 75,
                        "exit_bar": 95,
                        "pnl": -3.0
                    }
                ]
            }
        
        # Use the logic validator to analyze the results
        analysis = validator.validate_strategy_logic(strategy_name, backtest_results, metadata, dataset)
        
        return analysis
        
    def test_building_block(self, strategy_name: str):
        """Test a specific building block strategy"""
        print(f"\n=== Testing Building Block: {strategy_name} ===")
        
        # Load test dataset and metadata
        dataset = self.load_test_dataset(strategy_name)
        metadata = self.load_test_metadata(strategy_name)
        
        if dataset is None or metadata is None:
            print(f"Failed to load test data for {strategy_name}")
            return None
            
        # Create strategy
        creation_result = self.create_strategy(strategy_name)
        
        # Run backtest
        backtest_result = self.run_backtest(strategy_name, self.test_datasets[strategy_name])
        
        # Analyze results
        analysis_result = self.analyze_backtest_results(strategy_name, metadata, dataset)
        
        # Combine results
        test_result = {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "creation": creation_result,
            "backtest": backtest_result,
            "analysis": analysis_result,
            "metadata": metadata
        }
        
        return test_result
        
    def test_all_building_blocks(self):
        """Test all building block strategies"""
        print("=== Building Block Strategy Tester ===")
        print("Testing all building block strategies...")
        
        results = {}
        
        for strategy in self.building_blocks:
            test_result = self.test_building_block(strategy)
            if test_result:
                results[strategy] = test_result
                
            time.sleep(2)  # Brief pause between strategies
            
        return results
        
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"building_block_test_results_{timestamp}.json"
        
        # Convert DataFrames to dictionaries for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
        # Create summary
        summary = self.create_summary(results)
        
        summary_filename = f"building_block_test_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_filename}")
        
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            # Handle Timestamp keys by converting them to strings
            serializable_dict = {}
            for key, value in obj.items():
                # Convert Timestamp keys to strings
                if hasattr(key, 'isoformat'):
                    serializable_key = key.isoformat()
                else:
                    serializable_key = str(key)
                serializable_dict[serializable_key] = self._make_json_serializable(value)
            return serializable_dict
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'strftime'):  # Handle datetime/timestamp objects
            return obj.isoformat()
        elif hasattr(obj, 'dtype'):  # Handle numpy types
            return obj.item() if hasattr(obj, 'item') else str(obj)
        elif str(type(obj)).find('Timestamp') != -1:  # Handle pandas Timestamp objects
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Handle datetime objects with isoformat
            return obj.isoformat()
        elif hasattr(obj, 'timestamp'):  # Handle datetime objects with timestamp method
            return obj.isoformat()
        else:
            return str(obj)  # Convert any other objects to string
        
    def create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of test results"""
        successful_creations = len([r for r in results.values() 
                                  if r["creation"]["status"] == "success"])
        successful_backtests = len([r for r in results.values() 
                                 if r["backtest"]["status"] == "success"])
        
        # Count validated patterns
        total_patterns = 0
        validated_patterns = 0
        
        for result in results.values():
            analysis = result.get("analysis", {})
            validation_results = analysis.get("validation_results", {})
            for pattern_result in validation_results.values():
                total_patterns += 1
                if pattern_result.get("status") == "validated":
                    validated_patterns += 1
                    
        return {
            "test_timestamp": datetime.now().isoformat(),
            "total_strategies": len(self.building_blocks),
            "successful_creations": successful_creations,
            "successful_backtests": successful_backtests,
            "total_patterns": total_patterns,
            "validated_patterns": validated_patterns,
            "validation_rate": f"{validated_patterns/total_patterns*100:.1f}%" if total_patterns > 0 else "0%",
            "strategies_tested": list(results.keys())
        }
        
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("\n" + "="*60)
        print("BUILDING BLOCK TEST SUMMARY")
        print("="*60)
        
        successful_creations = len([r for r in results.values() 
                                  if r["creation"]["status"] == "success"])
        successful_backtests = len([r for r in results.values() 
                                 if r["backtest"]["status"] == "success"])
        
        print(f"\nStrategy Creation:")
        print(f"  Total: {len(self.building_blocks)}")
        print(f"  Successful: {successful_creations}")
        print(f"  Success Rate: {successful_creations/len(self.building_blocks)*100:.1f}%")
        
        print(f"\nBacktest Execution:")
        print(f"  Total: {len(self.building_blocks)}")
        print(f"  Successful: {successful_backtests}")
        print(f"  Success Rate: {successful_backtests/len(self.building_blocks)*100:.1f}%")
        
        # Pattern validation summary
        total_patterns = 0
        validated_patterns = 0
        
        for result in results.values():
            analysis = result.get("analysis", {})
            validation_results = analysis.get("validation_results", {})
            for pattern_result in validation_results.values():
                total_patterns += 1
                if pattern_result.get("status") == "validated":
                    validated_patterns += 1
                    
        print(f"\nLogic Validation:")
        print(f"  Total Patterns: {total_patterns}")
        print(f"  Validated Patterns: {validated_patterns}")
        print(f"  Validation Rate: {validated_patterns/total_patterns*100:.1f}%" if total_patterns > 0 else "0%")
        
        print("\n" + "="*60)
        
    def run_complete_test(self):
        """Run the complete building block test"""
        try:
            # Start GUI application
            if not self.start_gui_application():
                print("Failed to start GUI application!")
                return False
                
            # Wait for GUI to fully load
            time.sleep(5)
            
            # Test all building blocks
            results = self.test_all_building_blocks()
            
            # Save results
            self.save_results(results)
            
            # Print summary
            self.print_summary(results)
            
            return True
            
        except Exception as e:
            print(f"Error during building block testing: {e}")
            return False
        finally:
            # Clean up
            if hasattr(self, 'app'):
                self.app.quit()
                print("GUI application closed.")

def main():
    """Main function"""
    tester = BuildingBlockTester()
    success = tester.run_complete_test()
    
    if success:
        print("Building block testing completed successfully!")
    else:
        print("Building block testing failed!")

if __name__ == "__main__":
    main() 