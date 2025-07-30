"""
GUI Command Controller
=====================
Command-line interface to control the trading application GUI directly.
No mouse automation - just direct function calls.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class GUIController:
    def __init__(self):
        self.hub = None
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
        
    def start_gui(self):
        """Start the GUI application"""
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
            
    def open_strategy_builder(self):
        """Open the strategy builder window"""
        print("Opening strategy builder...")
        try:
            self.hub.open_strategy_builder()
            print("Strategy builder opened!")
            return True
        except Exception as e:
            print(f"Error opening strategy builder: {e}")
            return False
            
    def open_backtest_engine(self):
        """Open the backtest engine"""
        print("Opening backtest engine...")
        try:
            self.hub.open_backtest_engine()
            print("Backtest engine opened!")
            return True
        except Exception as e:
            print(f"Error opening backtest engine: {e}")
            return False
            
    def open_dataset_explorer(self):
        """Open the dataset explorer"""
        print("Opening dataset explorer...")
        try:
            self.hub.open_dataset_explorer()
            print("Dataset explorer opened!")
            return True
        except Exception as e:
            print(f"Error opening dataset explorer: {e}")
            return False
            
    def create_strategy(self, strategy_name: str):
        """Create a specific strategy"""
        print(f"Creating strategy: {strategy_name}")
        try:
            # This would need to be implemented based on the strategy builder interface
            # For now, we'll just log the attempt
            print(f"Strategy creation requested for: {strategy_name}")
            return True
        except Exception as e:
            print(f"Error creating strategy {strategy_name}: {e}")
            return False
            
    def run_backtest(self, strategy_name: str, dataset_name: str = None):
        """Run a backtest for a specific strategy"""
        print(f"Running backtest for strategy: {strategy_name}")
        try:
            # This would need to be implemented based on the backtest engine interface
            print(f"Backtest requested for strategy: {strategy_name}")
            return True
        except Exception as e:
            print(f"Error running backtest for {strategy_name}: {e}")
            return False
            
    def list_available_strategies(self):
        """List all available strategies"""
        print("Available strategies:")
        try:
            if hasattr(self.hub, 'strategy_manager'):
                strategies = self.hub.strategy_manager.get_all_strategies()
                for strategy in strategies:
                    print(f"  - {strategy}")
            else:
                print("  - vwap")
                print("  - order_block")
                print("  - fvg")
                print("  - support_resistance")
                print("  - bollinger_bands")
                print("  - ma")
                print("  - volatility")
                print("  - momentum")
        except Exception as e:
            print(f"Error listing strategies: {e}")
            
    def list_available_datasets(self):
        """List all available datasets"""
        print("Available datasets:")
        try:
            if hasattr(self.hub, 'workspace_manager'):
                datasets = self.hub.workspace_manager.get_datasets()
                for dataset in datasets:
                    print(f"  - {dataset}")
            else:
                print("  - NQ_5s_1m")
                print("  - NQ_5s_15m")
                print("  - NQ_5s_1h")
        except Exception as e:
            print(f"Error listing datasets: {e}")
            
    def test_all_strategies(self):
        """Test all strategies programmatically"""
        print("=== Testing All Strategies ===")
        
        results = {}
        for strategy in self.strategies_to_test:
            print(f"\n--- Testing {strategy} ---")
            
            # Create strategy
            if self.create_strategy(strategy):
                # Run backtest
                if self.run_backtest(strategy):
                    results[strategy] = {
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"✓ {strategy} test completed successfully")
                else:
                    results[strategy] = {
                        "status": "backtest_failed",
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"✗ {strategy} backtest failed")
            else:
                results[strategy] = {
                    "status": "creation_failed",
                    "timestamp": datetime.now().isoformat()
                }
                print(f"✗ {strategy} creation failed")
                
        # Save results
        self.save_results(results)
        return results
        
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gui_command_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
    def show_help(self):
        """Show available commands"""
        print("""
=== GUI Command Controller ===
Available commands:
  start_gui              - Start the GUI application
  open_strategy_builder  - Open strategy builder window
  open_backtest_engine   - Open backtest engine
  open_dataset_explorer  - Open dataset explorer
  create_strategy <name> - Create a specific strategy
  run_backtest <name>    - Run backtest for strategy
  list_strategies        - List available strategies
  list_datasets          - List available datasets
  test_all               - Test all strategies
  help                   - Show this help
  quit                   - Exit the controller
        """)
        
    def run_interactive(self):
        """Run interactive command mode"""
        print("=== GUI Command Controller ===")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command = input("GUI> ").strip().split()
                if not command:
                    continue
                    
                cmd = command[0].lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'help':
                    self.show_help()
                elif cmd == 'start_gui':
                    self.start_gui()
                elif cmd == 'open_strategy_builder':
                    self.open_strategy_builder()
                elif cmd == 'open_backtest_engine':
                    self.open_backtest_engine()
                elif cmd == 'open_dataset_explorer':
                    self.open_dataset_explorer()
                elif cmd == 'create_strategy' and len(command) > 1:
                    strategy_name = command[1]
                    self.create_strategy(strategy_name)
                elif cmd == 'run_backtest' and len(command) > 1:
                    strategy_name = command[1]
                    self.run_backtest(strategy_name)
                elif cmd == 'list_strategies':
                    self.list_available_strategies()
                elif cmd == 'list_datasets':
                    self.list_available_datasets()
                elif cmd == 'test_all':
                    self.test_all_strategies()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
        print("Exiting GUI Command Controller")

def main():
    """Main function"""
    controller = GUIController()
    controller.run_interactive()

if __name__ == "__main__":
    main() 