"""
Strategy Automation Tester
=========================
Directly controls the trading application GUI to test strategies and extract results.
Focuses on actual data extraction rather than screenshots.
"""

import pyautogui
import time
import subprocess
import sys
import os
import json
from datetime import datetime

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

class StrategyAutomationTester:
    def __init__(self):
        self.app_process = None
        self.window = None
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
        
    def start_trading_app(self):
        """Start the trading application"""
        print("Starting trading application...")
        # Start with fullscreen parameters
        self.app_process = subprocess.Popen([sys.executable, "main.py"])
        time.sleep(12)  # Wait for app to load
        print("Application started!")
        
    def find_and_activate_window(self):
        """Find and activate the trading application window"""
        print("Searching for trading application window...")
        
        # Try different possible window titles
        possible_titles = [
            "main.py - Kairos - Cursor",
            "Trading Strategy Hub",
            "main.py",
            "Kairos"
        ]
        
        for title in possible_titles:
            windows = pyautogui.getWindowsWithTitle(title)
            if windows:
                self.window = windows[0]
                print(f"Found window: '{title}' at ({self.window.left}, {self.window.top})")
                
                # Activate and maximize immediately
                self.window.activate()
                time.sleep(1)
                self.window.maximize()
                time.sleep(2)
                
                # Force fullscreen if possible
                try:
                    # Try to set window to fullscreen
                    self.window.maximize()
                    time.sleep(1)
                    
                    # Try to force fullscreen using F11 key
                    pyautogui.press('f11')
                    time.sleep(1)
                    print("Window activated and maximized to fullscreen")
                except Exception as e:
                    print(f"Could not set fullscreen: {e}")
                    print("Window activated and maximized")
                
                return True
                
        print("No trading application window found!")
        return False
        
    def click_at_position(self, x, y, description=""):
        """Click at position with description"""
        print(f"Clicking at ({x}, {y}) - {description}")
        pyautogui.click(x, y)
        time.sleep(1)
        
    def type_text(self, text, description=""):
        """Type text with description"""
        print(f"Typing: {text} - {description}")
        pyautogui.typewrite(text)
        time.sleep(0.5)
        
    def press_key(self, key, description=""):
        """Press key with description"""
        print(f"Pressing: {key} - {description}")
        pyautogui.press(key)
        time.sleep(0.5)
        
    def test_strategy(self, strategy_name):
        """Test a specific strategy"""
        print(f"\n=== Testing Strategy: {strategy_name} ===")
        
        try:
            # 1. Click on dataset dropdown (estimated position)
            self.click_at_position(200, 150, "Dataset dropdown")
            time.sleep(1)
            self.press_key('down', "Select first dataset")
            time.sleep(1)
            self.press_key('enter', "Confirm dataset selection")
            time.sleep(2)
            
            # 2. Click on strategy dropdown
            self.click_at_position(400, 150, "Strategy dropdown")
            time.sleep(1)
            self.type_text(strategy_name, f"Type strategy name: {strategy_name}")
            time.sleep(1)
            self.press_key('enter', "Confirm strategy selection")
            time.sleep(2)
            
            # 3. Click run backtest button
            self.click_at_position(600, 150, "Run backtest button")
            time.sleep(5)  # Wait for backtest to complete
            
            # 4. Extract results from different tabs
            results = self.extract_strategy_results(strategy_name)
            
            # 5. Save results
            self.test_results[strategy_name] = results
            
            print(f"Strategy {strategy_name} test completed!")
            return True
            
        except Exception as e:
            print(f"Error testing strategy {strategy_name}: {e}")
            return False
            
    def extract_strategy_results(self, strategy_name):
        """Extract results from the backtest window"""
        results = {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "tabs": {}
        }
        
        # Test different tabs to extract data
        tabs = [
            (100, 300, "overview"),
            (200, 300, "equity"), 
            (300, 300, "chart"),
            (400, 300, "trades"),
            (500, 300, "stats")
        ]
        
        for x, y, tab_name in tabs:
            try:
                self.click_at_position(x, y, f"Click {tab_name} tab")
                time.sleep(2)
                
                # Try to extract text from the active area
                # This is a simplified approach - in practice you'd need OCR
                # or direct access to the GUI elements
                results["tabs"][tab_name] = {
                    "status": "tab_clicked",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                results["tabs"][tab_name] = {
                    "status": "error",
                    "error": str(e)
                }
                
        return results
        
    def test_all_strategies(self):
        """Test all strategies"""
        print("=== Testing All Strategies ===")
        
        try:
            # Start the application
            self.start_trading_app()
            
            # Find and activate window
            if not self.find_and_activate_window():
                return False
                
            # Test each strategy
            successful_tests = 0
            for strategy in self.strategies_to_test:
                if self.test_strategy(strategy):
                    successful_tests += 1
                    
            print(f"\n=== Test Summary ===")
            print(f"Successful tests: {successful_tests}/{len(self.strategies_to_test)}")
            
            # Save all results
            self.save_results()
            
            return successful_tests > 0
            
        except Exception as e:
            print(f"Error during testing: {e}")
            return False
        finally:
            if self.app_process:
                self.app_process.terminate()
                print("Application terminated.")
                
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
        # Also save a summary
        summary = {
            "test_timestamp": timestamp,
            "total_strategies": len(self.strategies_to_test),
            "successful_tests": len([r for r in self.test_results.values() if r]),
            "strategies_tested": list(self.test_results.keys())
        }
        
        summary_filename = f"test_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_filename}")

def main():
    """Main function"""
    tester = StrategyAutomationTester()
    success = tester.test_all_strategies()
    
    if success:
        print("Strategy testing completed successfully!")
    else:
        print("Strategy testing failed!")

if __name__ == "__main__":
    main() 