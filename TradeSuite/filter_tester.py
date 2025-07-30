"""
Filter Tester
============
Comprehensive testing system for all microstructure and advanced filters.
Tests each filter with synthetic data to ensure proper functionality.
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class FilterTester:
    def __init__(self):
        self.test_results = {}
        
        # All filters to test
        self.filters_to_test = {
            # Microstructure Filters
            "tick_frequency": {
                "type": "tick_frequency",
                "max_ticks_per_second": 50,
                "min_book_depth": 100
            },
            "spread": {
                "type": "spread", 
                "max_spread_ticks": 2,
                "normal_spread_multiple": 5
            },
            "order_flow": {
                "type": "order_flow",
                "min_cvd_threshold": 1000,
                "large_trade_ratio": 0.35
            },
            
            # Advanced Filters
            "volume": {
                "type": "volume",
                "min_volume": 1000,
                "volume_ratio": 1.5
            },
            "time": {
                "type": "time",
                "start_time": "09:30:00",
                "end_time": "16:00:00"
            },
            "volatility": {
                "type": "volatility",
                "min_atr_ratio": 0.001,
                "max_atr_ratio": 0.1
            },
            "momentum": {
                "type": "momentum",
                "momentum_threshold": 0.001,
                "rsi_range": [20, 80]
            },
            "price": {
                "type": "price",
                "min_price": 1000.0,
                "max_price": 10000.0
            },
            
            # Basic Filters
            "vwap": {
                "type": "vwap",
                "condition": "near",
                "tolerance": 0.001,
                "period": 20
            },
            "bollinger_bands": {
                "type": "bollinger_bands",
                "period": 20,
                "std_dev": 2,
                "condition": "inside"
            }
        }
        
    def create_test_data(self, filter_type: str) -> pd.DataFrame:
        """Create synthetic test data for a specific filter"""
        print(f"Creating test data for {filter_type} filter...")
        
        # Create base data
        if filter_type == 'time':
            # For time filter, create data with proper time components
            dates = pd.date_range('2024-01-01 09:00:00', periods=100, freq='1min')
        else:
            dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        base_price = 5000.0
        
        # Generate price data with some patterns
        np.random.seed(42)  # For reproducible results
        prices = [base_price]
        
        for i in range(1, 100):
            # Add some volatility and trends
            change = np.random.normal(0, 0.001)
            if i % 20 == 0:  # Add some spikes
                change += np.random.normal(0, 0.005)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Create realistic OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.0005)))
            low = price * (1 - abs(np.random.normal(0, 0.0005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Create volume with some patterns
            base_volume = 1000
            if filter_type == "volume":
                # Add high volume periods for volume filter testing
                if 20 <= i <= 30 or 60 <= i <= 70:
                    volume = base_volume * np.random.uniform(2, 5)
                else:
                    volume = base_volume * np.random.uniform(0.5, 1.5)
            elif filter_type == "tick_frequency":
                # Add high tick frequency periods
                if 25 <= i <= 35:
                    volume = base_volume * np.random.uniform(3, 6)  # High volume = high ticks
                else:
                    volume = base_volume * np.random.uniform(0.3, 1.2)
            elif filter_type == "spread":
                # Add high volatility periods for spread testing
                if 40 <= i <= 50:
                    high = price * (1 + abs(np.random.normal(0, 0.002)))  # Higher volatility
                    low = price * (1 - abs(np.random.normal(0, 0.002)))
                volume = base_volume * np.random.uniform(0.8, 1.5)
            elif filter_type == "order_flow":
                # Add large trade periods
                if 45 <= i <= 55:
                    volume = base_volume * np.random.uniform(2, 4)
                else:
                    volume = base_volume * np.random.uniform(0.5, 1.2)
            else:
                volume = base_volume * np.random.uniform(0.8, 1.5)
            
            data.append({
                'datetime': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        
        print(f"Created test data: {len(df)} bars for {filter_type} filter")
        return df
        
    def test_filter(self, filter_name: str, filter_config: Dict) -> Dict:
        """Test a specific filter"""
        print(f"\n=== Testing Filter: {filter_name} ===")
        
        try:
            # Create test data
            test_data = self.create_test_data(filter_name)
            
            # Import the filter testing function
            from strategies.strategy_builders import Action
            
            # Create a test action with the filter
            test_action = Action(
                name=f"Test_{filter_name}",
                pattern=None,
                filters=[filter_config]
            )
            
            # Apply the filter
            signals = test_action.apply(test_data)
            
            # Analyze results
            total_bars = len(test_data)
            signal_count = signals.sum()
            signal_rate = signal_count / total_bars if total_bars > 0 else 0
            
            # Check if filter is working (should generate some signals but not too many)
            is_working = signal_count > 0 and signal_rate < 0.95  # Allow up to 95% signal rate
            
            result = {
                "filter_name": filter_name,
                "filter_config": filter_config,
                "total_bars": int(total_bars),
                "signal_count": int(signal_count),
                "signal_rate": float(signal_rate),
                "is_working": bool(is_working),
                "timestamp": datetime.now().isoformat(),
                "action_details": {}  # Empty dict since apply() doesn't return action_details
            }
            
            print(f"✅ {filter_name} filter test completed:")
            print(f"   - Total bars: {total_bars}")
            print(f"   - Signals generated: {signal_count}")
            print(f"   - Signal rate: {signal_rate:.2%}")
            print(f"   - Working: {'Yes' if is_working else 'No'}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error testing {filter_name} filter: {e}")
            return {
                "filter_name": filter_name,
                "filter_config": filter_config,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def test_all_filters(self):
        """Test all filters"""
        print("=== Filter Tester ===")
        print("Testing all microstructure and advanced filters...")
        
        results = {}
        
        for filter_name, filter_config in self.filters_to_test.items():
            result = self.test_filter(filter_name, filter_config)
            results[filter_name] = result
            
            time.sleep(1)  # Brief pause between tests
            
        return results
        
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filter_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {filename}")
        
        # Create summary
        summary = self.create_summary(results)
        
        summary_filename = f"filter_test_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary saved to: {summary_filename}")
        
    def create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of test results"""
        working_filters = len([r for r in results.values() if r.get("is_working", False)])
        error_filters = len([r for r in results.values() if "error" in r])
        
        # Calculate average signal rates
        signal_rates = []
        for result in results.values():
            if "signal_rate" in result:
                signal_rates.append(result["signal_rate"])
        
        avg_signal_rate = np.mean(signal_rates) if signal_rates else 0
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "total_filters": len(self.filters_to_test),
            "working_filters": working_filters,
            "error_filters": error_filters,
            "success_rate": f"{working_filters/len(self.filters_to_test)*100:.1f}%",
            "average_signal_rate": f"{avg_signal_rate:.2%}",
            "filters_tested": list(results.keys())
        }
        
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("\n" + "="*60)
        print("FILTER TEST SUMMARY")
        print("="*60)
        
        working_filters = len([r for r in results.values() if r.get("is_working", False)])
        error_filters = len([r for r in results.values() if "error" in r])
        
        print(f"\nFilter Testing:")
        print(f"  Total Filters: {len(self.filters_to_test)}")
        print(f"  Working Filters: {working_filters}")
        print(f"  Error Filters: {error_filters}")
        print(f"  Success Rate: {working_filters/len(self.filters_to_test)*100:.1f}%")
        
        # Show individual filter results
        print(f"\nIndividual Filter Results:")
        for filter_name, result in results.items():
            status = "✅ Working" if result.get("is_working", False) else "❌ Failed"
            if "error" in result:
                status = "❌ Error"
            signal_rate = result.get("signal_rate", 0)
            print(f"  {filter_name}: {status} (Signal rate: {signal_rate:.2%})")
        
        print("\n" + "="*60)
        
    def run_complete_test(self):
        """Run the complete filter test"""
        try:
            print("🚀 Starting comprehensive filter testing...")
            
            # Test all filters
            results = self.test_all_filters()
            
            # Save results
            self.save_results(results)
            
            # Print summary
            self.print_summary(results)
            
            return True
            
        except Exception as e:
            print(f"❌ Error during filter testing: {e}")
            return False

def main():
    """Main function"""
    tester = FilterTester()
    success = tester.run_complete_test()
    
    if success:
        print("Filter testing completed successfully!")
    else:
        print("Filter testing failed!")

if __name__ == "__main__":
    main() 