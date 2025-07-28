import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDateTime, Qt
import time
import datetime as dt

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_date_range_auto_setting():
    """Test that date ranges are automatically set to dataset's actual range"""
    print("Testing date range auto-setting functionality...")
    
    # Create test application
    app = QApplication(sys.argv)
    
    # Create test dataset with known date range
    start_date = datetime(2024, 1, 1, 9, 30, 0)
    end_date = datetime(2024, 1, 31, 16, 0, 0)
    
    # Generate test data
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    test_data = pd.DataFrame({
        'open': [100 + i * 0.01 for i in range(len(dates))],
        'high': [101 + i * 0.01 for i in range(len(dates))],
        'low': [99 + i * 0.01 for i in range(len(dates))],
        'close': [100.5 + i * 0.01 for i in range(len(dates))],
        'volume': [1000 + i for i in range(len(dates))]
    }, index=dates)
    
    print(f"Test dataset created:")
    print(f"  Start date: {test_data.index.min()} (tzinfo: {getattr(test_data.index.min(), 'tzinfo', None)})")
    print(f"  End date: {test_data.index.max()} (tzinfo: {getattr(test_data.index.max(), 'tzinfo', None)})")
    print(f"  Total bars: {len(test_data)}")
    
    # Import the backtest window
    from gui.backtest_window import BacktestWindow
    
    # Create backtest window
    backtest_window = BacktestWindow()
    
    # Simulate dataset loading by directly setting the data
    backtest_window.parent_window = type('MockParent', (), {
        'datasets': {
            'Test Dataset': {
                'data': test_data,
                'metadata': {}  # Empty metadata to test auto-setting
            }
        }
    })()
    
    # Populate datasets dropdown
    backtest_window._populate_datasets()
    
    # Find and select our test dataset
    test_dataset_index = backtest_window.dataset_combo.findText('Test Dataset')
    if test_dataset_index > 0:
        backtest_window.dataset_combo.setCurrentIndex(test_dataset_index)
        print(f"Selected test dataset at index {test_dataset_index}")
        
        # Check if date range was automatically set
        start_date_widget = backtest_window.start_date.dateTime()
        end_date_widget = backtest_window.end_date.dateTime()
        
        print(f"Widget start date: {start_date_widget.toString(Qt.DateFormat.ISODate)}")
        print(f"Widget end date: {end_date_widget.toString(Qt.DateFormat.ISODate)}")
        
        # Convert widget dates to datetime for comparison
        widget_start = start_date_widget.toPyDateTime()
        widget_end = end_date_widget.toPyDateTime()
        print(f"Widget start (local): {widget_start} (tzinfo: {getattr(widget_start, 'tzinfo', None)})")
        print(f"Widget end (local): {widget_end} (tzinfo: {getattr(widget_end, 'tzinfo', None)})")
        print(f"Widget start (UTC): {start_date_widget.toUTC().toPyDateTime()}")
        print(f"Widget end (UTC): {end_date_widget.toUTC().toPyDateTime()}")
        
        # Check if the dates match (within a reasonable tolerance)
        start_diff = abs((widget_start - test_data.index.min()).total_seconds())
        end_diff = abs((widget_end - test_data.index.max()).total_seconds())
        
        print(f"Start date difference: {start_diff} seconds")
        print(f"End date difference: {end_diff} seconds")
        
        # Allow 1 minute tolerance for timezone/formatting differences
        tolerance = 60  # seconds
        
        if start_diff <= tolerance and end_diff <= tolerance:
            print("✅ PASS: Date range was automatically set correctly!")
            return True
        else:
            print("❌ FAIL: Date range was not set correctly")
            print(f"Expected start: {test_data.index.min()}")
            print(f"Expected end: {test_data.index.max()}")
            print(f"Actual start: {widget_start}")
            print(f"Actual end: {widget_end}")
            return False
    else:
        print("❌ FAIL: Could not find test dataset in dropdown")
        return False

def test_date_range_with_metadata():
    """Test that metadata date ranges are respected when valid"""
    print("\nTesting date range with valid metadata...")
    
    # Create test application
    app = QApplication(sys.argv)
    
    # Create test dataset
    start_date = datetime(2024, 1, 1, 9, 30, 0)
    end_date = datetime(2024, 1, 31, 16, 0, 0)
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    test_data = pd.DataFrame({
        'open': [100 + i * 0.01 for i in range(len(dates))],
        'high': [101 + i * 0.01 for i in range(len(dates))],
        'low': [99 + i * 0.01 for i in range(len(dates))],
        'close': [100.5 + i * 0.01 for i in range(len(dates))],
        'volume': [1000 + i for i in range(len(dates))]
    }, index=dates)
    
    # Create metadata with a specific date range
    metadata_start = datetime(2024, 1, 15, 10, 0, 0)
    metadata_end = datetime(2024, 1, 20, 15, 0, 0)
    
    from gui.backtest_window import BacktestWindow
    backtest_window = BacktestWindow()
    
    # Simulate dataset with metadata
    backtest_window.parent_window = type('MockParent', (), {
        'datasets': {
            'Test Dataset with Metadata': {
                'data': test_data,
                'metadata': {
                    'selected_date_range': [
                        metadata_start.isoformat(),
                        metadata_end.isoformat()
                    ]
                }
            }
        }
    })()
    
    # Populate and select dataset
    backtest_window._populate_datasets()
    test_dataset_index = backtest_window.dataset_combo.findText('Test Dataset with Metadata')
    if test_dataset_index > 0:
        backtest_window.dataset_combo.setCurrentIndex(test_dataset_index)
        
        # Check if metadata date range was used
        start_date_widget = backtest_window.start_date.dateTime()
        end_date_widget = backtest_window.end_date.dateTime()
        
        widget_start = start_date_widget.toPyDateTime()
        widget_end = end_date_widget.toPyDateTime()
        
        print(f"Widget start (local): {widget_start} (tzinfo: {getattr(widget_start, 'tzinfo', None)})")
        print(f"Widget end (local): {widget_end} (tzinfo: {getattr(widget_end, 'tzinfo', None)})")
        print(f"Widget start (UTC): {start_date_widget.toUTC().toPyDateTime()}")
        print(f"Widget end (UTC): {end_date_widget.toUTC().toPyDateTime()}")
        
        start_diff = abs((widget_start - metadata_start).total_seconds())
        end_diff = abs((widget_end - metadata_end).total_seconds())
        
        print(f"Metadata start: {metadata_start} (tzinfo: {getattr(metadata_start, 'tzinfo', None)})")
        print(f"Metadata end: {metadata_end} (tzinfo: {getattr(metadata_end, 'tzinfo', None)})")
        print(f"Widget start: {widget_start} (tzinfo: {getattr(widget_start, 'tzinfo', None)})")
        print(f"Widget end: {widget_end} (tzinfo: {getattr(widget_end, 'tzinfo', None)})")
        print(f"Start difference: {start_diff} seconds")
        print(f"End difference: {end_diff} seconds")
        
        tolerance = 60  # seconds
        if start_diff <= tolerance and end_diff <= tolerance:
            print("✅ PASS: Metadata date range was respected!")
            return True
        else:
            print("❌ FAIL: Metadata date range was not respected")
            return False
    else:
        print("❌ FAIL: Could not find test dataset with metadata")
        return False

if __name__ == "__main__":
    print("Running date range fix tests...")
    
    test1_passed = test_date_range_auto_setting()
    test2_passed = test_date_range_with_metadata()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Date range fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Date range fix needs attention.")
    
    print("\nTest completed.")

    print(f"System time zone: {time.tzname}")
    print(f"Python datetime default tzinfo: {dt.datetime.now().tzinfo}") 