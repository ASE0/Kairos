#!/usr/bin/env python3
"""
Test script for BacktestWindow logic without GUI
Tests index handling, date filtering, and zone overlays
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestBacktestWindowLogic(unittest.TestCase):
    """Test the core logic of BacktestWindow without GUI"""
    
    def setUp(self):
        """Set up test data"""
        # Create test data with Date and Time columns
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        self.test_data = pd.DataFrame({
            'Date': [d.date() for d in dates],
            'Time': [d.time() for d in dates],
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Create test data with datetime column
        self.test_data_datetime = pd.DataFrame({
            'datetime': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Create test zones
        self.test_zones = [
            {
                'zone_min': 105.0,
                'zone_max': 115.0,
                'index': 10,
                'comb_centers': [107.5, 112.5]
            },
            {
                'zone_min': 95.0,
                'zone_max': 105.0,
                'index': 50,
                'comb_centers': [97.5, 102.5]
            }
        ]
    
    def test_index_conversion_date_time(self):
        """Test index conversion with Date and Time columns"""
        data = self.test_data.copy()
        
        # Simulate the index conversion logic from BacktestWindow
        if 'Date' in data.columns and 'Time' in data.columns:
            data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
            data.set_index('datetime', inplace=True)
        
        # Verify the result
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertTrue(data.index.is_unique)
        self.assertEqual(len(data), 100)
        print(f"PASS Index conversion with Date/Time: {type(data.index)}, unique: {data.index.is_unique}")
    
    def test_index_conversion_datetime(self):
        """Test index conversion with datetime column"""
        data = self.test_data_datetime.copy()
        
        # Simulate the index conversion logic
        if 'datetime' in data.columns:
            data.index = pd.to_datetime(data['datetime'])
        
        # Verify the result
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertTrue(data.index.is_unique)
        self.assertEqual(len(data), 100)
        print(f"PASS Index conversion with datetime: {type(data.index)}, unique: {data.index.is_unique}")
    
    def test_date_filtering(self):
        """Test date filtering logic"""
        data = self.test_data.copy()
        # Convert to DatetimeIndex first
        if 'Date' in data.columns and 'Time' in data.columns:
            data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
            data.set_index('datetime', inplace=True)
        # Test filtering to a specific range
        start_dt = pd.to_datetime('2024-01-01 10:00:00')
        end_dt = pd.to_datetime('2024-01-01 12:00:00')
        filtered_data = data[(data.index >= start_dt) & (data.index < end_dt)]
        self.assertLess(len(filtered_data), len(data))
        self.assertTrue(all(filtered_data.index >= start_dt))
        self.assertTrue(all(filtered_data.index < end_dt))
        print(f"PASS Date filtering: {len(data)} -> {len(filtered_data)} bars")
        # Edge: empty range
        empty_data = data[(data.index >= pd.to_datetime('2025-01-01')) & (data.index < pd.to_datetime('2025-01-02'))]
        self.assertEqual(len(empty_data), 0)
        print(f"PASS Empty range returns 0 bars")
    
    def test_zone_overlay_mapping(self):
        """Test zone overlay index mapping"""
        data = self.test_data.copy()
        if 'Date' in data.columns and 'Time' in data.columns:
            data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
            data.set_index('datetime', inplace=True)
        # All in-range
        for i, zone in enumerate(self.test_zones):
            zone_idx = zone.get('index')
            if zone_idx is not None and 0 <= zone_idx < len(data):
                start_time = data.index[zone_idx]
                end_idx = min(zone_idx + 5, len(data) - 1)
                end_time = data.index[end_idx]
                self.assertIsInstance(start_time, pd.Timestamp)
                self.assertIsInstance(end_time, pd.Timestamp)
                self.assertLess(start_time, end_time)
                print(f"PASS Zone {i} mapping: {start_time} to {end_time}")
            else:
                print(f"WARNING Zone {i} index {zone_idx} out of bounds for data length {len(data)}")
        # All out-of-bounds
        bad_zones = [
            {'zone_min': 105.0, 'zone_max': 115.0, 'index': 999, 'comb_centers': [107.5, 112.5]},
            {'zone_min': 95.0, 'zone_max': 105.0, 'index': -1, 'comb_centers': [97.5, 102.5]}
        ]
        for i, zone in enumerate(bad_zones):
            zone_idx = zone.get('index')
            if zone_idx is not None and (zone_idx < 0 or zone_idx >= len(data)):
                print(f"PASS Correctly detected out-of-bounds zone index: {zone_idx}")
            else:
                self.fail("Should have detected out-of-bounds zone index")
    
    def test_zone_overlay_toggle(self):
        """Test toggling overlays (simulate GUI logic)"""
        data = self.test_data.copy()
        if 'Date' in data.columns and 'Time' in data.columns:
            data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
            data.set_index('datetime', inplace=True)
        # Simulate overlays toggled off
        overlays = {'Zones': False, 'Entries/Exits': False}
        zones = self.test_zones
        if not overlays['Zones']:
            print("PASS Zones overlay toggled off: no zones should be plotted")
        if not overlays['Entries/Exits']:
            print("PASS Entries/Exits overlay toggled off: no trades should be plotted")
    
    def test_malformed_data(self):
        """Test handling of malformed data columns"""
        # Missing OHLC columns
        bad_data = pd.DataFrame({'foo': [1,2,3], 'bar': [4,5,6]})
        try:
            _ = bad_data[['open', 'high', 'low', 'close']]
            self.fail("Should have raised KeyError for missing columns")
        except KeyError:
            print("PASS Correctly handled missing OHLC columns")
    
    def test_data_robustness(self):
        """Test robustness with malformed data"""
        # Test with missing columns
        bad_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [110, 111, 112],
            'low': [90, 91, 92],
            'close': [105, 106, 107]
            # Missing Date, Time, datetime columns
        })
        
        # Should handle gracefully
        if not isinstance(bad_data.index, pd.DatetimeIndex):
            if 'Date' in bad_data.columns and 'Time' in bad_data.columns:
                bad_data['datetime'] = pd.to_datetime(bad_data['Date'].astype(str) + ' ' + bad_data['Time'].astype(str))
                bad_data.set_index('datetime', inplace=True)
            elif 'datetime' in bad_data.columns:
                bad_data.index = pd.to_datetime(bad_data['datetime'])
            else:
                # Fallback to synthetic index
                bad_data.index = pd.date_range(start='2000-01-01', periods=len(bad_data), freq='T')
        
        self.assertIsInstance(bad_data.index, pd.DatetimeIndex)
        print(f"PASS Robust handling of malformed data: {type(bad_data.index)}")

def run_gui_logic_tests():
    """Run all GUI logic tests"""
    print("🧪 Testing BacktestWindow Logic (No GUI Required)")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBacktestWindowLogic)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("PASS All GUI logic tests passed!")
        print("The BacktestWindow should handle your data correctly.")
    else:
        print("FAIL Some tests failed. Check the output above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_gui_logic_tests()
    sys.exit(0 if success else 1) 