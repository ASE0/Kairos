#!/usr/bin/env python3
"""
Test script to verify heatmap integration with the main application
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.results_viewer_window import ResultsViewerWindow
from PyQt6.QtWidgets import QApplication

def create_comprehensive_test_data():
    """Create comprehensive test data with multiple building blocks"""
    
    # Create sample time series data
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    end_time = datetime(2024, 1, 1, 16, 0, 0)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create sample OHLC data
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.1, len(time_index))
    prices = base_price + np.cumsum(price_changes)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 0.5, len(time_index)),
        'low': prices - np.random.uniform(0, 0.5, len(time_index)),
        'close': prices + np.random.normal(0, 0.1, len(time_index)),
        'volume': np.random.randint(100, 1000, len(time_index))
    }, index=time_index)
    
    # Create sample trades
    trades = [
        {
            'entry_time': start_time + timedelta(minutes=10),
            'exit_time': start_time + timedelta(minutes=25),
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'size': 1.0
        },
        {
            'entry_time': start_time + timedelta(minutes=45),
            'exit_time': start_time + timedelta(minutes=60),
            'entry_price': 101.0,
            'exit_price': 100.8,
            'pnl': -0.2,
            'size': 1.0
        },
        {
            'entry_time': start_time + timedelta(minutes=120),
            'exit_time': start_time + timedelta(minutes=135),
            'entry_price': 100.8,
            'exit_price': 101.5,
            'pnl': 0.7,
            'size': 1.0
        }
    ]
    
    # Create sample zones
    zones = [
        {
            'timestamp': start_time + timedelta(minutes=5),
            'zone_min': 100.0,
            'zone_max': 101.0,
            'direction': 'bullish'
        },
        {
            'timestamp': start_time + timedelta(minutes=40),
            'zone_min': 100.5,
            'zone_max': 101.5,
            'direction': 'bearish'
        },
        {
            'timestamp': start_time + timedelta(minutes=80),
            'zone_min': 100.8,
            'zone_max': 101.8,
            'direction': 'bullish'
        }
    ]
    
    # Create realistic action_details with different signal patterns
    fvg_signals = pd.Series([True if i % 20 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 15 == 0 else False for i in range(len(time_index))], index=time_index)
    ob_signals = pd.Series([True if i % 25 == 0 else False for i in range(len(time_index))], index=time_index)
    momentum_signals = pd.Series([True if i % 30 == 0 else False for i in range(len(time_index))], index=time_index)
    volume_signals = pd.Series([True if i % 35 == 0 else False for i in range(len(time_index))], index=time_index)
    
    action_details = {
        'fvg': str(fvg_signals),
        'vwap': str(vwap_signals),
        'order_block': str(ob_signals),
        'momentum': str(momentum_signals),
        'volume': str(volume_signals)
    }
    
    # Create comprehensive result data
    result_data = {
        'strategy_name': 'Comprehensive_Multi_Strategy',
        'data': data,
        'trades': trades,
        'zones': zones,
        'action_details': action_details,
        'strategy_params': {
            'filters': [
                {'type': 'vwap'},
                {'type': 'volume'},
                {'type': 'momentum'}
            ],
            'location_strategies': ['VWAP', 'Order Block', 'FVG'],
            'fvg_enabled': True,
            'vwap_enabled': True,
            'order_block_enabled': True,
            'momentum_enabled': True,
            'volume_enabled': True
        },
        'gates_enabled': {
            'location_gate': True,
            'volatility_gate': True,
            'momentum_gate': False,
            'regime_gate': True
        },
        'component_summary': {
            'filters': ['vwap', 'volume', 'momentum'],
            'patterns': ['hammer', 'engulfing'],
            'locations': ['VWAP', 'Order Block', 'FVG'],
            'gates': ['location_gate', 'volatility_gate', 'regime_gate']
        },
        'total_trades': len(trades),
        'total_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'win_rate': 0.67,
        'equity_curve': [100000] + [100000 + i * 10 for i in range(1, len(time_index))],
        'timeframe': '1min',
        'interval': '1min',
        'result_display_name': 'Comprehensive_Multi_Strategy_20240101_090000'
    }
    
    return result_data

def test_heatmap_integration():
    """Test the heatmap integration with comprehensive data"""
    print("Testing heatmap integration with comprehensive data...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Create comprehensive test data
    result_data = create_comprehensive_test_data()
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Test_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened with comprehensive test data.")
    print("Navigate to the 'Heatmap' tab to see the building blocks heatmap.")
    print("")
    print("Expected building blocks:")
    print("- fvg (Fair Value Gap signals)")
    print("- vwap (VWAP signals)")
    print("- order_block (Order Block signals)")
    print("- momentum (Momentum filter signals)")
    print("- volume (Volume filter signals)")
    print("- Trades (Trade entries/exits)")
    print("- Zones (Zone activations)")
    print("")
    print("The heatmap should show:")
    print("- Different signal patterns for each building block")
    print("- Time periods on X-axis (HH:MM format)")
    print("- Building blocks on Y-axis")
    print("- Color intensity representing signal density")
    print("- Darker colors when multiple sub-strategies fire simultaneously")
    print("")
    print("You can test:")
    print("- Different time binning options")
    print("- Different color schemes")
    print("- Show/hide legend")
    print("- Real-time updates when changing settings")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_heatmap_integration() 