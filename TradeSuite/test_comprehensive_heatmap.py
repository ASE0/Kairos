#!/usr/bin/env python3
"""
Test comprehensive heatmap with multiple building blocks
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.results_viewer_window import ResultsViewerWindow
from PyQt6.QtWidgets import QApplication

def create_comprehensive_strategy_data():
    """Create a comprehensive strategy with multiple building blocks"""
    
    # Create 7 hours of 1-minute data (420 periods)
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    end_time = datetime(2024, 1, 1, 16, 0, 0)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': [100 + i * 0.01 + np.random.normal(0, 0.1) for i in range(len(time_index))],
        'high': [100.5 + i * 0.01 + np.random.normal(0, 0.1) for i in range(len(time_index))],
        'low': [99.5 + i * 0.01 + np.random.normal(0, 0.1) for i in range(len(time_index))],
        'close': [100.2 + i * 0.01 + np.random.normal(0, 0.1) for i in range(len(time_index))],
        'volume': [1000 + i + np.random.randint(-100, 100) for i in range(len(time_index))]
    }, index=time_index)
    
    # Create action_details with REAL signal data
    fvg_signals = pd.Series([True if i % 15 == 0 else False for i in range(len(time_index))], index=time_index)
    vwap_signals = pd.Series([True if i % 20 == 5 else False for i in range(len(time_index))], index=time_index)
    mmr_signals = pd.Series([True if i % 12 == 3 else False for i in range(len(time_index))], index=time_index)
    
    action_details = {
        'fvg': fvg_signals,
        'vwap': vwap_signals,
        'mmr': mmr_signals
    }
    
    # Create trades based on signals
    trades = [
        {
            'entry_time': start_time + timedelta(minutes=15),
            'exit_time': start_time + timedelta(minutes=35),
            'entry_price': 100.5,
            'exit_price': 101.2,
            'pnl': 0.7,
            'size': 1.0
        },
        {
            'entry_time': start_time + timedelta(minutes=60),
            'exit_time': start_time + timedelta(minutes=80),
            'entry_price': 101.0,
            'exit_price': 101.8,
            'pnl': 0.8,
            'size': 1.0
        },
        {
            'entry_time': start_time + timedelta(minutes=120),
            'exit_time': start_time + timedelta(minutes=140),
            'entry_price': 101.5,
            'exit_price': 102.1,
            'pnl': 0.6,
            'size': 1.0
        }
    ]
    
    # Create comprehensive strategy params
    strategy_params = {
        'bar_interval_minutes': 1,
        'fvg_enabled': True,
        'fvg_epsilon': 0.01,
        'vwap_enabled': True,
        'vwap_lookback': 20,
        'order_block_enabled': True,
        'ob_lookback': 30,
        'momentum_threshold': 0.5,
        'volatility_window': 10,
        'imbalance_threshold': 0.3,
        'filters': [
            {'type': 'fvg'},
            {'type': 'vwap'},
            {'type': 'momentum'}
        ],
        'location_strategies': ['FVG', 'VWAP', 'Order_Block']
    }
    
    # Create gates
    gates_enabled = {
        'location_gate': True,
        'volatility_gate': True,
        'momentum_gate': True,
        'tick_validation_gate': True,
        'fvg_gate': True,
        'order_block_gate': True
    }
    
    # Create result data
    result_data = {
        'strategy_name': 'Comprehensive_Multi_Building_Block_Strategy',
        'data': data,
        'trades': trades,
        'zones': [],
        'action_details': action_details,
        'strategy_params': strategy_params,
        'gates_enabled': gates_enabled,
        'total_trades': len(trades),
        'total_return': 2.1,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.05,
        'win_rate': 1.0,
        'equity_curve': [100000] + [100000 + i * 0.1 for i in range(1, len(time_index))],
        'timeframe': '1min',
        'interval': '1min',
        'result_display_name': 'Comprehensive_Strategy_20240101_090000'
    }
    
    return result_data

def test_comprehensive_heatmap():
    """Test the comprehensive heatmap"""
    print("=== TESTING COMPREHENSIVE HEATMAP ===")
    
    # Create data
    result_data = create_comprehensive_strategy_data()
    
    print(f"Created strategy with:")
    print(f"  - {len(result_data['data'])} time periods")
    print(f"  - {len(result_data['action_details'])} action_details building blocks")
    print(f"  - {len(result_data['trades'])} trades")
    print(f"  - Strategy params: {len(result_data['strategy_params'])} parameters")
    print(f"  - Gates enabled: {sum(result_data['gates_enabled'].values())} gates")
    
    # Start GUI
    app = QApplication(sys.argv)
    results_viewer = ResultsViewerWindow()
    
    # Add result
    strategy_name = result_data.get('strategy_name', 'Comprehensive_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show window
    results_viewer.show()
    
    print("\n=== EXPECTED HEATMAP ===")
    print("✅ X-axis: Time (1-minute intervals)")
    print("✅ Y-axis: Multiple building blocks:")
    print("   - mmr (tick frequency)")
    print("   - fvg (fair value gap)")
    print("   - vwap (volume weighted average price)")
    print("   - order_block")
    print("   - momentum")
    print("   - volatility")
    print("   - imbalance")
    print("   - trades")
    print("   - location")
    print("   - tick_validation")
    print("✅ Colors: Different intensities for each building block's signals")
    print("✅ Time labels: Readable time stamps on X-axis")
    print("✅ Building block labels: Clear names on Y-axis")
    
    print("\nNavigate to the 'Heatmap' tab to see the comprehensive view!")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    test_comprehensive_heatmap()