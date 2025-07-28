#!/usr/bin/env python3
"""
Test script to verify GUI components are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from gui.strategy_builder_window import StrategyBuilderWindow

def test_gui_components():
    """Test that the GUI components are properly implemented"""
    app = QApplication(sys.argv)
    
    # Create the strategy builder window
    window = StrategyBuilderWindow()
    
    # Check if microstructure gates are present
    microstructure_gates = [
        'market_environment_gate_check',
        'news_time_gate_check', 
        'tick_validation_gate_check'
    ]
    
    # Check if advanced gates are present
    advanced_gates = [
        'fvg_gate_check',
        'momentum_gate_check',
        'volume_gate_check',
        'time_gate_check',
        'correlation_gate_check',
        'order_block_gate_check'
    ]
    
    # Check if microstructure filters are present
    microstructure_filters = [
        'tick_frequency_filter_check',
        'spread_filter_check',
        'order_flow_filter_check'
    ]
    
    # Check if advanced filters are present
    advanced_filters = [
        'volume_filter_check',
        'time_filter_check',
        'volatility_filter_check',
        'momentum_filter_check',
        'price_filter_check'
    ]
    
    # Test gates
    print("Testing Gates:")
    for gate in microstructure_gates + advanced_gates:
        if hasattr(window, gate):
            checkbox = getattr(window, gate)
            print(f"✓ {gate}: {checkbox.text()}")
        else:
            print(f"✗ {gate}: Missing")
    
    # Test filters
    print("\nTesting Filters:")
    for filter_name in microstructure_filters + advanced_filters:
        if hasattr(window, filter_name):
            checkbox = getattr(window, filter_name)
            print(f"✓ {filter_name}: {checkbox.text()}")
        else:
            print(f"✗ {filter_name}: Missing")
    
    # Show the window briefly
    window.show()
    print("\nGUI window opened. Check if you can see:")
    print("1. 'Execution Gates ▼' button in the Strategy Configuration panel")
    print("2. Click the button to reveal the gates")
    print("3. Look for 'Microstructure Gates' and 'Advanced Gates' sections")
    print("4. In the Action Builder panel, look for 'Microstructure Filters' and 'Advanced Filters' sections")
    
    # Run the app for a few seconds
    import time
    time.sleep(5)
    
    app.quit()
    print("\nTest completed!")

if __name__ == "__main__":
    test_gui_components() 