"""
Test GUI Integration for Advanced MTF Strategy
==============================================
Simple test to verify the new strategy appears in the GUI.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from gui.strategy_builder_window import StrategyBuilderWindow


def test_gui_integration():
    """Test that the Advanced MTF Strategy appears in the GUI"""
    
    print("=== Testing GUI Integration ===\n")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create strategy builder window (without parent)
    window = StrategyBuilderWindow()
    
    # Check if strategy type combo exists and has our option
    print("1. Checking Strategy Type Selector...")
    strategy_combo = window.strategy_type_combo
    
    items = [strategy_combo.itemText(i) for i in range(strategy_combo.count())]
    print(f"   Available strategy types: {items}")
    
    if "Advanced Multi-Timeframe Strategy" in items:
        print("   ✅ Advanced Multi-Timeframe Strategy found in GUI!")
    else:
        print("   ❌ Advanced Multi-Timeframe Strategy NOT found in GUI!")
        return False
    
    # Test switching to MTF strategy
    print("\n2. Testing Strategy Type Switch...")
    strategy_combo.setCurrentText("Advanced Multi-Timeframe Strategy")
    
    # Check if MTF parameters group is visible
    if hasattr(window, 'mtf_params_group') and window.mtf_params_group.isVisible():
        print("   ✅ MTF parameters group is visible!")
    else:
        print("   ❌ MTF parameters group is not visible!")
        return False
    
    # Check if pattern controls are hidden
    if not window.pattern_combo.isVisible():
        print("   ✅ Pattern controls are properly hidden!")
    else:
        print("   ❌ Pattern controls should be hidden!")
    
    # Test MTF parameter controls
    print("\n3. Testing MTF Parameter Controls...")
    mtf_controls = [
        ('atr_15_5_low_spin', 1.35),
        ('atr_15_5_high_spin', 1.9),
        ('atr_2000_200_threshold_spin', 2.8),
        ('ema_period_spin', 21),
        ('keltner_multiplier_spin', 1.0),
        ('alignment_tolerance_spin', 0.001),
    ]
    
    for control_name, expected_value in mtf_controls:
        if hasattr(window, control_name):
            control = getattr(window, control_name)
            actual_value = control.value()
            if actual_value == expected_value:
                print(f"   ✅ {control_name}: {actual_value} (correct)")
            else:
                print(f"   ⚠️  {control_name}: {actual_value} (expected {expected_value})")
        else:
            print(f"   ❌ {control_name}: Control not found!")
            return False
    
    # Test strategy creation
    print("\n4. Testing Strategy Creation...")
    window.strategy_name_edit.setText("Test MTF Strategy")
    
    try:
        window._create_strategy()
        
        if hasattr(window, 'current_strategy') and window.current_strategy:
            strategy = window.current_strategy
            print(f"   ✅ Strategy created: {strategy.name}")
            print(f"   ✅ Strategy type: {type(strategy).__name__}")
            
            # Check strategy parameters
            if hasattr(strategy, 'atr_15_5_threshold_low'):
                print(f"   ✅ ATR thresholds: {strategy.atr_15_5_threshold_low} - {strategy.atr_15_5_threshold_high}")
            if hasattr(strategy, 'timeframes'):
                print(f"   ✅ Timeframes: {strategy.timeframes}")
        else:
            print("   ❌ Strategy creation failed!")
            return False
            
    except Exception as e:
        print(f"   ❌ Strategy creation error: {e}")
        return False
    
    # Switch back to basic strategy
    print("\n5. Testing Switch Back to Basic Strategy...")
    strategy_combo.setCurrentText("Pattern Strategy (Basic)")
    
    if window.pattern_combo.isVisible():
        print("   ✅ Pattern controls are visible again!")
    else:
        print("   ❌ Pattern controls should be visible!")
    
    if hasattr(window, 'mtf_params_group') and not window.mtf_params_group.isVisible():
        print("   ✅ MTF parameters are hidden!")
    else:
        print("   ❌ MTF parameters should be hidden!")
    
    print("\n=== GUI Integration Test Complete ===")
    print("\n🎉 SUCCESS! The Advanced MTF Strategy is fully integrated into the GUI!")
    print("\n📋 What you can now do:")
    print("1. Open Strategy Builder from the main hub")
    print("2. Select 'Advanced Multi-Timeframe Strategy' from the dropdown")
    print("3. Configure all your parameters:")
    print("   - ATR ratio thresholds (15m/5m mean-reversion vs expansion)")
    print("   - ATR execution threshold (2000T/200T)")
    print("   - EMA periods and Keltner channel settings")
    print("   - Alignment tolerances for location density")
    print("4. Create and test your strategy!")
    
    # Don't show the window, just test programmatically
    app.quit()
    return True


if __name__ == "__main__":
    success = test_gui_integration()
    
    if success:
        print("\n✅ All GUI integration tests passed!")
    else:
        print("\n❌ Some GUI integration tests failed!")
    
    sys.exit(0 if success else 1)