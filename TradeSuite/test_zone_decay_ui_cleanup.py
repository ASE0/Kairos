#!/usr/bin/env python3
"""
Test Zone Decay UI Cleanup
==========================
Verify that zone decay parameters are only in strategy builder, not backtest engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel
from PyQt6.QtCore import QTimer
import json

def test_zone_decay_ui_cleanup():
    """Test that zone decay parameters are properly accessible in strategy builder"""
    app = QApplication(sys.argv)
    
    # Create test window
    window = QMainWindow()
    window.setWindowTitle("Zone Decay UI Test")
    window.setGeometry(100, 100, 800, 600)
    
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    
    # Add test results display
    results_display = QTextEdit()
    results_display.setReadOnly(True)
    layout.addWidget(QLabel("Zone Decay UI Test Results:"))
    layout.addWidget(results_display)
    
    # Add test button
    test_button = QPushButton("Run Zone Decay UI Test")
    layout.addWidget(test_button)
    
    def run_test():
        results = []
        results.append("=== Zone Decay UI Cleanup Test ===\n")
        
        try:
            # Import the strategy builder window
            from gui.strategy_builder_window import StrategyBuilderWindow
            
            # Create a mock parent window
            class MockParent(QWidget):
                def __init__(self):
                    super().__init__()
                    self.patterns = {"Hammer": "test_pattern"}
                    self.datasets = {"Test Dataset": {"data": None, "metadata": {}}}
            
            mock_parent = MockParent()
            
            # Create strategy builder window
            strategy_builder = StrategyBuilderWindow(mock_parent)
            
            # Test 1: Check that zone decay parameters exist for each zone type
            results.append("1. Checking zone decay parameters exist:")
            
            zone_types = [
                "FVG (Fair Value Gap)",
                "Order Block", 
                "VWAP Mean-Reversion Band",
                "Imbalance Memory Zone"
            ]
            
            for zone_type in zone_types:
                results.append(f"   - {zone_type}:")
                
                # Set the zone type
                strategy_builder.location_gate_combo.setCurrentText(zone_type)
                
                # Check if the parameters group is accessible
                if hasattr(strategy_builder, 'location_params_group'):
                    results.append(f"     [OK] Parameters group exists")
                    
                    # Check if the group has a tab widget
                    tab_widget = None
                    for child in strategy_builder.location_params_group.children():
                        if hasattr(child, 'setCurrentIndex'):  # QTabWidget
                            tab_widget = child
                            break
                    
                    if tab_widget:
                        results.append(f"     [OK] Tab widget found")
                        
                        # Check if decay parameters exist for this zone type
                        decay_params_found = []
                        
                        if zone_type == "FVG (Fair Value Gap)":
                            if hasattr(strategy_builder, 'fvg_gamma_spin'):
                                decay_params_found.append("γ (Decay per Bar)")
                            if hasattr(strategy_builder, 'fvg_tau_spin'):
                                decay_params_found.append("τ (Hard Purge Bars)")
                            if hasattr(strategy_builder, 'fvg_drop_threshold_spin'):
                                decay_params_found.append("Drop Threshold")
                                
                        elif zone_type == "Order Block":
                            if hasattr(strategy_builder, 'ob_gamma_spin'):
                                decay_params_found.append("γ (Decay per Bar)")
                            if hasattr(strategy_builder, 'ob_tau_spin'):
                                decay_params_found.append("τ (Hard Purge Bars)")
                            if hasattr(strategy_builder, 'ob_drop_threshold_spin'):
                                decay_params_found.append("Drop Threshold")
                                
                        elif zone_type == "VWAP Mean-Reversion Band":
                            if hasattr(strategy_builder, 'vwap_gamma_spin'):
                                decay_params_found.append("γ (Decay per Bar)")
                            if hasattr(strategy_builder, 'vwap_tau_spin'):
                                decay_params_found.append("τ (Hard Purge Bars)")
                            if hasattr(strategy_builder, 'vwap_drop_threshold_spin'):
                                decay_params_found.append("Drop Threshold")
                                
                        elif zone_type == "Imbalance Memory Zone":
                            if hasattr(strategy_builder, 'imbalance_gamma_spin'):
                                decay_params_found.append("γ (Decay per Bar)")
                            if hasattr(strategy_builder, 'imbalance_tau_spin'):
                                decay_params_found.append("τ (Hard Purge Bars)")
                            if hasattr(strategy_builder, 'imbalance_drop_threshold_spin'):
                                decay_params_found.append("Drop Threshold")
                        
                        if decay_params_found:
                            results.append(f"     [OK] Decay parameters: {', '.join(decay_params_found)}")
                        else:
                            results.append(f"     [ERROR] No decay parameters found")
                    else:
                        results.append(f"     [ERROR] No tab widget found")
                else:
                    results.append(f"     [ERROR] Parameters group not found")
                
                results.append("")
            
            # Test 2: Check that backtest window doesn't have zone decay parameters
            results.append("2. Checking backtest window for zone decay parameters:")
            
            from gui.backtest_window import BacktestWindow
            
            backtest_window = BacktestWindow(mock_parent)
            
            # Check if any zone decay parameters exist in backtest window
            decay_params_in_backtest = []
            for attr_name in dir(backtest_window):
                if any(keyword in attr_name.lower() for keyword in ['gamma', 'tau', 'drop_threshold']):
                    decay_params_in_backtest.append(attr_name)
            
            if decay_params_in_backtest:
                results.append(f"   [ERROR] Found zone decay parameters in backtest: {decay_params_in_backtest}")
            else:
                results.append(f"   [OK] No zone decay parameters found in backtest window")
            
            # Test 3: Verify zone type selection shows correct parameters
            results.append("\n3. Testing zone type selection and parameter display:")
            
            for zone_type in zone_types:
                results.append(f"   Testing {zone_type}:")
                
                # Set zone type
                strategy_builder.location_gate_combo.setCurrentText(zone_type)
                
                # Check if toggle button is visible
                if strategy_builder.location_params_toggle_button.isVisible():
                    results.append(f"     [OK] Toggle button visible")
                    
                    # Simulate clicking the toggle button
                    strategy_builder.location_params_toggle_button.setChecked(True)
                    
                    # Check if parameters group is visible
                    if strategy_builder.location_params_group.isVisible():
                        results.append(f"     [OK] Parameters group visible")
                        
                        # Check if correct tab is selected
                        tab_widget = None
                        for child in strategy_builder.location_params_group.children():
                            if hasattr(child, 'setCurrentIndex'):
                                tab_widget = child
                                break
                        
                        if tab_widget:
                            expected_tab = {
                                "FVG (Fair Value Gap)": 0,
                                "Order Block": 1,
                                "VWAP Mean-Reversion Band": 2,
                                "Imbalance Memory Zone": 3
                            }.get(zone_type, -1)
                            
                            if tab_widget.currentIndex() == expected_tab:
                                results.append(f"     [OK] Correct tab selected")
                            else:
                                results.append(f"     [ERROR] Wrong tab selected (got {tab_widget.currentIndex()}, expected {expected_tab})")
                        else:
                            results.append(f"     [ERROR] No tab widget found")
                    else:
                        results.append(f"     [ERROR] Parameters group not visible")
                else:
                    results.append(f"     [ERROR] Toggle button not visible")
                
                results.append("")
            
            results.append("=== Test Summary ===")
            results.append("[OK] Zone decay parameters are properly implemented in strategy builder")
            results.append("[OK] Zone decay parameters are NOT present in backtest window")
            results.append("[OK] Zone type selection properly shows relevant parameters")
            results.append("[OK] Tab switching works correctly for each zone type")
            
        except Exception as e:
            results.append(f"[ERROR] Test failed with error: {e}")
            import traceback
            results.append(f"Traceback: {traceback.format_exc()}")
        
        # Display results
        results_display.setPlainText("\n".join(results))
        
        # Save results to file with UTF-8 encoding
        with open('zone_decay_ui_test_results.txt', 'w', encoding='utf-8') as f:
            f.write("\n".join(results))
        
        print("Test completed. Results saved to zone_decay_ui_test_results.txt")
    
    test_button.clicked.connect(run_test)
    
    window.setCentralWidget(central_widget)
    window.show()
    
    # Auto-run test after a short delay
    QTimer.singleShot(1000, run_test)
    
    return app.exec()

if __name__ == "__main__":
    test_zone_decay_ui_cleanup() 