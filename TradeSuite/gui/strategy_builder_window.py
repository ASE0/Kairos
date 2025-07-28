"""
gui/strategy_builder_window.py
==============================
Window for building trading strategies
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from copy import deepcopy

from core.data_structures import BaseStrategy, ProbabilityMetrics
from strategies.strategy_builders import Action, PatternStrategy, StrategyFactory, BacktestEngine
from patterns.candlestick_patterns import CandlestickPattern
from core.pattern_registry import registry

DEFAULT_LOCATION_PARAMS = {
    # Candlestick parameters
    "sigma_b": 0.05,      # Body sensitivity [0.01, 0.1]
    "sigma_w": 0.10,      # Wick symmetry [0.05, 0.2]
    
    # Impulse parameters
    "gamma": 2.0,         # Range ratio exponent [1.0, 3.0]
    "delta": 1.5,         # Wick-to-body exponent [0.5, 2.0]
    "epsilon": 1e-4,      # Small constant
    
    # Location parameters
    "beta1": 0.7,         # Base weight [0.6, 0.8]
    "beta2": 0.3,         # Comb weight [0.2, 0.4]
    "N": 3,               # Peak count [1, 10]
    "sigma": 0.1,         # Peak width [0.01, 0.5]
    "lambda_skew": 0.0,   # Skew parameter [-2, 2]
    
    # Momentum parameters
    "kappa_m": 0.5,       # Momentum factor [0, 2]
    "phi": 0.2,           # Expansion factor [0, 0.5]
    
    # Volatility parameters
    "kappa_v": 0.5,       # Volatility factor [0.1, 2.0]
    
    # Execution parameters
    "gate_threshold": 0.7,  # Execution threshold [50, 100]
    "lookback": 100,        # Lookback period
    
    # Legacy parameters (kept for compatibility)
    "sr_window": 20, "sr_threshold": 0.015, "lambda_skew": 0.0,
    "gamma_z": 1.0, "delta_y": 0.0, "omega_mem": 1.0,
    "kernel_xi": 0.5, "kernel_alpha": 2.0, "comb_N": 3,
    "comb_beta1": 0.4, "comb_beta2": 0.6, "impulse_gamma": 2.0,
    "impulse_delta": 1.0, "gate_threshold": 0.4,
}

class StrategyBuilderWindow(QMainWindow):
    """Window for building trading strategies"""

    # Signals
    strategy_created = pyqtSignal(object)  # BaseStrategy

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Strategy Builder")
        self.setGeometry(300, 300, 1000, 800)

        # Get available patterns and datasets
        self.available_patterns = parent.patterns if parent else {}
        self.available_datasets = parent.datasets if parent else {}

        # Current strategy components
        self.actions = []
        self.current_strategy = None

        # Setup UI
        self._setup_ui()

        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #3c3c3c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
                background-color: #3c3c3c;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                padding: 8px 12px;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border: 1px solid #777777;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
                border: 1px solid #444444;
            }
            QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 5px;
                color: #ffffff;
                border-radius: 3px;
            }
            QTextEdit:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #4a4a4a;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
            QListWidget, QTableWidget {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #ffffff;
                gridline-color: #555555;
                alternate-background-color: #2a2a2a;
            }
            QListWidget::item:selected, QTableWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QListWidget::item:hover, QTableWidget::item:hover {
                background-color: #4a4a4a;
            }
            QHeaderView::section {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                color: #ffffff;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #3c3c3c;
            }
            QTabBar::tab {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                padding: 8px 12px;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                border-bottom: 2px solid #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #5a5a5a;
            }
            QScrollBar:vertical {
                background-color: #3c3c3c;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #3c3c3c;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #666666;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #777777;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #0078d4;
            }
            QCheckBox::indicator:checked::after {
                content: "✓";
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
            }
            QRadioButton {
                color: #ffffff;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555555;
                border-radius: 8px;
                background-color: #3c3c3c;
            }
            QRadioButton::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #0078d4;
            }
            QRadioButton::indicator:checked::after {
                content: "";
                width: 6px;
                height: 6px;
                border-radius: 3px;
                background-color: #ffffff;
                margin: 4px;
            }
            QLabel {
                color: #ffffff;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-bottom: 1px solid #555555;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
            }
            QMenuBar::item:selected {
                background-color: #4a4a4a;
            }
            QMenu {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #ffffff;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
            QToolBar {
                background-color: #3c3c3c;
                border: none;
            }
            QToolButton {
                color: #0078d4;
                font-weight: bold;
            }
            QToolButton:hover {
                color: #4ba3e6;
            }
        """)

    def _setup_ui(self):
        """Setup the main UI components"""
        # --- Main Layout ---
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # --- Left Panel: Action Builder ---
        left_panel = self._create_action_builder()
        main_layout.addWidget(left_panel, 1)

        # --- Right Panel: Strategy and Testing ---
        right_layout = QVBoxLayout()
        strategy_panel = self._create_strategy_panel()
        testing_panel = self._create_testing_panel()
        right_layout.addWidget(strategy_panel, 1)
        right_layout.addWidget(testing_panel, 2)
        main_layout.addLayout(right_layout, 2)
        
        self.setCentralWidget(main_widget)

    def _create_action_builder(self) -> QWidget:
        """Create the action builder panel"""
        action_group = QGroupBox("Action Builder")
        layout = QVBoxLayout()

        form_layout = QGridLayout()
        
        form_layout.addWidget(QLabel("Action Name:"), 0, 0)
        self.action_name_edit = QLineEdit()
        self.action_name_edit.setPlaceholderText("e.g., 'Buy on Hammer'")
        form_layout.addWidget(self.action_name_edit, 0, 1, 1, 2)
        
        form_layout.addWidget(QLabel("Pattern:"), 1, 0)
        # Remove FVG from available_patterns for the pattern dropdown
        filtered_patterns = [k for k in self.available_patterns.keys() if k.lower() not in ["fvg", "fvg (fair value gap)"]]
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Location Only"] + filtered_patterns)
        form_layout.addWidget(self.pattern_combo, 1, 1, 1, 2)

        form_layout.addWidget(QLabel("Zone Gate:"), 2, 0)
        self.location_gate_combo = QComboBox()
        self.location_gate_combo.addItems([
            "None", "FVG (Fair Value Gap)", "VWAP Mean-Reversion Band", "Support/Resistance Band", "Imbalance Memory Zone", "Order Block"
        ])
        form_layout.addWidget(self.location_gate_combo, 2, 1)

        # --- Location Gate Parameters ---
        self.location_params_toggle_button = QToolButton()
        self.location_params_toggle_button.setText("Edit Parameters ▼")
        self.location_params_toggle_button.setCheckable(True)
        self.location_params_toggle_button.setChecked(False)
        self.location_params_toggle_button.setStyleSheet("QToolButton { border: none; }")
        form_layout.addWidget(self.location_params_toggle_button, 2, 2)
        
        layout.addLayout(form_layout)
        
        # Create and add the location params group to the layout
        self.location_params_group = self._create_location_params_group()
        self.location_params_group.setVisible(False)
        layout.addWidget(self.location_params_group)
        
        # Connect the toggle button to show/hide parameters
        self.location_params_toggle_button.toggled.connect(self._toggle_params_visibility)
        
        # Connect the zone type combo box to update the tab and button visibility
        self.location_gate_combo.currentTextChanged.connect(self._on_zone_type_changed)
        
        # --- Indicator Filters ---
        indicator_filters_group = self._create_indicator_filters_group()
        layout.addWidget(indicator_filters_group)
        
        layout.addWidget(QLabel("Time Range for Action:"))
        time_range_layout = QHBoxLayout()
        self.time_range_value = QSpinBox()
        self.time_range_value.setRange(1, 1000)
        self.time_range_unit = QComboBox()
        self.time_range_unit.addItems(['minutes', 'hours', 'days'])
        time_range_layout.addWidget(self.time_range_value)
        time_range_layout.addWidget(self.time_range_unit)
        layout.addLayout(time_range_layout)
        
        self.add_action_button = QPushButton("Add Action")
        self.add_action_button.clicked.connect(self._add_action)
        layout.addWidget(self.add_action_button)
        
        layout.addStretch()
        
        # --- Actions List ---
        layout.addWidget(QLabel("Current Actions:"))
        self.actions_list = QListWidget()
        layout.addWidget(self.actions_list)
        
        button_layout = QHBoxLayout()
        self.remove_action_button = QPushButton("Remove Selected Action")
        self.remove_action_button.clicked.connect(self._remove_action)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._clear_actions)
        button_layout.addWidget(self.remove_action_button)
        button_layout.addWidget(clear_button)
        layout.addLayout(button_layout)
        
        action_group.setLayout(layout)
        self._toggle_location_params_button("None") # Set initial state

        # Connect signal for when the selected action changes
        self.actions_list.currentItemChanged.connect(self._display_action_details)

        return action_group

    def _toggle_location_params_button(self, text):
        """Show/hide the location params button based on combo box selection."""
        is_zone_selected = (text != "None")
        self.location_params_toggle_button.setVisible(is_zone_selected)
        if not is_zone_selected:
            # Also hide the group and uncheck the button if the gate is disabled
            self.location_params_group.setVisible(False)
            self.location_params_toggle_button.setChecked(False)

    def _toggle_params_visibility(self, checked):
        """Toggle the visibility of the parameters group box."""
        self.location_params_group.setVisible(checked)
        self.location_params_toggle_button.setText("Edit Parameters ▼" if not checked else "Edit Parameters ▲")
        
        # If showing parameters, switch to the correct tab for the selected zone type
        if checked:
            current_zone_type = self.location_gate_combo.currentText()
            self._switch_to_zone_tab(current_zone_type)

    def _on_zone_type_changed(self, zone_type):
        """Handle zone type selection change"""
        # Update button visibility
        self._toggle_location_params_button(zone_type)
        
        # If parameters are visible, switch to the relevant tab
        if self.location_params_group.isVisible():
            self._switch_to_zone_tab(zone_type)
    
    def _switch_to_zone_tab(self, zone_type):
        """Switch to the appropriate tab based on zone type"""
        tab_mapping = {
            "FVG (Fair Value Gap)": 0,  # FVG tab
            "VWAP Mean-Reversion Band": 1,  # VWAP tab
            "Support/Resistance Band": 2,  # Support/Resistance tab
            "Imbalance Memory Zone": 3,       # Imbalance tab
            "Order Block": 4                  # Order Block tab
        }
        
        if zone_type in tab_mapping:
            # Get the tab widget from the location params group
            for child in self.location_params_group.children():
                if isinstance(child, QTabWidget):
                    child.setCurrentIndex(tab_mapping[zone_type])
                    break

    def _create_indicator_filters_group(self) -> QGroupBox:
        """Creates the group box for indicator-based filters."""
        group = QGroupBox("Indicator Filters")
        layout = QVBoxLayout()

        # Basic filters section
        basic_filters_label = QLabel("Basic Filters:")
        basic_filters_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        layout.addWidget(basic_filters_label)

        # --- MA Filter ---
        self.ma_filter_check = QCheckBox("Moving Average")
        self.ma_filter_widget = QWidget()
        ma_layout = QHBoxLayout(self.ma_filter_widget)
        self.ma_period_spin = QSpinBox()
        self.ma_period_spin.setRange(1, 200)
        self.ma_period_spin.setValue(20)
        self.ma_condition_combo = QComboBox()
        self.ma_condition_combo.addItems(["above", "below", "near"])
        ma_layout.addWidget(QLabel("Period:"))
        ma_layout.addWidget(self.ma_period_spin)
        ma_layout.addWidget(QLabel("Price is:"))
        ma_layout.addWidget(self.ma_condition_combo)
        ma_layout.setContentsMargins(10, 0, 0, 0)
        self.ma_filter_widget.setVisible(False)
        self.ma_filter_check.toggled.connect(self.ma_filter_widget.setVisible)
        layout.addWidget(self.ma_filter_check)
        layout.addWidget(self.ma_filter_widget)

        # --- VWAP Filter ---
        self.vwap_filter_check = QCheckBox("VWAP")
        self.vwap_filter_widget = QWidget()
        vwap_layout = QHBoxLayout(self.vwap_filter_widget)
        self.vwap_condition_combo = QComboBox()
        self.vwap_condition_combo.addItems(["above", "below", "near"])
        vwap_layout.addWidget(QLabel("Price is:"))
        vwap_layout.addWidget(self.vwap_condition_combo)
        vwap_layout.setContentsMargins(10, 0, 0, 0)
        self.vwap_filter_widget.setVisible(False)
        self.vwap_filter_check.toggled.connect(self.vwap_filter_widget.setVisible)
        layout.addWidget(self.vwap_filter_check)
        layout.addWidget(self.vwap_filter_widget)

        # --- Bollinger Bands Filter ---
        self.bb_filter_check = QCheckBox("Bollinger Bands")
        self.bb_filter_widget = QWidget()
        bb_layout = QHBoxLayout(self.bb_filter_widget)
        self.bb_period_spin = QSpinBox()
        self.bb_period_spin.setRange(1, 200)
        self.bb_period_spin.setValue(20)
        self.bb_std_spin = QDoubleSpinBox()
        self.bb_std_spin.setRange(0.5, 5.0)
        self.bb_std_spin.setValue(2.0)
        self.bb_condition_combo = QComboBox()
        self.bb_condition_combo.addItems(["inside", "outside", "touching_upper", "touching_lower"])
        bb_layout.addWidget(QLabel("Period:"))
        bb_layout.addWidget(self.bb_period_spin)
        bb_layout.addWidget(QLabel("Std Dev:"))
        bb_layout.addWidget(self.bb_std_spin)
        bb_layout.addWidget(QLabel("Price is:"))
        bb_layout.addWidget(self.bb_condition_combo)
        bb_layout.setContentsMargins(10, 0, 0, 0)
        self.bb_filter_widget.setVisible(False)
        self.bb_filter_check.toggled.connect(self.bb_filter_widget.setVisible)
        layout.addWidget(self.bb_filter_check)
        layout.addWidget(self.bb_filter_widget)

        # Microstructure filters section
        microstructure_filters_label = QLabel("Microstructure Filters:")
        microstructure_filters_label.setStyleSheet("font-weight: bold; color: #ffffff; margin-top: 10px;")
        layout.addWidget(microstructure_filters_label)

        # --- Tick Frequency Filter ---
        self.tick_frequency_filter_check = QCheckBox("Tick Frequency Filter")
        self.tick_frequency_filter_widget = QWidget()
        tick_freq_layout = QHBoxLayout(self.tick_frequency_filter_widget)
        self.max_ticks_per_second_spin = QSpinBox()
        self.max_ticks_per_second_spin.setRange(10, 1000)
        self.max_ticks_per_second_spin.setValue(50)
        self.min_book_depth_spin = QSpinBox()
        self.min_book_depth_spin.setRange(10, 1000)
        self.min_book_depth_spin.setValue(100)
        tick_freq_layout.addWidget(QLabel("Max Ticks/sec:"))
        tick_freq_layout.addWidget(self.max_ticks_per_second_spin)
        tick_freq_layout.addWidget(QLabel("Min Book Depth:"))
        tick_freq_layout.addWidget(self.min_book_depth_spin)
        tick_freq_layout.setContentsMargins(10, 0, 0, 0)
        self.tick_frequency_filter_widget.setVisible(False)
        self.tick_frequency_filter_check.toggled.connect(self.tick_frequency_filter_widget.setVisible)
        layout.addWidget(self.tick_frequency_filter_check)
        layout.addWidget(self.tick_frequency_filter_widget)

        # --- Spread Filter ---
        self.spread_filter_check = QCheckBox("Spread Filter")
        self.spread_filter_widget = QWidget()
        spread_layout = QHBoxLayout(self.spread_filter_widget)
        self.max_spread_ticks_spin = QSpinBox()
        self.max_spread_ticks_spin.setRange(1, 10)
        self.max_spread_ticks_spin.setValue(2)
        self.normal_spread_multiple_spin = QDoubleSpinBox()
        self.normal_spread_multiple_spin.setRange(1.0, 10.0)
        self.normal_spread_multiple_spin.setValue(5.0)
        spread_layout.addWidget(QLabel("Max Spread (ticks):"))
        spread_layout.addWidget(self.max_spread_ticks_spin)
        spread_layout.addWidget(QLabel("Normal Multiple:"))
        spread_layout.addWidget(self.normal_spread_multiple_spin)
        spread_layout.setContentsMargins(10, 0, 0, 0)
        self.spread_filter_widget.setVisible(False)
        self.spread_filter_check.toggled.connect(self.spread_filter_widget.setVisible)
        layout.addWidget(self.spread_filter_check)
        layout.addWidget(self.spread_filter_widget)

        # --- Order Flow Filter ---
        self.order_flow_filter_check = QCheckBox("Order Flow Filter")
        self.order_flow_filter_widget = QWidget()
        order_flow_layout = QHBoxLayout(self.order_flow_filter_widget)
        self.min_cvd_threshold_spin = QSpinBox()
        self.min_cvd_threshold_spin.setRange(100, 10000)
        self.min_cvd_threshold_spin.setValue(1000)
        self.large_trade_ratio_spin = QDoubleSpinBox()
        self.large_trade_ratio_spin.setRange(0.1, 1.0)
        self.large_trade_ratio_spin.setValue(0.35)
        self.large_trade_ratio_spin.setSingleStep(0.05)
        order_flow_layout.addWidget(QLabel("Min CVD Threshold:"))
        order_flow_layout.addWidget(self.min_cvd_threshold_spin)
        order_flow_layout.addWidget(QLabel("Large Trade Ratio:"))
        order_flow_layout.addWidget(self.large_trade_ratio_spin)
        order_flow_layout.setContentsMargins(10, 0, 0, 0)
        self.order_flow_filter_widget.setVisible(False)
        self.order_flow_filter_check.toggled.connect(self.order_flow_filter_widget.setVisible)
        layout.addWidget(self.order_flow_filter_check)
        layout.addWidget(self.order_flow_filter_widget)

        # Advanced filters section
        advanced_filters_label = QLabel("Advanced Filters:")
        advanced_filters_label.setStyleSheet("font-weight: bold; color: #ffffff; margin-top: 10px;")
        layout.addWidget(advanced_filters_label)

        # --- Volume Filter ---
        self.volume_filter_check = QCheckBox("Volume Filter")
        self.volume_filter_widget = QWidget()
        volume_layout = QHBoxLayout(self.volume_filter_widget)
        self.min_volume_spin = QSpinBox()
        self.min_volume_spin.setRange(100, 100000)
        self.min_volume_spin.setValue(1000)
        self.volume_ratio_spin = QDoubleSpinBox()
        self.volume_ratio_spin.setRange(1.0, 5.0)
        self.volume_ratio_spin.setValue(1.5)
        volume_layout.addWidget(QLabel("Min Volume:"))
        volume_layout.addWidget(self.min_volume_spin)
        volume_layout.addWidget(QLabel("Volume Ratio:"))
        volume_layout.addWidget(self.volume_ratio_spin)
        volume_layout.setContentsMargins(10, 0, 0, 0)
        self.volume_filter_widget.setVisible(False)
        self.volume_filter_check.toggled.connect(self.volume_filter_widget.setVisible)
        layout.addWidget(self.volume_filter_check)
        layout.addWidget(self.volume_filter_widget)

        # --- Time Filter ---
        self.time_filter_check = QCheckBox("Time Filter")
        self.time_filter_widget = QWidget()
        time_layout = QHBoxLayout(self.time_filter_widget)
        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setTime(QTime(9, 30))
        self.end_time_edit = QTimeEdit()
        self.end_time_edit.setTime(QTime(16, 0))
        time_layout.addWidget(QLabel("Start Time:"))
        time_layout.addWidget(self.start_time_edit)
        time_layout.addWidget(QLabel("End Time:"))
        time_layout.addWidget(self.end_time_edit)
        time_layout.setContentsMargins(10, 0, 0, 0)
        self.time_filter_widget.setVisible(False)
        self.time_filter_check.toggled.connect(self.time_filter_widget.setVisible)
        layout.addWidget(self.time_filter_check)
        layout.addWidget(self.time_filter_widget)

        # --- Volatility Filter ---
        self.volatility_filter_check = QCheckBox("Volatility Filter")
        self.volatility_filter_widget = QWidget()
        volatility_layout = QHBoxLayout(self.volatility_filter_widget)
        self.min_atr_ratio_spin = QDoubleSpinBox()
        self.min_atr_ratio_spin.setRange(0.001, 0.1)
        self.min_atr_ratio_spin.setValue(0.01)
        self.max_atr_ratio_spin = QDoubleSpinBox()
        self.max_atr_ratio_spin.setRange(0.01, 0.2)
        self.max_atr_ratio_spin.setValue(0.05)
        volatility_layout.addWidget(QLabel("Min ATR Ratio:"))
        volatility_layout.addWidget(self.min_atr_ratio_spin)
        volatility_layout.addWidget(QLabel("Max ATR Ratio:"))
        volatility_layout.addWidget(self.max_atr_ratio_spin)
        volatility_layout.setContentsMargins(10, 0, 0, 0)
        self.volatility_filter_widget.setVisible(False)
        self.volatility_filter_check.toggled.connect(self.volatility_filter_widget.setVisible)
        layout.addWidget(self.volatility_filter_check)
        layout.addWidget(self.volatility_filter_widget)

        # --- Momentum Filter ---
        self.momentum_filter_check = QCheckBox("Momentum Filter")
        self.momentum_filter_widget = QWidget()
        momentum_layout = QHBoxLayout(self.momentum_filter_widget)
        self.momentum_threshold_spin = QDoubleSpinBox()
        self.momentum_threshold_spin.setRange(0.001, 0.1)
        self.momentum_threshold_spin.setValue(0.02)
        self.rsi_min_spin = QSpinBox()
        self.rsi_min_spin.setRange(0, 100)
        self.rsi_min_spin.setValue(30)
        self.rsi_max_spin = QSpinBox()
        self.rsi_max_spin.setRange(0, 100)
        self.rsi_max_spin.setValue(70)
        momentum_layout.addWidget(QLabel("Momentum Threshold:"))
        momentum_layout.addWidget(self.momentum_threshold_spin)
        momentum_layout.addWidget(QLabel("RSI Range:"))
        momentum_layout.addWidget(self.rsi_min_spin)
        momentum_layout.addWidget(QLabel("-"))
        momentum_layout.addWidget(self.rsi_max_spin)
        momentum_layout.setContentsMargins(10, 0, 0, 0)
        self.momentum_filter_widget.setVisible(False)
        self.momentum_filter_check.toggled.connect(self.momentum_filter_widget.setVisible)
        layout.addWidget(self.momentum_filter_check)
        layout.addWidget(self.momentum_filter_widget)

        # --- Price Filter ---
        self.price_filter_check = QCheckBox("Price Filter")
        self.price_filter_widget = QWidget()
        price_layout = QHBoxLayout(self.price_filter_widget)
        self.min_price_spin = QDoubleSpinBox()
        self.min_price_spin.setRange(0.01, 1000.0)
        self.min_price_spin.setValue(1.0)
        self.max_price_spin = QDoubleSpinBox()
        self.max_price_spin.setRange(1.0, 10000.0)
        self.max_price_spin.setValue(1000.0)
        price_layout.addWidget(QLabel("Min Price:"))
        price_layout.addWidget(self.min_price_spin)
        price_layout.addWidget(QLabel("Max Price:"))
        price_layout.addWidget(self.max_price_spin)
        price_layout.setContentsMargins(10, 0, 0, 0)
        self.price_filter_widget.setVisible(False)
        self.price_filter_check.toggled.connect(self.price_filter_widget.setVisible)
        layout.addWidget(self.price_filter_check)
        layout.addWidget(self.price_filter_widget)

        group.setLayout(layout)
        return group

    def _create_location_params_group(self) -> QGroupBox:
        """Create location parameters group with zone-specific parameters"""
        group = QGroupBox("Zone Parameters")
        layout = QVBoxLayout()
        
        # Create tabs for different parameter categories
        tabs = QTabWidget()
        
        # FVG parameters tab
        fvg_tab = QWidget()
        fvg_layout = QFormLayout()
        
        # FVG detection parameters
        fvg_layout.addRow(QLabel("Detection Parameters:"))
        fvg_layout.addRow(QLabel(""))  # Spacer
        
        self.fvg_epsilon_spin = QDoubleSpinBox()
        self.fvg_epsilon_spin.setRange(1, 5)
        self.fvg_epsilon_spin.setValue(2)
        self.fvg_epsilon_spin.setSingleStep(0.5)
        self.fvg_epsilon_spin.setToolTip("Zone buffer points [1, 5]")
        fvg_layout.addRow("ε (Buffer Points):", self.fvg_epsilon_spin)
        
        self.fvg_N_spin = QSpinBox()
        self.fvg_N_spin.setRange(1, 10)
        self.fvg_N_spin.setValue(3)
        self.fvg_N_spin.setToolTip("Number of Gaussian peaks [1, 10]")
        fvg_layout.addRow("N (Peak Count):", self.fvg_N_spin)
        
        self.fvg_sigma_spin = QDoubleSpinBox()
        self.fvg_sigma_spin.setRange(0.01, 0.5)
        self.fvg_sigma_spin.setValue(0.1)
        self.fvg_sigma_spin.setSingleStep(0.01)
        self.fvg_sigma_spin.setToolTip("Std-dev of Gaussian peaks [0.01, 0.5]")
        fvg_layout.addRow("σ (Peak Width):", self.fvg_sigma_spin)
        
        self.fvg_beta1_spin = QDoubleSpinBox()
        self.fvg_beta1_spin.setRange(0.6, 0.8)
        self.fvg_beta1_spin.setValue(0.7)
        self.fvg_beta1_spin.setSingleStep(0.05)
        self.fvg_beta1_spin.setToolTip("Flat base weight [0.6, 0.8]")
        fvg_layout.addRow("β₁ (Base Weight):", self.fvg_beta1_spin)
        
        self.fvg_beta2_spin = QDoubleSpinBox()
        self.fvg_beta2_spin.setRange(0.2, 0.4)
        self.fvg_beta2_spin.setValue(0.3)
        self.fvg_beta2_spin.setSingleStep(0.05)
        self.fvg_beta2_spin.setToolTip("Micro-comb weight [0.2, 0.4]")
        fvg_layout.addRow("β₂ (Comb Weight):", self.fvg_beta2_spin)
        
        self.fvg_phi_spin = QDoubleSpinBox()
        self.fvg_phi_spin.setRange(0.0, 0.5)
        self.fvg_phi_spin.setValue(0.2)
        self.fvg_phi_spin.setSingleStep(0.05)
        self.fvg_phi_spin.setToolTip("Momentum warp factor [0, 0.5]")
        fvg_layout.addRow("φ (Momentum Warp):", self.fvg_phi_spin)
        
        self.fvg_lambda_spin = QDoubleSpinBox()
        self.fvg_lambda_spin.setRange(-2.0, 2.0)
        self.fvg_lambda_spin.setValue(0.0)
        self.fvg_lambda_spin.setSingleStep(0.1)
        self.fvg_lambda_spin.setToolTip("Directional skew slope [-2, 2]")
        fvg_layout.addRow("λ (Directional Skew):", self.fvg_lambda_spin)
        
        # FVG decay parameters
        fvg_layout.addRow(QLabel(""))  # Spacer
        fvg_layout.addRow(QLabel("Decay Parameters:"))
        fvg_layout.addRow(QLabel(""))  # Spacer
        
        self.fvg_gamma_spin = QDoubleSpinBox()
        self.fvg_gamma_spin.setRange(0.8, 0.99)
        self.fvg_gamma_spin.setValue(0.95)
        self.fvg_gamma_spin.setSingleStep(0.01)
        self.fvg_gamma_spin.setToolTip("Exponential decay per bar [0.8, 0.99]")
        fvg_layout.addRow("γ (Decay per Bar):", self.fvg_gamma_spin)
        
        self.fvg_tau_spin = QSpinBox()
        self.fvg_tau_spin.setRange(5, 200)
        self.fvg_tau_spin.setValue(50)
        self.fvg_tau_spin.setToolTip("Hard purge after τ bars [5, 200]")
        fvg_layout.addRow("τ (Hard Purge Bars):", self.fvg_tau_spin)
        
        self.fvg_drop_threshold_spin = QDoubleSpinBox()
        self.fvg_drop_threshold_spin.setRange(0.001, 0.1)
        self.fvg_drop_threshold_spin.setValue(0.01)
        self.fvg_drop_threshold_spin.setSingleStep(0.001)
        self.fvg_drop_threshold_spin.setToolTip("Minimum strength before early purge [0.001, 0.1]")
        fvg_layout.addRow("Drop Threshold:", self.fvg_drop_threshold_spin)
        
        fvg_tab.setLayout(fvg_layout)
        tabs.addTab(fvg_tab, "FVG")
        
        # VWAP parameters tab
        vwap_tab = QWidget()
        vwap_layout = QFormLayout()
        
        # VWAP detection parameters
        vwap_layout.addRow(QLabel("Detection Parameters:"))
        vwap_layout.addRow(QLabel(""))  # Spacer
        
        self.vwap_k_spin = QDoubleSpinBox()
        self.vwap_k_spin.setRange(0.5, 2.0)
        self.vwap_k_spin.setValue(1.0)
        self.vwap_k_spin.setSingleStep(0.1)
        self.vwap_k_spin.setToolTip("Multiplier for VWAP stdev band [0.5, 2.0]")
        vwap_layout.addRow("k (Stdev Multiplier):", self.vwap_k_spin)
        
        self.vwap_lookback_spin = QSpinBox()
        self.vwap_lookback_spin.setRange(10, 50)
        self.vwap_lookback_spin.setValue(20)
        self.vwap_lookback_spin.setToolTip("VWAP calculation lookback [10, 50]")
        vwap_layout.addRow("VWAP Lookback:", self.vwap_lookback_spin)
        
        # VWAP decay parameters
        vwap_layout.addRow(QLabel(""))  # Spacer
        vwap_layout.addRow(QLabel("Decay Parameters:"))
        vwap_layout.addRow(QLabel(""))  # Spacer
        
        self.vwap_gamma_spin = QDoubleSpinBox()
        self.vwap_gamma_spin.setRange(0.8, 0.99)
        self.vwap_gamma_spin.setValue(0.95)
        self.vwap_gamma_spin.setSingleStep(0.01)
        self.vwap_gamma_spin.setToolTip("Exponential decay per bar [0.8, 0.99]")
        vwap_layout.addRow("γ (Decay per Bar):", self.vwap_gamma_spin)
        
        self.vwap_tau_spin = QSpinBox()
        self.vwap_tau_spin.setRange(5, 200)
        self.vwap_tau_spin.setValue(15)  # VWAP zones are shorter-lived
        self.vwap_tau_spin.setToolTip("Hard purge after τ bars [5, 200]")
        vwap_layout.addRow("τ (Hard Purge Bars):", self.vwap_tau_spin)
        
        self.vwap_drop_threshold_spin = QDoubleSpinBox()
        self.vwap_drop_threshold_spin.setRange(0.001, 0.1)
        self.vwap_drop_threshold_spin.setValue(0.01)
        self.vwap_drop_threshold_spin.setSingleStep(0.001)
        self.vwap_drop_threshold_spin.setToolTip("Minimum strength before early purge [0.001, 0.1]")
        vwap_layout.addRow("Drop Threshold:", self.vwap_drop_threshold_spin)
        
        vwap_tab.setLayout(vwap_layout)
        tabs.addTab(vwap_tab, "VWAP")
        
        # Support/Resistance parameters tab
        sr_tab = QWidget()
        sr_layout = QFormLayout()
        
        # Support/Resistance detection parameters
        sr_layout.addRow(QLabel("Detection Parameters:"))
        sr_layout.addRow(QLabel(""))  # Spacer
        
        self.sr_window_spin = QSpinBox()
        self.sr_window_spin.setRange(10, 100)
        self.sr_window_spin.setValue(20)
        self.sr_window_spin.setSingleStep(5)
        self.sr_window_spin.setToolTip("Window size for support/resistance detection [10, 100]")
        sr_layout.addRow("Window Size:", self.sr_window_spin)
        
        self.sr_buffer_pts_spin = QDoubleSpinBox()
        self.sr_buffer_pts_spin.setRange(0.5, 10.0)
        self.sr_buffer_pts_spin.setValue(2.0)
        self.sr_buffer_pts_spin.setSingleStep(0.5)
        self.sr_buffer_pts_spin.setToolTip("Buffer points for zone edges [0.5, 10.0]")
        sr_layout.addRow("Buffer Points:", self.sr_buffer_pts_spin)
        
        self.sr_sigma_r_spin = QDoubleSpinBox()
        self.sr_sigma_r_spin.setRange(1.0, 20.0)
        self.sr_sigma_r_spin.setValue(5.0)
        self.sr_sigma_r_spin.setSingleStep(0.5)
        self.sr_sigma_r_spin.setToolTip("Spatial standard deviation [1.0, 20.0]")
        sr_layout.addRow("σ_r (Spatial):", self.sr_sigma_r_spin)
        
        self.sr_sigma_t_spin = QDoubleSpinBox()
        self.sr_sigma_t_spin.setRange(1.0, 10.0)
        self.sr_sigma_t_spin.setValue(3.0)
        self.sr_sigma_t_spin.setSingleStep(0.5)
        self.sr_sigma_t_spin.setToolTip("Temporal standard deviation [1.0, 10.0]")
        sr_layout.addRow("σ_t (Temporal):", self.sr_sigma_t_spin)
        
        # Support/Resistance decay parameters
        sr_layout.addRow(QLabel(""))  # Spacer
        sr_layout.addRow(QLabel("Decay Parameters:"))
        sr_layout.addRow(QLabel(""))  # Spacer
        
        self.sr_gamma_spin = QDoubleSpinBox()
        self.sr_gamma_spin.setRange(0.8, 0.99)
        self.sr_gamma_spin.setValue(0.95)
        self.sr_gamma_spin.setSingleStep(0.01)
        self.sr_gamma_spin.setToolTip("Exponential decay per bar [0.8, 0.99]")
        sr_layout.addRow("γ (Decay per Bar):", self.sr_gamma_spin)
        
        self.sr_tau_spin = QSpinBox()
        self.sr_tau_spin.setRange(5, 200)
        self.sr_tau_spin.setValue(60)  # Support/Resistance zones are medium-lived
        self.sr_tau_spin.setToolTip("Hard purge after τ bars [5, 200]")
        sr_layout.addRow("τ (Hard Purge Bars):", self.sr_tau_spin)
        
        self.sr_drop_threshold_spin = QDoubleSpinBox()
        self.sr_drop_threshold_spin.setRange(0.001, 0.1)
        self.sr_drop_threshold_spin.setValue(0.01)
        self.sr_drop_threshold_spin.setSingleStep(0.001)
        self.sr_drop_threshold_spin.setToolTip("Minimum strength before early purge [0.001, 0.1]")
        sr_layout.addRow("Drop Threshold:", self.sr_drop_threshold_spin)
        
        sr_tab.setLayout(sr_layout)
        tabs.addTab(sr_tab, "Support/Resistance")
        
        # Imbalance parameters tab
        imbalance_tab = QWidget()
        imbalance_layout = QFormLayout()
        
        # Imbalance detection parameters
        imbalance_layout.addRow(QLabel("Detection Parameters:"))
        imbalance_layout.addRow(QLabel(""))  # Spacer
        
        self.imbalance_threshold_spin = QDoubleSpinBox()
        self.imbalance_threshold_spin.setRange(10, 500)
        self.imbalance_threshold_spin.setValue(100)
        self.imbalance_threshold_spin.setSingleStep(10)
        self.imbalance_threshold_spin.setToolTip("Threshold points to register imbalance [10, 500]")
        imbalance_layout.addRow("τ_imbalance (Threshold):", self.imbalance_threshold_spin)
        
        self.imbalance_gamma_mem_spin = QDoubleSpinBox()
        self.imbalance_gamma_mem_spin.setRange(0.001, 0.1)
        self.imbalance_gamma_mem_spin.setValue(0.01)
        self.imbalance_gamma_mem_spin.setSingleStep(0.001)
        self.imbalance_gamma_mem_spin.setToolTip("Decay factor for imbalance memory [0.001, 0.1]")
        imbalance_layout.addRow("γ_mem (Memory Decay):", self.imbalance_gamma_mem_spin)
        
        self.imbalance_sigma_rev_spin = QDoubleSpinBox()
        self.imbalance_sigma_rev_spin.setRange(5, 50)
        self.imbalance_sigma_rev_spin.setValue(20)
        self.imbalance_sigma_rev_spin.setSingleStep(1)
        self.imbalance_sigma_rev_spin.setToolTip("Width of Gaussian influence for revisit [5, 50 pts]")
        imbalance_layout.addRow("σ_rev (Revisit Width):", self.imbalance_sigma_rev_spin)
        
        # Imbalance decay parameters
        imbalance_layout.addRow(QLabel(""))  # Spacer
        imbalance_layout.addRow(QLabel("Decay Parameters:"))
        imbalance_layout.addRow(QLabel(""))  # Spacer
        
        self.imbalance_gamma_spin = QDoubleSpinBox()
        self.imbalance_gamma_spin.setRange(0.8, 0.99)
        self.imbalance_gamma_spin.setValue(0.95)
        self.imbalance_gamma_spin.setSingleStep(0.01)
        self.imbalance_gamma_spin.setToolTip("Exponential decay per bar [0.8, 0.99]")
        imbalance_layout.addRow("γ (Decay per Bar):", self.imbalance_gamma_spin)
        
        self.imbalance_tau_spin = QSpinBox()
        self.imbalance_tau_spin.setRange(5, 200)
        self.imbalance_tau_spin.setValue(100)  # Imbalance zones can last longer
        self.imbalance_tau_spin.setToolTip("Hard purge after τ bars [5, 200]")
        imbalance_layout.addRow("τ (Hard Purge Bars):", self.imbalance_tau_spin)
        
        self.imbalance_drop_threshold_spin = QDoubleSpinBox()
        self.imbalance_drop_threshold_spin.setRange(0.001, 0.1)
        self.imbalance_drop_threshold_spin.setValue(0.01)
        self.imbalance_drop_threshold_spin.setSingleStep(0.001)
        self.imbalance_drop_threshold_spin.setToolTip("Minimum strength before early purge [0.001, 0.1]")
        imbalance_layout.addRow("Drop Threshold:", self.imbalance_drop_threshold_spin)
        
        imbalance_tab.setLayout(imbalance_layout)
        tabs.addTab(imbalance_tab, "Imbalance")
        
        # Order Block tab
        ob_tab = QWidget()
        ob_layout = QFormLayout()
        ob_layout.addRow(QLabel("Detection Parameters:"))
        ob_layout.addRow(QLabel(""))  # Spacer
        self.ob_impulse_threshold_spin = QDoubleSpinBox()
        self.ob_impulse_threshold_spin.setRange(0.01, 0.1)
        self.ob_impulse_threshold_spin.setValue(0.02)
        self.ob_impulse_threshold_spin.setSingleStep(0.005)
        self.ob_impulse_threshold_spin.setToolTip("Minimum impulse move [0.01, 0.1]")
        ob_layout.addRow("Impulse Threshold:", self.ob_impulse_threshold_spin)
        self.ob_lookback_spin = QSpinBox()
        self.ob_lookback_spin.setRange(5, 50)
        self.ob_lookback_spin.setValue(10)
        self.ob_lookback_spin.setToolTip("Lookback for impulse detection [5, 50]")
        ob_layout.addRow("Lookback Period:", self.ob_lookback_spin)
        ob_layout.addRow(QLabel(""))  # Spacer
        ob_layout.addRow(QLabel("Decay Parameters:"))
        ob_layout.addRow(QLabel(""))  # Spacer
        self.ob_gamma_spin = QDoubleSpinBox()
        self.ob_gamma_spin.setRange(0.8, 0.99)
        self.ob_gamma_spin.setValue(0.95)
        self.ob_gamma_spin.setSingleStep(0.01)
        self.ob_gamma_spin.setToolTip("Exponential decay per bar [0.8, 0.99]")
        ob_layout.addRow("γ (Decay per Bar):", self.ob_gamma_spin)
        self.ob_tau_spin = QSpinBox()
        self.ob_tau_spin.setRange(5, 200)
        self.ob_tau_spin.setValue(50)
        self.ob_tau_spin.setToolTip("Hard purge after τ bars [5, 200]")
        ob_layout.addRow("τ (Hard Purge Bars):", self.ob_tau_spin)
        self.ob_buffer_spin = QDoubleSpinBox()
        self.ob_buffer_spin.setRange(0.01, 1.0)
        self.ob_buffer_spin.setSingleStep(0.01)
        self.ob_buffer_spin.setValue(0.1)
        ob_layout.addRow("Buffer (pts)", self.ob_buffer_spin)
        ob_tab.setLayout(ob_layout)
        tabs.addTab(ob_tab, "Order Block")
        
        # Global settings tab
        global_tab = QWidget()
        global_layout = QFormLayout()
        
        global_layout.addRow(QLabel("Global Settings:"))
        global_layout.addRow(QLabel(""))  # Spacer
        
        self.bar_interval_minutes_spin = QSpinBox()
        self.bar_interval_minutes_spin.setRange(1, 1440)
        self.bar_interval_minutes_spin.setValue(1)
        self.bar_interval_minutes_spin.setToolTip("Minutes per bar for calendar conversion")
        global_layout.addRow("Bar Interval (min):", self.bar_interval_minutes_spin)
        
        global_tab.setLayout(global_layout)
        tabs.addTab(global_tab, "Global")
        
        layout.addWidget(tabs)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_location_params)
        layout.addWidget(reset_btn)

        group.setLayout(layout)
        return group

    def _reset_location_params(self):
        """Reset all parameters to spec-compliant defaults"""
        self.fvg_epsilon_spin.setValue(2)
        self.fvg_N_spin.setValue(3)
        self.fvg_sigma_spin.setValue(0.1)
        self.fvg_beta1_spin.setValue(0.7)
        self.fvg_beta2_spin.setValue(0.3)
        self.fvg_phi_spin.setValue(0.2)
        self.fvg_lambda_spin.setValue(0.0)
        self.fvg_gamma_spin.setValue(0.95)
        self.fvg_tau_spin.setValue(50)
        self.fvg_drop_threshold_spin.setValue(0.01)
        self.vwap_k_spin.setValue(1.0)
        self.vwap_lookback_spin.setValue(20)
        self.vwap_gamma_spin.setValue(0.95)
        self.vwap_tau_spin.setValue(15)
        self.vwap_drop_threshold_spin.setValue(0.01)
        self.sr_window_spin.setValue(20)
        self.sr_buffer_pts_spin.setValue(2.0)
        self.sr_sigma_r_spin.setValue(5.0)
        self.sr_sigma_t_spin.setValue(3.0)
        self.sr_gamma_spin.setValue(0.95)
        self.sr_tau_spin.setValue(60)
        self.sr_drop_threshold_spin.setValue(0.01)
        self.imbalance_threshold_spin.setValue(100)
        self.imbalance_gamma_mem_spin.setValue(0.01)
        self.imbalance_sigma_rev_spin.setValue(20)
        self.imbalance_gamma_spin.setValue(0.95)
        self.imbalance_tau_spin.setValue(100)
        self.imbalance_drop_threshold_spin.setValue(0.01)
        self.ob_impulse_threshold_spin.setValue(0.02)
        self.ob_lookback_spin.setValue(10)
        self.ob_gamma_spin.setValue(0.95)
        self.ob_tau_spin.setValue(50)
        self.ob_buffer_spin.setValue(2.0)
        self.bar_interval_minutes_spin.setValue(1)

    def _create_strategy_panel(self) -> QWidget:
        """Create the strategy management panel"""
        strategy_group = QGroupBox("Strategy Configuration")
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.strategy_name_edit = QLineEdit()
        self.strategy_name_edit.setPlaceholderText("e.g., 'My Hammer Strategy'")
        form_layout.addRow("Strategy Name:", self.strategy_name_edit)
        
        self.combination_logic_combo = QComboBox()
        self.combination_logic_combo.addItems(['AND', 'OR'])
        form_layout.addRow("Combination Logic:", self.combination_logic_combo)
        
        layout.addLayout(form_layout)

        # --- Execution Gates ---
        self.gates_toggle_button = QToolButton()
        self.gates_toggle_button.setText("Execution Gates ▼")
        self.gates_toggle_button.setCheckable(True)
        self.gates_toggle_button.setChecked(False)
        self.gates_toggle_button.setStyleSheet("QToolButton { border: none; }")
        
        self.gates_group = self._create_gates_group()
        self.gates_group.setVisible(False)
        self.gates_toggle_button.toggled.connect(self.gates_group.setVisible)

        layout.addWidget(self.gates_toggle_button)
        layout.addWidget(self.gates_group)
        
        self.create_strategy_button = QPushButton("Create Strategy")
        self.create_strategy_button.clicked.connect(self._create_strategy)
        self.duplicate_strategy_button = QPushButton("Duplicate Strategy")
        self.duplicate_strategy_button.clicked.connect(self._duplicate_strategy)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.create_strategy_button)
        button_layout.addWidget(self.duplicate_strategy_button)
        layout.addLayout(button_layout)
        
        strategy_group.setLayout(layout)
        return strategy_group

    def _create_gates_group(self) -> QGroupBox:
        """Creates the group box for execution gate toggles."""
        group = QGroupBox("Gates")
        layout = QVBoxLayout()
        
        # Basic gates
        basic_gates_label = QLabel("Basic Gates:")
        basic_gates_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        layout.addWidget(basic_gates_label)
        
        self.volatility_gate_check = QCheckBox("Volatility Gate")
        self.regime_gate_check = QCheckBox("Regime Gate")
        self.bayesian_gate_check = QCheckBox("Bayesian State Gate")
        
        layout.addWidget(self.volatility_gate_check)
        layout.addWidget(self.regime_gate_check)
        layout.addWidget(self.bayesian_gate_check)
        
        # Microstructure gates
        microstructure_gates_label = QLabel("Microstructure Gates:")
        microstructure_gates_label.setStyleSheet("font-weight: bold; color: #ffffff; margin-top: 10px;")
        layout.addWidget(microstructure_gates_label)
        
        self.market_environment_gate_check = QCheckBox("Market Environment Gate")
        self.news_time_gate_check = QCheckBox("News Time Gate")
        self.tick_validation_gate_check = QCheckBox("Tick Validation Gate")
        
        layout.addWidget(self.market_environment_gate_check)
        layout.addWidget(self.news_time_gate_check)
        layout.addWidget(self.tick_validation_gate_check)
        
        # Advanced gates
        advanced_gates_label = QLabel("Advanced Gates:")
        advanced_gates_label.setStyleSheet("font-weight: bold; color: #ffffff; margin-top: 10px;")
        layout.addWidget(advanced_gates_label)
        
        self.fvg_gate_check = QCheckBox("FVG Gate")
        self.momentum_gate_check = QCheckBox("Momentum Gate")
        self.volume_gate_check = QCheckBox("Volume Gate")
        self.time_gate_check = QCheckBox("Time Gate")
        self.correlation_gate_check = QCheckBox("Correlation Gate")
        self.order_block_gate_check = QCheckBox("Order Block Gate")
        
        layout.addWidget(self.fvg_gate_check)
        layout.addWidget(self.momentum_gate_check)
        layout.addWidget(self.volume_gate_check)
        layout.addWidget(self.time_gate_check)
        layout.addWidget(self.correlation_gate_check)
        layout.addWidget(self.order_block_gate_check)

        group.setLayout(layout)
        return group

    def _create_testing_panel(self) -> QWidget:
        """Create the strategy testing panel"""
        testing_group = QGroupBox("Strategy Testing")
        layout = QVBoxLayout()

        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        if self.available_datasets:
            self.dataset_combo.addItems(self.available_datasets.keys())
        dataset_layout.addWidget(self.dataset_combo)

        self.load_dataset_button = QPushButton("Load External")
        self.load_dataset_button.clicked.connect(self._load_external_dataset)
        dataset_layout.addWidget(self.load_dataset_button)
        layout.addLayout(dataset_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.test_strategy_button = QPushButton("Test")
        self.test_strategy_button.setEnabled(False)
        self.test_strategy_button.clicked.connect(self._test_strategy)
        
        self.accept_strategy_button = QPushButton("Accept and Close")
        self.accept_strategy_button.setEnabled(False)
        self.accept_strategy_button.clicked.connect(self._accept_strategy)
        
        self.reject_strategy_button = QPushButton("Reject and Close")
        self.reject_strategy_button.clicked.connect(self._reject_strategy)
        
        button_layout.addWidget(self.test_strategy_button)
        button_layout.addWidget(self.accept_strategy_button)
        button_layout.addWidget(self.reject_strategy_button)
        layout.addLayout(button_layout)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        testing_group.setLayout(layout)
        return testing_group

    def _add_action(self):
        """Add a new action to the strategy"""
        action_name = self.action_name_edit.text()
        if not action_name:
            QMessageBox.warning(self, "Warning", "Please enter an action name.")
            return

        pattern_name = self.pattern_combo.currentText()
        if pattern_name == "Location Only":
            pattern = None
        elif pattern_name == "FVG (Fair Value Gap)":
            # PATCH: Always use the FVG pattern object from available_patterns
            pattern = self.available_patterns.get("FVG (Fair Value Gap)")
            if pattern is None:
                QMessageBox.warning(self, "Warning", "FVG pattern object not found in available_patterns.")
                return
        else:
            pattern = self.available_patterns.get(pattern_name)
        
        if not pattern and pattern_name != "Location Only":
            QMessageBox.warning(self, "Warning", f"Pattern '{pattern_name}' not found.")
            return

        time_range = {
            'value': self.time_range_value.value(),
            'unit': self.time_range_unit.currentText()
        }
        
        # Zone strategy
        location_strategy = self.location_gate_combo.currentText()
        if location_strategy == "None":
            location_strategy = None
            
        # Indicator Filters
        filters = []
        if self.ma_filter_check.isChecked():
            filters.append({
                'type': 'ma',
                'period': self.ma_period_spin.value(),
                'condition': self.ma_condition_combo.currentText()
            })
        if self.vwap_filter_check.isChecked():
            filters.append({
                'type': 'vwap',
                'condition': self.vwap_condition_combo.currentText()
            })
        if self.bb_filter_check.isChecked():
            filters.append({
                'type': 'bollinger_bands',
                'period': self.bb_period_spin.value(),
                'std_dev': self.bb_std_spin.value(),
                'condition': self.bb_condition_combo.currentText()
            })
        
        # Microstructure Filters
        if self.tick_frequency_filter_check.isChecked():
            filters.append({
                'type': 'tick_frequency',
                'max_ticks_per_second': self.max_ticks_per_second_spin.value(),
                'min_book_depth': self.min_book_depth_spin.value()
            })
        if self.spread_filter_check.isChecked():
            filters.append({
                'type': 'spread',
                'max_spread_ticks': self.max_spread_ticks_spin.value(),
                'normal_spread_multiple': self.normal_spread_multiple_spin.value()
            })
        if self.order_flow_filter_check.isChecked():
            filters.append({
                'type': 'order_flow',
                'min_cvd_threshold': self.min_cvd_threshold_spin.value(),
                'large_trade_ratio': self.large_trade_ratio_spin.value()
            })
        
        # Advanced Filters
        if self.volume_filter_check.isChecked():
            filters.append({
                'type': 'volume',
                'min_volume': self.min_volume_spin.value(),
                'volume_ratio': self.volume_ratio_spin.value()
            })
        if self.time_filter_check.isChecked():
            filters.append({
                'type': 'time',
                'start_time': self.start_time_edit.time().toString('HH:mm'),
                'end_time': self.end_time_edit.time().toString('HH:mm')
            })
        if self.volatility_filter_check.isChecked():
            filters.append({
                'type': 'volatility',
                'min_atr_ratio': self.min_atr_ratio_spin.value(),
                'max_atr_ratio': self.max_atr_ratio_spin.value()
            })
        if self.momentum_filter_check.isChecked():
            filters.append({
                'type': 'momentum',
                'momentum_threshold': self.momentum_threshold_spin.value(),
                'rsi_range': [self.rsi_min_spin.value(), self.rsi_max_spin.value()]
            })
        if self.price_filter_check.isChecked():
            filters.append({
                'type': 'price',
                'min_price': self.min_price_spin.value(),
                'max_price': self.max_price_spin.value()
            })

        action = Action(
            name=action_name,
            pattern=pattern,
            time_range=time_range,
            location_strategy=location_strategy,
            filters=filters
        )

        self.actions.append(action)
        self._update_actions_list()
        self.action_name_edit.clear()
        self.pattern_combo.setCurrentIndex(0)
        self.location_gate_combo.setCurrentIndex(0)
        self.ma_filter_check.setChecked(False)
        self.vwap_filter_check.setChecked(False)
        self.bb_filter_check.setChecked(False)

    def _update_actions_list(self):
        """Update the list of actions in the UI"""
        self.actions_list.clear()
        for action in self.actions:
            pattern_str = action.pattern.name if action.pattern else "Location Only"
            self.actions_list.addItem(f"{action.name} (Pattern: {pattern_str})")

    def _remove_action(self):
        """Remove the selected action"""
        current_row = self.actions_list.currentRow()
        if current_row >= 0:
            self.actions.pop(current_row)
            self._update_actions_list()

    def _clear_actions(self):
        """Clear all actions"""
        self.actions = []
        self._update_actions_list()

    def _create_strategy(self):
        """Create and save the current strategy"""
        # Always get the latest name from the QLineEdit
        name = self.strategy_name_edit.text()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a strategy name.")
            return

        if not self.actions:
            QMessageBox.warning(self, "Warning", "A strategy must have at least one action.")
            return

        # Collect location parameters
        location_params = {
            # Global settings
            'bar_interval_minutes': self.bar_interval_minutes_spin.value(),
            
            # FVG parameters
            'fvg_epsilon': self.fvg_epsilon_spin.value(),
            'fvg_N': self.fvg_N_spin.value(),
            'fvg_sigma': self.fvg_sigma_spin.value(),
            'fvg_beta1': self.fvg_beta1_spin.value(),
            'fvg_beta2': self.fvg_beta2_spin.value(),
            'fvg_phi': self.fvg_phi_spin.value(),
            'fvg_lambda': self.fvg_lambda_spin.value(),
            'fvg_gamma': self.fvg_gamma_spin.value(),
            'fvg_tau_bars': self.fvg_tau_spin.value(),
            'fvg_drop_threshold': self.fvg_drop_threshold_spin.value(),
            
            # VWAP parameters
            'vwap_k': self.vwap_k_spin.value(),
            'vwap_lookback': self.vwap_lookback_spin.value(),
            'vwap_gamma': self.vwap_gamma_spin.value(),
            'vwap_tau_bars': self.vwap_tau_spin.value(),
            'vwap_drop_threshold': self.vwap_drop_threshold_spin.value(),
            
            # Support/Resistance parameters
            'sr_window': self.sr_window_spin.value(),
            'sr_buffer_pts': self.sr_buffer_pts_spin.value(),
            'sr_sigma_r': self.sr_sigma_r_spin.value(),
            'sr_sigma_t': self.sr_sigma_t_spin.value(),
            'sr_gamma': self.sr_gamma_spin.value(),
            'sr_tau_bars': self.sr_tau_spin.value(),
            'sr_drop_threshold': self.sr_drop_threshold_spin.value(),
            
            # Imbalance parameters
            'imbalance_threshold': self.imbalance_threshold_spin.value(),
            'imbalance_gamma_mem': self.imbalance_gamma_mem_spin.value(),
            'imbalance_sigma_rev': self.imbalance_sigma_rev_spin.value(),
            'imbalance_gamma': self.imbalance_gamma_spin.value(),
            'imbalance_tau_bars': self.imbalance_tau_spin.value(),
            'imbalance_drop_threshold': self.imbalance_drop_threshold_spin.value(),
            
            # Order Block parameters
            'ob_impulse_threshold': self.ob_impulse_threshold_spin.value(),
            'ob_lookback': self.ob_lookback_spin.value(),
            'ob_gamma': self.ob_gamma_spin.value(),
            'ob_tau_bars': self.ob_tau_spin.value(),
            'ob_buffer_points': self.ob_buffer_spin.value(),
            
            # Legacy parameters for compatibility
            'sr_threshold': 0.015,
            'gamma_z': 1.0,
            'delta_y': 0.0,
            'omega_mem': 1.0,
            'kernel_xi': 0.5,
            'kernel_alpha': 2.0,
        }
        
        # The main 'location_gate' is active if any action uses a zone type
        zone_in_use = any(action.location_strategy and action.location_strategy != "None" for action in self.actions)
        
        # Map UI zone names to internal zone types
        zone_type_mapping = {
            "FVG (Fair Value Gap)": "FVG",
            "VWAP Mean-Reversion Band": "VWAP",
            # Support/Resistance Band removed
            "Imbalance Memory Zone": "Imbalance",
            "Order Block": "Order Block"
        }
        
        # Update actions with mapped zone types
        for action in self.actions:
            if action.location_strategy and action.location_strategy in zone_type_mapping:
                action.location_strategy = zone_type_mapping[action.location_strategy]

        # Always use the latest name from the QLineEdit
        self.current_strategy = PatternStrategy(
            id=f"strat_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=name,
            actions=self.actions,
            combination_logic=self.combination_logic_combo.currentText(),
            gates_and_logic={
                'location_gate': zone_in_use,
                'volatility_gate': self.volatility_gate_check.isChecked(),
                'regime_gate': self.regime_gate_check.isChecked(),
                'bayesian_gate': self.bayesian_gate_check.isChecked(),
                # Microstructure gates
                'market_environment_gate': self.market_environment_gate_check.isChecked(),
                'news_time_gate': self.news_time_gate_check.isChecked(),
                'tick_validation_gate': self.tick_validation_gate_check.isChecked(),
                # Advanced gates
                'fvg_gate': self.fvg_gate_check.isChecked(),
                'momentum_gate': self.momentum_gate_check.isChecked(),
                'volume_gate': self.volume_gate_check.isChecked(),
                'time_gate': self.time_gate_check.isChecked(),
                'correlation_gate': self.correlation_gate_check.isChecked(),
                'order_block_gate': self.order_block_gate_check.isChecked()
            },
            location_gate_params=location_params
        )
        
        self.test_strategy_button.setEnabled(True)
        self.accept_strategy_button.setEnabled(True)
        
        self.results_text.setText(f"Strategy '{self.current_strategy.name}' has been created/updated.\n"
                                  "It is now ready for testing.")
        
        QMessageBox.information(self, "Strategy Created", 
                                f"Strategy '{self.current_strategy.name}' has been created and is ready for testing.")

    def _test_strategy(self):
        """Run a test of the current strategy"""
        if not self.current_strategy:
            QMessageBox.warning(self, "Warning", "Please create a strategy first.")
            return

        dataset_name = self.dataset_combo.currentText()
        if not dataset_name or not self.available_datasets:
            QMessageBox.warning(self, "Warning", "Please select or load a dataset.")
            return

        dataset = self.available_datasets[dataset_name]['data']
        
        # Instantiate backtest engine
        engine = BacktestEngine()
        
        try:
            results = engine.run_backtest(self.current_strategy, dataset)
            
            # Format results for display
            results_str = (
                f"Backtest Results for '{self.current_strategy.name}':\n"
                f"--------------------------------------------------\n"
                f"Total Trades: {results.get('total_trades', 0)}\n"
                f"Win Rate: {results.get('win_rate', 0):.2%}\n"
                f"Profit Factor: {results.get('profit_factor', 0):.2f}\n"
                f"Total Return: {results.get('total_return', 0):.2%}\n"
                f"Max Drawdown: {results.get('max_drawdown', 0):.2%}\n"
                f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
                f"Final Capital: ${results.get('final_capital', 0):,.2f}\n"
            )
            
            self.results_text.setText(results_str)
            
        except Exception as e:
            self.results_text.setText(f"An error occurred during backtesting: {e}")
            import traceback
            traceback.print_exc()

    def _accept_strategy(self):
        """Emit signal and close"""
        if self.current_strategy:
            self.strategy_created.emit(self.current_strategy)
            if self.parent_window:
                self.parent_window.save_strategy(self.current_strategy)
                self.parent_window.update_strategy_list()
                self.parent_window.show_status_message(f"Strategy '{self.current_strategy.name}' saved.")
            self.close()
        else:
            QMessageBox.warning(self, "Warning", "No strategy to accept.")

    def _reject_strategy(self):
        """Close without saving"""
        self.close()

    def _load_external_dataset(self):
        """Load a dataset from a file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                dataset_name = file_path.split('/')[-1]
                self.available_datasets[dataset_name] = {'data': df, 'metadata': {}}
                self.dataset_combo.addItem(dataset_name)
                self.dataset_combo.setCurrentText(dataset_name)
                self.results_text.setText(f"Loaded external dataset: {dataset_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def load_strategy_for_editing(self, strategy: PatternStrategy):
        """Load a strategy into the builder for editing"""
        if self.parent_window:
            self.parent_window.show_status_message(f"Loading strategy '{strategy.name}' for editing...")

        self.strategy_name_edit.setText(strategy.name)
        self.combination_logic_combo.setCurrentText(strategy.combination_logic)
        
        # Clear existing UI state and load actions
        self._clear_actions()
        self.actions = list(strategy.actions) # Use a copy
        self._update_actions_list()

        # Load gates
        gates = strategy.gates_and_logic or {}
        self.volatility_gate_check.setChecked(gates.get('volatility_gate', False))
        self.regime_gate_check.setChecked(gates.get('regime_gate', False))
        self.bayesian_gate_check.setChecked(gates.get('bayesian_gate', False))
        
        # The location gate dropdowns are handled by selecting an action
        # For now, just ensure the params button is in the correct state
        self._toggle_location_params_button(self.location_gate_combo.currentText())

        # Load location parameters for the Definitive Zone
        if strategy.location_gate_params:
            for name, widget in self.location_param_widgets.items():
                if name in strategy.location_gate_params:
                    widget.setValue(strategy.location_gate_params[name])
                else:
                    widget.setValue(DEFAULT_LOCATION_PARAMS[name])
        else:
            self._reset_location_params()

        # Load filter settings
        for f in strategy.filters:
            if f.get('type') == 'ma':
                self.ma_filter_check.setChecked(True)
                self.ma_period_spin.setValue(f.get('period', 20))
                self.ma_condition_combo.setCurrentText(f.get('condition', 'above'))
            elif f.get('type') == 'vwap':
                self.vwap_filter_check.setChecked(True)
                self.vwap_condition_combo.setCurrentText(f.get('condition', 'above'))
            elif f.get('type') == 'bollinger_bands':
                self.bb_filter_check.setChecked(True)
                self.bb_period_spin.setValue(f.get('period', 20))
                self.bb_std_spin.setValue(f.get('std_dev', 2.0))
                self.bb_condition_combo.setCurrentText(f.get('condition', 'inside'))

        self.current_strategy = strategy
        self.test_strategy_button.setEnabled(True)
        self.accept_strategy_button.setEnabled(True)
        
        self.results_text.setText(f"Loaded strategy '{strategy.name}' for editing.")
        self.tabs.setCurrentIndex(0) # Switch to builder tab
        self.show()
        self.raise_()
        self.activateWindow()

    def _display_action_details(self, current, previous):
        """When an action is selected in the list, display its details in the UI."""
        if current is None:
            return
            
        row = self.actions_list.row(current)
        if row < 0 or row >= len(self.actions):
            return

        action = self.actions[row]
        
        # --- Reset all filter checks before loading action details ---
        self.ma_filter_check.setChecked(False)
        self.vwap_filter_check.setChecked(False)
        self.bb_filter_check.setChecked(False)

        self.action_name_edit.setText(action.name)
        
        if action.pattern:
            self.pattern_combo.setCurrentText(action.pattern.name)
        else:
            self.pattern_combo.setCurrentText("Location Only")
            
        if action.location_strategy:
            self.location_gate_combo.setCurrentText(action.location_strategy)
        else:
            self.location_gate_combo.setCurrentText("None")

        if action.time_range:
            self.time_range_value.setValue(action.time_range.get('value', 1))
            self.time_range_unit.setCurrentText(action.time_range.get('unit', 'minutes'))
            
        # Load filter settings
        for f in action.filters:
            if f.get('type') == 'ma':
                self.ma_filter_check.setChecked(True)
                self.ma_period_spin.setValue(f.get('period', 20))
                self.ma_condition_combo.setCurrentText(f.get('condition', 'above'))
            elif f.get('type') == 'vwap':
                self.vwap_filter_check.setChecked(True)
                self.vwap_condition_combo.setCurrentText(f.get('condition', 'above'))
            elif f.get('type') == 'bollinger_bands':
                self.bb_filter_check.setChecked(True)
                self.bb_period_spin.setValue(f.get('period', 20))
                self.bb_std_spin.setValue(f.get('std_dev', 2.0))
                self.bb_condition_combo.setCurrentText(f.get('condition', 'inside'))

    def _duplicate_strategy(self):
        """Duplicate the current strategy, including all actions, gates, and parameters."""
        if not self.current_strategy:
            QMessageBox.warning(self, "Warning", "No strategy to duplicate.")
            return
        # Deep copy the strategy and actions
        new_strategy = deepcopy(self.current_strategy)
        new_strategy.name = f"{self.current_strategy.name}_copy"
        self.load_strategy_for_editing(new_strategy)
        QMessageBox.information(self, "Strategy Duplicated", f"Strategy '{self.current_strategy.name}' duplicated. You can now add or remove actions.")