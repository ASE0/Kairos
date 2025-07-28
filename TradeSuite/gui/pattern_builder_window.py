"""
gui/pattern_builder_window.py
=============================
Window for building custom candlestick patterns
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
from typing import Dict, List, Optional

from patterns.candlestick_patterns import CustomPattern, PatternFactory, CandlestickPattern
from core.data_structures import OHLCRatio, TimeRange
from core.pattern_registry import registry


class PatternBuilderWindow(QMainWindow):
    """Window for building custom patterns"""
    
    # Signals
    pattern_created = pyqtSignal(str, object)  # name, pattern
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pattern Builder")
        self.setGeometry(250, 250, 800, 600)
        
        # Current pattern components
        self.ohlc_ratios = []
        self.custom_formulas = []
        
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
                border: 1px solid #555555;
                spacing: 3px;
                padding: 3px;
            }
            QToolButton {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QToolButton:hover {
                background-color: #5a5a5a;
            }
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
                border-top: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background-color: #3c3c3c;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                width: 16px;
                border-radius: 8px;
                margin: -4px 0;
            }
            QSlider::handle:horizontal:hover {
                background-color: #1a8fd4;
            }
            QDateTimeEdit, QTimeEdit, QDateEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 5px;
                color: #ffffff;
                border-radius: 3px;
            }
            QDateTimeEdit::drop-down, QTimeEdit::drop-down, QDateEdit::drop-down {
                border: none;
                background-color: #4a4a4a;
            }
            QCalendarWidget {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QCalendarWidget QToolButton {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QCalendarWidget QMenu {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #ffffff;
            }
            QCalendarWidget QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #ffffff;
            }
        """)
        
    def _setup_ui(self):
        """Setup UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Main content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Pattern name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Pattern Name:"))
        self.pattern_name = QLineEdit()
        self.pattern_name.setPlaceholderText("e.g., My Custom Pattern")
        name_layout.addWidget(self.pattern_name)
        layout.addLayout(name_layout)
        
        # Pattern type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Base Pattern Type:"))
        self.pattern_type = QComboBox()
        # Populate from registry
        pattern_names = registry.get_pattern_names()
        self.pattern_type.addItems(pattern_names)
        self.pattern_type.currentTextChanged.connect(self._on_pattern_type_changed)
        type_layout.addWidget(self.pattern_type)
        layout.addLayout(type_layout)
        
        # Main content area
        self.content_stack = QStackedWidget()
        
        # Custom pattern widget
        self.custom_widget = self._create_custom_pattern_widget()
        self.content_stack.addWidget(self.custom_widget)
        
        # Predefined pattern widget
        self.predefined_widget = self._create_predefined_pattern_widget()
        self.content_stack.addWidget(self.predefined_widget)
        
        layout.addWidget(self.content_stack)
        
        # Timeframes
        tf_layout = QHBoxLayout()
        tf_layout.addWidget(QLabel("Timeframes:"))
        self.timeframes_input = QLineEdit()
        self.timeframes_input.setPlaceholderText("e.g., 1m,5m,15m,1h")
        self.timeframes_input.setText("5m")
        tf_layout.addWidget(self.timeframes_input)
        layout.addLayout(tf_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("Test Pattern")
        self.test_btn.clicked.connect(self._test_pattern)
        button_layout.addWidget(self.test_btn)
        
        self.save_btn = QPushButton("Save Pattern")
        self.save_btn.clicked.connect(self._save_pattern)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Pattern preview
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Pattern Definition:"))
        layout.addWidget(self.preview_text)
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Main layout for central widget
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(scroll_area)
        
        # Set size policies for resizable window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(600, 400)  # Minimum size to ensure usability
        
        self._update_ratio_inputs()  # <-- Now both formula_input and preview_text exist
        
    def _create_custom_pattern_widget(self) -> QWidget:
        """Create widget for custom pattern building"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Number of bars
        bars_layout = QHBoxLayout()
        bars_layout.addWidget(QLabel("Number of Bars:"))
        self.num_bars_spin = QSpinBox()
        self.num_bars_spin.setRange(1, 10)
        self.num_bars_spin.setValue(1)
        self.num_bars_spin.valueChanged.connect(self._update_ratio_inputs)
        bars_layout.addWidget(self.num_bars_spin)
        bars_layout.addStretch()
        layout.addLayout(bars_layout)
        
        # OHLC Ratios section
        layout.addWidget(QLabel("OHLC Ratios (per bar):"))
        
        self.ratios_scroll = QScrollArea()
        self.ratios_widget = QWidget()
        self.ratios_layout = QVBoxLayout(self.ratios_widget)
        self.ratios_scroll.setWidget(self.ratios_widget)
        self.ratios_scroll.setWidgetResizable(True)
        self.ratios_scroll.setMaximumHeight(200)
        layout.addWidget(self.ratios_scroll)
        
        # Custom formula section
        layout.addWidget(QLabel("Custom Formula (optional):"))
        
        formula_help = QLabel(
            "Available variables: open[i], high[i], low[i], close[i], volume[i]\n"
            "Example: close[0] > open[0] and volume[0] > volume[1] * 1.5"
        )
        formula_help.setWordWrap(True)
        formula_help.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(formula_help)
        
        self.formula_input = QTextEdit()
        self.formula_input.setMaximumHeight(60)
        self.formula_input.setPlaceholderText("Enter Python expression...")
        layout.addWidget(self.formula_input)

        # --- Advanced Features Section ---
        advanced_group = QGroupBox("Advanced Features (Quantification & Context)")
        advanced_layout = QFormLayout()

        # Body size
        self.body_size_check = QCheckBox("Body Size (Bt)")
        self.body_size_check.setToolTip("|C-O|: Absolute body size of the candle")
        advanced_layout.addRow(self.body_size_check)

        # Upper wick
        self.upper_wick_check = QCheckBox("Upper Wick (Wu)")
        self.upper_wick_check.setToolTip("H - max(O,C): Upper wick length")
        advanced_layout.addRow(self.upper_wick_check)

        # Lower wick
        self.lower_wick_check = QCheckBox("Lower Wick (Wl)")
        self.lower_wick_check.setToolTip("min(O,C) - L: Lower wick length")
        advanced_layout.addRow(self.lower_wick_check)

        # Doji-ness
        self.doji_ness_check = QCheckBox("Doji-ness (Dt)")
        self.doji_ness_check.setToolTip("exp[-(Bt/Range)^2/(2σ_b^2)] · exp[-(Ŵu-Ŵl)^2/(2σ_w^2)]")
        self.sigma_b_spin = QDoubleSpinBox(); self.sigma_b_spin.setRange(0.01, 1.0); self.sigma_b_spin.setValue(0.1)
        self.sigma_w_spin = QDoubleSpinBox(); self.sigma_w_spin.setRange(0.01, 1.0); self.sigma_w_spin.setValue(0.1)
        doji_params = QWidget(); doji_params_layout = QHBoxLayout(doji_params)
        doji_params_layout.addWidget(QLabel("σ_b:")); doji_params_layout.addWidget(self.sigma_b_spin)
        doji_params_layout.addWidget(QLabel("σ_w:")); doji_params_layout.addWidget(self.sigma_w_spin)
        advanced_layout.addRow(self.doji_ness_check, doji_params)

        # Two-bar strength
        self.twobar_check = QCheckBox("Two-Bar Strength (A₂bar)")
        self.twobar_check.setToolTip("β_pat·(Body₂/Body₁): Two-bar pattern strength")
        self.beta_pat_spin = QDoubleSpinBox(); self.beta_pat_spin.setRange(0.1, 5.0); self.beta_pat_spin.setValue(1.0)
        advanced_layout.addRow(self.twobar_check, self.beta_pat_spin)

        # Location context
        self.location_check = QCheckBox("Location Context (L_total)")
        self.location_check.setToolTip("Location score using FVG, peaks, skew, etc.")
        advanced_layout.addRow(self.location_check)

        # Momentum context
        self.momentum_check = QCheckBox("Momentum Context (L_mom)")
        self.momentum_check.setToolTip("Momentum-adaptive location score boost")
        self.kappa_m_spin = QDoubleSpinBox(); self.kappa_m_spin.setRange(0.0, 2.0); self.kappa_m_spin.setValue(0.5)
        advanced_layout.addRow(self.momentum_check, self.kappa_m_spin)

        # Reversion context
        self.reversion_check = QCheckBox("Reversion/Imbalance Context (R_imb)")
        self.reversion_check.setToolTip("Imbalance memory and reversion bump")
        advanced_layout.addRow(self.reversion_check)

        # Pattern confidence
        self.pattern_confidence_check = QCheckBox("Pattern Confidence (q_T)")
        self.pattern_confidence_check.setToolTip("Pattern confidence: q_T = σ[κ(Corr_T - τ)]")
        self.kappa_conf_spin = QDoubleSpinBox(); self.kappa_conf_spin.setRange(0.1, 10.0); self.kappa_conf_spin.setValue(2.0)
        self.tau_conf_spin = QDoubleSpinBox(); self.tau_conf_spin.setRange(0.1, 1.0); self.tau_conf_spin.setValue(0.7)
        conf_params = QWidget(); conf_params_layout = QHBoxLayout(conf_params)
        conf_params_layout.addWidget(QLabel("κ:")); conf_params_layout.addWidget(self.kappa_conf_spin)
        conf_params_layout.addWidget(QLabel("τ:")); conf_params_layout.addWidget(self.tau_conf_spin)
        advanced_layout.addRow(self.pattern_confidence_check, conf_params)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        # --- End Advanced Features Section ---

        widget.setLayout(layout)
        return widget
        
    def _create_predefined_pattern_widget(self) -> QWidget:
        """Create widget for predefined pattern configuration"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Parameters based on pattern type
        self.param_stack = QStackedWidget()
        
        # II Bars parameters
        ii_widget = QWidget()
        ii_layout = QFormLayout()
        self.ii_min_bars = QSpinBox()
        self.ii_min_bars.setRange(2, 10)
        self.ii_min_bars.setValue(2)
        ii_layout.addRow("Minimum Bars:", self.ii_min_bars)
        ii_widget.setLayout(ii_layout)
        self.param_stack.addWidget(ii_widget)
        
        # Double Wick parameters
        dw_widget = QWidget()
        dw_layout = QFormLayout()
        self.dw_min_wick = QDoubleSpinBox()
        self.dw_min_wick.setRange(0.1, 0.9)
        self.dw_min_wick.setValue(0.3)
        self.dw_min_wick.setSingleStep(0.1)
        dw_layout.addRow("Min Wick Ratio:", self.dw_min_wick)
        self.dw_max_body = QDoubleSpinBox()
        self.dw_max_body.setRange(0.1, 0.9)
        self.dw_max_body.setValue(0.4)
        self.dw_max_body.setSingleStep(0.1)
        dw_layout.addRow("Max Body Ratio:", self.dw_max_body)
        dw_widget.setLayout(dw_layout)
        self.param_stack.addWidget(dw_widget)
        
        # Engulfing parameters
        engulf_widget = QWidget()
        engulf_layout = QFormLayout()
        self.engulf_type = QComboBox()
        self.engulf_type.addItems(['Both', 'Bullish', 'Bearish'])
        engulf_layout.addRow("Pattern Type:", self.engulf_type)
        engulf_widget.setLayout(engulf_layout)
        self.param_stack.addWidget(engulf_widget)
        
        # Doji parameters
        doji_widget = QWidget()
        doji_layout = QFormLayout()
        self.doji_max_body = QDoubleSpinBox()
        self.doji_max_body.setRange(0.01, 0.2)
        self.doji_max_body.setValue(0.1)
        self.doji_max_body.setSingleStep(0.01)
        doji_layout.addRow("Max Body Ratio:", self.doji_max_body)
        doji_widget.setLayout(doji_layout)
        self.param_stack.addWidget(doji_widget)
        
        # Body parameters
        body_widget = QWidget()
        body_layout = QFormLayout()
        self.body_threshold = QDoubleSpinBox()
        self.body_threshold.setRange(0.1, 0.9)
        self.body_threshold.setValue(0.6)
        self.body_threshold.setSingleStep(0.1)
        body_layout.addRow("Body Threshold:", self.body_threshold)
        body_widget.setLayout(body_layout)
        self.param_stack.addWidget(body_widget)
        
        # Momentum parameters
        momentum_widget = QWidget()
        momentum_layout = QFormLayout()
        self.momentum_threshold = QDoubleSpinBox()
        self.momentum_threshold.setRange(0.01, 0.1)
        self.momentum_threshold.setValue(0.02)
        self.momentum_threshold.setSingleStep(0.01)
        momentum_layout.addRow("Momentum Threshold:", self.momentum_threshold)
        momentum_widget.setLayout(momentum_layout)
        self.param_stack.addWidget(momentum_widget)
        
        # Volatility parameters
        volatility_widget = QWidget()
        volatility_layout = QFormLayout()
        self.volatility_threshold = QDoubleSpinBox()
        self.volatility_threshold.setRange(0.01, 0.1)
        self.volatility_threshold.setValue(0.03)
        self.volatility_threshold.setSingleStep(0.01)
        volatility_layout.addRow("Volatility Threshold:", self.volatility_threshold)
        volatility_widget.setLayout(volatility_layout)
        self.param_stack.addWidget(volatility_widget)
        
        # Multi-bar parameters
        multibar_widget = QWidget()
        multibar_layout = QFormLayout()
        self.multibar_count = QSpinBox()
        self.multibar_count.setRange(3, 5)
        self.multibar_count.setValue(3)
        multibar_layout.addRow("Bar Count:", self.multibar_count)
        multibar_widget.setLayout(multibar_layout)
        self.param_stack.addWidget(multibar_widget)
        
        # Trend parameters
        trend_widget = QWidget()
        trend_layout = QFormLayout()
        self.trend_strength = QDoubleSpinBox()
        self.trend_strength.setRange(0.1, 1.0)
        self.trend_strength.setValue(0.5)
        self.trend_strength.setSingleStep(0.1)
        trend_layout.addRow("Trend Strength:", self.trend_strength)
        trend_widget.setLayout(trend_layout)
        self.param_stack.addWidget(trend_widget)
        
        # Gap parameters
        gap_widget = QWidget()
        gap_layout = QFormLayout()
        self.gap_size = QDoubleSpinBox()
        self.gap_size.setRange(0.001, 0.05)
        self.gap_size.setValue(0.01)
        self.gap_size.setSingleStep(0.001)
        gap_layout.addRow("Min Gap Size:", self.gap_size)
        gap_widget.setLayout(gap_layout)
        self.param_stack.addWidget(gap_widget)
        
        # Market structure parameters
        market_widget = QWidget()
        market_layout = QFormLayout()
        self.structure_threshold = QDoubleSpinBox()
        self.structure_threshold.setRange(0.1, 1.0)
        self.structure_threshold.setValue(0.5)
        self.structure_threshold.setSingleStep(0.1)
        market_layout.addRow("Structure Threshold:", self.structure_threshold)
        market_widget.setLayout(market_layout)
        self.param_stack.addWidget(market_widget)
        
        # Default parameters
        default_widget = QWidget()
        default_layout = QFormLayout()
        default_layout.addRow(QLabel("Standard pattern configuration"))
        default_widget.setLayout(default_layout)
        self.param_stack.addWidget(default_widget)
        
        layout.addWidget(self.param_stack)
        
        widget.setLayout(layout)
        return widget
        
    def _on_pattern_type_changed(self, pattern_type: str):
        """Handle pattern type change"""
        if pattern_type == 'Custom':
            self.content_stack.setCurrentIndex(0)
            self._update_preview()
        else:
            self.content_stack.setCurrentIndex(1)
            # Update parameter stack based on pattern type
            if pattern_type == 'II Bars':
                self.param_stack.setCurrentIndex(0)
            elif pattern_type == 'Double Wick':
                self.param_stack.setCurrentIndex(1)
            elif pattern_type == 'Engulfing':
                self.param_stack.setCurrentIndex(2)
            elif pattern_type in ['Doji', 'Doji Standard', 'Dragonfly Doji', 'Gravestone Doji', 'Four Price Doji']:
                self.param_stack.setCurrentIndex(3)  # Doji parameters
            elif pattern_type in ['Strong Body', 'Weak Body']:
                self.param_stack.setCurrentIndex(4)  # Body parameters
            elif pattern_type in ['Momentum Breakout', 'Momentum Reversal']:
                self.param_stack.setCurrentIndex(5)  # Momentum parameters
            elif pattern_type in ['High Volatility', 'Low Volatility', 'Volatility Expansion', 'Volatility Contraction']:
                self.param_stack.setCurrentIndex(6)  # Volatility parameters
            elif pattern_type in ['Three White Soldiers', 'Three Black Crows']:
                self.param_stack.setCurrentIndex(7)  # Multi-bar parameters
            elif pattern_type in ['Trend Continuation', 'Trend Reversal']:
                self.param_stack.setCurrentIndex(8)  # Trend parameters
            elif pattern_type in ['Gap Up', 'Gap Down']:
                self.param_stack.setCurrentIndex(9)  # Gap parameters
            elif pattern_type in ['Consolidation', 'Breakout', 'Exhaustion', 'Accumulation', 'Distribution']:
                self.param_stack.setCurrentIndex(10)  # Market structure parameters
            else:
                self.param_stack.setCurrentIndex(11)  # Default parameters
            self._update_preview()
            
    def _update_ratio_inputs(self):
        """Update ratio input fields based on number of bars"""
        # Clear existing inputs
        while self.ratios_layout.count():
            item = self.ratios_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.ratio_inputs = []
        num_bars = self.num_bars_spin.value()
        
        for i in range(num_bars):
            # Create frame for each bar
            bar_frame = QFrame()
            bar_frame.setFrameStyle(QFrame.Shape.Box)
            bar_layout = QFormLayout()
            
            bar_layout.addRow(QLabel(f"<b>Bar {i}:</b>"))
            
            # Body ratio
            body_spin = QDoubleSpinBox()
            body_spin.setRange(0, 1)
            body_spin.setSingleStep(0.1)
            body_spin.setSpecialValueText("Any")
            bar_layout.addRow("Body Ratio:", body_spin)
            
            # Upper wick ratio
            upper_spin = QDoubleSpinBox()
            upper_spin.setRange(0, 1)
            upper_spin.setSingleStep(0.1)
            upper_spin.setSpecialValueText("Any")
            bar_layout.addRow("Upper Wick:", upper_spin)
            
            # Lower wick ratio
            lower_spin = QDoubleSpinBox()
            lower_spin.setRange(0, 1)
            lower_spin.setSingleStep(0.1)
            lower_spin.setSpecialValueText("Any")
            bar_layout.addRow("Lower Wick:", lower_spin)
            
            bar_frame.setLayout(bar_layout)
            self.ratios_layout.addWidget(bar_frame)
            
            self.ratio_inputs.append({
                'body': body_spin,
                'upper': upper_spin,
                'lower': lower_spin
            })
            
        self.ratios_layout.addStretch()
        self._update_preview()
        
    def _update_preview(self):
        """Update pattern preview"""
        pattern_type = self.pattern_type.currentText()
        
        if pattern_type == 'Custom':
            preview = f"Custom Pattern: {self.pattern_name.text() or 'Unnamed'}\n"
            preview += f"Bars: {self.num_bars_spin.value()}\n\n"
            
            for i, inputs in enumerate(self.ratio_inputs):
                preview += f"Bar {i}:\n"
                if inputs['body'].value() > 0:
                    preview += f"  Body Ratio: {inputs['body'].value()}\n"
                if inputs['upper'].value() > 0:
                    preview += f"  Upper Wick: {inputs['upper'].value()}\n"
                if inputs['lower'].value() > 0:
                    preview += f"  Lower Wick: {inputs['lower'].value()}\n"
                    
            if self.formula_input.toPlainText():
                preview += f"\nCustom Formula:\n{self.formula_input.toPlainText()}"
        else:
            preview = f"Predefined Pattern: {pattern_type}\n"
            preview += self._get_predefined_description()
            
        self.preview_text.setText(preview)
        
    def _get_predefined_description(self) -> str:
        """Get description for predefined pattern"""
        pattern_type = self.pattern_type.currentText()
        
        if pattern_type == 'II Bars':
            return f"Inside-Inside bars with minimum {self.ii_min_bars.value()} bars"
        elif pattern_type == 'Double Wick':
            return (f"Min wick ratio: {self.dw_min_wick.value()}\n"
                   f"Max body ratio: {self.dw_max_body.value()}")
        elif pattern_type == 'Engulfing':
            return f"Type: {self.engulf_type.currentText()}"
        elif pattern_type in ['Doji', 'Doji Standard', 'Dragonfly Doji', 'Gravestone Doji', 'Four Price Doji']:
            return f"Max body ratio: {self.doji_max_body.value()}"
        elif pattern_type in ['Strong Body', 'Weak Body']:
            return f"Body threshold: {self.body_threshold.value()}"
        elif pattern_type in ['Momentum Breakout', 'Momentum Reversal']:
            return f"Momentum threshold: {self.momentum_threshold.value()}"
        elif pattern_type in ['High Volatility', 'Low Volatility', 'Volatility Expansion', 'Volatility Contraction']:
            return f"Volatility threshold: {self.volatility_threshold.value()}"
        elif pattern_type in ['Three White Soldiers', 'Three Black Crows']:
            return f"Bar count: {self.multibar_count.value()}"
        elif pattern_type in ['Trend Continuation', 'Trend Reversal']:
            return f"Trend strength: {self.trend_strength.value()}"
        elif pattern_type in ['Gap Up', 'Gap Down']:
            return f"Min gap size: {self.gap_size.value()}"
        elif pattern_type in ['Consolidation', 'Breakout', 'Exhaustion', 'Accumulation', 'Distribution']:
            return f"Structure threshold: {self.structure_threshold.value()}"
        else:
            return "Standard pattern configuration"
            
    def _test_pattern(self):
        """Test pattern on sample data"""
        # This would open a dialog to test on data
        QMessageBox.information(self, "Test Pattern", 
                              "Pattern testing would open a dialog with sample data")
        
    def _save_pattern(self):
        """Save the pattern"""
        name = self.pattern_name.text()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a pattern name")
            return
            
        pattern_type = self.pattern_type.currentText()
        
        # Parse timeframes
        timeframes = []
        for tf_str in self.timeframes_input.text().split(','):
            tf_str = tf_str.strip()
            if tf_str:
                import re
                match = re.match(r'(\d+)([smhd])', tf_str)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    timeframes.append(TimeRange(value, unit))
                    
        if not timeframes:
            timeframes = [TimeRange(5, 'm')]  # Default
            
        # Gather advanced features selections
        advanced_features = {
            'body_size': self.body_size_check.isChecked(),
            'upper_wick': self.upper_wick_check.isChecked(),
            'lower_wick': self.lower_wick_check.isChecked(),
            'doji_ness': self.doji_ness_check.isChecked(),
            'doji_sigma_b': self.sigma_b_spin.value(),
            'doji_sigma_w': self.sigma_w_spin.value(),
            'two_bar_strength': self.twobar_check.isChecked(),
            'beta_pat': self.beta_pat_spin.value(),
            'location_context': self.location_check.isChecked(),
            'momentum_context': self.momentum_check.isChecked(),
            'kappa_m': self.kappa_m_spin.value(),
            'reversion_context': self.reversion_check.isChecked(),
            'pattern_confidence': self.pattern_confidence_check.isChecked(),
            'kappa_conf': self.kappa_conf_spin.value(),
            'tau_conf': self.tau_conf_spin.value(),
        }
        
        # Create pattern based on type
        if pattern_type == 'Custom':
            # Create OHLC ratios
            ohlc_ratios = []
            for inputs in self.ratio_inputs:
                ratio = OHLCRatio(
                    body_ratio=inputs['body'].value() if inputs['body'].value() > 0 else None,
                    upper_wick_ratio=inputs['upper'].value() if inputs['upper'].value() > 0 else None,
                    lower_wick_ratio=inputs['lower'].value() if inputs['lower'].value() > 0 else None
                )
                ohlc_ratios.append(ratio)
                
            pattern = CustomPattern(
                name=name,
                timeframes=timeframes,
                ohlc_ratios=ohlc_ratios,
                custom_formula=self.formula_input.toPlainText() or None,
                required_bars=self.num_bars_spin.value(),
                advanced_features=advanced_features
            )
        else:
            # Create predefined pattern
            kwargs = {'timeframes': timeframes}
            
            if pattern_type == 'II Bars':
                kwargs['min_bars'] = self.ii_min_bars.value()
                pattern = PatternFactory.create_pattern('ii_bars', **kwargs)
            elif pattern_type == 'Double Wick':
                kwargs['min_wick_ratio'] = self.dw_min_wick.value()
                kwargs['max_body_ratio'] = self.dw_max_body.value()
                pattern = PatternFactory.create_pattern('double_wick', **kwargs)
            elif pattern_type == 'Engulfing':
                kwargs['pattern_type'] = self.engulf_type.currentText().lower()
                pattern = PatternFactory.create_pattern('engulfing', **kwargs)
            else:
                pattern = CustomPattern(name=name, timeframes=timeframes, 
                                      ohlc_ratios=[], required_bars=1)
                
            pattern.name = name  # Override with custom name
            
        # Emit signal
        self.pattern_created.emit(name, pattern)
        
        QMessageBox.information(self, "Success", f"Pattern '{name}' created successfully")
        self.close()

    def _on_accept(self):
        """Handle pattern acceptance"""
        pattern = self._create_pattern_from_ui()
        if pattern:
            self.pattern_created.emit(pattern)
            QMessageBox.information(self, "Success", f"Pattern '{pattern.name}' created.")
            self.close()

    def load_pattern_for_editing(self, pattern: CandlestickPattern):
        """Loads an existing pattern's data into the builder UI."""
        self.setWindowTitle(f"Edit Pattern - {pattern.name}")
        
        # Load basic info
        self.pattern_name_input.setText(pattern.name)
        
        # This assumes we are editing a 'CustomPattern'.
        # A more robust implementation would handle different pattern types.
        if isinstance(pattern, CustomPattern):
            # Load OHLCRatios
            self.param_table.setRowCount(len(pattern.ohlc_ratios))
            for i, ratio in enumerate(pattern.ohlc_ratios):
                self.param_table.item(i, 0).setText(ratio.part1)
                self.param_table.item(i, 1).setText(ratio.part2)
                self.param_table.cellWidget(i, 2).setCurrentText(ratio.operator)
                self.param_table.cellWidget(i, 3).setValue(ratio.value)

            # Load advanced features
            if hasattr(pattern, 'advanced_features') and pattern.advanced_features:
                 for key, value in pattern.advanced_features.items():
                    widget = self.adv_feature_inputs.get(key)
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(value)
                    elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                        widget.setValue(value)
        else:
            QMessageBox.warning(self, "Edit Not Supported",
                                f"Editing for '{type(pattern).__name__}' is not yet supported. "
                                "Only 'CustomPattern' types can be edited.")
            # We can still show the basic info but disable editing complex fields
            self.param_table.setEnabled(False)
            self.adv_features_group.setEnabled(False)

    def _on_cancel(self):
        """Handle cancellation"""
        self.close()
