"""
gui/strategy_combiner_window.py
================================
Window for combining pattern and risk strategies
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from strategies.strategy_builders import PatternStrategy, RiskStrategy, CombinedStrategy
from statistics1.probability_calculator import ProbabilityCalculator, StatisticalValidator


class StrategyCombinerWindow(QMainWindow):
    """Window for combining strategies"""

    # Signals
    combination_created = pyqtSignal(object)  # CombinedStrategy

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Strategy Combiner")
        self.setGeometry(350, 300, 1000, 700)

        # Get available strategies
        self.available_strategies = parent.strategies if parent else {'pattern': {}, 'risk': {}}

        # Analysis tools
        self.prob_calculator = ProbabilityCalculator()
        self.validator = StatisticalValidator()

        # Current combination
        self.current_combination = None

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

        # Combination name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Combination Name:"))
        self.combination_name = QLineEdit()
        self.combination_name.setPlaceholderText("e.g., VWAP Bounce with ATR Risk")
        name_layout.addWidget(self.combination_name)
        layout.addLayout(name_layout)

        # Strategy selection
        selection_layout = QHBoxLayout()

        # Pattern strategy selection
        pattern_group = QGroupBox("Pattern Strategy")
        pattern_layout = QVBoxLayout()

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItem("-- Select Pattern Strategy --")
        for strategy_id, strategy in self.available_strategies.get('pattern', {}).items():
            self.pattern_combo.addItem(strategy.name, strategy_id)
        pattern_layout.addWidget(self.pattern_combo)

        self.pattern_details = QTextEdit()
        self.pattern_details.setReadOnly(True)
        self.pattern_details.setMaximumHeight(100)
        pattern_layout.addWidget(self.pattern_details)

        pattern_group.setLayout(pattern_layout)
        selection_layout.addWidget(pattern_group)

        # Risk strategy selection
        risk_group = QGroupBox("Risk Strategy")
        risk_layout = QVBoxLayout()

        self.risk_combo = QComboBox()
        self.risk_combo.addItem("-- Select Risk Strategy --")
        for strategy_id, strategy in self.available_strategies.get('risk', {}).items():
            self.risk_combo.addItem(strategy.name, strategy_id)
        risk_layout.addWidget(self.risk_combo)

        self.risk_details = QTextEdit()
        self.risk_details.setReadOnly(True)
        self.risk_details.setMaximumHeight(100)
        risk_layout.addWidget(self.risk_details)

        risk_group.setLayout(risk_layout)
        selection_layout.addWidget(risk_group)

        layout.addLayout(selection_layout)

        # Connect selection changes
        self.pattern_combo.currentIndexChanged.connect(self._on_pattern_selected)
        self.risk_combo.currentIndexChanged.connect(self._on_risk_selected)

        # Combination logic
        logic_group = QGroupBox("Combination Logic")
        logic_layout = QFormLayout()

        # Entry logic
        self.entry_logic = QComboBox()
        self.entry_logic.addItems(['Pattern signals entry', 'Risk overrides pattern',
                                   'Both must agree', 'Either can signal'])
        logic_layout.addRow("Entry Logic:", self.entry_logic)

        # Exit logic
        self.exit_logic = QComboBox()
        self.exit_logic.addItems(['Risk controls exit', 'Pattern controls exit',
                                  'First exit signal', 'Both must agree'])
        logic_layout.addRow("Exit Logic:", self.exit_logic)

        # Position sizing
        self.sizing_logic = QComboBox()
        self.sizing_logic.addItems(['Risk strategy sizing', 'Fixed percentage',
                                    'Pattern confidence based', 'Volatility adjusted'])
        logic_layout.addRow("Position Sizing:", self.sizing_logic)

        logic_group.setLayout(logic_layout)
        layout.addWidget(logic_group)

        # Analysis section
        analysis_group = QGroupBox("Statistical Analysis")
        analysis_layout = QVBoxLayout()

        # Analysis controls
        analysis_controls = QHBoxLayout()

        self.analyze_btn = QPushButton("Analyze Combination")
        self.analyze_btn.clicked.connect(self._analyze_combination)
        analysis_controls.addWidget(self.analyze_btn)

        self.prob_method = QComboBox()
        self.prob_method.addItems(['Bayesian', 'Frequency', 'Monte Carlo', 'Conditional'])
        analysis_controls.addWidget(QLabel("Method:"))
        analysis_controls.addWidget(self.prob_method)

        analysis_layout.addLayout(analysis_controls)

        # Results display
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setMaximumHeight(150)
        analysis_layout.addWidget(self.analysis_results)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Action buttons
        action_layout = QHBoxLayout()

        self.create_btn = QPushButton("Create Combination")
        self.create_btn.clicked.connect(self._create_combination)
        action_layout.addWidget(self.create_btn)

        self.save_btn = QPushButton("Save & Accept")
        self.save_btn.clicked.connect(self._save_combination)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)

        layout.addLayout(action_layout)

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Main layout for central widget
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(scroll_area)

        # Set size policies for resizable window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(800, 500)  # Minimum size to ensure usability

    def _on_pattern_selected(self):
        """Handle pattern strategy selection"""
        if self.pattern_combo.currentIndex() <= 0:
            self.pattern_details.clear()
            return

        strategy_id = self.pattern_combo.currentData()
        strategy = self.available_strategies['pattern'].get(strategy_id)

        if strategy:
            details = f"Actions: {len(strategy.actions)}\n"
            if hasattr(strategy, 'probability_metrics') and strategy.probability_metrics:
                details += f"Probability: {strategy.probability_metrics.probability:.2%}\n"
            details += f"Created: {strategy.created_at.strftime('%Y-%m-%d %H:%M')}"
            self.pattern_details.setText(details)

    def _on_risk_selected(self):
        """Handle risk strategy selection"""
        if self.risk_combo.currentIndex() <= 0:
            self.risk_details.clear()
            return

        strategy_id = self.risk_combo.currentData()
        strategy = self.available_strategies['risk'].get(strategy_id)

        if strategy:
            details = f"Entry: {strategy.entry_method}\n"
            details += f"Stop Loss: {strategy.stop_loss_pct:.1%}\n"
            details += f"Risk/Reward: 1:{strategy.risk_reward_ratio}"
            self.risk_details.setText(details)

    def _analyze_combination(self):
        """Analyze the combination statistically"""
        if self.pattern_combo.currentIndex() <= 0 or self.risk_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select both strategies")
            return

        self.analysis_results.clear()
        self.analysis_results.append("Analyzing combination...\n")

        # Get strategies
        pattern_strategy = self.available_strategies['pattern'].get(self.pattern_combo.currentData())
        risk_strategy = self.available_strategies['risk'].get(self.risk_combo.currentData())

        # Mock analysis results
        method = self.prob_method.currentText()

        # Probability calculation
        if pattern_strategy.probability_metrics:
            base_prob = pattern_strategy.probability_metrics.probability
        else:
            base_prob = 0.5

        # Adjust based on risk strategy
        if risk_strategy.risk_reward_ratio >= 2:
            adjusted_prob = base_prob * 0.95  # Slightly lower due to tighter stops
        else:
            adjusted_prob = base_prob * 1.05

        adjusted_prob = min(max(adjusted_prob, 0), 1)

        self.analysis_results.append(f"Method: {method}")
        self.analysis_results.append(f"Base Probability: {base_prob:.2%}")
        self.analysis_results.append(f"Adjusted Probability: {adjusted_prob:.2%}")
        self.analysis_results.append(
            f"Confidence Interval: [{max(0, adjusted_prob - 0.05):.2%}, {min(1, adjusted_prob + 0.05):.2%}]")

        # Risk metrics
        expected_win = risk_strategy.risk_reward_ratio * risk_strategy.stop_loss_pct
        expected_loss = risk_strategy.stop_loss_pct
        expectancy = (adjusted_prob * expected_win) - ((1 - adjusted_prob) * expected_loss)

        self.analysis_results.append(f"\nRisk Metrics:")
        self.analysis_results.append(f"Expected Win: {expected_win:.2%}")
        self.analysis_results.append(f"Expected Loss: {expected_loss:.2%}")
        self.analysis_results.append(f"Expectancy: {expectancy:.3%}")

        # Recommendations
        if expectancy > 0 and adjusted_prob > 0.5:
            self.analysis_results.append("\n✓ Combination shows positive expectancy")
            self.save_btn.setEnabled(True)
        else:
            self.analysis_results.append("\n✗ Combination needs improvement")
            self.save_btn.setEnabled(False)

    def _create_combination(self):
        """Create the combination"""
        if self.pattern_combo.currentIndex() <= 0 or self.risk_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select both strategies")
            return

        name = self.combination_name.text()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a combination name")
            return

        # Get strategies
        pattern_strategy = self.available_strategies['pattern'].get(self.pattern_combo.currentData())
        risk_strategy = self.available_strategies['risk'].get(self.risk_combo.currentData())

        # Create combined strategy
        self.current_combination = CombinedStrategy(
            name=name,
            pattern_strategy=pattern_strategy,
            risk_strategy=risk_strategy,
            entry_logic=self.entry_logic.currentText(),
            exit_logic=self.exit_logic.currentText(),
            sizing_logic=self.sizing_logic.currentText()
        )

        self.analysis_results.append(f"\nCombination '{name}' created successfully")
        self.save_btn.setEnabled(True)

    def _save_combination(self):
        """Save and accept the combination"""
        if not self.current_combination:
            return

        # Update with analysis results
        from core.data_structures import ProbabilityMetrics

        metrics = ProbabilityMetrics()
        # Extract probability from analysis
        text = self.analysis_results.toPlainText()
        if "Adjusted Probability:" in text:
            prob_line = [line for line in text.split('\n') if "Adjusted Probability:" in line][0]
            prob_str = prob_line.split(":")[1].strip().rstrip('%')
            metrics.probability = float(prob_str) / 100

        self.current_combination.update_probability(metrics)

        # Emit signal
        self.combination_created.emit(self.current_combination)

        QMessageBox.information(self, "Success",
                                f"Combination '{self.current_combination.name}' saved successfully")

        # Reset
        self.combination_name.clear()
        self.pattern_combo.setCurrentIndex(0)
        self.risk_combo.setCurrentIndex(0)
        self.analysis_results.clear()
        self.current_combination = None
        self.save_btn.setEnabled(False)