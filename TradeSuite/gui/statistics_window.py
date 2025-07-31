"""
gui/statistics_window.py
========================
Window for statistical analysis of strategies and combinations
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pyqtgraph as pg
from scipy import stats
import itertools

from statistics1.probability_calculator import ProbabilityCalculator, StatisticalValidator, AcceptanceCalculator
from core.data_structures import ProbabilityMetrics
from strategies.strategy_builders import BacktestEngine


class StatisticsWindow(QMainWindow):
    """Window for statistical analysis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Statistics Analyzer")
        self.setGeometry(400, 200, 1000, 800)
        
        # Analysis tools
        self.prob_calculator = ProbabilityCalculator()
        self.validator = StatisticalValidator()
        self.acceptance_calc = AcceptanceCalculator()
        
        # Shared UI components for validation tab to prevent load-order errors
        self.test_layout = QVBoxLayout()
        self.test_checkboxes = {}
        self.single_strategy_tests = {
            'positive_mean': ("Positive Returns (T-test)", True),
            'sharpe_z_test': ("Sharpe Ratio > 0 (Z-test)", True),
            'normality': ("Normality of Returns", False)
        }
        self.multi_strategy_tests = {
            'independence': ("Independence (Pearson r)", True),
            'performance_diff': ("Performance Difference (T-test)", True)
        }
        
        # Current analysis data
        self.current_results = {}
        
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
        
        # Input section
        input_group = self._create_input_section()
        layout.addWidget(input_group)
        
        # Analysis tabs
        self.analysis_tabs = QTabWidget()
        
        # Probability tab
        self.prob_tab = self._create_probability_tab()
        self.analysis_tabs.addTab(self.prob_tab, "Probability Analysis")
        
        # Validation tab
        self.validation_tab = self._create_validation_tab()
        self.analysis_tabs.addTab(self.validation_tab, "Statistical Validation")
        
        # Acceptance tab
        self.acceptance_tab = self._create_acceptance_tab()
        self.analysis_tabs.addTab(self.acceptance_tab, "Acceptance Scoring")
        
        # Correlation tab
        self.correlation_tab = self._create_correlation_tab()
        self.analysis_tabs.addTab(self.correlation_tab, "Correlation Analysis")
        
        layout.addWidget(self.analysis_tabs)
        
        # Connect tab change to update UI
        self.analysis_tabs.currentChanged.connect(self._on_tab_changed)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self._run_analysis)
        button_layout.addWidget(self.analyze_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Main layout for central widget
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(scroll_area)
        
        # Set size policies for resizable window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size to ensure usability
        
    def _create_input_section(self) -> QGroupBox:
        """Create input section"""
        group = QGroupBox("Analysis Input")
        layout = QVBoxLayout()
        
        # Strategy selection
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Select Strategies:"))
        
        self.strategy_list = QListWidget()
        self.strategy_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.strategy_list.setMaximumHeight(100)
        
        # Populate with available strategies
        self._populate_strategies()
                    
        strategy_layout.addWidget(self.strategy_list)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        refresh_btn.setMaximumWidth(100)
        strategy_layout.addWidget(refresh_btn)
        
        layout.addLayout(strategy_layout)
        
        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Test Dataset:"))
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("-- Select Dataset --")
        # Populate from available datasets
        self._populate_datasets()
        # Connect dataset selection to timeframe adjustment
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        dataset_layout.addWidget(self.dataset_combo)
        
        layout.addLayout(dataset_layout)
        
        # Timeframe selection
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe Range:"))
        
        self.start_date = QDateTimeEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        timeframe_layout.addWidget(QLabel("From:"))
        timeframe_layout.addWidget(self.start_date)
        
        self.end_date = QDateTimeEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        timeframe_layout.addWidget(QLabel("To:"))
        timeframe_layout.addWidget(self.end_date)
        
        layout.addLayout(timeframe_layout)
        
        # Analysis parameters
        params_layout = QFormLayout()
        
        self.lookback_days = QSpinBox()
        self.lookback_days.setRange(1, 365)
        self.lookback_days.setValue(30)
        params_layout.addRow("Lookback Days:", self.lookback_days)
        
        self.confidence_level = QDoubleSpinBox()
        self.confidence_level.setRange(0.8, 0.99)
        self.confidence_level.setValue(0.95)
        self.confidence_level.setSingleStep(0.01)
        params_layout.addRow("Confidence Level:", self.confidence_level)
        
        layout.addLayout(params_layout)
        
        # Add a listener to strategy selection change
        self.strategy_list.itemSelectionChanged.connect(self._update_validation_tests_ui)
        
        group.setLayout(layout)
        return group
        
    def _populate_strategies(self):
        """Populate strategy list, ensuring every item has a strategy object."""
        self.strategy_list.clear()
        
        # Get all strategies and results from the main hub
        all_strategies = self.parent_window.strategies if self.parent_window else {}
        all_results = self.parent_window.results if self.parent_window else {}
        
        # Create a lookup from name to strategy object
        strategy_by_name = {}
        for s_type in all_strategies:
            for s_id, strategy in all_strategies[s_type].items():
                strategy_by_name[strategy.name] = strategy

        # Determine which strategies have backtest results
        strategies_with_results = {res.get('strategy_name') for res in all_results.values()}

        # Populate the list, prioritizing backtested strategies
        for name, strategy in strategy_by_name.items():
            if name in strategies_with_results:
                item_text = f"[backtest] {name}"
            else:
                item_text = f"[{strategy.type}] {name}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, strategy) # Attach the object
            self.strategy_list.addItem(item)
        
        self._update_validation_tests_ui() # Update on initial population
        
    def _populate_datasets(self):
        """Populate dataset dropdown with available datasets"""
        self.dataset_combo.clear()
        self.dataset_combo.addItem("-- Select Dataset --")
        
        if self.parent_window and hasattr(self.parent_window, 'datasets'):
            for dataset_name in self.parent_window.datasets.keys():
                self.dataset_combo.addItem(dataset_name)
        # Auto-select last loaded dataset if available
        if self.parent_window and hasattr(self.parent_window, 'last_loaded_dataset'):
            last = self.parent_window.last_loaded_dataset
            idx = self.dataset_combo.findText(last)
            if idx > 0:
                self.dataset_combo.setCurrentIndex(idx)
                
    def _on_dataset_selected(self, idx):
        """Set date pickers to the dataset's saved date range if available, otherwise use actual data range."""
        if idx <= 0:
            return
        dataset_name = self.dataset_combo.currentText()
        if self.parent_window and hasattr(self.parent_window, 'datasets'):
            dataset_info = self.parent_window.datasets.get(dataset_name)
            if dataset_info and 'data' in dataset_info:
                data = dataset_info['data']
                metadata = dataset_info.get('metadata', {})
                
                # Try to get date range from metadata first
                date_range = None
                if hasattr(metadata, 'selected_date_range'):
                    date_range = metadata.selected_date_range
                elif isinstance(metadata, dict) and 'selected_date_range' in metadata:
                    date_range = metadata['selected_date_range']
                
                print(f"[DEBUG] Statistics: Dataset '{dataset_name}' date_range in metadata: {date_range}")
                
                # Try to parse metadata date range
                if date_range and len(date_range) == 2:
                    try:
                        # Parse the ISO format strings manually to avoid timezone issues
                        from PyQt6.QtCore import QDate, QTime, Qt
                        
                        # Parse start date
                        start_str = date_range[0]
                        if 'T' in start_str:
                            date_part, time_part = start_str.split('T')
                            time_part = time_part.split('.')[0]  # Remove microseconds if present
                        else:
                            date_part = start_str
                            time_part = "00:00:00"
                        
                        start_date_parts = date_part.split('-')
                        start_time_parts = time_part.split(':')
                        
                        start_date = QDate(int(start_date_parts[0]), int(start_date_parts[1]), int(start_date_parts[2]))
                        start_time = QTime(int(start_time_parts[0]), int(start_time_parts[1]), int(start_time_parts[2]))
                        from_date = QDateTime(start_date, start_time, Qt.TimeSpec.LocalTime)
                        
                        # Parse end date
                        end_str = date_range[1]
                        if 'T' in end_str:
                            date_part, time_part = end_str.split('T')
                            time_part = time_part.split('.')[0]  # Remove microseconds if present
                        else:
                            date_part = end_str
                            time_part = "00:00:00"
                        
                        end_date_parts = date_part.split('-')
                        end_time_parts = time_part.split(':')
                        
                        end_date = QDate(int(end_date_parts[0]), int(end_date_parts[1]), int(end_date_parts[2]))
                        end_time = QTime(int(end_time_parts[0]), int(end_time_parts[1]), int(end_time_parts[2]))
                        to_date = QDateTime(end_date, end_time, Qt.TimeSpec.LocalTime)
                        
                        print(f"[DEBUG] Statistics: Parsed from_date: {from_date.toString(Qt.DateFormat.ISODate)}, valid: {from_date.isValid()}")
                        print(f"[DEBUG] Statistics: Parsed to_date: {to_date.toString(Qt.DateFormat.ISODate)}, valid: {to_date.isValid()}")
                        
                        # Only use metadata dates if they're valid
                        if from_date.isValid() and to_date.isValid():
                            self.start_date.setDateTime(from_date)
                            self.end_date.setDateTime(to_date)
                            print(f"[DEBUG] Statistics: Set date range from metadata: {from_date.toString(Qt.DateFormat.ISODate)} to {to_date.toString(Qt.DateFormat.ISODate)}")
                            return
                    except Exception as e:
                        print(f"[DEBUG] Statistics: Exception parsing date range: {e}")
                
                # If metadata date range is not available or invalid, use actual data range
                if isinstance(data.index, pd.DatetimeIndex):
                    actual_start = data.index.min()
                    actual_end = data.index.max()
                    
                    # Convert pandas timestamps to QDateTime using QDate and QTime to avoid timezone issues
                    from PyQt6.QtCore import QDate, QTime, Qt
                    
                    # Extract date and time components
                    start_date = QDate(actual_start.year, actual_start.month, actual_start.day)
                    start_time = QTime(actual_start.hour, actual_start.minute, actual_start.second)
                    from_date = QDateTime(start_date, start_time, Qt.TimeSpec.LocalTime)
                    
                    end_date = QDate(actual_end.year, actual_end.month, actual_end.day)
                    end_time = QTime(actual_end.hour, actual_end.minute, actual_end.second)
                    to_date = QDateTime(end_date, end_time, Qt.TimeSpec.LocalTime)
                    
                    if from_date.isValid() and to_date.isValid():
                        self.start_date.setDateTime(from_date)
                        self.end_date.setDateTime(to_date)
                        print(f"[DEBUG] Statistics: Set date range from actual data: {from_date.toString(Qt.DateFormat.ISODate)} to {to_date.toString(Qt.DateFormat.ISODate)}")
                    else:
                        print(f"[DEBUG] Statistics: Failed to create valid QDateTime from actual data dates: {actual_start} to {actual_end}")
                else:
                    print(f"[DEBUG] Statistics: Dataset does not have DatetimeIndex, cannot set date range automatically")
                
    def refresh_data(self):
        """Refresh strategy and dataset lists"""
        self._populate_strategies()
        self._populate_datasets()
        
    def _create_probability_tab(self) -> QWidget:
        """Create probability analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Probability method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Calculation Method:"))
        
        self.prob_method = QComboBox()
        self.prob_method.addItems(['Bayesian', 'Frequency', 'Monte Carlo', 'Conditional'])
        method_layout.addWidget(self.prob_method)
        
        layout.addLayout(method_layout)
        
        # Results display
        self.prob_results = QTextEdit()
        self.prob_results.setReadOnly(True)
        layout.addWidget(self.prob_results)
        
        # Probability chart
        self.prob_chart = pg.PlotWidget()
        self.prob_chart.setLabel('left', 'Probability')
        self.prob_chart.setLabel('bottom', 'Strategy')
        layout.addWidget(self.prob_chart)
        
        widget.setLayout(layout)
        return widget
        
    def _create_validation_tab(self) -> QWidget:
        """Create validation tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # The group box now uses the pre-initialized layout from __init__
        group = QGroupBox("Statistical Tests")
        group.setLayout(self.test_layout)
        layout.addWidget(group)
        
        # Validation results
        self.validation_results = QTextEdit()
        self.validation_results.setReadOnly(True)
        layout.addWidget(self.validation_results)
        
        # Test statistics table
        self.test_table = QTableWidget()
        self.test_table.setColumnCount(5)
        self.test_table.setHorizontalHeaderLabels(['Strategy', 'Test', 'Statistic', 'P-Value', 'Result'])
        self.test_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.test_table)
        
        widget.setLayout(layout)
        return widget
        
    def _create_acceptance_tab(self) -> QWidget:
        """Create acceptance scoring tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Criteria weights
        weights_group = QGroupBox("Criteria Weights")
        weights_layout = QFormLayout()
        
        self.weight_inputs = {}
        default_weights = {
            'probability': 0.3,
            'sharpe_ratio': 0.25,
            'max_drawdown': 0.2,
            'consistency': 0.15,
            'sample_size': 0.1
        }
        
        for criterion, default in default_weights.items():
            spin = QDoubleSpinBox()
            spin.setRange(0, 1)
            spin.setValue(default)
            spin.setSingleStep(0.05)
            weights_layout.addRow(f"{criterion.replace('_', ' ').title()}:", spin)
            self.weight_inputs[criterion] = spin
            
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # Acceptance results
        self.acceptance_results = QTextEdit()
        self.acceptance_results.setReadOnly(True)
        layout.addWidget(self.acceptance_results)
        
        # Score visualization
        self.score_chart = pg.PlotWidget()
        self.score_chart.setLabel('left', 'Score')
        self.score_chart.setLabel('bottom', 'Strategy')
        layout.addWidget(self.score_chart)
        
        widget.setLayout(layout)
        return widget
        
    def _create_correlation_tab(self) -> QWidget:
        """Create correlation analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Correlation settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Correlation Window:"))
        
        self.corr_window = QSpinBox()
        self.corr_window.setRange(10, 500)
        self.corr_window.setValue(50)
        settings_layout.addWidget(self.corr_window)
        
        settings_layout.addWidget(QLabel("bars"))
        settings_layout.addStretch()
        
        layout.addLayout(settings_layout)
        
        # Correlation matrix
        self.corr_table = QTableWidget()
        layout.addWidget(QLabel("Correlation Matrix:"))
        layout.addWidget(self.corr_table)
        
        # Correlation heatmap
        self.corr_heatmap_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(QLabel("Correlation Heatmap:"))
        layout.addWidget(self.corr_heatmap_widget)
        
        widget.setLayout(layout)
        return widget
        
    def _run_analysis(self):
        """Run selected analyses, either on-the-fly or with historical data."""
        selected_items = self.strategy_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one strategy to analyze.")
            return
            
        dataset_name = self.dataset_combo.currentText()
        run_live_backtest = dataset_name != "-- Select Dataset --"

        # This dictionary will hold the results used for the analysis.
        analysis_results = {}
        selected_strategies = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]

        if run_live_backtest:
            # --- LIVE ANALYSIS ---
            dataset_info = self.parent_window.datasets.get(dataset_name)
            if not dataset_info or 'data' not in dataset_info:
                QMessageBox.critical(self, "Error", f"Could not load data for '{dataset_name}'.")
                return
            
            data = dataset_info['data']
            
            # Apply timeframe filtering
            data = self._filter_data_by_timeframe(data)
            if data is None or len(data) == 0:
                QMessageBox.warning(self, "No Data", "No data available for the selected timeframe range.")
                return
            
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            try:
                engine = BacktestEngine()
                for strategy in selected_strategies:
                    if not strategy: continue
                    # Run a temporary backtest.
                    temp_results = engine.run_backtest(strategy, data.copy())
                    analysis_results[strategy.name] = temp_results
                
                info_message = f"Live analysis on '{dataset_name}' complete."
            finally:
                QApplication.restoreOverrideCursor()

        else:
            # --- HISTORICAL ANALYSIS ---
            for strategy in selected_strategies:
                if not strategy: continue
                # Find the most recent saved result for this strategy.
                for result_id, result in reversed(list(self.parent_window.results.items())):
                    if result.get('strategy_name') == strategy.name:
                        analysis_results[strategy.name] = result
                        break # Found most recent
            
            info_message = "Historical analysis complete."

        if not analysis_results:
            QMessageBox.warning(self, "No Data", "Could not find or generate any results for the selected strategies.")
            return

        # Clear previous results and run all analyses with the new data
        self.current_results = analysis_results
        self._run_probability_analysis(analysis_results)
        self._run_validation_tests(analysis_results)
        self._run_acceptance_scoring(analysis_results)
        self._run_correlation_analysis(analysis_results)
        
        QMessageBox.information(self, "Analysis Complete", info_message)
        
    def _get_selected_strategy_objects(self):
        """Helper to get strategy objects from list widget selection."""
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.strategy_list.selectedItems() if item.data(Qt.ItemDataRole.UserRole)]
        
    def _filter_data_by_timeframe(self, data):
        """Filter data to the selected timeframe range, using the same logic as backtester."""
        if not isinstance(data.index, pd.DatetimeIndex):
            print("[DEBUG] Statistics: Data does not have DatetimeIndex, cannot filter by timeframe")
            return data
            
        # Get the selected date range
        start_dt = pd.to_datetime(self.start_date.dateTime().toString(Qt.DateFormat.ISODate))
        end_dt = pd.to_datetime(self.end_date.dateTime().toString(Qt.DateFormat.ISODate))
        
        # Get actual data range
        actual_start = data.index.min()
        actual_end = data.index.max()
        
        print(f"[DEBUG] Statistics: Selected timeframe: {start_dt} to {end_dt}")
        print(f"[DEBUG] Statistics: Actual data range: {actual_start} to {actual_end}")
        
        # Check if the selected range overlaps with available data
        if start_dt > actual_end or end_dt < actual_start:
            print(f"[DEBUG] Statistics: WARNING: Selected timeframe ({start_dt} to {end_dt}) is outside dataset range!")
            print(f"[DEBUG] Statistics: Using full dataset range instead: {actual_start} to {actual_end}")
            start_dt = actual_start
            end_dt = actual_end
        
        # Filter the data
        before_filter = len(data)
        filtered_data = data[(data.index >= start_dt) & (data.index < end_dt)]
        
        print(f"[DEBUG] Statistics: Filtered to timeframe: {start_dt} to {end_dt} ({before_filter} -> {len(filtered_data)} bars)")
        
        if len(filtered_data) == 0:
            print(f"[DEBUG] Statistics: ERROR: No data available for the selected timeframe range")
            return None
            
        return filtered_data

    def _run_probability_analysis(self, source_results: Dict[str, Any]):
        """Run probability analysis using the provided results data."""
        method = self.prob_method.currentText().lower()
        
        results_text = f"Probability Analysis ({method} method):\n\n"
        
        probabilities = []
        labels = []
        
        for strategy_name, result in source_results.items():
            prob = result.get('win_rate', 0.5)
            conf_low = max(0, prob - 0.1) # Wider mock interval
            conf_high = min(1, prob + 0.1)
            
            results_text += f"{strategy_name}:\n"
            results_text += f"  Probability: {prob:.2%}\n"
            results_text += f"  95% CI: [{conf_low:.2%}, {conf_high:.2%}]\n\n"
            
            probabilities.append(prob)
            labels.append(strategy_name)
            
        self.prob_results.setText(results_text)
        
        # Update chart
        self.prob_chart.clear()
        if probabilities:
            x = np.arange(len(probabilities))
            self.prob_chart.plot(x, probabilities, pen=None, symbol='o', symbolSize=10, 
                               symbolBrush=pg.mkBrush(100, 150, 255, 150))
        
        # Add labels
        ax = self.prob_chart.getAxis('bottom')
        ax.setTicks([[(i, labels[i]) for i in range(len(labels))]])
        
        # Set y-axis range
        self.prob_chart.setYRange(0, 1)
        
    def _run_validation_tests(self, source_results: Dict[str, Any]):
        """Run validation tests on single or multiple strategies."""
        self.test_table.setRowCount(0)
        self.validation_results.clear()

        num_strategies = len(source_results)

        if num_strategies == 0:
            self.validation_results.setText("Please select at least one strategy to validate.")
            self.test_layout.parentWidget().setVisible(False)
            return

        self.test_layout.parentWidget().setVisible(True)

        strategy_data = {}
        for name, result in source_results.items():
            trades = result.get('trades', [])
            if trades:
                # Use non-zero PNL for more meaningful stats
                returns = pd.Series([t['pnl'] for t in trades if t['pnl'] != 0])
            else:
                returns = pd.Series([])
            strategy_data[name] = {'returns': returns}

        if num_strategies == 1:
            self._validate_single_strategy(source_results, strategy_data)
        else:
            self._validate_multiple_strategies(source_results, strategy_data)
            
        self.test_table.resizeColumnsToContents()

    def _validate_single_strategy(self, source_results, strategy_data):
        """Run validation tests for a single selected strategy."""
        self.test_table.setColumnCount(5)
        self.test_table.setHorizontalHeaderLabels(['Strategy', 'Test', 'Statistic', 'P-Value', 'Result'])
        
        s_name = list(source_results.keys())[0]
        returns = strategy_data[s_name]['returns']
        results_text = f"Single Strategy Validation for: {s_name}\n\n"

        if len(returns) < 8:
            self.validation_results.setText(results_text + "Not enough trades to perform validation.")
            return

        # Test 1: One-sample T-test (are returns significantly positive?)
        if self.test_checkboxes.get('positive_mean', QCheckBox()).isChecked():
            t_stat, p_val = stats.ttest_1samp(returns, 0, alternative='greater')
            result = {
                'test': 'Positive Mean Returns (T-test)', 'statistic': t_stat, 
                'p_value': p_val, 'passed': p_val < 0.05
            }
            self._add_single_validation_row(s_name, result)
            results_text += self._interpret_single_test(s_name, result)

        # Test 2: Sharpe Ratio Z-test
        if self.test_checkboxes.get('sharpe_z_test', QCheckBox()).isChecked():
            # Simplified Z-test for Sharpe Ratio > 0
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            n = len(returns)
            z_stat = sharpe * np.sqrt(n)
            p_val = 1 - stats.norm.cdf(z_stat)
            result = {
                'test': 'Sharpe Ratio > 0 (Z-test)', 'statistic': z_stat,
                'p_value': p_val, 'passed': p_val < 0.05
            }
            self._add_single_validation_row(s_name, result)
            results_text += self._interpret_single_test(s_name, result)

        # Test 3: Shapiro-Wilk test for normality of returns
        if self.test_checkboxes.get('normality', QCheckBox()).isChecked():
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            result = {
                'test': 'Normality of Returns (Shapiro-Wilk)', 'statistic': shapiro_stat, 
                'p_value': shapiro_p, 'passed': shapiro_p > 0.05 # Null is that it IS normal
            }
            self._add_single_validation_row(s_name, result)
            results_text += self._interpret_single_test(s_name, result)

        self.validation_results.setText(results_text)

    def _validate_multiple_strategies(self, source_results, strategy_data):
        """Run pairwise validation tests for multiple strategies."""
        self.test_table.setColumnCount(6)
        self.test_table.setHorizontalHeaderLabels(['Strategy 1', 'Strategy 2', 'Test', 'Statistic', 'P-Value', 'Result'])
        
        validator = StatisticalValidator()
        strategy_names = list(source_results.keys())
        results_text = "Statistical Validation Results (Pairwise Comparison):\n\n"

        for s1_name, s2_name in itertools.combinations(strategy_names, 2):
            data1_returns = strategy_data[s1_name]['returns']
            data2_returns = strategy_data[s2_name]['returns']
            results_text += f"--- Comparison: {s1_name} vs. {s2_name} ---\n"
            
            # --- Independence Test ---
            if self.test_checkboxes.get('independence', QCheckBox()).isChecked() and len(data1_returns) > 1 and len(data2_returns) > 1:
                # Align series for correlation calculation
                min_len = min(len(data1_returns), len(data2_returns))
                corr, p_val = stats.pearsonr(data1_returns[:min_len], data2_returns[:min_len])
                result = {
                    'test': 'Returns Independence (Pearson r)', 'statistic': corr,
                    'p_value': p_val, 'passed': abs(corr) < 0.5
                }
                self._add_multi_validation_row(s1_name, s2_name, result)
                results_text += self._interpret_multi_test(s1_name, s2_name, result)

            # --- Performance Improvement Test (Welch's T-test) ---
            if self.test_checkboxes.get('performance_diff', QCheckBox()).isChecked() and len(data1_returns) > 1 and len(data2_returns) > 1:
                stat, p_val = stats.ttest_ind(data1_returns, data2_returns, equal_var=False)
                result = {
                    'test': 'Performance Difference (T-test)', 'statistic': stat,
                    'p_value': p_val, 'passed': p_val < 0.05
                }
                self._add_multi_validation_row(s1_name, s2_name, result)
                results_text += self._interpret_multi_test(s1_name, s2_name, result)
            results_text += "\n"
        self.validation_results.setText(results_text)

    def _interpret_single_test(self, s_name: str, result: Dict[str, Any]) -> str:
        """Generate a plain-English interpretation for a single-strategy test."""
        test_name = result['test']
        passed = result['passed']
        
        if 'Positive Mean' in test_name:
            if passed:
                return f"- The strategy's returns are **statistically positive**, suggesting a real profitable edge.\n"
            else:
                return f"- The strategy's returns are **not statistically positive**. Its profitability in the backtest could be due to chance.\n"
        if 'Sharpe' in test_name:
            if passed:
                return f"- The strategy's risk-adjusted performance is **statistically significant**.\n"
            else:
                return f"- The strategy's risk-adjusted performance is **not statistically significant**.\n"
        if 'Normality' in test_name:
            if passed:
                return f"- The returns **appear to be normally distributed**. This is unusual for financial data.\n"
            else:
                return f"- The returns are **not normally distributed**, which is typical and expected for trading strategies (e.g., 'fat tails').\n"
        return ""

    def _interpret_multi_test(self, s1_name: str, s2_name: str, result: Dict[str, Any]) -> str:
        """Generate a plain-English interpretation for a multi-strategy test."""
        test_name = result['test']
        passed = result['passed']

        if 'Independence' in test_name:
            if passed:
                return f"- The strategies' returns appear to be **independent**. Good for diversification.\n"
            else:
                return f"- The strategies' returns are **highly correlated**. They are redundant and not good for diversification.\n"
        if 'Performance Difference' in test_name:
            if passed:
                return f"- There is a **statistically significant difference** in performance between the two strategies.\n"
            else:
                return f"- There is **no statistically significant difference** in performance. Neither can be considered superior.\n"
        return ""

    def _add_single_validation_row(self, s_name: str, test_result: Dict[str, Any]):
        """Helper to add a row to the single-strategy validation table."""
        row = self.test_table.rowCount()
        self.test_table.insertRow(row)
        
        self.test_table.setItem(row, 0, QTableWidgetItem(s_name))
        self.test_table.setItem(row, 1, QTableWidgetItem(test_result.get('test', 'N/A')))
        self.test_table.setItem(row, 2, QTableWidgetItem(f"{test_result.get('statistic', 0):.4f}"))
        self.test_table.setItem(row, 3, QTableWidgetItem(f"{test_result.get('p_value', 0):.4f}"))
        
        passed = test_result.get('passed', False)
        result_item = QTableWidgetItem('PASS' if passed else 'FAIL')
        result_item.setBackground(QColor("#AED581") if passed else QColor("#E57373"))
        self.test_table.setItem(row, 4, result_item)

    def _add_multi_validation_row(self, s1_name: str, s2_name: str, test_result: Dict[str, Any]):
        """Helper to add a row to the validation results table."""
        row = self.test_table.rowCount()
        self.test_table.insertRow(row)
        
        self.test_table.setItem(row, 0, QTableWidgetItem(s1_name))
        self.test_table.setItem(row, 1, QTableWidgetItem(s2_name))
        self.test_table.setItem(row, 2, QTableWidgetItem(test_result.get('test', 'N/A')))
        self.test_table.setItem(row, 3, QTableWidgetItem(f"{test_result.get('statistic', 0):.4f}"))
        self.test_table.setItem(row, 4, QTableWidgetItem(f"{test_result.get('p_value', 0):.4f}"))
        
        passed = test_result.get('passed', False)
        result_item = QTableWidgetItem('PASS' if passed else 'FAIL')
        result_item.setBackground(QColor("#AED581") if passed else QColor("#E57373"))
        self.test_table.setItem(row, 5, result_item)

    def _run_acceptance_scoring(self, source_results: Dict[str, Any]):
        """Run acceptance scoring using the provided results data."""
        # Get weights
        weights = {k: v.value() for k, v in self.weight_inputs.items()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0: return # Avoid division by zero
        weights = {k: v/total_weight for k, v in weights.items()}
        
        results_text = "Acceptance Scoring Results:\n\n"
        
        # Calculate scores for each strategy from the source results
        strategy_scores = {}
        for strategy_name, result in source_results.items():
            scores = {
                'probability': result.get('win_rate', 0.5),
                'sharpe_ratio': min(1.0, max(0.0, (result.get('sharpe_ratio', 0.0) + 1) / 3)),
                'max_drawdown': max(0.0, 1.0 - abs(result.get('max_drawdown', 0.0))),
                'consistency': result.get('profit_factor', 1.0) / 3,
                'sample_size': min(1.0, result.get('total_trades', 0) / 100)
            }
            strategy_scores[strategy_name] = scores
        
        # Calculate overall scores
        overall_scores = {}
        for strategy_name, scores in strategy_scores.items():
            weighted_score = sum(scores[k] * weights[k] for k in weights)
            overall_scores[strategy_name] = weighted_score
        
        # Display results
        if overall_scores:
            best_strategy = max(overall_scores.items(), key=lambda x: x[1])
            results_text += f"Best Strategy: {best_strategy[0]} (Score: {best_strategy[1]:.2f})\n\n"
            
            results_text += "Strategy Scores:\n"
            for strategy_name, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
                status = 'ACCEPTED' if score >= 0.6 else 'REJECTED'
                results_text += f"  {strategy_name}: {score:.2f}/1.00 ({status})\n"
                
                # Show breakdown for best strategy
                if strategy_name == best_strategy[0]:
                    results_text += f"    Breakdown:\n"
                    for criterion, weight in weights.items():
                        criterion_score = strategy_scores[strategy_name][criterion]
                        results_text += f"      {criterion.replace('_', ' ').title()}: {criterion_score:.2f} (weight: {weight:.2f})\n"
            
        self.acceptance_results.setText(results_text)
        
        # Update chart
        self.score_chart.clear()
        if overall_scores:
            x = np.arange(len(overall_scores))
            y = list(overall_scores.values())
            labels = list(overall_scores.keys())
            
            # Create bars
            bars = pg.BarGraphItem(x=x, height=y, width=0.6, brush=pg.mkBrush(100, 150, 255, 150))
            self.score_chart.addItem(bars)
            
            # Add threshold line
            threshold = 0.6
            self.score_chart.addLine(y=threshold, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        
        # Add labels
        ax = self.score_chart.getAxis('bottom')
        ax.setTicks([[(i, labels[i][:10] + '...' if len(labels[i]) > 10 else labels[i]) for i in range(len(labels))]])
        
        # Set y-axis range
        self.score_chart.setYRange(0, 1)
        
    def _run_correlation_analysis(self, source_results: Dict[str, Any]):
        """Run correlation analysis on the equity curves of the backtest results."""
        if len(source_results) < 2:
            self.corr_table.setRowCount(0)
            self.corr_table.setColumnCount(0)
            self.corr_heatmap_widget.clear()
            return # Not enough data to correlate

        strategy_names = list(source_results.keys())
        equity_curves = []

        # Extract equity curves and align them
        for name in strategy_names:
            result = source_results[name]
            if 'equity_curve' in result:
                equity_curves.append(pd.Series(result['equity_curve']))

        if len(equity_curves) < 2:
            return # Not enough valid data

        # Combine into a single DataFrame to handle different lengths
        equity_df = pd.concat(equity_curves, axis=1, keys=strategy_names)
        equity_df.ffill(inplace=True) # Forward fill to align curves
        
        # Calculate returns to get a stationary series for correlation
        returns_df = equity_df.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr().values
        
        # Handle NaN values that can occur with zero-variance series (no trades)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Ensure the diagonal is always 1.0, as a strategy is perfectly correlated with itself.
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Update table
        n = len(strategy_names)
        self.corr_table.setRowCount(n)
        self.corr_table.setColumnCount(n)
        
        self.corr_table.setHorizontalHeaderLabels(strategy_names)
        self.corr_table.setVerticalHeaderLabels(strategy_names)
        self.corr_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.corr_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{corr_matrix[i, j]:.3f}")
                
                # Color code based on correlation strength
                val = corr_matrix[i, j]
                if i == j:
                    item.setBackground(QColor("#cccccc"))
                elif abs(val) > 0.7:
                    item.setBackground(QColor("#E57373")) # Red
                elif abs(val) > 0.5:
                    item.setBackground(QColor("#FFB74D")) # Orange
                elif abs(val) > 0.3:
                    item.setBackground(QColor("#FFF176")) # Yellow
                else:
                    item.setBackground(QColor("#AED581")) # Green
                    
                self.corr_table.setItem(i, j, item)
                
        self._update_correlation_heatmap(corr_matrix, strategy_names)

    def _update_correlation_heatmap(self, corr_matrix, labels):
        """Update the correlation heatmap plot using GraphicsLayoutWidget."""
        self.corr_heatmap_widget.clear()
        
        # Add a plot item to the layout
        plot = self.corr_heatmap_widget.addPlot(row=0, col=0)
        
        # Add the image
        img = pg.ImageItem(image=corr_matrix)
        plot.addItem(img)
        
        # Add a color bar to the layout, using the backward-compatible 'colorMap'
        colormap = pg.colormap.get('viridis')
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=colormap)
        self.corr_heatmap_widget.addItem(bar, row=0, col=1)
        
        # Set axis labels
        ax_bottom = plot.getAxis('bottom')
        ax_left = plot.getAxis('left')
        ticks = [[(i, label) for i, label in enumerate(labels)]]
        ax_bottom.setTicks(ticks)
        ax_left.setTicks(ticks)
        
        # Add text overlays
        for i in range(len(labels)):
            for j in range(len(labels)):
                text_color = 'k' if 0.2 < corr_matrix[i, j] < 0.8 else 'w'
                text = pg.TextItem(
                    f"{corr_matrix[i, j]:.2f}",
                    anchor=(0.5, 0.5),
                    color=text_color
                )
                text.setPos(j + 0.5, i + 0.5)
                plot.addItem(text)
                
        plot.getViewBox().setAspectLocked(True)

    def _on_tab_changed(self, index):
        """Called when the user switches tabs."""
        if self.analysis_tabs.tabText(index) == "Statistical Validation":
            self._update_validation_tests_ui()

    def _update_validation_tests_ui(self):
        """Dynamically update the checkboxes in the validation tab based on selection."""
        # Clear existing checkboxes
        for i in reversed(range(self.test_layout.count())): 
            self.test_layout.itemAt(i).widget().setParent(None)
        self.test_checkboxes.clear()

        num_selected = len(self.strategy_list.selectedItems())
        
        if num_selected == 1:
            tests_to_show = self.single_strategy_tests
        else:
            tests_to_show = self.multi_strategy_tests
            
        for key, (name, is_checked) in tests_to_show.items():
            checkbox = QCheckBox(name)
            checkbox.setChecked(is_checked)
            self.test_layout.addWidget(checkbox)
            self.test_checkboxes[key] = checkbox
        
    def _export_results(self):
        """Export analysis results"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "HTML Files (*.html);;CSV Files (*.csv)"
        )
        
        if filepath:
            # Would implement export functionality
            QMessageBox.information(self, "Export", f"Results exported to {filepath}")
