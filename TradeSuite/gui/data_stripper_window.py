"""
gui/data_stripper_window.py
===========================
Window for data stripping and preprocessing
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime
import os

from processors.data_processor import DataStripper, MultiTimeframeProcessor, VolatilityCalculator
from processors.data_source_integration import EnhancedDataStripper, DataSourceDetector
from core.data_structures import TimeRange, DatasetMetadata, VolatilityProfile

logger = logging.getLogger(__name__)

RECENT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'recent_dataset')
RECENT_DATASET_PATH = os.path.join(RECENT_DATASET_DIR, 'most_recent.csv')

class DataStripperWindow(QMainWindow):
    """Window for stripping and processing data"""

    # Signals
    data_processed = pyqtSignal(str, pd.DataFrame, DatasetMetadata)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Stripper")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize components
        self.stripper = DataStripper()
        self.multi_timeframe_processor = MultiTimeframeProcessor()
        self.volatility_calc = VolatilityCalculator()

        # Data storage
        self.current_data = None
        self.processed_data = None
        self.resampled_data = {}
        self.current_metadata = None
        self.detected_source = None

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

        # File loading section
        file_section = self._create_file_section()
        layout.addWidget(file_section)

        # Source selection section
        source_section = self._create_source_section()
        layout.addWidget(source_section)

        # Column mapping section
        column_section = self._create_column_section()
        layout.addWidget(column_section)

        # Data preview
        self.data_preview = self._create_data_preview()
        layout.addWidget(self.data_preview)

        # Processing options
        processing_section = self._create_processing_section()
        layout.addWidget(processing_section)

        # Action buttons
        button_layout = QHBoxLayout()

        self.process_btn = QPushButton("Process Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)

        self.save_btn = QPushButton("Save Dataset")
        self.save_btn.clicked.connect(self.save_dataset)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)

        # Main layout for central widget
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(scroll_area)

        # Set size policies for resizable window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(800, 500)  # Minimum size to ensure usability

        # Status bar
        self.status_bar = self.statusBar()

    def _create_file_section(self) -> QGroupBox:
        """Create file loading section"""
        group = QGroupBox("Load Data File")
        layout = QHBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select CSV file...")
        layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        layout.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_file)
        layout.addWidget(load_btn)

        group.setLayout(layout)
        return group

    def _create_source_section(self) -> QGroupBox:
        """Create data source selection section"""
        group = QGroupBox("Data Source")
        layout = QVBoxLayout()

        # Auto-detect option
        detect_layout = QHBoxLayout()
        self.auto_detect_check = QCheckBox("Auto-detect source")
        self.auto_detect_check.setChecked(True)
        self.auto_detect_check.toggled.connect(self._on_auto_detect_changed)
        detect_layout.addWidget(self.auto_detect_check)

        self.detected_source_label = QLabel("Source: Not detected")
        detect_layout.addWidget(self.detected_source_label)
        detect_layout.addStretch()
        layout.addLayout(detect_layout)

        # Manual source selection
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Manual Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            'Auto',
            'Sierra Chart',
            'Zorro',
            'MetaTrader',
            'NinjaTrader',
            'Generic OHLC'
        ])
        self.source_combo.setEnabled(False)
        manual_layout.addWidget(self.source_combo)

        # Show column mapping button
        self.show_mapping_btn = QPushButton("Show Column Mapping")
        self.show_mapping_btn.clicked.connect(self._show_column_mapping)
        manual_layout.addWidget(self.show_mapping_btn)

        layout.addLayout(manual_layout)

        # Source-specific information
        self.source_info = QTextEdit()
        self.source_info.setMaximumHeight(80)
        self.source_info.setReadOnly(True)
        self.source_info.setPlaceholderText("Source-specific information will appear here...")
        layout.addWidget(self.source_info)

        group.setLayout(layout)
        return group

    def _create_column_section(self) -> QGroupBox:
        """Create column mapping section"""
        group = QGroupBox("Column Configuration")
        layout = QVBoxLayout()

        # Expected columns info
        info_text = QLabel("Expected columns will be mapped automatically based on source.")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)

        # Column selection
        selection_layout = QHBoxLayout()

        selection_layout.addWidget(QLabel("Columns to keep:"))
        self.columns_to_keep = QLineEdit()
        self.columns_to_keep.setPlaceholderText("e.g., open,high,low,close,volume (or leave empty for all)")
        selection_layout.addWidget(self.columns_to_keep)

        layout.addLayout(selection_layout)

        # Custom mapping
        mapping_layout = QHBoxLayout()

        mapping_layout.addWidget(QLabel("Custom mapping:"))
        self.custom_mapping = QLineEdit()
        self.custom_mapping.setPlaceholderText("e.g., Last:close,# of Trades:trades")
        mapping_layout.addWidget(self.custom_mapping)

        layout.addLayout(mapping_layout)

        group.setLayout(layout)
        return group

    def _create_data_preview(self) -> QGroupBox:
        """Create data preview section"""
        group = QGroupBox("Data Preview")
        layout = QVBoxLayout()

        # Table for preview
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        layout.addWidget(self.preview_table)

        # Statistics
        self.stats_label = QLabel("No data loaded")
        layout.addWidget(self.stats_label)

        group.setLayout(layout)
        return group

    def _create_processing_section(self) -> QGroupBox:
        """Create processing options section"""
        group = QGroupBox("Processing Options")
        layout = QVBoxLayout()

        # Multi-timeframe Processing
        mtf_layout = QHBoxLayout()

        self.enable_lcm = QCheckBox("Enable Multi-Timeframe Processing")
        self.enable_lcm.setChecked(True)
        mtf_layout.addWidget(self.enable_lcm)

        mtf_layout.addWidget(QLabel("Timeframes:"))
        self.timeframes_edit = QLineEdit()
        self.timeframes_edit.setPlaceholderText("e.g., 5s,1m,5m,15m,1h")
        self.timeframes_edit.setText("1m,5m,15m")
        mtf_layout.addWidget(self.timeframes_edit)

        layout.addLayout(mtf_layout)

        # Date range filter
        date_layout = QHBoxLayout()

        self.enable_date_filter = QCheckBox("Filter Date Range")
        date_layout.addWidget(self.enable_date_filter)

        date_layout.addWidget(QLabel("From:"))
        self.date_from = QDateTimeEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDateTime(QDateTime.currentDateTime().addDays(-30))
        date_layout.addWidget(self.date_from)

        date_layout.addWidget(QLabel("To:"))
        self.date_to = QDateTimeEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDateTime(QDateTime.currentDateTime())
        date_layout.addWidget(self.date_to)

        layout.addLayout(date_layout)

        # JSON date filter
        json_date_layout = QHBoxLayout()

        self.enable_json_filter = QCheckBox("Filter by JSON dates")
        json_date_layout.addWidget(self.enable_json_filter)

        self.json_file_edit = QLineEdit()
        self.json_file_edit.setPlaceholderText("Select JSON file with dates...")
        json_date_layout.addWidget(self.json_file_edit)

        self.json_browse_btn = QPushButton("Browse")
        self.json_browse_btn.clicked.connect(self.browse_json_file)
        json_date_layout.addWidget(self.json_browse_btn)

        layout.addLayout(json_date_layout)

        # Time period filter
        time_period_layout = QHBoxLayout()

        self.enable_time_filter = QCheckBox("Filter time periods")
        time_period_layout.addWidget(self.enable_time_filter)

        time_period_layout.addWidget(QLabel("From:"))
        self.time_from = QTimeEdit()
        self.time_from.setTime(QTime(7, 30))
        time_period_layout.addWidget(self.time_from)

        time_period_layout.addWidget(QLabel("To:"))
        self.time_to = QTimeEdit()
        self.time_to.setTime(QTime(8, 30))
        time_period_layout.addWidget(self.time_to)

        self.add_time_period_btn = QPushButton("Add Period")
        self.add_time_period_btn.clicked.connect(self.add_time_period)
        time_period_layout.addWidget(self.add_time_period_btn)

        layout.addLayout(time_period_layout)

        # Time periods list
        self.time_periods_list = QListWidget()
        self.time_periods_list.setMaximumHeight(60)
        layout.addWidget(self.time_periods_list)

        # Volatility calculation
        vol_layout = QHBoxLayout()

        vol_layout.addWidget(QLabel("Volatility Method:"))
        self.volatility_method = QComboBox()
        self.volatility_method.addItems(['atr', 'std', 'parkinson', 'garman_klass'])
        vol_layout.addWidget(self.volatility_method)

        vol_layout.addWidget(QLabel("Period:"))
        self.volatility_period = QSpinBox()
        self.volatility_period.setRange(5, 100)
        self.volatility_period.setValue(20)
        vol_layout.addWidget(self.volatility_period)

        layout.addLayout(vol_layout)

        # Dataset name
        name_layout = QHBoxLayout()

        name_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name = QLineEdit()
        self.dataset_name.setPlaceholderText("Enter dataset name...")
        name_layout.addWidget(self.dataset_name)

        layout.addLayout(name_layout)

        group.setLayout(layout)
        return group

    def _on_auto_detect_changed(self, checked: bool):
        """Handle auto-detect checkbox change"""
        self.source_combo.setEnabled(not checked)
        if checked:
            self.source_combo.setCurrentIndex(0)  # Set to Auto

    def _show_column_mapping(self):
        """Show column mapping for selected source"""
        source = self.source_combo.currentText().lower().replace(' ', '_')
        if source == 'auto' and self.detected_source:
            source = self.detected_source

        from processors.data_source_integration import DataSourceMapper
        mapping = DataSourceMapper.get_mapping(source)

        if mapping:
            mapping_text = "Column Mapping:\n"
            for original, standard in mapping.items():
                mapping_text += f"  {original} → {standard}\n"
        else:
            mapping_text = "No specific mapping defined for this source."

        QMessageBox.information(self, "Column Mapping", mapping_text)

    def browse_file(self):
        """Browse for data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv);;All Files (*.*)"
        )

        if file_path:
            self.file_path_edit.setText(file_path)
            # Auto-generate dataset name
            import os
            base_name = os.path.basename(file_path).split('.')[0]
            self.dataset_name.setText(base_name)

    def browse_json_file(self):
        """Browse for JSON date filter file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            self.json_file_edit.setText(file_path)

    def add_time_period(self):
        """Add time period to filter list"""
        from_time = self.time_from.time().toString("HH:mm")
        to_time = self.time_to.time().toString("HH:mm")
        period = f"{from_time} - {to_time}"

        # Check if already exists
        for i in range(self.time_periods_list.count()):
            if self.time_periods_list.item(i).text() == period:
                return

        self.time_periods_list.addItem(period)

    def load_file(self):
        """Load selected file"""
        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a file")
            return

        try:
            # Detect source if auto-detect is enabled
            source = None
            if self.auto_detect_check.isChecked():
                detector = DataSourceDetector()
                self.detected_source = detector.detect_source(file_path)
                self.detected_source_label.setText(f"Source: {self.detected_source}")

                # Show source-specific info
                if self.detected_source == 'zorro':
                    self.source_info.setText(
                        "Detected Zorro export file.\n"
                        "Will extract patterns and indicators if present.\n"
                        "Date format: YYYY-MM-DD HH:MM:SS"
                    )
                elif self.detected_source == 'sierra_chart':
                    self.source_info.setText(
                        "Detected Sierra Chart export.\n"
                        "Will combine Date and Time columns.\n"
                        "Using 'Last' column as Close price."
                    )
                else:
                    self.source_info.setText(f"Detected {self.detected_source} format.")
            else:
                # Use manual selection
                source_text = self.source_combo.currentText()
                if source_text != 'Auto':
                    source = source_text.lower().replace(' ', '_')

            # Parse custom mapping if provided
            column_mapping = None
            if self.custom_mapping.text():
                column_mapping = {}
                for mapping in self.custom_mapping.text().split(','):
                    if ':' in mapping:
                        old, new = mapping.strip().split(':')
                        column_mapping[old.strip()] = new.strip()

            # Load data with enhanced loader
            self.current_data = self.stripper.load_data(file_path, column_mapping)

            # Use detected source for info
            source_info = {'source': self.detected_source or 'unknown'}

            # Show metadata if from Zorro
            if source_info['source'] == 'zorro':
                # For Zorro, we would need to check if there are pattern columns
                pattern_cols = [col for col in self.current_data.columns if 'pattern' in col.lower()]
                if pattern_cols:
                    self.source_info.append(f"\nFound patterns: {', '.join(pattern_cols)}")
                
                indicator_cols = [col for col in self.current_data.columns 
                                if any(ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd'])]
                if indicator_cols:
                    self.source_info.append(f"\nFound indicators: {', '.join(indicator_cols)}")

            # Update preview
            self._update_preview()

            # Enable processing
            self.process_btn.setEnabled(True)

            self.status_bar.showMessage(f"Loaded {len(self.current_data)} rows from {source_info['source']}")

            # After loading, save a copy to recent_dataset
            if not os.path.exists(RECENT_DATASET_DIR):
                os.makedirs(RECENT_DATASET_DIR)
            try:
                self.current_data.to_csv(RECENT_DATASET_PATH, index=False)
            except Exception as e:
                print(f"[WARNING] Could not save recent dataset: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
            logger.error(f"File load error: {e}")

    def _update_preview(self):
        """Update data preview"""
        if self.current_data is None:
            return

        # Show first 10 rows in table
        preview_data = self.current_data.head(10)

        self.preview_table.setRowCount(len(preview_data))
        self.preview_table.setColumnCount(len(preview_data.columns))
        self.preview_table.setHorizontalHeaderLabels(preview_data.columns.tolist())

        for i in range(len(preview_data)):
            for j, col in enumerate(preview_data.columns):
                value = str(preview_data.iloc[i][col])
                self.preview_table.setItem(i, j, QTableWidgetItem(value))

        # Update statistics
        stats_text = f"""
Rows: {len(self.current_data)}
Columns: {len(self.current_data.columns)}
Date Range: {self.current_data.index.min()} to {self.current_data.index.max()}
"""
        self.stats_label.setText(stats_text.strip())

    def process_data(self):
        """Process the loaded data and resample to all requested timeframes"""
        if self.current_data is None:
            return

        try:
            self.processed_data = self.current_data.copy()
            metadata = DatasetMetadata(
                name=self.dataset_name.text() or "processed_data",
                rows_original=len(self.current_data)
            )
            # Strip columns if specified
            if self.columns_to_keep.text():
                columns = [col.strip() for col in self.columns_to_keep.text().split(',')]
                # Only keep columns that exist
                columns = [col for col in columns if col in self.processed_data.columns]
                if columns:
                    # Create a simple DataStripper for column operations
                    from processors.data_processor import DataStripper
                    basic_stripper = DataStripper()
                    basic_stripper.processed_data = self.processed_data
                    basic_stripper.metadata = metadata
                    self.processed_data = basic_stripper.strip_columns(columns)
                    metadata = basic_stripper.metadata

            # Apply date filter if enabled
            if self.enable_date_filter.isChecked():
                date_from = self.date_from.dateTime().toPyDateTime()
                date_to = self.date_to.dateTime().toPyDateTime()

                mask = (self.processed_data.index >= date_from) & (self.processed_data.index <= date_to)
                self.processed_data = self.processed_data[mask]

            # Apply JSON date filter if enabled
            if self.enable_json_filter.isChecked() and self.json_file_edit.text():
                try:
                    with open(self.json_file_edit.text(), 'r') as f:
                        date_list = json.load(f)

                    # Convert MM/DD/YYYY to datetime
                    valid_dates = []
                    for date_str in date_list:
                        try:
                            dt = pd.to_datetime(date_str, format='%m/%d/%Y')
                            valid_dates.append(dt.date())
                        except:
                            pass

                    if valid_dates:
                        # Filter data to only include these dates
                        self.processed_data = self.processed_data[
                            self.processed_data.index.date.isin(valid_dates)
                        ]
                except Exception as e:
                    logger.error(f"Error loading JSON dates: {e}")

            # Apply time period filter if enabled
            if self.enable_time_filter.isChecked() and self.time_periods_list.count() > 0:
                # Ensure index is DatetimeIndex before time filter
                if not isinstance(self.processed_data.index, pd.DatetimeIndex):
                    self.processed_data.index = pd.to_datetime(self.processed_data.index)
                time_masks = []

                for i in range(self.time_periods_list.count()):
                    period = self.time_periods_list.item(i).text()
                    from_time, to_time = period.split(' - ')
                    from_h, from_m = map(int, from_time.split(':'))
                    to_h, to_m = map(int, to_time.split(':'))

                    # Create time mask
                    time_mask = (
                        (self.processed_data.index.hour > from_h) |
                        ((self.processed_data.index.hour == from_h) &
                         (self.processed_data.index.minute >= from_m))
                    ) & (
                        (self.processed_data.index.hour < to_h) |
                        ((self.processed_data.index.hour == to_h) &
                         (self.processed_data.index.minute <= to_m))
                    )
                    time_masks.append(time_mask)

                # Combine all time masks with OR
                if time_masks:
                    combined_mask = time_masks[0]
                    for mask in time_masks[1:]:
                        combined_mask = combined_mask | mask
                    self.processed_data = self.processed_data[combined_mask]

            # Apply multi-timeframe processing if enabled
            if self.enable_lcm.isChecked() and self.timeframes_edit.text():
                try:
                    # Parse timeframes
                    timeframes = []
                    for tf_str in self.timeframes_edit.text().split(','):
                        tf_str = tf_str.strip()
                        if tf_str:
                            # Extract value and unit
                            import re
                            match = re.match(r'(\d+)([smhd])', tf_str)
                            if match:
                                value = int(match.group(1))
                                unit = match.group(2)
                                timeframes.append(TimeRange(value, unit))

                    if timeframes:
                        logger.info(f"Creating multi-timeframe datasets with {len(timeframes)} timeframes")
                        logger.info(f"Data index type: {type(self.processed_data.index)}")
                        logger.info(f"Data shape: {self.processed_data.shape}")
                        logger.info(f"Columns before processing: {list(self.processed_data.columns)}")
                        
                        # Ensure data has a proper datetime index
                        if not isinstance(self.processed_data.index, pd.DatetimeIndex):
                            logger.warning("Converting index to DatetimeIndex")
                            self.processed_data.index = pd.to_datetime(self.processed_data.index)
                        
                        # Create timeframe datasets without filtering original data
                        self.resampled_data = self.multi_timeframe_processor.create_timeframe_datasets(
                            self.processed_data,
                            timeframes
                        )
                        
                        # Use the highest resolution timeframe as the main processed data
                        if self.resampled_data:
                            # Find highest resolution (most rows)
                            highest_res_tf = max(self.resampled_data.keys(), 
                                               key=lambda x: len(self.resampled_data[x]))
                            self.processed_data = self.resampled_data[highest_res_tf]
                            logger.info(f"Using {highest_res_tf} as main timeframe with {len(self.processed_data)} rows")
                        
                        logger.info(f"Created {len(self.resampled_data)} timeframe datasets")
                except Exception as e:
                    logger.error(f"Multi-timeframe processing error: {e}")
                    QMessageBox.warning(self, "Warning", f"Multi-timeframe processing failed: {str(e)}")

            # Calculate volatility
            logger.info(f"Calculating volatility with columns: {list(self.processed_data.columns)}")
            volatility_profile = self.volatility_calc.calculate_volatility(
                self.processed_data,
                self.volatility_method.currentText(),
                self.volatility_period.value()
            )

            # Add source information
            metadata.tags.append(f"source:{self.stripper.source if hasattr(self.stripper, 'source') else 'unknown'}")
            if self.detected_source:
                metadata.tags.append(f"detected:{self.detected_source}")

            # Update metadata with processing info
            metadata.rows_processed = len(self.processed_data)
            metadata.volatility = volatility_profile

            # Store selected date range in metadata
            if hasattr(self, 'date_from') and hasattr(self, 'date_to'):
                metadata.selected_date_range = (
                    self.date_from.dateTime().toPyDateTime().isoformat(),
                    self.date_to.dateTime().toPyDateTime().isoformat()
                )

            # Store metadata for save
            self.current_metadata = metadata

            # Show results
            result_text = f"""
                                    Processing Complete!
                                    
                                    Original rows: {len(self.current_data)}
                                    Processed rows: {len(self.processed_data)}
                                    Reduction: {(1 - len(self.processed_data)/len(self.current_data))*100:.1f}%
                                    
                                    Volatility: {volatility_profile.value} ({volatility_profile.category})
                                    """
            
            # Add multi-timeframe information if available
            if hasattr(self, 'resampled_data') and self.resampled_data:
                result_text += f"\n\nMulti-Timeframe Datasets Created:"
                for tf_str, df in self.resampled_data.items():
                    result_text += f"\n  {tf_str}: {len(df)} rows"
                
                if hasattr(self, 'multi_timeframe_processor') and self.multi_timeframe_processor.get_original_data() is not None:
                    original_rows = len(self.multi_timeframe_processor.get_original_data())
                    result_text += f"\n\nOriginal data preserved: {original_rows} rows"
                    result_text += f"\nNo data was filtered or lost!"

            QMessageBox.information(self, "Processing Complete", result_text)

            # Enable save
            self.save_btn.setEnabled(True)

            # --- PATCH: Robust DatetimeIndex creation before resampling ---
            if not isinstance(self.processed_data.index, pd.DatetimeIndex):
                cols = [c.strip().lower() for c in self.processed_data.columns]
                col_map = {c.strip().lower(): c for c in self.processed_data.columns}
                # Try 'datetime' column
                if 'datetime' in cols:
                    col = col_map['datetime']
                    self.processed_data[col] = pd.to_datetime(self.processed_data[col])
                    self.processed_data.set_index(col, inplace=True)
                # Try 'date' + 'time' columns
                elif 'date' in cols and 'time' in cols:
                    date_col = col_map['date']
                    time_col = col_map['time']
                    self.processed_data['datetime'] = pd.to_datetime(
                        self.processed_data[date_col].astype(str) + ' ' + self.processed_data[time_col].astype(str)
                    )
                    self.processed_data.set_index('datetime', inplace=True)
                # Try just 'date' column
                elif 'date' in cols:
                    date_col = col_map['date']
                    self.processed_data['datetime'] = pd.to_datetime(self.processed_data[date_col].astype(str))
                    self.processed_data.set_index('datetime', inplace=True)
                else:
                    # Print debug info for user
                    print('Available columns:', list(self.processed_data.columns))
                    print('First 5 rows:', self.processed_data.head())
                    QMessageBox.critical(self, "Error", f"No datetime information found for resampling!\nAvailable columns: {list(self.processed_data.columns)}\nPlease ensure your data has a 'datetime', or 'date' and 'time' columns.")
                    raise Exception('No datetime information found for resampling!')

            # --- PATCH: Resample to all requested timeframes ---
            self.resampled_data = {}  # Dict of {interval_str: DataFrame}
            timeframes = []
            if self.timeframes_edit.text():
                import re
                for tf_str in self.timeframes_edit.text().split(','):
                    tf_str = tf_str.strip()
                    if tf_str:
                        match = re.match(r'(\d+)([smhd])', tf_str)
                        if match:
                            value = int(match.group(1))
                            unit = match.group(2)
                            timeframes.append((tf_str, value, unit))
            if timeframes:
                for tf_str, value, unit in timeframes:
                    resample_map = {'s': 'S', 'm': 'T', 'h': 'H', 'd': 'D'}
                    resample_str = f"{value}{resample_map.get(unit, unit)}"
                    df_resampled = self.processed_data.resample(resample_str).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    self.resampled_data[tf_str] = df_resampled
                # Default to 1st interval for preview and downstream
                first_tf = timeframes[0][0]
                self.processed_data = self.resampled_data[first_tf]
            else:
                self.resampled_data = {'raw': self.processed_data}

            # Update preview with processed data
            self.current_data = self.processed_data
            self._update_preview()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
            logger.error(f"Processing error: {e}")

    def save_dataset(self):
        """Save all timeframe datasets and register them in the workspace manager"""
        if not hasattr(self, 'resampled_data') or not self.resampled_data:
            QMessageBox.warning(self, "Warning", "No multi-timeframe data to save.")
            return
        
        try:
            # Get original data from multi-timeframe processor
            original_data = None
            if hasattr(self, 'multi_timeframe_processor'):
                original_data = self.multi_timeframe_processor.get_original_data()
            
            if original_data is None:
                QMessageBox.warning(self, "Warning", "Original data not available for multi-timeframe saving.")
                return
            
            # Convert timeframes to TimeRange objects
            from core.data_structures import TimeRange
            timeframes = []
            for tf_str in self.resampled_data.keys():
                import re
                match = re.match(r'(\d+)([smhd])', tf_str)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    timeframes.append(TimeRange(value, unit))
            
            # Save using workspace manager's multi-timeframe functionality
            if self.parent() and hasattr(self.parent(), "workspace_manager"):
                workspace_manager = self.parent().workspace_manager
                
                # Create metadata
                metadata = getattr(self, "current_metadata", {})
                if hasattr(metadata, 'to_dict'):
                    metadata = metadata.to_dict()
                elif not isinstance(metadata, dict):
                    metadata = dict(metadata)
                metadata.update({
                    'source': 'data_stripper',
                    'original_timeframes': [f"{tf.value}{tf.unit}" for tf in timeframes],
                    'created_at': datetime.now().isoformat(),
                    'original_rows': len(original_data)
                })
                
                # Save multi-timeframe dataset
                success = workspace_manager.save_multi_timeframe_dataset(
                    self.dataset_name.text(), 
                    original_data, 
                    timeframes, 
                    metadata
                )
                
                if success:
                    # Also register in parent window's datasets for immediate use
                    if hasattr(self.parent(), 'datasets'):
                        self.parent().datasets[self.dataset_name.text()] = {
                            'data': original_data,
                            'metadata': metadata
                        }
                    
                    QMessageBox.information(self, "Save Complete", 
                                          f"Multi-timeframe dataset '{self.dataset_name.text()}' saved successfully!\n\n"
                                          f"Timeframes created: {list(self.resampled_data.keys())}\n"
                                          f"Original data: {len(original_data)} rows\n"
                                          f"All datasets registered in workspace manager.")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save multi-timeframe dataset.")
            else:
                # Fallback to old method if no workspace manager
                saved_files = []
                for tf_str, df in self.resampled_data.items():
                    filename = f"{self.dataset_name.text()}_{tf_str}.csv"
                    df.to_csv(filename)
                    saved_files.append(f"{filename} ({len(df)} rows)")
                    logger.info(f"Saved {filename} with {len(df)} rows.")
                
                # Also save original data if available
                if original_data is not None:
                    original_filename = f"{self.dataset_name.text()}_original.csv"
                    original_data.to_csv(original_filename)
                    saved_files.append(f"{original_filename} ({len(original_data)} rows)")
                    logger.info(f"Saved original data: {original_filename} with {len(original_data)} rows.")
                
                files_text = "\n".join(saved_files)
                QMessageBox.information(self, "Save Complete", 
                                      f"Saved {len(saved_files)} files:\n\n{files_text}\n\n"
                                      f"Note: Multi-timeframe functionality not available.")
        
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save dataset: {e}")