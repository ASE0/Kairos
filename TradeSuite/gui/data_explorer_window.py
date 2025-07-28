"""
gui/data_explorer_window.py
==============================
Dataset explorer with search, filtering, and visualization
"""

import logging
logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Matplotlib imports with error handling
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.info("Warning: matplotlib not available. Visualization features will be disabled.")

from core.dataset_manager import DatasetManager, DatasetInfo, DatasetCombiner
from core.data_structures import ProbabilityMetrics, VolatilityProfile


class DatasetExplorerWindow(QMainWindow):
    """Window for exploring and managing datasets"""
    
    # Signals
    dataset_selected = pyqtSignal(str, pd.DataFrame, DatasetInfo)
    datasets_combined = pyqtSignal(str, pd.DataFrame, ProbabilityMetrics)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Explorer")
        self.setGeometry(150, 150, 1400, 900)
        
        # Dataset manager
        self.dataset_manager = DatasetManager()
        self.combiner = DatasetCombiner(self.dataset_manager)
        
        # Current selection
        self.selected_datasets = []
        self.current_data = None
        self.current_info = None
        
        # Setup UI
        self._setup_ui()
        
        # Load initial data
        self._refresh_datasets()
        
        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333333;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTableWidget {
                gridline-color: #e0e0e0;
                selection-background-color: #e3f2fd;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 8px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
        """)
        
    def _setup_ui(self):
        """Setup UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QHBoxLayout(central_widget)
        
        # Left panel - Search and filters
        left_panel = self._create_left_panel()
        layout.addWidget(left_panel, 1)
        
        # Center panel - Dataset list and details
        center_panel = self._create_center_panel()
        layout.addWidget(center_panel, 2)
        
        # Right panel - Data visualization
        right_panel = self._create_right_panel()
        layout.addWidget(right_panel, 1)
        
        # Setup toolbar
        self._setup_toolbar()
        
        # Status bar
        self.status_bar = self.statusBar()
        
    def _create_left_panel(self) -> QWidget:
        """Create left panel with search and filters"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Search section
        search_group = QGroupBox("Search & Filter")
        search_layout = QVBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search datasets by name, patterns, strategies...")
        self.search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(self.search_input)
        
        # Quick filters
        quick_filter_layout = QHBoxLayout()
        
        self.quick_accepted_btn = QPushButton("Accepted")
        self.quick_accepted_btn.setCheckable(True)
        self.quick_accepted_btn.clicked.connect(self._apply_quick_filter)
        quick_filter_layout.addWidget(self.quick_accepted_btn)
        
        self.quick_rejected_btn = QPushButton("Rejected")
        self.quick_rejected_btn.setCheckable(True)
        self.quick_rejected_btn.clicked.connect(self._apply_quick_filter)
        quick_filter_layout.addWidget(self.quick_rejected_btn)
        
        self.quick_pending_btn = QPushButton("Pending")
        self.quick_pending_btn.setCheckable(True)
        self.quick_pending_btn.clicked.connect(self._apply_quick_filter)
        quick_filter_layout.addWidget(self.quick_pending_btn)
        
        search_layout.addLayout(quick_filter_layout)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Advanced filters section
        filters_group = QGroupBox("Advanced Filters")
        filters_layout = QVBoxLayout()
        
        # Status filter
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(['All', 'Accepted', 'Rejected', 'Pending'])
        self.status_filter.currentTextChanged.connect(self._apply_filters)
        status_layout.addWidget(self.status_filter)
        filters_layout.addLayout(status_layout)
        
        # Pattern filter
        filters_layout.addWidget(QLabel("Patterns:"))
        self.pattern_list = QListWidget()
        self.pattern_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.pattern_list.setMaximumHeight(100)
        self.pattern_list.itemSelectionChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.pattern_list)
        
        # Timeframe filter
        filters_layout.addWidget(QLabel("Timeframes:"))
        self.timeframe_list = QListWidget()
        self.timeframe_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.timeframe_list.setMaximumHeight(100)
        self.timeframe_list.itemSelectionChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.timeframe_list)
        
        # Location strategy filter
        filters_layout.addWidget(QLabel("Location Strategies:"))
        self.location_list = QListWidget()
        self.location_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.location_list.setMaximumHeight(100)
        self.location_list.itemSelectionChanged.connect(self._apply_filters)
        filters_layout.addWidget(self.location_list)
        
        # Probability filter
        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Min Probability:"))
        self.min_prob_spin = QDoubleSpinBox()
        self.min_prob_spin.setRange(0, 1)
        self.min_prob_spin.setSingleStep(0.05)
        self.min_prob_spin.setValue(0)
        self.min_prob_spin.setSuffix(" %")
        self.min_prob_spin.setDecimals(1)
        self.min_prob_spin.valueChanged.connect(self._apply_filters)
        prob_layout.addWidget(self.min_prob_spin)
        filters_layout.addLayout(prob_layout)
        
        # Date filter
        self.date_filter_check = QCheckBox("Filter by date")
        self.date_filter_check.toggled.connect(self._apply_filters)
        filters_layout.addWidget(self.date_filter_check)
        
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("After:"))
        self.date_after = QDateEdit()
        self.date_after.setCalendarPopup(True)
        self.date_after.setDate(QDate.currentDate().addDays(-30))
        self.date_after.dateChanged.connect(self._apply_filters)
        date_layout.addWidget(self.date_after)
        filters_layout.addLayout(date_layout)
        
        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)
        
        # Clear filters button
        clear_btn = QPushButton("Clear All Filters")
        clear_btn.clicked.connect(self._clear_filters)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
        
        return panel
    
    def _create_center_panel(self) -> QWidget:
        """Create center panel with dataset list and details"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Dataset table
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(9)
        self.dataset_table.setHorizontalHeaderLabels([
            'Name', 'Status', 'Probability', 'Rows', 
            'Patterns', 'Timeframes', 'Created', 'Size', 'Select'
        ])
        self.dataset_table.horizontalHeader().setStretchLastSection(False)
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_table.itemDoubleClicked.connect(self._on_dataset_double_click)
        self.dataset_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self.dataset_table)
        
        # Details panel
        details_group = QGroupBox("Dataset Details")
        details_layout = QVBoxLayout()
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.clicked.connect(self._load_selected)
        self.load_btn.setEnabled(False)
        action_layout.addWidget(self.load_btn)
        
        self.combine_btn = QPushButton("Combine Selected")
        self.combine_btn.clicked.connect(self._combine_selected)
        self.combine_btn.setEnabled(False)
        action_layout.addWidget(self.combine_btn)
        
        self.accept_btn = QPushButton("Accept Dataset")
        self.accept_btn.clicked.connect(lambda: self._update_status('accepted'))
        self.accept_btn.setEnabled(False)
        action_layout.addWidget(self.accept_btn)
        
        self.reject_btn = QPushButton("Reject Dataset")
        self.reject_btn.clicked.connect(lambda: self._update_status('rejected'))
        self.reject_btn.setEnabled(False)
        action_layout.addWidget(self.reject_btn)
        
        self.delete_btn = QPushButton("Delete Dataset")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        action_layout.addWidget(self.delete_btn)
        
        layout.addLayout(action_layout)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with data visualization"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Visualization controls
        viz_group = QGroupBox("Data Visualization")
        viz_layout = QVBoxLayout()
        
        if not MATPLOTLIB_AVAILABLE:
            # Show warning if matplotlib is not available
            warning_label = QLabel("⚠️ Visualization disabled: matplotlib not available")
            warning_label.setStyleSheet("color: orange; font-weight: bold;")
            viz_layout.addWidget(warning_label)
        else:
            # Chart type selector
            chart_layout = QHBoxLayout()
            chart_layout.addWidget(QLabel("Chart Type:"))
            self.chart_type_combo = QComboBox()
            self.chart_type_combo.addItems(['Price Chart', 'Volume Chart', 'Distribution', 'Correlation Matrix'])
            self.chart_type_combo.currentTextChanged.connect(self._update_visualization)
            chart_layout.addWidget(self.chart_type_combo)
            viz_layout.addLayout(chart_layout)
            
            # Time range selector
            range_layout = QHBoxLayout()
            range_layout.addWidget(QLabel("Time Range:"))
            self.time_range_combo = QComboBox()
            self.time_range_combo.addItems(['All Data', 'Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'Last Year'])
            self.time_range_combo.currentTextChanged.connect(self._update_visualization)
            range_layout.addWidget(self.time_range_combo)
            viz_layout.addLayout(range_layout)
            
            # Chart area
            self.figure = Figure(figsize=(6, 8))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Statistics panel
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        return panel
    
    def _setup_toolbar(self):
        """Setup toolbar"""
        toolbar = self.addToolBar("Dataset Tools")
        
        refresh_action = QAction("🔄 Refresh", self)
        refresh_action.triggered.connect(self._refresh_datasets)
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        export_action = QAction("📤 Export", self)
        export_action.triggered.connect(self._export_dataset)
        toolbar.addAction(export_action)
        
        import_action = QAction("📥 Import", self)
        import_action.triggered.connect(self._import_dataset)
        toolbar.addAction(import_action)
        
        toolbar.addSeparator()
        
        create_sample_action = QAction("➕ Create Sample", self)
        create_sample_action.triggered.connect(self._create_sample_dataset)
        toolbar.addAction(create_sample_action)
        
    def _refresh_datasets(self):
        """Refresh dataset list"""
        # Clear current data
        self.dataset_table.setRowCount(0)
        self.selected_datasets.clear()
        
        # Get all unique values for filters
        all_patterns = set()
        all_timeframes = set()
        all_locations = set()
        
        # Get all datasets
        all_datasets = self.dataset_manager.search_datasets()
        
        for info in all_datasets:
            all_patterns.update(info.patterns_included)
            all_timeframes.update(info.timeframes)
            all_locations.update(info.location_strategies)
        
        # Update filter lists
        self.pattern_list.clear()
        self.pattern_list.addItems(sorted(all_patterns))
        
        self.timeframe_list.clear()
        self.timeframe_list.addItems(sorted(all_timeframes))
        
        self.location_list.clear()
        self.location_list.addItems(sorted(all_locations))
        
        # Apply current filters
        self._apply_filters()
        
        self.status_bar.showMessage(f"Loaded {len(all_datasets)} datasets")
        
    def _apply_quick_filter(self):
        """Apply quick filter buttons"""
        # Uncheck other buttons
        sender = self.sender()
        if sender == self.quick_accepted_btn:
            self.quick_rejected_btn.setChecked(False)
            self.quick_pending_btn.setChecked(False)
            if self.quick_accepted_btn.isChecked():
                self.status_filter.setCurrentText('Accepted')
            else:
                self.status_filter.setCurrentText('All')
        elif sender == self.quick_rejected_btn:
            self.quick_accepted_btn.setChecked(False)
            self.quick_pending_btn.setChecked(False)
            if self.quick_rejected_btn.isChecked():
                self.status_filter.setCurrentText('Rejected')
            else:
                self.status_filter.setCurrentText('All')
        elif sender == self.quick_pending_btn:
            self.quick_accepted_btn.setChecked(False)
            self.quick_rejected_btn.setChecked(False)
            if self.quick_pending_btn.isChecked():
                self.status_filter.setCurrentText('Pending')
            else:
                self.status_filter.setCurrentText('All')
        
        self._apply_filters()
        
    def _apply_filters(self):
        """Apply current filters to dataset list"""
        # Build filter dict
        filters = {}
        
        # Status filter
        status_text = self.status_filter.currentText()
        if status_text != 'All':
            filters['acceptance_status'] = status_text.lower()
        
        # Pattern filter
        selected_patterns = [item.text() for item in self.pattern_list.selectedItems()]
        if selected_patterns:
            filters['patterns'] = selected_patterns
        
        # Timeframe filter
        selected_timeframes = [item.text() for item in self.timeframe_list.selectedItems()]
        if selected_timeframes:
            filters['timeframes'] = selected_timeframes
        
        # Location filter
        selected_locations = [item.text() for item in self.location_list.selectedItems()]
        if selected_locations:
            filters['location_strategies'] = selected_locations
        
        # Probability filter
        if self.min_prob_spin.value() > 0:
            filters['min_probability'] = self.min_prob_spin.value() / 100.0
        
        # Date filter
        if self.date_filter_check.isChecked():
            filters['created_after'] = self.date_after.date().toPyDate()
        
        # Search with filters
        query = self.search_input.text()
        results = self.dataset_manager.search_datasets(query, filters)
        
        # Update table
        self._update_table(results)
        
    def _update_table(self, datasets: List[DatasetInfo]):
        """Update dataset table"""
        self.dataset_table.setRowCount(len(datasets))
        
        for row, info in enumerate(datasets):
            # Name
            self.dataset_table.setItem(row, 0, QTableWidgetItem(info.name))
            
            # Status with color
            status_item = QTableWidgetItem(info.acceptance_status.capitalize())
            if info.acceptance_status == 'accepted':
                status_item.setBackground(QColor(0, 255, 0, 50))
            elif info.acceptance_status == 'rejected':
                status_item.setBackground(QColor(255, 0, 0, 50))
            elif info.acceptance_status == 'pending':
                status_item.setBackground(QColor(255, 255, 0, 50))
            self.dataset_table.setItem(row, 1, status_item)
            
            # Probability
            prob_text = f"{info.probability:.1%}" if info.probability else "N/A"
            self.dataset_table.setItem(row, 2, QTableWidgetItem(prob_text))
            
            # Row count
            self.dataset_table.setItem(row, 3, QTableWidgetItem(f"{info.row_count:,}"))
            
            # Patterns
            patterns_text = ", ".join(info.patterns_included[:3])
            if len(info.patterns_included) > 3:
                patterns_text += "..."
            self.dataset_table.setItem(row, 4, QTableWidgetItem(patterns_text))
            
            # Timeframes
            tf_text = ", ".join(info.timeframes[:3])
            if len(info.timeframes) > 3:
                tf_text += "..."
            self.dataset_table.setItem(row, 5, QTableWidgetItem(tf_text))
            
            # Created date
            date_text = info.created_at.strftime("%Y-%m-%d %H:%M")
            self.dataset_table.setItem(row, 6, QTableWidgetItem(date_text))
            
            # Size (estimated)
            size_mb = info.row_count * 0.001  # Rough estimate
            size_text = f"{size_mb:.1f} MB" if size_mb > 1 else f"{size_mb*1024:.0f} KB"
            self.dataset_table.setItem(row, 7, QTableWidgetItem(size_text))
            
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(lambda state, id=info.id: self._on_selection_changed(id, state))
            self.dataset_table.setCellWidget(row, 8, checkbox)
            
            # Store dataset ID in first column
            self.dataset_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, info.id)
        
        self.dataset_table.resizeColumnsToContents()
        
    def _on_selection_changed(self, dataset_id: str, state: int):
        """Handle dataset selection change"""
        if state == Qt.CheckState.Checked.value:
            if dataset_id not in self.selected_datasets:
                self.selected_datasets.append(dataset_id)
        else:
            if dataset_id in self.selected_datasets:
                self.selected_datasets.remove(dataset_id)
        
        # Update button states
        has_selection = len(self.selected_datasets) > 0
        self.load_btn.setEnabled(has_selection)
        self.combine_btn.setEnabled(len(self.selected_datasets) >= 2)
        self.accept_btn.setEnabled(has_selection)
        self.reject_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        
        # Update status bar
        self.status_bar.showMessage(f"Selected: {len(self.selected_datasets)} datasets")
        
    def _on_table_selection_changed(self):
        """Handle table row selection change"""
        current_row = self.dataset_table.currentRow()
        if current_row >= 0:
            dataset_id = self.dataset_table.item(current_row, 0).data(Qt.ItemDataRole.UserRole)
            if dataset_id:
                # Load and display details
                info = self.dataset_manager.dataset_index.get(dataset_id)
                if info:
                    self._show_dataset_details(info)
                    self._load_dataset_for_visualization(dataset_id)
    
    def _on_dataset_double_click(self, item: QTableWidgetItem):
        """Handle double click on dataset"""
        row = item.row()
        dataset_id = self.dataset_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        if dataset_id:
            # Load and display details
            info = self.dataset_manager.dataset_index.get(dataset_id)
            if info:
                self._show_dataset_details(info)
                self._load_dataset_for_visualization(dataset_id)
    
    def _show_dataset_details(self, info: DatasetInfo):
        """Show detailed information about dataset"""
        details = f"""
Dataset: {info.name}
ID: {info.id}
Created: {info.created_at.strftime("%Y-%m-%d %H:%M:%S")}
Status: {info.acceptance_status.capitalize()}

Data Properties:
- Rows: {info.row_count:,}
- Date Range: {info.date_range[0]} to {info.date_range[1]}
- Probability: {f"{info.probability:.2%}" if info.probability else "N/A"}
- Confidence: {f"[{info.confidence_interval[0]:.2%}, {info.confidence_interval[1]:.2%}]" if info.confidence_interval else "N/A"}

Strategies Used:
{chr(10).join(f"  • {s}" for s in info.strategies_used)}

Patterns Included:
{chr(10).join(f"  • {p}" for p in info.patterns_included)}

Timeframes:
{chr(10).join(f"  • {t}" for t in info.timeframes)}

Location Strategies:
{chr(10).join(f"  • {l}" for l in info.location_strategies)}

Tags: {", ".join(info.tags) if info.tags else "None"}

Notes:
{info.notes if info.notes else "No notes"}
"""
        
        if info.parent_datasets:
            details += f"\nParent Datasets:\n"
            for parent_id in info.parent_datasets:
                parent_info = self.dataset_manager.dataset_index.get(parent_id)
                if parent_info:
                    details += f"  • {parent_info.name} ({parent_id})\n"
            details += f"Combination Method: {info.combination_method}\n"
        
        self.details_text.setText(details)
    
    def _load_dataset_for_visualization(self, dataset_id: str):
        """Load dataset for visualization"""
        try:
            data, info = self.dataset_manager.load_dataset(dataset_id)
            self.current_data = data
            self.current_info = info
            self._update_visualization()
            self._update_statistics()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load dataset for visualization: {str(e)}")
    
    def _update_visualization(self):
        """Update the visualization chart"""
        if not MATPLOTLIB_AVAILABLE or self.current_data is None:
            return
            
        # Clear previous plot
        self.figure.clear()
        
        # Get chart type and time range
        chart_type = self.chart_type_combo.currentText()
        time_range = self.time_range_combo.currentText()
        
        # Filter data by time range
        data = self._filter_data_by_time_range(self.current_data, time_range)
        
        if len(data) == 0:
            self.canvas.draw()
            return
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        if chart_type == 'Price Chart':
            self._plot_price_chart(ax, data)
        elif chart_type == 'Volume Chart':
            self._plot_volume_chart(ax, data)
        elif chart_type == 'Distribution':
            self._plot_distribution(ax, data)
        elif chart_type == 'Correlation Matrix':
            self._plot_correlation_matrix(ax, data)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _filter_data_by_time_range(self, data: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """Filter data by time range"""
        if time_range == 'All Data':
            return data
        
        end_date = data.index.max()
        
        if time_range == 'Last 30 Days':
            start_date = end_date - timedelta(days=30)
        elif time_range == 'Last 90 Days':
            start_date = end_date - timedelta(days=90)
        elif time_range == 'Last 6 Months':
            start_date = end_date - timedelta(days=180)
        elif time_range == 'Last Year':
            start_date = end_date - timedelta(days=365)
        else:
            return data
        
        return data[data.index >= start_date]
    
    def _plot_price_chart(self, ax, data: pd.DataFrame):
        """Plot price chart"""
        try:
            if 'Close' in data.columns:
                ax.plot(data.index, data['Close'], label='Close Price', linewidth=1)
            if 'High' in data.columns and 'Low' in data.columns:
                ax.fill_between(data.index, data['Low'], data['High'], alpha=0.3, label='High-Low Range')
            
            ax.set_title(f'Price Chart - {self.current_info.name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting price chart: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_volume_chart(self, ax, data: pd.DataFrame):
        """Plot volume chart"""
        try:
            if 'Volume' in data.columns:
                ax.bar(data.index, data['Volume'], alpha=0.7, label='Volume')
                ax.set_title(f'Volume Chart - {self.current_info.name}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Volume')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, 'No Volume column found in data', 
                       ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting volume chart: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_distribution(self, ax, data: pd.DataFrame):
        """Plot distribution chart"""
        try:
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'Returns Distribution - {self.current_info.name}')
                ax.set_xlabel('Returns')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Close column found for returns calculation', 
                       ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting distribution: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_correlation_matrix(self, ax, data: pd.DataFrame):
        """Plot correlation matrix"""
        try:
            # Select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                ax.text(0.5, 0.5, 'Not enough numeric columns for correlation matrix', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            corr_matrix = data[numeric_cols].corr()
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45)
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black")
            
            ax.set_title(f'Correlation Matrix - {self.current_info.name}')
            self.figure.colorbar(im, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting correlation matrix: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _update_statistics(self):
        """Update statistics panel"""
        if self.current_data is None:
            self.stats_text.clear()
            return
        
        stats = []
        stats.append(f"Dataset: {self.current_info.name}")
        stats.append(f"Rows: {len(self.current_data):,}")
        stats.append(f"Columns: {len(self.current_data.columns)}")
        stats.append(f"Date Range: {self.current_data.index.min()} to {self.current_data.index.max()}")
        
        # Numeric statistics
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats.append("\nNumeric Statistics:")
            for col in numeric_cols[:5]:  # Show first 5 columns
                col_stats = self.current_data[col].describe()
                stats.append(f"{col}:")
                stats.append(f"  Mean: {col_stats['mean']:.4f}")
                stats.append(f"  Std: {col_stats['std']:.4f}")
                stats.append(f"  Min: {col_stats['min']:.4f}")
                stats.append(f"  Max: {col_stats['max']:.4f}")
        
        self.stats_text.setText('\n'.join(stats))
    
    def _on_search(self):
        """Handle search input change"""
        self._apply_filters()
    
    def _clear_filters(self):
        """Clear all filters"""
        self.search_input.clear()
        self.status_filter.setCurrentIndex(0)
        self.quick_accepted_btn.setChecked(False)
        self.quick_rejected_btn.setChecked(False)
        self.quick_pending_btn.setChecked(False)
        self.pattern_list.clearSelection()
        self.timeframe_list.clearSelection()
        self.location_list.clearSelection()
        self.min_prob_spin.setValue(0)
        self.date_filter_check.setChecked(False)
        self._apply_filters()
    
    def _load_selected(self):
        """Load selected dataset"""
        if not self.selected_datasets:
            return
        
        # Load first selected dataset
        dataset_id = self.selected_datasets[0]
        
        try:
            data, info = self.dataset_manager.load_dataset(dataset_id)
            self.dataset_selected.emit(dataset_id, data, info)
            
            QMessageBox.information(self, "Success", 
                                  f"Dataset '{info.name}' loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def _combine_selected(self):
        """Combine selected datasets"""
        if len(self.selected_datasets) < 2:
            QMessageBox.warning(self, "Warning", 
                              "Please select at least 2 datasets to combine")
            return
        
        # Get combination parameters
        dialog = CombineDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            method = dialog.get_method()
            name = dialog.get_name()
            
            try:
                # Combine datasets
                combined_data, probability = self.combiner.combine_datasets(
                    self.selected_datasets, method, name
                )
                
                # Get parent strategies
                all_strategies = []
                for dataset_id in self.selected_datasets:
                    info = self.dataset_manager.dataset_index[dataset_id]
                    all_strategies.extend(info.strategies_used)
                
                # Save combined dataset
                new_info = self.dataset_manager.save_dataset(
                    name=name,
                    data=combined_data,
                    strategies=[],  # Would need actual strategy objects
                    probability=probability,
                    parent_datasets=self.selected_datasets,
                    combination_method=method,
                    notes=f"Combined from {len(self.selected_datasets)} datasets using {method}",
                    tags=['combined', method.lower()]
                )
                
                # Emit signal
                self.datasets_combined.emit(new_info.id, combined_data, probability)
                
                # Refresh display
                self._refresh_datasets()
                
                QMessageBox.information(self, "Success", 
                                      f"Created combined dataset '{name}' with {len(combined_data)} rows")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to combine datasets: {str(e)}")
    
    def _update_status(self, status: str):
        """Update status of selected datasets"""
        if not self.selected_datasets:
            return
        
        # Get additional info
        dialog = StatusUpdateDialog(status, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            probability = dialog.get_probability()
            notes = dialog.get_notes()
            
            # Update each selected dataset
            for dataset_id in self.selected_datasets:
                self.dataset_manager.update_acceptance(
                    dataset_id, status, probability, notes
                )
            
            # Refresh display
            self._refresh_datasets()
            
            QMessageBox.information(self, "Success", 
                                  f"Updated {len(self.selected_datasets)} datasets to {status}")
    
    def _delete_selected(self):
        """Delete selected datasets"""
        if not self.selected_datasets:
            return
        
        reply = QMessageBox.question(self, "Confirm Deletion", 
                                   f"Are you sure you want to delete {len(self.selected_datasets)} dataset(s)?\n\nThis action cannot be undone.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                for dataset_id in self.selected_datasets:
                    # This would need to be implemented in DatasetManager
                    # For now, just remove from index
                    if dataset_id in self.dataset_manager.dataset_index:
                        del self.dataset_manager.dataset_index[dataset_id]
                
                # Refresh display
                self._refresh_datasets()
                
                QMessageBox.information(self, "Success", 
                                      f"Deleted {len(self.selected_datasets)} dataset(s)")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete datasets: {str(e)}")
    
    def _export_dataset(self):
        """Export selected dataset"""
        if not self.selected_datasets:
            QMessageBox.warning(self, "Warning", "Please select a dataset to export")
            return
        
        dataset_id = self.selected_datasets[0]
        info = self.dataset_manager.dataset_index[dataset_id]
        
        # Get export format
        formats = "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
        filepath, selected_format = QFileDialog.getSaveFileName(
            self, "Export Dataset", f"{info.name}.csv", formats
        )
        
        if filepath:
            try:
                if "csv" in selected_format:
                    format = 'csv'
                elif "xlsx" in selected_format:
                    format = 'excel'
                else:
                    format = 'json'
                
                self.dataset_manager.export_dataset(dataset_id, filepath, format)
                QMessageBox.information(self, "Success", f"Dataset exported to {filepath}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def _import_dataset(self):
        """Import external dataset"""
        # Get file to import
        formats = "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*.*)"
        filepath, selected_format = QFileDialog.getOpenFileName(
            self, "Import Dataset", "", formats
        )
        
        if filepath:
            try:
                # Determine format
                if filepath.endswith('.csv'):
                    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                elif filepath.endswith('.xlsx'):
                    data = pd.read_excel(filepath, index_col=0, parse_dates=True)
                elif filepath.endswith('.json'):
                    with open(filepath, 'r') as f:
                        import json
                        json_data = json.load(f)
                        if 'data' in json_data:
                            data = pd.DataFrame(json_data['data'])
                        else:
                            data = pd.DataFrame(json_data)
                else:
                    QMessageBox.warning(self, "Warning", "Unsupported file format")
                    return
                
                # Get dataset name
                name, ok = QInputDialog.getText(self, "Dataset Name", 
                                              "Enter a name for this dataset:")
                if not ok or not name:
                    return
                
                # Save imported dataset
                from strategies.strategy_builders import PatternStrategy
                dummy_strategy = PatternStrategy(name="Imported")
                
                info = self.dataset_manager.save_dataset(
                    name=name,
                    data=data,
                    strategies=[dummy_strategy],
                    notes=f"Imported from {filepath}",
                    tags=['imported']
                )
                
                # Refresh display
                self._refresh_datasets()
                
                QMessageBox.information(self, "Success", 
                                      f"Imported dataset '{name}' with {len(data)} rows")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Import failed: {str(e)}")
    
    def _create_sample_dataset(self):
        """Create a sample dataset for testing"""
        try:
            # Generate sample data
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
            np.random.seed(42)
            
            # Generate OHLCV data
            base_price = 100
            returns = np.random.normal(0, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = pd.DataFrame({
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
            data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
            
            # Get dataset name
            name, ok = QInputDialog.getText(self, "Sample Dataset Name", 
                                          "Enter a name for the sample dataset:")
            if not ok or not name:
                return
            
            # Save sample dataset
            from strategies.strategy_builders import PatternStrategy
            dummy_strategy = PatternStrategy(name="Sample")
            
            info = self.dataset_manager.save_dataset(
                name=name,
                data=data,
                strategies=[dummy_strategy],
                notes="Sample dataset for testing",
                tags=['sample', 'test']
            )
            
            # Refresh display
            self._refresh_datasets()
            
            QMessageBox.information(self, "Success", 
                                  f"Created sample dataset '{name}' with {len(data)} rows")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create sample dataset: {str(e)}")


class CombineDialog(QDialog):
    """Dialog for combining datasets"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine Datasets")
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout()
        
        # Method selection
        layout.addWidget(QLabel("Combination Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['AND', 'OR', 'XOR'])
        layout.addWidget(self.method_combo)
        
        # Name input
        layout.addWidget(QLabel("Combined Dataset Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name for combined dataset...")
        layout.addWidget(self.name_input)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_method(self) -> str:
        return self.method_combo.currentText()
    
    def get_name(self) -> str:
        return self.name_input.text() or f"Combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class StatusUpdateDialog(QDialog):
    """Dialog for updating dataset status"""
    
    def __init__(self, status: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Update to {status.capitalize()}")
        self.setModal(True)
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # Probability input
        layout.addWidget(QLabel("Probability (optional):"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0, 1)
        self.prob_spin.setSingleStep(0.01)
        self.prob_spin.setValue(0)
        self.prob_spin.setSuffix(" %")
        self.prob_spin.setDecimals(1)
        layout.addWidget(self.prob_spin)
        
        # Notes input
        layout.addWidget(QLabel("Notes:"))
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Add notes about this decision...")
        self.notes_input.setMaximumHeight(100)
        layout.addWidget(self.notes_input)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_probability(self) -> Optional[float]:
        return self.prob_spin.value() / 100.0 if self.prob_spin.value() > 0 else None
    
    def get_notes(self) -> str:
        return self.notes_input.toPlainText()
