"""
gui/dataset_explorer_window.py
==============================
Dataset explorer with search and filtering
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.dataset_manager import DatasetManager, DatasetInfo, DatasetCombiner
from core.data_structures import ProbabilityMetrics


class DatasetExplorerWindow(QMainWindow):
    """Window for exploring and managing datasets"""
    
    # Signals
    dataset_selected = pyqtSignal(str, pd.DataFrame, DatasetInfo)
    datasets_combined = pyqtSignal(str, pd.DataFrame, ProbabilityMetrics)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Explorer")
        self.setGeometry(150, 150, 1200, 800)
        
        # Dataset manager
        self.dataset_manager = DatasetManager()
        self.combiner = DatasetCombiner(self.dataset_manager)
        
        # Current selection
        self.selected_datasets = []
        
        # Setup UI
        self._setup_ui()
        
        # Load initial data
        self._refresh_datasets()
        
        # Apply light stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QPushButton {
                background-color: #cccccc;
                border: 1px solid #888;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #b0b0b0;
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
        
        # Right panel - Dataset list and details
        right_panel = self._create_right_panel()
        layout.addWidget(right_panel, 2)
        
        # Setup toolbar
        self._setup_toolbar()
        
        # Status bar
        self.status_bar = self.statusBar()
        
    def _create_left_panel(self) -> QWidget:
        """Create left panel with search and filters"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Search section
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search datasets...")
        self.search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(self.search_input)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Filters section
        filters_group = QGroupBox("Filters")
        filters_layout = QVBoxLayout()
        
        # Acceptance status filter
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
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with dataset list"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Dataset table
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(8)
        self.dataset_table.setHorizontalHeaderLabels([
            'Name', 'Status', 'Probability', 'Rows', 
            'Patterns', 'Timeframes', 'Created', 'Select'
        ])
        self.dataset_table.horizontalHeader().setStretchLastSection(False)
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dataset_table.itemDoubleClicked.connect(self._on_dataset_double_click)
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
        
        layout.addLayout(action_layout)
        
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
            filters['min_probability'] = self.min_prob_spin.value()
        
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
            self.dataset_table.setItem(row, 1, status_item)
            
            # Probability
            prob_text = f"{info.probability:.2%}" if info.probability else "N/A"
            self.dataset_table.setItem(row, 2, QTableWidgetItem(prob_text))
            
            # Row count
            self.dataset_table.setItem(row, 3, QTableWidgetItem(str(info.row_count)))
            
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
            
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(lambda state, id=info.id: self._on_selection_changed(id, state))
            self.dataset_table.setCellWidget(row, 7, checkbox)
            
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
        
        # Update status bar
        self.status_bar.showMessage(f"Selected: {len(self.selected_datasets)} datasets")
        
    def _on_dataset_double_click(self, item: QTableWidgetItem):
        """Handle double click on dataset"""
        row = item.row()
        dataset_id = self.dataset_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        if dataset_id:
            # Load and display details
            info = self.dataset_manager.dataset_index.get(dataset_id)
            if info:
                self._show_dataset_details(info)
    
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
    
    def _on_search(self):
        """Handle search input change"""
        self._apply_filters()
    
    def _clear_filters(self):
        """Clear all filters"""
        self.search_input.clear()
        self.status_filter.setCurrentIndex(0)
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
        # This would implement importing from external sources
        QMessageBox.information(self, "Import", "Import functionality coming soon")


class CombineDialog(QDialog):
    """Dialog for combining datasets"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine Datasets")
        self.setModal(True)
        
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
        
        layout = QVBoxLayout()
        
        # Probability input
        layout.addWidget(QLabel("Probability (optional):"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0, 1)
        self.prob_spin.setSingleStep(0.01)
        self.prob_spin.setValue(0)
        self.prob_spin.setSpecialValueText("Not set")
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
        return self.prob_spin.value() if self.prob_spin.value() > 0 else None
    
    def get_notes(self) -> str:
        return self.notes_input.toPlainText()
