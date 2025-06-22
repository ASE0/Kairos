"""
gui/workspace_manager_dialog.py
==============================
Workspace management dialog for managing saved patterns, strategies, and datasets
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import os
from pathlib import Path
from typing import Dict, List, Any

from core.workspace_manager import WorkspaceManager


class WorkspaceManagerDialog(QDialog):
    """Dialog for managing workspace components"""
    
    def __init__(self, workspace_manager: WorkspaceManager, parent=None):
        super().__init__(parent)
        self.workspace_manager = workspace_manager
        self.setWindowTitle("Workspace Manager")
        self.setGeometry(200, 200, 800, 600)
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """Setup UI layout"""
        # Main scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Main content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Tab widget for different components
        self.tab_widget = QTabWidget()
        
        # Patterns tab
        self.patterns_tab = self.create_patterns_tab()
        self.tab_widget.addTab(self.patterns_tab, "Patterns")
        
        # Strategies tab
        self.strategies_tab = self.create_strategies_tab()
        self.tab_widget.addTab(self.strategies_tab, "Strategies")
        
        # Datasets tab
        self.datasets_tab = self.create_datasets_tab()
        self.tab_widget.addTab(self.datasets_tab, "Datasets")
        
        # Configs tab
        self.configs_tab = self.create_configs_tab()
        self.tab_widget.addTab(self.configs_tab, "Configurations")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_data)
        button_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("Export Workspace")
        self.export_btn.clicked.connect(self.export_workspace)
        button_layout.addWidget(self.export_btn)
        
        self.import_btn = QPushButton("Import Workspace")
        self.import_btn.clicked.connect(self.import_workspace)
        button_layout.addWidget(self.import_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Main layout for dialog
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        
        # Set size policies for resizable dialog
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(600, 400)  # Minimum size to ensure usability
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog, QWidget {
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
        
    def create_patterns_tab(self) -> QWidget:
        """Create patterns management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Patterns list
        self.patterns_list = QListWidget()
        self.patterns_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(QLabel("Saved Patterns:"))
        layout.addWidget(self.patterns_list)
        
        # Pattern actions
        pattern_actions = QHBoxLayout()
        
        self.load_pattern_btn = QPushButton("Load Pattern")
        self.load_pattern_btn.clicked.connect(self.load_selected_pattern)
        pattern_actions.addWidget(self.load_pattern_btn)
        
        self.delete_pattern_btn = QPushButton("Delete Pattern")
        self.delete_pattern_btn.clicked.connect(self.delete_selected_pattern)
        pattern_actions.addWidget(self.delete_pattern_btn)
        
        self.rename_pattern_btn = QPushButton("Rename Pattern")
        self.rename_pattern_btn.clicked.connect(self.rename_selected_pattern)
        pattern_actions.addWidget(self.rename_pattern_btn)
        
        pattern_actions.addStretch()
        layout.addLayout(pattern_actions)
        
        # Pattern details
        self.pattern_details = QTextEdit()
        self.pattern_details.setReadOnly(True)
        self.pattern_details.setMaximumHeight(150)
        layout.addWidget(QLabel("Pattern Details:"))
        layout.addWidget(self.pattern_details)
        
        # Connect selection change
        self.patterns_list.currentItemChanged.connect(self.on_pattern_selection_changed)
        
        return widget
    
    def create_strategies_tab(self) -> QWidget:
        """Create strategies management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Strategies list
        self.strategies_list = QListWidget()
        self.strategies_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(QLabel("Saved Strategies:"))
        layout.addWidget(self.strategies_list)
        
        # Strategy actions
        strategy_actions = QHBoxLayout()
        
        self.load_strategy_btn = QPushButton("Load Strategy")
        self.load_strategy_btn.clicked.connect(self.load_selected_strategy)
        strategy_actions.addWidget(self.load_strategy_btn)
        
        self.delete_strategy_btn = QPushButton("Delete Strategy")
        self.delete_strategy_btn.clicked.connect(self.delete_selected_strategy)
        strategy_actions.addWidget(self.delete_strategy_btn)
        
        self.rename_strategy_btn = QPushButton("Rename Strategy")
        self.rename_strategy_btn.clicked.connect(self.rename_selected_strategy)
        strategy_actions.addWidget(self.rename_strategy_btn)
        
        strategy_actions.addStretch()
        layout.addLayout(strategy_actions)
        
        # Strategy details
        self.strategy_details = QTextEdit()
        self.strategy_details.setReadOnly(True)
        self.strategy_details.setMaximumHeight(150)
        layout.addWidget(QLabel("Strategy Details:"))
        layout.addWidget(self.strategy_details)
        
        # Connect selection change
        self.strategies_list.currentItemChanged.connect(self.on_strategy_selection_changed)
        
        return widget
    
    def create_datasets_tab(self) -> QWidget:
        """Create datasets management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Datasets list
        self.datasets_list = QListWidget()
        self.datasets_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(QLabel("Saved Datasets:"))
        layout.addWidget(self.datasets_list)
        
        # Dataset actions
        dataset_actions = QHBoxLayout()
        
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.clicked.connect(self.load_selected_dataset)
        dataset_actions.addWidget(self.load_dataset_btn)
        
        self.delete_dataset_btn = QPushButton("Delete Dataset")
        self.delete_dataset_btn.clicked.connect(self.delete_selected_dataset)
        dataset_actions.addWidget(self.delete_dataset_btn)
        
        self.rename_dataset_btn = QPushButton("Rename Dataset")
        self.rename_dataset_btn.clicked.connect(self.rename_selected_dataset)
        dataset_actions.addWidget(self.rename_dataset_btn)
        
        dataset_actions.addStretch()
        layout.addLayout(dataset_actions)
        
        # Dataset details
        self.dataset_details = QTextEdit()
        self.dataset_details.setReadOnly(True)
        self.dataset_details.setMaximumHeight(150)
        layout.addWidget(QLabel("Dataset Details:"))
        layout.addWidget(self.dataset_details)
        
        # Connect selection change
        self.datasets_list.currentItemChanged.connect(self.on_dataset_selection_changed)
        
        return widget
    
    def create_configs_tab(self) -> QWidget:
        """Create configurations management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Configs list
        self.configs_list = QListWidget()
        self.configs_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(QLabel("Saved Configurations:"))
        layout.addWidget(self.configs_list)
        
        # Config actions
        config_actions = QHBoxLayout()
        
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_selected_config)
        config_actions.addWidget(self.load_config_btn)
        
        self.delete_config_btn = QPushButton("Delete Config")
        self.delete_config_btn.clicked.connect(self.delete_selected_config)
        config_actions.addWidget(self.delete_config_btn)
        
        self.rename_config_btn = QPushButton("Rename Config")
        self.rename_config_btn.clicked.connect(self.rename_selected_config)
        config_actions.addWidget(self.rename_config_btn)
        
        config_actions.addStretch()
        layout.addLayout(config_actions)
        
        # Config details
        self.config_details = QTextEdit()
        self.config_details.setReadOnly(True)
        self.config_details.setMaximumHeight(150)
        layout.addWidget(QLabel("Configuration Details:"))
        layout.addWidget(self.config_details)
        
        # Connect selection change
        self.configs_list.currentItemChanged.connect(self.on_config_selection_changed)
        
        return widget
    
    def load_data(self):
        """Load all workspace data"""
        # Load patterns
        self.patterns_list.clear()
        patterns = self.workspace_manager.list_patterns()
        for pattern in patterns:
            self.patterns_list.addItem(pattern)
        
        # Load strategies
        self.strategies_list.clear()
        strategies = self.workspace_manager.list_strategies()
        for strategy in strategies:
            self.strategies_list.addItem(strategy)
        
        # Load datasets
        self.datasets_list.clear()
        datasets = self.workspace_manager.list_datasets()
        for dataset in datasets:
            self.datasets_list.addItem(dataset)
        
        # Load configs
        self.configs_list.clear()
        configs = self.workspace_manager.list_configs()
        for config in configs:
            self.configs_list.addItem(config)
    
    def on_pattern_selection_changed(self, current, previous):
        """Handle pattern selection change"""
        if current:
            pattern_name = current.text()
            pattern = self.workspace_manager.load_pattern(pattern_name)
            if pattern:
                details = f"Name: {pattern.name}\n"
                details += f"Type: {type(pattern).__name__}\n"
                details += f"Timeframes: {[f'{tf.value}{tf.unit}' for tf in pattern.timeframes]}\n"
                details += f"Required Bars: {pattern.required_bars}\n"
                
                if hasattr(pattern, 'advanced_features') and pattern.advanced_features:
                    details += f"\nAdvanced Features:\n"
                    for feature, enabled in pattern.advanced_features.items():
                        if isinstance(enabled, bool) and enabled:
                            details += f"  - {feature}\n"
                
                self.pattern_details.setText(details)
    
    def on_strategy_selection_changed(self, current, previous):
        """Handle strategy selection change"""
        if current:
            strategy_name = current.text()
            strategy = self.workspace_manager.load_strategy(strategy_name)
            if strategy:
                details = f"Name: {strategy.name}\n"
                details += f"Type: {strategy.type}\n"
                details += f"Actions: {len(strategy.actions)}\n"
                details += f"Combination Logic: {strategy.combination_logic}\n"
                details += f"Min Actions Required: {strategy.min_actions_required}\n"
                
                if strategy.gates_and_logic:
                    details += f"\nGates & Logic:\n"
                    for gate, enabled in strategy.gates_and_logic.items():
                        if isinstance(enabled, bool) and enabled:
                            details += f"  - {gate}\n"
                
                self.strategy_details.setText(details)
    
    def on_dataset_selection_changed(self, current, previous):
        """Handle dataset selection change"""
        if current:
            dataset_name = current.text()
            dataset_info = self.workspace_manager.load_dataset(dataset_name)
            if dataset_info:
                data = dataset_info['data']
                metadata = dataset_info.get('metadata', {})
                
                details = f"Name: {dataset_name}\n"
                details += f"Rows: {len(data)}\n"
                details += f"Columns: {list(data.columns)}\n"
                details += f"Date Range: {data.index[0]} to {data.index[-1]}\n"
                
                if metadata:
                    details += f"\nMetadata:\n"
                    for key, value in metadata.items():
                        if key not in ['rows', 'columns']:
                            details += f"  {key}: {value}\n"
                
                self.dataset_details.setText(details)
    
    def on_config_selection_changed(self, current, previous):
        """Handle config selection change"""
        if current:
            config_name = current.text()
            config = self.workspace_manager.load_workspace_config(config_name)
            if config:
                details = f"Name: {config_name}\n"
                details += f"Saved At: {config.get('saved_at', 'Unknown')}\n"
                
                if 'patterns' in config:
                    details += f"Patterns: {len(config['patterns'])}\n"
                if 'strategies' in config:
                    details += f"Strategies: {len(config['strategies'])}\n"
                if 'datasets' in config:
                    details += f"Datasets: {len(config['datasets'])}\n"
                
                self.config_details.setText(details)
    
    def load_selected_pattern(self):
        """Load selected pattern into main application"""
        current_item = self.patterns_list.currentItem()
        if current_item:
            pattern_name = current_item.text()
            pattern = self.workspace_manager.load_pattern(pattern_name)
            if pattern and hasattr(self.parent(), 'patterns'):
                self.parent().patterns[pattern_name] = pattern
                QMessageBox.information(self, "Success", f"Pattern '{pattern_name}' loaded successfully")
    
    def delete_selected_pattern(self):
        """Delete selected pattern"""
        current_item = self.patterns_list.currentItem()
        if current_item:
            pattern_name = current_item.text()
            reply = QMessageBox.question(self, "Confirm Delete", 
                                       f"Are you sure you want to delete pattern '{pattern_name}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.workspace_manager.delete_pattern(pattern_name):
                    self.load_data()
                    QMessageBox.information(self, "Success", f"Pattern '{pattern_name}' deleted")
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete pattern '{pattern_name}'")
    
    def rename_selected_pattern(self):
        """Rename selected pattern"""
        current_item = self.patterns_list.currentItem()
        if current_item:
            old_name = current_item.text()
            new_name, ok = QInputDialog.getText(self, "Rename Pattern", 
                                              "Enter new name:", text=old_name)
            if ok and new_name and new_name != old_name:
                # Load pattern, save with new name, delete old
                pattern = self.workspace_manager.load_pattern(old_name)
                if pattern:
                    if self.workspace_manager.save_pattern(pattern, new_name):
                        self.workspace_manager.delete_pattern(old_name)
                        self.load_data()
                        QMessageBox.information(self, "Success", f"Pattern renamed to '{new_name}'")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to rename pattern")
    
    # Similar methods for strategies, datasets, and configs
    def load_selected_strategy(self):
        current_item = self.strategies_list.currentItem()
        if current_item:
            strategy_name = current_item.text()
            strategy = self.workspace_manager.load_strategy(strategy_name)
            if strategy and hasattr(self.parent(), 'strategies'):
                self.parent().strategies['pattern'][strategy_name] = strategy
                QMessageBox.information(self, "Success", f"Strategy '{strategy_name}' loaded successfully")
    
    def delete_selected_strategy(self):
        current_item = self.strategies_list.currentItem()
        if current_item:
            strategy_name = current_item.text()
            reply = QMessageBox.question(self, "Confirm Delete", 
                                       f"Are you sure you want to delete strategy '{strategy_name}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.workspace_manager.delete_strategy(strategy_name):
                    self.load_data()
                    QMessageBox.information(self, "Success", f"Strategy '{strategy_name}' deleted")
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete strategy '{strategy_name}'")
    
    def rename_selected_strategy(self):
        current_item = self.strategies_list.currentItem()
        if current_item:
            old_name = current_item.text()
            new_name, ok = QInputDialog.getText(self, "Rename Strategy", 
                                              "Enter new name:", text=old_name)
            if ok and new_name and new_name != old_name:
                strategy = self.workspace_manager.load_strategy(old_name)
                if strategy:
                    if self.workspace_manager.save_strategy(strategy, new_name):
                        self.workspace_manager.delete_strategy(old_name)
                        self.load_data()
                        QMessageBox.information(self, "Success", f"Strategy renamed to '{new_name}'")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to rename strategy")
    
    def load_selected_dataset(self):
        current_item = self.datasets_list.currentItem()
        if current_item:
            dataset_name = current_item.text()
            dataset_info = self.workspace_manager.load_dataset(dataset_name)
            if dataset_info and hasattr(self.parent(), 'datasets'):
                self.parent().datasets[dataset_name] = dataset_info
                QMessageBox.information(self, "Success", f"Dataset '{dataset_name}' loaded successfully")
    
    def delete_selected_dataset(self):
        current_item = self.datasets_list.currentItem()
        if current_item:
            dataset_name = current_item.text()
            reply = QMessageBox.question(self, "Confirm Delete", 
                                       f"Are you sure you want to delete dataset '{dataset_name}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.workspace_manager.delete_dataset(dataset_name):
                    self.load_data()
                    QMessageBox.information(self, "Success", f"Dataset '{dataset_name}' deleted")
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete dataset '{dataset_name}'")
    
    def rename_selected_dataset(self):
        current_item = self.datasets_list.currentItem()
        if current_item:
            old_name = current_item.text()
            new_name, ok = QInputDialog.getText(self, "Rename Dataset", 
                                              "Enter new name:", text=old_name)
            if ok and new_name and new_name != old_name:
                dataset_info = self.workspace_manager.load_dataset(old_name)
                if dataset_info:
                    if self.workspace_manager.save_dataset(dataset_info['data'], new_name, dataset_info.get('metadata', {})):
                        self.workspace_manager.delete_dataset(old_name)
                        self.load_data()
                        QMessageBox.information(self, "Success", f"Dataset renamed to '{new_name}'")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to rename dataset")
    
    def load_selected_config(self):
        current_item = self.configs_list.currentItem()
        if current_item:
            config_name = current_item.text()
            config = self.workspace_manager.load_workspace_config(config_name)
            if config:
                QMessageBox.information(self, "Success", f"Configuration '{config_name}' loaded successfully")
    
    def delete_selected_config(self):
        current_item = self.configs_list.currentItem()
        if current_item:
            config_name = current_item.text()
            reply = QMessageBox.question(self, "Confirm Delete", 
                                       f"Are you sure you want to delete config '{config_name}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                config_file = self.workspace_manager.configs_dir / f"{config_name}.json"
                if config_file.exists():
                    config_file.unlink()
                    self.load_data()
                    QMessageBox.information(self, "Success", f"Config '{config_name}' deleted")
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete config '{config_name}'")
    
    def rename_selected_config(self):
        current_item = self.configs_list.currentItem()
        if current_item:
            old_name = current_item.text()
            new_name, ok = QInputDialog.getText(self, "Rename Config", 
                                              "Enter new name:", text=old_name)
            if ok and new_name and new_name != old_name:
                config = self.workspace_manager.load_workspace_config(old_name)
                if config:
                    if self.workspace_manager.save_workspace_config(config, new_name):
                        old_file = self.workspace_manager.configs_dir / f"{old_name}.json"
                        if old_file.exists():
                            old_file.unlink()
                        self.load_data()
                        QMessageBox.information(self, "Success", f"Config renamed to '{new_name}'")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to rename config")
    
    def export_workspace(self):
        """Export workspace to zip file"""
        from zipfile import ZipFile
        import tempfile
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Workspace", "", "ZIP Files (*.zip)")
        if filepath:
            try:
                with ZipFile(filepath, 'w') as zipf:
                    for file_path in self.workspace_manager.workspace_dir.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.workspace_manager.workspace_dir)
                            zipf.write(file_path, arcname)
                
                QMessageBox.information(self, "Success", f"Workspace exported to {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export workspace: {e}")
    
    def import_workspace(self):
        """Import workspace from zip file"""
        from zipfile import ZipFile
        
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Workspace", "", "ZIP Files (*.zip)")
        if filepath:
            try:
                with ZipFile(filepath, 'r') as zipf:
                    zipf.extractall(self.workspace_manager.workspace_dir)
                
                self.load_data()
                QMessageBox.information(self, "Success", f"Workspace imported from {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to import workspace: {e}") 