"""
gui/strategy_pattern_manager_window.py
======================================
Window for managing saved strategies and patterns.
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from gui.strategy_builder_window import StrategyBuilderWindow
from gui.pattern_builder_window import PatternBuilderWindow

class StrategyPatternManagerWindow(QMainWindow):
    """Window for viewing, editing, and deleting saved strategies and patterns."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Strategy/Pattern Manager")
        self.setGeometry(200, 200, 900, 700)
        
        self.strategy_manager = parent.strategy_manager if parent else None
        
        self._setup_ui()
        self._apply_stylesheet()
        self.populate_all_lists()

    def _setup_ui(self):
        """Setup the main UI layout with tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create Strategy and Pattern tabs
        self.strategy_tab = self._create_management_tab("Strategies")
        self.pattern_tab = self._create_management_tab("Patterns")

        self.tabs.addTab(self.strategy_tab, "Strategies")
        self.tabs.addTab(self.pattern_tab, "Patterns")
        
        # Connect signals for strategy tab
        self.strategy_tab.findChild(QListWidget, "list_widget").itemSelectionChanged.connect(
            lambda: self.display_details("strategy")
        )
        self.strategy_tab.findChild(QPushButton, "edit_button").clicked.connect(
            lambda: self.edit_item("strategy")
        )
        self.strategy_tab.findChild(QPushButton, "delete_button").clicked.connect(
            lambda: self.delete_item("strategy")
        )
        
        # Connect signals for pattern tab
        self.pattern_tab.findChild(QListWidget, "list_widget").itemSelectionChanged.connect(
            lambda: self.display_details("pattern")
        )
        self.pattern_tab.findChild(QPushButton, "edit_button").clicked.connect(
            lambda: self.edit_item("pattern")
        )
        self.pattern_tab.findChild(QPushButton, "delete_button").clicked.connect(
            lambda: self.delete_item("pattern")
        )

    def _create_management_tab(self, title: str) -> QWidget:
        """Creates a reusable tab for managing items (strategies or patterns)."""
        container = QWidget()
        main_layout = QHBoxLayout(container)

        # Left side: List
        list_container = QGroupBox(f"Saved {title}")
        list_layout = QVBoxLayout()
        list_widget = QListWidget()
        list_widget.setObjectName("list_widget") # Set object name for later retrieval
        list_layout.addWidget(list_widget)
        
        refresh_button = QPushButton("🔄 Refresh List")
        refresh_button.clicked.connect(self.populate_all_lists)
        list_layout.addWidget(refresh_button)
        list_container.setLayout(list_layout)

        # Right side: Details and Actions
        details_container = QGroupBox("Details")
        details_layout = QVBoxLayout()
        details_text = QTextEdit()
        details_text.setObjectName("details_text")
        details_text.setReadOnly(True)
        details_layout.addWidget(details_text)
        
        button_layout = QHBoxLayout()
        edit_button = QPushButton(f"Edit {title[:-1]}")
        edit_button.setObjectName("edit_button")
        edit_button.setEnabled(False)
        
        delete_button = QPushButton(f"Delete {title[:-1]}")
        delete_button.setObjectName("delete_button")
        delete_button.setEnabled(False)
        
        button_layout.addWidget(edit_button)
        button_layout.addWidget(delete_button)
        details_layout.addLayout(button_layout)
        details_container.setLayout(details_layout)

        main_layout.addWidget(list_container, 1)
        main_layout.addWidget(details_container, 2)
        
        return container

    def populate_all_lists(self):
        """Fetches and displays all saved strategies and patterns."""
        self._populate_list("strategy")
        self._populate_list("pattern")
        
    def _populate_list(self, item_type: str):
        """Populate the list for either strategies or patterns."""
        if item_type == "strategy":
            tab = self.strategy_tab
            source = self.parent_window.strategies if self.parent_window else {}
        else: # pattern
            tab = self.pattern_tab
            source = self.parent_window.patterns if self.parent_window else {}

        list_widget = tab.findChild(QListWidget, "list_widget")
        details_text = tab.findChild(QTextEdit, "details_text")
        edit_button = tab.findChild(QPushButton, "edit_button")
        delete_button = tab.findChild(QPushButton, "delete_button")

        list_widget.clear()
        details_text.clear()
        edit_button.setEnabled(False)
        delete_button.setEnabled(False)
        
        if item_type == "strategy":
            for s_type in source:
                for s_id, strategy in source[s_type].items():
                    item = QListWidgetItem(strategy.name)
                    item.setData(Qt.ItemDataRole.UserRole, strategy)
                    list_widget.addItem(item)
        else:
             for p_name, pattern in source.items():
                item = QListWidgetItem(pattern.name)
                item.setData(Qt.ItemDataRole.UserRole, pattern)
                list_widget.addItem(item)

    def display_details(self, item_type: str):
        """Display details of selected strategy or pattern."""
        if item_type == "strategy":
            tab = self.strategy_tab
            list_widget = tab.findChild(QListWidget, "list_widget")
            details_text = tab.findChild(QTextEdit, "details_text")
        else:
            tab = self.pattern_tab
            list_widget = tab.findChild(QListWidget, "list_widget")
            details_text = tab.findChild(QTextEdit, "details_text")
        
        if not list_widget.selectedItems():
            details_text.clear()
            return
            
        item = list_widget.selectedItems()[0]
        item_id = item.data(Qt.ItemDataRole.UserRole)
        
        if item_type == "strategy":
            strategy = item_id
            details = f"Name: {strategy.name}\nID: {strategy.id}\nType: {strategy.type.capitalize()}\n\n"
            if hasattr(strategy, 'actions'):
                details += "Actions:\n"
                for i, action in enumerate(strategy.actions, 1):
                    details += f"  {i}. {action.name} (Pattern: {action.pattern.name})\n"
        else:
            pattern = item_id
            details = f"Name: {pattern.name}\nType: {type(pattern).__name__}\n"
            details += f"Required Bars: {pattern.get_required_bars()}"
        
        details_text.setText(details)
        
        tab.findChild(QPushButton, "edit_button").setEnabled(True)
        tab.findChild(QPushButton, "delete_button").setEnabled(True)

    def edit_item(self, item_type: str):
        """Opens the selected item in its respective builder."""
        tab = self.strategy_tab if item_type == "strategy" else self.pattern_tab
        list_widget = tab.findChild(QListWidget, "list_widget")
        selected_items = list_widget.selectedItems()
        if not selected_items: return

        item_to_edit = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not item_to_edit: return

        if item_type == "strategy":
            builder_window = StrategyBuilderWindow(self.parent_window)
            builder_window.load_strategy_for_editing(item_to_edit)
        else: # pattern
            builder_window = PatternBuilderWindow(self.parent_window)
            # This method needs to be created in PatternBuilderWindow
            builder_window.load_pattern_for_editing(item_to_edit)
            
        builder_window.show()
        self.parent_window.open_windows.append(builder_window)

    def delete_item(self, item_type: str):
        """Deletes the selected item from the workspace."""
        tab = self.strategy_tab if item_type == "strategy" else self.pattern_tab
        list_widget = tab.findChild(QListWidget, "list_widget")
        selected_items = list_widget.selectedItems()
        if not selected_items: return

        item_to_delete = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not item_to_delete or not self.strategy_manager: return

        warning_message = f"Are you sure you want to permanently delete '{item_to_delete.name}'?"
        if item_type == "pattern":
            warning_message += ("\n\nWarning: Deleting a pattern may break strategies that depend on it. "
                              "This action cannot be undone.")

        reply = QMessageBox.question(self, "Confirm Delete", warning_message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            if item_type == "strategy":
                success = self.strategy_manager.delete_strategy(item_to_delete.name)
                if success:
                    del self.parent_window.strategies[item_to_delete.type][item_to_delete.id]
            else: # pattern
                success = self.strategy_manager.delete_pattern(item_to_delete.name)
                if success:
                    del self.parent_window.patterns[item_to_delete.name]
            
            if success:
                QMessageBox.information(self, "Success", f"'{item_to_delete.name}' was deleted.")
                self.parent_window._load_strategies_and_results() # Reload all data
                self.populate_all_lists()
            else:
                QMessageBox.warning(self, "Error", f"Failed to delete '{item_to_delete.name}'.")

    def _apply_stylesheet(self):
        """Apply a dark theme stylesheet."""
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
            }
            QListWidget, QTextEdit {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
        """) 

    def _delete_selected(self):
        """Delete selected strategy or pattern."""
        if self.tabs.currentIndex() == 0:  # Strategies tab
            list_widget = self.strategy_tab.findChild(QListWidget, "list_widget")
            item_type = "strategy"
            manager = self.parent_window.strategy_manager
        else:  # Patterns tab
            list_widget = self.pattern_tab.findChild(QListWidget, "list_widget")
            item_type = "pattern"
            manager = self.parent_window.strategy_manager  # Assuming same manager for now

        if not list_widget.selectedItems():
            QMessageBox.warning(self, "No Selection", f"Please select a {item_type} to delete.")
            return

        item = list_widget.selectedItems()[0]
        item_id = item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to delete '{item.text()}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if item_type == "strategy":
                    manager.delete_strategy(item_id)
                else:
                    manager.delete_pattern(item_id) # Assumes this method exists
                
                # Remove from list widget
                list_widget.takeItem(list_widget.row(item))
                
                # Clear details pane
                self.pattern_tab.findChild(QTextEdit, "details_text").clear()
                self.parent_window._load_strategies_and_results() # Refresh main hub
                QMessageBox.information(self, "Success", f"'{item.text()}' has been deleted.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete {item_type}: {e}") 