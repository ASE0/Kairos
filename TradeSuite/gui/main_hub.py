"""
gui/main_hub.py
===============
Main Trading Strategy Hub GUI Application
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import logging

# Import all modules
from core.data_structures import *
from core.dataset_manager import DatasetInfo
from core.workspace_manager import WorkspaceManager
from core.strategy_manager import StrategyManager
from patterns.candlestick_patterns import *
from patterns.enhanced_candlestick_patterns import FVGPattern
from processors.data_processor import *
from strategies.strategy_builders import *
from statistics1.probability_calculator import *
from gui.data_stripper_window import DataStripperWindow
from gui.strategy_builder_window import StrategyBuilderWindow
from gui.strategy_combiner_window import StrategyCombinerWindow
from gui.statistics_window import StatisticsWindow
from gui.workspace_manager_dialog import WorkspaceManagerDialog
from gui.strategy_pattern_manager_window import StrategyPatternManagerWindow

logger = logging.getLogger(__name__)


class TradingStrategyHub(QMainWindow):
    """Main hub application for trading strategy building"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trading Strategy Hub - Professional Edition")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize managers
        self.workspace_manager = WorkspaceManager()
        self.strategy_manager = StrategyManager()
        
        # Storage
        self.datasets = {}
        self.actions = {}
        self.patterns = {}
        self.strategies = {
            'pattern': {},
            'risk': {},
            'combined': {}
        }
        self.results = {}
        
        # Windows
        self.open_windows = []
        
        # Session management
        self.last_session_file = 'last_session.json'
        self.last_loaded_dataset = None
        
        # Setup
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._load_defaults()
        
        # Apply dark theme
        self._apply_theme()
        
        # Load workspace and strategies
        self._load_workspace()
        self._load_strategies_and_results()
        
        self._load_last_session()
        
    def _setup_ui(self):
        """Setup main UI layout"""
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
        
        # Welcome panel
        welcome_panel = self._create_welcome_panel()
        layout.addWidget(welcome_panel)
        
        # Quick access buttons
        quick_access = self._create_quick_access_panel()
        layout.addWidget(quick_access)
        
        # Status dashboard
        self.dashboard = self._create_dashboard()
        layout.addWidget(self.dashboard)
        
        # Workspace info panel
        workspace_panel = self._create_workspace_panel()
        layout.addWidget(workspace_panel)
        
        # Log viewer
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(150)
        layout.addWidget(QLabel("System Log:"))
        layout.addWidget(self.log_viewer)
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Main layout for central widget
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(scroll_area)
        
        # Set size policies for resizable window
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(1000, 600)  # Minimum size to ensure usability
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
    def _create_welcome_panel(self) -> QWidget:
        """Create welcome panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Trading Strategy Hub - Professional Edition")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Advanced Trading Strategy Development Platform with Quantification & Execution Logic")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)
        
        # Version info
        version = QLabel("Version 2.0 - Complete Integration")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(version)
        
        return panel
        
    def _create_quick_access_panel(self) -> QWidget:
        """Create quick access buttons panel"""
        panel = QWidget()
        layout = QGridLayout(panel)
        
        # Create buttons
        buttons_config = [
            ("Data Stripper", "📊", self.open_data_stripper, 0, 0),
            ("Dataset Explorer", "🔍", self.open_dataset_explorer, 0, 1),
            ("Pattern Builder", "📈", self.open_pattern_builder, 0, 2),
            ("Strategy Builder", "🎯", self.open_strategy_builder, 0, 3),
            ("Risk Manager", "🛡️", self.open_risk_manager, 1, 0),
            ("Strategy Combiner", "🔗", self.open_strategy_combiner, 1, 1),
            ("AI Strategy Optimizer", "🤖", self.open_strategy_optimizer, 1, 2),
            ("Statistics Analyzer", "📉", self.open_statistics_analyzer, 1, 3),
            ("Backtest Engine", "⚡", self.open_backtest_engine, 2, 0),
            ("Results Viewer", "📋", self.open_results_viewer, 2, 1),
            ("Workspace Manager", "💾", self.open_workspace_manager, 2, 2),
            ("Strategy/Pattern Manager", "🗂️", self.open_strategy_pattern_manager, 2, 3)
        ]
        
        for text, icon, callback, row, col in buttons_config:
            btn = QPushButton(f"{icon} {text}")
            btn.clicked.connect(callback)
            btn.setMinimumHeight(60)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #cccccc;
                }
            """)
            layout.addWidget(btn, row, col)
            
        return panel
        
    def _create_dashboard(self) -> QWidget:
        """Create status dashboard"""
        dashboard = QGroupBox("System Status")
        layout = QGridLayout(dashboard)
        
        # Status items
        self.status_labels = {}
        
        status_items = [
            ("Datasets Loaded", "datasets", 0, 0),
            ("Actions Created", "actions", 0, 1),
            ("Patterns Defined", "patterns", 0, 2),
            ("Strategies Built", "strategies", 0, 3),
            ("Backtests Run", "backtests", 1, 0),
            ("Combined Strategies", "combined", 1, 1),
            ("Success Rate", "success_rate", 1, 2),
            ("Active Windows", "windows", 1, 3)
        ]
        
        for label, key, row, col in status_items:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            
            title_label = QLabel(label)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(title_label)
            
            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 20px; font-weight: bold;")
            container_layout.addWidget(value_label)
            
            self.status_labels[key] = value_label
            layout.addWidget(container, row, col)
            
        self._update_dashboard()
        return dashboard
        
    def _create_workspace_panel(self) -> QWidget:
        """Create workspace information panel"""
        panel = QGroupBox("Workspace Status")
        layout = QGridLayout(panel)
        
        # Workspace info
        self.workspace_labels = {}
        
        workspace_items = [
            ("Saved Patterns", "saved_patterns", 0, 0),
            ("Saved Strategies", "saved_strategies", 0, 1),
            ("Saved Datasets", "saved_datasets", 0, 2),
            ("Saved Configs", "saved_configs", 0, 3),
            ("Workspace Size", "workspace_size", 1, 0),
            ("Last Saved", "last_saved", 1, 1),
            ("Auto Save", "auto_save", 1, 2),
            ("Backup Status", "backup_status", 1, 3)
        ]
        
        for label, key, row, col in workspace_items:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            
            title_label = QLabel(label)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(title_label)
            
            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 14px; font-weight: bold;")
            container_layout.addWidget(value_label)
            
            self.workspace_labels[key] = value_label
            layout.addWidget(container, row, col)
            
        self._update_workspace_panel()
        return panel
        
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_workspace = QAction("New Workspace", self)
        new_workspace.setShortcut("Ctrl+N")
        new_workspace.triggered.connect(self.new_workspace)
        file_menu.addAction(new_workspace)
        
        open_workspace = QAction("Open Workspace", self)
        open_workspace.setShortcut("Ctrl+O")
        open_workspace.triggered.connect(self.open_workspace)
        file_menu.addAction(open_workspace)
        
        save_workspace = QAction("Save Workspace", self)
        save_workspace.setShortcut("Ctrl+S")
        save_workspace.triggered.connect(self.save_workspace)
        file_menu.addAction(save_workspace)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        for action_name, callback in [
            ("Data Stripper", self.open_data_stripper),
            ("Pattern Builder", self.open_pattern_builder),
            ("Strategy Builder", self.open_strategy_builder),
            ("Risk Manager", self.open_risk_manager),
            ("Strategy Combiner", self.open_strategy_combiner),
            ("AI Strategy Optimizer", self.open_strategy_optimizer),
            ("Statistics Analyzer", self.open_statistics_analyzer)
        ]:
            action = QAction(action_name, self)
            action.triggered.connect(callback)
            tools_menu.addAction(action)
            
        # View menu
        view_menu = menubar.addMenu("View")
        
        toggle_log = QAction("Toggle Log", self)
        toggle_log.triggered.connect(lambda: self.log_viewer.setVisible(not self.log_viewer.isVisible()))
        view_menu.addAction(toggle_log)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about = QAction("About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _setup_toolbar(self):
        """Setup toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        # Add quick actions (text only, no icons for now)
        actions = [
            ("Data Stripper", self.open_data_stripper),
            ("Pattern Builder", self.open_pattern_builder),
            ("Strategy Builder", self.open_strategy_builder),
            ("Combine Strategies", self.open_strategy_combiner),
            ("AI Strategy Optimizer", self.open_strategy_optimizer),
            ("Analyze Statistics", self.open_statistics_analyzer)
        ]

        for text, callback in actions:
            action = QAction(text, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)
            
    def _load_defaults(self):
        """Load default patterns and configurations"""
        # Create comprehensive default patterns based on advanced concepts
        default_patterns = {
            # Basic candlestick patterns
            'ii_bars': IIBarsPattern(timeframes=[TimeRange(5, 'm')]),
            'double_wick': DoubleWickPattern(timeframes=[TimeRange(15, 'm')]),
            'hammer': HammerPattern(timeframes=[TimeRange(5, 'm')]),
            
            # Advanced quantification patterns
            'doji_standard': CustomPattern(
                name="Doji_Standard",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.1, upper_wick_ratio=0.4, lower_wick_ratio=0.4)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.05,
                    'doji_sigma_w': 0.1
                }
            ),
            
            'doji_long_leg': CustomPattern(
                name="Doji_Long_Leg",
                timeframes=[TimeRange(30, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.05, upper_wick_ratio=0.45, lower_wick_ratio=0.45)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.03,
                    'doji_sigma_w': 0.15
                }
            ),
            
            'strong_body': CustomPattern(
                name="Strong_Body",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.7, upper_wick_ratio=0.15, lower_wick_ratio=0.15)],
                advanced_features={
                    'body_size': True,
                    'two_bar_strength': True,
                    'beta_pat': 1.5
                }
            ),
            
            'weak_body': CustomPattern(
                name="Weak_Body",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.2, upper_wick_ratio=0.4, lower_wick_ratio=0.4)],
                advanced_features={
                    'body_size': True,
                    'doji_ness': True,
                    'doji_sigma_b': 0.1,
                    'doji_sigma_w': 0.1
                }
            ),
            
            # Momentum patterns
            'momentum_breakout': CustomPattern(
                name="Momentum_Breakout",
                timeframes=[TimeRange(5, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)],
                advanced_features={
                    'momentum_context': True,
                    'kappa_m': 0.8,
                    'two_bar_strength': True,
                    'beta_pat': 2.0
                }
            ),
            
            'momentum_reversal': CustomPattern(
                name="Momentum_Reversal",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.25, lower_wick_ratio=0.25)],
                advanced_features={
                    'momentum_context': True,
                    'kappa_m': 0.3,
                    'location_context': True
                }
            ),
            
            # Volatility patterns
            'high_volatility': CustomPattern(
                name="High_Volatility",
                timeframes=[TimeRange(5, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)],
                advanced_features={
                    'location_context': True,
                    'momentum_context': True,
                    'kappa_m': 1.0
                }
            ),
            
            'low_volatility': CustomPattern(
                name="Low_Volatility",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.3, upper_wick_ratio=0.35, lower_wick_ratio=0.35)],
                advanced_features={
                    'body_size': True,
                    'doji_ness': True,
                    'doji_sigma_b': 0.08,
                    'doji_sigma_w': 0.08
                }
            ),
            
            # Location-based patterns
            'support_bounce': CustomPattern(
                name="Support_Bounce",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.2, lower_wick_ratio=0.3)],
                advanced_features={
                    'location_context': True,
                    'momentum_context': True,
                    'kappa_m': 0.5
                }
            ),
            
            'resistance_rejection': CustomPattern(
                name="Resistance_Rejection",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.3, lower_wick_ratio=0.2)],
                advanced_features={
                    'location_context': True,
                    'momentum_context': True,
                    'kappa_m': 0.5
                }
            ),
            
            # Multi-bar patterns
            'three_white_soldiers': CustomPattern(
                name="Three_White_Soldiers",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[
                    OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                    OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                    OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)
                ],
                required_bars=3,
                advanced_features={
                    'two_bar_strength': True,
                    'beta_pat': 1.8,
                    'momentum_context': True,
                    'kappa_m': 0.7
                }
            ),
            
            'three_black_crows': CustomPattern(
                name="Three_Black_Crows",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[
                    OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                    OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2),
                    OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)
                ],
                required_bars=3,
                advanced_features={
                    'two_bar_strength': True,
                    'beta_pat': 1.8,
                    'momentum_context': True,
                    'kappa_m': 0.7
                }
            ),
            
            # Advanced doji patterns
            'four_price_doji': CustomPattern(
                name="Four_Price_Doji",
                timeframes=[TimeRange(30, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.01, upper_wick_ratio=0.495, lower_wick_ratio=0.495)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.01,
                    'doji_sigma_w': 0.2
                }
            ),
            
            'dragonfly_doji': CustomPattern(
                name="Dragonfly_Doji",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.1, upper_wick_ratio=0.0, lower_wick_ratio=0.8)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.05,
                    'doji_sigma_w': 0.1,
                    'location_context': True
                }
            ),
            
            'gravestone_doji': CustomPattern(
                name="Gravestone_Doji",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.1, upper_wick_ratio=0.8, lower_wick_ratio=0.0)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.05,
                    'doji_sigma_w': 0.1,
                    'location_context': True
                }
            ),
            
            # Volatility breakout patterns
            'volatility_expansion': CustomPattern(
                name="Volatility_Expansion",
                timeframes=[TimeRange(5, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)],
                advanced_features={
                    'momentum_context': True,
                    'kappa_m': 1.2,
                    'location_context': True
                }
            ),
            
            'volatility_contraction': CustomPattern(
                name="Volatility_Contraction",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.2, upper_wick_ratio=0.4, lower_wick_ratio=0.4)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.1,
                    'doji_sigma_w': 0.1,
                    'body_size': True
                }
            ),
            
            # Trend continuation patterns
            'trend_continuation': CustomPattern(
                name="Trend_Continuation",
                timeframes=[TimeRange(30, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)],
                advanced_features={
                    'two_bar_strength': True,
                    'beta_pat': 1.3,
                    'momentum_context': True,
                    'kappa_m': 0.6
                }
            ),
            
            # Reversal patterns
            'trend_reversal': CustomPattern(
                name="Trend_Reversal",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.5, upper_wick_ratio=0.25, lower_wick_ratio=0.25)],
                advanced_features={
                    'location_context': True,
                    'momentum_context': True,
                    'kappa_m': 0.4,
                    'two_bar_strength': True,
                    'beta_pat': 1.5
                }
            ),
            
            # Gap patterns
            'gap_up': CustomPattern(
                name="Gap_Up",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.7, upper_wick_ratio=0.15, lower_wick_ratio=0.15)],
                advanced_features={
                    'momentum_context': True,
                    'kappa_m': 1.0,
                    'two_bar_strength': True,
                    'beta_pat': 2.0
                }
            ),
            
            'gap_down': CustomPattern(
                name="Gap_Down",
                timeframes=[TimeRange(15, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.7, upper_wick_ratio=0.15, lower_wick_ratio=0.15)],
                advanced_features={
                    'momentum_context': True,
                    'kappa_m': 1.0,
                    'two_bar_strength': True,
                    'beta_pat': 2.0
                }
            ),
            
            # Consolidation patterns
            'consolidation': CustomPattern(
                name="Consolidation",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.3, upper_wick_ratio=0.35, lower_wick_ratio=0.35)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.1,
                    'doji_sigma_w': 0.1,
                    'body_size': True
                }
            ),
            
            # Breakout patterns
            'breakout': CustomPattern(
                name="Breakout",
                timeframes=[TimeRange(30, 'm')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.6, upper_wick_ratio=0.2, lower_wick_ratio=0.2)],
                advanced_features={
                    'momentum_context': True,
                    'kappa_m': 0.8,
                    'location_context': True,
                    'two_bar_strength': True,
                    'beta_pat': 1.7
                }
            ),
            
            # Exhaustion patterns
            'exhaustion': CustomPattern(
                name="Exhaustion",
                timeframes=[TimeRange(1, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)],
                advanced_features={
                    'doji_ness': True,
                    'doji_sigma_b': 0.08,
                    'doji_sigma_w': 0.08,
                    'momentum_context': True,
                    'kappa_m': 0.2
                }
            ),
            
            # Accumulation patterns
            'accumulation': CustomPattern(
                name="Accumulation",
                timeframes=[TimeRange(4, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)],
                advanced_features={
                    'location_context': True,
                    'momentum_context': True,
                    'kappa_m': 0.3,
                    'body_size': True
                }
            ),
            
            # Distribution patterns
            'distribution': CustomPattern(
                name="Distribution",
                timeframes=[TimeRange(4, 'h')],
                ohlc_ratios=[OHLCRatio(body_ratio=0.4, upper_wick_ratio=0.3, lower_wick_ratio=0.3)],
                advanced_features={
                    'location_context': True,
                    'momentum_context': True,
                    'kappa_m': 0.3,
                    'body_size': True
                }
            ),
            # Add FVGPattern for GUI use
            'FVG (Fair Value Gap)': FVGPattern(timeframes=[TimeRange(1, 'm')], min_gap_size=0.001)
        }
        
        for name, pattern in default_patterns.items():
            self.patterns[name] = pattern
            
        self._log(f"Loaded {len(default_patterns)} comprehensive default patterns")
        self._update_dashboard()
        
    def _apply_theme(self):
        """Apply dark theme with modern styling"""
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
        
    def _load_workspace(self):
        """Load existing workspace data and restore last dataset if available."""
        try:
            # Load saved patterns
            saved_patterns = self.workspace_manager.list_patterns()
            for pattern_name in saved_patterns:
                pattern = self.workspace_manager.load_pattern(pattern_name)
                if pattern:
                    self.patterns[pattern_name] = pattern
            # Load saved strategies
            saved_strategies = self.workspace_manager.list_strategies()
            for strategy_name in saved_strategies:
                strategy = self.workspace_manager.load_strategy(strategy_name)
                if strategy:
                    self.strategies['pattern'][strategy_name] = strategy
            # Load saved datasets
            saved_datasets = self.workspace_manager.list_datasets()
            for dataset_name in saved_datasets:
                dataset_info = self.workspace_manager.load_dataset(dataset_name)
                if dataset_info:
                    self.datasets[dataset_name] = dataset_info
            self._log(f"Loaded workspace: {len(saved_patterns)} patterns, {len(saved_strategies)} strategies, {len(saved_datasets)} datasets")
            # Restore last loaded dataset if available
            self._load_last_session()
            if self.last_loaded_dataset and self.last_loaded_dataset in self.datasets:
                # Optionally, set as active in UI (if such a method exists)
                self._log(f"Restored last loaded dataset: {self.last_loaded_dataset}")
                # You may want to call a method to update the UI selection here
        except Exception as e:
            self._log(f"Error loading workspace: {e}")
    
    def _save_workspace(self):
        """Save current workspace"""
        try:
            # Save all patterns
            for name, pattern in self.patterns.items():
                self.workspace_manager.save_pattern(pattern, name)
            
            # Save all strategies
            for name, strategy in self.strategies['pattern'].items():
                self.workspace_manager.save_strategy(strategy, name)
            
            # Save all datasets
            for name, dataset_info in self.datasets.items():
                if 'data' in dataset_info:
                    self.workspace_manager.save_dataset(dataset_info['data'], name, dataset_info.get('metadata', {}))
            
            # Save workspace config
            config = {
                'patterns': list(self.patterns.keys()),
                'strategies': list(self.strategies['pattern'].keys()),
                'datasets': list(self.datasets.keys()),
                'saved_at': datetime.now().isoformat()
            }
            self.workspace_manager.save_workspace_config(config, 'current_workspace')
            
            self._log("Workspace saved successfully")
            self._update_workspace_panel()
            
        except Exception as e:
            self._log(f"Error saving workspace: {e}")
    
    def _update_workspace_panel(self):
        """Update workspace panel information"""
        try:
            # Count saved items
            saved_patterns = len(self.workspace_manager.list_patterns())
            saved_strategies = len(self.workspace_manager.list_strategies())
            saved_datasets = len(self.workspace_manager.list_datasets())
            saved_configs = len(self.workspace_manager.list_configs())
            
            # Calculate workspace size
            workspace_size = 0
            for file_path in self.workspace_manager.workspace_dir.rglob("*"):
                if file_path.is_file():
                    workspace_size += file_path.stat().st_size
            
            workspace_size_mb = workspace_size / (1024 * 1024)
            
            # Update labels
            self.workspace_labels['saved_patterns'].setText(str(saved_patterns))
            self.workspace_labels['saved_strategies'].setText(str(saved_strategies))
            self.workspace_labels['saved_datasets'].setText(str(saved_datasets))
            self.workspace_labels['saved_configs'].setText(str(saved_configs))
            self.workspace_labels['workspace_size'].setText(f"{workspace_size_mb:.1f} MB")
            self.workspace_labels['last_saved'].setText(datetime.now().strftime("%H:%M"))
            self.workspace_labels['auto_save'].setText("Enabled")
            self.workspace_labels['backup_status'].setText("OK")
            
        except Exception as e:
            self._log(f"Error updating workspace panel: {e}")
    
    def open_workspace_manager(self):
        """Open workspace management window"""
        dialog = WorkspaceManagerDialog(self.workspace_manager, self)
        dialog.exec()
        self._update_workspace_panel()
    
    def on_pattern_created(self, pattern: CandlestickPattern):
        """Handle created pattern"""
        self.patterns[pattern.name] = pattern
        self._log(f"Pattern '{pattern.name}' created")
        
        # Save the pattern to disk
        self.strategy_manager.save_pattern(pattern)
        
        self._update_dashboard()
    
    def on_strategy_created(self, strategy: BaseStrategy):
        """Handle created strategy"""
        strategy_type = strategy.type
        self.strategies[strategy_type][strategy.id] = strategy
        
        # Ensure strategy has a proper name
        if not strategy.name or strategy.name.strip() == "":
            strategy.name = f"Strategy_{strategy.id}"
        
        self._log(f"{strategy_type.capitalize()} strategy '{strategy.name}' created")
        
        # Save the strategy to disk
        self.strategy_manager.save_strategy(strategy)
        
        # Refresh any open statistics windows
        self._refresh_statistics_windows()
        
        self._update_dashboard()
    
    def save_strategy(self, strategy: BaseStrategy):
        """Save a strategy to disk"""
        try:
            self.strategy_manager.save_strategy(strategy)
            self._log(f"Strategy '{strategy.name}' saved successfully")
        except Exception as e:
            self._log(f"Error saving strategy: {e}")
            raise
    
    def update_strategy_list(self):
        """Update the strategy list in any open windows"""
        # Refresh any open windows that display strategies
        for window in self.open_windows:
            if hasattr(window, 'refresh_datasets'):
                try:
                    window.refresh_datasets()
                except Exception as e:
                    self._log(f"Error refreshing window: {e}")
    
    def show_status_message(self, message: str):
        """Show a status message in the status bar"""
        self.status_bar.showMessage(message)
        self._log(message)
    
    def _refresh_statistics_windows(self):
        """Refresh any open statistics windows"""
        if hasattr(self, 'open_windows'):
            for window in self.open_windows:
                if hasattr(window, 'refresh_data'):
                    try:
                        window.refresh_data()
                    except Exception as e:
                        self._log(f"Error refreshing statistics window: {e}")
    
    def _add_to_results_viewers(self, results: Dict[str, Any], strategy_name: str):
        """Add results to any open results viewer windows"""
        if hasattr(self, 'open_windows'):
            for window in self.open_windows:
                if hasattr(window, 'add_result') and hasattr(window, 'results_history'):
                    try:
                        window.add_result(results, strategy_name)
                    except Exception as e:
                        self._log(f"Error adding to results viewer: {e}")
                        
    def _refresh_results_viewers_from_disk(self):
        """Refresh all open results viewer windows from disk"""
        if hasattr(self, 'open_windows'):
            for window in self.open_windows:
                if hasattr(window, '_refresh_results_from_disk'):
                    try:
                        window._refresh_results_from_disk()
                    except Exception as e:
                        self._log(f"Error refreshing results viewer from disk: {e}")
        
    # Window opening methods
    def open_data_stripper(self):
        """Open data stripper window"""
        from gui.data_stripper_window import DataStripperWindow
        window = DataStripperWindow(self)
        window.data_processed.connect(self.on_data_processed)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_dataset_explorer(self):
        """Open dataset explorer window"""
        from gui.data_explorer_window import DatasetExplorerWindow
        window = DatasetExplorerWindow(self)
        window.dataset_selected.connect(self.on_dataset_selected)
        window.datasets_combined.connect(self.on_datasets_combined)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_pattern_builder(self):
        """Open pattern builder window"""
        from gui.pattern_builder_window import PatternBuilderWindow
        window = PatternBuilderWindow(self)
        window.pattern_created.connect(self.on_pattern_created)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_strategy_builder(self):
        """Open strategy builder window"""
        window = StrategyBuilderWindow(self)
        window.strategy_created.connect(self.on_strategy_created)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_risk_manager(self):
        """Open risk manager window"""
        from gui.risk_manager_window import RiskManagerWindow
        window = RiskManagerWindow(self)
        window.risk_strategy_created.connect(self.on_risk_strategy_created)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_strategy_combiner(self):
        """Open strategy combiner window"""
        window = StrategyCombinerWindow(self)
        window.combination_created.connect(self.on_combination_created)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_statistics_analyzer(self):
        """Open statistics1 analyzer window"""
        window = StatisticsWindow(self)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_backtest_engine(self):
        """Open backtest engine window"""
        from gui.backtest_window import BacktestWindow
        window = BacktestWindow(self)
        window.backtest_complete.connect(self.on_backtest_complete)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_results_viewer(self):
        """Open results viewer window"""
        from gui.results_viewer_window import ResultsViewerWindow
        window = ResultsViewerWindow(self)
        window.show()
        self.open_windows.append(window)
        
        # Refresh the results viewer to load any existing results from disk
        window._refresh_results_from_disk()
        
        self._update_dashboard()
        
    def open_strategy_optimizer(self):
        """Open AI strategy optimizer window"""
        from gui.strategy_optimizer_window import StrategyOptimizerWindow
        window = StrategyOptimizerWindow(self)
        window.show()
        self.open_windows.append(window)
        self._update_dashboard()
        
    def open_strategy_pattern_manager(self):
        """Open the strategy/pattern manager window."""
        from gui.strategy_pattern_manager_window import StrategyPatternManagerWindow
        window = StrategyPatternManagerWindow(self)
        window.show()
        self.open_windows.append(window)
        
    # Slot methods
    def on_data_processed(self, dataset_name: str, data: pd.DataFrame, metadata: DatasetMetadata):
        """Handle processed data from stripper"""
        self.datasets[dataset_name] = {
            'data': data,
            'metadata': metadata
        }
        self.last_loaded_dataset = dataset_name
        self._log(f"Dataset '{dataset_name}' processed: {len(data)} rows")
        self._update_dashboard()
        
    def on_risk_strategy_created(self, strategy: RiskStrategy):
        """Handle created risk strategy"""
        print(f"[DEBUG] Received risk strategy with name: '{strategy.name}', ID: '{strategy.id}'")
        self.strategies['risk'][strategy.id] = strategy
        print(f"[DEBUG] Stored strategy in self.strategies['risk'] with key: '{strategy.id}'")
        self._log(f"Risk strategy '{strategy.name}' created")
        self._update_dashboard()
        
    def on_dataset_selected(self, dataset_id: str, data: pd.DataFrame, info: DatasetInfo):
        """Handle dataset selection from explorer"""
        self._log(f"Dataset '{info.name}' loaded from explorer")
        self.datasets[info.name] = {'data': data, 'metadata': info}
        self.last_loaded_dataset = info.name
        self._update_dashboard()
        self._refresh_statistics_windows()
        
    def on_datasets_combined(self, dataset_id: str, data: pd.DataFrame, probability: ProbabilityMetrics):
        """Handle combined datasets from explorer"""
        self._log(f"Combined dataset created with {len(data)} rows, probability: {probability.probability:.2%}")
        self._update_dashboard()
        
    def on_backtest_complete(self, results: Dict[str, Any]):
        """Handle backtest results"""
        print(f"[DEBUG] MainHub: Received backtest_complete signal with results keys: {list(results.keys())}")
        # Save results to disk
        self.strategy_manager.save_backtest_results(results)
        
        # Reload all results from disk to ensure consistency
        self.results = self.strategy_manager.load_all_results()
        
        self._log(f"Backtest completed for: {results.get('strategy_name', 'Unknown')}")
        
        # Add to any open results viewers
        self._add_to_results_viewers(results, results.get('strategy_name', 'Unknown'))
        
        # Refresh any open statistics windows
        self._refresh_statistics_windows()
        
        self._update_dashboard()
        
    def on_combination_created(self, combination):
        """Handle created combined strategy"""
        self.strategies['combined'][combination.name] = combination
        self._log(f"Combined strategy '{combination.name}' created")
        self._update_dashboard()
        
    # Workspace methods
    def new_workspace(self):
        """Create new workspace"""
        reply = QMessageBox.question(
            self, "New Workspace",
            "This will clear all current data. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.datasets.clear()
            self.actions.clear()
            self.patterns.clear()
            self.strategies = {'pattern': {}, 'risk': {}, 'combined': {}}
            self.results.clear()
            
            # Clear workspace files
            # Note: This is a destructive action. Should add more safety checks.
            for path in [self.strategy_manager.strategies_path, self.strategy_manager.results_path, self.strategy_manager.patterns_path]:
                if os.path.exists(path):
                    # Remove subdirectories for results
                    if path == self.strategy_manager.results_path:
                        for folder in os.listdir(path):
                            folder_path = os.path.join(path, folder)
                            if os.path.isdir(folder_path):
                                for f in os.listdir(folder_path):
                                    os.remove(os.path.join(folder_path, f))
                                os.rmdir(folder_path)
                    else:
                        for f in os.listdir(path):
                            os.remove(os.path.join(path, f))

            self._log("New workspace created")
            self._update_dashboard()
            
    def save_workspace(self):
        """Save current workspace"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Workspace", "", "JSON Files (*.json)"
        )
        
        if filepath:
            workspace = {
                'created_at': datetime.now().isoformat(),
                'datasets': list(self.datasets.keys()),
                'patterns': {name: p.__class__.__name__ for name, p in self.patterns.items()},
                'strategies': {
                    stype: {sid: s.to_dict() for sid, s in strategies.items()}
                    for stype, strategies in self.strategies.items()
                },
                'results': self.results
            }
            
            with open(filepath, 'w') as f:
                json.dump(workspace, f, indent=2, default=str)
                
            self._log(f"Workspace saved to {filepath}")
            
    def open_workspace(self):
        """Open saved workspace"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Workspace", "", "JSON Files (*.json)"
        )
        
        if filepath:
            with open(filepath, 'r') as f:
                workspace = json.load(f)
                
            # TODO: Implement workspace loading
            self._log(f"Workspace loaded from {filepath}")
            self._update_dashboard()
            
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Trading Strategy Hub",
            "Trading Strategy Hub v1.0\n\n"
            "A comprehensive platform for building, testing, and combining trading strategies.\n\n"
            "Features:\n"
            "• Advanced pattern recognition\n"
            "• Statistical strategy validation\n"
            "• Multi-timeframe analysis\n"
            "• Risk management integration\n"
            "• Machine learning ready\n\n"
            "© 2024 Trading Technologies"
        )
        
    def _log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_viewer.append(f"[{timestamp}] {message}")
        logger.info(message)
        
    def _update_dashboard(self):
        """Update dashboard statistics1"""
        self.status_labels['datasets'].setText(str(len(self.datasets)))
        self.status_labels['actions'].setText(str(len(self.actions)))
        self.status_labels['patterns'].setText(str(len(self.patterns)))
        
        # Fix the strategies count calculation to handle non-dictionary values
        total_strategies = 0
        for strategy_type, strategies in self.strategies.items():
            if isinstance(strategies, dict):
                total_strategies += len(strategies)
            else:
                # If it's not a dict, it's likely a strategy object that was incorrectly added
                # Remove it and log the issue
                if hasattr(strategies, 'id') and hasattr(strategies, 'type'):
                    # Move it to the correct location
                    correct_type = strategies.type
                    if correct_type not in self.strategies:
                        self.strategies[correct_type] = {}
                    self.strategies[correct_type][strategies.id] = strategies
                # Remove the incorrect entry
                del self.strategies[strategy_type]
        
        self.status_labels['strategies'].setText(str(total_strategies))
        
        self.status_labels['combined'].setText(str(len(self.strategies.get('combined', {}))))
        self.status_labels['backtests'].setText(str(len(self.results)))
        
        # Calculate average success rate
        if self.results:
            success_rates = [r.get('win_rate', 0) for r in self.results.values()]
            avg_success = sum(success_rates) / len(success_rates) * 100
            self.status_labels['success_rate'].setText(f"{avg_success:.1f}%")
        else:
            self.status_labels['success_rate'].setText("N/A")
            
        # Count open windows
        open_count = len([w for w in self.open_windows if w.isVisible()])
        self.status_labels['windows'].setText(str(open_count))
        
    def closeEvent(self, event):
        """Handle application close"""
        # Close all open windows
        for window in self.open_windows:
            window.close()
            
        self._save_last_session()
        event.accept()

    def _load_strategies_and_results(self):
        """Load all saved strategies and results from the workspace."""
        # Load strategies
        loaded_strategies = self.strategy_manager.load_strategies()
        for sid, strategy in loaded_strategies.items():
            # Determine strategy type
            strategy_type = 'pattern'  # Default type
            if hasattr(strategy, 'type'):
                strategy_type = strategy.type
            elif hasattr(strategy, '__class__'):
                class_name = strategy.__class__.__name__.lower()
                if 'risk' in class_name:
                    strategy_type = 'risk'
                elif 'combined' in class_name:
                    strategy_type = 'combined'
                else:
                    strategy_type = 'pattern'
            # Only set fallback name if missing or blank, never overwrite a valid name
            if not hasattr(strategy, 'name') or not strategy.name or not strategy.name.strip():
                strategy.name = f"Strategy_{sid}"
            self.strategies[strategy_type][sid] = strategy
        
        # Load patterns
        self.patterns.update(self.strategy_manager.load_patterns())
        
        # Load results
        self.results = self.strategy_manager.load_all_results()
        
        count_s = sum(len(s) for s in self.strategies.values())
        count_p = len(self.patterns)
        count_r = len(self.results)
        self._log(f"Loaded {count_s} strategies, {count_p} patterns, and {count_r} result sets from workspace.")
        self._update_dashboard()

    def _load_last_session(self):
        """Load last session info (e.g., last loaded dataset) from file."""
        try:
            with open(self.last_session_file, 'r') as f:
                session = json.load(f)
                last_dataset = session.get('last_dataset')
                if last_dataset and last_dataset in self.datasets:
                    self.last_loaded_dataset = last_dataset
        except Exception:
            pass

    def _save_last_session(self):
        """Save last session info (e.g., last loaded dataset) to file."""
        try:
            session = {'last_dataset': self.last_loaded_dataset}
            with open(self.last_session_file, 'w') as f:
                json.dump(session, f)
        except Exception:
            pass
