"""
gui/results_viewer_window.py
============================
Window for viewing and comparing backtest results
"""

import sys
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyqtgraph as pg
from scipy import stats
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplfinance as mpf


class ResultsViewerWindow(QMainWindow):
    """Window for viewing and comparing results"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Results Viewer")
        self.setGeometry(500, 200, 1200, 800)
        
        # Results storage
        self.results_history = {}
        
        # Setup UI
        self._setup_ui()
        
        # Load any existing results from disk
        self._refresh_results_from_disk()
        
        # Apply stylesheet
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
        layout = QVBoxLayout(central_widget)
        # Results selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select Results:"))
        self.results_combo = QComboBox()
        selection_layout.addWidget(self.results_combo)
        # Add Load Selected Result button
        self.load_result_btn = QPushButton("Load Selected Result")
        self.load_result_btn.clicked.connect(self._on_load_selected_result)
        selection_layout.addWidget(self.load_result_btn)
        self.compare_check = QCheckBox("Compare Mode")
        self.compare_check.toggled.connect(self._toggle_compare_mode)
        selection_layout.addWidget(self.compare_check)
        self.delete_btn = QPushButton("Delete Result")
        self.delete_btn.clicked.connect(self._delete_result)
        selection_layout.addWidget(self.delete_btn)
        self.run_backtest_btn = QPushButton("Run Backtest")
        self.run_backtest_btn.clicked.connect(self._run_backtest_on_strategy)
        selection_layout.addWidget(self.run_backtest_btn)
        
        # Add refresh button
        self.refresh_btn = QPushButton("🔄 Refresh from Disk")
        self.refresh_btn.clicked.connect(self._refresh_results_from_disk)
        selection_layout.addWidget(self.refresh_btn)
        selection_layout.addStretch()
        layout.addLayout(selection_layout)
        # Main content area
        self.content_stack = QStackedWidget()
        self.single_view = self._create_single_view()
        self.content_stack.addWidget(self.single_view)
        self.compare_view = self._create_compare_view()
        self.content_stack.addWidget(self.compare_view)
        layout.addWidget(self.content_stack)
        # Export section
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        export_layout.addWidget(self.export_btn)
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.clicked.connect(self._generate_report)
        export_layout.addWidget(self.generate_report_btn)
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
    def _create_single_view(self) -> QWidget:
        """Create single result view"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Result info
        self.result_info = QTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMaximumHeight(150)
        layout.addWidget(self.result_info)
        
        # Tabs for different views
        self.single_tabs = QTabWidget()
        
        # Performance tab
        perf_widget = QWidget()
        perf_layout = QVBoxLayout()
        
        # Metrics grid
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout()
        
        self.single_metrics = {}
        metrics = [
            ('Total Return', 'total_return', 0, 0),
            ('Sharpe Ratio', 'sharpe_ratio', 0, 1),
            ('Max Drawdown', 'max_drawdown', 0, 2),
            ('Win Rate', 'win_rate', 1, 0),
            ('Profit Factor', 'profit_factor', 1, 1),
            ('Total Trades', 'total_trades', 1, 2),
            ('Avg Win', 'avg_win', 2, 0),
            ('Avg Loss', 'avg_loss', 2, 1),
            ('Expectancy', 'expectancy', 2, 2)
        ]
        
        for label, key, row, col in metrics:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            
            title = QLabel(label)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(title)
            
            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            container_layout.addWidget(value_label)
            
            self.single_metrics[key] = value_label
            metrics_layout.addWidget(container, row, col)
            
        metrics_group.setLayout(metrics_layout)
        perf_layout.addWidget(metrics_group)
        
        perf_widget.setLayout(perf_layout)
        self.single_tabs.addTab(perf_widget, "Performance")
        
        # Charts tab
        charts_widget = QWidget()
        charts_layout = QVBoxLayout()
        
        # Equity curve
        self.single_equity_chart = pg.PlotWidget()
        self.single_equity_chart.setLabel('left', 'Equity')
        self.single_equity_chart.setLabel('bottom', 'Time')
        self.single_equity_chart.showGrid(True, True)
        charts_layout.addWidget(QLabel("Equity Curve:"))
        charts_layout.addWidget(self.single_equity_chart)
        
        # Drawdown chart
        self.single_drawdown_chart = pg.PlotWidget()
        self.single_drawdown_chart.setLabel('left', 'Drawdown %')
        self.single_drawdown_chart.setLabel('bottom', 'Time')
        self.single_drawdown_chart.showGrid(True, True)
        charts_layout.addWidget(QLabel("Drawdown:"))
        charts_layout.addWidget(self.single_drawdown_chart)
        
        charts_widget.setLayout(charts_layout)
        self.single_tabs.addTab(charts_widget, "Charts")
        
        # Strategy details tab
        details_widget = QWidget()
        details_layout = QVBoxLayout()
        
        self.strategy_details = QTextEdit()
        self.strategy_details.setReadOnly(True)
        details_layout.addWidget(self.strategy_details)
        
        details_widget.setLayout(details_layout)
        self.single_tabs.addTab(details_widget, "Strategy Details")

        # Spec Metrics tab
        spec_widget = QWidget()
        spec_layout = QVBoxLayout()
        
        # Spec-compliant metrics grid
        spec_metrics_group = QGroupBox("Spec-Compliant Metrics")
        spec_metrics_layout = QGridLayout()
        
        self.spec_metrics = {}
        spec_metrics = [
            ('S_adj Scores', 'S_adj_scores', 0, 0),
            ('S_net Scores', 'S_net_scores', 0, 1),
            ('Per-Zone Strengths', 'per_zone_strengths', 0, 2),
            ('Momentum Scores', 'momentum_scores', 1, 0),
            ('Volatility Scores', 'volatility_scores', 1, 1),
            ('Imbalance Scores', 'imbalance_scores', 1, 2),
            ('Enhanced Momentum', 'enhanced_momentum', 2, 0),
            ('Avg S_adj', 'avg_S_adj', 2, 1),
            ('Max S_adj', 'max_S_adj', 2, 2)
        ]
        
        for label, key, row, col in spec_metrics:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            
            title = QLabel(label)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(title)
            
            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E86AB;")
            container_layout.addWidget(value_label)
            
            self.spec_metrics[key] = value_label
            spec_metrics_layout.addWidget(container, row, col)
            
        spec_metrics_group.setLayout(spec_metrics_layout)
        spec_layout.addWidget(spec_metrics_group)
        
        # Strategy parameters display
        params_group = QGroupBox("Strategy Parameters")
        params_layout = QVBoxLayout()
        self.strategy_params_display = QTextEdit()
        self.strategy_params_display.setReadOnly(True)
        self.strategy_params_display.setMaximumHeight(200)
        params_layout.addWidget(self.strategy_params_display)
        params_group.setLayout(params_layout)
        spec_layout.addWidget(params_group)
        
        spec_widget.setLayout(spec_layout)
        self.single_tabs.addTab(spec_widget, "Spec Metrics")

        # Trade Returns tab
        returns_widget = QWidget()
        returns_layout = QVBoxLayout()
        returns_layout.addWidget(QLabel("Trade Returns Distribution (PnL):"))
        self.returns_hist = pg.PlotWidget()
        self.returns_hist.setLabel('left', 'Frequency')
        self.returns_hist.setLabel('bottom', 'PnL')
        returns_layout.addWidget(self.returns_hist)
        returns_widget.setLayout(returns_layout)
        self.single_tabs.addTab(returns_widget, "Trade Returns")

        # MAE Analysis tab
        mae_widget = QWidget()
        mae_layout = QVBoxLayout()
        mae_layout.addWidget(QLabel("MAE vs. PnL per Trade:"))
        self.mae_plot = pg.PlotWidget()
        self.mae_plot.setLabel('left', 'Profit/Loss (PnL)')
        self.mae_plot.setLabel('bottom', 'Max Adverse Excursion (MAE)')
        mae_layout.addWidget(self.mae_plot)
        self.mae_placeholder = QLabel("No MAE data available in trades.")
        self.mae_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mae_layout.addWidget(self.mae_placeholder)
        mae_widget.setLayout(mae_layout)
        self.single_tabs.addTab(mae_widget, "MAE Analysis")

        # Heatmap tab
        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout()
        
        # Heatmap controls
        controls_layout = QHBoxLayout()
        
        # Time binning control
        controls_layout.addWidget(QLabel("Time Binning:"))
        self.time_binning_combo = QComboBox()
        self.time_binning_combo.addItems(["1 minute", "5 minutes", "15 minutes", "1 hour", "4 hours", "1 day"])
        self.time_binning_combo.setCurrentText("15 minutes")
        self.time_binning_combo.currentTextChanged.connect(self._on_heatmap_control_changed)
        controls_layout.addWidget(self.time_binning_combo)
        
        # Color scheme control
        controls_layout.addWidget(QLabel("Color Scheme:"))
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Spectral"])
        self.color_scheme_combo.setCurrentText("Viridis")
        self.color_scheme_combo.currentTextChanged.connect(self._on_heatmap_control_changed)
        controls_layout.addWidget(self.color_scheme_combo)
        
        # Show legend checkbox
        self.show_legend_check = QCheckBox("Show Legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.toggled.connect(self._on_heatmap_control_changed)
        controls_layout.addWidget(self.show_legend_check)
        
        controls_layout.addStretch()
        heatmap_layout.addLayout(controls_layout)
        
        # Heatmap plot
        self.heatmap_plot = pg.PlotWidget()
        self.heatmap_plot.setLabel('left', 'Building Blocks')
        self.heatmap_plot.setLabel('bottom', 'Time')
        self.heatmap_plot.showGrid(True, True)
        heatmap_layout.addWidget(self.heatmap_plot)
        
        # Heatmap info
        self.heatmap_info = QLabel("No heatmap data available.")
        self.heatmap_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heatmap_layout.addWidget(self.heatmap_info)
        
        heatmap_widget.setLayout(heatmap_layout)
        self.single_tabs.addTab(heatmap_widget, "Heatmap")

        # State Change Heatmap tab
        state_heatmap_widget = self._create_state_heatmap_tab()
        self.single_tabs.addTab(state_heatmap_widget, "State Changes")

        # Playback tab
        playback_widget = self._create_playback_tab()
        self.single_tabs.addTab(playback_widget, "Playback")
        
        layout.addWidget(self.single_tabs)
        
        widget.setLayout(layout)
        return widget
        
    def _create_compare_view(self) -> QWidget:
        """Create comparison view"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Results selection for comparison
        selection_group = QGroupBox("Select Results to Compare")
        selection_layout = QVBoxLayout()
        
        self.compare_list = QListWidget()
        self.compare_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        selection_layout.addWidget(self.compare_list)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Comparison table
        self.compare_table = QTableWidget()
        layout.addWidget(QLabel("Comparison Table:"))
        layout.addWidget(self.compare_table)
        
        # Comparison charts
        charts_layout = QHBoxLayout()
        
        # Equity curves comparison
        self.compare_equity_chart = pg.PlotWidget()
        self.compare_equity_chart.setLabel('left', 'Equity')
        self.compare_equity_chart.setLabel('bottom', 'Time')
        self.compare_equity_chart.addLegend()
        charts_layout.addWidget(self.compare_equity_chart)
        
        # Metrics comparison bar chart
        self.compare_metrics_chart = pg.PlotWidget()
        self.compare_metrics_chart.setLabel('left', 'Value')
        self.compare_metrics_chart.setLabel('bottom', 'Metric')
        charts_layout.addWidget(self.compare_metrics_chart)
        
        layout.addLayout(charts_layout)
        
        # Update comparison button
        self.update_compare_btn = QPushButton("Update Comparison")
        self.update_compare_btn.clicked.connect(self._update_comparison)
        layout.addWidget(self.update_compare_btn)
        
        widget.setLayout(layout)
        return widget
        
    def add_result(self, result_data: Dict[str, Any], strategy_name: str):
        """Add a new result"""
        import os
        # Delete all previous results for this strategy
        folder = strategy_name.replace(' ', '_')
        results_dir = os.path.join('workspaces', 'results', folder)
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    try:
                        os.remove(os.path.join(results_dir, filename))
                    except Exception as e:
                        print(f"[WARNING] Could not delete old result: {filename}: {e}")
        # Save the new result as before
        result_id = result_data.get('result_display_name')
        if not result_id:
            result_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_history[result_id] = {
            'data': result_data,
            'strategy': strategy_name,
            'timestamp': datetime.now(),
            'id': result_id
        }
        display_name = result_data.get('result_display_name', f"{strategy_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.results_combo.addItem(display_name, result_id)
        self.compare_list.addItem(display_name)
        self._save_results_history()
        
        # Automatically display the result when added
        self._display_result(result_data)
        
    def _on_load_selected_result(self):
        """Load and display the currently selected result from the combo box."""
        index = self.results_combo.currentIndex()
        if index < 0:
            return
        result_id = self.results_combo.currentData()
        if result_id and result_id in self.results_history:
            result_data = self.results_history[result_id]
            if 'data' in result_data:
                actual_result = result_data['data']
                self._display_result(actual_result)
            else:
                self._display_result(result_data)

    def _display_result(self, result_data: Dict[str, Any]):
        """Display a single result"""
        # Store current result data for heatmap updates
        self.current_result_data = result_data
        
        # Update performance metrics
        self._update_performance_metrics(result_data)
        
        # Update strategy details
        self._update_strategy_details(result_data)
        
        # Update equity chart
        self._update_equity_chart(result_data)
        
        # Update drawdown chart
        self._update_drawdown_chart(result_data)
        
        # Update trade returns chart
        self._update_trade_returns_chart(result_data)
        
        # Initialize playback data with multi-timeframe support
        self._initialize_playback_data(result_data)
        
        # Update MAE analysis
        self._update_mae_analysis(result_data)
        
        # Update heatmap
        self._update_heatmap(result_data)
        
        # Update state change heatmap
        self._update_state_heatmap(result_data)
        
    def _update_equity_chart(self, data: Dict[str, Any]):
        """Update the equity chart"""
        self.single_equity_chart.clear()
        
        # Access equity curve directly from result data
        equity_curve = data.get('equity_curve', [])
        
        # Handle both old list format and new Series format
        if isinstance(equity_curve, pd.Series):
            if not equity_curve.empty:
                x = np.arange(len(equity_curve))
                self.single_equity_chart.plot(x, equity_curve.values, pen='w')
        elif isinstance(equity_curve, list):
            if equity_curve:
                x = np.arange(len(equity_curve))
                self.single_equity_chart.plot(x, equity_curve, pen='w')
    
    def _update_drawdown_chart(self, data: Dict[str, Any]):
        """Update the drawdown chart"""
        self.single_drawdown_chart.clear()
        
        # Access equity curve directly from result data
        equity_curve = data.get('equity_curve', [])
        
        # Handle both old list format and new Series format
        if isinstance(equity_curve, pd.Series):
            if not equity_curve.empty:
                equity_series = equity_curve
        elif isinstance(equity_curve, list):
            if equity_curve:
                equity_series = pd.Series(equity_curve)
        else:
            return  # No valid equity curve data
            
        if 'equity_series' in locals():
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max * 100
            x = np.arange(len(equity_curve))
            self.single_drawdown_chart.plot(x, drawdown, pen='r', fillLevel=0, brush=(255, 0, 0, 50))
        
    def _update_trade_returns_chart(self, result_data: Dict[str, Any]):
        """Update the trade returns histogram chart"""
        self.returns_hist.clear()
        
        # Access trades directly from result data
        trades = result_data.get('trades', [])
        print(f"[DEBUG] Trade Returns Chart - Found {len(trades)} trades")
        
        if trades and isinstance(trades, list) and len(trades) > 0:
            print(f"[DEBUG] First trade keys: {list(trades[0].keys()) if trades[0] else 'No first trade'}")
            
            if 'pnl' in trades[0]:
                # Extract PnL values from trades
                pnl_values = [t['pnl'] for t in trades if 'pnl' in t]
                print(f"[DEBUG] Extracted {len(pnl_values)} PnL values")
                print(f"[DEBUG] PnL range: {min(pnl_values) if pnl_values else 'N/A'} to {max(pnl_values) if pnl_values else 'N/A'}")
                
                if pnl_values:
                    # Create histogram
                    hist, bins = np.histogram(pnl_values, bins=20, density=True)
                    
                    # Plot histogram bars
                    x = (bins[:-1] + bins[1:]) / 2  # Center of each bin
                    width = bins[1] - bins[0]
                    
                    # Color bars based on positive/negative PnL
                    for i, (h, x_pos) in enumerate(zip(hist, x)):
                        if x_pos >= 0:
                            color = (0, 255, 0, 150)  # Green for positive
                        else:
                            color = (255, 0, 0, 150)  # Red for negative
                        
                        bar = pg.QtWidgets.QGraphicsRectItem(x_pos - width/2, 0, width, h)
                        bar.setBrush(pg.mkBrush(color))
                        bar.setPen(pg.mkPen(color))
                        self.returns_hist.addItem(bar)
                    
                    # Add vertical line at zero
                    zero_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=2))
                    self.returns_hist.addItem(zero_line)
                    
                    # Add statistics text
                    mean_pnl = np.mean(pnl_values)
                    std_pnl = np.std(pnl_values)
                    stats_text = f"Mean: ${mean_pnl:.2f}\nStd: ${std_pnl:.2f}\nTrades: {len(pnl_values)}"
                    
                    # Add text item to chart
                    text_item = pg.TextItem(text=stats_text, color='w')
                    text_item.setPos(0.8 * max(bins), 0.8 * max(hist))
                    self.returns_hist.addItem(text_item)
                    
            else:
                print(f"[DEBUG] No 'pnl' key found in trades")
                # No trade data available
                text_item = pg.TextItem(text="No PnL data in trades", color='w')
                text_item.setPos(0, 0)
                self.returns_hist.addItem(text_item)
        else:
            print(f"[DEBUG] No trades found or trades is empty")
            # No trade data available
            text_item = pg.TextItem(text="No trade data available", color='w')
            text_item.setPos(0, 0)
            self.returns_hist.addItem(text_item)
    
    def _update_strategy_details(self, result: Dict[str, Any]):
        """Update strategy details with multi-timeframe information"""
        strategy = result.get('strategy', {})
        if not strategy:
            self.strategy_details.setPlainText("No strategy details available")
            return
            
        # Get multi-timeframe information
        multi_tf_data = result.get('multi_tf_data', {})
        strategy_timeframes = []
        
        # Handle both old dict format and new DataFrame format
        if isinstance(multi_tf_data, dict) and multi_tf_data:
            for tf_key in multi_tf_data.keys():
                if tf_key != 'execution':
                    strategy_timeframes.append(tf_key)
        elif isinstance(multi_tf_data, pd.DataFrame) and not multi_tf_data.empty:
            # New architecture returns DataFrame, treat as single timeframe
            strategy_timeframes.append('1min')
        
        details = f"Strategy: {result.get('strategy_name', 'Unknown')}\n"
        details += f"Display Timeframe: {result.get('timeframe', 'Unknown')}\n"
        
        if strategy_timeframes:
            details += f"Strategy Timeframes: {', '.join(strategy_timeframes)}\n"
            
            # Handle execution data differently for old vs new architecture
            if isinstance(multi_tf_data, dict):
                details += f"Execution Timeframe: {len(multi_tf_data.get('execution', []))} bars\n\n"
            elif isinstance(multi_tf_data, pd.DataFrame):
                details += f"Execution Timeframe: {len(multi_tf_data)} bars\n\n"
        
        # Strategy actions
        actions = strategy.get('actions', [])
        if actions:
            details += "Strategy Actions:\n"
            for i, action in enumerate(actions, 1):
                details += f"{i}. {action.get('name', 'Unknown Action')}\n"
                pattern = action.get('pattern', {})
                if pattern:
                    details += f"   Pattern: {pattern.get('name', 'Unknown')}\n"
                    details += f"   Type: {pattern.get('type', 'Unknown')}\n"
                time_range = action.get('time_range', {})
                if time_range:
                    details += f"   Time Range: {time_range.get('value', 0)} {time_range.get('unit', 'minutes')}\n"
                details += "\n"
        
        # Combination logic
        combination = strategy.get('combination', {})
        if combination:
            details += f"Combination Logic: {combination.get('type', 'Unknown')}\n"
            if combination.get('type') == 'WEIGHTED':
                weights = combination.get('weights', [])
                details += f"Weights: {weights}\n"
        
        # Performance summary
        details += f"\nPerformance Summary:\n"
        details += f"Total Return: {result.get('total_return', 0):.2%}\n"
        details += f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}\n"
        details += f"Max Drawdown: {result.get('max_drawdown', 0):.2%}\n"
        details += f"Total Trades: {result.get('total_trades', 0)}\n"
        details += f"Win Rate: {result.get('win_rate', 0):.2%}\n"
        details += f"Profit Factor: {result.get('profit_factor', 0):.2f}\n"
        
        self.strategy_details.setPlainText(details)
        
    def _toggle_compare_mode(self, checked: bool):
        """Toggle between single and comparison view"""
        if checked:
            self.content_stack.setCurrentIndex(1)
            self._populate_compare_list()
        else:
            self.content_stack.setCurrentIndex(0)
            
    def _populate_compare_list(self):
        """Populate comparison list"""
        self.compare_list.clear()
        
        for result_id, result in self.results_history.items():
            strategy_name = result.get('strategy', 'Unknown Strategy')
            item_text = f"{strategy_name} - {result['timestamp'].strftime('%Y-%m-%d %H:%M')}"
            self.compare_list.addItem(item_text)
            
    def _update_comparison(self):
        """Update comparison view"""
        selected_indices = [item.row() for item in self.compare_list.selectedIndexes()]
        
        if len(selected_indices) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 results to compare")
            return
            
        # Get selected results
        result_ids = list(self.results_history.keys())
        selected_results = [self.results_history[result_ids[i]] for i in selected_indices]
        
        # Update comparison table
        self._update_comparison_table(selected_results)
        
        # Update comparison charts
        self._update_comparison_charts(selected_results)
        
    def _update_comparison_table(self, results: List[Dict[str, Any]]):
        """Update comparison table"""
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                  'profit_factor', 'total_trades', 'avg_win', 'avg_loss']
        
        self.compare_table.setRowCount(len(metrics))
        self.compare_table.setColumnCount(len(results) + 1)
        
        # Set headers
        headers = ['Metric'] + [r.get('strategy', 'Unknown Strategy') for r in results]
        self.compare_table.setHorizontalHeaderLabels(headers)
        
        # Fill table
        for i, metric in enumerate(metrics):
            # Metric name
            self.compare_table.setItem(i, 0, QTableWidgetItem(metric.replace('_', ' ').title()))
            
            # Values for each result
            for j, result in enumerate(results):
                value = result['data'].get(metric, 0)
                
                if metric in ['total_return', 'max_drawdown', 'win_rate']:
                    text = f"{value:.2%}"
                elif metric in ['sharpe_ratio', 'profit_factor']:
                    text = f"{value:.2f}"
                elif metric == 'total_trades':
                    text = str(value)
                else:
                    text = f"${value:.2f}"
                    
                item = QTableWidgetItem(text)
                
                # Color coding
                if metric == 'total_return':
                    if value > 0:
                        item.setBackground(QColor(0, 255, 0, 50))
                    else:
                        item.setBackground(QColor(255, 0, 0, 50))
                        
                self.compare_table.setItem(i, j + 1, item)
                
    def _update_comparison_charts(self, results: List[Dict[str, Any]]):
        """Update comparison charts"""
        # Clear charts
        self.compare_equity_chart.clear()
        self.compare_metrics_chart.clear()
        
        # Plot equity curves
        colors = ['w', 'g', 'r', 'b', 'y', 'c', 'm']
        
        for i, result in enumerate(results):
            equity_curve = result['data'].get('equity_curve', [])
            
            # Handle both old list format and new Series format
            plot_data = None
            if isinstance(equity_curve, pd.Series):
                if not equity_curve.empty:
                    plot_data = equity_curve.values
            elif isinstance(equity_curve, list):
                if equity_curve:
                    plot_data = equity_curve
            
            if plot_data is not None:
                x = np.arange(len(plot_data))
                color = colors[i % len(colors)]
                strategy_name = result.get('strategy', 'Unknown Strategy')
                self.compare_equity_chart.plot(x, plot_data, pen=color, 
                                             name=strategy_name)
                
        # Plot metrics comparison
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'win_rate']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(results)
        
        for i, result in enumerate(results):
            values = [result['data'].get(m, 0) for m in metrics_to_plot]
            x_offset = x + (i - len(results)/2) * width
            
            color = colors[i % len(colors)]
            strategy_name = result.get('strategy', 'Unknown Strategy')
            self.compare_metrics_chart.plot(x_offset, values, pen=None, 
                                          symbol='s', symbolSize=20,
                                          symbolBrush=color, name=strategy_name)
            
        # Set x-axis labels
        ax = self.compare_metrics_chart.getAxis('bottom')
        ax.setTicks([[(i, m.replace('_', ' ').title()) for i, m in enumerate(metrics_to_plot)]])
        
    def _delete_result(self):
        """Delete selected result and its file from disk"""
        import os
        if self.results_combo.currentIndex() < 0:
            return
        reply = QMessageBox.question(self, "Delete Result", 
                                   "Are you sure you want to delete this result?",
                                   QMessageBox.StandardButton.Yes | 
                                   QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            result_id = self.results_combo.currentData()
            if result_id in self.results_history:
                # Try to delete the file from disk
                result = self.results_history[result_id]
                strategy_name = result.get('strategy', None)
                timestamp = result.get('timestamp', None)
                # Try to reconstruct the file path
                if strategy_name and timestamp:
                    # Try to find the file in workspaces/results/STRATEGY_NAME/
                    folder = strategy_name.replace(' ', '_')
                    # Try to extract the result file name from result_id
                    # result_id is usually STRATEGYNAME_result_YYYYMMDD_HHMMSS
                    if '_' in result_id:
                        file_part = result_id.split('_', 1)[1]
                        file_name = f"result_{file_part}.json"
                        file_path = os.path.join('workspaces', 'results', folder, file_name)
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                print(f"[INFO] Deleted result file: {file_path}")
                            except Exception as e:
                                print(f"[WARNING] Could not delete result file: {file_path}: {e}")
                del self.results_history[result_id]
                self.results_combo.removeItem(self.results_combo.currentIndex())
                self._save_results_history()

    def _export_results(self):
        """Export results"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if filepath:
            if filepath.endswith('.csv'):
                # Export as CSV
                data = []
                for result_id, result in self.results_history.items():
                    row = {
                        'ID': result_id,
                        'Strategy': result.get('strategy', 'Unknown Strategy'),
                        'Timestamp': result['timestamp'],
                        **result['data']
                    }
                    data.append(row)
                    
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
                
            elif filepath.endswith('.json'):
                # Export as JSON
                with open(filepath, 'w') as f:
                    json.dump(self.results_history, f, indent=2, default=str)
                    
            QMessageBox.information(self, "Export Complete", f"Results exported to {filepath}")
            
    def _generate_report(self):
        """Generate HTML report"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Generate Report", "", "HTML Files (*.html)"
        )
        
        if filepath:
            # Generate HTML report
            html = self._create_html_report()
            
            with open(filepath, 'w') as f:
                f.write(html)
                
            QMessageBox.information(self, "Report Generated", f"Report saved to {filepath}")
            
            # Open in browser
            import webbrowser
            webbrowser.open(filepath)
            
    def _create_html_report(self) -> str:
        """Create HTML report"""
        html = """
        <html>
        <head>
            <title>Trading Strategy Results Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .positive { color: green; }
                .negative { color: red; }
                .metric-box { display: inline-block; margin: 10px; padding: 10px; 
                             border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Trading Strategy Results Report</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <h2>Results Summary</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Date</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Total Trades</th>
                </tr>
        """
        
        for result_id, result in self.results_history.items():
            data = result['data']
            total_return = data.get('total_return', 0)
            
            html += f"""
                <tr>
                    <td>{result['strategy']}</td>
                    <td>{result['timestamp'].strftime('%Y-%m-%d')}</td>
                    <td class="{'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</td>
                    <td>{data.get('sharpe_ratio', 0):.2f}</td>
                    <td class="negative">{data.get('max_drawdown', 0):.2%}</td>
                    <td>{data.get('win_rate', 0):.2%}</td>
                    <td>{data.get('total_trades', 0)}</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
        
    def _save_results_history(self):
        """Save results history to file"""
        # Would implement saving to file
        pass
        
    def _load_results_history(self):
        """Load results history from file, but do NOT auto-select or auto-display any result."""
        print(f"[DEBUG] _load_results_history called")
        if self.parent_window and hasattr(self.parent_window, 'results'):
            print(f"[DEBUG] Parent window has {len(self.parent_window.results)} results")
            print(f"[DEBUG] Parent window result keys: {list(self.parent_window.results.keys())}")
            for result_id, result in self.parent_window.results.items():
                print(f"[DEBUG] Loading result {result_id} with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                display_name = result.get('result_display_name')
                if not display_name:
                    strategy_name = result.get('strategy_name', f'Strategy_{result_id}')
                    timestamp = result.get('timestamp', None)
                    display_name = f"{strategy_name} - {timestamp}"
                self.results_history[result_id] = {
                    'data': result,
                    'strategy': result.get('strategy_name', f'Strategy_{result_id}'),
                    'timestamp': result.get('timestamp', None),
                    'id': result_id
                }
                self.results_combo.addItem(display_name, result_id)
                self.compare_list.addItem(display_name)
            print(f"[DEBUG] Loaded {len(self.results_history)} results into history")
        else:
            print(f"[DEBUG] No parent window or no results found")

    def _run_backtest_on_strategy(self):
        """Run backtest on a strategy that doesn't have results yet"""
        if not self.parent_window or not hasattr(self.parent_window, 'strategies'):
            QMessageBox.warning(self, "Warning", "No strategies available")
            return
            
        # Get strategies without results
        strategies_without_results = []
        for strategy_type, strategies in self.parent_window.strategies.items():
            for strategy_id, strategy in strategies.items():
                # Check if this strategy has results
                has_results = False
                for result_id, result in self.results_history.items():
                    if result['strategy'] == strategy.name:
                        has_results = True
                        break
                
                if not has_results:
                    strategies_without_results.append((strategy_type, strategy))
        
        if not strategies_without_results:
            QMessageBox.information(self, "Info", "All strategies already have results")
            return
        
        # Let user select strategy
        strategy_names = [f"[{stype}] {strategy.name}" for stype, strategy in strategies_without_results]
        strategy_name, ok = QInputDialog.getItem(
            self, "Select Strategy", 
            "Choose a strategy to backtest:",
            strategy_names, 0, False
        )
        
        if not ok:
            return
            
        # Get selected strategy
        selected_index = strategy_names.index(strategy_name)
        strategy_type, strategy = strategies_without_results[selected_index]
        
        # Check if we have datasets
        if not self.parent_window.datasets:
            QMessageBox.warning(self, "Warning", "No datasets available for backtesting")
            return
        
        # Let user select dataset
        dataset_names = list(self.parent_window.datasets.keys())
        dataset_name, ok = QInputDialog.getItem(
            self, "Select Dataset", 
            "Choose a dataset for backtesting:",
            dataset_names, 0, False
        )
        
        if not ok:
            return
            
        # Get dataset
        dataset_info = self.parent_window.datasets[dataset_name]
        if 'data' not in dataset_info:
            QMessageBox.warning(self, "Warning", "Selected dataset has no data")
            return
        
        # Run backtest
        try:
            from strategies.strategy_builders import BacktestEngine
            engine = BacktestEngine()
            
            # Show progress dialog
            progress = QProgressDialog("Running backtest...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Run backtest
            results = engine.run_backtest(
                strategy, 
                dataset_info['data'],
                initial_capital=100000,
                risk_per_trade=0.02
            )
            
            progress.setValue(100)
            
            # Add strategy name to results
            results['strategy_name'] = strategy.name
            
            # Add to results
            self.add_result(results, strategy.name)
            
            # Also add to parent window
            if self.parent_window and hasattr(self.parent_window, 'results'):
                result_id = datetime.now().strftime("%Y%m%d%H%M%S")
                self.parent_window.results[result_id] = results
            
            QMessageBox.information(self, "Success", 
                                  f"Backtest completed: {results['total_trades']} trades, "
                                  f"{results['total_return']:.2%} return")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Backtest failed: {str(e)}")

    def _create_playback_tab(self) -> QWidget:
        """Create the playback tab with full playback controls, overlays, and timeframe selection."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        widget = QWidget()
        layout = QVBoxLayout()
        # --- Controls ---
        control_panel = QHBoxLayout()
        # Play/Pause/Stop
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setCheckable(True)
        control_panel.addWidget(self.play_btn)
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self._pause_playback)
        control_panel.addWidget(self.pause_btn)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        control_panel.addWidget(self.stop_btn)
        # Speed control
        control_panel.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        control_panel.addWidget(self.speed_combo)

        # Overlay toggles
        self.show_price = QCheckBox("Price")
        self.show_price.setChecked(True)
        self.show_price.toggled.connect(self._update_chart)
        control_panel.addWidget(self.show_price)
        self.show_zones = QCheckBox("Zones")
        self.show_zones.setChecked(True)
        self.show_zones.toggled.connect(self._update_chart)
        control_panel.addWidget(self.show_zones)
        self.show_indicators = QCheckBox("Indicators")
        self.show_indicators.setChecked(False)
        self.show_indicators.toggled.connect(self._update_chart)
        control_panel.addWidget(self.show_indicators)
        self.show_patterns = QCheckBox("Patterns")
        self.show_patterns.setChecked(False)
        self.show_patterns.toggled.connect(self._update_chart)
        control_panel.addWidget(self.show_patterns)
        self.show_signals = QCheckBox("Entry/Exit")
        self.show_signals.setChecked(True)
        self.show_signals.toggled.connect(self._update_chart)
        control_panel.addWidget(self.show_signals)
        self.show_volume = QCheckBox("Volume")
        self.show_volume.setChecked(False)
        self.show_volume.toggled.connect(self._update_chart)
        control_panel.addWidget(self.show_volume)
        control_panel.addStretch()
        layout.addLayout(control_panel)
        # Timeline slider
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(QLabel("Timeline:"))
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self._on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider)
        self.time_label = QLabel("00:00:00")
        self.time_label.setMinimumWidth(80)
        timeline_layout.addWidget(self.time_label)
        layout.addLayout(timeline_layout)
        # Chart area
        chart_layout = QHBoxLayout()
        self.playback_fig, self.playback_ax = plt.subplots(figsize=(10, 5))
        self.playback_canvas = FigureCanvas(self.playback_fig)
        self.playback_toolbar = NavigationToolbar(self.playback_canvas, widget)
        chart_container = QWidget()
        chart_container_layout = QVBoxLayout(chart_container)
        chart_container_layout.addWidget(self.playback_toolbar)
        chart_container_layout.addWidget(self.playback_canvas)
        chart_layout.addWidget(chart_container)
        layout.addLayout(chart_layout)
        # Playback state
        self.playback_data = None
        self.current_index = 0
        self.is_playing = False
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback)
        self.playback_speed = 1000
        widget.setLayout(layout)
        return widget

    # --- Playback logic (restore previous methods for play/pause/slider/interval) ---
    def _toggle_playback(self):
        if self.playback_data is None:
            return
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()
    def _start_playback(self):
        if self.playback_data is None:
            return
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.play_btn.setChecked(True)
        self.playback_timer.start(self.playback_speed)
    def _pause_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.play_btn.setChecked(False)
        self.playback_timer.stop()
    def _stop_playback(self):
        self._pause_playback()
        self.current_index = 0
        self.timeline_slider.setValue(0)
        self._update_chart()
    def _on_speed_changed(self, speed_text: str):
        speed_map = {"0.25x": 4000, "0.5x": 2000, "1x": 1000, "2x": 500, "4x": 250, "8x": 125}
        self.playback_speed = speed_map.get(speed_text, 1000)
        if self.is_playing:
            self.playback_timer.setInterval(self.playback_speed)

    def _on_timeline_changed(self, value: int):
        if self.playback_data is None:
            return
        max_index = len(self.playback_data) - 1
        if max_index <= 0:
            return
        target_index = int((value / 100.0) * max_index)
        target_index = max(0, min(target_index, max_index))
        if target_index != self.current_index:
            self.current_index = target_index
            self._update_chart()
    def _update_playback(self):
        if self.playback_data is None:
            self._stop_playback()
            return
        if self.current_index >= len(self.playback_data) - 1:
            self._stop_playback()
            return
        self.current_index += 1
        max_index = len(self.playback_data) - 1
        if max_index > 0:
            slider_value = int((self.current_index / max_index) * 100)
            slider_value = max(0, min(slider_value, 100))
            self.timeline_slider.setValue(slider_value)
        self._update_chart()

    def _update_chart(self):
        if self.playback_data is not None and hasattr(self, 'current_result'):
            # Show data from the start up to the current playback index
            if self.current_index < 0:
                self.current_index = 0
            end_idx = min(len(self.playback_data), self.current_index + 1)
            window_data = self.playback_data.iloc[:end_idx]
            print(f"[DEBUG] _update_chart: current_index={self.current_index}, end_idx={end_idx}, window_data.shape={window_data.shape}")
            self._draw_playback_chart(window_data, self.current_result)
        else:
            print(f"[DEBUG] _update_chart: playback_data is None or no current_result")
            print(f"[DEBUG] playback_data is None: {self.playback_data is None}")
            print(f"[DEBUG] hasattr current_result: {hasattr(self, 'current_result')}")

    def _draw_playback_chart(self, window_data, result_data):
        """Draw the current playback window using the same overlays as the backtester (zones, signals, etc)."""
        import mplfinance as mpf
        import pandas as pd
        self.playback_ax.clear()
        df = window_data.copy()
        
        print(f"[DEBUG] _draw_playback_chart called with {len(df)} bars")
        print(f"[DEBUG] Data columns: {list(df.columns)}")
        print(f"[DEBUG] Data range: {df.index[0] if len(df) > 0 else 'Empty'} to {df.index[-1] if len(df) > 0 else 'Empty'}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df.index = pd.to_datetime(df['datetime'])
            elif 'Date' in df.columns:
                df.index = pd.to_datetime(df['Date'])
            else:
                df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
        
        # Only plot if enough data
        if len(df) > 0 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            print(f"[DEBUG] Plotting candlesticks for {len(df)} bars")
            
            # Price (candlesticks)
            if getattr(self, 'show_price', None) and self.show_price.isChecked():
                plot_df = df[['open', 'high', 'low', 'close']].copy()
                plot_df.columns = ['Open', 'High', 'Low', 'Close']
                
                # FIX: Properly draw candlesticks using mplfinance
                try:
                    # Use the axis directly instead of returnfig
                    mpf.plot(plot_df, type='candle', ax=self.playback_ax, style='charles', 
                            show_nontrading=True, returnfig=False)
                    print(f"[DEBUG] Successfully plotted candlesticks")
                except Exception as e:
                    print(f"[DEBUG] Error plotting candlesticks: {e}")
                    # Fallback: plot as line chart
                    self.playback_ax.plot(df.index, df['close'], color='black', linewidth=1)
                    print(f"[DEBUG] Fallback to line chart")
            
            # Volume
            if getattr(self, 'show_volume', None) and self.show_volume.isChecked() and 'volume' in df.columns:
                self.playback_ax2 = self.playback_ax.twinx()
                self.playback_ax2.bar(df.index, df['volume'], color='gray', alpha=0.2, width=0.8)
            
            # Zones
            if getattr(self, 'show_zones', None) and self.show_zones.isChecked():
                zones = result_data.get('zones', [])
                print(f"[DEBUG] Zone plotting: Found {len(zones)} zones")
                for i, zone in enumerate(zones):
                    print(f"[DEBUG] Zone {i}: {zone}")
                    
                    if 'zones' in zone and isinstance(zone['zones'], list):
                        zone_index = zone.get('index', 0)
                        print(f"[DEBUG] Complex zone {i}: index={zone_index}, subzones={len(zone['zones'])}")
                        if zone_index < len(df.index):
                            start_time = df.index[zone_index]
                            end_idx = min(zone_index + 5, len(df) - 1)
                            end_time = df.index[end_idx]
                            for subzone in zone['zones']:
                                if 'zone_min' in subzone and 'zone_max' in subzone:
                                    zone_min = subzone['zone_min']
                                    zone_max = subzone['zone_max']
                                    print(f"[DEBUG] Plotting complex subzone: {zone_min} to {zone_max} at {start_time} to {end_time}")
                                    self.playback_ax.fill_between([start_time, end_time], zone_min, zone_max, color='blue', alpha=0.08, zorder=1)
                                    if 'comb_centers' in subzone:
                                        for c in subzone['comb_centers']:
                                            self.playback_ax.hlines(c, start_time, end_time, color='orange', linestyle='--', alpha=0.5, linewidth=1, zorder=2)
                        else:
                            print(f"[DEBUG] Complex zone {i}: index {zone_index} out of range (max: {len(df.index)})")
                    else:
                        zone_min = zone.get('zone_min')
                        zone_max = zone.get('zone_max')
                        zone_idx = zone.get('index') or zone.get('creation_index')  # Try both 'index' and 'creation_index'
                        comb_centers = zone.get('comb_centers', [])
                        print(f"[DEBUG] Simple zone {i}: min={zone_min}, max={zone_max}, idx={zone_idx}, combs={len(comb_centers)}")
                        
                        if zone_min is not None and zone_max is not None and zone_idx is not None and 0 <= zone_idx < len(df):
                            # Get zone parameters for decay calculation
                            zone_type = zone.get('zone_type', 'Unknown')
                            zone_direction = zone.get('zone_direction', 'neutral')
                            initial_strength = zone.get('initial_strength', 1.0)
                            gamma = zone.get('gamma', 0.95)
                            tau_bars = zone.get('tau_bars', 50)
                            drop_threshold = zone.get('drop_threshold', 0.01)
                            
                            # Calculate dynamic zone duration using decay system (match backtester)
                            start_time = df.index[zone_idx]
                            
                            # Use zone-specific decay parameters to calculate end index
                            gamma = zone.get('gamma', 0.95)
                            tau_bars = zone.get('tau_bars', 50)
                            drop_threshold = zone.get('drop_threshold', 0.01)
                            initial_strength = zone.get('initial_strength', 1.0)
                            
                            # Calculate end index based on decay parameters
                            end_idx = zone_idx
                            for future_idx in range(zone_idx + 1, min(zone_idx + tau_bars + 1, len(df))):
                                bars_since_creation = future_idx - zone_idx
                                if bars_since_creation >= tau_bars:
                                    break
                                current_strength = initial_strength * (gamma ** bars_since_creation)
                                if current_strength < (drop_threshold * initial_strength):
                                    break
                                end_idx = future_idx
                            
                            end_time = df.index[end_idx]
                            
                            # Calculate alpha based on current strength (match backtester)
                            current_strength = initial_strength * (gamma ** 0)  # At creation
                            alpha = max(0.2, min(0.5, current_strength))
                            
                            # Color zones based on direction (match backtester)
                            zone_color = 'blue'
                            # Print debug info for zone type and direction before color assignment
                            print(f"[DEBUG] Zone color assignment: zone_type={zone_type}, zone_direction={zone_direction}")
                            if zone_type == 'FVG' or zone_type == 'VWAP':
                                if zone_direction == 'bullish':
                                    zone_color = 'green'
                                elif zone_direction == 'bearish':
                                    zone_color = 'red'
                                else:
                                    zone_color = 'blue'  # Fallback for neutral
                            elif zone_type == 'Imbalance':
                                zone_color = 'magenta'  # Magenta for imbalance zones
                            else:
                                zone_color = 'blue'  # All other zone types remain blue
                            
                            print(f"[DEBUG] Plotting zone: {zone_color} {zone_direction} FVG at {start_time} to {end_time}, alpha={alpha}")
                            self.playback_ax.fill_between([start_time, end_time], zone_min, zone_max, color=zone_color, alpha=alpha, zorder=10)
                            
                            # Plot comb centers as horizontal lines (match backtester)
                            if comb_centers:
                                for comb_price in comb_centers:
                                    if zone_min <= comb_price <= zone_max:
                                        self.playback_ax.plot([start_time, end_time], [comb_price, comb_price], 
                                               color='orange', linewidth=1, alpha=0.8, linestyle='--', zorder=11)
                                        # Add small marker at the comb center
                                        self.playback_ax.scatter(start_time, comb_price, color='orange', s=20, alpha=0.9, zorder=12)
                        else:
                            print(f"[DEBUG] Simple zone {i}: missing data or index out of range")
                            print(f"[DEBUG]   zone_min: {zone_min}, zone_max: {zone_max}, zone_idx: {zone_idx}")
                            print(f"[DEBUG]   index valid: {zone_idx is not None and 0 <= zone_idx < len(df)}")
                
                if zones:
                    self.playback_ax.plot([], [], color='blue', alpha=0.15, label='Zone')
                    self.playback_ax.plot([], [], color='orange', linestyle='--', alpha=0.7, label='Micro-Comb Peak')
                else:
                    print(f"[DEBUG] No zones found in result_data")
            
            # Add VWAP indicator to results viewer only if it's part of the strategy
            # Check if VWAP is in the strategy components
            has_vwap = False
            strategy_params = result_data.get('strategy_params', {})
            component_summary = result_data.get('component_summary', {})
            action_details = result_data.get('action_details', {})
            
            # Check various sources for VWAP
            if 'filters' in strategy_params:
                for filter_config in strategy_params['filters']:
                    if filter_config.get('type', '').lower() == 'vwap':
                        has_vwap = True
                        break
            
            if not has_vwap and component_summary:
                filters = component_summary.get('filters', [])
                if 'vwap' in [f.lower() for f in filters]:
                    has_vwap = True
            
            if not has_vwap and action_details:
                for action_name in action_details.keys():
                    if 'vwap' in action_name.lower():
                        has_vwap = True
                        break
            
            # Only add VWAP if it's actually part of the strategy
            if has_vwap and 'volume' in df.columns and len(df) > 20:
                try:
                    # Calculate VWAP
                    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                    self.playback_ax.plot(df.index, vwap, color='purple', linewidth=1, alpha=0.8, linestyle='-', label='VWAP', zorder=5)
                    print(f"[DEBUG] Results viewer: Added VWAP indicator (part of strategy)")
                except Exception as e:
                    print(f"[DEBUG] Results viewer: Failed to add VWAP indicator: {e}")
            else:
                print(f"[DEBUG] Results viewer: VWAP not part of strategy, skipping VWAP indicator")
            
            # Entry/Exit signals
            if getattr(self, 'show_signals', None) and self.show_signals.isChecked():
                trades = result_data.get('trades', [])
                entry_plotted = False
                exit_plotted = False
                for trade in trades:
                    entry_time = trade.get('entry_time')
                    exit_time = trade.get('exit_time')
                    entry_price = trade.get('entry_price')
                    exit_price = trade.get('exit_price')
                    entry_idx = trade.get('entry_idx')
                    exit_idx = trade.get('exit_idx')
                    entry_dt = None
                    if entry_time is not None:
                        try:
                            entry_dt = pd.to_datetime(entry_time)
                            if entry_dt in df.index:
                                self.playback_ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                                entry_plotted = True
                        except Exception:
                            pass
                    elif entry_idx is not None and entry_price is not None and 0 <= entry_idx < len(df):
                        entry_dt = df.index[entry_idx]
                        self.playback_ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                        entry_plotted = True
                    exit_dt = None
                    if exit_time is not None:
                        try:
                            exit_dt = pd.to_datetime(exit_time)
                            if exit_dt in df.index:
                                self.playback_ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                                exit_plotted = True
                        except Exception:
                            pass
                    elif exit_idx is not None and exit_price is not None and 0 <= exit_idx < len(df):
                        exit_dt = df.index[exit_idx]
                        self.playback_ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                        exit_plotted = True
                handles, labels = self.playback_ax.get_legend_handles_labels()
                if handles:
                    self.playback_ax.legend()
            # Patterns (future: add pattern markers if available)
            # Indicators (future: add indicator overlays if available)
        else:
            print(f"[DEBUG] No valid data for chart - len: {len(df)}, has OHLC: {all(col in df.columns for col in ['open', 'high', 'low', 'close'])}")
            self.playback_ax.text(0.5, 0.5, 'No real price data available for chart.', ha='center', va='center', color='red', fontsize=14)
        
        self.playback_ax.set_title('Playback Chart')
        self.playback_fig.tight_layout()
        self.playback_canvas.draw()
        print(f"[DEBUG] Chart drawing complete")

    def _update_performance_metrics(self, result_data: Dict[str, Any]):
        """Update performance metrics display"""
        print(f"[DEBUG] _update_performance_metrics called with keys: {list(result_data.keys())}")
        
        # Update result info
        status_text = ""
        if result_data.get('status') == 'created':
            status_text = "\nStatus: Strategy created (no backtest results yet)"
        
        info_text = f"""
Result ID: {result_data.get('id', 'N/A')}
Strategy: {result_data.get('strategy', 'N/A')}
Timestamp: {result_data.get('timestamp', 'N/A')}{status_text}

Summary:
Total Return: {result_data.get('total_return', 0):.2%}
Sharpe Ratio: {result_data.get('sharpe_ratio', 0):.2f}
Max Drawdown: {result_data.get('max_drawdown', 0):.2%}
Total Trades: {result_data.get('total_trades', 0)}
"""
        self.result_info.setText(info_text)
        
        # Update metrics - access directly from result data
        for key, label in self.single_metrics.items():
            value = result_data.get(key, 0)
            print(f"[DEBUG] Setting {key} = {value}")
            if key in ['total_return', 'max_drawdown', 'win_rate']:
                label.setText(f"{value:.2%}")
            elif key in ['sharpe_ratio', 'profit_factor']:
                label.setText(f"{value:.2f}")
            elif key == 'total_trades':
                label.setText(str(value))
            else:
                label.setText(f"${value:.2f}")
    
    def _update_spec_metrics(self, result_data: Dict[str, Any]):
        """Update spec-compliant metrics display"""
        try:
            # Calculate averages and maxes for display
            S_adj_scores = result_data.get('S_adj_scores', [])
            S_net_scores = result_data.get('S_net_scores', [])
            per_zone_strengths = result_data.get('per_zone_strengths', [])
            momentum_scores = result_data.get('momentum_scores', [])
            volatility_scores = result_data.get('volatility_scores', [])
            imbalance_scores = result_data.get('imbalance_scores', [])
            enhanced_momentum = result_data.get('enhanced_momentum', [])
            
            # Update metrics
            self.spec_metrics['S_adj_scores'].setText(f"{len(S_adj_scores)} scores")
            self.spec_metrics['S_net_scores'].setText(f"{len(S_net_scores)} scores")
            self.spec_metrics['per_zone_strengths'].setText(f"{len(per_zone_strengths)} zones")
            self.spec_metrics['momentum_scores'].setText(f"{len(momentum_scores)} scores")
            self.spec_metrics['volatility_scores'].setText(f"{len(volatility_scores)} scores")
            self.spec_metrics['imbalance_scores'].setText(f"{len(imbalance_scores)} scores")
            self.spec_metrics['enhanced_momentum'].setText(f"{len(enhanced_momentum)} scores")
            
            # Calculate averages and maxes
            avg_S_adj = np.mean(S_adj_scores) if S_adj_scores else 0.0
            max_S_adj = np.max(S_adj_scores) if S_adj_scores else 0.0
            
            self.spec_metrics['avg_S_adj'].setText(f"{avg_S_adj:.3f}")
            self.spec_metrics['max_S_adj'].setText(f"{max_S_adj:.3f}")
            
            # Update strategy parameters display
            strategy_params = result_data.get('strategy_params', {})
            gates_enabled = result_data.get('gates_enabled', {})
            
            params_text = "Strategy Parameters:\n"
            for key, value in strategy_params.items():
                params_text += f"{key}: {value}\n"
            
            params_text += "\nGates Enabled:\n"
            for gate, enabled in gates_enabled.items():
                params_text += f"{gate}: {enabled}\n"
            
            self.strategy_params_display.setText(params_text)
            
        except Exception as e:
            print(f"Error updating spec metrics: {e}")
            # Set default values on error
            for label in self.spec_metrics.values():
                label.setText("--")
    
    def _update_mae_analysis(self, result_data: Dict[str, Any]):
        """Update MAE analysis"""
        # Update the MAE vs PnL scatter plot
        self.mae_plot.clear()
        
        # Access trades directly from result data
        trades = result_data.get('trades', [])
        if trades and isinstance(trades, list) and trades and 'mae' in trades[0] and 'pnl' in trades[0]:
            self.mae_plot.setVisible(True)
            self.mae_placeholder.setVisible(False)

            profitable_trades = [t for t in trades if t['pnl'] > 0]
            unprofitable_trades = [t for t in trades if t['pnl'] <= 0]

            if profitable_trades:
                prof_maes = [t['mae'] for t in profitable_trades]
                prof_pnls = [t['pnl'] for t in profitable_trades]
                self.mae_plot.plot(prof_maes, prof_pnls, pen=None, symbol='o', symbolBrush=(0, 255, 0, 150), name='Profitable')

            if unprofitable_trades:
                unprof_maes = [t['mae'] for t in unprofitable_trades]
                unprof_pnls = [t['pnl'] for t in unprofitable_trades]
                self.mae_plot.plot(unprof_maes, unprof_pnls, pen=None, symbol='o', symbolBrush=(255, 0, 0, 150), name='Unprofitable')

            self.mae_plot.addLegend()
        else:
            self.mae_plot.setVisible(False)
            self.mae_placeholder.setVisible(True)

    def _initialize_playback_data(self, result_data: Dict[str, Any]):
        """Initialize playback data from result with multi-timeframe support"""
        import pandas as pd
        print(f"[DEBUG] Initializing playback data with keys: {list(result_data.keys())}")
        
        # Priority 1: Use multi-timeframe data if available (preserves strategy timeframes)
        data = None
        multi_tf_data = result_data.get('multi_tf_data', {})
        
        # Handle both old dict format and new DataFrame format
        if isinstance(multi_tf_data, dict):
            print(f"[DEBUG] Multi-timeframe data keys: {list(multi_tf_data.keys()) if multi_tf_data else 'None'}")
            
            if multi_tf_data:
                # Use execution timeframe data for playback (this is what the strategy actually ran on)
                execution_data = multi_tf_data.get('execution')
                print(f"[DEBUG] Execution data type: {type(execution_data)}")
                if execution_data is not None and isinstance(execution_data, pd.DataFrame):
                    data = execution_data
        elif isinstance(multi_tf_data, pd.DataFrame) and not multi_tf_data.empty:
            print(f"[DEBUG] Multi-timeframe data is DataFrame: {len(multi_tf_data)} bars")
            # New architecture returns DataFrame directly
            data = multi_tf_data
            print(f"[DEBUG] Using new architecture DataFrame with shape: {data.shape}")
            print(f"[DEBUG] DataFrame columns: {list(data.columns)}")
            print(f"[DEBUG] DataFrame data range: {data.index[0] if len(data) > 0 else 'Empty'} to {data.index[-1] if len(data) > 0 else 'Empty'}")
        
        # Check if we found execution data from multi-timeframe
        if data is not None and isinstance(data, pd.DataFrame):
            print(f"[DEBUG] Found execution data from multi-timeframe source")
        else:
            print(f"[DEBUG] No execution data found in multi-timeframe, checking fallback options")
        
        # Priority 2: Use the resampled data from the backtest results
        if data is None and 'data' in result_data and isinstance(result_data['data'], pd.DataFrame):
            data = result_data['data']
            print(f"[DEBUG] Using resampled data from backtest results with shape: {data.shape}")
            print(f"[DEBUG] Data columns: {list(data.columns)}")
            print(f"[DEBUG] Data index range: {data.index[0] if len(data) > 0 else 'Empty'} to {data.index[-1] if len(data) > 0 else 'Empty'}")
        elif data is None:
            print(f"[DEBUG] No 'data' key in result_data or not a DataFrame")
            print(f"[DEBUG] 'data' key exists: {'data' in result_data}")
            if 'data' in result_data:
                print(f"[DEBUG] 'data' type: {type(result_data['data'])}")
            
        # Priority 3: Create synthetic data from equity curve if no resampled data available
        elif data is None and 'equity_curve' in result_data:
            equity_curve = result_data['equity_curve']
            
            # Handle both old list format and new Series format
            has_equity_data = False
            if isinstance(equity_curve, pd.Series):
                has_equity_data = not equity_curve.empty
            elif isinstance(equity_curve, list):
                has_equity_data = bool(equity_curve)
            
            if has_equity_data:
                print(f"[DEBUG] ⚠️ FALLING BACK TO SYNTHETIC DATA - This will show diagonal line!")
                print(f"[DEBUG] Creating synthetic data from equity curve with {len(result_data['equity_curve'])} points")
                equity_curve = result_data['equity_curve']
                n_points = len(equity_curve)
                
                # Use the timeframe from the backtest results if available
                timeframe = result_data.get('timeframe', '1d')
                if timeframe == '1d':
                    freq = 'D'
                elif timeframe == '1h':
                    freq = 'H'
                elif timeframe == '15min':
                    freq = '15T'
                elif timeframe == '5min':
                    freq = '5T'
                elif timeframe == '1min':
                    freq = '1T'
                else:
                    freq = '1T'  # Default to 1 minute
                    
                time_index = pd.date_range('2023-01-01', periods=n_points, freq=freq)
                base_price = 100.0
                data = pd.DataFrame({
                    'open': [base_price + i * 0.1 for i in range(n_points)],
                    'high': [base_price + i * 0.1 + 0.5 for i in range(n_points)],
                    'low': [base_price + i * 0.1 - 0.5 for i in range(n_points)],
                    'close': [base_price + i * 0.1 + (equity_curve[i] - equity_curve[0]) / 1000 for i in range(n_points)],
                    'volume': [1000 + i * 10 for i in range(n_points)]
                }, index=time_index)
                print(f"[DEBUG] Created synthetic DataFrame with shape: {data.shape} and frequency: {freq}")
                print(f"[DEBUG] ⚠️ WARNING: This synthetic data will show diagonal line, not real price data!")
            
        # Priority 4: Only fall back to parent dataset if absolutely necessary (and with warning)
        elif data is None and hasattr(self, 'parent_window') and self.parent_window and hasattr(self.parent_window, 'datasets'):
            print(f"[WARNING] No resampled data found in results, falling back to original dataset")
            print(f"[DEBUG] Checking parent window datasets: {list(self.parent_window.datasets.keys())}")
            if self.parent_window.datasets:
                dataset_name = list(self.parent_window.datasets.keys())[0]
                dataset_info = self.parent_window.datasets[dataset_name]
                if 'data' in dataset_info and isinstance(dataset_info['data'], pd.DataFrame):
                    data = dataset_info['data']
                    print(f"[WARNING] Using original dataset '{dataset_name}' with shape: {data.shape}")
                    print(f"[WARNING] This may cause performance issues with high-frequency data")
                    
        # Priority 5: Last resort - check if result_data itself is a DataFrame
        elif data is None and isinstance(result_data, pd.DataFrame):
            data = result_data
            print(f"[DEBUG] Result data is DataFrame with shape: {data.shape}")
            
        if data is None:
            print(f"[DEBUG] No valid data found for playback")
            self.playback_data = None
            return
            
        try:
            self.playback_data = data.copy()
            
            # Store original data for multi-timeframe support
            self.original_playback_data = self.playback_data.copy()
            self.multi_tf_data = multi_tf_data
            
            # Handle date column if present
            if 'Date' in self.playback_data.columns:
                self.playback_data['Date'] = pd.to_datetime(self.playback_data['Date'])
                self.playback_data.set_index('Date', inplace=True)
                
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.playback_data.columns]
            if missing_columns:
                print(f"[DEBUG] Missing columns: {missing_columns}")
                for col in missing_columns:
                    if col == 'volume':
                        self.playback_data[col] = 1000
                    else:
                        self.playback_data[col] = self.playback_data['close']
                        
            # Ensure proper datetime index
            if not isinstance(self.playback_data.index, pd.DatetimeIndex):
                try:
                    if hasattr(self.playback_data.index, 'dtype') and 'datetime' in str(self.playback_data.index.dtype):
                        self.playback_data.index = pd.to_datetime(self.playback_data.index)
                    else:
                        self.playback_data.index = pd.to_datetime(self.playback_data.index)
                except Exception as e:
                    print(f"[DEBUG] Could not convert index to datetime: {e}")
                    # Use the timeframe from results to set appropriate frequency
                    timeframe = result_data.get('timeframe', '1d')
                    if timeframe == '1d':
                        freq = 'D'
                    elif timeframe == '1h':
                        freq = 'H'
                    elif timeframe == '15min':
                        freq = '15T'
                    elif timeframe == '5min':
                        freq = '5T'
                    elif timeframe == '1min':
                        freq = '1T'
                    else:
                        freq = '1T'
                    self.playback_data.index = pd.date_range('2023-01-01', periods=len(self.playback_data), freq=freq)
                    
            print(f"[DEBUG] Playback data index range: {self.playback_data.index[0]} to {self.playback_data.index[-1]}")
            print(f"[DEBUG] Playback data index type: {type(self.playback_data.index)}")
            print(f"[DEBUG] Playback data initialized with {len(self.playback_data)} rows")
            print(f"[DEBUG] Playback data columns: {list(self.playback_data.columns)}")
            
            # Log multi-timeframe information
            if multi_tf_data:
                print(f"[DEBUG] Multi-timeframe data available:")
                for tf, tf_data in multi_tf_data.items():
                    print(f"  {tf}: {len(tf_data)} bars")
            
            self.current_index = 0
            
            # Update chart
            self._draw_playback_chart(self.playback_data, result_data)
            
        except Exception as e:
            print(f"[DEBUG] Error initializing playback data: {e}")
            import traceback
            traceback.print_exc()
            self.playback_data = None

    def _refresh_results_from_disk(self):
        """Reload only the most recent result for each strategy from disk and auto-select the most recent overall."""
        import os
        print(f"[DEBUG] ResultsViewer: Starting _refresh_results_from_disk")
        if self.parent_window and hasattr(self.parent_window, 'strategy_manager'):
            all_results = self.parent_window.strategy_manager.load_all_results()
            print(f"[DEBUG] ResultsViewer: Loaded {len(all_results)} results from disk")
            print(f"[DEBUG] ResultsViewer: Result keys: {list(all_results.keys())}")
            # Keep only the most recent result for each strategy
            most_recent = {}
            for result_id, result in all_results.items():
                strategy = result.get('strategy_name', None)
                timestamp = result.get('timestamp', None)
                if not timestamp:
                    timestamp = result.get('trades', [{}])[0].get('entry_time', 'Unknown')
                if strategy:
                    if strategy not in most_recent or timestamp > most_recent[strategy]['timestamp']:
                        most_recent[strategy] = {'result_id': result_id, 'result': result, 'timestamp': timestamp}
            self.results_history.clear()
            self.results_combo.clear()
            # Add only the most recent result for each strategy
            sorted_results = sorted(most_recent.values(), key=lambda x: x['timestamp'], reverse=True)
            for entry in sorted_results:
                result_id = entry['result_id']
                result = entry['result']
                strategy_name = result.get('strategy_name', f'Strategy_{result_id}')
                timestamp = entry['timestamp']
                display_name = f"{strategy_name} - {timestamp}"
                self.results_history[result_id] = {
                    'data': result,
                    'strategy': strategy_name,
                    'timestamp': timestamp,
                    'id': result_id
                }
                self.results_combo.addItem(display_name, result_id)
            # Auto-select the most recent result
            if self.results_combo.count() > 0:
                self.results_combo.setCurrentIndex(0)
                self._on_load_selected_result()

    def _on_heatmap_control_changed(self):
        """Callback for heatmap UI controls"""
        if hasattr(self, 'current_result_data') and self.current_result_data:
            self._update_heatmap(self.current_result_data)
        else:
            print("[DEBUG] No current result data available for heatmap update")

    def _update_heatmap(self, result_data: Dict[str, Any] = None):
        """Update heatmap - SIMPLE VERSION"""
        print("[DEBUG] SIMPLE HEATMAP: Starting update")
        
        # Get result data
        if result_data is None:
            if hasattr(self, 'current_result_data') and self.current_result_data:
                result_data = self.current_result_data
            else:
                print("[DEBUG] No result data available")
                return
        
        if not isinstance(result_data, dict):
            print(f"[DEBUG] Invalid result data: {type(result_data)}")
            return
        
        try:
            # Clear the plot
            self.heatmap_plot.clear()
            
            # Step 1: Extract ONLY real building blocks
            building_blocks = []
        action_details = result_data.get('action_details', {})
            trades = result_data.get('trades', [])
            
            # Get building blocks with actual signals
            for block_name, signals in action_details.items():
                if signals is not None:
                    building_blocks.append(block_name)
                    print(f"[DEBUG] Found building block: {block_name}")
                    if isinstance(signals, pd.Series):
                        print(f"[DEBUG] {block_name} has {signals.sum()} signals")
                    elif isinstance(signals, str):
                        print(f"[DEBUG] {block_name} is string with {signals.count('True')} True values")
            
            # Add trades if they exist
            if trades:
                building_blocks.append('trades')
                print(f"[DEBUG] Added trades block")
        
        if not building_blocks:
                self.heatmap_info.setText("No building blocks found")
            return
            
            # Step 2: Get time index from data
            data = result_data.get('data')
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                print("[DEBUG] No valid data")
                self.heatmap_info.setText("No data available")
            return
            
            time_index = data.index
            print(f"[DEBUG] Time range: {time_index[0]} to {time_index[-1]} ({len(time_index)} points)")
            
            # Step 3: Create simple time windows (15-minute bins)
            start_time = time_index[0]
            end_time = time_index[-1]
            time_bins = pd.date_range(start=start_time, end=end_time, freq='15min')
            
            if len(time_bins) < 2:
                time_bins = pd.date_range(start=start_time, end=end_time, freq='1H')
            
            print(f"[DEBUG] Created {len(time_bins)} time bins")
            
            # Step 4: Create matrix
            matrix = np.zeros((len(building_blocks), len(time_bins)-1))
            
            for i, block_name in enumerate(building_blocks):
                if block_name == 'trades':
                    # Handle trades
                    for trade in trades:
                        entry_time = trade.get('entry_time')
                        if entry_time:
                            if isinstance(entry_time, str):
                                entry_time = pd.to_datetime(entry_time)
                            
                            # Find which time bin this trade belongs to
                            for j in range(len(time_bins)-1):
                                if time_bins[j] <= entry_time < time_bins[j+1]:
                                    matrix[i, j] += 1
                                    break
                else:
                    # Handle action_details
                    signals = action_details.get(block_name)
                    if signals is not None:
                        if isinstance(signals, pd.Series):
                            # Direct series
                            for j in range(len(time_bins)-1):
                                bin_start = time_bins[j]
                                bin_end = time_bins[j+1]
                                mask = (signals.index >= bin_start) & (signals.index < bin_end)
                                matrix[i, j] = signals[mask].sum()
                        elif isinstance(signals, str) and 'True' in signals:
                            # Parse string - simplified approach
                            true_count = signals.count('True')
                            if true_count > 0:
                                # Distribute signals across time bins
                                signals_per_bin = true_count / len(time_bins)
                                for j in range(len(time_bins)-1):
                                    matrix[i, j] = signals_per_bin
            
            # Step 5: Create labels
            time_labels = []
            for j in range(len(time_bins)-1):
                start_label = time_bins[j].strftime('%H:%M')
                time_labels.append(start_label)
            
            # Step 6: Plot proper heatmap tiles
            if matrix.max() > 0:
                # Normalize to 0-1
                normalized_matrix = matrix / matrix.max()
                
                # Create proper heatmap image
                img = pg.ImageItem()
                img.setImage(normalized_matrix, levels=(0, 1))
                
                # Create proper colormap: black (no signals) to white (many signals)
                lut = np.zeros((256, 4), dtype=np.ubyte)
                for i in range(256):
                    val = i
                    # Black to white gradient
                    lut[i] = [val, val, val, 255]
                img.setLookupTable(lut)
                
                self.heatmap_plot.addItem(img)
                
                # Set proper range for heatmap tiles
                self.heatmap_plot.setRange(
                    xRange=(-0.5, len(time_labels) - 0.5),
                    yRange=(-0.5, len(building_blocks) - 0.5)
                )
                
                # Set proper labels
                self.heatmap_plot.setLabel('bottom', 'Time Windows', size='12pt')
                self.heatmap_plot.setLabel('left', 'Building Blocks', size='12pt')
                self.heatmap_plot.setTitle('Signal Heatmap: White = Active, Black = Inactive', size='14pt')
                
                # Set proper ticks
                if building_blocks:
                    block_ticks = [(i, block) for i, block in enumerate(building_blocks)]
                    self.heatmap_plot.getAxis('left').setTicks([block_ticks])
                
                if time_labels:
                    # Show readable time labels
                    time_ticks = [(i, time_labels[i]) for i in range(0, len(time_labels), 2)]
                    self.heatmap_plot.getAxis('bottom').setTicks([time_ticks])
                
                print(f"[DEBUG] SIMPLE HEATMAP: Created {matrix.shape[0]}x{matrix.shape[1]} matrix")
                print(f"[DEBUG] SIMPLE HEATMAP: Signal counts - {matrix.sum():.1f} total signals")
        
        self.heatmap_info.setText(
                    f"Simple Heatmap: {len(building_blocks)} blocks, {len(time_labels)} time windows, "
                    f"{matrix.sum():.0f} total signals. White = More Active."
                )
            else:
                self.heatmap_info.setText("No signals found in any building blocks")
                
        except Exception as e:
            print(f"[DEBUG] SIMPLE HEATMAP ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.heatmap_info.setText(f"Heatmap Error: {str(e)}")

    def _create_state_heatmap_tab(self) -> QWidget:
        """Create state change heatmap tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # State heatmap controls
        controls_layout = QHBoxLayout()
        
        # State display mode
        controls_layout.addWidget(QLabel("State Mode:"))
        self.state_mode_combo = QComboBox()
        self.state_mode_combo.addItems([
            "All States", "Active/Inactive", "Confidence Levels", 
            "State Transitions", "Probability Thresholds"
        ])
        self.state_mode_combo.setCurrentText("All States")
        self.state_mode_combo.currentTextChanged.connect(self._on_state_heatmap_control_changed)
        controls_layout.addWidget(self.state_mode_combo)
        
        # Time resolution
        controls_layout.addWidget(QLabel("Time Resolution:"))
        self.state_time_resolution_combo = QComboBox()
        self.state_time_resolution_combo.addItems(["1 minute", "5 minutes", "15 minutes", "1 hour"])
        self.state_time_resolution_combo.setCurrentText("5 minutes")
        self.state_time_resolution_combo.currentTextChanged.connect(self._on_state_heatmap_control_changed)
        controls_layout.addWidget(self.state_time_resolution_combo)
        
        # Color scheme for states
        controls_layout.addWidget(QLabel("Color Scheme:"))
        self.state_color_scheme_combo = QComboBox()
        self.state_color_scheme_combo.addItems(["State Rainbow", "Traffic Light", "Thermal", "Discrete"])
        self.state_color_scheme_combo.setCurrentText("State Rainbow")
        self.state_color_scheme_combo.currentTextChanged.connect(self._on_state_heatmap_control_changed)
        controls_layout.addWidget(self.state_color_scheme_combo)
        
        # Show transitions checkbox
        self.show_transitions_check = QCheckBox("Show Transitions")
        self.show_transitions_check.setChecked(True)
        self.show_transitions_check.toggled.connect(self._on_state_heatmap_control_changed)
        controls_layout.addWidget(self.show_transitions_check)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # State heatmap plot
        self.state_heatmap_plot = pg.PlotWidget()
        self.state_heatmap_plot.setLabel('left', 'Building Blocks')
        self.state_heatmap_plot.setLabel('bottom', 'Time')
        self.state_heatmap_plot.setTitle('Building Block State Changes Over Time')
        self.state_heatmap_plot.showGrid(True, True)
        layout.addWidget(self.state_heatmap_plot)
        
        # State legend and info
        legend_layout = QHBoxLayout()
        
        # State legend
        self.state_legend = QLabel("State Legend: Loading...")
        self.state_legend.setAlignment(Qt.AlignmentFlag.AlignLeft)
        legend_layout.addWidget(self.state_legend)
        
        # State statistics
        self.state_stats = QLabel("State Statistics: Loading...")
        self.state_stats.setAlignment(Qt.AlignmentFlag.AlignRight)
        legend_layout.addWidget(self.state_stats)
        
        layout.addLayout(legend_layout)
        
        # State heatmap info
        self.state_heatmap_info = QLabel("No state change data available.")
        self.state_heatmap_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_heatmap_info)
        
        widget.setLayout(layout)
        return widget

    def _on_state_heatmap_control_changed(self):
        """Handle state heatmap control changes"""
        if hasattr(self, 'current_result_data') and self.current_result_data:
            self._update_state_heatmap(self.current_result_data)

    def _update_state_heatmap(self, result_data: Dict[str, Any]):
        """Update state change heatmap"""
        try:
            print(f"[DEBUG] STATE HEATMAP: Starting update")
            self.state_heatmap_plot.clear()
            
            # Extract state change data
            state_data = self._extract_state_change_data(result_data)
            
            if not state_data:
                self.state_heatmap_info.setText("No state change data found")
                return
            
            # Create state matrix
            state_matrix, building_blocks, time_labels, state_legend = self._create_state_matrix(state_data)
            
            if state_matrix is None or state_matrix.size == 0:
                self.state_heatmap_info.setText("No valid state data to display")
                return
            
            # Plot state heatmap
            self._plot_state_heatmap(state_matrix, building_blocks, time_labels, state_legend)
            
            print(f"[DEBUG] STATE HEATMAP: Successfully updated")
                                            
                                    except Exception as e:
            print(f"[DEBUG] STATE HEATMAP ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.state_heatmap_info.setText(f"Error creating state heatmap: {e}")

    def _extract_state_change_data(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state change data from backtest results"""
        print(f"[DEBUG] STATE HEATMAP: Extracting state data")
        
        state_data = {}
        
        # Extract from action_details (building block signals)
        action_details = result_data.get('action_details', {})
        print(f"[DEBUG] STATE HEATMAP: Found action_details with keys: {list(action_details.keys())}")
        
        # Get data for time index
        data = result_data.get('data')
        if data is None or len(data) == 0:
            print(f"[DEBUG] STATE HEATMAP: No data found")
            return {}
        
        time_index = data.index
        print(f"[DEBUG] STATE HEATMAP: Using time index with {len(time_index)} periods")
        
        # Process each building block
        for block_name, block_data in action_details.items():
            try:
                print(f"[DEBUG] STATE HEATMAP: Processing {block_name}")
                
                if isinstance(block_data, pd.Series):
                    # Convert boolean signals to state values
                    states = self._convert_signals_to_states(block_data, block_name)
                elif isinstance(block_data, str):
                    # Parse string representation of Series
                    try:
                        parsed_series = pd.read_json(block_data, typ='series')
                        states = self._convert_signals_to_states(parsed_series, block_name)
                                                    except:
                        print(f"[DEBUG] STATE HEATMAP: Failed to parse {block_name} as Series")
                        continue
                else:
                    print(f"[DEBUG] STATE HEATMAP: Unknown data type for {block_name}: {type(block_data)}")
                                                        continue
                                        
                # Align states with time index
                if len(states) != len(time_index):
                    states = states.reindex(time_index, fill_value=0)
                
                state_data[block_name] = states
                print(f"[DEBUG] STATE HEATMAP: {block_name} has {len(states)} state values")
                
                                except Exception as e:
                print(f"[DEBUG] STATE HEATMAP: Error processing {block_name}: {e}")
                                    continue
        
        # Add trades as a special building block
        trades = result_data.get('trades', [])
        if trades and len(trades) > 0:
            trade_states = self._convert_trades_to_states(trades, time_index)
            state_data['trades'] = trade_states
            print(f"[DEBUG] STATE HEATMAP: Added trades with {len(trade_states)} state values")
        
        print(f"[DEBUG] STATE HEATMAP: Extracted state data for {len(state_data)} building blocks")
        return state_data

    def _convert_signals_to_states(self, signals: pd.Series, block_name: str) -> pd.Series:
        """Convert boolean signals to state values for visualization"""
        # Define state mappings based on building block type and signal patterns
        states = signals.copy()
        
        if signals.dtype == bool:
            # Simple boolean signals: 0 = inactive, 1 = active
            states = signals.astype(int)
                                    else:
            # Numeric signals: convert to state levels
            # State 0: Inactive (value == 0 or False)
            # State 1: Low Activity (0 < value <= 0.3)  
            # State 2: Medium Activity (0.3 < value <= 0.7)
            # State 3: High Activity (0.7 < value <= 1.0)
            # State 4: Very High Activity (value > 1.0)
            
            states = pd.Series(0, index=signals.index)  # Default to inactive
            states[signals > 0] = 1      # Low activity
            states[signals > 0.3] = 2    # Medium activity  
            states[signals > 0.7] = 3    # High activity
            states[signals > 1.0] = 4    # Very high activity
        
        return states

    def _convert_trades_to_states(self, trades: List[Dict], time_index: pd.DatetimeIndex) -> pd.Series:
        """Convert trade events to state values"""
        trade_states = pd.Series(0, index=time_index)  # Default to no trade
        
        for trade in trades:
            try:
                entry_time = pd.to_datetime(trade.get('entry_time'))
                exit_time = pd.to_datetime(trade.get('exit_time'))
                
                # Find closest times in index
                entry_idx = time_index.get_indexer([entry_time], method='nearest')[0]
                exit_idx = time_index.get_indexer([exit_time], method='nearest')[0]
                
                if entry_idx >= 0 and entry_idx < len(time_index):
                    trade_states.iloc[entry_idx] = 2  # Entry signal
                
                if exit_idx >= 0 and exit_idx < len(time_index):
                    trade_states.iloc[exit_idx] = 3   # Exit signal
                
                # Mark positions as active (state 1) between entry and exit
                if entry_idx >= 0 and exit_idx >= 0 and entry_idx < exit_idx:
                    trade_states.iloc[entry_idx:exit_idx] = 1  # Position active
                    trade_states.iloc[entry_idx] = 2  # Re-mark entry
                    trade_states.iloc[exit_idx] = 3   # Re-mark exit
            
        except Exception as e:
                print(f"[DEBUG] STATE HEATMAP: Error processing trade: {e}")
                continue
        
        return trade_states

    def _create_state_matrix(self, state_data: Dict[str, pd.Series]) -> Tuple[np.ndarray, List[str], List[str], Dict]:
        """Create state matrix for heatmap visualization"""
        if not state_data:
            return None, [], [], {}
        
        # Get time resolution
        time_resolution = self.state_time_resolution_combo.currentText()
        time_delta_map = {
            "1 minute": "1min",
            "5 minutes": "5min", 
            "15 minutes": "15min",
            "1 hour": "1H"
        }
        time_delta = time_delta_map.get(time_resolution, "5min")
        
        # Get all building blocks
        building_blocks = list(state_data.keys())
        
        # Get time range from first series
        first_series = next(iter(state_data.values()))
        start_time = first_series.index[0]
        end_time = first_series.index[-1]
        
        # Create time windows
        time_windows = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        if len(time_windows) == 0:
            time_windows = [start_time, end_time]
            
        # Create matrix
        matrix = np.zeros((len(building_blocks), len(time_windows)))
        
        # Fill matrix with state values
        for i, block_name in enumerate(building_blocks):
            series = state_data[block_name]
            
            for j, window_time in enumerate(time_windows):
                # Find values in this time window
                if j < len(time_windows) - 1:
                    next_window = time_windows[j + 1]
                    window_mask = (series.index >= window_time) & (series.index < next_window)
                else:
                    window_mask = series.index >= window_time
                
                window_values = series[window_mask]
                
                if len(window_values) > 0:
                    # Get dominant state in this window
                    state_mode = self.state_mode_combo.currentText()
                    
                    if state_mode == "All States":
                        # Use most common state
                        matrix[i, j] = window_values.mode().iloc[0] if len(window_values.mode()) > 0 else 0
                    elif state_mode == "Active/Inactive":
                        # 0 = inactive, 1 = any activity
                        matrix[i, j] = 1 if (window_values > 0).any() else 0
                    elif state_mode == "Confidence Levels":
                        # Use maximum confidence level
                        matrix[i, j] = window_values.max()
                    elif state_mode == "State Transitions":
                        # Count number of state changes
                        transitions = (window_values.diff() != 0).sum()
                        matrix[i, j] = min(transitions, 4)  # Cap at 4 for visualization
                    elif state_mode == "Probability Thresholds":
                        # Use average state level
                        matrix[i, j] = window_values.mean()
        
        # Create time labels
        time_labels = [t.strftime("%H:%M") if t.time() != t.time().replace(hour=0, minute=0) 
                      else t.strftime("%m/%d") for t in time_windows]
        
        # Create state legend
        state_legend = self._create_state_legend()
        
        print(f"[DEBUG] STATE HEATMAP: Created matrix {matrix.shape} with {len(building_blocks)} blocks")
        
        return matrix, building_blocks, time_labels, state_legend

    def _create_state_legend(self) -> Dict[int, str]:
        """Create legend for state values"""
        state_mode = self.state_mode_combo.currentText()
        
        if state_mode == "All States":
            return {
                0: "Inactive",
                1: "Low Activity", 
                2: "Medium Activity",
                3: "High Activity",
                4: "Very High Activity"
            }
        elif state_mode == "Active/Inactive":
            return {
                0: "Inactive",
                1: "Active"
            }
        elif state_mode == "Confidence Levels":
            return {
                0: "No Confidence",
                1: "Low Confidence",
                2: "Medium Confidence", 
                3: "High Confidence",
                4: "Very High Confidence"
            }
        elif state_mode == "State Transitions":
            return {
                0: "No Transitions",
                1: "1 Transition",
                2: "2 Transitions",
                3: "3 Transitions", 
                4: "4+ Transitions"
            }
        elif state_mode == "Probability Thresholds":
            return {
                0: "Below Threshold",
                1: "Low Probability",
                2: "Medium Probability",
                3: "High Probability",
                4: "Very High Probability"
            }
        
        return {}

    def _plot_state_heatmap(self, matrix: np.ndarray, building_blocks: List[str], 
                           time_labels: List[str], state_legend: Dict[int, str]):
        """Plot the state change heatmap"""
        try:
            self.state_heatmap_plot.clear()
            
            # Create color map based on selected scheme
            color_scheme = self.state_color_scheme_combo.currentText()
            
            if color_scheme == "State Rainbow":
                # Rainbow colors for different states
                colors = [
                    [0, 0, 0, 255],        # Black - Inactive
                    [0, 255, 0, 255],      # Green - Low
                    [255, 255, 0, 255],    # Yellow - Medium  
                    [255, 165, 0, 255],    # Orange - High
                    [255, 0, 0, 255],      # Red - Very High
                ]
            elif color_scheme == "Traffic Light":
                colors = [
                    [100, 100, 100, 255],  # Gray - Inactive
                    [255, 255, 0, 255],    # Yellow - Low
                    [255, 165, 0, 255],    # Orange - Medium
                    [255, 0, 0, 255],      # Red - High
                    [139, 0, 0, 255],      # Dark Red - Very High
                ]
            elif color_scheme == "Thermal":
                colors = [
                    [0, 0, 0, 255],        # Black - Cold
                    [0, 0, 255, 255],      # Blue - Cool
                    [0, 255, 255, 255],    # Cyan - Warm
                    [255, 255, 0, 255],    # Yellow - Hot
                    [255, 0, 0, 255],      # Red - Very Hot
                ]
            else:  # Discrete
                colors = [
                    [50, 50, 50, 255],     # Dark Gray
                    [100, 150, 200, 255],  # Light Blue  
                    [150, 200, 100, 255],  # Light Green
                    [200, 150, 100, 255],  # Light Orange
                    [200, 100, 150, 255],  # Light Red
                ]
            
            # Ensure we have enough colors
            while len(colors) < int(matrix.max()) + 1:
                colors.append([255, 255, 255, 255])  # White for overflow
            
            # Create color map
            colormap = pg.ColorMap(pos=np.linspace(0, 1, len(colors)), color=colors)
            
            # Create image item
            img = pg.ImageItem()
            img.setImage(matrix, levels=(0, len(colors)-1))
            img.setColorMap(colormap)
            
            self.state_heatmap_plot.addItem(img)
            
            # Set axis labels
            if building_blocks:
                block_ticks = [(i, block) for i, block in enumerate(building_blocks)]
                self.state_heatmap_plot.getAxis('left').setTicks([block_ticks])
            
            if time_labels:
                # Show readable time labels (every 2nd or 3rd label to avoid crowding)
                step = max(1, len(time_labels) // 10)
                time_ticks = [(i, time_labels[i]) for i in range(0, len(time_labels), step)]
                self.state_heatmap_plot.getAxis('bottom').setTicks([time_ticks])
            
            # Update legend
            legend_text = " | ".join([f"{val}: {desc}" for val, desc in state_legend.items()])
            self.state_legend.setText(f"States: {legend_text}")
            
            # Update statistics
            total_states = matrix.size
            active_states = np.sum(matrix > 0)
            max_state = int(matrix.max())
            avg_state = matrix.mean()
            
            self.state_stats.setText(
                f"Active: {active_states}/{total_states} ({100*active_states/total_states:.1f}%) | "
                f"Max State: {max_state} | Avg: {avg_state:.2f}"
            )
            
            # Update info
            state_mode = self.state_mode_combo.currentText()
            time_resolution = self.state_time_resolution_combo.currentText()
            
            self.state_heatmap_info.setText(
                f"State Heatmap: {len(building_blocks)} blocks × {len(time_labels)} windows | "
                f"Mode: {state_mode} | Resolution: {time_resolution} | "
                f"Total States: {total_states} | Active: {active_states}"
            )
            
            print(f"[DEBUG] STATE HEATMAP: Plotted {matrix.shape[0]}×{matrix.shape[1]} matrix")
            
        except Exception as e:
            print(f"[DEBUG] STATE HEATMAP PLOT ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.state_heatmap_info.setText(f"Error plotting state heatmap: {e}")
