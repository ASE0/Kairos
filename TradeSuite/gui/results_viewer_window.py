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
        
        # Load any existing results
        self._load_results_history()
        
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
        self.current_result = result_data
        
        # Update result info with multi-timeframe information
        strategy_name = result_data.get('strategy_name', 'Unknown Strategy')
        timeframe = result_data.get('timeframe', 'Unknown')
        
        # Get multi-timeframe information
        multi_tf_data = result_data.get('multi_tf_data', {})
        strategy_timeframes = []
        if multi_tf_data:
            for tf_key in multi_tf_data.keys():
                if tf_key != 'execution':
                    strategy_timeframes.append(tf_key)
        
        # Create info text
        info_text = f"""
Strategy: {strategy_name}
Display Timeframe: {timeframe}
"""
        
        if strategy_timeframes:
            info_text += f"Strategy Timeframes: {', '.join(strategy_timeframes)}\n"
            execution_data = multi_tf_data.get('execution')
            if execution_data is not None:
                info_text += f"Execution Timeframe: {len(execution_data)} bars\n"
        
        info_text += f"""
Total Return: {result_data.get('total_return', 0):.2%}
Total Trades: {result_data.get('total_trades', 0)}
Sharpe Ratio: {result_data.get('sharpe_ratio', 0):.2f}
Max Drawdown: {result_data.get('max_drawdown', 0):.2%}
"""
        
        self.result_info.setPlainText(info_text)
        
        # Update metrics
        self._update_performance_metrics(result_data)
        
        # Update spec-compliant metrics
        self._update_spec_metrics(result_data)
        
        # Update strategy details
        self._update_strategy_details(result_data)
        
        # Update charts
        self._update_equity_chart(result_data)
        self._update_drawdown_chart(result_data)
        self._update_trade_returns_chart(result_data)
        
        # Initialize playback data with multi-timeframe support
        self._initialize_playback_data(result_data)
        
        # Update MAE analysis
        self._update_mae_analysis(result_data)
        
    def _update_equity_chart(self, data: Dict[str, Any]):
        """Update the equity chart"""
        self.single_equity_chart.clear()
        
        # Access equity curve directly from result data
        equity_curve = data.get('equity_curve', [])
        if equity_curve:
            x = np.arange(len(equity_curve))
            self.single_equity_chart.plot(x, equity_curve, pen='w')
    
    def _update_drawdown_chart(self, data: Dict[str, Any]):
        """Update the drawdown chart"""
        self.single_drawdown_chart.clear()
        
        # Access equity curve directly from result data
        equity_curve = data.get('equity_curve', [])
        if equity_curve:
            equity_series = pd.Series(equity_curve)
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
        if multi_tf_data:
            for tf_key in multi_tf_data.keys():
                if tf_key != 'execution':
                    strategy_timeframes.append(tf_key)
        
        details = f"Strategy: {result.get('strategy_name', 'Unknown')}\n"
        details += f"Display Timeframe: {result.get('timeframe', 'Unknown')}\n"
        
        if strategy_timeframes:
            details += f"Strategy Timeframes: {', '.join(strategy_timeframes)}\n"
            details += f"Execution Timeframe: {len(multi_tf_data.get('execution', []))} bars\n\n"
        
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
            if equity_curve:
                x = np.arange(len(equity_curve))
                color = colors[i % len(colors)]
                strategy_name = result.get('strategy', 'Unknown Strategy')
                self.compare_equity_chart.plot(x, equity_curve, pen=color, 
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
            
            # Add VWAP indicator to results viewer
            if 'volume' in df.columns and len(df) > 20:
                try:
                    # Calculate VWAP
                    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                    self.playback_ax.plot(df.index, vwap, color='purple', linewidth=1, alpha=0.8, linestyle='-', label='VWAP', zorder=5)
                    print(f"[DEBUG] Results viewer: Added VWAP indicator")
                except Exception as e:
                    print(f"[DEBUG] Results viewer: Failed to add VWAP indicator: {e}")
            
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
        print(f"[DEBUG] Multi-timeframe data keys: {list(multi_tf_data.keys()) if multi_tf_data else 'None'}")
        
        if multi_tf_data:
            # Use execution timeframe data for playback (this is what the strategy actually ran on)
            execution_data = multi_tf_data.get('execution')
            print(f"[DEBUG] Execution data type: {type(execution_data)}")
            if execution_data is not None and isinstance(execution_data, pd.DataFrame):
                data = execution_data
                print(f"[DEBUG] Using multi-timeframe execution data with shape: {data.shape}")
                print(f"[DEBUG] Execution data columns: {list(data.columns)}")
                print(f"[DEBUG] Strategy timeframes: {[tf for tf in multi_tf_data.keys() if tf != 'execution']}")
                print(f"[DEBUG] Execution timeframe data range: {data.index[0] if len(data) > 0 else 'Empty'} to {data.index[-1] if len(data) > 0 else 'Empty'}")
            else:
                print(f"[DEBUG] Multi-timeframe data found but no execution data available")
                print(f"[DEBUG] Execution data is None: {execution_data is None}")
                print(f"[DEBUG] Execution data is DataFrame: {isinstance(execution_data, pd.DataFrame)}")
        
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
        elif data is None and 'equity_curve' in result_data and result_data['equity_curve']:
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
        if self.parent_window and hasattr(self.parent_window, 'strategy_manager'):
            all_results = self.parent_window.strategy_manager.load_all_results()
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
