"""
gui/results_viewer_window.py
============================
Window for viewing and comparing backtest results
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pyqtgraph as pg
from datetime import datetime
import json


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
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Results selection
        selection_layout = QHBoxLayout()
        
        selection_layout.addWidget(QLabel("Select Results:"))
        
        self.results_combo = QComboBox()
        self.results_combo.currentIndexChanged.connect(self._on_result_selected)
        selection_layout.addWidget(self.results_combo)
        
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
        
        # Single result view
        self.single_view = self._create_single_view()
        self.content_stack.addWidget(self.single_view)
        
        # Comparison view
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

        # Heatmaps tab
        heatmaps_widget = QWidget()
        heatmaps_layout = QVBoxLayout()
        
        # Strength score heatmap
        heatmaps_layout.addWidget(QLabel("Pattern Strength Heatmap:"))
        self.strength_heatmap = pg.ImageView()
        heatmaps_layout.addWidget(self.strength_heatmap)
        self.strength_heatmap.setVisible(False)  # Hide by default
        self.strength_placeholder = QLabel("No strength score data available.")
        self.strength_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heatmaps_layout.addWidget(self.strength_placeholder)

        # Z-matrix heatmap
        heatmaps_layout.addWidget(QLabel("Z-Matrix Heatmap:"))
        self.zmatrix_heatmap = pg.ImageView()
        heatmaps_layout.addWidget(self.zmatrix_heatmap)
        self.zmatrix_heatmap.setVisible(False)
        self.zmatrix_placeholder = QLabel("No Z-matrix data available.")
        self.zmatrix_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heatmaps_layout.addWidget(self.zmatrix_placeholder)

        heatmaps_widget.setLayout(heatmaps_layout)
        self.single_tabs.addTab(heatmaps_widget, "Heatmaps")
        
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
        result_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.results_history[result_id] = {
            'data': result_data,
            'strategy': strategy_name,
            'timestamp': datetime.now(),
            'id': result_id
        }
        
        # Update UI
        self.results_combo.addItem(f"{strategy_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}", result_id)
        self.compare_list.addItem(f"{strategy_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Save results
        self._save_results_history()
        
    def _on_result_selected(self, index: int):
        """Handle result selection"""
        if index < 0:
            return
            
        result_id = self.results_combo.currentData()
        if result_id and result_id in self.results_history:
            self._display_result(self.results_history[result_id])
            
    def _display_result(self, result: Dict[str, Any]):
        """Display a single result"""
        # Update result info
        status_text = ""
        if result['data'].get('status') == 'created':
            status_text = "\nStatus: Strategy created (no backtest results yet)"
        
        info_text = f"""
Result ID: {result['id']}
Strategy: {result['strategy']}
Timestamp: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}{status_text}

Summary:
Total Return: {result['data'].get('total_return', 0):.2%}
Sharpe Ratio: {result['data'].get('sharpe_ratio', 0):.2f}
Max Drawdown: {result['data'].get('max_drawdown', 0):.2%}
Total Trades: {result['data'].get('total_trades', 0)}
"""
        self.result_info.setText(info_text)
        
        # Update metrics
        for key, label in self.single_metrics.items():
            value = result['data'].get(key, 0)
            if key in ['total_return', 'max_drawdown', 'win_rate']:
                label.setText(f"{value:.2%}")
            elif key in ['sharpe_ratio', 'profit_factor']:
                label.setText(f"{value:.2f}")
            elif key == 'total_trades':
                label.setText(str(value))
            else:
                label.setText(f"${value:.2f}")
                
        # Update charts
        self._update_single_charts(result['data'])
        
        # Update strategy details
        self._update_strategy_details(result)

        # Update analysis tabs
        self._update_returns_histogram(result['data'])
        self._update_mae_plot(result['data'])
        self._update_heatmaps(result['data'])
        
    def _update_single_charts(self, data: Dict[str, Any]):
        """Update single view charts"""
        # Clear charts
        self.single_equity_chart.clear()
        self.single_drawdown_chart.clear()
        
        # Equity curve
        equity_curve = data.get('equity_curve', [])
        if equity_curve:
            x = np.arange(len(equity_curve))
            self.single_equity_chart.plot(x, equity_curve, pen='w')
            
        # Drawdown
        if equity_curve:
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max * 100
            self.single_drawdown_chart.plot(x, drawdown, pen='r', fillLevel=0, brush=(255, 0, 0, 50))
            
    def _update_returns_histogram(self, data: Dict[str, Any]):
        """Update the trade returns histogram"""
        self.returns_hist.clear()
        trades = data.get('trades', [])
        if trades and isinstance(trades, list) and trades and 'pnl' in trades[0]:
            pnls = [t.get('pnl', 0) for t in trades]
            if pnls:
                y, x = np.histogram(pnls, bins=30)
                x_centers = (x[:-1] + x[1:]) / 2
                bg = pg.BarGraphItem(x=x_centers, height=y, width=(x[1] - x[0]), brush='b')
                self.returns_hist.addItem(bg)

    def _update_mae_plot(self, data: Dict[str, Any]):
        """Update the MAE vs PnL scatter plot"""
        self.mae_plot.clear()
        trades = data.get('trades', [])
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

    def _update_heatmaps(self, data: Dict[str, Any]):
        """Update the strength and Z-matrix heatmaps"""
        # Strength heatmap
        strength = data.get('strength', None)
        if strength is not None and np.array(strength).size > 0:
            self.strength_heatmap.setVisible(True)
            self.strength_placeholder.setVisible(False)
            arr = np.array(strength)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.strength_heatmap.setImage(arr)
        else:
            self.strength_heatmap.setVisible(False)
            self.strength_placeholder.setVisible(True)
        # Z-matrix heatmap
        zmat = data.get('z_matrix', None)
        if zmat is not None and np.array(zmat).size > 0:
            self.zmatrix_heatmap.setVisible(True)
            self.zmatrix_placeholder.setVisible(False)
            arr = np.array(zmat)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.zmatrix_heatmap.setImage(arr)
        else:
            self.zmatrix_heatmap.setVisible(False)
            self.zmatrix_placeholder.setVisible(True)
        
    def _update_strategy_details(self, result: Dict[str, Any]):
        """Update strategy details"""
        details = f"""
Strategy Configuration:
=====================
Name: {result['strategy']}
Type: Pattern Strategy

Parameters:
-----------
Initial Capital: $100,000
Position Size: 2%
Commission: $1.00 per trade
Slippage: 0.01%

Entry Rules:
-----------
- Pattern detection confirmed
- Volume above average
- Risk/Reward ratio >= 2:1

Exit Rules:
----------
- Fixed stop loss at 2%
- Take profit at 1:2 RR
- Time-based exit after 20 bars

Notes:
------
This strategy performed well in trending markets but struggled during consolidation periods.
Consider adding volatility filters to improve performance.
"""
        self.strategy_details.setText(details)
        
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
            item_text = f"{result['strategy']} - {result['timestamp'].strftime('%Y-%m-%d %H:%M')}"
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
        headers = ['Metric'] + [r['strategy'] for r in results]
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
                self.compare_equity_chart.plot(x, equity_curve, pen=color, 
                                             name=result['strategy'])
                
        # Plot metrics comparison
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'win_rate']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(results)
        
        for i, result in enumerate(results):
            values = [result['data'].get(m, 0) for m in metrics_to_plot]
            x_offset = x + (i - len(results)/2) * width
            
            color = colors[i % len(colors)]
            self.compare_metrics_chart.plot(x_offset, values, pen=None, 
                                          symbol='s', symbolSize=20,
                                          symbolBrush=color, name=result['strategy'])
            
        # Set x-axis labels
        ax = self.compare_metrics_chart.getAxis('bottom')
        ax.setTicks([[(i, m.replace('_', ' ').title()) for i, m in enumerate(metrics_to_plot)]])
        
    def _delete_result(self):
        """Delete selected result"""
        if self.results_combo.currentIndex() < 0:
            return
            
        reply = QMessageBox.question(self, "Delete Result", 
                                   "Are you sure you want to delete this result?",
                                   QMessageBox.StandardButton.Yes | 
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            result_id = self.results_combo.currentData()
            if result_id in self.results_history:
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
                        'Strategy': result['strategy'],
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
        """Load results history from file"""
        # Would implement loading from file
        # For now, add some mock data
        if self.parent_window and hasattr(self.parent_window, 'results'):
            for result_id, result in self.parent_window.results.items():
                self.add_result(result, f"Strategy_{result_id}")

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
