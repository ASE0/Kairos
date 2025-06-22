"""
gui/backtest_window.py
======================
Window for backtesting strategies
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pyqtgraph as pg
from datetime import datetime, timedelta
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import mplfinance as mpf

from core.data_structures import BaseStrategy


class CandlestickPlotWorker(QThread):
    candlestick_signal = pyqtSignal(int, float, float, float, float)  # index, open, close, low, high
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, ohlc_data, delay=0.02):
        super().__init__()
        self.ohlc_data = ohlc_data
        self.delay = delay  # seconds between candles
        self._is_running = True

    def run(self):
        for i, (o, h, l, c) in enumerate(self.ohlc_data):
            if not self._is_running:
                break
            self.candlestick_signal.emit(i, o, c, l, h)
            self.log_signal.emit(f"Plotted candlestick {i+1}/{len(self.ohlc_data)}: O={o}, H={h}, L={l}, C={c}")
            self.msleep(int(self.delay * 1000))
        self.finished.emit()

    def stop(self):
        self._is_running = False


class BacktestWorker(QThread):
    """Worker thread for running backtests"""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    trade_signal = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, parameters: Dict[str, Any]):
        super().__init__()
        self.data = data
        self.strategy = strategy
        self.parameters = parameters
        self.is_running = True

    def run(self):
        """Run backtest"""
        try:
            self.log.emit("Starting backtest...")

            # Import BacktestEngine here to avoid circular imports
            from strategies.strategy_builders import BacktestEngine
            engine = BacktestEngine()

            initial_capital = self.parameters.get('initial_capital', 100000)
            position_size = self.parameters.get('position_size', 0.02)
            # Use risk_per_trade as position_size for compatibility
            results = engine.run_backtest(
                self.strategy,
                self.data,
                initial_capital=initial_capital,
                risk_per_trade=position_size
            )

            self.log.emit("Backtest complete!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Stop backtest"""
        self.is_running = False

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float], initial_capital: float) -> Dict[str, Any]:
        """Calculate backtest metrics"""
        if not trades or not equity_curve:
            return {
                'total_trades': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'equity_curve': equity_curve,
                'trades': trades
            }

        returns = pd.Series(equity_curve).pct_change().dropna()
        total_return = (equity_curve[-1] - initial_capital) / initial_capital

        # Annual return calculation
        n_days = len(equity_curve)
        if n_days > 1:
            years = n_days / 252  # Assume 252 trading days/year
            annual_return = (equity_curve[-1] / initial_capital) ** (1 / years) - 1
        else:
            annual_return = 0

        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative = pd.Series(equity_curve)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        max_dd_duration = (drawdown == max_drawdown).sum() if max_drawdown < 0 else 0

        # Win rate
        n_wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        n_sells = sum(1 for t in trades if t['type'] == 'SELL')
        win_rate = n_wins / n_sells if n_sells > 0 else 0

        print(f"[DEBUG] Annual return: {annual_return:.4f}, Sharpe: {sharpe_ratio:.4f}, Max DD: {max_drawdown:.4f}, Win rate: {win_rate:.4f}")

        return {
            'total_trades': n_sells,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_dd_duration': max_dd_duration,
            'win_rate': win_rate,
            'equity_curve': equity_curve,
            'trades': trades
        }


class BacktestWindow(QMainWindow):
    """Window for backtesting strategies"""

    # Signals
    backtest_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Backtest Engine")
        self.setGeometry(450, 250, 1200, 800)

        # Current backtest
        self.current_worker = None
        self.current_results = None

        # Setup UI
        self._setup_ui()

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

        # Configuration section
        config_group = self._create_config_section()
        layout.addWidget(config_group)

        # Progress section
        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.start_btn = QPushButton("Start Backtest")
        self.start_btn.clicked.connect(self._start_backtest)
        progress_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_backtest)
        self.stop_btn.setEnabled(False)
        progress_layout.addWidget(self.stop_btn)

        layout.addLayout(progress_layout)

        # Tabs for results
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_overview_tab(), "Overview")
        self.tabs.addTab(self._create_equity_tab(), "Equity Curve")
        self.tabs.addTab(self._create_chart_tab(), "Chart")
        self.tabs.addTab(self._create_trades_tab(), "Trades")
        self.tabs.addTab(self._create_stats_tab(), "Statistics")
        self.tabs.addTab(self._create_log_tab(), "Log")
        layout.addWidget(self.tabs)

        # Populate datasets after UI setup
        self._populate_datasets()

    def _populate_datasets(self):
        """Populate dataset dropdown with available datasets from parent window."""
        self.dataset_combo.clear()
        self.dataset_combo.addItem("-- Select Dataset --")
        if self.parent_window and hasattr(self.parent_window, 'datasets'):
            for dataset_name in self.parent_window.datasets.keys():
                self.dataset_combo.addItem(dataset_name)
        # Connect signal for when user selects a dataset
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)

    def _on_dataset_selected(self, idx):
        """Set date pickers to the dataset's saved date range if available."""
        if idx <= 0:
            return
        dataset_name = self.dataset_combo.currentText()
        if self.parent_window and hasattr(self.parent_window, 'datasets'):
            dataset_info = self.parent_window.datasets.get(dataset_name)
            if dataset_info and 'metadata' in dataset_info:
                metadata = dataset_info['metadata']
                # Support both dict and dataclass
                date_range = None
                if hasattr(metadata, 'selected_date_range'):
                    date_range = metadata.selected_date_range
                elif isinstance(metadata, dict) and 'selected_date_range' in metadata:
                    date_range = metadata['selected_date_range']
                print(f"[DEBUG] Dataset '{dataset_name}' date_range in metadata: {date_range}")
                if date_range and len(date_range) == 2:
                    try:
                        # Try full ISO format first
                        from_date = QDateTime.fromString(date_range[0], Qt.DateFormat.ISODate)
                        to_date = QDateTime.fromString(date_range[1], Qt.DateFormat.ISODate)
                        # Fallback: try date only if time fails
                        if not from_date.isValid():
                            from_date = QDateTime.fromString(date_range[0][:19], "yyyy-MM-ddTHH:mm:ss")
                        if not to_date.isValid():
                            to_date = QDateTime.fromString(date_range[1][:19], "yyyy-MM-ddTHH:mm:ss")
                        print(f"[DEBUG] Parsed from_date: {from_date.toString(Qt.DateFormat.ISODate)}, valid: {from_date.isValid()}")
                        print(f"[DEBUG] Parsed to_date: {to_date.toString(Qt.DateFormat.ISODate)}, valid: {to_date.isValid()}")
                        if from_date.isValid():
                            self.start_date.setDateTime(from_date)
                        if to_date.isValid():
                            self.end_date.setDateTime(to_date)
                    except Exception as e:
                        print(f"[DEBUG] Exception parsing date range: {e}")

    def refresh_datasets(self):
        """Public method to refresh dataset dropdown."""
        self._populate_datasets()

    def _create_config_section(self) -> QGroupBox:
        """Create configuration section"""
        group = QGroupBox("Backtest Configuration")
        layout = QFormLayout()

        # Strategy selection
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("-- Select Strategy --")

        # Populate with strategies
        if self.parent_window and hasattr(self.parent_window, 'strategies'):
            for strategy_type, strategies in self.parent_window.strategies.items():
                for strategy_id, strategy in strategies.items():
                    self.strategy_combo.addItem(f"[{strategy_type}] {strategy.name}", strategy_id)

        layout.addRow("Strategy:", self.strategy_combo)

        # Sensitivity sliders area
        self.sensitivity_group = QGroupBox("Pattern Sensitivity")
        self.sensitivity_layout = QVBoxLayout()
        self.sensitivity_group.setLayout(self.sensitivity_layout)
        layout.addRow(self.sensitivity_group)

        self.strategy_combo.currentIndexChanged.connect(self._update_sensitivity_sliders)
        self.sensitivity_sliders = []

        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("-- Select Dataset --")
        layout.addRow("Dataset:", self.dataset_combo)

        # Date range
        date_layout = QHBoxLayout()

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-6))
        date_layout.addWidget(self.start_date)

        date_layout.addWidget(QLabel("to"))

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(self.end_date)

        layout.addRow("Date Range:", date_layout)

        # Capital settings
        self.initial_capital = QSpinBox()
        self.initial_capital.setRange(1000, 10000000)
        self.initial_capital.setValue(100000)
        self.initial_capital.setSingleStep(10000)
        self.initial_capital.setPrefix("$")
        layout.addRow("Initial Capital:", self.initial_capital)

        # Position sizing
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(0.01, 1.0)
        self.position_size.setValue(0.02)
        self.position_size.setSingleStep(0.01)
        self.position_size.setSuffix("%")
        layout.addRow("Position Size:", self.position_size)

        # Commission
        self.commission = QDoubleSpinBox()
        self.commission.setRange(0, 100)
        self.commission.setValue(1.0)
        self.commission.setSingleStep(0.5)
        self.commission.setPrefix("$")
        layout.addRow("Commission:", self.commission)

        # Slippage
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 0.01)
        self.slippage.setValue(0.0001)
        self.slippage.setSingleStep(0.0001)
        self.slippage.setDecimals(4)
        self.slippage.setSuffix("%")
        layout.addRow("Slippage:", self.slippage)

        group.setLayout(layout)
        return group

    def _update_sensitivity_sliders(self):
        # Remove old sliders
        for i in reversed(range(self.sensitivity_layout.count())):
            widget = self.sensitivity_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.sensitivity_sliders = []

        # Get selected strategy
        idx = self.strategy_combo.currentIndex()
        if idx <= 0 or not self.parent_window or not hasattr(self.parent_window, 'strategies'):
            return
        selected_text = self.strategy_combo.currentText()
        strategy = None
        for strategy_type, strategies in self.parent_window.strategies.items():
            for strategy_id, s in strategies.items():
                if f"[{strategy_type}] {s.name}" == selected_text:
                    strategy = s
                    break
            if strategy:
                break
        if not strategy or not hasattr(strategy, 'actions'):
            return

        # Only one slider for the whole strategy
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setSingleStep(1)
        slider.setValue(50)
        value_label = QLabel()
        value_label.setText(f"Sensitivity: {slider.value()/100:.2f}")

        def on_change(val):
            # Invert: right = more sensitive, left = less
            sensitivity = val / 100.0
            value_label.setText(f"Sensitivity: {sensitivity:.2f}")
            for action in getattr(strategy, 'actions', []):
                pattern = getattr(action, 'pattern', None)
                if not pattern:
                    continue
                pattern_name = getattr(pattern, 'name', type(pattern).__name__).lower()
                # DoubleWickPattern: min_wick_ratio (0.001-0.9), max_body_ratio (2.0-0.01)
                if pattern_name.startswith('double_wick'):
                    pattern.min_wick_ratio = 0.9 - sensitivity * (0.9 - 0.001)
                    pattern.max_body_ratio = 0.01 + sensitivity * (2.0 - 0.01)
                # IIBarsPattern: min_bars (30-1)
                elif pattern_name.startswith('ii_bars'):
                    pattern.min_bars = int(30 - sensitivity * (30-1))
                # HammerPattern: lower wick/body, upper wick/body, body/total_range
                elif pattern_name.startswith('hammer'):
                    # At high sensitivity, allow almost any bar: lower wick >= 0.1*body, upper wick <= 2.0*body, body/total_range <= 2.0
                    # At low sensitivity, strict: lower wick >= 10*body, upper wick <= 0.01*body, body/total_range <= 0.05
                    pattern._min_lower_wick_to_body = 10.0 - sensitivity * (10.0 - 0.1)
                    pattern._max_upper_wick_to_body = 0.01 + sensitivity * (2.0 - 0.01)
                    pattern._max_body_to_range = 0.05 + sensitivity * (2.0 - 0.05)
                # CustomPattern: ohlc_ratios[0].body_ratio, upper_wick_ratio, lower_wick_ratio (0.9-0.001)
                elif pattern_name.startswith('custom') or pattern_name.startswith('distribution'):
                    if hasattr(pattern, 'ohlc_ratios') and pattern.ohlc_ratios:
                        ratio = pattern.ohlc_ratios[0]
                        if hasattr(ratio, 'body_ratio') and ratio.body_ratio is not None:
                            ratio.body_ratio = 0.9 - sensitivity * (0.9 - 0.001)
                        if hasattr(ratio, 'upper_wick_ratio') and ratio.upper_wick_ratio is not None:
                            ratio.upper_wick_ratio = 0.9 - sensitivity * (0.9 - 0.001)
                        if hasattr(ratio, 'lower_wick_ratio') and ratio.lower_wick_ratio is not None:
                            ratio.lower_wick_ratio = 0.9 - sensitivity * (0.9 - 0.001)
                # Add more pattern types as needed

        slider.valueChanged.connect(on_change)
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.addWidget(QLabel("Strategy Sensitivity"))
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        self.sensitivity_layout.addWidget(row_widget)
        self.sensitivity_sliders.append((slider, value_label))

    def _create_overview_tab(self) -> QWidget:
        """Create overview tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Summary metrics
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        layout.addWidget(self.summary_text)

        # Key metrics grid
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QGridLayout()

        self.metric_labels = {}
        metrics = [
            ('Total Return', 'total_return', 0, 0),
            ('Sharpe Ratio', 'sharpe_ratio', 0, 1),
            ('Max Drawdown', 'max_drawdown', 0, 2),
            ('Win Rate', 'win_rate', 1, 0),
            ('Total Trades', 'total_trades', 1, 1),
            ('Profit Factor', 'profit_factor', 1, 2)
        ]

        for label, key, row, col in metrics:
            container = QWidget()
            container_layout = QVBoxLayout(container)

            title = QLabel(label)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(title)

            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 20px; font-weight: bold;")
            container_layout.addWidget(value_label)

            self.metric_labels[key] = value_label
            metrics_layout.addWidget(container, row, col)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def _create_equity_tab(self) -> QWidget:
        """Create equity curve tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Equity curve chart
        self.equity_chart = pg.PlotWidget()
        self.equity_chart.setLabel('left', 'Equity ($)')
        self.equity_chart.setLabel('bottom', 'Time')
        self.equity_chart.showGrid(True, True)

        # Add legend
        self.equity_chart.addLegend()

        layout.addWidget(self.equity_chart)

        # Statistics below chart
        stats_layout = QHBoxLayout()

        self.equity_stats = QLabel("Equity statistics1 will appear here")
        stats_layout.addWidget(self.equity_stats)

        layout.addLayout(stats_layout)

        widget.setLayout(layout)
        return widget

    def _create_chart_tab(self) -> QWidget:
        """Create chart tab with price, trades, patterns, and overlays (Matplotlib version)"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Matplotlib Figure and Canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Overlay toggles (keep for future use)
        self.overlay_toggles = {}
        overlays = ['Entries/Exits', 'Pattern Signals', 'FVGs', 'Indicators']
        toggle_layout = QHBoxLayout()
        for name in overlays:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_chart_overlays)
            self.overlay_toggles[name] = cb
            toggle_layout.addWidget(cb)
        layout.addLayout(toggle_layout)

        widget.setLayout(layout)
        return widget

    def _create_trades_tab(self) -> QWidget:
        """Create trades tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            'Exit Time', 'Type', 'Exit Price', 'Size', 'P&L', 'Entry Price', 'Holding Period', 'Cumulative P&L'
        ])
        layout.addWidget(self.trades_table)

        # Add method to update trades table
        def update_trades_table(trades):
            self.trades_table.setRowCount(len(trades))
            cum_pnl = 0
            for row, trade in enumerate(trades):
                # Time: exit_time
                self.trades_table.setItem(row, 0, QTableWidgetItem(str(trade.get('exit_time', ''))))
                # Type: always 'SELL' (since only closing trades are recorded)
                self.trades_table.setItem(row, 1, QTableWidgetItem('SELL'))
                # Exit Price
                self.trades_table.setItem(row, 2, QTableWidgetItem(f"{trade.get('exit_price', 0):.2f}"))
                # Size
                self.trades_table.setItem(row, 3, QTableWidgetItem(f"{trade.get('size', 0):.4f}"))
                # P&L
                pnl = trade.get('pnl', 0)
                self.trades_table.setItem(row, 4, QTableWidgetItem(f"{pnl:.2f}"))
                # Entry Price
                self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade.get('entry_price', 0):.2f}"))
                # Holding Period
                holding = ''
                if trade.get('entry_time') and trade.get('exit_time'):
                    try:
                        holding = str(pd.to_datetime(trade['exit_time']) - pd.to_datetime(trade['entry_time']))
                    except Exception:
                        holding = ''
                self.trades_table.setItem(row, 6, QTableWidgetItem(holding))
                # Cumulative P&L
                cum_pnl += pnl
                self.trades_table.setItem(row, 7, QTableWidgetItem(f"{cum_pnl:.2f}"))

        self.update_trades_table = update_trades_table
        widget.setLayout(layout)
        return widget

    def _create_stats_tab(self) -> QWidget:
        """Create detailed statistics1 tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Monthly returns table
        layout.addWidget(QLabel("Monthly Returns:"))
        self.monthly_table = QTableWidget()
        self.monthly_table.setMaximumHeight(200)
        layout.addWidget(self.monthly_table)

        # Distribution chart
        self.distribution_chart = pg.PlotWidget()
        self.distribution_chart.setLabel('left', 'Frequency')
        self.distribution_chart.setLabel('bottom', 'Returns (%)')
        layout.addWidget(QLabel("Returns Distribution:"))
        layout.addWidget(self.distribution_chart)

        # Additional statistics1
        self.detailed_stats = QTextEdit()
        self.detailed_stats.setReadOnly(True)
        layout.addWidget(QLabel("Detailed Statistics:"))
        layout.addWidget(self.detailed_stats)

        widget.setLayout(layout)
        return widget

    def _create_log_tab(self) -> QWidget:
        """Create log tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        widget.setLayout(layout)
        return widget

    def _start_backtest(self):
        """Start backtest"""
        # Validate inputs
        if self.strategy_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select a strategy")
            return

        if self.dataset_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select a dataset")
            return

        # Create mock data for testing
        dates = pd.date_range(
            start=self.start_date.date().toPyDate(),
            end=self.end_date.date().toPyDate(),
            freq='5T'  # 5-minute bars
        )

        # Create mock OHLCV data
        np.random.seed(42)
        base_price = 100
        data = pd.DataFrame(index=dates)

        # Generate realistic price data
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        data['open'] = prices * (1 + np.random.uniform(-0.001, 0.001, len(dates)))
        data['high'] = prices * (1 + np.random.uniform(0, 0.002, len(dates)))
        data['low'] = prices * (1 + np.random.uniform(-0.002, 0, len(dates)))
        data['close'] = prices
        data['volume'] = np.random.lognormal(10, 1, len(dates))

        # Get strategy
        strategy = None
        if self.parent_window and hasattr(self.parent_window, 'strategies'):
            selected_text = self.strategy_combo.currentText()
            found = False
            for strategy_type, strategies in self.parent_window.strategies.items():
                for strategy_id, s in strategies.items():
                    if f"[{strategy_type}] {s.name}" == selected_text:
                        strategy = s
                        found = True
                        break
                if found:
                    break
        if strategy is None:
            QMessageBox.warning(self, "Warning", "Could not find the selected strategy object.")
            return

        # Prepare parameters
        parameters = {
            'initial_capital': self.initial_capital.value(),
            'position_size': self.position_size.value() / 100,
            'commission': self.commission.value(),
            'slippage': self.slippage.value() / 100
        }

        # Create worker thread
        self.current_worker = BacktestWorker(data, strategy, parameters)
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.log.connect(self._add_log)
        self.current_worker.trade_signal.connect(self._on_trade_signal)
        self.current_worker.finished.connect(self._on_backtest_complete)
        self.current_worker.error.connect(self._on_error)

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Clear previous results
        self._clear_results()

        # Start backtest
        self.current_worker.start()

    def _stop_backtest(self):
        """Stop running backtest"""
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker.quit()
            self.current_worker.wait()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._add_log("Backtest stopped by user")

    def _add_log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _on_trade_signal(self, trade: Dict[str, Any]):
        """Handle trade signal"""
        # Add to trades table
        row = self.trades_table.rowCount()
        self.trades_table.insertRow(row)

        self.trades_table.setItem(row, 0, QTableWidgetItem(str(trade['exit_time'])))
        self.trades_table.setItem(row, 1, QTableWidgetItem('SELL'))
        self.trades_table.setItem(row, 2, QTableWidgetItem(f"{trade['exit_price']:.2f}"))
        self.trades_table.setItem(row, 3, QTableWidgetItem(f"{trade['size']:.2f}"))
        self.trades_table.setItem(row, 4, QTableWidgetItem(f"{trade['pnl']:.2f}"))
        self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade['entry_price']:.2f}"))
        self.trades_table.setItem(row, 6, QTableWidgetItem(str(trade['holding_period'])))
        self.trades_table.setItem(row, 7, QTableWidgetItem(f"{trade['cumulative_pnl']:.2f}"))

    def _on_backtest_complete(self, results: Dict[str, Any]):
        """Handle backtest completion"""
        self.current_results = results

        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        # Update overview
        self._update_overview(results)

        # Update equity curve
        self._update_equity_curve(results)

        # Update trade statistics
        self._update_trade_stats(results)

        # Update detailed statistics1
        self._update_detailed_stats(results)

        # Update chart
        self._update_chart_tab(results)

        # Emit signal
        self.backtest_complete.emit(results)

        self._add_log(f"Backtest complete: {results['total_trades']} trades, "
                     f"{results['total_return']:.2%} return")

    def _on_error(self, error: str):
        """Handle backtest error"""
        QMessageBox.critical(self, "Backtest Error", error)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._add_log(f"ERROR: {error}")

    def _clear_results(self):
        """Clear previous results"""
        self.summary_text.clear()
        self.trades_table.setRowCount(0)
        self.equity_chart.clear()
        self.distribution_chart.clear()
        self.detailed_stats.clear()

        for label in self.metric_labels.values():
            label.setText("--")

    def _update_overview(self, results: Dict[str, Any]):
        """Update overview tab"""
        # Summary text
        summary = f"""
Backtest Summary:
================
Total Trades: {results['total_trades']}
Total Return: {results['total_return']:.2%}
Sharpe Ratio: {results['sharpe_ratio']:.2f}
Max Drawdown: {results['max_drawdown']:.2%}
Win Rate: {results['win_rate']:.2%}

Strategy: {self.strategy_combo.currentText()}
Period: {self.start_date.date().toString()} to {self.end_date.date().toString()}
Initial Capital: ${self.initial_capital.value():,}
"""
        self.summary_text.setText(summary)

        # Update metric labels
        self.metric_labels['total_return'].setText(f"{results['total_return']:.2%}")
        self.metric_labels['sharpe_ratio'].setText(f"{results['sharpe_ratio']:.2f}")
        self.metric_labels['max_drawdown'].setText(f"{results['max_drawdown']:.2%}")
        self.metric_labels['win_rate'].setText(f"{results['win_rate']:.2%}")
        self.metric_labels['total_trades'].setText(str(results['total_trades']))
        self.metric_labels['profit_factor'].setText("1.85")  # Mock value

        # Color code metrics
        if results['total_return'] > 0:
            self.metric_labels['total_return'].setStyleSheet("color: green;")
        else:
            self.metric_labels['total_return'].setStyleSheet("color: red;")

        if results['sharpe_ratio'] > 1:
            self.metric_labels['sharpe_ratio'].setStyleSheet("color: green;")
        elif results['sharpe_ratio'] < 0:
            self.metric_labels['sharpe_ratio'].setStyleSheet("color: red;")
        else:
            self.metric_labels['sharpe_ratio'].setStyleSheet("color: orange;")

    def _update_equity_curve(self, results: Dict[str, Any]):
        """Update equity curve chart"""
        self.equity_chart.clear()

        equity_curve = results.get('equity_curve', [])
        if not equity_curve:
            return

        # Plot equity curve
        x = np.arange(len(equity_curve))
        self.equity_chart.plot(x, equity_curve, pen='w', name='Equity')

        # Add initial capital line
        self.equity_chart.plot([0, len(equity_curve)-1],
                             [self.initial_capital.value(), self.initial_capital.value()],
                             pen=pg.mkPen('y', style=Qt.PenStyle.DashLine),
                             name='Initial Capital')

        # Calculate and display statistics1
        equity_series = pd.Series(equity_curve)
        stats_text = f"Final Equity: ${equity_curve[-1]:,.2f} | "
        stats_text += f"Peak: ${equity_series.max():,.2f} | "
        stats_text += f"Max Drawdown: {results['max_drawdown']:.2%}"
        self.equity_stats.setText(stats_text)

    def _update_trade_stats(self, results: Dict[str, Any]):
        """Update trade statistics tab (skip avg_win_label, etc. as they no longer exist)"""
        # This method previously updated labels that have been removed.
        # If you want to show summary stats, add them as a summary label or in the table.
        # For now, just update the trades table:
        trades = results.get('trades', [])
        self.update_trades_table(trades)

    def _update_detailed_stats(self, results: Dict[str, Any]):
        """Update detailed statistics tab with real data"""
        # --- Monthly returns ---
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            equity_series = pd.Series(equity_curve)
            n = len(equity_series)
            if 'trades' in results and results['trades'] and 'entry_time' in results['trades'][0]:
                try:
                    start = pd.to_datetime(results['trades'][0]['entry_time'])
                    end = pd.to_datetime(results['trades'][-1]['exit_time'])
                    idx = pd.date_range(start, end, periods=n)
                    equity_series.index = idx
                except Exception:
                    equity_series.index = pd.date_range('2023-01-01', periods=n, freq='D')
            else:
                equity_series.index = pd.date_range('2023-01-01', periods=n, freq='D')
            monthly_returns = equity_series.resample('M').last().pct_change().dropna()
            # If only one month, set monthly return to total return
            if len(monthly_returns) == 1:
                monthly_returns.iloc[0] = results.get('total_return', 0)
        else:
            monthly_returns = pd.Series(dtype=float)

        months = list(monthly_returns.index.strftime('%b %Y'))
        returns = monthly_returns.values
        self.monthly_table.setRowCount(1)
        self.monthly_table.setColumnCount(len(months))
        self.monthly_table.setHorizontalHeaderLabels(months)
        for i, ret in enumerate(returns):
            item = QTableWidgetItem(f"{ret:.2%}")
            if ret > 0:
                item.setBackground(QColor(0, 255, 0, 50))
            else:
                item.setBackground(QColor(255, 0, 0, 50))
            self.monthly_table.setItem(0, i, item)

        # --- Returns distribution (trade P&L) ---
        self.distribution_chart.clear()
        trades = results.get('trades', [])
        trade_returns = [t.get('pnl', 0) / t['entry_price'] for t in trades if t.get('pnl') is not None and t.get('entry_price')]
        if trade_returns:
            y, x = np.histogram(trade_returns, bins=30)
            # For stepMode=True, x must be len(y)+1
            self.distribution_chart.plot(x, y, stepMode=True, fillLevel=0, brush='b')

        # --- Detailed statistics ---
        total_return = results.get('total_return', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)
        trades_list = trades
        win_rate = results.get('win_rate', 0)
        n_trades = results.get('total_trades', 0)
        avg_trade = np.mean(trade_returns) if trade_returns else 0
        best_trade = np.max(trade_returns) if trade_returns else 0
        worst_trade = np.min(trade_returns) if trade_returns else 0
        # Calculate average holding time if timestamps are available
        avg_holding = ''
        if trades_list and 'entry_time' in trades_list[0] and 'exit_time' in trades_list[0]:
            holding_times = [pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time']) for t in trades_list if t.get('entry_time') and t.get('exit_time')]
            if holding_times:
                avg_holding = str(sum(holding_times, pd.Timedelta(0)) / len(holding_times))
        stats_text = f"""
Performance Statistics:
=====================
Total Return: {total_return:.2%}
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.2%}

Trade Analysis:
==============
Total Trades: {n_trades}
Win Rate: {win_rate:.2%}
Average Trade: {avg_trade:.2%}
Best Trade: {best_trade:.2%}
Worst Trade: {worst_trade:.2%}
Avg Holding Time: {avg_holding}
"""
        self.detailed_stats.setText(stats_text)

    def _update_chart_tab(self, results: dict):
        import logging
        logger = logging.getLogger("ChartTabDebug")
        self.ax.clear()
        data = results.get('data')
        logger.info(f"_update_chart_tab called. results keys: {list(results.keys())}")
        if data is None and hasattr(self, 'current_worker') and self.current_worker is not None:
            data = getattr(self.current_worker, 'data', None)
            logger.info(f"Tried current_worker, got data: {type(data)}")
        if data is None:
            if hasattr(self, 'last_data'):
                data = self.last_data
                logger.info(f"Tried last_data, got data: {type(data)}")
        if data is not None and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            try:
                # Prepare DataFrame for mplfinance
                df = data.copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'datetime' in df.columns:
                        df.index = pd.to_datetime(df['datetime'])
                    elif 'Date' in df.columns:
                        df.index = pd.to_datetime(df['Date'])
                    else:
                        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
                df = df[['open', 'high', 'low', 'close']]
                df.columns = ['Open', 'High', 'Low', 'Close']
                # Plot candlesticks
                mpf.plot(df, type='candle', ax=self.ax, style='charles', show_nontrading=True)

                # Plot zones if they exist
                zones = results.get('zones')
                if zones:
                    logger.info(f"Plotting {len(zones)} zones.")
                    for zone in zones:
                        zone_min = zone.get('zone_min')
                        zone_max = zone.get('zone_max')
                        zone_idx = zone.get('index')
                        
                        # Ensure the zone's index is within the plotted data's range
                        if zone_min is not None and zone_max is not None and zone_idx is not None and 0 <= zone_idx < len(df):
                            # Define the time range for the zone visualization
                            start_time = df.index[zone_idx]
                            # Extend the zone visualization for a few bars for better visibility
                            end_idx = min(zone_idx + 5, len(df) - 1)
                            end_time = df.index[end_idx]
                            
                            self.ax.fill_between([start_time, end_time], zone_min, zone_max, 
                                                 color='blue', alpha=0.15, zorder=1)

                # Overlay entries/exits
                if self.overlay_toggles['Entries/Exits'].isChecked() and 'trades' in results:
                    trades = results['trades']
                    logger.info(f"Plotting {len(trades)} trades.")
                    entry_plotted = False
                    exit_plotted = False
                    for trade in trades:
                        # Try to use entry_time/exit_time if available
                        entry_time = trade.get('entry_time')
                        exit_time = trade.get('exit_time')
                        entry_price = trade.get('entry_price')
                        exit_price = trade.get('exit_price')
                        entry_idx = trade.get('entry_idx')
                        exit_idx = trade.get('exit_idx')
                        # Entry marker
                        entry_dt = None
                        if entry_time is not None:
                            try:
                                entry_dt = pd.to_datetime(entry_time)
                                if entry_dt in df.index:
                                    self.ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                                    print(f"[DEBUG] Plotted entry at {entry_dt} price {entry_price}")
                                    entry_plotted = True
                                else:
                                    print(f"[DEBUG] Entry time {entry_dt} not in df.index")
                            except Exception as e:
                                print(f"[DEBUG] Could not parse entry_time {entry_time}: {e}")
                        elif entry_idx is not None and entry_price is not None and 0 <= entry_idx < len(df):
                            entry_dt = df.index[entry_idx]
                            self.ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                            print(f"[DEBUG] Plotted entry at idx {entry_idx} (time {entry_dt}) price {entry_price}")
                            entry_plotted = True
                        else:
                            print(f"[DEBUG] Skipped entry: idx={entry_idx}, price={entry_price}, time={entry_time}")
                        # Exit marker
                        exit_dt = None
                        if exit_time is not None:
                            try:
                                exit_dt = pd.to_datetime(exit_time)
                                if exit_dt in df.index:
                                    self.ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                                    print(f"[DEBUG] Plotted exit at {exit_dt} price {exit_price}")
                                    exit_plotted = True
                                else:
                                    print(f"[DEBUG] Exit time {exit_dt} not in df.index")
                            except Exception as e:
                                print(f"[DEBUG] Could not parse exit_time {exit_time}: {e}")
                        elif exit_idx is not None and exit_price is not None and 0 <= exit_idx < len(df):
                            exit_dt = df.index[exit_idx]
                            self.ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                            print(f"[DEBUG] Plotted exit at idx {exit_idx} (time {exit_dt}) price {exit_price}")
                            exit_plotted = True
                        else:
                            print(f"[DEBUG] Skipped exit: idx={exit_idx}, price={exit_price}, time={exit_time}")
                    # Only show legend if at least one entry or exit was plotted
                    handles, labels = self.ax.get_legend_handles_labels()
                    if handles:
                        self.ax.legend()
            except Exception as e:
                logger.error(f"Matplotlib candlestick plotting failed: {e}")
        else:
            self.ax.text(0.5, 0.5, 'No real price data available for chart.', ha='center', va='center', color='red', fontsize=14)
        self.canvas.draw()

    def _update_chart_overlays(self):
        if self.current_results:
            self._update_chart_tab(self.current_results)