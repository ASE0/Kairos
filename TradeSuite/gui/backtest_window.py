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
import pprint

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
        self._should_stop = False

    def run(self):
        """Run backtest with progress updates"""
        try:
            self.log.emit("Starting backtest...")

            initial_capital = self.parameters.get('initial_capital', 100000)
            position_size = self.parameters.get('position_size', 0.02)
            
            # Check if we should use the new architecture
            use_new_architecture = self._should_use_new_architecture()
            
            if use_new_architecture:
                self.log.emit("Using NEW MODULAR ARCHITECTURE...")
                print(f"[DEBUG] BacktestWorker: Using new architecture for strategy: {self.strategy.name}")
                
                # Convert strategy to new architecture format
                strategy_config = self._convert_to_new_architecture_config()
                
                # Use new architecture
                try:
                    from core.new_gui_integration import new_gui_integration
                    results = new_gui_integration.run_strategy_backtest(
                        strategy_config, 
                        self.data,
                        initial_capital=initial_capital,
                        risk_per_trade=position_size
                    )
                    print(f"[DEBUG] BacktestWorker: New architecture backtest completed successfully")
                except Exception as e:
                    print(f"[DEBUG] BacktestWorker: New architecture failed: {e}")
                    self.log.emit(f"New architecture failed, falling back to old system: {e}")
                    use_new_architecture = False
            
            if not use_new_architecture:
                self.log.emit("Using legacy backtest engine...")
                print(f"[DEBUG] BacktestWorker: Using old architecture for strategy: {self.strategy.name}")
                
                # Import MultiTimeframeBacktestEngine here to avoid circular imports
                from strategies.strategy_builders import MultiTimeframeBacktestEngine
                engine = MultiTimeframeBacktestEngine()
                
                # Process in chunks to allow cancellation and progress updates
                chunk_size = 1000
                total_bars = len(self.data)
                
                for i in range(0, total_bars, chunk_size):
                    if self._should_stop:
                        self.log.emit("Backtest cancelled by user")
                        return
                    
                    # Update progress
                    progress = int((i / total_bars) * 100)
                    self.progress.emit(progress)
                    
                    # Allow GUI updates
                    self.msleep(1)
                
                # Use risk_per_trade as position_size for compatibility
                results = engine.run_backtest(
                    self.strategy,
                    self.data,
                    initial_capital=initial_capital,
                    risk_per_trade=position_size
                )

            import pprint
            with open('gui_debug_output.txt', 'a', encoding='utf-8') as f:
                f.write("\n[DEBUG] Strategy actions and patterns:\n")
                for i, action in enumerate(self.strategy.actions):
                    f.write(f"  Action {i}: {repr(action)}\n")
                    if hasattr(action, 'pattern'):
                        f.write(f"    Pattern: {repr(getattr(action, 'pattern', None))}\n")
                f.write("[DEBUG] Raw results['zones']:\n")
                pprint.pprint(results.get('zones', []), stream=f)

            self.progress.emit(100)
            self.log.emit("Backtest complete!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Stop backtest gracefully"""
        self._should_stop = True
        self.is_running = False

    def _calculate_metrics(self, trades: List[Dict], equity_curve, initial_capital: float) -> Dict[str, Any]:
        """Calculate backtest metrics"""
        # Handle both old list format and new Series format for equity_curve
        equity_empty = False
        if isinstance(equity_curve, pd.Series):
            equity_empty = equity_curve.empty
        elif isinstance(equity_curve, list):
            equity_empty = not equity_curve
        else:
            equity_empty = True
            
        if not trades or equity_empty:
            return {
                'total_trades': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'equity_curve': equity_curve,
                'trades': trades
            }

        # Handle both Series and list formats
        if isinstance(equity_curve, pd.Series):
            returns = equity_curve.pct_change().dropna()
            final_value = equity_curve.iloc[-1]
            n_days = len(equity_curve)
        else:
            returns = pd.Series(equity_curve).pct_change().dropna()
            final_value = equity_curve[-1]
            n_days = len(equity_curve)
        
        total_return = (final_value - initial_capital) / initial_capital

        # Annual return calculation
        if n_days > 1:
            years = n_days / 252  # Assume 252 trading days/year
            annual_return = (final_value / initial_capital) ** (1 / years) - 1
        else:
            annual_return = 0

        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0

        # Max drawdown
        if isinstance(equity_curve, pd.Series):
            cumulative = equity_curve
        else:
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
    
    def _should_use_new_architecture(self) -> bool:
        """Determine if we should use the new modular architecture"""
        try:
            # Check if this is a JSON-based strategy (new format)
            if hasattr(self.strategy, 'to_dict'):
                strategy_dict = self.strategy.to_dict()
            elif hasattr(self.strategy, '__dict__'):
                strategy_dict = self.strategy.__dict__
            else:
                return False
            
            # Look for filter-only strategies with recognized filters
            if 'actions' in strategy_dict:
                for action in strategy_dict['actions']:
                    filters = action.get('filters', [])
                    for filter_config in filters:
                        filter_type = filter_config.get('type', '')
                        # Check if it's a filter type that the new architecture supports
                        if filter_type in ['vwap', 'momentum', 'volatility', 'ma', 'bollinger_bands']:
                            print(f"[DEBUG] BacktestWorker: Found supported filter type: {filter_type}")
                            return True
            
            return False
            
        except Exception as e:
            print(f"[DEBUG] BacktestWorker: Error checking for new architecture: {e}")
            return False
    
    def _convert_to_new_architecture_config(self) -> dict:
        """Convert strategy to new architecture format"""
        try:
            if hasattr(self.strategy, 'to_dict'):
                strategy_dict = self.strategy.to_dict()
            elif hasattr(self.strategy, '__dict__'):
                strategy_dict = self.strategy.__dict__.copy()
            else:
                strategy_dict = {
                    'name': getattr(self.strategy, 'name', 'Unknown Strategy'),
                    'actions': []
                }
            
            # Ensure it has the right structure
            if 'name' not in strategy_dict:
                strategy_dict['name'] = getattr(self.strategy, 'name', 'Unknown Strategy')
            
            if 'actions' not in strategy_dict:
                strategy_dict['actions'] = []
            
            if 'combination_logic' not in strategy_dict:
                strategy_dict['combination_logic'] = 'AND'
            
            if 'gates_and_logic' not in strategy_dict:
                strategy_dict['gates_and_logic'] = {}
            
            print(f"[DEBUG] BacktestWorker: Converted strategy config: {strategy_dict}")
            return strategy_dict
            
        except Exception as e:
            print(f"[DEBUG] BacktestWorker: Error converting strategy config: {e}")
            return {
                'name': 'Fallback Strategy',
                'actions': [],
                'combination_logic': 'AND',
                'gates_and_logic': {}
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
        
        # Data caching for performance
        self._data_cache = {}  # Cache processed datasets

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

        # Control buttons
        self.start_button = QPushButton("Start Backtest")
        self.start_button.clicked.connect(self._start_backtest)
        
        self.stop_button = QPushButton("Stop Backtest")
        self.stop_button.clicked.connect(self._stop_backtest)
        self.stop_button.setEnabled(False)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()

        progress_layout.addLayout(button_layout)

        # Add View Results button
        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.clicked.connect(self._open_results_viewer)
        self.view_results_btn.setEnabled(False)  # Initially disabled
        progress_layout.addWidget(self.view_results_btn)

        layout.addLayout(progress_layout)

        # Tabs for results
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_overview_tab(), "Overview")
        self.tabs.addTab(self._create_equity_tab(), "Equity Curve")
        self.chart_tab = self._create_chart_tab()
        self.tabs.addTab(self.chart_tab, "Chart")
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
        # Auto-select last loaded dataset if available
        if self.parent_window and hasattr(self.parent_window, 'last_loaded_dataset'):
            last = self.parent_window.last_loaded_dataset
            idx = self.dataset_combo.findText(last)
            if idx > 0:
                self.dataset_combo.setCurrentIndex(idx)

    def _on_dataset_selected(self, idx):
        """Set date pickers to the dataset's saved date range if available, otherwise use actual data range."""
        if idx <= 0:
            return
        dataset_name = self.dataset_combo.currentText()
        if self.parent_window and hasattr(self.parent_window, 'datasets'):
            dataset_info = self.parent_window.datasets.get(dataset_name)
            if dataset_info and 'data' in dataset_info:
                data = dataset_info['data']
                metadata = dataset_info.get('metadata', {})
                
                # Try to get date range from metadata first
                date_range = None
                if hasattr(metadata, 'selected_date_range'):
                    date_range = metadata.selected_date_range
                elif isinstance(metadata, dict) and 'selected_date_range' in metadata:
                    date_range = metadata['selected_date_range']
                
                print(f"[DEBUG] Dataset '{dataset_name}' date_range in metadata: {date_range}")
                
                # Try to parse metadata date range
                if date_range and len(date_range) == 2:
                    try:
                        # Parse the ISO format strings manually to avoid timezone issues
                        from PyQt6.QtCore import QDate, QTime, Qt
                        
                        # Parse start date
                        start_str = date_range[0]
                        if 'T' in start_str:
                            date_part, time_part = start_str.split('T')
                            time_part = time_part.split('.')[0]  # Remove microseconds if present
                        else:
                            date_part = start_str
                            time_part = "00:00:00"
                        
                        start_date_parts = date_part.split('-')
                        start_time_parts = time_part.split(':')
                        
                        start_date = QDate(int(start_date_parts[0]), int(start_date_parts[1]), int(start_date_parts[2]))
                        start_time = QTime(int(start_time_parts[0]), int(start_time_parts[1]), int(start_time_parts[2]))
                        from_date = QDateTime(start_date, start_time, Qt.TimeSpec.LocalTime)
                        
                        # Parse end date
                        end_str = date_range[1]
                        if 'T' in end_str:
                            date_part, time_part = end_str.split('T')
                            time_part = time_part.split('.')[0]  # Remove microseconds if present
                        else:
                            date_part = end_str
                            time_part = "00:00:00"
                        
                        end_date_parts = date_part.split('-')
                        end_time_parts = time_part.split(':')
                        
                        end_date = QDate(int(end_date_parts[0]), int(end_date_parts[1]), int(end_date_parts[2]))
                        end_time = QTime(int(end_time_parts[0]), int(end_time_parts[1]), int(end_time_parts[2]))
                        to_date = QDateTime(end_date, end_time, Qt.TimeSpec.LocalTime)
                        
                        print(f"[DEBUG] Parsed from_date: {from_date.toString(Qt.DateFormat.ISODate)}, valid: {from_date.isValid()}")
                        print(f"[DEBUG] Parsed to_date: {to_date.toString(Qt.DateFormat.ISODate)}, valid: {to_date.isValid()}")
                        
                        # Only use metadata dates if they're valid
                        if from_date.isValid() and to_date.isValid():
                            self.start_date.setDateTime(from_date)
                            self.end_date.setDateTime(to_date)
                            print(f"[DEBUG] Set date range from metadata: {from_date.toString(Qt.DateFormat.ISODate)} to {to_date.toString(Qt.DateFormat.ISODate)}")
                            return
                    except Exception as e:
                        print(f"[DEBUG] Exception parsing date range: {e}")
                
                # If metadata date range is not available or invalid, use actual data range
                if isinstance(data.index, pd.DatetimeIndex):
                    actual_start = data.index.min()
                    actual_end = data.index.max()
                    
                    # Convert pandas timestamps to QDateTime using QDate and QTime to avoid timezone issues
                    from PyQt6.QtCore import QDate, QTime, Qt
                    
                    # Extract date and time components
                    start_date = QDate(actual_start.year, actual_start.month, actual_start.day)
                    start_time = QTime(actual_start.hour, actual_start.minute, actual_start.second)
                    from_date = QDateTime(start_date, start_time, Qt.TimeSpec.LocalTime)
                    
                    end_date = QDate(actual_end.year, actual_end.month, actual_end.day)
                    end_time = QTime(actual_end.hour, actual_end.minute, actual_end.second)
                    to_date = QDateTime(end_date, end_time, Qt.TimeSpec.LocalTime)
                    
                    if from_date.isValid() and to_date.isValid():
                        self.start_date.setDateTime(from_date)
                        self.end_date.setDateTime(to_date)
                        print(f"[DEBUG] Set date range from actual data: {from_date.toString(Qt.DateFormat.ISODate)} to {to_date.toString(Qt.DateFormat.ISODate)}")
                    else:
                        print(f"[DEBUG] Failed to create valid QDateTime from actual data dates: {actual_start} to {actual_end}")
                else:
                    print(f"[DEBUG] Dataset does not have DatetimeIndex, cannot set date range automatically")

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
                if isinstance(strategies, dict):
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
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Matplotlib Figure and Canvas - Use larger size to match popout chart
        self.fig, self.ax = plt.subplots(figsize=(12, 6))  # Increased from (10, 5) to (12, 6)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, tab)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Overlay toggles (keep for future use)
        self.overlay_toggles = {}
        overlays = ['Entries/Exits', 'Pattern Signals', 'Zones', 'Indicators']
        toggle_layout = QHBoxLayout()
        for name in overlays:
            cb = QCheckBox(name)
            cb.setChecked(True)
            if name == 'Zones':
                cb.stateChanged.connect(lambda _: self._update_chart_overlays())
            else:
                cb.stateChanged.connect(self._update_chart_overlays)
            self.overlay_toggles[name] = cb
            toggle_layout.addWidget(cb)
        layout.addLayout(toggle_layout)

        # Add pop-out button
        self.popout_btn = QPushButton("Pop Out Chart")
        self.popout_btn.setToolTip("Open chart in a separate matplotlib window")
        self.popout_btn.clicked.connect(self._popout_chart)
        self.popout_btn.setEnabled(False)  # Disabled until backtest is run
        layout.addWidget(self.popout_btn)
        # Attach the button to the tab for external access
        tab.popout_btn = self.popout_btn

        tab.setLayout(layout)
        return tab

    def _popout_chart(self):
        zones_plotted = 0  # PATCH: Initialize to avoid UnboundLocalError
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        from PyQt6.QtWidgets import QDialog, QVBoxLayout
        import pandas as pd
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle('Backtest Chart (Popped Out)')
        dialog.setMinimumSize(1200, 700)
        dialog.setSizeGripEnabled(True)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.Window)
        layout = QVBoxLayout(dialog)
        # Create new figure/canvas/toolbar
        fig, ax = plt.subplots(figsize=(14, 7))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        # Draw chart
        results = getattr(self, 'current_results', None)
        if results is None or results.get('data') is None:
            ax.set_title('No data to display')
            fig.tight_layout()
            canvas.draw()
            dialog.exec()
            return
        data = results.get('data')
        if not (isinstance(data, pd.DataFrame) and all(col in data.columns for col in ['open', 'high', 'low', 'close'])):
            ax.set_title('No valid OHLC data to display')
            fig.tight_layout()
            canvas.draw()
            dialog.exec()
            return
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
        mpf.plot(df, type='candle', ax=ax, style='charles', show_nontrading=True, returnfig=True)
        
        # Auto-format x-axis based on data timeframe
        try:
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta
            
            # Detect timeframe from data
            if len(df) >= 2:
                time_diff = df.index[1] - df.index[0]
                total_duration = df.index[-1] - df.index[0]
                
                # Determine appropriate date format based on timeframe
                if time_diff <= timedelta(minutes=1):
                    # 1-minute or less data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(df)//20)))
                    print(f"[DEBUG] Popout: Set x-axis format for 1-minute data: {time_diff}")
                elif time_diff <= timedelta(minutes=5):
                    # 5-minute data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(5, len(df)//15)))
                    print(f"[DEBUG] Popout: Set x-axis format for 5-minute data: {time_diff}")
                elif time_diff <= timedelta(hours=1):
                    # Hourly data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//12)))
                    print(f"[DEBUG] Popout: Set x-axis format for hourly data: {time_diff}")
                elif time_diff <= timedelta(days=1):
                    # Daily data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
                    print(f"[DEBUG] Popout: Set x-axis format for daily data: {time_diff}")
                else:
                    # Weekly or longer
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    print(f"[DEBUG] Popout: Set x-axis format for weekly+ data: {time_diff}")
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
            print(f"[DEBUG] Popout: X-axis formatting complete. Time range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"[DEBUG] Popout: Error formatting x-axis: {e}")
            # Fallback to default formatting
            pass
        # Plot zones and micro-comb peaks if overlay is enabled
        if self.overlay_toggles.get('Zones', QCheckBox()).isChecked():
            zones = results.get('zones', [])
            print(f"[DEBUG] Popout chart: {len(zones)} zones in results['zones']")
            zones_plotted = 0
            # Defensive patch: ensure all zones have 'direction', 'zone_direction', 'type', and 'zone_type'
            for zone in zones:
                if 'direction' not in zone and 'zone_direction' in zone:
                    zone['direction'] = zone['zone_direction']
                if 'zone_direction' not in zone and 'direction' in zone:
                    zone['zone_direction'] = zone['direction']
                if 'direction' not in zone:
                    zone['direction'] = 'unknown'
                if 'zone_direction' not in zone:
                    zone['zone_direction'] = 'unknown'
                if 'zone_type' not in zone and 'type' in zone:
                    zone['zone_type'] = zone['type']
                if 'type' not in zone and 'zone_type' in zone:
                    zone['type'] = zone['zone_type']
            for i, zone in enumerate(zones):
                ts = zone.get('timestamp')
                zmin = zone.get('zone_min')
                zmax = zone.get('zone_max')
                comb_centers = zone.get('comb_centers', [])
                zone_type = zone.get('zone_type', 'Unknown')
                zone_direction = zone.get('zone_direction', 'neutral')
                # Zone decay parameters
                initial_strength = zone.get('initial_strength', 1.0)
                creation_index = zone.get('creation_index')
                gamma = zone.get('gamma', 0.95)
                tau_bars = zone.get('tau_bars', 50)
                drop_threshold = zone.get('drop_threshold', 0.01)
                if ts not in df.index:
                    print(f"[DEBUG] Skipping zone: timestamp {ts} not in df.index")
                    continue
                if zmin is None or zmax is None:
                    print(f"[DEBUG] Skipping zone: zmin or zmax is None (zmin={zmin}, zmax={zmax})")
                    continue
                idx = df.index.get_loc(ts)
                # Calculate dynamic zone duration using decay system
                if creation_index is not None:
                    creation_ts = ts
                    if creation_ts in df.index:
                        creation_idx = df.index.get_loc(creation_ts)
                        bars_since_creation = idx - creation_idx
                        if bars_since_creation >= tau_bars:
                            continue
                        current_strength = initial_strength * (gamma ** bars_since_creation)
                        if current_strength < (drop_threshold * initial_strength):
                            continue
                        end_idx = idx
                        for future_idx in range(idx + 1, min(idx + tau_bars + 1, len(df))):
                            future_bars_since_creation = future_idx - creation_idx
                            if future_bars_since_creation >= tau_bars:
                                break
                            future_strength = initial_strength * (gamma ** future_bars_since_creation)
                            if future_strength < (drop_threshold * initial_strength):
                                break
                            end_idx = future_idx
                    else:
                        end_idx = min(idx + 5, len(df) - 1)
                else:
                    end_idx = min(idx + 5, len(df) - 1)
                if creation_index is not None and ts in df.index:
                    bars_since_creation = idx - df.index.get_loc(ts)
                    current_strength = initial_strength * (gamma ** bars_since_creation)
                    alpha = max(0.2, min(0.5, current_strength))  # Higher alpha for visibility (match popout)
                else:
                    alpha = 0.3  # Higher default alpha (match popout)
                zone_color = 'blue'  # All zones are blue bands (match popout)
                # Color zones based on direction
                if zone_type == 'FVG' or zone_type == 'OrderBlock':
                    if zone_direction == 'bullish':
                        zone_color = 'green'
                    elif zone_direction == 'bearish':
                        zone_color = 'red'
                    else:
                        zone_color = 'blue'  # Fallback for neutral
                elif zone_type == 'VWAP':
                    zone_color = 'cyan'  # Cyan for VWAP zones
                # Support/Resistance zones removed
                elif zone_type == 'Imbalance':
                    zone_color = 'magenta'  # Magenta for imbalance zones
                else:
                    zone_color = 'blue'  # All other zone types remain blue
                # Print debug info for every zone
                print(f"[DEBUG] Plotting zone {i}: idx={idx}, end_idx={end_idx}, zmin={zmin}, zmax={zmax}, alpha={alpha}, type={zone_type}, direction={zone_direction}, color={zone_color}")
                # Plot with higher zorder (match popout)
                ax.fill_between(df.index[idx:end_idx+1], zmin, zmax, color=zone_color, alpha=alpha, zorder=10)
                # Plot comb centers as vertical lines within the zone
                if comb_centers:
                    for comb_price in comb_centers:
                        if zmin <= comb_price <= zmax:
                            ax.plot([df.index[idx], df.index[end_idx]], [comb_price, comb_price], 
                                   color='orange', linewidth=1, alpha=0.8, linestyle='--', zorder=11)
                            # Add small marker at the comb center
                            ax.scatter(df.index[idx], comb_price, color='orange', s=20, alpha=0.9, zorder=12)
                zones_plotted += 1
        
        # NEW ARCHITECTURE: Add indicators from strategy visualization data (SAME AS CHART TAB)
        indicators_added = False
        try:
            from core.new_gui_integration import new_gui_integration
            
            # Check if results have new architecture visualization data
            viz_data = results.get('visualization_data', {})
            if viz_data:
                print(f"[DEBUG] Popout chart: Using NEW ARCHITECTURE visualization data")
                
                # Render all line indicators (VWAP, MA, RSI, etc.)
                for line_data in viz_data.get('lines', []):
                    line_name = line_data.get('name', 'Unknown')
                    line_values = line_data.get('data')
                    line_config = line_data.get('config', {})
                    
                    if line_values is not None and not line_values.empty:
                        # Match line data index to chart data index
                        common_index = df.index.intersection(line_values.index)
                        if len(common_index) > 0:
                            chart_values = line_values.reindex(common_index)
                            
                            ax.plot(common_index, chart_values.values,
                                   color=line_config.get('color', 'purple'),
                                   linewidth=line_config.get('linewidth', 1),
                                   alpha=line_config.get('alpha', 0.8),
                                   linestyle=line_config.get('linestyle', '-'),
                                   label=line_config.get('label', line_name),
                                   zorder=line_config.get('zorder', 5))
                            
                            print(f"[DEBUG] Popout chart: Added {line_name} indicator from new architecture")
                            indicators_added = True
                        else:
                            print(f"[DEBUG] Popout chart: No common index for {line_name}")
                
                # Render zones from new architecture
                for zone_data in viz_data.get('zones', []):
                    start_idx = zone_data.get('start_idx', 0)
                    end_idx = zone_data.get('end_idx', len(df) - 1)
                    min_price = zone_data.get('min_price')
                    max_price = zone_data.get('max_price')
                    color = zone_data.get('color', 'blue')
                    alpha = zone_data.get('alpha', 0.3)
                    
                    if min_price is not None and max_price is not None and start_idx < len(df) and end_idx < len(df):
                        x_range = df.index[start_idx:end_idx+1]
                        ax.fill_between(x_range, min_price, max_price, 
                                       color=color, alpha=alpha, zorder=10)
                        print(f"[DEBUG] Popout chart: Added zone from new architecture")
                        
        except Exception as e:
            print(f"[DEBUG] Popout chart: Failed to load new architecture data: {e}")
        
        # FALLBACK: Add VWAP indicator if no new architecture data
        if not indicators_added and 'volume' in df.columns and len(df) > 20:
            try:
                print(f"[DEBUG] Popout chart: Using FALLBACK VWAP calculation")
                # Calculate VWAP
                vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                ax.plot(df.index, vwap, color='purple', linewidth=1, alpha=0.8, linestyle='-', label='VWAP', zorder=5)
                print(f"[DEBUG] Popout chart: Added fallback VWAP indicator")
            except Exception as e:
                print(f"[DEBUG] Popout chart: Failed to add VWAP indicator: {e}")
        
        # Overlay entries/exits
        if self.overlay_toggles.get('Entries/Exits', QCheckBox()).isChecked() and 'trades' in results:
            trades = results['trades']
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
                            ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                            entry_plotted = True
                    except Exception as e:
                        pass
                elif entry_idx is not None and entry_price is not None and 0 <= entry_idx < len(df):
                    entry_dt = df.index[entry_idx]
                    ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                    entry_plotted = True
                exit_dt = None
                if exit_time is not None:
                    try:
                        exit_dt = pd.to_datetime(exit_time)
                        if exit_dt in df.index:
                            ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                            exit_plotted = True
                    except Exception as e:
                        pass
                elif exit_idx is not None and exit_price is not None and 0 <= exit_idx < len(df):
                    exit_dt = df.index[exit_idx]
                    ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                    exit_plotted = True
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        ax.set_title('Backtest Chart (Popped Out)')
        fig.tight_layout()
        canvas.draw()
        dialog.exec()

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
        """Start backtest with multi-timeframe support and warning if fallback is used"""
        # Validate inputs
        if self.strategy_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select a strategy")
            return
        if self.dataset_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "Warning", "Please select a dataset")
            return
        
        # Get selected strategy and dataset
        strategy = self._get_selected_strategy()
        dataset_name = self.dataset_combo.currentText()
        
        if not strategy or not dataset_name:
            QMessageBox.warning(self, "Warning", "Could not load strategy or dataset.")
            return
        
        # Get required timeframes from strategy
        required_timeframes = self._get_strategy_timeframes(strategy)
        available_timeframes = []
        base_dataset_name = dataset_name
        mtf_info = None
        # Try to load multi-timeframe config for the selected dataset
        if self.parent_window and hasattr(self.parent_window, 'workspace_manager'):
            workspace_manager = self.parent_window.workspace_manager
            mtf_info = workspace_manager.load_multi_timeframe_dataset(dataset_name)
            if not mtf_info:
                # Try to infer base name (strip _1m, _5m, _15m, etc)
                import re
                m = re.match(r"(.+)_([0-9]+[smhd])$", dataset_name)
                if m:
                    base_dataset_name = m.group(1)
                    mtf_info = workspace_manager.load_multi_timeframe_dataset(base_dataset_name)
            if mtf_info and 'datasets' in mtf_info:
                available_timeframes = list(mtf_info['datasets'].keys())
        # If still no available timeframes, treat loaded data as single timeframe
        if not available_timeframes:
            # Try to infer timeframe from dataset name (e.g., NQ_5s_1m -> 1m)
            import re
            m = re.match(r".+_([0-9]+[smhd])$", dataset_name)
            if m:
                loaded_tf = m.group(1)
                available_timeframes = [loaded_tf]
            else:
                available_timeframes = []
        # Only warn if strategy needs a finer (higher resolution) timeframe than what is loaded
        def tf_to_seconds(tf):
            import re
            m = re.match(r"(\d+)([smhd])", tf)
            if not m:
                return float('inf')
            value, unit = int(m.group(1)), m.group(2)
            if unit == 's': return value
            if unit == 'm': return value * 60
            if unit == 'h': return value * 3600
            if unit == 'd': return value * 86400
            return float('inf')
        if available_timeframes:
            # Find the finest (lowest seconds) available timeframe
            finest_tf = min(available_timeframes, key=tf_to_seconds)
            finest_sec = tf_to_seconds(finest_tf)
            missing_timeframes = [tf for tf in required_timeframes if tf_to_seconds(tf) < finest_sec]
        else:
            missing_timeframes = required_timeframes
        if missing_timeframes:
            QMessageBox.warning(self, "Timeframe Warning",
                f"The loaded dataset does not contain the required timeframes for this strategy: {', '.join(missing_timeframes)}.\n"
                f"The system will attempt to create them from the original data if possible. If not, the strategy will be forced to use the available timeframe(s): {', '.join(available_timeframes) if available_timeframes else 'none'}.\n\n"
                f"Chart will display the loaded dataset's timeframe, but the strategy will use its assigned timeframes in the background.")
        # Actually start the backtest (this was missing)
        data = self._get_dataset_data(dataset_name)
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "Warning", "Selected dataset has no data.")
            return
        # Filter data by selected date range
        start_dt = self.start_date.date().toPyDate()
        end_dt = self.end_date.date().toPyDate()
        # If data index is datetime, filter by date
        if hasattr(data, 'index') and hasattr(data.index, 'to_pydatetime'):
            mask = (data.index.date >= start_dt) & (data.index.date <= end_dt)
            data = data.loc[mask]
        self._run_backtest_worker(data)

    def _get_selected_strategy(self):
        """Get the currently selected strategy"""
        if not self.parent_window or not hasattr(self.parent_window, 'strategies'):
            return None
        
        selected_text = self.strategy_combo.currentText()
        for strategy_type, strategies in self.parent_window.strategies.items():
            for strategy_id, strategy in strategies.items():
                if f"[{strategy_type}] {strategy.name}" == selected_text:
                    return strategy
        return None
    
    def _get_strategy_timeframes(self, strategy):
        """Extract required timeframes from a strategy"""
        timeframes = set()
        unit_map = {'minute': 'm', 'minutes': 'm', 'hour': 'h', 'hours': 'h', 'day': 'd', 'days': 'd', 'second': 's', 'seconds': 's'}
        
        if hasattr(strategy, 'actions'):
            for action in strategy.actions:
                # PREFER action.time_range over pattern.timeframes for display
                if action.time_range:
                    # Handle both TimeRange objects and dictionaries
                    if hasattr(action.time_range, 'value') and hasattr(action.time_range, 'unit'):
                        value = action.time_range.value
                        unit = action.time_range.unit
                    elif isinstance(action.time_range, dict):
                        value = action.time_range.get('value')
                        unit = action.time_range.get('unit')
                    else:
                        value = None
                        unit = None
                    if value is not None and unit is not None:
                        abbr_unit = unit_map.get(str(unit).lower(), str(unit)[0])
                        timeframes.add(f"{value}{abbr_unit}")
                # Fallback to pattern timeframes if no action time_range
                elif action.pattern and hasattr(action.pattern, 'timeframes') and action.pattern.timeframes:
                    tf = action.pattern.timeframes[0]
                    if hasattr(tf, 'value') and hasattr(tf, 'unit'):
                        value = tf.value
                        unit = tf.unit
                        abbr_unit = unit_map.get(str(unit).lower(), str(unit)[0])
                        timeframes.add(f"{value}{abbr_unit}")
        
        # If no timeframes specified, assume single timeframe
        if not timeframes:
            timeframes.add("1m")  # Default to 1 minute
        
        return sorted(list(timeframes))
    
    def _get_dataset_data(self, dataset_name):
        """Get dataset data from parent window"""
        if not self.parent_window or not hasattr(self.parent_window, 'datasets'):
            return None
        
        datasets = self.parent_window.datasets
        if dataset_name in datasets:
            dataset_obj = datasets[dataset_name]
            if isinstance(dataset_obj, dict) and 'data' in dataset_obj:
                return dataset_obj['data']
            elif isinstance(dataset_obj, pd.DataFrame):
                return dataset_obj
        return None

    def _stop_backtest(self):
        """Stop running backtest"""
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker.quit()
            self.current_worker.wait()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
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
        # Get strategy name for naming
        strategy_name = None
        if self.strategy_combo.currentIndex() > 0:
            strategy_name = self.strategy_combo.currentText().split('] ', 1)[-1]
        else:
            strategy_name = results.get('strategy_name', 'UnknownStrategy')
        
        # Add naming info to results
        results['strategy_name'] = strategy_name
        
        # Get strategy's timeframe instead of dataset timeframe
        strategy = self._get_selected_strategy()
        strategy_timeframes = self._get_strategy_timeframes(strategy) if strategy else []
        
        # Use the first strategy timeframe, or fallback to dataset timeframe
        if strategy_timeframes:
            strategy_timeframe = strategy_timeframes[0]
        else:
            # Fallback to dataset timeframe
            dataset_name = self.dataset_combo.currentText()
            import re
            m = re.match(r".+_([0-9]+[smhd])$", dataset_name)
            if m:
                strategy_timeframe = m.group(1)
            else:
                strategy_timeframe = 'unknown'
        
        results['timeframe'] = strategy_timeframe
        results['interval'] = strategy_timeframe
        
        # Compose a display/result name
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results['result_display_name'] = f"{strategy_name}_{strategy_timeframe}_{timestamp}"
        self.current_results = results

        # --- PATCH: Print debug info about results ---
        print(f"[DEBUG] Backtest results: keys={list(results.keys())}")
        print(f"[DEBUG] Trades: {len(results.get('trades', []))}, Equity curve: {len(results.get('equity_curve', []))}")
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        # Enable View Results button after successful backtest
        self.view_results_btn.setEnabled(True)
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
        # Enable pop-out button if results have data
        if hasattr(self, 'chart_tab') and hasattr(self.chart_tab, 'popout_btn'):
            if results.get('data') is not None:
                self.chart_tab.popout_btn.setEnabled(True)
            else:
                self.chart_tab.popout_btn.setEnabled(False)
        # Emit signal (pass results with naming info)
        self.backtest_complete.emit(results)
        # --- PATCH: Improved user feedback for no trades ---
        if len(results.get('trades', [])) == 0:
            QMessageBox.information(self, "Backtest Complete", "Backtest completed, but no trades were generated.\nCheck your strategy logic, pattern sensitivity, and dataset.")
        self._add_log(f"Backtest complete: {results.get('total_trades', 0)} trades, "
                     f"{results.get('total_return', 0):.2%} return")

    def _on_error(self, error: str):
        """Handle backtest error"""
        QMessageBox.critical(self, "Backtest Error", error)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
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
            
        # Disable View Results button when clearing results
        self.view_results_btn.setEnabled(False)

    def _update_overview(self, results: Dict[str, Any]):
        """Update overview tab with multi-timeframe information"""
        # Get multi-timeframe information
        multi_tf_data = results.get('multi_tf_data', {})
        strategy_timeframes = []
        
        # Handle both old dict format and new DataFrame format
        if isinstance(multi_tf_data, dict) and multi_tf_data:
            for tf_key in multi_tf_data.keys():
                if tf_key != 'execution':
                    strategy_timeframes.append(tf_key)
        elif isinstance(multi_tf_data, pd.DataFrame) and not multi_tf_data.empty:
            # New architecture returns DataFrame, treat as single timeframe
            strategy_timeframes.append('1min')
        # Summary text
        summary = f"""
Backtest Results

Strategy: {results.get('strategy_name', 'Unknown')}
"""
        if strategy_timeframes:
            summary += f"Strategy Timeframes: {', '.join(strategy_timeframes)}\n"
            
            # Handle execution data differently for old vs new architecture
            if isinstance(multi_tf_data, dict):
                execution_data = multi_tf_data.get('execution')
                if execution_data is not None:
                    summary += f"Execution Timeframe: {len(execution_data)} bars\n"
            elif isinstance(multi_tf_data, pd.DataFrame):
                summary += f"Execution Timeframe: {len(multi_tf_data)} bars\n"
        summary += f"""
Initial Capital: ${results.get('initial_capital', 0):,.2f}
Final Capital: ${results.get('final_capital', 0):,.2f}
Total Return: {results.get('total_return', 0):.2%}
Total Trades: {results.get('total_trades', 0)}
"""
        self.summary_text.setPlainText(summary)
        # Update metric labels
        metrics = {
            'total_return': f"{results.get('total_return', 0):.2%}",
            'sharpe_ratio': f"{results.get('sharpe_ratio', 0):.2f}",
            'max_drawdown': f"{results.get('max_drawdown', 0):.2%}",
            'win_rate': f"{results.get('win_rate', 0):.2%}",
            'total_trades': f"{results.get('total_trades', 0)}",
            'profit_factor': f"{results.get('profit_factor', 0):.2f}"
        }
        for key, value in metrics.items():
            if key in self.metric_labels:
                self.metric_labels[key].setText(value)

    def _update_equity_curve(self, results: Dict[str, Any]):
        """Update equity curve chart"""
        self.equity_chart.clear()

        equity_curve = results.get('equity_curve', [])
        
        # Handle both old list format and new Series format
        if isinstance(equity_curve, pd.Series):
            if equity_curve.empty:
                return
        elif isinstance(equity_curve, list):
            if not equity_curve:
                return
        else:
            return  # Unknown format

        # Plot equity curve
        if isinstance(equity_curve, pd.Series):
            # Convert Series to numpy arrays for plotting
            if isinstance(equity_curve.index, pd.DatetimeIndex):
                # Use time-based x-axis for datetime index
                x = np.arange(len(equity_curve))
                y = equity_curve.values
            else:
                # Use index values as x
                x = np.arange(len(equity_curve))
                y = equity_curve.values
            
            # Plot
            self.equity_chart.plot(x, y, pen='w', name='Equity')
            
            # Add initial capital line
            self.equity_chart.plot([0, len(equity_curve)-1],
                                 [self.initial_capital.value(), self.initial_capital.value()],
                                 pen=pg.mkPen('y', style=Qt.PenStyle.DashLine),
                                 name='Initial Capital')
            
            # Calculate and display statistics
            stats_text = f"Final Equity: ${equity_curve.iloc[-1]:,.2f} | "
            stats_text += f"Peak: ${equity_curve.max():,.2f} | "
            stats_text += f"Max Drawdown: {results['max_drawdown']:.2%}"
            
        else:
            # Handle old list format
            x = np.arange(len(equity_curve))
            self.equity_chart.plot(x, equity_curve, pen='w', name='Equity')

            # Add initial capital line
            self.equity_chart.plot([0, len(equity_curve)-1],
                                 [self.initial_capital.value(), self.initial_capital.value()],
                                 pen=pg.mkPen('y', style=Qt.PenStyle.DashLine),
                                 name='Initial Capital')

            # Calculate and display statistics
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
        
        # Handle both old list format and new Series format
        equity_has_data = False
        if isinstance(equity_curve, pd.Series):
            equity_has_data = not equity_curve.empty
        elif isinstance(equity_curve, list):
            equity_has_data = bool(equity_curve)
        
        if equity_has_data:
            # Handle both Series and list input
            if isinstance(equity_curve, pd.Series):
                equity_series = equity_curve
            else:
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

        # Defensive check for proper datetime index
        if len(monthly_returns) > 0 and hasattr(monthly_returns.index, 'strftime'):
            months = list(monthly_returns.index.strftime('%b %Y'))
        else:
            # Fallback for non-datetime index or empty series
            months = [f"Month {i+1}" for i in range(len(monthly_returns))]
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
        """Update the chart tab to always use the original dataset's timeframe for candlesticks"""
        # Always use the original dataset's data for plotting
        data = results.get('dataset_data') if results else None
        if data is not None:
            self._plot_candlesticks(data)
            self._add_log("Chart display updated using original dataset timeframe")
        else:
            # Fallback to dataset data if no results data available
            dataset_name = self.dataset_combo.currentText()
            data = None
            if self.parent_window and hasattr(self.parent_window, 'datasets'):
                datasets = self.parent_window.datasets
                if dataset_name in datasets:
                    dataset_obj = datasets[dataset_name]
                    if isinstance(dataset_obj, dict) and 'data' in dataset_obj:
                        data = dataset_obj['data']
                    elif isinstance(dataset_obj, pd.DataFrame):
                        data = dataset_obj
            if data is not None:
                self._plot_candlesticks(data)
                self._add_log(f"Chart display updated using dataset data (fallback): {dataset_name}")
            else:
                self._add_log("No data available for chart display")

    def _plot_candlesticks(self, data):
        """Plot candlestick chart on the chart tab using the exact same logic as popout chart"""
        import pandas as pd
        import mplfinance as mpf
        import matplotlib.pyplot as plt
        from datetime import timedelta
        from PyQt6.QtWidgets import QCheckBox
        
        # Clear the figure
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        
        # Draw chart - use exact same logic as popout
        results = getattr(self, 'current_results', None)
        if results is None or results.get('data') is None:
            ax.set_title('No data to display')
            self.fig.tight_layout()
            self.canvas.draw()
            return
        
        # Use the data passed in (which should be the dataset data)
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df.index = pd.to_datetime(df['datetime'])
            elif 'Date' in df.columns:
                df.index = pd.to_datetime(df['Date'])
            else:
                df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
        # --- DEBUG: Print detected bar interval ---
        if len(df) >= 2:
            bar_interval = df.index[1] - df.index[0]
            print(f"[DEBUG] Chart tab: Detected bar interval (dataset timeframe): {bar_interval}")
        # --- END DEBUG ---
        if not (isinstance(df, pd.DataFrame) and all(col in df.columns for col in ['open', 'high', 'low', 'close'])):
            ax.set_title('No valid OHLC data to display')
            self.fig.tight_layout()
            self.canvas.draw()
            return
        
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
        mpf.plot(df, type='candle', ax=ax, style='charles', show_nontrading=True, returnfig=True)
        
        # Auto-format x-axis based on data timeframe (exact same as popout)
        try:
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta
            
            # Detect timeframe from data
            if len(df) >= 2:
                time_diff = df.index[1] - df.index[0]
                total_duration = df.index[-1] - df.index[0]
                
                # Determine appropriate date format based on timeframe
                if time_diff <= timedelta(minutes=1):
                    # 1-minute or less data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(df)//20)))
                    print(f"[DEBUG] Chart tab: Set x-axis format for 1-minute data: {time_diff}")
                elif time_diff <= timedelta(minutes=5):
                    # 5-minute data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(5, len(df)//15)))
                    print(f"[DEBUG] Chart tab: Set x-axis format for 5-minute data: {time_diff}")
                elif time_diff <= timedelta(hours=1):
                    # Hourly data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//12)))
                    print(f"[DEBUG] Chart tab: Set x-axis format for hourly data: {time_diff}")
                elif time_diff <= timedelta(days=1):
                    # Daily data
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
                    print(f"[DEBUG] Chart tab: Set x-axis format for daily data: {time_diff}")
                else:
                    # Weekly or longer
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    print(f"[DEBUG] Chart tab: Set x-axis format for weekly+ data: {time_diff}")
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
            print(f"[DEBUG] Chart tab: X-axis formatting complete. Time range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"[DEBUG] Chart tab: Error formatting x-axis: {e}")
            # Fallback to default formatting
            pass
        
        # Plot zones and micro-comb peaks if overlay is enabled (exact same as popout)
        if self.overlay_toggles.get('Zones', QCheckBox()).isChecked():
            zones = results.get('zones', [])
            print(f"[DEBUG] Chart tab: {len(zones)} zones in results['zones']")
            zones_plotted = 0
            # Defensive patch: ensure all zones have 'direction', 'zone_direction', 'type', and 'zone_type'
            for zone in zones:
                if 'direction' not in zone and 'zone_direction' in zone:
                    zone['direction'] = zone['zone_direction']
                if 'zone_direction' not in zone and 'direction' in zone:
                    zone['zone_direction'] = zone['direction']
                if 'direction' not in zone:
                    zone['direction'] = 'unknown'
                if 'zone_direction' not in zone:
                    zone['zone_direction'] = 'unknown'
                if 'zone_type' not in zone and 'type' in zone:
                    zone['zone_type'] = zone['type']
                if 'type' not in zone and 'zone_type' in zone:
                    zone['type'] = zone['zone_type']
            for i, zone in enumerate(zones):
                ts = zone.get('timestamp')
                zmin = zone.get('zone_min')
                zmax = zone.get('zone_max')
                comb_centers = zone.get('comb_centers', [])
                zone_type = zone.get('zone_type', 'Unknown')
                zone_direction = zone.get('zone_direction', 'neutral')
                # --- PATCH: Always plot all zones, regardless of expiry/decay ---
                # Remove/skip any logic that would skip plotting expired zones
                if ts not in df.index:
                    print(f"[DEBUG] Chart tab: Skipping zone: timestamp {ts} not in df.index")
                    continue
                if zmin is None or zmax is None:
                    print(f"[DEBUG] Chart tab: Skipping zone: zmin or zmax is None (zmin={zmin}, zmax={zmax})")
                    continue
                idx = df.index.get_loc(ts)
                # PATCH: Plot zone for full tau_bars duration, even if expired
                creation_index = zone.get('creation_index')
                tau_bars = zone.get('tau_bars', 50)
                if creation_index is not None and ts in df.index:
                    creation_idx = df.index.get_loc(ts)
                    end_idx = min(creation_idx + tau_bars, len(df) - 1)
                else:
                    end_idx = min(idx + 5, len(df) - 1)
                alpha = 0.3  # Use a fixed alpha for all zones
                zone_color = 'blue'
                if zone_type == 'FVG' or zone_type == 'OrderBlock':
                    if zone_direction == 'bullish':
                        zone_color = 'green'
                    elif zone_direction == 'bearish':
                        zone_color = 'red'
                    else:
                        zone_color = 'blue'
                elif zone_type == 'VWAP':
                    zone_color = 'cyan'
                # Support/Resistance zones removed
                elif zone_type == 'Imbalance':
                    zone_color = 'magenta'
                else:
                    zone_color = 'blue'
                print(f"[DEBUG] Chart tab: Plotting zone {i}: idx={idx}, end_idx={end_idx}, zmin={zmin}, zmax={zmax}, alpha={alpha}, type={zone_type}, direction={zone_direction}, color={zone_color}")
                ax.fill_between(df.index[idx:end_idx+1], zmin, zmax, color=zone_color, alpha=alpha, zorder=10)
                if comb_centers:
                    for comb_price in comb_centers:
                        if zmin <= comb_price <= zmax:
                            ax.plot([df.index[idx], df.index[end_idx]], [comb_price, comb_price], 
                                   color='orange', linewidth=1, alpha=0.8, linestyle='--', zorder=11)
                            ax.scatter(df.index[idx], comb_price, color='orange', s=20, alpha=0.9, zorder=12)
                zones_plotted += 1
        
        # NEW ARCHITECTURE: Add indicators from strategy visualization data
        try:
            from core.new_gui_integration import new_gui_integration
            
            # Check if results have new architecture visualization data
            viz_data = results.get('visualization_data', {})
            if viz_data:
                print(f"[DEBUG] Chart tab: Using NEW ARCHITECTURE visualization data")
                
                # Render all line indicators (VWAP, MA, RSI, etc.)
                for line_data in viz_data.get('lines', []):
                    line_name = line_data.get('name', 'Unknown')
                    line_values = line_data.get('data')
                    line_config = line_data.get('config', {})
                    
                    if line_values is not None and not line_values.empty:
                        # Match line data index to chart data index
                        common_index = df.index.intersection(line_values.index)
                        if len(common_index) > 0:
                            chart_values = line_values.reindex(common_index)
                            
                            ax.plot(common_index, chart_values.values,
                                   color=line_config.get('color', 'purple'),
                                   linewidth=line_config.get('linewidth', 1),
                                   alpha=line_config.get('alpha', 0.8),
                                   linestyle=line_config.get('linestyle', '-'),
                                   label=line_config.get('label', line_name),
                                   zorder=line_config.get('zorder', 5))
                            
                            print(f"[DEBUG] Chart tab: Added {line_name} indicator from new architecture")
                        else:
                            print(f"[DEBUG] Chart tab: No common index for {line_name} indicator")
                
                # Render zones from new architecture
                for zone_data in viz_data.get('zones', []):
                    start_idx = zone_data.get('start_idx', 0)
                    end_idx = zone_data.get('end_idx', len(df) - 1)
                    min_price = zone_data.get('min_price')
                    max_price = zone_data.get('max_price')
                    color = zone_data.get('color', 'blue')
                    alpha = zone_data.get('alpha', 0.3)
                    
                    if min_price is not None and max_price is not None and start_idx < len(df) and end_idx < len(df):
                        x_range = df.index[start_idx:end_idx+1]
                        ax.fill_between(x_range, min_price, max_price, 
                                       color=color, alpha=alpha, zorder=10)
                        print(f"[DEBUG] Chart tab: Added zone from new architecture")
                
                # Add legend for new architecture indicators
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
                    
            else:
                print(f"[DEBUG] Chart tab: No new architecture data, using FALLBACK VWAP")
                # FALLBACK: Use old hardcoded VWAP calculation if no new architecture data
                if 'volume' in df.columns and len(df) > 20:
                    # Calculate VWAP
                    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                    ax.plot(df.index, vwap, color='purple', linewidth=1, alpha=0.8, linestyle='-', label='VWAP', zorder=5)
                    print(f"[DEBUG] Chart tab: Added fallback VWAP indicator")
                    
        except Exception as e:
            print(f"[DEBUG] Chart tab: Error with new architecture indicators: {e}")
            print(f"[DEBUG] Chart tab: Using FALLBACK VWAP")
            # FALLBACK: Use old hardcoded VWAP calculation
            if 'volume' in df.columns and len(df) > 20:
                try:
                    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
                    ax.plot(df.index, vwap, color='purple', linewidth=1, alpha=0.8, linestyle='-', label='VWAP', zorder=5)
                    print(f"[DEBUG] Chart tab: Added fallback VWAP indicator")
                except Exception as e2:
                    print(f"[DEBUG] Chart tab: Failed to add fallback VWAP: {e2}")
        
        print(f"[DEBUG] Chart tab: Plotted {zones_plotted} zones")
        
        # Overlay entries/exits (exact same as popout)
        if self.overlay_toggles.get('Entries/Exits', QCheckBox()).isChecked() and 'trades' in results:
            trades = results['trades']
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
                            ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                            entry_plotted = True
                    except Exception as e:
                        pass
                elif entry_idx is not None and entry_price is not None and 0 <= entry_idx < len(df):
                    entry_dt = df.index[entry_idx]
                    ax.scatter(entry_dt, entry_price, marker='^', color='green', s=120, label='Entry' if not entry_plotted else None, zorder=10)
                    entry_plotted = True
                exit_dt = None
                if exit_time is not None:
                    try:
                        exit_dt = pd.to_datetime(exit_time)
                        if exit_dt in df.index:
                            ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                            exit_plotted = True
                    except Exception as e:
                        pass
                elif exit_idx is not None and exit_price is not None and 0 <= exit_idx < len(df):
                    exit_dt = df.index[exit_idx]
                    ax.scatter(exit_dt, exit_price, marker='x', color='red', s=120, label='Exit' if not exit_plotted else None, zorder=10)
                    exit_plotted = True
            
            # Add legend if any entries/exits were plotted (exact same as popout)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        
        # Set title and finalize (exact same as popout)
        ax.set_title('Backtest Chart')
        self.fig.tight_layout()
        self.canvas.draw()
        print(f"[DEBUG] Chart tab: Chart updated with exact popout logic")

    def _update_chart_overlays(self):
        if self.current_results:
            self._update_chart_tab(self.current_results)

    def _run_backtest_worker(self, data):
        """Run the backtest with the processed data"""
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
        
        if not strategy:
            QMessageBox.warning(self, "Warning", "Please select a strategy")
            return
        
        # Get backtest parameters
        initial_capital = float(self.initial_capital.value())
        risk_per_trade = float(self.position_size.value())
        
        # Create worker
        self.current_worker = BacktestWorker(data, strategy, {
            'initial_capital': initial_capital,
            'position_size': risk_per_trade  # BacktestWorker expects 'position_size'
        })
        
        # Connect signals
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.log.connect(self._add_log)
        self.current_worker.trade_signal.connect(self._on_trade_signal)
        self.current_worker.finished.connect(self._on_backtest_complete)
        self.current_worker.error.connect(self._on_error)
        
        # Start backtest
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.current_worker.start()

    def _open_results_viewer(self):
        """Open results viewer with current backtest result pre-selected"""
        if not hasattr(self, 'current_results') or self.current_results is None:
            QMessageBox.warning(self, "No Results", "No backtest results available to view.")
            return
            
        # Open results viewer from parent window
        if self.parent_window:
            from gui.results_viewer_window import ResultsViewerWindow
            results_viewer = ResultsViewerWindow(self.parent_window)
            
            # Add the current result to the results viewer
            strategy_name = self.current_results.get('strategy_name', 'Unknown')
            results_viewer.add_result(self.current_results, strategy_name)
            
            # Set the combo box to select the current result
            result_display_name = self.current_results.get('result_display_name')
            if result_display_name:
                # Find the index of the current result in the combo box
                for i in range(results_viewer.results_combo.count()):
                    if results_viewer.results_combo.itemText(i) == result_display_name:
                        results_viewer.results_combo.setCurrentIndex(i)
                        break
                
                # Load the selected result
                results_viewer._on_load_selected_result()
            
            # Show the results viewer
            results_viewer.show()
            self.parent_window.open_windows.append(results_viewer)
            
            self._add_log(f"Opened results viewer for: {result_display_name}")
        else:
            QMessageBox.warning(self, "Error", "Cannot open results viewer: no parent window available.")

    def _process_data_for_backtest(self, data):
        """Process data for backtest with proper datetime handling and filtering"""
        try:
            # --- 1. Convert to DatetimeIndex if needed ---
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns and 'Time' in data.columns:
                    data = data.copy()
                    data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
                    data.set_index('datetime', inplace=True)
                    self._add_log("[PATCH] Set index to combined 'Date' and 'Time' columns.")
                else:
                    for col in ['datetime', 'date', 'Date', 'timestamp', 'Timestamp']:
                        if col in data.columns:
                            data = data.copy()
                            data[col] = pd.to_datetime(data[col])
                            data.set_index(col, inplace=True)
                            self._add_log(f"Converted '{col}' to DatetimeIndex.")
                            break
            
            self._add_log(f"[DEBUG] Index type: {type(data.index)}, unique: {data.index.is_unique}, sample: {list(data.index[:5])}")
            
            # --- PATCH: Check for required columns ---
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                QMessageBox.critical(self, "Error", f"Dataset is missing required columns: {missing_cols}\nColumns found: {list(data.columns)}")
                return None
            
            # --- 2. Filter to selected date range ---
            if isinstance(data.index, pd.DatetimeIndex):
                actual_start = data.index.min()
                actual_end = data.index.max()
                self._add_log(f"Dataset date range: {actual_start} to {actual_end}")
                
                # PATCH: Use both date and time from QDateEdit if available
                start_dt = pd.to_datetime(self.start_date.date().toString(Qt.DateFormat.ISODate))
                end_dt = pd.to_datetime(self.end_date.date().toString(Qt.DateFormat.ISODate)) + pd.Timedelta(days=1)
                
                # Check if the selected range overlaps with available data
                if start_dt > actual_end or end_dt < actual_start:
                    self._add_log(f"⚠️ WARNING: Selected date range ({start_dt} to {end_dt}) is outside dataset range!")
                    self._add_log(f"Using full dataset range instead: {actual_start} to {actual_end}")
                    start_dt = actual_start
                    end_dt = actual_end + pd.Timedelta(days=1)
                
                before_filter = len(data)
                filtered_data = data[(data.index >= start_dt) & (data.index < end_dt)]
                self._add_log(f"[DEBUG] Filtered to date range: {start_dt} to {end_dt} ({before_filter} -> {len(filtered_data)} bars)")
                
                # PATCH: Show popup if filter is empty
                if len(filtered_data) == 0:
                    QMessageBox.warning(self, "No Data", f"No data available for the selected date range: {start_dt} to {end_dt}\nAvailable range: {actual_start} to {actual_end}")
                    return None
                
                data = filtered_data
            else:
                self._add_log("WARNING: Data does not have a DatetimeIndex after conversion. Cannot filter by date range.")
            
            # --- 3. Multi-timeframe engine will handle resampling internally ---
            self._add_log("✅ Using Multi-Timeframe Backtest Engine - strategy timeframes will be preserved")
            self._add_log(f"Original data: {len(data)} bars, timeframe: {data.index.freq if hasattr(data.index, 'freq') else 'Unknown'}")
            
            # --- 4. Drop rows with missing OHLCV data ---
            data = data.dropna()
            self._add_log(f"Final data for backtest: {len(data)} bars, columns: {list(data.columns)}")
            
            if len(data) == 0:
                QMessageBox.warning(self, "Warning", "No data available for the selected date range after dropping missing values.")
                return None
            
            # --- 5. Cache the processed data ---
            cache_key = f"{self.dataset_combo.currentText()}_{start_dt}_{end_dt}"
            if cache_key in self._data_cache:
                data = self._data_cache[cache_key]
                self._add_log("✅ Using cached data")
            else:
                self._add_log("🔄 Processing and caching data")
                self._data_cache[cache_key] = data
                if len(self._data_cache) > 5:
                    oldest_key = next(iter(self._data_cache))
                    del self._data_cache[oldest_key]
                    self._add_log(f"🗑️ Cleared old cache entry: {oldest_key}")
            
            return data
            
        except Exception as e:
            self._add_log(f"❌ Error processing data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to process data: {e}")
            return None

    def _run_backtest(self, strategy, dataset):
        print("[DEBUG] Strategy actions and patterns:")
        for i, action in enumerate(strategy.actions):
            print(f"  Action {i}: {repr(action)}")
            if hasattr(action, 'pattern'):
                print(f"    Pattern: {repr(getattr(action, 'pattern', None))}")
        # Run the backtest
        results = self.engine.run_backtest(strategy, dataset)
        print("[DEBUG] Raw results['zones']:")
        pprint.pprint(results.get('zones', []))
        return results