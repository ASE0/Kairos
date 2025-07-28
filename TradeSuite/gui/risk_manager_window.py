"""
gui/risk_manager_window.py
==========================
Window for creating risk management strategies
"""

import logging
logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from typing import Dict, List, Optional

from strategies.strategy_builders import RiskStrategy
from patterns.candlestick_patterns import CandlestickPattern


class RiskManagerWindow(QMainWindow):
    """Window for creating risk management strategies"""
    
    # Signals
    risk_strategy_created = pyqtSignal(object)  # RiskStrategy
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Risk Manager")
        self.setGeometry(300, 200, 800, 700)
        
        # Get available patterns from parent
        self.available_patterns = parent.patterns if parent else {}
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Strategy name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Risk Strategy Name:"))
        self.strategy_name = QLineEdit()
        self.strategy_name.setPlaceholderText("e.g., Conservative 2% Risk")
        name_layout.addWidget(self.strategy_name)
        layout.addLayout(name_layout)
        
        # Tab widget for different aspects
        self.tab_widget = QTabWidget()
        
        # Entry tab
        self.entry_tab = self._create_entry_tab()
        self.tab_widget.addTab(self.entry_tab, "Entry Rules")
        
        # Stop loss tab
        self.stop_tab = self._create_stop_tab()
        self.tab_widget.addTab(self.stop_tab, "Stop Loss")
        
        # Take profit tab
        self.profit_tab = self._create_profit_tab()
        self.tab_widget.addTab(self.profit_tab, "Take Profit")
        
        # Position sizing tab
        self.sizing_tab = self._create_sizing_tab()
        self.tab_widget.addTab(self.sizing_tab, "Position Sizing")
        
        # Exit patterns tab
        self.exit_patterns_tab = self._create_exit_patterns_tab()
        self.tab_widget.addTab(self.exit_patterns_tab, "Exit Patterns")
        
        layout.addWidget(self.tab_widget)
        
        # Summary section
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Strategy Summary:"))
        layout.addWidget(self.summary_text)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.update_btn = QPushButton("Update Summary")
        self.update_btn.clicked.connect(self._update_summary)
        button_layout.addWidget(self.update_btn)
        
        self.save_btn = QPushButton("Save Risk Strategy")
        self.save_btn.clicked.connect(self._save_strategy)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Initial update
        self._update_summary()
        
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
        
    def _create_entry_tab(self) -> QWidget:
        """Create entry rules tab"""
        widget = QWidget()
        layout = QFormLayout()
        
        # Entry method
        self.entry_method = QComboBox()
        self.entry_method.addItems(['Market', 'Limit', 'Stop', 'Stop Limit'])
        self.entry_method.currentTextChanged.connect(self._on_entry_method_changed)
        layout.addRow("Entry Method:", self.entry_method)
        
        # Limit offset
        self.limit_offset_widget = QWidget()
        limit_layout = QHBoxLayout(self.limit_offset_widget)
        self.limit_offset = QDoubleSpinBox()
        self.limit_offset.setRange(0, 0.1)
        self.limit_offset.setValue(0.001)
        self.limit_offset.setSingleStep(0.0001)
        self.limit_offset.setDecimals(4)
        limit_layout.addWidget(self.limit_offset)
        limit_layout.addWidget(QLabel("(fraction of price)"))
        limit_layout.addStretch()
        layout.addRow("Limit Offset:", self.limit_offset_widget)
        self.limit_offset_widget.setVisible(False)
        
        # Entry timing
        layout.addWidget(QLabel("<b>Entry Timing:</b>"))
        
        self.immediate_entry = QRadioButton("Immediate on signal")
        self.immediate_entry.setChecked(True)
        layout.addWidget(self.immediate_entry)
        
        self.wait_for_close = QRadioButton("Wait for bar close")
        layout.addWidget(self.wait_for_close)
        
        self.wait_for_confirmation = QRadioButton("Wait for confirmation")
        layout.addWidget(self.wait_for_confirmation)
        
        # Confirmation settings
        self.confirmation_widget = QWidget()
        conf_layout = QFormLayout(self.confirmation_widget)
        
        self.confirmation_bars = QSpinBox()
        self.confirmation_bars.setRange(1, 10)
        self.confirmation_bars.setValue(2)
        conf_layout.addRow("Confirmation Bars:", self.confirmation_bars)
        
        self.confirmation_type = QComboBox()
        self.confirmation_type.addItems(['Price holds', 'Volume increase', 'Pattern completion'])
        conf_layout.addRow("Confirmation Type:", self.confirmation_type)
        
        layout.addWidget(self.confirmation_widget)
        self.confirmation_widget.setVisible(False)
        
        # Connect confirmation visibility
        self.wait_for_confirmation.toggled.connect(self.confirmation_widget.setVisible)
        
        widget.setLayout(layout)
        return widget
        
    def _create_stop_tab(self) -> QWidget:
        """Create stop loss tab"""
        widget = QWidget()
        layout = QFormLayout()
        
        # Stop method
        self.stop_method = QComboBox()
        self.stop_method.addItems(['Fixed Percentage', 'ATR Based', 'Pattern Based', 
                                  'Trailing Stop'])
        self.stop_method.currentTextChanged.connect(self._on_stop_method_changed)
        layout.addRow("Stop Method:", self.stop_method)
        
        # Method-specific settings
        self.stop_settings_stack = QStackedWidget()
        
        # Fixed percentage
        fixed_widget = QWidget()
        fixed_layout = QFormLayout()
        self.stop_loss_pct = QDoubleSpinBox()
        self.stop_loss_pct.setMinimum(-1e9)
        self.stop_loss_pct.setMaximum(1e9)
        self.stop_loss_pct.setValue(2.0)
        self.stop_loss_pct.setSingleStep(0.1)
        self.stop_loss_pct.setSuffix("%")
        fixed_layout.addRow("Stop Loss %:", self.stop_loss_pct)
        fixed_widget.setLayout(fixed_layout)
        self.stop_settings_stack.addWidget(fixed_widget)
        
        # ATR based
        atr_widget = QWidget()
        atr_layout = QFormLayout()
        self.atr_period = QSpinBox()
        self.atr_period.setMinimum(-1000000000)
        self.atr_period.setMaximum(1000000000)
        self.atr_period.setValue(14)
        atr_layout.addRow("ATR Period:", self.atr_period)
        self.atr_multiplier = QDoubleSpinBox()
        self.atr_multiplier.setMinimum(-1e9)
        self.atr_multiplier.setMaximum(1e9)
        self.atr_multiplier.setValue(2.0)
        self.atr_multiplier.setSingleStep(0.1)
        atr_layout.addRow("ATR Multiplier:", self.atr_multiplier)
        atr_widget.setLayout(atr_layout)
        self.stop_settings_stack.addWidget(atr_widget)
        
        # Pattern based
        pattern_widget = QWidget()
        pattern_layout = QFormLayout()
        self.pattern_buffer = QDoubleSpinBox()
        self.pattern_buffer.setMinimum(-1e9)
        self.pattern_buffer.setMaximum(1e9)
        self.pattern_buffer.setValue(0.001)
        self.pattern_buffer.setSingleStep(0.0001)
        self.pattern_buffer.setDecimals(4)
        pattern_layout.addRow("Buffer below pattern:", self.pattern_buffer)
        pattern_widget.setLayout(pattern_layout)
        self.stop_settings_stack.addWidget(pattern_widget)
        
        # Trailing stop
        trail_widget = QWidget()
        trail_layout = QFormLayout()
        self.trail_activation = QDoubleSpinBox()
        self.trail_activation.setMinimum(-1e9)
        self.trail_activation.setMaximum(1e9)
        self.trail_activation.setValue(1.5)
        self.trail_activation.setSingleStep(0.1)
        trail_layout.addRow("Activation (R):", self.trail_activation)
        self.trail_distance = QDoubleSpinBox()
        self.trail_distance.setMinimum(-1e9)
        self.trail_distance.setMaximum(1e9)
        self.trail_distance.setValue(1.0)
        self.trail_distance.setSingleStep(0.1)
        self.trail_distance.setSuffix("%")
        trail_layout.addRow("Trail Distance:", self.trail_distance)
        trail_widget.setLayout(trail_layout)
        self.stop_settings_stack.addWidget(trail_widget)
        
        layout.addWidget(self.stop_settings_stack)
        
        # Additional stop options
        layout.addWidget(QLabel("<b>Additional Options:</b>"))
        
        self.time_stop_check = QCheckBox("Time-based stop")
        layout.addWidget(self.time_stop_check)
        
        self.time_stop_widget = QWidget()
        time_layout = QHBoxLayout(self.time_stop_widget)
        time_layout.addWidget(QLabel("Exit after:"))
        self.time_stop_bars = QSpinBox()
        self.time_stop_bars.setMinimum(-1000000000)
        self.time_stop_bars.setMaximum(1000000000)
        self.time_stop_bars.setValue(20)
        time_layout.addWidget(self.time_stop_bars)
        time_layout.addWidget(QLabel("bars"))
        time_layout.addStretch()
        layout.addWidget(self.time_stop_widget)
        self.time_stop_widget.setVisible(False)
        
        self.time_stop_check.toggled.connect(self.time_stop_widget.setVisible)
        
        widget.setLayout(layout)
        return widget
        
    def _create_profit_tab(self) -> QWidget:
        """Create take profit tab"""
        widget = QWidget()
        layout = QFormLayout()
        
        # Take profit method
        self.profit_method = QComboBox()
        self.profit_method.addItems(['Fixed Risk/Reward', 'ATR Multiple', 
                                    'Resistance Level', 'Trailing Profit'])
        self.profit_method.currentTextChanged.connect(self._on_profit_method_changed)
        layout.addRow("Take Profit Method:", self.profit_method)
        
        # Method-specific settings
        self.profit_settings_stack = QStackedWidget()
        
        # Fixed RR
        rr_widget = QWidget()
        rr_layout = QFormLayout()
        self.risk_reward_ratio = QDoubleSpinBox()
        self.risk_reward_ratio.setMinimum(-1e9)
        self.risk_reward_ratio.setMaximum(1e9)
        self.risk_reward_ratio.setValue(2.0)
        self.risk_reward_ratio.setSingleStep(0.1)
        rr_layout.addRow("Risk/Reward Ratio:", self.risk_reward_ratio)
        rr_widget.setLayout(rr_layout)
        self.profit_settings_stack.addWidget(rr_widget)
        
        # ATR multiple
        atr_profit_widget = QWidget()
        atr_profit_layout = QFormLayout()
        self.profit_atr_multiple = QDoubleSpinBox()
        self.profit_atr_multiple.setMinimum(-1e9)
        self.profit_atr_multiple.setMaximum(1e9)
        self.profit_atr_multiple.setValue(3.0)
        self.profit_atr_multiple.setSingleStep(0.5)
        atr_profit_layout.addRow("ATR Multiple:", self.profit_atr_multiple)
        atr_profit_widget.setLayout(atr_profit_layout)
        self.profit_settings_stack.addWidget(atr_profit_widget)
        
        # Resistance level
        resistance_widget = QWidget()
        resistance_layout = QFormLayout()
        self.resistance_lookback = QSpinBox()
        self.resistance_lookback.setMinimum(-1000000000)
        self.resistance_lookback.setMaximum(1000000000)
        self.resistance_lookback.setValue(50)
        resistance_layout.addRow("Lookback Bars:", self.resistance_lookback)
        resistance_widget.setLayout(resistance_layout)
        self.profit_settings_stack.addWidget(resistance_widget)
        
        # Trailing profit
        trail_profit_widget = QWidget()
        trail_profit_layout = QFormLayout()
        self.profit_trail_trigger = QDoubleSpinBox()
        self.profit_trail_trigger.setMinimum(-1e9)
        self.profit_trail_trigger.setMaximum(1e9)
        self.profit_trail_trigger.setValue(1.0)
        self.profit_trail_trigger.setSingleStep(0.1)
        trail_profit_layout.addRow("Trigger (R):", self.profit_trail_trigger)
        self.profit_trail_distance = QDoubleSpinBox()
        self.profit_trail_distance.setMinimum(-1e9)
        self.profit_trail_distance.setMaximum(1e9)
        self.profit_trail_distance.setValue(0.5)
        self.profit_trail_distance.setSingleStep(0.1)
        self.profit_trail_distance.setSuffix("%")
        trail_profit_layout.addRow("Trail Distance:", self.profit_trail_distance)
        trail_profit_widget.setLayout(trail_profit_layout)
        self.profit_settings_stack.addWidget(trail_profit_widget)
        
        layout.addWidget(self.profit_settings_stack)
        
        # Partial profits
        layout.addWidget(QLabel("<b>Partial Profits:</b>"))
        
        self.partial_profits_check = QCheckBox("Take partial profits")
        layout.addWidget(self.partial_profits_check)
        
        self.partial_widget = QWidget()
        partial_layout = QFormLayout(self.partial_widget)
        
        self.partial_1_pct = QSpinBox()
        self.partial_1_pct.setMinimum(-1000000000)
        self.partial_1_pct.setMaximum(1000000000)
        self.partial_1_pct.setValue(50)
        self.partial_1_pct.setSuffix("%")
        partial_layout.addRow("First partial %:", self.partial_1_pct)
        
        self.partial_1_target = QDoubleSpinBox()
        self.partial_1_target.setMinimum(-1e9)
        self.partial_1_target.setMaximum(1e9)
        self.partial_1_target.setValue(1.0)
        self.partial_1_target.setSingleStep(0.1)
        partial_layout.addRow("At R multiple:", self.partial_1_target)
        
        layout.addWidget(self.partial_widget)
        self.partial_widget.setVisible(False)
        
        self.partial_profits_check.toggled.connect(self.partial_widget.setVisible)
        
        widget.setLayout(layout)
        return widget
        
    def _create_sizing_tab(self) -> QWidget:
        """Create position sizing tab"""
        widget = QWidget()
        layout = QFormLayout()
        
        # Sizing method
        self.sizing_method = QComboBox()
        self.sizing_method.addItems(['Fixed Risk %', 'Fixed Dollar', 'Kelly Criterion', 
                                    'Volatility Adjusted', 'Equal Weight'])
        self.sizing_method.currentTextChanged.connect(self._on_sizing_method_changed)
        layout.addRow("Sizing Method:", self.sizing_method)
        
        # Method-specific settings
        self.sizing_settings_stack = QStackedWidget()
        
        # Fixed risk %
        risk_widget = QWidget()
        risk_layout = QFormLayout()
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setMinimum(-1e9)
        self.risk_per_trade.setMaximum(1e9)
        self.risk_per_trade.setValue(1.0)
        self.risk_per_trade.setSingleStep(0.1)
        self.risk_per_trade.setSuffix("%")
        risk_layout.addRow("Risk per Trade:", self.risk_per_trade)
        risk_widget.setLayout(risk_layout)
        self.sizing_settings_stack.addWidget(risk_widget)
        
        # Fixed dollar
        dollar_widget = QWidget()
        dollar_layout = QFormLayout()
        self.fixed_dollar = QSpinBox()
        self.fixed_dollar.setMinimum(-1000000000)
        self.fixed_dollar.setMaximum(1000000000)
        self.fixed_dollar.setValue(1000)
        self.fixed_dollar.setSingleStep(100)
        self.fixed_dollar.setPrefix("$")
        dollar_layout.addRow("Position Size:", self.fixed_dollar)
        dollar_widget.setLayout(dollar_layout)
        self.sizing_settings_stack.addWidget(dollar_widget)
        
        # Kelly criterion
        kelly_widget = QWidget()
        kelly_layout = QFormLayout()
        self.kelly_fraction = QDoubleSpinBox()
        self.kelly_fraction.setMinimum(-1e9)
        self.kelly_fraction.setMaximum(1e9)
        self.kelly_fraction.setValue(0.25)
        self.kelly_fraction.setSingleStep(0.05)
        kelly_layout.addRow("Kelly Fraction:", self.kelly_fraction)
        kelly_widget.setLayout(kelly_layout)
        self.sizing_settings_stack.addWidget(kelly_widget)
        
        # Volatility adjusted
        vol_widget = QWidget()
        vol_layout = QFormLayout()
        self.vol_target = QDoubleSpinBox()
        self.vol_target.setMinimum(-1e9)
        self.vol_target.setMaximum(1e9)
        self.vol_target.setValue(10.0)
        self.vol_target.setSingleStep(1.0)
        self.vol_target.setSuffix("%")
        vol_layout.addRow("Target Volatility:", self.vol_target)
        vol_widget.setLayout(vol_layout)
        self.sizing_settings_stack.addWidget(vol_widget)
        
        # Equal weight
        equal_widget = QWidget()
        equal_layout = QFormLayout()
        self.max_positions = QSpinBox()
        self.max_positions.setMinimum(-1000000000)
        self.max_positions.setMaximum(1000000000)
        self.max_positions.setValue(5)
        equal_layout.addRow("Max Positions:", self.max_positions)
        equal_widget.setLayout(equal_layout)
        self.sizing_settings_stack.addWidget(equal_widget)
        
        layout.addWidget(self.sizing_settings_stack)
        
        # Position limits
        layout.addWidget(QLabel("<b>Position Limits:</b>"))
        
        max_pos_layout = QHBoxLayout()
        max_pos_layout.addWidget(QLabel("Max position size:"))
        self.max_position_pct = QDoubleSpinBox()
        self.max_position_pct.setMinimum(-1e9)
        self.max_position_pct.setMaximum(1e9)
        self.max_position_pct.setValue(10.0)
        self.max_position_pct.setSingleStep(1.0)
        self.max_position_pct.setSuffix("% of portfolio")
        max_pos_layout.addWidget(self.max_position_pct)
        max_pos_layout.addStretch()
        layout.addRow(max_pos_layout)
        
        widget.setLayout(layout)
        return widget
        
    def _create_exit_patterns_tab(self) -> QWidget:
        """Create exit patterns tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select patterns that trigger exit:"))
        
        # Pattern list
        self.exit_pattern_list = QListWidget()
        self.exit_pattern_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        for pattern_name in self.available_patterns.keys():
            self.exit_pattern_list.addItem(pattern_name)
            
        layout.addWidget(self.exit_pattern_list)
        
        # Exit on opposite signal
        self.exit_on_opposite = QCheckBox("Exit on opposite signal")
        self.exit_on_opposite.setChecked(True)
        layout.addWidget(self.exit_on_opposite)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
        
    def _on_entry_method_changed(self, method: str):
        """Handle entry method change"""
        self.limit_offset_widget.setVisible(method in ['Limit', 'Stop Limit'])
        
    def _on_stop_method_changed(self, method: str):
        """Handle stop method change"""
        index_map = {
            'Fixed Percentage': 0,
            'ATR Based': 1,
            'Pattern Based': 2,
            'Trailing Stop': 3
        }
        self.stop_settings_stack.setCurrentIndex(index_map.get(method, 0))
        
    def _on_profit_method_changed(self, method: str):
        """Handle profit method change"""
        index_map = {
            'Fixed Risk/Reward': 0,
            'ATR Multiple': 1,
            'Resistance Level': 2,
            'Trailing Profit': 3
        }
        self.profit_settings_stack.setCurrentIndex(index_map.get(method, 0))
        
    def _on_sizing_method_changed(self, method: str):
        """Handle sizing method change"""
        index_map = {
            'Fixed Risk %': 0,
            'Fixed Dollar': 1,
            'Kelly Criterion': 2,
            'Volatility Adjusted': 3,
            'Equal Weight': 4
        }
        self.sizing_settings_stack.setCurrentIndex(index_map.get(method, 0))
        
    def _update_summary(self):
        """Update strategy summary"""
        summary = f"Risk Strategy: {self.strategy_name.text() or 'Unnamed'}\n\n"
        
        # Entry
        summary += f"Entry: {self.entry_method.currentText()}"
        if self.immediate_entry.isChecked():
            summary += " (Immediate)\n"
        elif self.wait_for_close.isChecked():
            summary += " (Wait for close)\n"
        else:
            summary += f" (Wait {self.confirmation_bars.value()} bars)\n"
            
        # Stop loss
        summary += f"Stop: {self.stop_method.currentText()}"
        if self.stop_method.currentText() == 'Fixed Percentage':
            summary += f" ({self.stop_loss_pct.value()}%)\n"
        elif self.stop_method.currentText() == 'ATR Based':
            summary += f" ({self.atr_multiplier.value()}x ATR)\n"
        else:
            summary += "\n"
            
        # Take profit
        summary += f"Profit: {self.profit_method.currentText()}"
        if self.profit_method.currentText() == 'Fixed Risk/Reward':
            summary += f" (1:{self.risk_reward_ratio.value()})\n"
        else:
            summary += "\n"
            
        # Position sizing
        summary += f"Sizing: {self.sizing_method.currentText()}"
        if self.sizing_method.currentText() == 'Fixed Risk %':
            summary += f" ({self.risk_per_trade.value()}%)\n"
        else:
            summary += "\n"
            
        # Exit patterns
        selected_patterns = [item.text() for item in self.exit_pattern_list.selectedItems()]
        if selected_patterns:
            summary += f"\nExit Patterns: {', '.join(selected_patterns)}"
            
        self.summary_text.setText(summary)
        
    def _save_strategy(self):
        """Save the risk strategy"""
        name = self.strategy_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a strategy name")
            return
            
        logger.info(f"[DEBUG] Creating risk strategy with name: '{name}'")
            
        # Get exit patterns
        exit_patterns = []
        for item in self.exit_pattern_list.selectedItems():
            pattern_name = item.text()
            if pattern_name in self.available_patterns:
                exit_patterns.append(self.available_patterns[pattern_name])
                
        # Create risk strategy
        strategy = RiskStrategy(
            name=name,
            entry_method=self.entry_method.currentText().lower().replace(' ', '_'),
            stop_method=self._get_stop_method(),
            exit_method=self._get_exit_method(),
            stop_loss_pct=self.stop_loss_pct.value() / 100,
            risk_reward_ratio=self.risk_reward_ratio.value(),
            atr_multiplier=self.atr_multiplier.value(),
            trailing_stop_pct=self.trail_distance.value() / 100,
            exit_patterns=exit_patterns
        )
        
        logger.info(f"[DEBUG] Created strategy with name: '{strategy.name}', ID: '{strategy.id}'")
        
        # Emit signal
        self.risk_strategy_created.emit(strategy)
        
        QMessageBox.information(self, "Success", f"Risk strategy '{name}' created successfully")
        self.close()
        
    def _get_stop_method(self) -> str:
        """Get stop method code"""
        method_map = {
            'Fixed Percentage': 'fixed',
            'ATR Based': 'atr',
            'Pattern Based': 'pattern',
            'Trailing Stop': 'trailing'
        }
        return method_map.get(self.stop_method.currentText(), 'fixed')
        
    def _get_exit_method(self) -> str:
        """Get exit method code"""
        method_map = {
            'Fixed Risk/Reward': 'fixed_rr',
            'ATR Multiple': 'atr_multiple',
            'Resistance Level': 'resistance',
            'Trailing Profit': 'trailing'
        }
        return method_map.get(self.profit_method.currentText(), 'fixed_rr')
