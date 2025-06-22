"""
Trading Strategy Hub - Central Application
==========================================
A comprehensive GUI for building, testing, and combining trading strategies
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_hub.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # High DPI scaling is enabled by default in PyQt6
    # No need to set AA_EnableHighDpiScaling

    # Set application style
    app.setStyle('Fusion')

    # Import and create main window
    from gui.main_hub import TradingStrategyHub
    hub = TradingStrategyHub()
    hub.show()

    sys.exit(app.exec())


# ==================== DATA STRUCTURES ====================

@dataclass
class CandlestickPattern:
    """Defines a candlestick pattern"""
    name: str
    type: str  # 'ii_bars', 'double_wick', 'hammer', 'doji', etc.
    conditions: Dict[str, Any]
    timeframes: List[str]
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


@dataclass
class Action:
    """Trading action with time range"""
    name: str
    pattern: CandlestickPattern
    time_range: Dict[str, int]  # {'value': 5, 'unit': 'minutes'}
    location_strategy: Optional[str] = None  # VWAP, POC, etc.
    filters: List[Dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))


if __name__ == "__main__":
    main()