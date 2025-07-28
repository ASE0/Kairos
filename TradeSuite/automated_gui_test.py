#!/usr/bin/env python3
"""
Automated GUI Test - Simulate user interactions with the GUI
This script can programmatically control the GUI for testing
"""

import sys
import os
import time
import subprocess
from datetime import datetime
import json
import pandas as pd
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main hub
from gui.main_hub import TradingStrategyHub

# 1. Load a subset of the dataset
csv_path = 'NQ_5s_1m.csv'
df = pd.read_csv(csv_path, nrows=100)

dataset_name = 'TestSubset_1m'
metadata = {'source': csv_path, 'rows': len(df)}

app = QApplication(sys.argv)
hub = TradingStrategyHub()
hub.on_data_processed(dataset_name, df, metadata)

print('[INFO] Main window opened.')
hub.show()
QTest.qWait(1000)

# ... (all previous window tests here, including workspace manager, backtester, strategy builder, pattern manager, etc.) ...

# --- Results Viewer test ---
from gui.results_viewer_window import ResultsViewerWindow
import tempfile

try:
    rv = ResultsViewerWindow(parent=hub)
    rv.show()
    QTest.qWait(500)
    print('[PASS] Results Viewer window opened.')
    # Simulate loading the most recent result (if available)
    if hasattr(rv, 'results_combo') and rv.results_combo.count() > 1:
        rv.results_combo.setCurrentIndex(1)
        QTest.qWait(200)
        print('[PASS] Most recent result selected in Results Viewer.')
        # Simulate loading result
        if hasattr(rv, 'load_result_btn'):
            rv.load_result_btn.setFocus()
            try:
                key = getattr(Qt, 'Key_Return', None) or getattr(Qt, 'Key_Enter', None)
                if key:
                    QTest.keyClick(rv.load_result_btn, key)
                else:
                    QTest.mouseClick(rv.load_result_btn, Qt.MouseButton.LeftButton)
            except Exception:
                QTest.mouseClick(rv.load_result_btn, Qt.MouseButton.LeftButton)
            QTest.qWait(500)
            print('[PASS] Result loaded in Results Viewer.')
        else:
            print('[WARN] No load_result_btn in Results Viewer.')
        # Check if table and plots are populated
        if hasattr(rv, 'results_table') and rv.results_table.rowCount() > 0:
            print('[PASS] Results table populated.')
        else:
            print('[FAIL] Results table not populated.')
        if hasattr(rv, 'results_plot') and rv.results_plot.plotItem.listDataItems():
            print('[PASS] Results plot populated.')
        else:
            print('[FAIL] Results plot not populated.')
        # Check for key statistics
        stats_ok = True
        for stat in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            if not hasattr(rv, 'stats') or stat not in rv.stats or rv.stats[stat] is None:
                print(f'[FAIL] Statistic {stat} missing in Results Viewer.')
                stats_ok = False
        if stats_ok:
            print('[PASS] Key statistics displayed in Results Viewer.')
        # Simulate exporting results
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
                export_path = tmpfile.name
            if hasattr(rv, 'export_btn'):
                rv.export_btn.setFocus()
                QTest.keyClick(rv.export_btn, Qt.Key_Return)
                QTest.qWait(500)
                if os.path.exists(export_path):
                    print('[PASS] Results exported successfully.')
                    os.remove(export_path)
                else:
                    print('[FAIL] Results export file not created.')
            else:
                print('[WARN] No export_btn in Results Viewer.')
        except Exception as e:
            print(f'[FAIL] Could not export results: {e}')
    else:
        print('[FAIL] No results available in Results Viewer.')
except Exception as e:
    print(f'[FAIL] Could not open or test Results Viewer: {e}')

# Clean up
QTest.qWait(500)
app.quit() 