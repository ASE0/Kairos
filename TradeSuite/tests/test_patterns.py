#!/usr/bin/env python3
"""
Comprehensive Pattern Detection Test Suite
========================================
Tests all implemented patterns using synthetic datasets based on mathematical framework.
"""

import pytest
from .data_factory import create_dataset, get_expected_detection_index, create_negative_dataset
from .run_headless import run_headless_test
import os

# Comprehensive list of all patterns that are implemented in main.py
PATTERNS = [
    # Zone types (working)
    ('vwap_mean_reversion_band', {}),
    ('imbalance_memory_zone', {}),
    
    # Basic candlestick patterns (implemented in main.py)
    ('fvg', {}),
    ('engulfing', {}),
    ('hammer', {}),
    ('double_wick', {}),
    ('ii_bars', {}),
    ('doji', {}),
    ('weak_body', {}),
    ('marubozu', {}),
    ('spinning_top', {}),
]

@pytest.mark.parametrize('pattern_name,params', PATTERNS)
def test_pattern(pattern_name, params):
    data_path = create_dataset(pattern_name, params)
    expected_idx = get_expected_detection_index(pattern_name, params)
    strategy = {'name': f'Test_{pattern_name}', 'actions': [{'name': pattern_name, 'pattern': {'params': params}}]}
    gui_idx, results = run_headless_test(strategy, data_path)
    passed = (gui_idx == expected_idx)
    notes = '' if passed else f'Expected {expected_idx}, got {gui_idx}. Data: {os.path.basename(data_path)}'
    # Log to TEST_REPORT.md
    with open(os.path.join(os.path.dirname(__file__), 'TEST_REPORT.md'), 'a') as f:
        f.write(f'| {pattern_name} | {expected_idx} | {gui_idx} | {"PASS" if passed else "FAIL"} | {notes} |\n')
    assert passed, notes 

@pytest.mark.parametrize('pattern_name,params', PATTERNS)
def test_pattern_negative(pattern_name, params):
    data_path = create_negative_dataset(pattern_name, params)
    strategy = {'name': f'Test_{pattern_name}', 'actions': [{'name': pattern_name, 'pattern': {'params': params}}]}
    gui_idx, results = run_headless_test(strategy, data_path)
    # Acceptable values for no detection: None, -1, [], or empty
    no_detection = (gui_idx is None) or (gui_idx == -1) or (gui_idx == []) or (gui_idx == "")
    notes = '' if no_detection else f'Expected no detection, got {gui_idx}. Data: {os.path.basename(data_path)}'
    with open(os.path.join(os.path.dirname(__file__), 'TEST_REPORT.md'), 'a') as f:
        f.write(f'| {pattern_name} | NEGATIVE | {gui_idx} | {"PASS" if no_detection else "FAIL"} | {notes} |\n')
    assert no_detection, notes 