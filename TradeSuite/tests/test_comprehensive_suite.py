#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Patterns, Gates, and Filters
============================================================
Tests all implemented patterns, gates, and filters using synthetic datasets.
"""

import pytest
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tests.data_factory import (
    create_dataset, create_gate_dataset, create_filter_dataset,
    get_expected_detection_index
)
from tests.run_headless import run_headless_test, create_single_pattern_strategy


# Define all patterns to test
PATTERNS_TO_TEST = [
    # Basic candlestick patterns
    'fvg', 'engulfing', 'hammer', 'double_wick', 'ii_bars',
    'doji', 'marubozu', 'spinning_top', 'weak_body', 'strong_body',
    
    # Advanced patterns
    'engulfing_bullish', 'engulfing_bearish', 'doji_standard',
    'momentum_breakout', 'momentum_reversal', 'high_volatility', 'low_volatility',
    'support_bounce', 'resistance_rejection', 'three_white_soldiers', 'three_black_crows',
    'four_price_doji', 'dragonfly_doji', 'gravestone_doji', 'volatility_expansion',
    'volatility_contraction', 'trend_continuation', 'trend_reversal', 'gap_up', 'gap_down',
    'consolidation', 'breakout', 'exhaustion', 'accumulation', 'distribution'
]

# Define all gates to test
GATES_TO_TEST = [
    'location_gate', 'volatility_gate', 'regime_gate', 'bayesian_gate',
    'fvg_gate', 'momentum_gate', 'volume_gate',
    'time_gate', 'correlation_gate'
]

# Define all filters to test
FILTERS_TO_TEST = [
    'volume', 'time', 'volatility', 'momentum', 'price', 'regime', 'advanced'
]


class ComprehensiveTestReport:
    """Comprehensive test report generator"""
    
    def __init__(self):
        self.pattern_results = []
        self.gate_results = []
        self.filter_results = []
    
    def add_pattern_result(self, pattern: str, expected: int, actual: int, passed: bool, notes: str = ""):
        """Add a pattern test result"""
        self.pattern_results.append({
            'type': 'pattern',
            'name': pattern,
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'notes': notes
        })
    
    def add_gate_result(self, gate: str, expected: int, actual: int, passed: bool, notes: str = ""):
        """Add a gate test result"""
        self.gate_results.append({
            'type': 'gate',
            'name': gate,
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'notes': notes
        })
    
    def add_filter_result(self, filter_name: str, expected: int, actual: int, passed: bool, notes: str = ""):
        """Add a filter test result"""
        self.filter_results.append({
            'type': 'filter',
            'name': filter_name,
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'notes': notes
        })
    
    def generate_report(self) -> str:
        """Generate the comprehensive test report"""
        report = "# Comprehensive Pattern, Gate, and Filter Test Report\n\n"
        
        # Patterns section
        report += "## Pattern Detection Tests\n\n"
        report += "| Pattern | Expected | GUI | Result | Notes |\n"
        report += "|---------|----------|-----|--------|-------|\n"
        
        for result in self.pattern_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"| {result['name']} | {result['expected']} | {result['actual']} | {status} | {result['notes']} |\n"
        
        # Gates section
        report += "\n## Gate Tests\n\n"
        report += "| Gate | Expected | GUI | Result | Notes |\n"
        report += "|------|----------|-----|--------|-------|\n"
        
        for result in self.gate_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"| {result['name']} | {result['expected']} | {result['actual']} | {status} | {result['notes']} |\n"
        
        # Filters section
        report += "\n## Filter Tests\n\n"
        report += "| Filter | Expected | GUI | Result | Notes |\n"
        report += "|--------|----------|-----|--------|-------|\n"
        
        for result in self.filter_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"| {result['name']} | {result['expected']} | {result['actual']} | {status} | {result['notes']} |\n"
        
        # Summary
        total_patterns = len(self.pattern_results)
        passed_patterns = sum(1 for r in self.pattern_results if r['passed'])
        total_gates = len(self.gate_results)
        passed_gates = sum(1 for r in self.gate_results if r['passed'])
        total_filters = len(self.filter_results)
        passed_filters = sum(1 for r in self.filter_results if r['passed'])
        
        total_tests = total_patterns + total_gates + total_filters
        total_passed = passed_patterns + passed_gates + passed_filters
        
        report += f"\n## Summary\n"
        report += f"- **Total Tests**: {total_tests}\n"
        report += f"- **Patterns**: {passed_patterns}/{total_patterns} passed\n"
        report += f"- **Gates**: {passed_gates}/{total_gates} passed\n"
        report += f"- **Filters**: {passed_filters}/{total_filters} passed\n"
        report += f"- **Overall Success Rate**: {total_passed/total_tests*100:.1f}%\n"
        
        return report


@pytest.mark.parametrize("pattern_name", PATTERNS_TO_TEST)
def test_pattern_detection(pattern_name: str):
    """
    Test pattern detection for each pattern.
    """
    print(f"\n{'='*60}")
    print(f"TESTING PATTERN: {pattern_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Create synthetic dataset
        data_path = create_dataset(pattern_name)
        print(f"Created dataset: {data_path}")
        
        # Get expected detection index
        expected_index = get_expected_detection_index(pattern_name)
        print(f"Expected detection index: {expected_index}")
        
        # Create strategy configuration
        strategy_config = create_single_pattern_strategy(pattern_name, {})
        print(f"Strategy config: {json.dumps(strategy_config, indent=2)}")
        
        # Run headless test
        actual_index, results = run_headless_test(strategy_config, data_path)
        print(f"Actual detection index: {actual_index}")
        print(f"Results keys: {list(results.keys()) if results else 'No results'}")
        
        # Determine if test passed
        passed = actual_index == expected_index
        notes = ""
        
        if not passed:
            if actual_index == -1:
                notes = "No detection found"
            elif actual_index == 0:
                notes = "Detected at first bar (likely false positive)"
            else:
                notes = f"Detected at wrong index"
        
        # Assert the result
        assert passed, f"Pattern {pattern_name}: Expected index {expected_index}, got {actual_index}. {notes}"
        
        print(f"✅ PASS: {pattern_name} detected correctly at index {actual_index}")
        
    except Exception as e:
        print(f"❌ FAIL: {pattern_name} - {str(e)}")
        raise


@pytest.mark.parametrize("gate_name", GATES_TO_TEST)
def test_gate_detection(gate_name: str):
    """
    Test gate detection for each gate.
    """
    print(f"\n{'='*60}")
    print(f"TESTING GATE: {gate_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Create synthetic dataset
        data_path = create_gate_dataset(gate_name)
        print(f"Created dataset: {data_path}")
        
        # Get expected detection index (gates typically trigger at bar 5)
        expected_index = 5
        print(f"Expected detection index: {expected_index}")
        
        # Create strategy configuration
        strategy_config = create_single_pattern_strategy(gate_name, {})
        print(f"Strategy config: {json.dumps(strategy_config, indent=2)}")
        
        # Run headless test
        actual_index, results = run_headless_test(strategy_config, data_path)
        print(f"Actual detection index: {actual_index}")
        print(f"Results keys: {list(results.keys()) if results else 'No results'}")
        
        # Determine if test passed
        passed = actual_index == expected_index
        notes = ""
        
        if not passed:
            if actual_index == -1:
                notes = "No detection found"
            elif actual_index == 0:
                notes = "Detected at first bar (likely false positive)"
            else:
                notes = f"Detected at wrong index"
        
        # Assert the result
        assert passed, f"Gate {gate_name}: Expected index {expected_index}, got {actual_index}. {notes}"
        
        print(f"✅ PASS: {gate_name} detected correctly at index {actual_index}")
        
    except Exception as e:
        print(f"❌ FAIL: {gate_name} - {str(e)}")
        raise


@pytest.mark.parametrize("filter_name", FILTERS_TO_TEST)
def test_filter_detection(filter_name: str):
    """
    Test filter detection for each filter.
    """
    print(f"\n{'='*60}")
    print(f"TESTING FILTER: {filter_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Create synthetic dataset
        data_path = create_filter_dataset(filter_name)
        print(f"Created dataset: {data_path}")
        
        # Get expected detection index (filters typically trigger at bar 5)
        expected_index = 5
        print(f"Expected detection index: {expected_index}")
        
        # Create strategy configuration
        strategy_config = create_single_pattern_strategy(filter_name, {})
        print(f"Strategy config: {json.dumps(strategy_config, indent=2)}")
        
        # Run headless test
        actual_index, results = run_headless_test(strategy_config, data_path)
        print(f"Actual detection index: {actual_index}")
        print(f"Results keys: {list(results.keys()) if results else 'No results'}")
        
        # Determine if test passed
        passed = actual_index == expected_index
        notes = ""
        
        if not passed:
            if actual_index == -1:
                notes = "No detection found"
            elif actual_index == 0:
                notes = "Detected at first bar (likely false positive)"
            else:
                notes = f"Detected at wrong index"
        
        # Assert the result
        assert passed, f"Filter {filter_name}: Expected index {expected_index}, got {actual_index}. {notes}"
        
        print(f"✅ PASS: {filter_name} detected correctly at index {actual_index}")
        
    except Exception as e:
        print(f"❌ FAIL: {filter_name} - {str(e)}")
        raise


def test_comprehensive_suite():
    """
    Run comprehensive test suite for all patterns, gates, and filters.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PATTERN, GATE, AND FILTER TEST SUITE")
    print("="*80)
    
    report = ComprehensiveTestReport()
    
    # Test all patterns
    print("\nTesting patterns...")
    for pattern_name in PATTERNS_TO_TEST:
        print(f"  Testing {pattern_name}...")
        
        try:
            # Create synthetic dataset
            data_path = create_dataset(pattern_name)
            
            # Get expected detection index
            expected_index = get_expected_detection_index(pattern_name)
            
            # Create strategy configuration
            strategy_config = create_single_pattern_strategy(pattern_name, {})
            
            # Run headless test
            actual_index, results = run_headless_test(strategy_config, data_path)
            
            # Determine if test passed
            passed = actual_index == expected_index
            notes = ""
            
            if not passed:
                if actual_index == -1:
                    notes = "No detection found"
                elif actual_index == 0:
                    notes = "Detected at first bar (likely false positive)"
                else:
                    notes = f"Detected at wrong index"
            
            # Add to report
            report.add_pattern_result(pattern_name, expected_index, actual_index, passed, notes)
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {status}: Expected {expected_index}, Got {actual_index}")
            
        except Exception as e:
            print(f"    ❌ ERROR: {str(e)}")
            report.add_pattern_result(pattern_name, -1, -1, False, f"Error: {str(e)}")
    
    # Test all gates
    print("\nTesting gates...")
    for gate_name in GATES_TO_TEST:
        print(f"  Testing {gate_name}...")
        
        try:
            # Create synthetic dataset
            data_path = create_gate_dataset(gate_name)
            
            # Get expected detection index
            expected_index = 5
            
            # Create strategy configuration
            strategy_config = create_single_pattern_strategy(gate_name, {})
            
            # Run headless test
            actual_index, results = run_headless_test(strategy_config, data_path)
            
            # Determine if test passed
            passed = actual_index == expected_index
            notes = ""
            
            if not passed:
                if actual_index == -1:
                    notes = "No detection found"
                elif actual_index == 0:
                    notes = "Detected at first bar (likely false positive)"
                else:
                    notes = f"Detected at wrong index"
            
            # Add to report
            report.add_gate_result(gate_name, expected_index, actual_index, passed, notes)
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {status}: Expected {expected_index}, Got {actual_index}")
            
        except Exception as e:
            print(f"    ❌ ERROR: {str(e)}")
            report.add_gate_result(gate_name, -1, -1, False, f"Error: {str(e)}")
    
    # Test all filters
    print("\nTesting filters...")
    for filter_name in FILTERS_TO_TEST:
        print(f"  Testing {filter_name}...")
        
        try:
            # Create synthetic dataset
            data_path = create_filter_dataset(filter_name)
            
            # Get expected detection index
            expected_index = 5
            
            # Create strategy configuration
            strategy_config = create_single_pattern_strategy(filter_name, {})
            
            # Run headless test
            actual_index, results = run_headless_test(strategy_config, data_path)
            
            # Determine if test passed
            passed = actual_index == expected_index
            notes = ""
            
            if not passed:
                if actual_index == -1:
                    notes = "No detection found"
                elif actual_index == 0:
                    notes = "Detected at first bar (likely false positive)"
                else:
                    notes = f"Detected at wrong index"
            
            # Add to report
            report.add_filter_result(filter_name, expected_index, actual_index, passed, notes)
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {status}: Expected {expected_index}, Got {actual_index}")
            
        except Exception as e:
            print(f"    ❌ ERROR: {str(e)}")
            report.add_filter_result(filter_name, -1, -1, False, f"Error: {str(e)}")
    
    # Generate and save report
    report_content = report.generate_report()
    
    with open("tests/COMPREHENSIVE_TEST_REPORT.md", "w") as f:
        f.write(report_content)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST REPORT GENERATED: tests/COMPREHENSIVE_TEST_REPORT.md")
    print("="*80)
    print(report_content)
    
    # Return summary for pytest
    total_patterns = len(report.pattern_results)
    passed_patterns = sum(1 for r in report.pattern_results if r['passed'])
    total_gates = len(report.gate_results)
    passed_gates = sum(1 for r in report.gate_results if r['passed'])
    total_filters = len(report.filter_results)
    passed_filters = sum(1 for r in report.filter_results if r['passed'])
    
    total_tests = total_patterns + total_gates + total_filters
    total_passed = passed_patterns + passed_gates + passed_filters
    
    assert total_passed == total_tests, f"Only {total_passed}/{total_tests} tests passed. Check COMPREHENSIVE_TEST_REPORT.md for details."


if __name__ == "__main__":
    # Run the comprehensive test suite
    test_comprehensive_suite() 