#!/usr/bin/env python3
"""
Comprehensive Filter Testing - Validate All Indicators Against Mathematical Framework
================================================================================
This script tests every indicator filter in the strategy builder against the mathematical
framework documentation to ensure they're implemented correctly.

The mathematical framework specifies exact formulas and parameters that must be followed.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import subprocess
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our GUI-compatible tester
from gui_test_simulator import run_gui_compatible_test, HeadlessBacktestWindow


class FilterValidator:
    """Validates filters against mathematical framework specifications"""
    
    def __init__(self):
        self.test_results = {}
        
    def create_test_data(self, filter_type: str) -> pd.DataFrame:
        """Create test data specific to each filter type"""
        if filter_type == 'vwap':
            # Create data with known VWAP patterns
            times = pd.date_range('2024-03-07 09:00:00', periods=100, freq='1min')
            data = []
            for i in range(100):
                if i < 50:
                    # First half: price above VWAP
                    base_price = 100 + (i * 0.1)
                    data.append({
                        'open': base_price,
                        'high': base_price + 1.0,
                        'low': base_price - 1.0,
                        'close': base_price + 0.2,
                        'volume': 1000
                    })
                else:
                    # Second half: price below VWAP
                    base_price = 95 - (i * 0.1)
                    data.append({
                        'open': base_price,
                        'high': base_price + 1.0,
                        'low': base_price - 1.0,
                        'close': base_price - 0.2,
                        'volume': 1000
                    })
            return pd.DataFrame(data, index=times)
            
        elif filter_type == 'momentum':
            # Create data with momentum patterns
            times = pd.date_range('2024-03-07 09:00:00', periods=100, freq='1min')
            data = []
            for i in range(100):
                if i < 30:
                    # Strong uptrend
                    base_price = 100 + (i * 0.5)
                elif i < 70:
                    # Sideways
                    base_price = 115 + (i * 0.1)
                else:
                    # Strong downtrend
                    base_price = 120 - (i * 0.5)
                
                data.append({
                    'open': base_price,
                    'high': base_price + 1.0,
                    'low': base_price - 1.0,
                    'close': base_price + 0.2,
                    'volume': 1000
                })
            return pd.DataFrame(data, index=times)
            
        elif filter_type == 'volatility':
            # Create data with volatility patterns
            times = pd.date_range('2024-03-07 09:00:00', periods=100, freq='1min')
            data = []
            for i in range(100):
                if i < 50:
                    # Low volatility
                    base_price = 100 + (i * 0.1)
                    volatility = 0.5
                else:
                    # High volatility
                    base_price = 105 + (i * 0.1)
                    volatility = 2.0
                
                data.append({
                    'open': base_price,
                    'high': base_price + volatility,
                    'low': base_price - volatility,
                    'close': base_price + (volatility * 0.2),
                    'volume': 1000
                })
            return pd.DataFrame(data, index=times)
            
        else:
            # Default test data
            times = pd.date_range('2024-03-07 09:00:00', periods=100, freq='1min')
            data = []
            for i in range(100):
                base_price = 100 + (i * 0.1)
                data.append({
                    'open': base_price,
                    'high': base_price + 1.0,
                    'low': base_price - 1.0,
                    'close': base_price + 0.2,
                    'volume': 1000
                })
            return pd.DataFrame(data, index=times)
    
    def test_vwap_filter(self) -> Dict[str, Any]:
        """Test VWAP filter against mathematical framework"""
        print("\n" + "="*60)
        print("TESTING VWAP FILTER")
        print("="*60)
        
        # Create test data
        test_data = self.create_test_data('vwap')
        test_data_path = "test_vwap_data.csv"
        test_data.to_csv(test_data_path)
        
        # According to mathematical framework, VWAP should be:
        # VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)
        # And VWAP bands should be: VWAP ± k × σ_vwap
        
        # Calculate expected VWAP manually
        vwap_expected = (test_data['close'] * test_data['volume']).cumsum() / test_data['volume'].cumsum()
        price_deviations = test_data['close'] - vwap_expected
        sigma_vwap = price_deviations.rolling(window=20).std()
        
        print(f"Expected VWAP calculation:")
        print(f"  - VWAP formula: Σ(Price × Volume) / Σ(Volume)")
        print(f"  - VWAP bands: VWAP ± k × σ_vwap")
        print(f"  - σ_vwap = std(Price - VWAP) over lookback period")
        
        # Test current implementation
        strategy_config = {
            "name": "VWAP Test Strategy",
            "actions": [
                {
                    "name": "test_action",
                    "location_strategy": "FVG",
                    "location_params": {},
                    "filters": [
                        {
                            "type": "vwap",
                            "condition": "above"  # Test above VWAP condition
                        }
                    ]
                }
            ],
            "combination_logic": "AND",
            "gates_and_logic": {"location_gate": True}
        }
        
        # Run test
        results = run_gui_compatible_test(strategy_config, test_data_path, "test_vwap_results.json")
        
        # Analyze results
        if 'error' in results:
            return {'status': 'ERROR', 'error': results['error']}
        
        # Check if VWAP filter is working correctly
        signals = results.get('signals', [])
        if isinstance(signals, list):
            signals = pd.Series(signals, index=test_data.index)
        
        # Expected behavior: signals should be True when price > VWAP
        expected_signals = test_data['close'] > vwap_expected
        
        # Compare actual vs expected
        correct_signals = (signals == expected_signals).sum()
        total_signals = len(signals)
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        print(f"VWAP Filter Results:")
        print(f"  - Total signals: {total_signals}")
        print(f"  - Correct signals: {correct_signals}")
        print(f"  - Accuracy: {accuracy:.2%}")
        
        # Check specific conditions
        above_vwap_count = (test_data['close'] > vwap_expected).sum()
        below_vwap_count = (test_data['close'] < vwap_expected).sum()
        
        print(f"  - Bars above VWAP: {above_vwap_count}")
        print(f"  - Bars below VWAP: {below_vwap_count}")
        
        return {
            'status': 'PASS' if accuracy > 0.8 else 'FAIL',
            'accuracy': accuracy,
            'total_signals': total_signals,
            'correct_signals': correct_signals,
            'above_vwap_count': above_vwap_count,
            'below_vwap_count': below_vwap_count
        }
    
    def test_momentum_filter(self) -> Dict[str, Any]:
        """Test momentum filter against mathematical framework"""
        print("\n" + "="*60)
        print("TESTING MOMENTUM FILTER")
        print("="*60)
        
        # Create test data
        test_data = self.create_test_data('momentum')
        test_data_path = "test_momentum_data.csv"
        test_data.to_csv(test_data_path)
        
        # According to mathematical framework, momentum should be:
        # M(t,y) = (1/n) Σ |r_i|·sign(r_i)
        # where r_i are returns over lookback period
        
        # Calculate expected momentum manually
        returns = test_data['close'].pct_change().dropna()
        lookback = 10
        momentum_expected = []
        
        for i in range(lookback, len(returns)):
            recent_returns = returns.iloc[i-lookback:i]
            # M(t,y) = (1/n) Σ |r_i|·sign(r_i)
            momentum = np.mean(np.abs(recent_returns) * np.sign(recent_returns))
            momentum_expected.append(momentum)
        
        print(f"Expected Momentum calculation:")
        print(f"  - Momentum formula: M(t,y) = (1/n) Σ |r_i|·sign(r_i)")
        print(f"  - Where r_i are returns over lookback period")
        
        # Test current implementation
        strategy_config = {
            "name": "Momentum Test Strategy",
            "actions": [
                {
                    "name": "test_action",
                    "location_strategy": "FVG",
                    "location_params": {},
                    "filters": [
                        {
                            "type": "momentum",
                            "momentum_threshold": 0.02,
                            "rsi_range": [30, 70]
                        }
                    ]
                }
            ],
            "combination_logic": "AND",
            "gates_and_logic": {"location_gate": True}
        }
        
        # Run test
        results = run_gui_compatible_test(strategy_config, test_data_path, "test_momentum_results.json")
        
        # Analyze results
        if 'error' in results:
            return {'status': 'ERROR', 'error': results['error']}
        
        signals = results.get('signals', [])
        if isinstance(signals, list):
            signals = pd.Series(signals, index=test_data.index)
        
        # Expected behavior: signals should be True when momentum > threshold
        momentum_series = pd.Series(momentum_expected, index=test_data.index[lookback:])
        expected_signals = momentum_series > 0.02
        
        # Compare actual vs expected
        correct_signals = (signals == expected_signals).sum()
        total_signals = len(signals)
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        print(f"Momentum Filter Results:")
        print(f"  - Total signals: {total_signals}")
        print(f"  - Correct signals: {correct_signals}")
        print(f"  - Accuracy: {accuracy:.2%}")
        
        return {
            'status': 'PASS' if accuracy > 0.8 else 'FAIL',
            'accuracy': accuracy,
            'total_signals': total_signals,
            'correct_signals': correct_signals
        }
    
    def test_volatility_filter(self) -> Dict[str, Any]:
        """Test volatility filter against mathematical framework"""
        print("\n" + "="*60)
        print("TESTING VOLATILITY FILTER")
        print("="*60)
        
        # Create test data
        test_data = self.create_test_data('volatility')
        test_data_path = "test_volatility_data.csv"
        test_data.to_csv(test_data_path)
        
        # According to mathematical framework, volatility should be:
        # ATR = Average True Range
        # Realized Vol = std(returns)
        
        # Calculate expected ATR manually
        high_low = test_data['high'] - test_data['low']
        high_close = np.abs(test_data['high'] - test_data['close'].shift())
        low_close = np.abs(test_data['low'] - test_data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_expected = true_range.rolling(window=14).mean()
        
        # Calculate realized volatility
        returns = test_data['close'].pct_change().dropna()
        realized_vol_expected = returns.rolling(window=20).std()
        
        print(f"Expected Volatility calculation:")
        print(f"  - ATR = Average True Range")
        print(f"  - Realized Vol = std(returns)")
        print(f"  - ATR Ratio = ATR / Average Price")
        
        # Test current implementation
        strategy_config = {
            "name": "Volatility Test Strategy",
            "actions": [
                {
                    "name": "test_action",
                    "location_strategy": "FVG",
                    "location_params": {},
                    "filters": [
                        {
                            "type": "volatility",
                            "min_atr_ratio": 0.01,
                            "max_atr_ratio": 0.05
                        }
                    ]
                }
            ],
            "combination_logic": "AND",
            "gates_and_logic": {"location_gate": True}
        }
        
        # Run test
        results = run_gui_compatible_test(strategy_config, test_data_path, "test_volatility_results.json")
        
        # Analyze results
        if 'error' in results:
            return {'status': 'ERROR', 'error': results['error']}
        
        signals = results.get('signals', [])
        if isinstance(signals, list):
            signals = pd.Series(signals, index=test_data.index)
        
        # Expected behavior: signals should be True when ATR ratio is within bounds
        avg_price = test_data['close'].rolling(window=14).mean()
        atr_ratio = atr_expected / avg_price
        expected_signals = (atr_ratio >= 0.01) & (atr_ratio <= 0.05)
        
        # Compare actual vs expected
        correct_signals = (signals == expected_signals).sum()
        total_signals = len(signals)
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        print(f"Volatility Filter Results:")
        print(f"  - Total signals: {total_signals}")
        print(f"  - Correct signals: {correct_signals}")
        print(f"  - Accuracy: {accuracy:.2%}")
        
        return {
            'status': 'PASS' if accuracy > 0.8 else 'FAIL',
            'accuracy': accuracy,
            'total_signals': total_signals,
            'correct_signals': correct_signals
        }
    
    def test_all_filters(self):
        """Test all filters systematically"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FILTER TESTING")
        print("="*80)
        
        # Test each filter
        filters_to_test = [
            ('vwap', self.test_vwap_filter),
            ('momentum', self.test_momentum_filter),
            ('volatility', self.test_volatility_filter)
        ]
        
        results = {}
        
        for filter_name, test_func in filters_to_test:
            print(f"\nTesting {filter_name.upper()} filter...")
            try:
                result = test_func()
                results[filter_name] = result
                print(f"✅ {filter_name.upper()}: {result['status']}")
            except Exception as e:
                print(f"❌ {filter_name.upper()}: ERROR - {e}")
                results[filter_name] = {'status': 'ERROR', 'error': str(e)}
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for filter_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASS':
                print(f"✅ {filter_name.upper()}: PASS")
            elif status == 'FAIL':
                print(f"❌ {filter_name.upper()}: FAIL")
            else:
                print(f"⚠️ {filter_name.upper()}: {status}")
        
        # Clean up test files
        for file in ["test_vwap_data.csv", "test_momentum_data.csv", "test_volatility_data.csv",
                    "test_vwap_results.json", "test_momentum_results.json", "test_volatility_results.json"]:
            if os.path.exists(file):
                os.remove(file)
        
        return results


def main():
    """Run comprehensive filter testing"""
    validator = FilterValidator()
    results = validator.test_all_filters()
    
    print(f"\n{'='*80}")
    print("MATHEMATICAL FRAMEWORK COMPLIANCE")
    print(f"{'='*80}")
    print("The mathematical framework specifies exact formulas that must be followed:")
    print()
    print("1. VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)")
    print("2. VWAP Bands = VWAP ± k × σ_vwap")
    print("3. Momentum = (1/n) Σ |r_i|·sign(r_i)")
    print("4. ATR = Average True Range")
    print("5. Realized Vol = std(returns)")
    print()
    print("All filters must implement these exact formulas to be compliant.")


if __name__ == "__main__":
    main() 