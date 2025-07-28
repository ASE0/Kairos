#!/usr/bin/env python3
"""
Headless Runner for Pattern Testing
==================================
Runs the GUI in headless mode and extracts detection results.
"""

import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def run_headless_test(strategy_config: Dict[str, Any], data_path: Path) -> Tuple[int, Dict[str, Any]]:
    """
    Run a headless test with the given strategy and data.
    
    Args:
        strategy_config: Strategy configuration dictionary
        data_path: Path to the data CSV file
        
    Returns:
        Tuple of (detection_index, full_results)
    """
    # Debug: Print strategy config
    print(f"[DEBUG] Strategy config: {json.dumps(strategy_config, indent=2)}")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as strategy_file:
        json.dump(strategy_config, strategy_file)
        strategy_path = strategy_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Use absolute path for data file
        data_path = Path(data_path).resolve()
        print(f"[DEBUG] Data path: {data_path}")
        print(f"[DEBUG] Data file exists: {data_path.exists()}")
        
        # Run the headless command
        cmd = [
            'python', 'main.py',
            '--headless',
            '--strategy', strategy_path,
            '--data', str(data_path),
            '--output', output_path
        ]
        
        print(f"[DEBUG] Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()  # Run from current directory where main.py is located
        )
        
        print(f'[DEBUG] STDOUT: {result.stdout}')
        print(f'[DEBUG] STDERR: {result.stderr}')
        print(f'[DEBUG] Return code: {result.returncode}')
        
        if result.returncode != 0:
            print(f"Headless test failed with return code {result.returncode}")
            return -1, {}
        
        # Read the results
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                results = json.load(f)
            # Debug: Print results before cleanup
            print(f"[DEBUG] Results: {json.dumps(results, indent=2)}")
        else:
            print(f"Output file not found: {output_path}")
            return -1, {}
        
        # Extract detection index
        detection_index = _extract_detection_index(results)
        
        return detection_index, results
        
    finally:
        # Clean up temporary files
        for temp_file in [strategy_path, output_path]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def _extract_detection_index(results: Dict[str, Any]) -> int:
    """
    Extract the detection index from the results.
    
    Args:
        results: Results dictionary from headless run
        
    Returns:
        Detection index (-1 if not found)
    """
    print(f"[DEBUG] Extracting detection index from results: {list(results.keys())}")
    
    # Check for signals in results
    if 'signals' in results and results['signals']:
        print(f"[DEBUG] Found signals: {len(results['signals'])}")
        # Find the first True signal
        for i, signal in enumerate(results['signals']):
            if signal:
                print(f"[DEBUG] Found signal at index {i}")
                return i
        print("[DEBUG] No True signals found")
        return -1
    
    # Check for zones in results
    if 'zones' in results and results['zones']:
        print(f"[DEBUG] Found zones: {len(results['zones'])}")
        # Return the creation index of the first zone
        first_zone = results['zones'][0]
        print(f"[DEBUG] First zone: {first_zone}")
        
        if 'creation_index' in first_zone:
            return first_zone['creation_index']
        elif 'index' in first_zone:
            return first_zone['index']
        elif 'bar_index' in first_zone:
            return first_zone['bar_index']
        else:
            print("[DEBUG] No index found in zone, defaulting to 0")
            return 0  # Default to first bar if no index found
    
    # Check for trades in results
    if 'trades' in results and results['trades']:
        print(f"[DEBUG] Found trades: {len(results['trades'])}")
        # Return the entry index of the first trade
        first_trade = results['trades'][0]
        print(f"[DEBUG] First trade: {first_trade}")
        
        if 'entry_index' in first_trade:
            return first_trade['entry_index']
        elif 'index' in first_trade:
            return first_trade['index']
        else:
            print("[DEBUG] No index found in trade, defaulting to 0")
            return 0  # Default to first bar if no index found
    
    print("[DEBUG] No detection found in results")
    return -1


def create_single_pattern_strategy(pattern_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a strategy configuration for a single pattern.
    
    Args:
        pattern_name: Name of the pattern
        params: Pattern parameters
        
    Returns:
        Strategy configuration dictionary
    """
    if params is None:
        params = {}
    
    # Map pattern names to action names
    pattern_to_action = {
        # Basic patterns
        'fvg': 'fvg',
        'engulfing': 'engulfing',
        'hammer': 'hammer',
        'double_wick': 'double_wick',
        'ii_bars': 'ii_bars',
        'doji': 'doji',
        'marubozu': 'marubozu',
        'spinning_top': 'spinning_top',
        'weak_body': 'weak_body',
        'strong_body': 'strong_body',
        
        # Advanced patterns
        'engulfing_bullish': 'engulfing',
        'engulfing_bearish': 'engulfing',
        'doji_standard': 'doji',
        'momentum_breakout': 'momentum_breakout',
        'momentum_reversal': 'momentum_reversal',
        'high_volatility': 'high_volatility',
        'low_volatility': 'low_volatility',
        'support_bounce': 'support_bounce',
        'resistance_rejection': 'resistance_rejection',
        'three_white_soldiers': 'three_white_soldiers',
        'three_black_crows': 'three_black_crows',
        'four_price_doji': 'four_price_doji',
        'dragonfly_doji': 'dragonfly_doji',
        'gravestone_doji': 'gravestone_doji',
        'volatility_expansion': 'volatility_expansion',
        'volatility_contraction': 'volatility_contraction',
        'trend_continuation': 'trend_continuation',
        'trend_reversal': 'trend_reversal',
        'gap_up': 'gap_up',
        'gap_down': 'gap_down',
        'consolidation': 'consolidation',
        'breakout': 'breakout',
        'exhaustion': 'exhaustion',
        'accumulation': 'accumulation',
        'distribution': 'distribution'
    }
    
    action_name = pattern_to_action.get(pattern_name, pattern_name)
    
    # Handle special cases for pattern parameters
    pattern_params = params.copy() if params else {}
    
    # Add specific parameters for certain patterns
    if pattern_name == 'fvg':
        pattern_params.setdefault('min_gap_size', 0.001)
        pattern_params.setdefault('max_touches', 3)
    elif pattern_name in ['engulfing', 'engulfing_bullish', 'engulfing_bearish']:
        pattern_params.setdefault('pattern_type', 'both' if pattern_name == 'engulfing' else pattern_name.split('_')[1])
    elif pattern_name == 'hammer':
        pattern_params.setdefault('pattern_type', 'both')
    elif pattern_name in ['doji', 'doji_standard']:
        pattern_params.setdefault('min_body_ratio', 0.1)
        pattern_params.setdefault('wick_symmetry_tolerance', 0.3)
    elif pattern_name == 'marubozu':
        pattern_params.setdefault('min_body_ratio', 0.8)
    elif pattern_name == 'spinning_top':
        pattern_params.setdefault('min_wick_ratio', 0.3)
        pattern_params.setdefault('max_body_ratio', 0.4)
    elif pattern_name == 'double_wick':
        pattern_params.setdefault('min_wick_ratio', 0.3)
        pattern_params.setdefault('max_body_ratio', 0.4)
    elif pattern_name == 'ii_bars':
        pattern_params.setdefault('min_bars', 2)
    elif pattern_name == 'weak_body':
        pattern_params.setdefault('max_body_ratio', 0.3)
    elif pattern_name == 'strong_body':
        pattern_params.setdefault('min_body_ratio', 0.7)
    
    # Create strategy configuration
    strategy_config = {
        'name': f'Test_{pattern_name}',
        'actions': [
            {
                'name': action_name,
                'pattern': {
                    'params': pattern_params
                }
            }
        ]
    }
    
    return strategy_config 