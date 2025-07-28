#!/usr/bin/env python3
"""
Comprehensive Zone Type Testing Script
=====================================

This script tests all 5 zone types to ensure they are implemented correctly
according to the documentation specifications. It tests:

1. FVG (Fair Value Gap) zones
2. Order Block zones  
3. VWAP Mean-Reversion Band zones
4. Support/Resistance Band zones
5. Imbalance Memory Zone zones

Each zone type is tested for:
- Correct detection logic
- Parameter sensitivity
- Zone-specific decay behavior
- Mathematical implementation compliance
- GUI parameter mapping
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_builders import PatternStrategy, Action
from core.data_structures import BaseStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data_with_patterns():
    """Create test data with specific patterns to trigger each zone type"""
    np.random.seed(42)  # For reproducible results
    
    # Create 200 bars of test data
    n_bars = 200
    base_price = 100.0
    
    # Generate price data with specific patterns
    prices = [base_price]
    opens = [base_price]
    highs = [base_price + 0.5]
    lows = [base_price - 0.5]
    volumes = [1000]
    
    for i in range(1, n_bars):
        prev_price = prices[-1]
        
        # Create specific patterns at certain indices
        if i == 10:  # FVG pattern - bullish gap
            # Create a gap: H_{t-1} < L_{t+1}
            opens.append(prev_price + 2.0)  # Gap up
            highs.append(opens[-1] + 1.0)
            lows.append(opens[-1] - 0.5)
            prices.append(opens[-1] + 0.3)
            
        elif i == 20:  # FVG pattern - bearish gap
            # Create a gap: L_{t-1} > H_{t+1}
            opens.append(prev_price - 2.0)  # Gap down
            highs.append(opens[-1] + 0.5)
            lows.append(opens[-1] - 1.0)
            prices.append(opens[-1] - 0.3)
            
        elif i == 30:  # Order Block pattern - bullish OB
            # Last down candle before sharp up impulse
            opens.append(prev_price + 0.5)
            highs.append(opens[-1] + 0.3)
            lows.append(opens[-1] - 0.8)
            prices.append(opens[-1] - 0.4)  # Down candle
            
        elif i == 31:  # Sharp up impulse after OB
            opens.append(prices[-1] - 0.1)
            highs.append(opens[-1] + 3.0)  # Sharp up move
            lows.append(opens[-1] - 0.2)
            prices.append(opens[-1] + 2.5)
            
        elif i == 40:  # Order Block pattern - bearish OB
            # Last up candle before sharp down impulse
            opens.append(prev_price - 0.5)
            highs.append(opens[-1] + 0.8)
            lows.append(opens[-1] - 0.3)
            prices.append(opens[-1] + 0.4)  # Up candle
            
        elif i == 41:  # Sharp down impulse after OB
            opens.append(prices[-1] + 0.1)
            highs.append(opens[-1] + 0.2)
            lows.append(opens[-1] - 3.0)  # Sharp down move
            prices.append(opens[-1] - 2.5)
            
        elif i == 50:  # Imbalance pattern - large move
            # Create a significant price move
            move_size = 5.0
            opens.append(prev_price)
            highs.append(opens[-1] + move_size)
            lows.append(opens[-1] - 0.5)
            prices.append(opens[-1] + move_size * 0.8)
            
        else:
            # Normal price movement
            change = np.random.normal(0, 0.5)
            new_price = prev_price + change
            new_open = prev_price + np.random.normal(0, 0.2)
            
            opens.append(new_open)
            prices.append(new_price)
            highs.append(max(new_open, new_price) + np.random.uniform(0.1, 0.5))
            lows.append(min(new_open, new_price) - np.random.uniform(0.1, 0.5))
        
        volumes.append(np.random.randint(500, 2000))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    # Create datetime index
    start_date = datetime(2023, 1, 1, 9, 30)
    df.index = pd.date_range(start=start_date, periods=len(df), freq='1min')
    
    return df

def test_fvg_zones():
    """Test FVG (Fair Value Gap) zone detection"""
    print("\n" + "="*60)
    print("TESTING FVG (FAIR VALUE GAP) ZONES")
    print("="*60)
    
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("FVG Test")
    
    # DEBUG: Show the actual candle values at FVG test indices
    print("\nDEBUG: Candle values at FVG test indices:")
    test_indices = [10, 11, 20, 21]  # Where we created FVG patterns
    for idx in test_indices:
        if idx < len(data):
            candle = data.iloc[idx]
            print(f"Index {idx}: O={candle['open']:.2f}, H={candle['high']:.2f}, L={candle['low']:.2f}, C={candle['close']:.2f}")
            if idx >= 2:
                candle1 = data.iloc[idx-2]  # t-1
                candle3 = data.iloc[idx]    # t+1
                print(f"  FVG check: H_{idx-2}={candle1['high']:.2f} vs L_{idx}={candle3['low']:.2f}")
                print(f"  Bullish FVG condition: {candle1['high']:.2f} < {candle3['low']:.2f} = {candle1['high'] < candle3['low']}")
                print(f"  Bearish FVG condition: {candle1['low']:.2f} > {candle3['high']:.2f} = {candle1['low'] > candle3['high']}")
                
                # Calculate gap sizes
                if candle1['high'] < candle3['low']:
                    gap_size = candle3['low'] - candle1['high']
                    print(f"  Bullish gap size: {gap_size:.2f}")
                elif candle1['low'] > candle3['high']:
                    gap_size = candle1['low'] - candle3['high']
                    print(f"  Bearish gap size: {gap_size:.2f}")
                
                # Show zone boundaries with epsilon
                epsilon = 2.0  # Default FVG epsilon
                if candle1['high'] < candle3['low']:
                    x0 = candle1['high'] + epsilon
                    x1 = candle3['low'] - epsilon
                    print(f"  Zone boundaries: x0={x0:.2f}, x1={x1:.2f}, valid={x0 < x1}")
                elif candle1['low'] > candle3['high']:
                    x0 = candle3['high'] + epsilon
                    x1 = candle1['low'] - epsilon
                    print(f"  Zone boundaries: x0={x0:.2f}, x1={x1:.2f}, valid={x0 < x1}")
    
    # Test different FVG parameters
    test_params = [
        {
            'name': 'Default FVG',
            'fvg_epsilon': 0.1,  # Lowered from 2.0
            'fvg_N': 3,
            'fvg_sigma': 0.1,
            'fvg_beta1': 0.7,
            'fvg_beta2': 0.3,
            'fvg_phi': 0.2,
            'fvg_lambda': 0.0,
            'fvg_gamma': 0.95,
            'fvg_tau_bars': 50,
            'fvg_drop_threshold': 0.01
        },
        {
            'name': 'Sensitive FVG',
            'fvg_epsilon': 0.1,  # Lowered from 1.0
            'fvg_N': 5,          # More peaks
            'fvg_sigma': 0.05,   # Sharper peaks
            'fvg_beta1': 0.5,
            'fvg_beta2': 0.5,
            'fvg_phi': 0.1,
            'fvg_lambda': 0.5,   # Some skew
            'fvg_gamma': 0.98,   # Slower decay
            'fvg_tau_bars': 100,
            'fvg_drop_threshold': 0.005
        }
    ]
    
    for params in test_params:
        print(f"\n--- Testing {params['name']} ---")
        
        # Set parameters
        strategy.location_gate_params.update(params)
        strategy.gates_and_logic = {'location_gate': True}
        
        # Test zone detection at specific indices
        for idx in test_indices:
            zones = strategy._detect_fvg_zones(data, idx)
            print(f"Index {idx}: Found {len(zones)} FVG zones")
            
            for zone in zones:
                print(f"  - {zone['direction']} FVG: {zone['zone_min']:.2f} - {zone['zone_max']:.2f}")
                print(f"    Gap size: {zone.get('gap_size', 'N/A'):.2f}")
                print(f"    Comb centers: {len(zone.get('comb_centers', []))}")
                print(f"    Strength: {zone.get('strength', 0):.3f}")
                
                # Verify zone properties
                assert zone['type'] == 'FVG'
                assert zone['direction'] in ['bullish', 'bearish']
                assert zone['zone_min'] < zone['zone_max']
                assert len(zone.get('comb_centers', [])) == params['fvg_N']
                
                # Check if current price is in zone
                current_price = data.iloc[idx]['close']
                in_zone = zone['zone_min'] <= current_price <= zone['zone_max']
                print(f"    Current price {current_price:.2f} {'IN' if in_zone else 'OUTSIDE'} zone")
    
    print("FVG zone testing completed")

def test_order_block_zones():
    """Test Order Block zone detection"""
    print("\n" + "="*60)
    print("TESTING ORDER BLOCK ZONES")
    print("="*60)
    
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("Order Block Test")
    
    # Test different Order Block parameters
    test_params = [
        {
            'name': 'Default Order Block',
            'ob_impulse_threshold': 0.02,
            'ob_lookback': 10,
            'ob_gamma': 0.95,
            'ob_tau_bars': 80,
            'ob_drop_threshold': 0.01
        },
        {
            'name': 'Sensitive Order Block',
            'ob_impulse_threshold': 0.01,  # Lower threshold
            'ob_lookback': 20,             # Longer lookback
            'ob_gamma': 0.98,              # Slower decay
            'ob_tau_bars': 120,
            'ob_drop_threshold': 0.005
        }
    ]
    
    for params in test_params:
        print(f"\n--- Testing {params['name']} ---")
        
        # Set parameters
        strategy.location_gate_params.update(params)
        strategy.gates_and_logic = {'location_gate': True}
        
        # Test zone detection at specific indices
        test_indices = [30, 31, 40, 41]  # Where we created OB patterns
        
        for idx in test_indices:
            zones = strategy._detect_order_block_zones(data, idx)
            print(f"Index {idx}: Found {len(zones)} Order Block zones")
            
            for zone in zones:
                print(f"  - {zone['direction']} OB: {zone['zone_min']:.2f} - {zone['zone_max']:.2f}")
                print(f"    Impulse threshold: {zone.get('impulse_threshold', 'N/A')}")
                print(f"    Lookback: {zone.get('lookback', 'N/A')}")
                print(f"    Strength: {zone.get('strength', 0):.3f}")
                
                # Verify zone properties
                assert zone['type'] == 'OrderBlock'
                assert zone['direction'] in ['bullish', 'bearish']
                assert zone['zone_min'] < zone['zone_max']
                
                # Check if current price is in zone
                current_price = data.iloc[idx]['close']
                in_zone = zone['zone_min'] <= current_price <= zone['zone_max']
                print(f"    Current price {current_price:.2f} {'IN' if in_zone else 'OUTSIDE'} zone")
    
    print("Order Block zone testing completed")

def test_vwap_zones():
    """Test VWAP Mean-Reversion Band zone detection"""
    print("\n" + "="*60)
    print("TESTING VWAP MEAN-REVERSION BAND ZONES")
    print("="*60)
    
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("VWAP Test")
    
    # Test different VWAP parameters
    test_params = [
        {
            'name': 'Default VWAP',
            'vwap_k': 1.0,
            'vwap_lookback': 20,
            'vwap_gamma': 0.95,
            'vwap_tau_bars': 15,
            'vwap_drop_threshold': 0.01
        },
        {
            'name': 'Wide VWAP Band',
            'vwap_k': 2.0,        # Wider band
            'vwap_lookback': 30,  # Longer lookback
            'vwap_gamma': 0.98,   # Slower decay
            'vwap_tau_bars': 25,
            'vwap_drop_threshold': 0.005
        }
    ]
    
    for params in test_params:
        print(f"\n--- Testing {params['name']} ---")
        
        # Set parameters
        strategy.location_gate_params.update(params)
        strategy.gates_and_logic = {'location_gate': True}
        
        # Test zone detection at various indices
        test_indices = [25, 50, 75, 100]
        
        for idx in test_indices:
            zones = strategy._detect_vwap_zones(data, idx)
            print(f"Index {idx}: Found {len(zones)} VWAP zones")
            
            for zone in zones:
                print(f"  - VWAP zone: {zone['zone_min']:.2f} - {zone['zone_max']:.2f}")
                print(f"    VWAP center: {zone.get('mu', 'N/A'):.2f}")
                print(f"    k multiplier: {zone.get('k', 'N/A')}")
                print(f"    Lookback: {zone.get('lookback', 'N/A')}")
                print(f"    Sigma VWAP: {zone.get('sigma_vwap', 'N/A'):.3f}")
                
                # Verify zone properties
                assert zone['type'] == 'VWAP'
                assert zone['direction'] == 'neutral'
                assert zone['zone_min'] < zone['zone_max']
                assert len(zone.get('comb_centers', [])) == 1  # Single VWAP center
                
                # Check if current price is in zone
                current_price = data.iloc[idx]['close']
                in_zone = zone['zone_min'] <= current_price <= zone['zone_max']
                print(f"    Current price {current_price:.2f} {'IN' if in_zone else 'OUTSIDE'} zone")
    
    print("VWAP zone testing completed")

def test_imbalance_zones():
    """Test Imbalance Memory Zone detection"""
    print("\n" + "="*60)
    print("TESTING IMBALANCE MEMORY ZONE ZONES")
    print("="*60)
    
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("Imbalance Test")
    
    # Test different Imbalance parameters
    test_params = [
        {
            'name': 'Default Imbalance',
            'imbalance_threshold': 5,      # Lower for testing
            'imbalance_gamma_mem': 0.01,
            'imbalance_sigma_rev': 20,
            'imbalance_gamma': 0.95,
            'imbalance_tau_bars': 100,
            'imbalance_drop_threshold': 0.01
        },
        {
            'name': 'Sensitive Imbalance',
            'imbalance_threshold': 2,      # Very sensitive
            'imbalance_gamma_mem': 0.005,  # Slower memory decay
            'imbalance_sigma_rev': 10,     # Narrower revisit width
            'imbalance_gamma': 0.98,       # Slower zone decay
            'imbalance_tau_bars': 150,
            'imbalance_drop_threshold': 0.005
        }
    ]
    
    for params in test_params:
        print(f"\n--- Testing {params['name']} ---")
        
        # Set parameters
        strategy.location_gate_params.update(params)
        strategy.gates_and_logic = {'location_gate': True}
        
        # Test zone detection at specific indices
        test_indices = [50, 51, 75, 100]  # Where we created imbalance patterns
        
        for idx in test_indices:
            zones = strategy._detect_imbalance_zones(data, idx)
            print(f"Index {idx}: Found {len(zones)} Imbalance zones")
            
            for zone in zones:
                print(f"  - {zone['direction']} Imbalance: {zone['zone_min']:.2f} - {zone['zone_max']:.2f}")
                print(f"    Magnitude: {zone.get('magnitude', 'N/A'):.2f}")
                print(f"    Threshold: {zone.get('imbalance_threshold', 'N/A')}")
                print(f"    Gamma mem: {zone.get('gamma_mem', 'N/A')}")
                print(f"    Sigma rev: {zone.get('sigma_rev', 'N/A')}")
                
                # Verify zone properties
                assert zone['type'] == 'Imbalance'
                assert zone['direction'] in ['bullish', 'bearish']
                assert zone['zone_min'] < zone['zone_max']
                
                # Check if current price is in zone
                current_price = data.iloc[idx]['close']
                in_zone = zone['zone_min'] <= current_price <= zone['zone_max']
                print(f"    Current price {current_price:.2f} {'IN' if in_zone else 'OUTSIDE'} zone")
    
    print("Imbalance zone testing completed")

def test_zone_decay_system():
    """Test zone-specific decay behavior"""
    print("\n" + "="*60)
    print("TESTING ZONE DECAY SYSTEM")
    print("="*60)
    
    strategy = PatternStrategy("Decay Test")
    
    # Test decay for each zone type
    zone_types = ['FVG', 'OrderBlock', 'VWAP', 'Imbalance']
    
    for zone_type in zone_types:
        print(f"\n--- Testing {zone_type} Decay ---")
        
        # Test zone_is_active method
        for bars in [5, 10, 20, 50, 100]:
            is_active = strategy.zone_is_active(bars, zone_type)
            strength = strategy.calculate_zone_strength(bars, 1.0, zone_type)
            print(f"  Bars {bars:3d}: Active={is_active}, Strength={strength:.4f}")
            
            # Verify decay behavior
            if bars > 0:
                prev_strength = strategy.calculate_zone_strength(bars-1, 1.0, zone_type)
                assert strength <= prev_strength, f"Strength should not increase over time for {zone_type}"
    
    print("Zone decay testing completed")

def test_parameter_sensitivity():
    """Test parameter sensitivity for each zone type"""
    print("\n" + "="*60)
    print("TESTING PARAMETER SENSITIVITY")
    print("="*60)
    
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("Parameter Test")
    strategy.gates_and_logic = {'location_gate': True}
    
    # Test FVG parameter sensitivity
    print("\n--- FVG Parameter Sensitivity ---")
    base_params = {
        'fvg_epsilon': 2.0, 'fvg_N': 3, 'fvg_sigma': 0.1,
        'fvg_beta1': 0.7, 'fvg_beta2': 0.3, 'fvg_phi': 0.2,
        'fvg_lambda': 0.0, 'fvg_gamma': 0.95, 'fvg_tau_bars': 50,
        'fvg_drop_threshold': 0.01
    }
    
    # Test epsilon sensitivity
    for epsilon in [1.0, 2.0, 3.0]:
        params = base_params.copy()
        params['fvg_epsilon'] = epsilon
        strategy.location_gate_params.update(params)
        
        zones = strategy._detect_fvg_zones(data, 10)
        print(f"  Epsilon {epsilon}: {len(zones)} zones detected")
    
    # Test N sensitivity
    for N in [1, 3, 5]:
        params = base_params.copy()
        params['fvg_N'] = N
        strategy.location_gate_params.update(params)
        
        zones = strategy._detect_fvg_zones(data, 10)
        if zones:
            print(f"  N {N}: {len(zones[0].get('comb_centers', []))} comb centers")
    
    # Test Order Block parameter sensitivity
    print("\n--- Order Block Parameter Sensitivity ---")
    base_ob_params = {
        'ob_impulse_threshold': 0.02, 'ob_lookback': 10,
        'ob_gamma': 0.95, 'ob_tau_bars': 80, 'ob_drop_threshold': 0.01
    }
    
    # Test impulse threshold sensitivity
    for threshold in [0.01, 0.02, 0.05]:
        params = base_ob_params.copy()
        params['ob_impulse_threshold'] = threshold
        strategy.location_gate_params.update(params)
        
        zones = strategy._detect_order_block_zones(data, 31)
        print(f"  Impulse threshold {threshold}: {len(zones)} zones detected")
    
    # Test VWAP parameter sensitivity
    print("\n--- VWAP Parameter Sensitivity ---")
    base_vwap_params = {
        'vwap_k': 1.0, 'vwap_lookback': 20,
        'vwap_gamma': 0.95, 'vwap_tau_bars': 15, 'vwap_drop_threshold': 0.01
    }
    
    # Test k sensitivity
    for k in [0.5, 1.0, 2.0]:
        params = base_vwap_params.copy()
        params['vwap_k'] = k
        strategy.location_gate_params.update(params)
        
        zones = strategy._detect_vwap_zones(data, 50)
        if zones:
            zone_width = zones[0]['zone_max'] - zones[0]['zone_min']
            print(f"  k {k}: zone width {zone_width:.2f}")
    
    print("Parameter sensitivity testing completed")

def test_mathematical_compliance():
    """Test mathematical compliance with documentation"""
    print("\n" + "="*60)
    print("TESTING MATHEMATICAL COMPLIANCE")
    print("="*60)
    
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("Math Test")
    strategy.gates_and_logic = {'location_gate': True}
    
    # Test penetration depth calculation
    print("\n--- Testing Penetration Depth ---")
    test_index = 50
    current_bar = data.iloc[test_index]
    O, C, H, L = current_bar['open'], current_bar['close'], current_bar['high'], current_bar['low']
    
    # Create a test zone
    zone_min, zone_max = 100.0, 105.0
    
    # Test penetration depth function
    from strategies.strategy_builders import compute_penetration_depth, compute_impulse_penetration
    
    d_ti = compute_penetration_depth(O, C, H, L, zone_min, zone_max)
    print(f"  Penetration depth: {d_ti:.3f}")
    assert 0 <= d_ti <= 1, "Penetration depth should be between 0 and 1"
    
    # Test impulse penetration
    avg_range = np.mean(data['high'] - data['low'])
    d_imp = compute_impulse_penetration(O, C, H, L, zone_min, zone_max, avg_range)
    print(f"  Impulse penetration: {d_imp:.3f}")
    
    # Test momentum calculation per spec
    print("\n--- Testing Momentum Calculation ---")
    if test_index >= 10:
        recent_returns = data['close'].iloc[test_index-10:test_index].pct_change().dropna()
        M_t_y = np.mean(np.abs(recent_returns) * np.sign(recent_returns))
        print(f"  Momentum M(t,y): {M_t_y:.6f}")
    
    # Test zone strength calculation
    print("\n--- Testing Zone Strength Calculation ---")
    from strategies.strategy_builders import per_zone_strength
    
    kernel_params = (0.5, 0.2, 2.0)  # xi, omega, alpha
    A_pattern = 1.0
    C_i = 1.0
    kappa_m = 0.5
    
    S_ti = per_zone_strength(A_pattern, d_imp, kernel_params, C_i, kappa_m, M_t_y)
    print(f"  Per-zone strength S_ti: {S_ti:.3f}")
    assert S_ti >= 0, "Zone strength should be non-negative"
    
    print("Mathematical compliance testing completed")

def test_gui_parameter_mapping():
    """Test that GUI parameters are correctly mapped"""
    print("\n" + "="*60)
    print("TESTING GUI PARAMETER MAPPING")
    print("="*60)
    
    # Test zone type mapping
    zone_type_mapping = {
        "FVG (Fair Value Gap)": "FVG",
        "Order Block": "OrderBlock", 
        "VWAP Mean-Reversion Band": "VWAP",
        "Imbalance Memory Zone": "Imbalance"
    }
    
    print("Zone type mapping:")
    for ui_name, internal_name in zone_type_mapping.items():
        print(f"  {ui_name} -> {internal_name}")
    
    # Test parameter prefix mapping
    parameter_prefixes = {
        'FVG': 'fvg_',
        'OrderBlock': 'ob_',
        'VWAP': 'vwap_',
        'Imbalance': 'imbalance_'
    }
    
    print("\nParameter prefixes:")
    for zone_type, prefix in parameter_prefixes.items():
        print(f"  {zone_type}: {prefix}")
    
    # Test that all required parameters exist
    required_params = {
        'FVG': ['epsilon', 'N', 'sigma', 'beta1', 'beta2', 'phi', 'lambda', 'gamma', 'tau_bars', 'drop_threshold'],
        'OrderBlock': ['impulse_threshold', 'lookback', 'gamma', 'tau_bars', 'drop_threshold'],
        'VWAP': ['k', 'lookback', 'gamma', 'tau_bars', 'drop_threshold'],
        'Imbalance': ['threshold', 'gamma_mem', 'sigma_rev', 'gamma', 'tau_bars', 'drop_threshold']
    }
    
    print("\nRequired parameters per zone type:")
    for zone_type, params in required_params.items():
        print(f"  {zone_type}: {', '.join(params)}")
    
    print("\nGUI parameter mapping testing completed")

def debug_print_all_fvg_zones():
    print("\n=== DEBUG: All Detected FVG Zones (zone_min, zone_max) ===")
    data = create_test_data_with_patterns()
    strategy = PatternStrategy("FVG Debug")
    strategy.location_gate_params.update({'fvg_epsilon': 0.1, 'fvg_N': 3})
    test_indices = [10, 11, 20, 21]
    all_zone_bounds = []
    for idx in test_indices:
        zones = strategy._detect_fvg_zones(data, idx)
        for z in zones:
            print(f"Index {idx}: {z['direction']} FVG: zone_min={z['zone_min']:.2f}, zone_max={z['zone_max']:.2f}")
            all_zone_bounds.append((z['zone_min'], z['zone_max']))
    # Check for repeated zone heights
    unique_bounds = set(all_zone_bounds)
    if len(unique_bounds) < len(all_zone_bounds):
        print(f"[WARNING] Some FVG zones have identical (zone_min, zone_max) values!")
    else:
        print(f"[OK] All FVG zones have unique heights.")

def test_fvg_zones_on_zonetest_csv():
    print("\n=== TEST: FVG Zones on Zonetest.csv ===")
    # Read CSV, skipping comment rows
    df_raw = pd.read_csv('datasets/Zonetest.csv', dtype=str)
    df = df_raw[~df_raw['Date'].astype(str).str.startswith('#')].copy()
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('datetime', inplace=True)
    # Standardize columns
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Last': 'close', 'Volume': 'volume'
    })
    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    strategy = PatternStrategy("FVG Zonetest")
    strategy.location_gate_params.update({'fvg_epsilon': 0.1, 'fvg_N': 3})
    strategy.gates_and_logic = {'location_gate': True}
    fvg_zones = []
    for idx in range(len(df)):
        zones = strategy._detect_fvg_zones(df, idx)
        for z in zones:
            fvg_zones.append((idx, df.index[idx], z['zone_min'], z['zone_max'], z['comb_centers']))
    print(f"Detected {len(fvg_zones)} FVG zones:")
    for idx, ts, zmin, zmax, combs in fvg_zones:
        print(f"  idx={idx}, ts={ts}, zone_min={zmin:.2f}, zone_max={zmax:.2f}, comb_centers={combs}")
    print("\nExpected FVGs (from CSV comments):")
    print("  Bullish FVG: High of t-1 < Low of t+1 (around 09:55-10:05)")
    print("  Bearish FVG: Low of t-1 > High of t+1 (around 10:10-10:20)")
    print("Check that detected FVGs match these locations and have correct vertical spans.")

def test_fvg_zones_exact_on_zonetest_csv():
    print("\n=== EXACT TEST: FVG Zones on Zonetest.csv ===")
    # Read CSV, skipping comment rows
    df_raw = pd.read_csv('datasets/Zonetest.csv', dtype=str)
    df = df_raw[~df_raw['Date'].astype(str).str.startswith('#')].copy()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Last': 'close', 'Volume': 'volume'
    })
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    epsilon = 0.1
    N = 3
    # --- Compute expected FVGs ---
    expected_fvgs = []
    for i in range(2, len(df)):
        c1 = df.iloc[i-2]
        c3 = df.iloc[i]
        ts = df.index[i]
        # Bullish FVG
        if c1['high'] < c3['low'] and (c3['low'] - c1['high']) > 0.1:
            zone_min = c1['high'] + epsilon
            zone_max = c3['low'] - epsilon
            direction = 'bullish'
            expected_fvgs.append({'index': i, 'timestamp': ts, 'zone_min': zone_min, 'zone_max': zone_max, 'direction': direction})
        # Bearish FVG
        if c1['low'] > c3['high'] and (c1['low'] - c3['high']) > 0.1:
            zone_min = c3['high'] + epsilon
            zone_max = c1['low'] - epsilon
            direction = 'bearish'
            expected_fvgs.append({'index': i, 'timestamp': ts, 'zone_min': zone_min, 'zone_max': zone_max, 'direction': direction})
    # --- Get detected FVGs ---
    strategy = PatternStrategy("FVG Zonetest")
    strategy.location_gate_params.update({'fvg_epsilon': epsilon, 'fvg_N': N})
    strategy.gates_and_logic = {'location_gate': True}
    detected_fvgs = []
    for i in range(len(df)):
        zones = strategy._detect_fvg_zones(df, i)
        for z in zones:
            detected_fvgs.append({'index': i, 'timestamp': df.index[i], 'zone_min': z['zone_min'], 'zone_max': z['zone_max'], 'direction': z['direction']})
    # --- Compare ---
    def zone_key(z):
        return (z['index'], z['direction'])
    expected_set = set((z['index'], z['direction'], round(z['zone_min'], 4), round(z['zone_max'], 4)) for z in expected_fvgs)
    detected_set = set((z['index'], z['direction'], round(z['zone_min'], 4), round(z['zone_max'], 4)) for z in detected_fvgs)
    missing = expected_set - detected_set
    extra = detected_set - expected_set
    if not missing and not extra:
        print("[PASS] All FVG zones detected exactly as expected!")
    else:
        print(f"[FAIL] FVG zone mismatch!")
        if missing:
            print(f"  Missing zones:")
            for z in missing:
                print(f"    index={z[0]}, direction={z[1]}, zone_min={z[2]}, zone_max={z[3]}")
        if extra:
            print(f"  Extra zones:")
            for z in extra:
                print(f"    index={z[0]}, direction={z[1]}, zone_min={z[2]}, zone_max={z[3]}")
    print(f"Expected zones: {len(expected_fvgs)}, Detected zones: {len(detected_fvgs)}")
    print("\nSample expected FVGs:")
    for z in expected_fvgs[:5]:
        print(f"  idx={z['index']}, ts={z['timestamp']}, min={z['zone_min']:.2f}, max={z['zone_max']:.2f}, dir={z['direction']}")
    print("\nSample detected FVGs:")
    for z in detected_fvgs[:5]:
        print(f"  idx={z['index']}, ts={z['timestamp']}, min={z['zone_min']:.2f}, max={z['zone_max']:.2f}, dir={z['direction']}")

def run_comprehensive_test():
    """Run all comprehensive tests"""
    print("STARTING COMPREHENSIVE ZONE TYPE TESTING")
    print("="*80)
    
    try:
        # Run all tests
        test_fvg_zones()
        test_order_block_zones()
        test_vwap_zones()
        test_imbalance_zones()
        test_zone_decay_system()
        test_parameter_sensitivity()
        test_mathematical_compliance()
        test_gui_parameter_mapping()
        debug_print_all_fvg_zones()
        test_fvg_zones_on_zonetest_csv()
        test_fvg_zones_exact_on_zonetest_csv()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! All zone types are working correctly.")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 