"""
Trading Strategy Hub - Central Application
==========================================
A comprehensive GUI for building, testing, and combining trading strategies
"""

import sys
import argparse
print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHONPATH:", sys.path)
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from scipy import stats
import logging

print('=== main.py executed ===')

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

RECENT_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'recent_dataset', 'most_recent.csv')

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Strategy Hub')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--strategy', type=str, help='Strategy configuration file')
    parser.add_argument('--data', type=str, help='Data file path')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.headless:
        return run_headless(args)
    
    # Import PyQt6 only when not in headless mode
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QFont
    import pyqtgraph as pg
    
    app = QApplication(sys.argv)

    # High DPI scaling is enabled by default in PyQt6
    # No need to set AA_EnableHighDpiScaling

    # Set application style
    app.setStyle('Fusion')

    # Import and create main window
    from gui.main_hub import TradingStrategyHub
    hub = TradingStrategyHub()
    # Auto-load most recent dataset if it exists
    if os.path.exists(RECENT_DATASET_PATH):
        try:
            df = pd.read_csv(RECENT_DATASET_PATH)
            # PATCH: Use both Date and Time columns for unique timestamp index if available
            from PyQt6.QtWidgets import QMessageBox
            debug_msg = ''
            if 'Date' in df.columns and 'Time' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='raise')
                    df.index = df['datetime']
                    debug_msg = "Set index to combined 'Date' and 'Time' columns."
                except Exception as e:
                    debug_msg = f"Failed to parse 'Date' and 'Time' columns: {e}\nUsing synthetic index."
                    df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
            elif 'datetime' in df.columns:
                df.index = pd.to_datetime(df['datetime'])
                debug_msg = "Set index to pd.to_datetime('datetime') column."
            elif 'Date' in df.columns:
                try:
                    df.index = pd.to_datetime(df['Date'], errors='raise')
                    debug_msg = "Set index to 'Date' column."
                except Exception as e:
                    debug_msg = f"Failed to parse 'Date' column: {e}\nUsing synthetic index."
                    df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
            elif 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
                debug_msg = "Set index to 'timestamp' column."
            else:
                df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='T')
                debug_msg = "No datetime column found, using synthetic date range index."
            # Debug: Show index info
            debug_msg += f"\nIndex type: {type(df.index)}\nNum rows: {len(df)}\nNum unique index: {df.index.nunique()}\nIndex min: {df.index.min()}\nIndex max: {df.index.max()}\nFirst 10: {list(df.index[:10])}"
            print(debug_msg)
            # QMessageBox.information(None, "Dataset Index Debug", debug_msg)
            
            # PATCH: Handle duplicate indices by aggregating OHLC data
            if not df.index.is_unique:
                print(f"[PATCH] Found duplicate indices. Aggregating OHLC data...")
                before_agg = len(df)
                # Aggregate OHLC data for duplicate timestamps
                df = df.groupby(df.index).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                })
                after_agg = len(df)
                print(f"[PATCH] Aggregated {before_agg} -> {after_agg} bars")
            
            # Map columns to standard OHLCV names for downstream compatibility
            col_map = {
                'open': None, 'high': None, 'low': None, 'close': None, 'volume': None
            }
            for col in df.columns:
                lcol = col.lower()
                if lcol.startswith('open'):
                    col_map['open'] = col
                elif lcol.startswith('high'):
                    col_map['high'] = col
                elif lcol.startswith('low'):
                    col_map['low'] = col
                elif lcol.startswith('close') or lcol.startswith('last'):
                    col_map['close'] = col
                elif lcol.startswith('vol'):
                    col_map['volume'] = col
            # Only keep and rename if all required columns are present
            if all(col_map.values()):
                df = df[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
            # Add to hub.datasets and set as active
            hub.datasets = getattr(hub, 'datasets', {})
            hub.datasets['Most Recent'] = {'data': df, 'metadata': {}}
            hub.last_loaded_dataset = 'Most Recent'
            print('[INFO] Auto-loaded most recent dataset.')
        except Exception as e:
            print(f'[WARNING] Could not auto-load most recent dataset: {e}')
    hub.show()

    sys.exit(app.exec())


def run_headless(args):
    """Run the application in headless mode for testing"""
    try:
        try:
            with open('headless_test.txt', 'w') as f:
                f.write("run_headless function called\n")
                f.write(f"Args: {args}\n")
        except Exception as e:
            print(f"[DEBUG] Failed to write headless_test.txt: {e}")
        
        # Load data
        if not args.data or not os.path.exists(args.data):
            print(f"Error: Data file not found: {args.data}")
            return 1
        
        df = pd.read_csv(args.data)
        df.reset_index(drop=True, inplace=True)
        # Set datetime index if needed for plotting, but keep RangeIndex for detection
        # if 'datetime' in df.columns:
        #     df['datetime'] = pd.to_datetime(df['datetime'])
        #     df.set_index('datetime', inplace=True)
        
        # Map columns to standard names
        col_map = {
            'open': None, 'high': None, 'low': None, 'close': None, 'volume': None
        }
        for col in df.columns:
            lcol = col.lower()
            if lcol.startswith('open'):
                col_map['open'] = col
            elif lcol.startswith('high'):
                col_map['high'] = col
            elif lcol.startswith('low'):
                col_map['low'] = col
            elif lcol.startswith('close') or lcol.startswith('last'):
                col_map['close'] = col
            elif lcol.startswith('vol'):
                col_map['volume'] = col
        
        if all(col_map.values()):
            df = df[[col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Load strategy
        if not args.strategy or not os.path.exists(args.strategy):
            print(f"Error: Strategy file not found: {args.strategy}")
            return 1
        strategy_config = None
        if args.strategy.endswith('.dill'):
            import dill
            with open(args.strategy, 'rb') as f:
                strategy_config = dill.load(f)
            # Convert to dict if needed
            if hasattr(strategy_config, 'to_dict'):
                strategy_config = strategy_config.to_dict()
            elif hasattr(strategy_config, '__dict__'):
                strategy_config = dict(strategy_config.__dict__)
        else:
            with open(args.strategy, 'r', encoding='utf-8') as f:
                strategy_config = json.load(f)
        
        # Run backtest
        from strategies.strategy_builders import MultiTimeframeBacktestEngine
        from core.data_structures import TimeRange
        
        engine = MultiTimeframeBacktestEngine()
        
        # Create strategy from config
        strategy_name = strategy_config.get('name', 'TestStrategy')
        actions = strategy_config.get('actions', [])
        
        # --- PATCH: Always use MultiTimeframeBacktestEngine for OB strategies ---
        is_ob_strategy = strategy_name and 'ob' in strategy_name.lower() or any(
            (a.get('location_strategy', '') and a.get('location_strategy', '').lower() in ('order block', 'orderblock')) for a in actions
        )
        if is_ob_strategy:
            from strategies.strategy_builders import PatternStrategy, Action
            # Reconstruct actions as Action objects
            action_objs = []
            for a in actions:
                action_objs.append(Action(
                    name=a.get('name', ''),
                    pattern=None,
                    time_range=a.get('time_range'),
                    location_strategy=a.get('location_strategy'),
                    location_params=a.get('location_params', {}),
                    filters=a.get('filters', [])
                ))
            strategy_obj = PatternStrategy(
                actions=action_objs,
                combination_logic=strategy_config.get('combination_logic', 'AND'),
                weights=strategy_config.get('weights'),
                min_actions_required=strategy_config.get('min_actions_required', 1),
                gates_and_logic=strategy_config.get('gates_and_logic', {'location_gate': True}),
                location_gate_params=strategy_config.get('location_gate_params', {})
            )
            results = engine.run_backtest(strategy_obj, df)
        else:
            # For now, just run a simple pattern detection
            results = {
                'strategy_name': strategy_name,
                'data_length': len(df),
                'signals': [],
                'trades': [],
                'zones': []
            }
            
            # Detect patterns based on strategy config
            for action in actions:
                pattern_name = action.get('name', 'unknown')
                pattern_config = action.get('pattern', {})
                if pattern_config is None:
                    pattern_config = {}
                try:
                    if pattern_name == 'fvg':
                        from patterns.enhanced_candlestick_patterns import FVGPattern
                        print(f"[DEBUG] FVG: DataFrame shape before detection: {df.shape}")
                        pattern = FVGPattern(**pattern_config.get('params', {}))
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                        zones = pattern.detect_zones(df)
                        results['zones'] = zones
                    elif pattern_name == 'engulfing':
                        from patterns.candlestick_patterns import EngulfingPattern
                        pattern = EngulfingPattern([TimeRange(1, 'min')], **pattern_config.get('params', {}))
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'hammer':
                        from patterns.candlestick_patterns import HammerPattern
                        pattern = HammerPattern([TimeRange(1, 'min')], **pattern_config.get('params', {}))
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'double_wick':
                        from patterns.candlestick_patterns import DoubleWickPattern
                        pattern = DoubleWickPattern([TimeRange(1, 'min')], **pattern_config.get('params', {}))
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'ii_bars':
                        from patterns.candlestick_patterns import IIBarsPattern
                        pattern = IIBarsPattern([TimeRange(1, 'min')], **pattern_config.get('params', {}))
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'doji':
                        from patterns.enhanced_candlestick_patterns import CustomParametricPattern, PredefinedPatterns, TimeRange
                        # Use custom parameters if provided, otherwise use default doji parameters
                        if pattern_config.get('params'):
                            # Create PatternParameters from custom params
                            from patterns.enhanced_candlestick_patterns import PatternParameters, PatternDirection
                            custom_params = pattern_config.get('params', {})
                            pattern_params = PatternParameters(
                                min_body_ratio=custom_params.get('min_body_ratio'),
                                max_body_ratio=custom_params.get('max_body_ratio'),
                                allowed_directions=[PatternDirection(d) for d in custom_params.get('allowed_directions', ['neutral'])]
                            )
                            pattern = CustomParametricPattern('CustomDoji', pattern_params)
                        else:
                            pattern = CustomParametricPattern('Doji', PredefinedPatterns.doji())
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'weak_body':
                        from core.pattern_registry import PatternRegistry
                        registry = PatternRegistry()
                        pattern = registry.create_pattern('weak_body')
                        print(f"[DEBUG] Created weak_body pattern: {pattern}")
                        print(f"[DEBUG] Pattern type: {type(pattern)}")
                        print(f"[DEBUG] Pattern ohlc_ratios: {pattern.ohlc_ratios}")
                        signals = pattern.detect(df)
                        print(f"[DEBUG] Weak body signals: {signals.tolist()}")
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'marubozu':
                        from core.pattern_registry import PatternRegistry
                        registry = PatternRegistry()
                        pattern = registry.create_pattern('marubozu')
                        print(f"[DEBUG] Created marubozu pattern: {pattern}")
                        print(f"[DEBUG] Pattern type: {type(pattern)}")
                        print(f"[DEBUG] Pattern ohlc_ratios: {getattr(pattern, 'ohlc_ratios', None)}")
                        signals = pattern.detect(df)
                        print(f"[DEBUG] Marubozu signals: {signals.tolist()}")
                        results['signals'] = signals.tolist()
                    elif pattern_name == 'spinning_top':
                        from patterns.enhanced_candlestick_patterns import CustomParametricPattern, PredefinedPatterns, TimeRange
                        pattern = CustomParametricPattern('SpinningTop', PredefinedPatterns.spinning_top())
                        signals = pattern.detect(df)
                        results['signals'] = signals.tolist()
                    # PATCH: Recognize both 'Order Block' and 'order_block_gate' as OB action names
                    # --- BEGIN REMOVAL: Order Block Gate logic ---
                    # elif pattern_name in ('order_block_gate', 'Order Block') or action.get('location_strategy') in ('Order Block', 'order_block_gate', 'OrderBlock'):
                    #     from core.order_block_gate import OrderBlockGate
                    #     bars = df.to_dict(orient='records')
                    #     params = pattern_config.get('params', {})
                    #     zones = OrderBlockGate.detect_zones(bars, **params)
                    #     results['zones'] = zones
                    #     signals = np.zeros(len(df), dtype=bool)
                    #     for z in zones:
                    #         idx = z.get('bar_index')
                    #         if idx is not None and 0 <= idx < len(signals):
                    #             signals[idx] = True
                    #     results['signals'] = signals.tolist()
                    # --- END REMOVAL: Order Block Gate logic ---
                    # --- BEGIN PATCH: Imbalance Memory Zone ---
                    elif pattern_name == 'imbalance_memory_zone':
                        # Use the PatternStrategy to detect Imbalance Memory Zone
                        from strategies.strategy_builders import PatternStrategy, Action
                        from core.data_structures import TimeRange
                        # Create a strategy with Imbalance detection
                        action = Action(
                            name="Imbalance Memory Zone",
                            location_strategy="Imbalance Memory Zone",
                            location_params={
                                'imbalance_threshold': pattern_config.get('params', {}).get('imbalance_threshold', 100),
                                'imbalance_gamma_mem': pattern_config.get('params', {}).get('imbalance_gamma_mem', 0.01),
                                'imbalance_sigma_rev': pattern_config.get('params', {}).get('imbalance_sigma_rev', 20),
                                'imbalance_gamma': pattern_config.get('params', {}).get('imbalance_gamma', 0.95),
                                'imbalance_tau_bars': pattern_config.get('params', {}).get('imbalance_tau_bars', 100),
                                'imbalance_drop_threshold': pattern_config.get('params', {}).get('imbalance_drop_threshold', 0.01)
                            }
                        )
                        strategy = PatternStrategy(actions=[action], gates_and_logic={'location_gate': True})
                        # Detect zones at each bar
                        zones = []
                        signals = np.zeros(len(df), dtype=bool)
                        for i in range(len(df)):
                            if i >= 10:  # Need enough history for Imbalance detection
                                imb_zones = strategy._detect_imbalance_zones(df, i)
                                if imb_zones:
                                    zones.extend(imb_zones)
                                    for zone in imb_zones:
                                        creation_index = zone.get('creation_index', i)
                                        if 0 <= creation_index < len(signals):
                                            signals[creation_index] = True
                        results['zones'] = zones
                        results['signals'] = signals.tolist()
                        print(f"[DEBUG] Imbalance detection: found {len(zones)} zones, {signals.sum()} signals")
                    # --- END PATCH: Imbalance Memory Zone ---
                except Exception as e:
                    results['error'] = str(e)
                    import traceback
                    results['traceback'] = traceback.format_exc()
        
        # DEBUG: Write actions list to file for inspection
        with open('headless_debug.txt', 'w') as f:
            f.write(f"Strategy config: {strategy_config}\n")
            f.write(f"Actions: {actions}\n")
            f.write(f"Data shape: {df.shape}\n")
            f.write(f"Data columns: {list(df.columns)}\n")
            if len(df) > 5:
                bar = df.iloc[5]
                f.write(f"Bar at index 5: {dict(bar)}\n")

        # DEBUG: Write hammer bar and features to file if hammer pattern
        if any(action.get('name') == 'hammer' for action in actions):
            if len(df) > 5:
                bar = df.iloc[5]
                open_, high_, low_, close_ = bar['open'], bar['high'], bar['low'], bar['close']
                body = abs(close_ - open_)
                upper = high_ - max(open_, close_)
                lower = min(open_, close_) - low_
                total_range = high_ - low_
                with open('headless_debug.txt', 'a') as f:
                    f.write(f"Hammer bar at idx 5: open={open_}, high={high_}, low={low_}, close={close_}\n")
                    f.write(f"Features: body={body}, upper={upper}, lower={lower}, total_range={total_range}\n")
        
        # Save results
        if args.output:
            # PATCH: Ensure all zones have 'zone_type' and 'zone_direction' fields for GUI compatibility
            if 'zones' in results and isinstance(results['zones'], list):
                for z in results['zones']:
                    if 'type' in z and 'zone_type' not in z:
                        z['zone_type'] = z['type']
                    if 'zone_type' in z and 'type' not in z:
                        z['type'] = z['zone_type']
                    if 'direction' not in z and 'zone_direction' in z:
                        z['direction'] = z['zone_direction']
                    if 'zone_direction' not in z and 'direction' in z:
                        z['zone_direction'] = z['direction']
                    if 'direction' not in z:
                        z['direction'] = 'unknown'
                    if 'zone_direction' not in z:
                        z['zone_direction'] = 'unknown'
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error in headless mode: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()