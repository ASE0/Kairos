"""
Logic Validator
==============
Comprehensive validation system that analyzes backtest results and validates
strategy logic against expected patterns from test datasets.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class LogicValidator:
    def __init__(self):
        self.validation_results = {}
        
    def extract_trade_signals(self, backtest_results: Dict) -> List[Dict]:
        """Extract trade signals from backtest results"""
        signals = []
        
        try:
            # Extract trades from backtest results
            if 'trades' in backtest_results:
                for trade in backtest_results['trades']:
                    signals.append({
                        'entry_time': trade.get('entry_time'),
                        'exit_time': trade.get('exit_time'),
                        'entry_price': trade.get('entry_price'),
                        'exit_price': trade.get('exit_price'),
                        'direction': trade.get('direction', 'UNKNOWN'),
                        'entry_bar': trade.get('entry_bar'),
                        'exit_bar': trade.get('exit_bar'),
                        'pnl': trade.get('pnl', 0)
                    })
            else:
                print("No trades found in backtest results")
                
        except Exception as e:
            print(f"Error extracting trade signals: {e}")
            
        return signals
        
    def analyze_vwap_logic(self, signals: List[Dict], metadata: Dict, dataset: pd.DataFrame) -> Dict:
        """Analyze VWAP strategy logic"""
        print("Analyzing VWAP strategy logic...")
        
        validation = {
            "strategy": "vwap",
            "timestamp": datetime.now().isoformat(),
            "patterns_validated": {},
            "overall_score": 0,
            "total_patterns": len(metadata.get("patterns", [])),
            "validated_patterns": 0
        }
        
        # Calculate VWAP for the dataset
        if 'volume' in dataset.columns:
            vwap = (dataset['close'] * dataset['volume']).cumsum() / dataset['volume'].cumsum()
        else:
            # Fallback if no volume data
            vwap = dataset['close'].rolling(window=20).mean()
            
        # Analyze each expected pattern
        for pattern in metadata.get("patterns", []):
            pattern_type = pattern.get("type", "unknown")
            expected_entry = pattern.get("expected_entry", "UNKNOWN")
            start_bar = pattern.get("start_bar", 0)
            end_bar = pattern.get("end_bar", 0)
            
            # Find signals in the expected range
            pattern_signals = []
            for signal in signals:
                if signal.get('entry_bar') is not None:
                    if start_bar <= signal['entry_bar'] <= end_bar:
                        pattern_signals.append(signal)
                        
            # Validate the pattern
            pattern_validation = self.validate_vwap_pattern(
                pattern_type, expected_entry, pattern_signals, 
                dataset, vwap, start_bar, end_bar
            )
            
            validation["patterns_validated"][pattern_type] = pattern_validation
            
            if pattern_validation["status"] == "validated":
                validation["validated_patterns"] += 1
                
        # Calculate overall score
        if validation["total_patterns"] > 0:
            validation["overall_score"] = validation["validated_patterns"] / validation["total_patterns"]
            
        return validation
        
    def validate_vwap_pattern(self, pattern_type: str, expected_entry: str, 
                            signals: List[Dict], dataset: pd.DataFrame, 
                            vwap: pd.Series, start_bar: int, end_bar: int) -> Dict:
        """Validate a specific VWAP pattern"""
        
        validation = {
            "pattern_type": pattern_type,
            "expected_entry": expected_entry,
            "expected_range": f"bars {start_bar}-{end_bar}",
            "signals_found": len(signals),
            "status": "failed",
            "notes": []
        }
        
        # Check if we found any signals in the expected range
        if len(signals) == 0:
            validation["notes"].append("No signals found in expected range")
            return validation
            
        # Validate signal direction
        correct_direction = 0
        for signal in signals:
            if signal.get('direction') == expected_entry:
                correct_direction += 1
                
        if correct_direction > 0:
            validation["status"] = "validated"
            validation["notes"].append(f"Found {correct_direction} correct {expected_entry} signals")
        else:
            validation["notes"].append(f"Expected {expected_entry} signals, found different directions")
            
        # Additional VWAP-specific validation
        if pattern_type == "reversion_down":
            # Check if price was above VWAP before reverting
            price_above_vwap = dataset.iloc[start_bar:end_bar]['close'] > vwap.iloc[start_bar:end_bar]
            if price_above_vwap.any():
                validation["notes"].append("Price correctly above VWAP before reversion")
            else:
                validation["notes"].append("Price not consistently above VWAP")
                
        elif pattern_type == "reversion_up":
            # Check if price was below VWAP before reverting
            price_below_vwap = dataset.iloc[start_bar:end_bar]['close'] < vwap.iloc[start_bar:end_bar]
            if price_below_vwap.any():
                validation["notes"].append("Price correctly below VWAP before reversion")
            else:
                validation["notes"].append("Price not consistently below VWAP")
                
        elif pattern_type == "support_bounce":
            # Check if VWAP acted as support
            vwap_support = dataset.iloc[start_bar:end_bar]['low'] >= vwap.iloc[start_bar:end_bar]
            if vwap_support.any():
                validation["notes"].append("VWAP correctly acted as support")
            else:
                validation["notes"].append("VWAP did not act as support")
                
        return validation
        
    def analyze_order_block_logic(self, signals: List[Dict], metadata: Dict, dataset: pd.DataFrame) -> Dict:
        """Analyze Order Block strategy logic"""
        print("Analyzing Order Block strategy logic...")
        
        validation = {
            "strategy": "order_block",
            "timestamp": datetime.now().isoformat(),
            "patterns_validated": {},
            "overall_score": 0,
            "total_patterns": len(metadata.get("patterns", [])),
            "validated_patterns": 0
        }
        
        # Analyze each expected pattern
        for pattern in metadata.get("patterns", []):
            pattern_type = pattern.get("type", "unknown")
            expected_entry = pattern.get("expected_entry", "UNKNOWN")
            start_bar = pattern.get("start_bar", 0)
            end_bar = pattern.get("end_bar", 0)
            
            # Find signals in the expected range
            pattern_signals = []
            for signal in signals:
                if signal.get('entry_bar') is not None:
                    if start_bar <= signal['entry_bar'] <= end_bar:
                        pattern_signals.append(signal)
                        
            # Validate the pattern
            pattern_validation = self.validate_order_block_pattern(
                pattern_type, expected_entry, pattern_signals, 
                dataset, start_bar, end_bar
            )
            
            validation["patterns_validated"][pattern_type] = pattern_validation
            
            if pattern_validation["status"] == "validated":
                validation["validated_patterns"] += 1
                
        # Calculate overall score
        if validation["total_patterns"] > 0:
            validation["overall_score"] = validation["validated_patterns"] / validation["total_patterns"]
            
        return validation
        
    def validate_order_block_pattern(self, pattern_type: str, expected_entry: str,
                                   signals: List[Dict], dataset: pd.DataFrame,
                                   start_bar: int, end_bar: int) -> Dict:
        """Validate a specific Order Block pattern"""
        
        validation = {
            "pattern_type": pattern_type,
            "expected_entry": expected_entry,
            "expected_range": f"bars {start_bar}-{end_bar}",
            "signals_found": len(signals),
            "status": "failed",
            "notes": []
        }
        
        # Check if we found any signals in the expected range
        if len(signals) == 0:
            validation["notes"].append("No signals found in expected range")
            return validation
            
        # Validate signal direction
        correct_direction = 0
        for signal in signals:
            if signal.get('direction') == expected_entry:
                correct_direction += 1
                
        if correct_direction > 0:
            validation["status"] = "validated"
            validation["notes"].append(f"Found {correct_direction} correct {expected_entry} signals")
        else:
            validation["notes"].append(f"Expected {expected_entry} signals, found different directions")
            
        # Additional Order Block-specific validation
        if pattern_type == "bullish_order_block":
            # Check for high volume and strong move before retracement
            volume_surge = dataset.iloc[start_bar-20:start_bar]['volume'].mean() > dataset.iloc[start_bar:end_bar]['volume'].mean() * 1.5
            if volume_surge:
                validation["notes"].append("Correctly identified bullish order block with volume surge")
            else:
                validation["notes"].append("Volume surge not detected in bullish order block")
                
        elif pattern_type == "bearish_order_block":
            # Check for high volume and strong move down before retracement
            volume_surge = dataset.iloc[start_bar-20:start_bar]['volume'].mean() > dataset.iloc[start_bar:end_bar]['volume'].mean() * 1.5
            if volume_surge:
                validation["notes"].append("Correctly identified bearish order block with volume surge")
            else:
                validation["notes"].append("Volume surge not detected in bearish order block")
                
        return validation
        
    def analyze_fvg_logic(self, signals: List[Dict], metadata: Dict, dataset: pd.DataFrame) -> Dict:
        """Analyze Fair Value Gap strategy logic"""
        print("Analyzing FVG strategy logic...")
        
        validation = {
            "strategy": "fvg",
            "timestamp": datetime.now().isoformat(),
            "patterns_validated": {},
            "overall_score": 0,
            "total_patterns": len(metadata.get("patterns", [])),
            "validated_patterns": 0
        }
        
        # Analyze each expected pattern
        for pattern in metadata.get("patterns", []):
            pattern_type = pattern.get("type", "unknown")
            expected_entry = pattern.get("expected_entry", "UNKNOWN")
            start_bar = pattern.get("start_bar", 0)
            end_bar = pattern.get("end_bar", 0)
            
            # Find signals in the expected range
            pattern_signals = []
            for signal in signals:
                if signal.get('entry_bar') is not None:
                    if start_bar <= signal['entry_bar'] <= end_bar:
                        pattern_signals.append(signal)
                        
            # Validate the pattern
            pattern_validation = self.validate_fvg_pattern(
                pattern_type, expected_entry, pattern_signals, 
                dataset, start_bar, end_bar
            )
            
            validation["patterns_validated"][pattern_type] = pattern_validation
            
            if pattern_validation["status"] == "validated":
                validation["validated_patterns"] += 1
                
        # Calculate overall score
        if validation["total_patterns"] > 0:
            validation["overall_score"] = validation["validated_patterns"] / validation["total_patterns"]
            
        return validation
        
    def validate_fvg_pattern(self, pattern_type: str, expected_entry: str,
                           signals: List[Dict], dataset: pd.DataFrame,
                           start_bar: int, end_bar: int) -> Dict:
        """Validate a specific FVG pattern"""
        
        validation = {
            "pattern_type": pattern_type,
            "expected_entry": expected_entry,
            "expected_range": f"bars {start_bar}-{end_bar}",
            "signals_found": len(signals),
            "status": "failed",
            "notes": []
        }
        
        # Check if we found any signals in the expected range
        if len(signals) == 0:
            validation["notes"].append("No signals found in expected range")
            return validation
            
        # Validate signal direction
        correct_direction = 0
        for signal in signals:
            if signal.get('direction') == expected_entry:
                correct_direction += 1
                
        if correct_direction > 0:
            validation["status"] = "validated"
            validation["notes"].append(f"Found {correct_direction} correct {expected_entry} signals")
        else:
            validation["notes"].append(f"Expected {expected_entry} signals, found different directions")
            
        # Additional FVG-specific validation
        if pattern_type == "bullish_fvg":
            # Check for gap up before retracement
            gap_up = dataset.iloc[start_bar-5:start_bar]['high'].max() < dataset.iloc[start_bar:start_bar+5]['low'].min()
            if gap_up:
                validation["notes"].append("Correctly identified bullish FVG with gap up")
            else:
                validation["notes"].append("Gap up not detected in bullish FVG")
                
        elif pattern_type == "bearish_fvg":
            # Check for gap down before retracement
            gap_down = dataset.iloc[start_bar-5:start_bar]['low'].min() > dataset.iloc[start_bar:start_bar+5]['high'].max()
            if gap_down:
                validation["notes"].append("Correctly identified bearish FVG with gap down")
            else:
                validation["notes"].append("Gap down not detected in bearish FVG")
                
        return validation
        
    def analyze_support_resistance_logic(self, signals: List[Dict], metadata: Dict, dataset: pd.DataFrame) -> Dict:
        """Analyze Support/Resistance strategy logic"""
        print("Analyzing Support/Resistance strategy logic...")
        
        validation = {
            "strategy": "support_resistance",
            "timestamp": datetime.now().isoformat(),
            "patterns_validated": {},
            "overall_score": 0,
            "total_patterns": len(metadata.get("patterns", [])),
            "validated_patterns": 0
        }
        
        # Analyze each expected pattern
        for pattern in metadata.get("patterns", []):
            pattern_type = pattern.get("type", "unknown")
            expected_entry = pattern.get("expected_entry", "UNKNOWN")
            start_bar = pattern.get("start_bar", 0)
            end_bar = pattern.get("end_bar", 0)
            
            # Find signals in the expected range
            pattern_signals = []
            for signal in signals:
                if signal.get('entry_bar') is not None:
                    if start_bar <= signal['entry_bar'] <= end_bar:
                        pattern_signals.append(signal)
                        
            # Validate the pattern
            pattern_validation = self.validate_support_resistance_pattern(
                pattern_type, expected_entry, pattern_signals, 
                dataset, start_bar, end_bar
            )
            
            validation["patterns_validated"][pattern_type] = pattern_validation
            
            if pattern_validation["status"] == "validated":
                validation["validated_patterns"] += 1
                
        # Calculate overall score
        if validation["total_patterns"] > 0:
            validation["overall_score"] = validation["validated_patterns"] / validation["total_patterns"]
            
        return validation
        
    def validate_support_resistance_pattern(self, pattern_type: str, expected_entry: str,
                                          signals: List[Dict], dataset: pd.DataFrame,
                                          start_bar: int, end_bar: int) -> Dict:
        """Validate a specific Support/Resistance pattern"""
        
        validation = {
            "pattern_type": pattern_type,
            "expected_entry": expected_entry,
            "expected_range": f"bars {start_bar}-{end_bar}",
            "signals_found": len(signals),
            "status": "failed",
            "notes": []
        }
        
        # Check if we found any signals in the expected range
        if len(signals) == 0:
            validation["notes"].append("No signals found in expected range")
            return validation
            
        # Validate signal direction
        correct_direction = 0
        for signal in signals:
            if signal.get('direction') == expected_entry:
                correct_direction += 1
                
        if correct_direction > 0:
            validation["status"] = "validated"
            validation["notes"].append(f"Found {correct_direction} correct {expected_entry} signals")
        else:
            validation["notes"].append(f"Expected {expected_entry} signals, found different directions")
            
        # Additional Support/Resistance-specific validation
        if pattern_type == "resistance_bounce":
            # Check for resistance level test
            resistance_test = dataset.iloc[start_bar:end_bar]['high'].max() >= dataset.iloc[start_bar-20:start_bar]['high'].max() * 0.98
            if resistance_test:
                validation["notes"].append("Correctly identified resistance level test")
            else:
                validation["notes"].append("Resistance level test not detected")
                
        elif pattern_type == "support_bounce":
            # Check for support level test
            support_test = dataset.iloc[start_bar:end_bar]['low'].min() <= dataset.iloc[start_bar-20:start_bar]['low'].min() * 1.02
            if support_test:
                validation["notes"].append("Correctly identified support level test")
            else:
                validation["notes"].append("Support level test not detected")
                
        elif pattern_type == "breakout_failure":
            # Check for failed breakout
            breakout_attempt = dataset.iloc[start_bar:start_bar+10]['high'].max() > dataset.iloc[start_bar-20:start_bar]['high'].max()
            if breakout_attempt:
                validation["notes"].append("Correctly identified breakout attempt")
            else:
                validation["notes"].append("Breakout attempt not detected")
                
        return validation
        
    def validate_strategy_logic(self, strategy_name: str, backtest_results: Dict, 
                              metadata: Dict, dataset: pd.DataFrame) -> Dict:
        """Main validation function for any strategy"""
        print(f"Validating {strategy_name} strategy logic...")
        
        # Extract trade signals
        signals = self.extract_trade_signals(backtest_results)
        
        # Route to appropriate validation function
        if strategy_name == "vwap":
            return self.analyze_vwap_logic(signals, metadata, dataset)
        elif strategy_name == "order_block":
            return self.analyze_order_block_logic(signals, metadata, dataset)
        elif strategy_name == "fvg":
            return self.analyze_fvg_logic(signals, metadata, dataset)
        elif strategy_name == "support_resistance":
            return self.analyze_support_resistance_logic(signals, metadata, dataset)
        else:
            return {
                "strategy": strategy_name,
                "timestamp": datetime.now().isoformat(),
                "status": "unknown_strategy",
                "notes": [f"Unknown strategy: {strategy_name}"]
            }

def main():
    """Test the logic validator"""
    validator = LogicValidator()
    print("Logic Validator initialized successfully!")

if __name__ == "__main__":
    main() 