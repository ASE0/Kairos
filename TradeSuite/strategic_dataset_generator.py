"""
Strategic Dataset Generator
==========================
Creates test datasets with specific patterns for each building block strategy.
Each dataset is designed to have at least 3 clear entry/exit opportunities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class StrategicDatasetGenerator:
    def __init__(self):
        self.base_price = 5000.0
        self.tick_size = 0.25
        self.base_volume = 1000
        
    def create_vwap_test_dataset(self):
        """Create dataset with clear VWAP mean reversion opportunities"""
        print("Creating VWAP test dataset...")
        
        # Create 1000 bars with specific VWAP patterns
        bars = []
        current_time = datetime(2024, 1, 1, 9, 30, 0)
        
        # Pattern 1: Price above VWAP, should revert down
        for i in range(200):
            if i < 50:  # Build up above VWAP
                price = self.base_price + 5 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 100:  # Peak above VWAP (entry opportunity)
                price = self.base_price + 8 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
            else:  # Revert back to VWAP (exit opportunity)
                price = self.base_price + 2 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 2: Price below VWAP, should revert up
        for i in range(200):
            if i < 50:  # Build down below VWAP
                price = self.base_price - 5 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 100:  # Bottom below VWAP (entry opportunity)
                price = self.base_price - 8 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
            else:  # Revert back to VWAP (exit opportunity)
                price = self.base_price - 2 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+200),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 3: Strong trend with VWAP as support
        for i in range(200):
            if i < 100:  # Trending up with VWAP support
                price = self.base_price + 10 + i * 0.1 + np.random.normal(0, 0.5)
                volume = self.base_volume * 1.2 + np.random.normal(0, 120)
            else:  # Pullback to VWAP (entry opportunity)
                price = self.base_price + 10 + 100 * 0.1 - (i-100) * 0.05 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.8 + np.random.normal(0, 180)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+400),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        
        # Save dataset
        filename = "test_vwap_dataset.csv"
        df.to_csv(filename)
        
        # Create metadata
        metadata = {
            "strategy": "vwap",
            "description": "VWAP mean reversion test dataset with 3 clear patterns",
            "patterns": [
                {"type": "reversion_down", "start_bar": 50, "end_bar": 100, "expected_entry": "SHORT"},
                {"type": "reversion_up", "start_bar": 250, "end_bar": 300, "expected_entry": "LONG"},
                {"type": "support_bounce", "start_bar": 500, "end_bar": 550, "expected_entry": "LONG"}
            ],
            "total_bars": len(df),
            "created": datetime.now().isoformat()
        }
        
        with open("test_vwap_dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"VWAP test dataset created: {filename}")
        print(f"Expected entries: {len(metadata['patterns'])}")
        return filename
        
    def create_order_block_test_dataset(self):
        """Create dataset with clear order block patterns"""
        print("Creating Order Block test dataset...")
        
        bars = []
        current_time = datetime(2024, 1, 1, 9, 30, 0)
        
        # Pattern 1: Bullish order block (institutional buying)
        for i in range(200):
            if i < 30:  # Normal trading
                price = self.base_price + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 50:  # Order block formation (high volume, strong move)
                price = self.base_price + 5 + np.random.normal(0, 0.3)
                volume = self.base_volume * 3 + np.random.normal(0, 300)
            elif i < 80:  # Retracement to order block (entry opportunity)
                price = self.base_price + 2 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            else:  # Continuation (exit opportunity)
                price = self.base_price + 8 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 2: Bearish order block (institutional selling)
        for i in range(200):
            if i < 30:  # Normal trading
                price = self.base_price + 10 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 50:  # Order block formation (high volume, strong move down)
                price = self.base_price + 2 + np.random.normal(0, 0.3)
                volume = self.base_volume * 3 + np.random.normal(0, 300)
            elif i < 80:  # Retracement to order block (entry opportunity)
                price = self.base_price + 5 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            else:  # Continuation down (exit opportunity)
                price = self.base_price - 2 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+200),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 3: Multiple order blocks in sequence
        for i in range(200):
            if i < 40:  # First order block
                price = self.base_price + 15 + np.random.normal(0, 0.3)
                volume = self.base_volume * 2.5 + np.random.normal(0, 250)
            elif i < 80:  # Retracement to first block (entry opportunity)
                price = self.base_price + 12 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 120:  # Second order block
                price = self.base_price + 18 + np.random.normal(0, 0.3)
                volume = self.base_volume * 2.5 + np.random.normal(0, 250)
            else:  # Final retracement (entry opportunity)
                price = self.base_price + 15 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+400),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        
        filename = "test_order_block_dataset.csv"
        df.to_csv(filename)
        
        metadata = {
            "strategy": "order_block",
            "description": "Order block test dataset with 3 clear institutional patterns",
            "patterns": [
                {"type": "bullish_order_block", "start_bar": 50, "end_bar": 80, "expected_entry": "LONG"},
                {"type": "bearish_order_block", "start_bar": 250, "end_bar": 280, "expected_entry": "SHORT"},
                {"type": "multiple_order_blocks", "start_bar": 480, "end_bar": 520, "expected_entry": "LONG"}
            ],
            "total_bars": len(df),
            "created": datetime.now().isoformat()
        }
        
        with open("test_order_block_dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Order Block test dataset created: {filename}")
        print(f"Expected entries: {len(metadata['patterns'])}")
        return filename
        
    def create_fvg_test_dataset(self):
        """Create dataset with clear Fair Value Gap patterns"""
        print("Creating FVG test dataset...")
        
        bars = []
        current_time = datetime(2024, 1, 1, 9, 30, 0)
        
        # Pattern 1: Bullish FVG (gap up)
        for i in range(200):
            if i < 50:  # Normal trading
                price = self.base_price + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 70:  # Gap up creating FVG
                price = self.base_price + 8 + np.random.normal(0, 0.3)
                volume = self.base_volume * 2 + np.random.normal(0, 200)
            elif i < 100:  # Retracement to fill FVG (entry opportunity)
                price = self.base_price + 4 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            else:  # Continuation up (exit opportunity)
                price = self.base_price + 12 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 2: Bearish FVG (gap down)
        for i in range(200):
            if i < 50:  # Normal trading
                price = self.base_price + 15 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 70:  # Gap down creating FVG
                price = self.base_price + 2 + np.random.normal(0, 0.3)
                volume = self.base_volume * 2 + np.random.normal(0, 200)
            elif i < 100:  # Retracement to fill FVG (entry opportunity)
                price = self.base_price + 6 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            else:  # Continuation down (exit opportunity)
                price = self.base_price - 2 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+200),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 3: Multiple FVGs in sequence
        for i in range(200):
            if i < 40:  # First FVG
                price = self.base_price + 20 + np.random.normal(0, 0.3)
                volume = self.base_volume * 2 + np.random.normal(0, 200)
            elif i < 80:  # Fill first FVG (entry opportunity)
                price = self.base_price + 16 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 120:  # Second FVG
                price = self.base_price + 25 + np.random.normal(0, 0.3)
                volume = self.base_volume * 2 + np.random.normal(0, 200)
            else:  # Fill second FVG (entry opportunity)
                price = self.base_price + 21 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+400),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        
        filename = "test_fvg_dataset.csv"
        df.to_csv(filename)
        
        metadata = {
            "strategy": "fvg",
            "description": "Fair Value Gap test dataset with 3 clear gap patterns",
            "patterns": [
                {"type": "bullish_fvg", "start_bar": 70, "end_bar": 100, "expected_entry": "LONG"},
                {"type": "bearish_fvg", "start_bar": 270, "end_bar": 300, "expected_entry": "SHORT"},
                {"type": "multiple_fvgs", "start_bar": 480, "end_bar": 520, "expected_entry": "LONG"}
            ],
            "total_bars": len(df),
            "created": datetime.now().isoformat()
        }
        
        with open("test_fvg_dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"FVG test dataset created: {filename}")
        print(f"Expected entries: {len(metadata['patterns'])}")
        return filename
        
    def create_support_resistance_test_dataset(self):
        """Create dataset with clear support and resistance levels"""
        print("Creating Support/Resistance test dataset...")
        
        bars = []
        current_time = datetime(2024, 1, 1, 9, 30, 0)
        
        # Pattern 1: Resistance level bounce
        resistance_level = self.base_price + 10
        
        for i in range(200):
            if i < 50:  # Approach resistance
                price = resistance_level - 3 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 80:  # Test resistance (entry opportunity)
                price = resistance_level - 0.5 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
            else:  # Bounce off resistance (exit opportunity)
                price = resistance_level - 4 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 2: Support level bounce
        support_level = self.base_price - 5
        
        for i in range(200):
            if i < 50:  # Approach support
                price = support_level + 3 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 80:  # Test support (entry opportunity)
                price = support_level + 0.5 + np.random.normal(0, 0.3)
                volume = self.base_volume * 1.5 + np.random.normal(0, 150)
            else:  # Bounce off support (exit opportunity)
                price = support_level + 4 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+200),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        # Pattern 3: Breakout from consolidation
        for i in range(200):
            if i < 100:  # Consolidation range
                price = self.base_price + 20 + np.random.normal(0, 0.8)
                volume = self.base_volume + np.random.normal(0, 100)
            elif i < 130:  # Breakout attempt (entry opportunity)
                price = self.base_price + 23 + np.random.normal(0, 0.5)
                volume = self.base_volume * 2 + np.random.normal(0, 200)
            else:  # Failed breakout (exit opportunity)
                price = self.base_price + 21 + np.random.normal(0, 0.5)
                volume = self.base_volume + np.random.normal(0, 100)
                
            bars.append({
                'timestamp': current_time + timedelta(minutes=i+400),
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.2),
                'volume': max(1, int(volume))
            })
            
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        
        filename = "test_support_resistance_dataset.csv"
        df.to_csv(filename)
        
        metadata = {
            "strategy": "support_resistance",
            "description": "Support/Resistance test dataset with 3 clear level patterns",
            "patterns": [
                {"type": "resistance_bounce", "start_bar": 50, "end_bar": 80, "expected_entry": "SHORT"},
                {"type": "support_bounce", "start_bar": 250, "end_bar": 280, "expected_entry": "LONG"},
                {"type": "breakout_failure", "start_bar": 500, "end_bar": 530, "expected_entry": "SHORT"}
            ],
            "total_bars": len(df),
            "created": datetime.now().isoformat()
        }
        
        with open("test_support_resistance_dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Support/Resistance test dataset created: {filename}")
        print(f"Expected entries: {len(metadata['patterns'])}")
        return filename
        
    def create_all_test_datasets(self):
        """Create test datasets for all building block strategies"""
        print("=== Creating Strategic Test Datasets ===")
        
        datasets = {}
        
        # Create datasets for each building block strategy
        datasets['vwap'] = self.create_vwap_test_dataset()
        datasets['order_block'] = self.create_order_block_test_dataset()
        datasets['fvg'] = self.create_fvg_test_dataset()
        datasets['support_resistance'] = self.create_support_resistance_test_dataset()
        
        # Create summary
        summary = {
            "created": datetime.now().isoformat(),
            "datasets": datasets,
            "total_datasets": len(datasets)
        }
        
        with open("test_datasets_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nCreated {len(datasets)} test datasets:")
        for strategy, filename in datasets.items():
            print(f"  - {strategy}: {filename}")
            
        return datasets

def main():
    """Main function"""
    generator = StrategicDatasetGenerator()
    datasets = generator.create_all_test_datasets()
    
    print("\nAll strategic test datasets created successfully!")

if __name__ == "__main__":
    main() 