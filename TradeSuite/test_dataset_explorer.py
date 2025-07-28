#!/usr/bin/env python3
"""
Test script for the Dataset Explorer Window
============================================
This script demonstrates the functionality of the dataset explorer.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.data_explorer_window import DatasetExplorerWindow
from core.dataset_manager import DatasetManager
from strategies.strategy_builders import PatternStrategy

def create_sample_datasets():
    """Create some sample datasets for testing"""
    dataset_manager = DatasetManager()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Dataset 1: Price data
    base_price = 100
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    price_data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    price_data['High'] = price_data[['Open', 'Close', 'High']].max(axis=1)
    price_data['Low'] = price_data[['Open', 'Close', 'Low']].min(axis=1)
    
    # Dataset 2: Volume data
    volume_data = pd.DataFrame({
        'Volume': np.random.randint(5000, 50000, len(dates)),
        'Price': np.random.uniform(50, 150, len(dates)),
        'Volatility': np.random.uniform(0.01, 0.05, len(dates))
    }, index=dates)
    
    # Create dummy strategies
    strategy1 = PatternStrategy(name="Price Strategy")
    strategy2 = PatternStrategy(name="Volume Strategy")
    
    # Save datasets
    try:
        info1 = dataset_manager.save_dataset(
            name="Sample Price Data",
            data=price_data,
            strategies=[strategy1],
            notes="Sample price data for testing",
            tags=['sample', 'price', 'test']
        )
        print(f"Created dataset: {info1.name} (ID: {info1.id})")
        
        info2 = dataset_manager.save_dataset(
            name="Sample Volume Data",
            data=volume_data,
            strategies=[strategy2],
            notes="Sample volume data for testing",
            tags=['sample', 'volume', 'test']
        )
        print(f"Created dataset: {info2.name} (ID: {info2.id})")
        
        return True
    except Exception as e:
        print(f"Error creating sample datasets: {e}")
        return False

def main():
    """Main function to test the dataset explorer"""
    app = QApplication(sys.argv)
    
    # Create sample datasets
    print("Creating sample datasets...")
    if not create_sample_datasets():
        print("Failed to create sample datasets. Exiting.")
        return
    
    # Create and show the dataset explorer
    print("Opening Dataset Explorer...")
    explorer = DatasetExplorerWindow()
    
    # Connect signals
    def on_dataset_selected(dataset_id, data, info):
        print(f"Dataset selected: {info.name} with {len(data)} rows")
    
    def on_datasets_combined(dataset_id, data, info):
        print(f"Datasets combined: {dataset_id} with {len(data)} rows")
    
    explorer.dataset_selected.connect(on_dataset_selected)
    explorer.datasets_combined.connect(on_datasets_combined)
    
    explorer.show()
    
    print("Dataset Explorer is now open!")
    print("Features available:")
    print("- Search and filter datasets")
    print("- View dataset details and statistics")
    print("- Visualize data (if matplotlib is available)")
    print("- Combine multiple datasets")
    print("- Import/export datasets")
    print("- Accept/reject datasets")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 