#!/usr/bin/env python3
"""
Test script to load a real backtest result and debug heatmap
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.results_viewer_window import ResultsViewerWindow
from PyQt6.QtWidgets import QApplication

def load_real_result():
    """Load a real backtest result from the results directory"""
    
    # Look for result files in the workspaces/results directory
    results_dir = "workspaces/results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found")
        return None
    
    # Find the most recent result file
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("No result files found")
        return None
    
    # Sort by modification time (most recent first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(result_files)} result files")
    for i, file in enumerate(result_files[:5]):  # Show first 5
        print(f"  {i+1}. {file}")
    
    # Try to load result files until we find a valid one
    for i, result_file in enumerate(result_files[:10]):  # Try first 10
        print(f"\nTrying result file {i+1}: {result_file}")
        
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            print(f"Successfully loaded result with keys: {list(result_data.keys())}")
            
            # Check if it has the required data
            if 'action_details' in result_data:
                print(f"Found action_details with keys: {list(result_data['action_details'].keys())}")
                return result_data
            else:
                print("No action_details found, trying next file")
                continue
                
        except Exception as e:
            print(f"Error loading result: {e}")
            continue
    
    print("Could not load any valid result files")
    return None

def test_real_heatmap():
    """Test heatmap with real backtest result"""
    print("Testing heatmap with real backtest result...")
    
    # Load real result
    result_data = load_real_result()
    if not result_data:
        print("Could not load real result, exiting")
        return
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create results viewer window
    results_viewer = ResultsViewerWindow()
    
    # Add the result
    strategy_name = result_data.get('strategy_name', 'Real_Strategy')
    results_viewer.add_result(result_data, strategy_name)
    
    # Show the window
    results_viewer.show()
    
    print("Results viewer window opened.")
    print("Navigate to the 'Heatmap' tab.")
    print("Check if the heatmap shows your strategy's building blocks.")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_real_heatmap() 