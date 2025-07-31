#!/usr/bin/env python3
"""
Simple heatmap test to prove the concept works
"""

import sys
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel

def test_simple_heatmap():
    """Test a simple heatmap"""
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Simple Heatmap Test")
    window.setGeometry(100, 100, 800, 600)
    
    # Create central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    # Add info label
    info_label = QLabel("This should show a 3x10 heatmap with clear patterns")
    layout.addWidget(info_label)
    
    # Create a simple test matrix (3 building blocks x 10 time periods)
    matrix = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Building block 1: every other period
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Building block 2: opposite pattern
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]   # Building block 3: pairs
    ])
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix:\n{matrix}")
    
    # Create plot widget
    plot_widget = pg.PlotWidget()
    plot_widget.setMinimumSize(600, 200)
    layout.addWidget(plot_widget)
    
    # Create image item
    img = pg.ImageItem()
    img.setImage(matrix)
    
    # Set levels for better visibility
    img.setLevels([0, 1])
    
    # Add to plot
    plot_widget.addItem(img)
    
    # Set range
    plot_widget.setRange(xRange=(0, matrix.shape[1]), yRange=(0, matrix.shape[0]))
    
    # Set labels
    plot_widget.setLabel('left', 'Building Blocks')
    plot_widget.setLabel('bottom', 'Time Periods')
    
    # Show window
    window.show()
    
    print("Simple heatmap created")
    print("You should see a 3x10 grid with clear patterns")
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_simple_heatmap()