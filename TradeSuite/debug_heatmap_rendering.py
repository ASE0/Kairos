#!/usr/bin/env python3
"""
Debug script to test heatmap rendering directly
"""

import sys
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_heatmap():
    """Test a simple heatmap to see if PyQtGraph is working"""
    
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
    
    # Create a simple test matrix
    matrix = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    
    print(f"Test matrix shape: {matrix.shape}")
    print(f"Matrix values:\n{matrix}")
    print(f"Matrix min: {matrix.min()}, max: {matrix.max()}")
    
    # Create plot widget
    plot_widget = pg.PlotWidget()
    layout.addWidget(plot_widget)
    
    # Create image item
    img = pg.ImageItem()
    img.setImage(matrix)
    
    # Set color map
    colormap = pg.colormap.get('viridis')
    img.setColorMap(colormap)
    
    # Add to plot
    plot_widget.addItem(img)
    
    # Set plot range
    plot_widget.setRange(xRange=(0, matrix.shape[1]), yRange=(0, matrix.shape[0]))
    
    # Set labels
    plot_widget.setLabel('left', 'Rows')
    plot_widget.setLabel('bottom', 'Columns')
    
    print("Created simple heatmap test")
    print("You should see a 4x5 grid with colored squares")
    
    # Show window
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_simple_heatmap() 