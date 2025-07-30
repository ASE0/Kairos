"""
Improved GUI Tester
==================
Enhanced version with better window handling and debugging.
"""

import pyautogui
import time
import subprocess
import sys
import os
from datetime import datetime

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1.0

def start_trading_app():
    """Start the trading application"""
    print("Starting trading application...")
    process = subprocess.Popen([sys.executable, "main.py"])
    time.sleep(10)  # Wait longer for app to load
    print("Application started!")
    return process

def find_and_activate_window():
    """Find and activate the trading application window"""
    print("Searching for trading application window...")
    
    # List all windows to debug
    all_windows = pyautogui.getAllWindows()
    print(f"Found {len(all_windows)} windows:")
    for i, window in enumerate(all_windows[:10]):  # Show first 10
        print(f"  {i}: '{window.title}' at ({window.left}, {window.top})")
    
    # Try different possible window titles
    possible_titles = [
        "main.py - Kairos - Cursor",
        "Trading Strategy Hub",
        "main.py",
        "Kairos"
    ]
    
    for title in possible_titles:
        windows = pyautogui.getWindowsWithTitle(title)
        if windows:
            window = windows[0]
            print(f"Found window with title '{title}' at ({window.left}, {window.top})")
            print(f"Window size: {window.width} x {window.height}")
            
            # Try to activate the window
            try:
                window.activate()
                time.sleep(2)
                print("Window activated")
                
                # Try to maximize the window
                print("Attempting to maximize window...")
                window.maximize()
                time.sleep(2)
                print("Window maximized")
                
                return window
            except Exception as e:
                print(f"Error activating/maximizing window: {e}")
                continue
    
    print("No trading application window found!")
    return None

def test_window_control():
    """Test window control and take screenshots"""
    print("Testing window control...")
    
    # Get screen info
    screen_size = pyautogui.size()
    print(f"Screen size: {screen_size}")
    
    # Take initial screenshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_screenshot = f"initial_{timestamp}.png"
    pyautogui.screenshot().save(initial_screenshot)
    print(f"Initial screenshot saved: {initial_screenshot}")
    
    # Find and activate window
    window = find_and_activate_window()
    
    if window:
        # Take screenshot after activation
        activated_screenshot = f"activated_{timestamp}.png"
        pyautogui.screenshot().save(activated_screenshot)
        print(f"Activated screenshot saved: {activated_screenshot}")
        
        # Get window info after activation
        print(f"Window after activation:")
        print(f"  Position: ({window.left}, {window.top})")
        print(f"  Size: {window.width} x {window.height}")
        
        return window
    else:
        print("Could not find or activate window!")
        return None

def test_mouse_control():
    """Test mouse control with better positioning"""
    print("Testing mouse control...")
    
    screen_width, screen_height = pyautogui.size()
    
    # Test positions relative to screen center
    test_positions = [
        (screen_width // 2, screen_height // 2),  # Center
        (screen_width // 4, screen_height // 4),  # Top-left quadrant
        (3 * screen_width // 4, screen_height // 4),  # Top-right quadrant
        (screen_width // 4, 3 * screen_height // 4),  # Bottom-left quadrant
        (3 * screen_width // 4, 3 * screen_height // 4),  # Bottom-right quadrant
    ]
    
    for i, (x, y) in enumerate(test_positions):
        print(f"Test {i+1}: Moving to ({x}, {y})")
        pyautogui.moveTo(x, y)
        time.sleep(0.5)
        
        # Click and take screenshot
        pyautogui.click(x, y)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot = f"click_test_{i+1}_{timestamp}.png"
        pyautogui.screenshot().save(screenshot)
        print(f"  Screenshot saved: {screenshot}")
        time.sleep(0.5)

def test_keyboard_control():
    """Test keyboard control"""
    print("Testing keyboard control...")
    
    # Test typing
    pyautogui.typewrite("GUI Test - Keyboard Control Working!")
    print("Typed test message")
    time.sleep(1)
    
    # Test key combinations
    pyautogui.hotkey('ctrl', 'a')  # Select all
    print("Pressed Ctrl+A")
    time.sleep(1)
    
    pyautogui.press('delete')  # Delete selection
    print("Pressed Delete")
    time.sleep(1)
    
    # Test individual keys
    for key in ['tab', 'enter', 'space']:
        pyautogui.press(key)
        print(f"Pressed {key}")
        time.sleep(0.5)

def main():
    """Main test function"""
    print("=== Improved GUI Tester ===")
    
    try:
        # Start the trading application
        app_process = start_trading_app()
        
        # Test window control
        window = test_window_control()
        
        if window:
            # Test mouse control
            test_mouse_control()
            
            # Test keyboard control
            test_keyboard_control()
            
            print("All tests completed successfully!")
        else:
            print("Could not control window - tests failed!")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'app_process' in locals():
            app_process.terminate()
            print("Application terminated.")

if __name__ == "__main__":
    main() 