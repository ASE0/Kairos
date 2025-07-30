"""
Automated GUI Tester
===================
Directly controls the trading application GUI to test strategies.
No interactive prompts - runs automatically.
"""

import pyautogui
import time
import subprocess
import sys
import os
from datetime import datetime

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1.0  # 1 second pause between actions

def start_trading_app():
    """Start the trading application"""
    print("Starting trading application...")
    process = subprocess.Popen([sys.executable, "main.py"])
    time.sleep(8)  # Wait for app to load
    print("Application started!")
    return process

def test_gui_control():
    """Test basic GUI control"""
    print("Testing GUI control...")
    
    # Get screen size
    screen_size = pyautogui.size()
    print(f"Screen size: {screen_size}")
    
    # Get current mouse position
    mouse_pos = pyautogui.position()
    print(f"Current mouse position: {mouse_pos}")
    
    # Take a screenshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_name = f"test_screenshot_{timestamp}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_name)
    print(f"Screenshot saved: {screenshot_name}")
    
    # Find the trading application window
    windows = pyautogui.getWindowsWithTitle("main.py - Kairos - Cursor")
    if windows:
        window = windows[0]
        print(f"Found trading window at: {window.left}, {window.top}")
        
        # Activate the window
        window.activate()
        time.sleep(2)
        
        # Maximize the window
        window.maximize()
        time.sleep(2)
        
        # Take another screenshot
        screenshot2 = pyautogui.screenshot()
        screenshot2.save(f"maximized_{screenshot_name}")
        print(f"Maximized screenshot saved: maximized_{screenshot_name}")
        
        return window
    else:
        print("Trading application window not found!")
        return None

def test_mouse_movement():
    """Test mouse movement and clicking"""
    print("Testing mouse movement...")
    
    # Move mouse to center of screen
    screen_width, screen_height = pyautogui.size()
    center_x, center_y = screen_width // 2, screen_height // 2
    
    pyautogui.moveTo(center_x, center_y)
    print(f"Moved mouse to center: ({center_x}, {center_y})")
    time.sleep(1)
    
    # Click at center
    pyautogui.click(center_x, center_y)
    print("Clicked at center")
    time.sleep(1)
    
    # Move to different positions and click
    positions = [
        (100, 100),
        (screen_width - 100, 100),
        (100, screen_height - 100),
        (screen_width - 100, screen_height - 100)
    ]
    
    for x, y in positions:
        pyautogui.moveTo(x, y)
        print(f"Moved to ({x}, {y})")
        time.sleep(0.5)
        pyautogui.click(x, y)
        print(f"Clicked at ({x}, {y})")
        time.sleep(0.5)

def test_keyboard_input():
    """Test keyboard input"""
    print("Testing keyboard input...")
    
    # Type some text
    pyautogui.typewrite("Hello GUI Test!")
    print("Typed: Hello GUI Test!")
    time.sleep(1)
    
    # Press some keys
    pyautogui.press('enter')
    print("Pressed Enter")
    time.sleep(1)
    
    pyautogui.press('tab')
    print("Pressed Tab")
    time.sleep(1)

def main():
    """Main test function"""
    print("=== Automated GUI Tester ===")
    
    try:
        # Start the trading application
        app_process = start_trading_app()
        
        # Test basic GUI control
        window = test_gui_control()
        
        if window:
            # Test mouse movement
            test_mouse_movement()
            
            # Test keyboard input
            test_keyboard_input()
            
            print("All tests completed successfully!")
        else:
            print("Could not find trading application window!")
            
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Clean up
        if 'app_process' in locals():
            app_process.terminate()
            print("Application terminated.")

if __name__ == "__main__":
    main() 