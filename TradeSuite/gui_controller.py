"""
Interactive GUI Controller
========================
Direct console control of the trading application GUI.
Use this to control the GUI like a human would through console commands.
"""

import pyautogui
import time
import json
from datetime import datetime
import subprocess
import sys
import os

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

class GUIController:
    def __init__(self):
        self.app_process = None
        self.test_log = []
        self.screenshot_dir = "test_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
    def start_app(self):
        """Start the trading application"""
        print("Starting trading application...")
        self.app_process = subprocess.Popen([sys.executable, "main.py"])
        time.sleep(5)  # Wait for app to load
        print("Application started!")
        
    def stop_app(self):
        """Stop the trading application"""
        if self.app_process:
            self.app_process.terminate()
            print("Application stopped!")
    
    def click(self, x, y):
        """Click at coordinates"""
        pyautogui.click(x, y)
        print(f"Clicked at ({x}, {y})")
        
    def type_text(self, text):
        """Type text"""
        pyautogui.typewrite(text)
        print(f"Typed: {text}")
        
    def press_key(self, key):
        """Press a key"""
        pyautogui.press(key)
        print(f"Pressed: {key}")
        
    def move_mouse(self, x, y):
        """Move mouse to coordinates"""
        pyautogui.moveTo(x, y)
        print(f"Moved mouse to ({x}, {y})")
        
    def screenshot(self, name):
        """Take screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/{name}_{timestamp}.png"
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Screenshot saved: {filename}")
        return filename
        
    def get_position(self):
        """Get current mouse position"""
        pos = pyautogui.position()
        print(f"Mouse position: {pos}")
        return pos
        
    def get_screen_size(self):
        """Get screen size"""
        size = pyautogui.size()
        print(f"Screen size: {size}")
        return size
        
    def find_window(self, title):
        """Find window by title"""
        windows = pyautogui.getWindowsWithTitle(title)
        if windows:
            window = windows[0]
            print(f"Found window: {title} at {window.left}, {window.top}")
            return window
        else:
            print(f"Window not found: {title}")
            return None
            
    def activate_window(self, title):
        """Activate window by title"""
        window = self.find_window(title)
        if window:
            window.activate()
            print(f"Activated window: {title}")
            
    def maximize_window(self, title):
        """Maximize window by title"""
        window = self.find_window(title)
        if window:
            window.maximize()
            print(f"Maximized window: {title}")

def main():
    """Interactive GUI controller"""
    controller = GUIController()
    
    print("=== GUI Controller ===")
    print("Commands:")
    print("  start - Start the trading application")
    print("  stop - Stop the trading application")
    print("  click <x> <y> - Click at coordinates")
    print("  type <text> - Type text")
    print("  press <key> - Press a key")
    print("  move <x> <y> - Move mouse to coordinates")
    print("  screenshot <name> - Take screenshot")
    print("  position - Get mouse position")
    print("  size - Get screen size")
    print("  find <title> - Find window by title")
    print("  activate <title> - Activate window")
    print("  maximize <title> - Maximize window")
    print("  test <strategy> - Test a strategy")
    print("  quit - Exit")
    
    while True:
        try:
            command = input("GUI> ").strip().split()
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == 'quit':
                controller.stop_app()
                break
            elif cmd == 'start':
                controller.start_app()
            elif cmd == 'stop':
                controller.stop_app()
            elif cmd == 'click' and len(command) >= 3:
                x, y = int(command[1]), int(command[2])
                controller.click(x, y)
            elif cmd == 'type' and len(command) > 1:
                text = ' '.join(command[1:])
                controller.type_text(text)
            elif cmd == 'press' and len(command) > 1:
                key = command[1]
                controller.press_key(key)
            elif cmd == 'move' and len(command) >= 3:
                x, y = int(command[1]), int(command[2])
                controller.move_mouse(x, y)
            elif cmd == 'screenshot' and len(command) > 1:
                name = command[1]
                controller.screenshot(name)
            elif cmd == 'position':
                controller.get_position()
            elif cmd == 'size':
                controller.get_screen_size()
            elif cmd == 'find' and len(command) > 1:
                title = ' '.join(command[1:])
                controller.find_window(title)
            elif cmd == 'activate' and len(command) > 1:
                title = ' '.join(command[1:])
                controller.activate_window(title)
            elif cmd == 'maximize' and len(command) > 1:
                title = ' '.join(command[1:])
                controller.maximize_window(title)
            elif cmd == 'test' and len(command) > 1:
                strategy = ' '.join(command[1:])
                test_strategy(controller, strategy)
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def test_strategy(controller, strategy_name):
    """Test a specific strategy"""
    print(f"Testing strategy: {strategy_name}")
    
    # This is a basic test - you can customize the coordinates
    # based on your actual GUI layout
    
    # 1. Find and activate the trading application window
    controller.activate_window("Trading Strategy Hub")
    time.sleep(1)
    
    # 2. Take initial screenshot
    controller.screenshot(f"before_{strategy_name}")
    
    # 3. Click on dataset dropdown (estimated coordinates)
    controller.click(100, 200)
    time.sleep(1)
    controller.press_key('down')
    time.sleep(1)
    controller.press_key('enter')
    time.sleep(1)
    
    # 4. Click on strategy dropdown
    controller.click(300, 200)
    time.sleep(1)
    controller.type_text(strategy_name)
    time.sleep(1)
    controller.press_key('enter')
    time.sleep(2)
    
    # 5. Click run backtest button
    controller.click(500, 200)
    time.sleep(5)  # Wait for backtest to complete
    
    # 6. Take screenshot of results
    controller.screenshot(f"after_{strategy_name}")
    
    # 7. Click on different tabs to capture data
    tabs = [
        (100, 300, "overview"),
        (200, 300, "equity"),
        (300, 300, "chart"),
        (400, 300, "trades"),
        (500, 300, "stats")
    ]
    
    for x, y, tab_name in tabs:
        controller.click(x, y)
        time.sleep(1)
        controller.screenshot(f"{strategy_name}_{tab_name}")
    
    print(f"Strategy {strategy_name} test completed!")

if __name__ == "__main__":
    main() 