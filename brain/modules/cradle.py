import pyautogui
import time
import platform
import subprocess

# Failsafe
pyautogui.FAILSAFE = True

class Cradle:
    """
    Cradle: The Physical Body of the Agent.
    Implements General Computer Control (GCC) Interface standards.
    Handles all IO (Mouse, Keyboard, Screen, System).
    """
    def __init__(self):
        self.os_name = platform.system()
        self.screen_width, self.screen_height = pyautogui.size()
        self.mouse_speed = 0.2 # Default duration
        
        # Action Space Mapping (Standardized)
        self.special_keys = {
            'enter': 'enter',
            'space': 'space',
            'backspace': 'backspace',
            'tab': 'tab',
            'esc': 'esc',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'ctrl': 'ctrl',
            'alt': 'alt',
            'shift': 'shift',
            'win': 'win', # Windows key
            'cmd': 'command', # Mac
        }

    def get_screen_size(self):
        return self.screen_width, self.screen_height

    def move_mouse(self, x, y, duration=None):
        """
        Move mouse to normalized coordinates (0.0-1.0).
        """
        if duration is None:
            duration = self.mouse_speed
            
        # Clamp
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        tx = int(x * self.screen_width)
        ty = int(y * self.screen_height)
        
        try:
            pyautogui.moveTo(tx, ty, duration=duration)
            return True
        except pyautogui.FailSafeException:
            print("CRADLE: Failsafe Triggered (Corner Reached)")
            return False

    def click(self, button='left', double=False):
        try:
            if double:
                pyautogui.doubleClick(button=button)
            else:
                pyautogui.click(button=button)
            return True
        except Exception as e:
            print(f"CRADLE Error (Click): {e}")
            return False

    def drag(self, x, y, duration=0.5):
        """Drag to normalized coordinates."""
        tx = int(x * self.screen_width)
        ty = int(y * self.screen_height)
        try:
            pyautogui.dragTo(tx, ty, duration=duration, button='left')
            return True
        except Exception:
            return False

    def type_text(self, text, interval=0.05):
        try:
            pyautogui.write(text, interval=interval)
            return True
        except Exception:
            return False

    def press_key(self, key_name):
        try:
            k = self.special_keys.get(key_name, key_name)
            pyautogui.press(k)
            return True
        except Exception:
            return False

    def hotkey(self, *keys):
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception:
            return False

    def scroll(self, clicks):
        try:
            pyautogui.scroll(clicks)
            return True
        except Exception:
            return False

    def execute_action(self, action_dict):
        """
        Execute a high-level action dictionary (GCC format).
        Example: {'type': 'click', 'button': 'left'}
                 {'type': 'type', 'text': 'hello'}
        """
        action_type = action_dict.get('type')
        
        if action_type == 'move':
            return self.move_mouse(action_dict['x'], action_dict['y'])
        elif action_type == 'click':
            return self.click(button=action_dict.get('button', 'left'), double=action_dict.get('double', False))
        elif action_type == 'drag':
            return self.drag(action_dict['x'], action_dict['y'])
        elif action_type == 'type':
            return self.type_text(action_dict['text'])
        elif action_type == 'key':
            return self.press_key(action_dict['key'])
        elif action_type == 'scroll':
            return self.scroll(action_dict['amount'])
        elif action_type == 'wait':
            time.sleep(action_dict.get('duration', 0.1))
            return True
            
        return False

    def execute_code(self, code, gaze_pos=None):
        """
        Executes an integer action code (Evolutionary Interface).
        gaze_pos: (x, y) tuple of current gaze/cursor position for relative movement.
        If None, fetches actual mouse position.
        """
        if gaze_pos is None:
            mx, my = pyautogui.position()
            gx, gy = mx / self.screen_width, my / self.screen_height
        else:
            gx, gy = gaze_pos
        
        try:
            if code == 1: return self.click()
            elif code == 2: return self.move_mouse(gx + 0.05, gy) # Right
            elif code == 3: return self.move_mouse(gx - 0.05, gy) # Left
            elif code == 4: return self.move_mouse(gx, gy + 0.05) # Down
            elif code == 5: return self.move_mouse(gx, gy - 0.05) # Up
            elif code == 6: return self.scroll(-100)
            elif code == 7: return self.scroll(100)
            elif code == 8: return self.press_key('enter')
            elif code == 9: return self.press_key('backspace')
            
            # Typing A-Z (15-40)
            elif 15 <= code <= 40:
                char = chr(code - 15 + 97)
                return self.type_text(char)
                
            # Special Keys
            elif code == 41: return self.press_key('space')
            elif 42 <= code <= 51: return self.type_text(str(code - 42)) # 0-9
            
            # Symbols
            symbols = {52: '+', 53: '-', 54: '*', 55: '=', 56: '?', 57: '(', 58: ')', 
                       59: '"', 60: ':', 61: '/', 62: '.', 63: ',', 64: '#'}
            if code in symbols:
                return self.type_text(symbols[code])
                
            elif code == 65: return self.press_key('f5')
            
            # Navigation (Moved to 66-69 to avoid conflict with Numbers 42-51)
            elif code == 66: return self.press_key('up')
            elif code == 67: return self.press_key('down')
            elif code == 68: return self.press_key('left')
            elif code == 69: return self.press_key('right')
            
            # Advanced Shortcuts (70+)
            elif code == 70: return self.press_key('esc')
            elif code == 71: return self.press_key('tab')
            elif code == 72: return self.hotkey('ctrl', 'c')
            elif code == 73: return self.hotkey('ctrl', 'v')
            elif code == 74: return self.hotkey('alt', 'tab')
            elif code == 75: return self.hotkey('win', 'r') # Run dialog
            
            return True
        except Exception as e:
            print(f"Cradle Execution Error: {e}")
            return False
