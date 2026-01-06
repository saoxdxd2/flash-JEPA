import pyautogui
import time

# Failsafe: Move mouse to corner to abort
pyautogui.FAILSAFE = True

class ControlInterface:
    """
    Interface for Physical Actions (Mouse/Keyboard).
    """
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    def move_mouse(self, x, y, duration=0.1):
        """
        Moves mouse to (x, y).
        Coordinates are normalized [0, 1] or absolute pixels.
        """
        try:
            # Assume normalized for safety
            if 0 <= x <= 1 and 0 <= y <= 1:
                tx = int(x * self.screen_width)
                ty = int(y * self.screen_height)
                pyautogui.moveTo(tx, ty, duration=duration)
            else:
                # Absolute
                pyautogui.moveTo(x, y, duration=duration)
        except pyautogui.FailSafeException:
            print("WARNING: Failsafe triggered in move_mouse! Resetting internal state.")
            # We can't move the mouse if it's stuck in a corner triggering failsafe,
            # but we can suppress the crash.
            pass

    def move_mouse_relative(self, dx, dy, duration=0.1):
        """Moves mouse relative to current position."""
        try:
            pyautogui.moveRel(dx, dy, duration=duration)
        except pyautogui.FailSafeException:
            print("WARNING: Failsafe triggered in move_mouse_relative!")
            pass

    def click(self):
        pyautogui.click()

    def type_text(self, text):
        pyautogui.write(text, interval=0.05)

    def press_key(self, key):
        pyautogui.press(key)

    def scroll(self, clicks):
        pyautogui.scroll(clicks)

    def execute_action_code(self, action_code):
        """
        Maps an integer action code to a physical action.
        Returns True if successful, False if Failsafe triggered.
        """
        try:
            if action_code == 0:
                pass # Wait
            else:
                print(f"DEBUG: Executing Action {action_code}") # Debug
                pass
                
            if action_code == 0:
                pass # Wait
            elif action_code == 1:
                self.click()
            elif action_code == 2:
                pyautogui.moveRel(10, 0) # Right
            elif action_code == 3:
                pyautogui.moveRel(-10, 0) # Left
            elif action_code == 4:
                pyautogui.moveRel(0, 10) # Down
            elif action_code == 5:
                pyautogui.moveRel(0, -10) # Up
            elif action_code == 6:
                self.scroll(-100) # Scroll Down
            elif action_code == 7:
                self.scroll(100) # Scroll Up
            elif action_code == 8:
                self.press_key('enter')
            elif action_code == 9:
                self.press_key('backspace')
            # Action 10 is SLEEP (Handled by Brain)
            # Actions 11-14 are SACCADES (Handled by Brain)
            
            # Actions 15-40: Typing A-Z
            elif 15 <= action_code <= 40:
                # 15 -> 'a', 16 -> 'b', ...
                char_code = action_code - 15 + 97 # 97 is ASCII for 'a'
                char = chr(char_code)
                self.press_key(char)
                
            # Actions 41-56: Special Keys & Numbers
            elif action_code == 41:
                self.press_key('space')
            
            # Numbers 0-9 (Actions 42-51)
            elif 42 <= action_code <= 51:
                num = action_code - 42
                self.press_key(str(num))
            
            # Symbols (Actions 52-56)
            elif action_code == 52:
                self.press_key('+')
            elif action_code == 53:
                self.press_key('-')
            elif action_code == 54:
                self.press_key('*')
            elif action_code == 55:
                self.press_key('=')
            elif action_code == 56:
                self.press_key('?')
            elif action_code == 57:
                self.press_key('(')
            elif action_code == 58:
                self.press_key(')')
            elif action_code == 59:
                self.press_key('"')
            elif action_code == 60:
                self.press_key(':')
            elif action_code == 61:
                self.press_key('/')
            elif action_code == 62:
                self.press_key('.')
            elif action_code == 63:
                self.press_key(',')
            elif action_code == 64:
                self.press_key('#') # Comments
            elif action_code == 65:
                self.press_key('f5') # RUN CODE
            # Actions 45-49: Navigation & Control
            # (Currently unused or remapped)
            elif action_code == 46:
                self.press_key('down')
            elif action_code == 47:
                self.press_key('left')
            elif action_code == 48:
                self.press_key('right')
            elif action_code == 49:
                # self.press_key('ctrl') # DISABLED FOR SAFETY
                print("DEBUG: Ctrl key disabled for safety.")
                pass

            return True

        except pyautogui.FailSafeException:
            print("WARNING: Failsafe triggered! Resetting mouse.")
            return False
        except Exception as e:
            print(f"Action Error: {e}")
            return False
