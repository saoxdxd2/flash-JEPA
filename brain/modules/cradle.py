import ctypes
import time
import platform
import math
from ctypes import wintypes

# --- Windows API Constants ---
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_WHEEL = 0x0800

# --- Ctypes Structures ---
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT),
                    ("ki", KEYBDINPUT),
                    ("hi", HARDWAREINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.c_ulong),
                ("_input", _INPUT)]

# --- Scan Code Map (Hardware Level) ---
# Maps index (0-35) to Scan Code
# 0-25: A-Z
# 26-35: 0-9
SCAN_CODE_MAP = [
    0x1E, 0x30, 0x2E, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17, 0x24, # A-J
    0x25, 0x26, 0x32, 0x31, 0x18, 0x19, 0x10, 0x13, 0x1F, 0x14, # K-T
    0x16, 0x2F, 0x11, 0x2D, 0x15, 0x2C,                         # U-Z
    0x0B, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A  # 0-9 (0 is 0x0B, 1 is 0x02...)
]

SPECIAL_KEYS_MAP = {
    'enter': 0x1C,
    'space': 0x39,
    'backspace': 0x0E,
    'tab': 0x0F,
    'esc': 0x01,
    'up': 0x48, # Extended
    'down': 0x50, # Extended
    'left': 0x4B, # Extended
    'right': 0x4D, # Extended
}

class Cradle:
    """
    Cradle: High-Performance Direct Neural Interface.
    Uses direct Windows API calls for millisecond-precision input.
    """
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.screen_width = self.user32.GetSystemMetrics(0)
        self.screen_height = self.user32.GetSystemMetrics(1)
        
    def _send_input(self, input_struct):
        self.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    def move_mouse(self, x, y):
        """Absolute movement (0.0-1.0) using normalized coordinates."""
        # Windows expects 0-65535 for absolute coordinates
        nx = int(x * 65535)
        ny = int(y * 65535)
        
        mi = MOUSEINPUT(dx=nx, dy=ny, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
        inp = INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi))
        self._send_input(inp)
        return True

    def click(self, button='left', double=False):
        if button == 'left':
            down = MOUSEEVENTF_LEFTDOWN
            up = MOUSEEVENTF_LEFTUP
        elif button == 'right':
            down = MOUSEEVENTF_RIGHTDOWN
            up = MOUSEEVENTF_RIGHTUP
        else:
            return False
            
        # Click = Down + Up
        mi_down = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=down, time=0, dwExtraInfo=None)
        inp_down = INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi_down))
        self._send_input(inp_down)
        
        mi_up = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=up, time=0, dwExtraInfo=None)
        inp_up = INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi_up))
        self._send_input(inp_up)
        
        if double:
            # Repeat
            self._send_input(inp_down)
            self._send_input(inp_up)
            
        return True

    def scroll(self, amount):
        mi = MOUSEINPUT(dx=0, dy=0, mouseData=amount, dwFlags=MOUSEEVENTF_WHEEL, time=0, dwExtraInfo=None)
        inp = INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi))
        self._send_input(inp)
        return True

    def press_key(self, scan_code, extended=False):
        flags = KEYEVENTF_SCANCODE
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY
            
        # Key Down
        ki_down = KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags, time=0, dwExtraInfo=None)
        inp_down = INPUT(type=INPUT_KEYBOARD, _input=INPUT._INPUT(ki=ki_down))
        self._send_input(inp_down)
        
        # Key Up
        ki_up = KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags | KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)
        inp_up = INPUT(type=INPUT_KEYBOARD, _input=INPUT._INPUT(ki=ki_up))
        self._send_input(inp_up)
        return True

    def execute_distributed(self, intent_code, params):
        """
        Executes a distributed action using Fast Input.
        """
        try:
            if hasattr(params, 'cpu'): params = params.cpu().numpy()
            if hasattr(params, 'flatten'): params = params.flatten()
            
            # 0: Move Mouse
            if intent_code == 0:
                x, y = params[0], params[1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                return self.move_mouse(x, y)
                
            # 1: Left Click
            elif intent_code == 1:
                return self.click(button='left')
                
            # 2: Right Click
            elif intent_code == 2:
                return self.click(button='right')
                
            # 3: Double Click
            elif intent_code == 3:
                return self.click(button='left', double=True)
                
            # 4: Type Character (Direct Scan Code Mapping)
            elif intent_code == 4:
                char_logits = params[10:] 
                if len(char_logits) == 0: return False
                idx = int(params[10:].argmax())
                
                if 0 <= idx < len(SCAN_CODE_MAP):
                    scan_code = SCAN_CODE_MAP[idx]
                    return self.press_key(scan_code)
                return False
                
            # 5: Special Key
            elif intent_code == 5:
                keys = ['enter', 'space', 'backspace', 'tab', 'esc', 'up', 'down', 'left', 'right']
                idx = int(params[10:].argmax()) % len(keys)
                key_name = keys[idx]
                scan_code = SPECIAL_KEYS_MAP.get(key_name, 0)
                extended = key_name in ['up', 'down', 'left', 'right']
                if scan_code:
                    return self.press_key(scan_code, extended=extended)
                return False
                
            # 6: Scroll Up
            elif intent_code == 6:
                return self.scroll(120)
                
            # 7: Scroll Down
            elif intent_code == 7:
                return self.scroll(-120)
                
            # 8: Drag (Swipe)
            elif intent_code == 8:
                # 1. Move to start (current position is implicit, but we drag TO the target)
                # Actually, drag usually implies "Press Down Here, Move There, Release".
                # But our action is instantaneous. So we:
                # - Press Left Down (at current pos)
                # - Move to Target (x, y)
                # - Release Left Up
                
                x, y = params[0], params[1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                
                # Down
                mi_down = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=None)
                self._send_input(INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi_down)))
                
                # Move
                nx = int(x * 65535)
                ny = int(y * 65535)
                mi_move = MOUSEINPUT(dx=nx, dy=ny, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
                self._send_input(INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi_move)))
                
                # Up
                mi_up = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=None)
                self._send_input(INPUT(type=INPUT_MOUSE, _input=INPUT._INPUT(mi=mi_up)))
                
                return True
                
            # 9: Wait
            elif intent_code == 9:
                return True
                
            return False
            
        except Exception as e:
            print(f"Cradle Fast Error: {e}")
            return False
            
    # Legacy Support
    def execute_code(self, code, gaze_pos=None):
        pass # Deprecated
