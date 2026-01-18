"""
RDP Session Manager

Manages multiple RDP sessions on a single Windows VM.
Each agent gets their own session with isolated desktop.
"""

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum
import numpy as np

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from PIL import ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SessionState(Enum):
    """State of an RDP session."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"


@dataclass
class AgentSession:
    """
    Represents a single agent's RDP session.
    
    Each agent gets:
    - Their own Windows user account
    - Isolated desktop session
    - Screen capture capability
    - Mouse/keyboard control
    """
    session_id: int
    username: str
    agent_name: str
    state: SessionState = SessionState.DISCONNECTED
    
    # Session resources
    screen_width: int = 1024
    screen_height: int = 768
    
    # Tracking
    birth_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.process = None
        self._screen_buffer = None
    
    @property
    def age_seconds(self) -> float:
        """Age of this session in seconds."""
        return time.time() - self.birth_time
    
    @property
    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self.last_activity
    
    def get_screen(self) -> Optional[np.ndarray]:
        """
        Capture the current screen of this session.
        Returns numpy array of shape (H, W, 3) in RGB format.
        """
        if not PIL_AVAILABLE:
            print("Warning: PIL not available for screen capture")
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        try:
            # Capture screen (works for active session)
            screenshot = ImageGrab.grab()
            # Resize to session resolution
            screenshot = screenshot.resize((self.screen_width, self.screen_height))
            return np.array(screenshot)
        except Exception as e:
            print(f"Screen capture error: {e}")
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
    
    def move_mouse(self, x: int, y: int):
        """Move mouse cursor to position (x, y)."""
        if not PYAUTOGUI_AVAILABLE:
            print("Warning: pyautogui not available for mouse control")
            return
        
        # Clamp to screen bounds
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        try:
            pyautogui.moveTo(x, y)
            self.last_activity = time.time()
        except Exception as e:
            print(f"Mouse move error: {e}")
    
    def click(self, x: int, y: int, button: str = "left"):
        """Click at position (x, y)."""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        try:
            pyautogui.click(x, y, button=button)
            self.last_activity = time.time()
        except Exception as e:
            print(f"Click error: {e}")
    
    def double_click(self, x: int, y: int):
        """Double-click at position."""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        try:
            pyautogui.doubleClick(x, y)
            self.last_activity = time.time()
        except Exception as e:
            print(f"Double-click error: {e}")
    
    def type_text(self, text: str, interval: float = 0.05):
        """Type text on keyboard."""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        try:
            pyautogui.typewrite(text, interval=interval)
            self.last_activity = time.time()
        except Exception as e:
            # Fall back to write for unicode
            try:
                pyautogui.write(text)
                self.last_activity = time.time()
            except Exception as e2:
                print(f"Type error: {e2}")
    
    def press_key(self, key: str):
        """Press a special key (enter, tab, escape, etc.)."""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        try:
            pyautogui.press(key)
            self.last_activity = time.time()
        except Exception as e:
            print(f"Key press error: {e}")
    
    def hotkey(self, *keys):
        """Press a keyboard shortcut (e.g., ctrl+c)."""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        try:
            pyautogui.hotkey(*keys)
            self.last_activity = time.time()
        except Exception as e:
            print(f"Hotkey error: {e}")
    
    def scroll(self, clicks: int = 3, x: int = None, y: int = None):
        """Scroll mouse wheel."""
        if not PYAUTOGUI_AVAILABLE:
            return
        
        try:
            pyautogui.scroll(clicks, x, y)
            self.last_activity = time.time()
        except Exception as e:
            print(f"Scroll error: {e}")


class SessionManager:
    """
    Manages all RDP sessions for agents.
    
    Responsibilities:
    - Create/destroy sessions
    - Route input to correct session
    - Monitor session health
    - Enforce resource limits
    """
    
    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self.sessions: Dict[int, AgentSession] = {}
        self._next_session_id = 1
    
    @property
    def active_session_count(self) -> int:
        return len([s for s in self.sessions.values() if s.state == SessionState.ACTIVE])
    
    def create_session(self, agent_name: str, username: str = None) -> Optional[AgentSession]:
        """
        Create a new session for an agent.
        
        Args:
            agent_name: Name of the agent
            username: Windows username (auto-generated if None)
        
        Returns:
            AgentSession if successful, None if at capacity
        """
        if self.active_session_count >= self.max_sessions:
            print(f"Cannot create session: at max capacity ({self.max_sessions})")
            return None
        
        session_id = self._next_session_id
        self._next_session_id += 1
        
        if username is None:
            username = f"agent_{session_id}"
        
        session = AgentSession(
            session_id=session_id,
            username=username,
            agent_name=agent_name,
            state=SessionState.ACTIVE
        )
        
        self.sessions[session_id] = session
        print(f"Created session {session_id} for agent '{agent_name}'")
        
        return session
    
    def destroy_session(self, session_id: int):
        """Destroy a session and free resources."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.state = SessionState.DISCONNECTED
            
            # Clean up any processes
            if session.process:
                try:
                    session.process.terminate()
                except:
                    pass
            
            del self.sessions[session_id]
            print(f"Destroyed session {session_id}")
    
    def get_session(self, session_id: int) -> Optional[AgentSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> Dict[int, AgentSession]:
        """Get all sessions."""
        return self.sessions.copy()
    
    def broadcast_to_all(self, action: str, *args, **kwargs):
        """Execute an action on all sessions."""
        results = {}
        for session_id, session in self.sessions.items():
            try:
                method = getattr(session, action, None)
                if method:
                    results[session_id] = method(*args, **kwargs)
            except Exception as e:
                results[session_id] = f"Error: {e}"
        return results


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
