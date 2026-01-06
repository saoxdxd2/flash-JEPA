import torch
import numpy as np
import time
import os
import sys
import io
import contextlib
import pickle
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.population import PopulationManager

class Curriculum:
    def __init__(self):
        self.levels = [
            {"id": 1, "text": "print(\"Hello World\")", "desc": "Print 'Hello World'", "type": "output", "expected": "Hello World", "code": 'print("Hello World")'},
            {"id": 1.5, "text": "print(\"Python\")", "desc": "Practice: Print 'Python'", "type": "output", "expected": "Python", "code": 'print("Python")', "practice": True},
            
            {"id": 2, "text": "x = 5", "desc": "Assign 5 to x", "type": "var", "var_name": "x", "expected": 5, "code": "x = 5"},
            {"id": 2.5, "text": "y = 10", "desc": "Practice: Assign 10 to y", "type": "var", "var_name": "y", "expected": 10, "code": "y = 10", "practice": True},
            
            {"id": 3, "text": "nums = [1, 2, 3]", "desc": "Create list [1, 2, 3]", "type": "var", "var_name": "nums", "expected": [1, 2, 3], "code": "nums = [1, 2, 3]"},
            {"id": 3.5, "text": "a = [4, 5]", "desc": "Practice: Create list [4, 5]", "type": "var", "var_name": "a", "expected": [4, 5], "code": "a = [4, 5]", "practice": True},
            
            {"id": 4, "text": "res = 1 + 2", "desc": "Add 1 and 2", "type": "var", "var_name": "res", "expected": 3, "code": "res = 1 + 2"},
            {"id": 4.5, "text": "sum = 5 + 5", "desc": "Practice: Add 5 and 5", "type": "var", "var_name": "sum", "expected": 10, "code": "sum = 5 + 5", "practice": True},
            
            {"id": 5, "text": "print(\"Hello %s\" % \"World\")", "desc": "Format string", "type": "output", "expected": "Hello World", "code": 'print("Hello %s" % "World")'},
            {"id": 5.5, "text": "print(\"Hi %s\" % \"You\")", "desc": "Practice: Format 'Hi You'", "type": "output", "expected": "Hi You", "code": 'print("Hi %s" % "You")', "practice": True},
            
            {"id": 6, "text": "print(\"Hello\"[0])", "desc": "Print first char", "type": "output", "expected": "H", "code": 'print("Hello"[0])'},
            {"id": 6.5, "text": "print(\"World\"[1])", "desc": "Practice: Print 2nd char of 'World'", "type": "output", "expected": "o", "code": 'print("World"[1])', "practice": True},
            
            {"id": 7, "text": "if x == 5: print(\"Yes\")", "desc": "If x is 5 print 'Yes'", "type": "output", "expected": "Yes", "code": 'if x == 5: print("Yes")', "setup": "x=5"},
            {"id": 7.5, "text": "if a == 1: print(\"1\")", "desc": "Practice: If a is 1 print '1'", "type": "output", "expected": "1", "code": 'if a == 1: print("1")', "setup": "a=1", "practice": True},
            
            {"id": 8, "text": "for i in range(5): print(i)", "desc": "Loop 5 times", "type": "output", "expected": "0\n1\n2\n3\n4", "code": 'for i in range(5): print(i)'},
            {"id": 8.5, "text": "for x in range(3): print(x)", "desc": "Practice: Loop 3 times", "type": "output", "expected": "0\n1\n2", "code": 'for x in range(3): print(x)', "practice": True},
            
            {"id": 9, "text": "def f(): print(\"Hi\")", "desc": "Define func", "type": "run_check", "check_code": "f()", "expected": "Hi", "code": 'def f(): print("Hi")'},
            {"id": 9.5, "text": "def g(): print(\"Go\")", "desc": "Practice: Define func g", "type": "run_check", "check_code": "g()", "expected": "Go", "code": 'def g(): print("Go")', "practice": True},
            
            {"id": 10, "text": "class MyClass: pass", "desc": "Define class", "type": "var", "var_name": "MyClass", "expected": "class", "code": "class MyClass: pass"},
            {"id": 10.5, "text": "class A: pass", "desc": "Practice: Define class A", "type": "var", "var_name": "A", "expected": "class", "code": "class A: pass", "practice": True},
            
            {"id": 11, "text": "d = {\"a\": 1}", "desc": "Create dict", "type": "var", "var_name": "d", "expected": {"a": 1}, "code": 'd = {"a": 1}'},
            {"id": 11.5, "text": "x = {\"b\": 2}", "desc": "Practice: Create dict x", "type": "var", "var_name": "x", "expected": {"b": 2}, "code": 'x = {"b": 2}', "practice": True},
        ]
        self.current_level_idx = 0
        # self.load_progress() # REMOVED: Progress managed by PopulationManager

    def get_current_level(self):
        return self.levels[self.current_level_idx]

    def next_level(self):
        if self.current_level_idx < len(self.levels) - 1:
            self.current_level_idx += 1
            # self.save_progress() # REMOVED
            return True
        return False

    # def save_progress(self): ... # REMOVED
    # def load_progress(self): ... # REMOVED

class BrowserEmulator:
    """
    Simulates a web browser for the agent to learn coding.
    Renders text and code editor to an image tensor.
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.code_buffer = ""
        self.output_buffer = ""
        self.last_result = ""
        self.cursor_pos = 0
        self.current_level = None
        
        # Font (Use default or load one if available)
        try:
            self.font = ImageFont.truetype("arial.ttf", 16)
            self.code_font = ImageFont.truetype("consola.ttf", 14)
        except:
            self.font = ImageFont.load_default()
            self.code_font = ImageFont.load_default()

    def set_level(self, level):
        self.current_level = level
        self.code_buffer = ""
        self.output_buffer = ""
        self.cursor_pos = 0

    def render(self):
        """Renders the current state to a PIL Image."""
        img = Image.new('RGB', (self.width, self.height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Header
        draw.rectangle([0, 0, self.width, 40], fill='#306998') # Python Blue
        draw.text((10, 10), "LearnPython.org Simulator", font=self.font, fill='white')
        
        if self.current_level:
            level_text = f"Level {self.current_level['id']}: {self.current_level['desc']}"
            draw.text((400, 10), level_text, font=self.font, fill='white')
            
            # Lesson Text
            draw.text((10, 50), f"Mission: {self.current_level['desc']}", font=self.font, fill='black')
            draw.text((10, 80), f"Type this:\n{self.current_level['text']}", font=self.font, fill='black')
        
        # Code Editor
        editor_y = 200
        draw.rectangle([10, editor_y, self.width-10, editor_y+150], fill='#f0f0f0', outline='gray')
        draw.text((15, editor_y+5), "Code Editor (main.py):", font=self.font, fill='gray')
        
        # Draw Code with Cursor
        code_view = self.code_buffer[:self.cursor_pos] + "|" + self.code_buffer[self.cursor_pos:]
        draw.text((20, editor_y+30), code_view, font=self.code_font, fill='black')
        
        # Output Console
        console_y = 360
        draw.rectangle([10, console_y, self.width-10, console_y+100], fill='black')
        draw.text((15, console_y+5), "Output:", font=self.code_font, fill='white')
        draw.text((20, console_y+25), self.output_buffer, font=self.code_font, fill='#00ff00')
        
        return img

    def type_key(self, char):
        """Handles typing characters."""
        self.code_buffer += char
        self.cursor_pos += 1

    def backspace(self):
        if self.cursor_pos > 0:
            self.code_buffer = self.code_buffer[:-1]
            self.cursor_pos -= 1

    def run_code(self, setup_code=None):
        """Executes the code in the buffer."""
        self.output_buffer = ""
        buffer = io.StringIO()
        local_scope = {}
        
        try:
            # Capture stdout
            with contextlib.redirect_stdout(buffer):
                if setup_code:
                    exec(setup_code, {"__builtins__": __builtins__}, local_scope)
                exec(self.code_buffer, {"__builtins__": __builtins__}, local_scope)
            
            self.output_buffer = buffer.getvalue().strip()
            self.last_result = "SUCCESS"
        except Exception as e:
            self.output_buffer = f"Error: {str(e)}"
            self.last_result = "ERROR"
            
        return self.output_buffer, local_scope

def main():
    print("=== Web Coding School (Multi-Level) ===")
    
    # Load Population
    pop_manager = PopulationManager.load()
    if not pop_manager.population:
        print("No population found. Creating genesis...")
        pop_manager.evolve() # Create initial
        
    individual = pop_manager.population[0]
    brain = EvolutionaryBrain(individual['genome'])
    
    # Load Model
    model_path = individual.get('model_path')
    if model_path and os.path.exists(model_path):
        brain.load_model(model_path)
    else:
        print("Starting fresh (no model found).")

    # Boost Dopamine
    brain.chemistry.dopamine = 1.0
    
    emulator = BrowserEmulator()
    curriculum = Curriculum()
    
    # Load Progress from Individual
    start_level = individual.get('school_level', 0)
    curriculum.current_level_idx = start_level
    print(f"Resuming from Level {start_level + 1} (saved in brain)")
    
    # Training Loop
    streak = 0
    max_steps_per_level = 2000
    
    while True:
        level = curriculum.get_current_level()
        print(f"\n--- Starting Level {level['id']}: {level['desc']} ---")
        emulator.set_level(level)
        
        step = 0
        while step < max_steps_per_level:
            step += 1
            action = 0 # Default to No-Op
            
            # Teacher Guidance (Imitation Learning)
            expected_code = level['code']
            current_len = len(emulator.code_buffer)
            is_practice = level.get('practice', False)
            
            # Disable hints for Practice Mode (unless stuck for VERY long)
            if is_practice:
                if step % 200 == 0 and current_len < len(expected_code): # Only hint every 200 steps
                     target_char = expected_code[current_len]
                     print(f"[Teacher]: Practice Mode! Try to remember... (Hint: '{target_char}'?)")
                     
                     # Find action code (Duplicate logic for now)
                     target_action = -1
                     if 'a' <= target_char <= 'z': target_action = ord(target_char) - 97 + 15
                     elif target_char == ' ': target_action = 41
                     elif target_char == '(': target_action = 57
                     elif target_char == ')': target_action = 58
                     elif target_char == '"': target_action = 59
                     elif target_char == ':': target_action = 60
                     elif target_char == '/': target_action = 61
                     elif target_char == '.': target_action = 62
                     elif target_char == ',': target_action = 63
                     elif target_char == '#': target_action = 64
                     elif '0' <= target_char <= '9': target_action = int(target_char) + 42
                     elif target_char == '+': target_action = 52
                     elif target_char == '=': target_action = 53
                     elif target_char == '[': target_action = 54
                     elif target_char == ']': target_action = 55
                     elif target_char == '%': target_action = 56
                     elif target_char == '-': target_action = 61 
                     elif target_char == '*': target_action = 64
                     
                     if target_action != -1:
                        # Inject Knowledge (But DO NOT force action)
                        img = emulator.render()
                        visual_state = brain.retina.process_image(img).unsqueeze(0)
                        semantic_text = level['desc']
                        semantic_vector = brain.broca.process_text(semantic_text).unsqueeze(0)
                        
                        full_input = torch.cat([
                            visual_state, 
                            torch.zeros(1, 256), 
                            semantic_vector, 
                            torch.zeros(1, 1), 
                            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
                            torch.zeros(1, 100), 
                            torch.zeros(1, 151)
                        ], dim=1).squeeze(0)
                        
                        brain.memory.store(full_input, target_action, 1.0)
                        print(f"[Teacher]: Injected knowledge (Action {target_action} for '{target_char}') - YOU MUST ACT!")

                else:
                    # Skip standard guidance
                    pass
            
            # Standard Guidance (Non-Practice)
            elif not is_practice:
                # If code is complete but not running, hint F5
                if emulator.code_buffer == expected_code and step % 50 == 0:
                    print(f"[Teacher]: Code Complete! Press F5 (Action 65) to Run.")
                    target_action = 65
                    
                    # Inject Knowledge
                    img = emulator.render()
                    visual_state = brain.retina.process_image(img).unsqueeze(0)
                    semantic_text = "run code"
                    semantic_vector = brain.broca.process_text(semantic_text).unsqueeze(0)
                    
                    full_input = torch.cat([
                        visual_state, 
                        torch.zeros(1, 256), 
                        semantic_vector, 
                        torch.zeros(1, 1), 
                        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
                        torch.zeros(1, 100), 
                        torch.zeros(1, 151)
                    ], dim=1).squeeze(0)
                    
                    brain.memory.store(full_input, target_action, 1.0)
                    brain.prev_action = target_action
                    print(f"[Teacher]: Injected knowledge (Action 65 for RUN)")
                    
                    # Force Action
                    output, locals_dict = emulator.run_code(setup_code=level.get('setup'))
                    action = 65 
                    # We will fall through to the execution block below!
                    # But we need to skip the brain decision.
                    
                # If stuck or wrong char, guide (ONLY IF NOT PRACTICE)
                elif step % 50 == 0 and current_len < len(expected_code):
                    target_char = expected_code[current_len]
                    print(f"[Teacher]: Hint! Try typing '{target_char}'")
                    
                    # Find action code
                    target_action = -1
                    if 'a' <= target_char <= 'z': target_action = ord(target_char) - 97 + 15
                    elif target_char == ' ': target_action = 41
                    elif target_char == '(': target_action = 57
                    elif target_char == ')': target_action = 58
                    elif target_char == '"': target_action = 59
                    elif target_char == ':': target_action = 60
                    elif target_char == '/': target_action = 61
                    elif target_char == '.': target_action = 62
                    elif target_char == ',': target_action = 63
                    elif target_char == '#': target_action = 64
                    elif '0' <= target_char <= '9': target_action = int(target_char) + 42
                    elif target_char == '+': target_action = 52
                    elif target_char == '=': target_action = 53
                    elif target_char == '[': target_action = 54
                    elif target_char == ']': target_action = 55
                    elif target_char == '%': target_action = 56
                    elif target_char == '-': target_action = 61 # Reuse /?
                    elif target_char == '*': target_action = 64 # Reuse #?
                    
                    if target_action != -1:
                        img = emulator.render()
                        visual_state = brain.retina.process_image(img).unsqueeze(0)
                        semantic_text = level['desc'] # Use description as semantic goal
                        semantic_vector = brain.broca.process_text(semantic_text).unsqueeze(0)
                        
                        full_input = torch.cat([
                            visual_state, 
                            torch.zeros(1, 256), 
                            semantic_vector, 
                            torch.zeros(1, 1), 
                            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
                            torch.zeros(1, 100), 
                            torch.zeros(1, 151)
                        ], dim=1).squeeze(0)
                        
                        brain.memory.store(full_input, target_action, 1.0)
                        brain.prev_action = target_action
                        print(f"[Teacher]: Injected knowledge (Action {target_action} for '{target_char}')")
                    
                    emulator.type_key(target_char)
                    continue
                


            else:
                # 1. Render
                img = emulator.render()
                
                # 2. Brain Decision
                visual_state = brain.retina.process_image(img).unsqueeze(0)
                semantic_text = level['desc']
                semantic_vector = brain.broca.process_text(semantic_text).unsqueeze(0)
                
                full_input = torch.cat([
                    visual_state, 
                    torch.zeros(1, 256), 
                    semantic_vector, 
                    torch.zeros(1, 1), 
                    torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
                    torch.zeros(1, 100), 
                    torch.zeros(1, 151)
                ], dim=1).squeeze(0)
                
                action, _ = brain.decide(full_input)
            
            # 3. Execute
            char = ""
            if 15 <= action <= 40: char = chr(action - 15 + 97)
            elif action == 41: char = " "
            elif 42 <= action <= 51: char = str(action - 42)
            elif action == 52: char = "+"
            elif action == 53: char = "="
            elif action == 54: char = "["
            elif action == 55: char = "]"
            elif action == 56: char = "%"
            elif action == 57: char = "("
            elif action == 58: char = ")"
            elif action == 59: char = '"'
            elif action == 60: char = ":"
            elif action == 61: char = "/" # or -
            elif action == 62: char = "."
            elif action == 63: char = ","
            elif action == 64: char = "#" # or *
            
            reward = -0.01 # Time Cost (Encourage Speed)
            
            if action == 0: # Idleness Penalty
                reward -= 0.1
            
            if char:
                emulator.type_key(char)
                # Shaping Reward
                if emulator.code_buffer == expected_code[:len(emulator.code_buffer)]:
                    reward += 2.0
                else:
                    reward -= 5.0
                    emulator.backspace()
            
            # CRITICAL FIX: Update the last memory with the REAL reward
            brain.memory.update_last_reward(reward)
            
            if action == 9: # Backspace
                emulator.backspace()
                reward -= 0.1
                
            if action == 65: # RUN
                output, locals_dict = emulator.run_code(setup_code=level.get('setup'))
                print(f"Agent ran code: '{emulator.code_buffer}' -> Output: '{output}'")
                
                success = False
                if level['type'] == 'output':
                    if output == level['expected']: success = True
                elif level['type'] == 'var':
                    val = locals_dict.get(level['var_name'])
                    if val == level['expected']: success = True
                    # Handle class check
                    if level['expected'] == 'class' and type(val).__name__ == 'type': success = True
                elif level['type'] == 'run_check':
                    # For functions, we run the check code
                    try:
                        buffer = io.StringIO()
                        with contextlib.redirect_stdout(buffer):
                            exec(level['check_code'], {"__builtins__": __builtins__}, locals_dict)
                        if buffer.getvalue().strip() == level['expected']: success = True
                    except:
                        pass

                if success:
                    reward = 20.0
                    print(f"*** LEVEL {level['id']} COMPLETE! ***")
                    brain.mutate_adaptive(reward)
                    
                    # Save Progress to Individual
                    individual['school_level'] = curriculum.current_level_idx + 1
                    pop_manager.save()
                    print(f"Progress Saved: Level {individual['school_level']}")
                    
                    if not curriculum.next_level():
                        print("ALL LEVELS COMPLETE! YOU ARE A PYTHON MASTER!")
                        return
                    break # Next level
                else:
                    reward = -1.0
                    print("Wrong Result.")
            
            brain.mutate_adaptive(reward)
            
            # Display
            if 'img' not in locals():
                img = emulator.render()
                
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imshow("Web School", img_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
