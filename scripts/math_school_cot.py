import time
import numpy as np
import cv2
import torch
import random
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.population import PopulationManager
from tools.control import ControlInterface

# --- Configuration ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FONT_SIZE = 32
MAX_STEPS = 200  # Increased for longer sequences

class Curriculum:
    def __init__(self):
        self.level = 1
        self.streak = 0
        self.history = []
    
    def get_problem(self):
        """Returns (question_text, expected_sequence_text)"""
        if self.level == 1:
            # Level 1: Single Digit Addition (Show Work)
            # 2 + 3 -> "2+3=5"
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            q = f"{a}+{b}"
            ans = a + b
            cot = f"{a}+{b}={ans}"
            return q, cot
            
        elif self.level == 2:
            # Level 2: Two Digit Addition (No Carry)
            # 12 + 13 -> "2+3=5\n1+1=2\n=25"
            # We select numbers that don't carry
            a1 = random.randint(1, 4)
            a2 = random.randint(0, 4)
            b1 = random.randint(1, 4)
            b2 = random.randint(0, 4)
            
            num_a = a1 * 10 + a2
            num_b = b1 * 10 + b2
            
            q = f"{num_a}+{num_b}"
            
            # Step 1: Ones
            ones_a = a2
            ones_b = b2
            ones_sum = ones_a + ones_b
            
            # Step 2: Tens
            tens_a = a1
            tens_b = b1
            tens_sum = tens_a + tens_b
            
            final = num_a + num_b
            
            cot = f"{ones_a}+{ones_b}={ones_sum}\n{tens_a}+{tens_b}={tens_sum}\n={final}"
            return q, cot
            
        elif self.level == 3:
            # Level 3: Two Digit Addition (With Carry)
            # 15 + 17 -> "5+7=12\n1+1+1=3\n=32"
            a = random.randint(10, 49)
            b = random.randint(10, 49)
            # Ensure carry for better training
            if (a % 10) + (b % 10) < 10:
                b += 5 # Force carry
                
            q = f"{a}+{b}"
            
            ones_a = a % 10
            ones_b = b % 10
            ones_sum = ones_a + ones_b
            carry = 1 if ones_sum >= 10 else 0
            
            tens_a = a // 10
            tens_b = b // 10
            tens_sum = tens_a + tens_b + carry
            
            final = a + b
            
            line1 = f"{ones_a}+{ones_b}={ones_sum}"
            if carry:
                line2 = f"{tens_a}+{tens_b}+1={tens_sum}"
            else:
                line2 = f"{tens_a}+{tens_b}={tens_sum}"
                
            cot = f"{line1}\n{line2}\n={final}"
            return q, cot
            
        return "1+1", "1+1=2"

    def update(self, correct):
        if correct:
            self.streak += 1
            if self.streak > 10:
                self.level = min(3, self.level + 1)
                self.streak = 0
                print(f"*** PROMOTED TO LEVEL {self.level} ***")
        else:
            self.streak = 0
            # Demotion logic? Maybe not for now.

class CoTTeacher:
    def __init__(self):
        self.curriculum = Curriculum()
        self.control = ControlInterface()
        self.font = ImageFont.load_default() # Use default for simplicity or load ttf
        try:
            self.font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            pass
            
    def render(self, question, typed_text, feedback=""):
        img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color=(30, 30, 30))
        d = ImageDraw.Draw(img)
        
        # Header
        d.text((10, 10), f"Math CoT School - Level {self.curriculum.level}", fill=(200, 200, 200), font=self.font)
        
        # Question
        d.text((50, 100), f"Problem: {question}", fill=(255, 255, 0), font=self.font)
        
        # Output Area
        d.text((50, 200), "Reasoning:", fill=(100, 200, 255), font=self.font)
        
        # Render multi-line typed text
        y = 250
        for line in typed_text.split('\n'):
            d.text((50, y), line, fill=(255, 255, 255), font=self.font)
            y += 40
            
        # Feedback
        if feedback:
            d.text((50, 500), feedback, fill=(255, 100, 100), font=self.font)
            
        return np.array(img)

def get_action_char(action):
    # Map actions to characters
    # 42-51 -> 0-9
    if 42 <= action <= 51: return str(action - 42)
    if action == 52: return '+'
    if action == 53: return '-'
    if action == 54: return '*'
    if action == 55: return '='
    if action == 9: return '\n' # Enter
    # Add more mappings as needed from control.py
    return ''

def get_action_code(char):
    if char.isdigit(): return 42 + int(char)
    if char == '+': return 52
    if char == '-': return 53
    if char == '*': return 54
    if char == '=': return 55
    if char == '\n': return 9
    return 0

def main():
    # Load Population
    pop_manager = PopulationManager.load()
    if len(pop_manager.population) == 0:
        pop_manager = PopulationManager(population_size=5)
    
    # Select Champion
    individual = pop_manager.population[0]
    brain = EvolutionaryBrain(individual['genome'])
    
    # Load Model
    model_path = individual.get('model_path')
    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        brain.load_model(model_path)
    
    brain.start()
    teacher = CoTTeacher()
    
    cv2.namedWindow("Math CoT School", cv2.WINDOW_NORMAL)
    
    total_attempts = 0
    
    while True:
        question, expected_cot = teacher.curriculum.get_problem()
        typed_cot = ""
        
        print(f"\nProblem: {question}")
        print(f"Target CoT:\n{expected_cot}")
        
        # Episode Loop
        steps = 0
        done = False
        
        while not done and steps < MAX_STEPS:
            # Render
            img = teacher.render(question, typed_cot)
            cv2.imshow("Math CoT School", img)
            cv2.waitKey(1)
            
            # Inject Vision (Useless for PredictiveRetina, but harmless)
            # brain.retina.process_image(Image.fromarray(img))
            
            # Wake Cycle
            action = brain.wake_cycle()
            
            if isinstance(action, str):
                # print(f"DEBUG: Brain Status: {action}")
                time.sleep(0.1)
                continue
                
            # Process Action
            char = get_action_char(action)
            
            reward = 0.0
            
            if char:
                typed_cot += char
                
                # Check correctness so far
                if expected_cot.startswith(typed_cot):
                    reward = 0.5 # Small reward for correct char
                    print(f"Correct char: {char}")
                    
                    # Check if complete
                    if typed_cot == expected_cot:
                        reward = 10.0
                        print("SUCCESS!")
                        teacher.curriculum.update(True)
                        brain.save_model(model_path or "models/saved/cot_student.pkl")
                        print("Model saved.")
                        done = True
                else:
                    reward = -1.0 # Wrong char
                    print(f"Wrong char: {char}. Expected: {expected_cot[len(typed_cot)-1]}")
                    # Undo typing (visual only)
                    typed_cot = typed_cot[:-1]
                    
            elif action == 0: # STOP
                # If stopped early
                if typed_cot == expected_cot:
                    reward = 10.0
                    brain.save_model(model_path or "models/saved/cot_student.pkl")
                    print("Model saved.")
                    done = True
                else:
                    reward = -0.1 # Penalty for stopping early
            else:
                # Non-typing action (e.g. Mouse Move)
                reward = -0.05 # Small penalty to trigger hint
            
            # Store Memory
            # Hack: We can't easily inject reward into the *past* step in wake_cycle.
            # But we can use the 'pain' signal or 'dopamine' directly.
            brain.chemistry.dopamine = max(0, min(1.0, brain.chemistry.dopamine + reward))
            
            # CRITICAL FIX: Update the last memory with the REAL reward
            brain.memory.update_last_reward(reward)
            
            # CRITICAL FIX 2: Update prev_reward so DDQN learns from this!
            brain.prev_reward = reward
            
            # Teacher Forcing / Hinting
            # If stuck or wrong, inject correct memory
            target_char = expected_cot[len(typed_cot)] if len(typed_cot) < len(expected_cot) else '\n'
            target_action = get_action_code(target_char)
            
            if reward < 0 or steps % 20 == 0:
                # Inject correct memory
                # We need the current state. 
                # brain.prev_state is available.
                if brain.prev_state is not None:
                    brain.memory.store(torch.tensor(brain.prev_state), target_action, 1.0)
                    print(f"Teacher Hint: Injected memory for '{target_char}'")
            
            steps += 1
            
        if not done:
            print("Failed. Resetting.")
            teacher.curriculum.update(False)
            
        total_attempts += 1
        
        # Evolution Check
        if total_attempts % 20 == 0:
            brain.mutate_adaptive(teacher.curriculum.streak / 10.0)
            brain.save_model(model_path or "models/saved/cot_student.pkl")

if __name__ == "__main__":
    main()
