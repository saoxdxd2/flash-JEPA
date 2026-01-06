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
MAX_STEPS = 300

class LanguageCurriculum:
    def __init__(self):
        self.level = 1
        self.streak = 0
        
        # Level 1: Basic Nouns (Spelling)
        self.nouns = ["cat", "dog", "ball", "tree", "book", "cup", "hat", "sun", "moon", "fish"]
        
        # Level 2: Verbs (Action)
        self.verbs = ["run", "jump", "eat", "sleep", "read", "write", "play", "walk", "fly", "swim"]
        
        # Level 3: Simple Sentences
        self.sentences = [
            "the cat runs",
            "the dog jumps",
            "i see a ball",
            "read the book",
            "hello world"
        ]
    
    def get_problem(self):
        """Returns (prompt_text, expected_text)"""
        if self.level == 1:
            word = random.choice(self.nouns)
            return f"Type: {word}", word
            
        elif self.level == 2:
            word = random.choice(self.verbs)
            return f"Action: {word}", word
            
        elif self.level == 3:
            sent = random.choice(self.sentences)
            return f"Say: {sent}", sent
            
        return "Type: hello", "hello"

    def update(self, correct):
        if correct:
            self.streak += 1
            if self.streak > 10:
                self.level = min(3, self.level + 1)
                self.streak = 0
                print(f"*** PROMOTED TO LEVEL {self.level} ***")
        else:
            self.streak = 0

class LanguageTeacher:
    def __init__(self):
        self.curriculum = LanguageCurriculum()
        self.control = ControlInterface()
        self.font = ImageFont.load_default()
        try:
            self.font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            pass
            
    def render(self, prompt, typed_text, feedback=""):
        img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color=(40, 30, 60)) # Purple background
        d = ImageDraw.Draw(img)
        
        # Header
        d.text((10, 10), f"Language School (NLHF) - Level {self.curriculum.level}", fill=(200, 200, 200), font=self.font)
        
        # Prompt
        d.text((50, 100), prompt, fill=(255, 255, 0), font=self.font)
        
        # Output Area
        d.text((50, 200), "Your Output:", fill=(100, 200, 255), font=self.font)
        d.text((50, 250), typed_text + "_", fill=(255, 255, 255), font=self.font)
            
        # Feedback
        if feedback:
            d.text((50, 400), feedback, fill=(255, 100, 100), font=self.font)
            
        return np.array(img)

def get_action_char(action):
    # Map actions to characters
    # 42-51 -> 0-9
    if 42 <= action <= 51: return str(action - 42)
    # 15-40 -> a-z (approximate mapping, need to check control.py)
    # control.py maps 15-40 to a-z
    if 15 <= action <= 40:
        return chr(ord('a') + (action - 15))
        
    if action == 52: return '+'
    if action == 53: return '-'
    if action == 54: return '*'
    if action == 55: return '='
    if action == 9: return '\n' # Enter
    if action == 66: return ' ' # Space (Assuming 66 is space, need to verify)
    
    return ''

def get_action_code(char):
    if char.isdigit(): return 42 + int(char)
    if 'a' <= char <= 'z': return 15 + (ord(char) - ord('a'))
    if char == ' ': return 66 # Placeholder for space
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
    teacher = LanguageTeacher()
    
    cv2.namedWindow("Language School", cv2.WINDOW_NORMAL)
    
    total_attempts = 0
    
    while True:
        prompt, expected_text = teacher.curriculum.get_problem()
        typed_text = ""
        
        print(f"\nPrompt: {prompt}")
        print(f"Target: {expected_text}")
        
        # Episode Loop
        steps = 0
        done = False
        
        while not done and steps < MAX_STEPS:
            # Render
            img = teacher.render(prompt, typed_text)
            cv2.imshow("Language School", img)
            cv2.waitKey(1)
            
            # Inject Vision
            brain.retina.process_image(Image.fromarray(img))
            
            # Wake Cycle
            action = brain.wake_cycle()
            
            if isinstance(action, str):
                continue
            
            # Process Action
            char = get_action_char(action)
            
            reward = 0.0
            
            if char:
                typed_text += char
                
                # Check correctness so far
                if expected_text.startswith(typed_text):
                    reward = 0.5 # Small reward for correct char
                    print(f"Correct char: {char}")
                    
                    # Check if complete
                    if typed_text == expected_text:
                        reward = 10.0
                        print("SUCCESS!")
                        teacher.curriculum.update(True)
                        teacher.curriculum.update(True)
                        brain.save_model(model_path or f"models/saved/gen_{pop_manager.generation}_elite.pt")
                        print("Model saved.")
                        done = True
                else:
                    reward = -1.0 # Wrong char
                    print(f"Wrong char: {char}. Expected: {expected_text[len(typed_text)-1]}")
                    # Undo typing (visual only)
                    typed_text = typed_text[:-1]
                    
            elif action == 0: # STOP
                pass
            else:
                # Non-typing action (e.g. Mouse Move)
                reward = -0.05 # Small penalty to trigger hint
            
            # Store Memory & Reward
            if not isinstance(action, str):
                 # Manually update dopamine since we nerfed agency reward
                 brain.chemistry.dopamine = max(0, min(1.0, brain.chemistry.dopamine + reward))
                 
                 # CRITICAL FIX: Update the last memory with the REAL reward
                 brain.memory.update_last_reward(reward)
                 
                 # Teacher Forcing / Hinting
                 target_char = expected_text[len(typed_text)] if len(typed_text) < len(expected_text) else '\n'
                 target_action = get_action_code(target_char)
                 
                 if reward < 0 or steps % 20 == 0:
                     if brain.prev_state is not None:
                         # Pad to full input size (1024)
                         padded_state = torch.zeros(brain.input_size)
                         padded_state[:len(brain.prev_state)] = torch.tensor(brain.prev_state)
                         brain.memory.store(padded_state, target_action, 1.0)
                         print(f"Teacher Hint: Injected memory for '{target_char}'")
            
            steps += 1
            
        if not done:
            print("Failed. Resetting.")
            teacher.curriculum.update(False)
            
        total_attempts += 1
        
        # Evolution Check
        if total_attempts % 20 == 0:
            brain.mutate_adaptive(teacher.curriculum.streak / 10.0)
            brain.save_model(model_path or f"models/saved/gen_{pop_manager.generation}_elite.pt")

if __name__ == "__main__":
    main()
