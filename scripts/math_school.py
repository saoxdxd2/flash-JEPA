import time
import random
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n import KnowledgeLoader
from brain.genome import Genome

from brain.population import PopulationManager

def decode_action(action_code):
    """Decodes action code to character."""
    if 15 <= action_code <= 40: # A-Z
        return chr(action_code - 15 + 65)
    if action_code == 41: return " "
    if 42 <= action_code <= 51: return str(action_code - 42) # 0-9
    if action_code == 52: return "+"
    if action_code == 53: return "-"
    if action_code == 54: return "*"
    if action_code == 55: return "="
    if action_code == 56: return "?"
    if action_code == 57: return "/"
    return None

def main():
    print("=== Math Training School ===")
    print("Teacher: I will ask questions. You answer. I will correct you.")
    
    # 1. Initialize Population & Select Student
    print("[Teacher]: Selecting the best student from the population...")
    pop_manager = PopulationManager.load()
    
    # We train the "Champion" (Index 0)
    # In a real scenario, we might train the whole population
    individual = pop_manager.population[0]
    genome = individual['genome']
    
    # Ensure genome is compatible with math student needs (optional override)
    # genome.species = 'math_student' 
    
    brain = EvolutionaryBrain(genome)
    
    # Load Saved Model from Population
    model_path = individual.get('model_path')
    
    if model_path and os.path.exists(model_path):
        print(f"\n[Teacher]: Welcome back! Loading your brain from {model_path}...")
        try:
            brain.load_model(model_path)
            print("[Teacher]: Brain loaded successfully. Let's see what you remember!")
        except Exception as e:
            print(f"[Teacher]: Error loading brain: {e}. Starting fresh.")
            n2n = KnowledgeLoader(brain)
            n2n.inject_knowledge()
    else:
        print("\n[Teacher]: New student! Injecting base knowledge...")
        # Inject Base Knowledge
        n2n = KnowledgeLoader(brain)
        n2n.inject_knowledge()
    
    # Boost Dopamine for School (Focus Mode)
    # Fixes startup instability/randomness
    brain.chemistry.dopamine = 1.0
    brain.chemistry.serotonin = 0.5 
    print("[Teacher]: Here is some chocolate (Dopamine Boost). Pay attention!")
    
    # 2. Curriculum
    # Level 1: Number Identification (0-9)
    # Level 2: Simple Addition (1+1, 1+2...)
    # Level 3: Simple Subtraction (Result >= 0)
    # Level 4: Simple Multiplication (Single Digit)
    # Level 5: Simple Division (Integer Result)
    
    lessons = []
    
    # Level 1: Number ID (0-9)
    for i in range(10):
        lessons.append((f"WHAT IS {i}", f"{i} "))
        
    # Level 2: Addition (0-9 + 0-9) -> Sums up to 18
    for i in range(10):
        for j in range(10):
            res = i + j
            lessons.append((f"WHAT IS {i}+{j}", f"{res} "))

    # Level 3: Subtraction (0-18 - 0-9) -> Result >= 0
    for i in range(19):
        for j in range(10):
            if i >= j:
                res = i - j
                lessons.append((f"WHAT IS {i}-{j}", f"{res} "))

    # Level 4: Multiplication (0-9 * 0-9) -> Products up to 81
    for i in range(10):
        for j in range(10):
            res = i * j
            lessons.append((f"WHAT IS {i}*{j}", f"{res} "))
            
    # Level 5: Division (Dividends up to 81)
    for i in range(1, 82):
        for j in range(1, 10):
            if i % j == 0:
                res = i // j
                lessons.append((f"WHAT IS {i}/{j}", f"{res} "))

    # Level 6: Mixed Operations (e.g. 2+3-1)
    import random
    random.seed(42) 
    for _ in range(50):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        c = random.randint(0, 9)
        op1 = random.choice(['+', '-', '*'])
        op2 = random.choice(['+', '-'])
        expr = f"{a}{op1}{b}{op2}{c}"
        try:
            res = eval(expr)
            if res >= 0 and res < 100:
                lessons.append((f"WHAT IS {expr}", f"{res} "))
        except:
            pass

    # === EXTREME CURRICULUM ===
    # Level 7: 2-Digit Addition (10-99 + 10-99)
    for _ in range(50):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        res = a + b
        lessons.append((f"WHAT IS {a}+{b}", f"{res} "))

    # Level 8: 2-Digit Subtraction (10-99 - 10-99)
    for _ in range(50):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        if a >= b:
            res = a - b
            lessons.append((f"WHAT IS {a}-{b}", f"{res} "))

    # Level 9: Hard Multiplication (10-20 * 2-9)
    for _ in range(50):
        a = random.randint(10, 20)
        b = random.randint(2, 9)
        res = a * b
        lessons.append((f"WHAT IS {a}*{b}", f"{res} "))

    # Level 10: Complex Mixed (a*b+c)
    for _ in range(50):
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        c = random.randint(10, 99)
        expr = f"{a}*{b}+{c}"
        res = eval(expr)
        lessons.append((f"WHAT IS {expr}", f"{res} "))

    print(f"\n[Teacher]: Generated {len(lessons)} Extreme Math Lessons.")
    
    # Vocab Map for Input
    vocab = {chr(65+i): i for i in range(26)}
    vocab[' '] = 26
    for i in range(10):
        vocab[str(i)] = 27 + i
    vocab['+'] = 37
    vocab['-'] = 38
    vocab['*'] = 39
    vocab['='] = 40
    vocab['?'] = 41
    vocab['/'] = 42 
    
    # DEBUG: Check Brain Structure
    print(f"DEBUG: Brain Input Size: {brain.input_size}")
    print(f"DEBUG: Brain Latent Dim: {brain.genome.latent_dim}")
    print(f"DEBUG: TRM Type: {type(brain.trm)}")
    if hasattr(brain.trm, 'visual_cortex'):
        print(f"DEBUG: Visual Cortex Output Size: {brain.trm.visual_cortex.output_size}")
    
    # Action Map for Correction
    def get_action_code(char):
        if 'A' <= char <= 'Z': return ord(char) - 65 + 15
        if char == ' ': return 41
        if '0' <= char <= '9': return int(char) + 42
        return 0

    correct_streak = 0
    total_attempts = 0
    
    while True:
        # Pick a random lesson
        question, expected = random.choice(lessons)
        expected = expected.strip() # Remove trailing space
        
        print(f"\nQuestion: {question}")
        
        # 1. Encode Input
        semantic_vector = brain.broca.process_text(question)
        
        # Construct State
        # Dynamic Indexing based on Latent Dim
        L = brain.retina.latent_size
        semantic_start = 2 * L
        semantic_end = 3 * L
        
        full_state = torch.zeros(brain.input_size)
        full_state[semantic_start:semantic_end] = semantic_vector
        
        # Flags (Bio/Context) start after 3*L
        # Surprise (1) + Bio (4) + Auto (100) + Context (151)
        # 760 was originally in Context/Bio area (512+256 = 768 is end of semantic in old model)
        # Old Layout: 0-256 (Fov), 256-512 (Per), 512-768 (Sem), 768...
        # Wait, 760 is INSIDE Semantic in old layout?
        # Old: Foveal(256) + Peripheral(256) + Semantic(256) = 768.
        # So 512:768 is Semantic.
        # 760 is inside Semantic? No, 760 is < 768.
        # If semantic vector is 256 dim, it fills 512 to 768.
        # The code said: full_state[760] = 1.0 # Conversation Flag
        # This overwrites part of the semantic vector! This might be a bug or intentional "flag" embedding.
        # Let's move the flag to the Context area.
        context_start = 3 * L + 105 # After Auto
        # Or just put it in the "Bio" or "Surprise" area?
        # Let's put it in the Context area which is safe.
        # Context starts at 3*L + 105.
        full_state[context_start] = 1.0 # Conversation Flag
        
        # Thinking Loop (Multi-Step Output)
        max_thinking_steps = 20 # Increased for multi-digit
        agent_response = ""
        silence_counter = 0
        
        start_time = time.time()
        
        # Reset Brain's Short-Term Memory (Autoregressive State)
        brain.reset_memory()
        
        # Autoregressive Indices
        auto_start = 3 * L + 5
        
        for step in range(max_thinking_steps):
            # Time encoding (in Context area)
            full_state[context_start + 13] = float(step) / max_thinking_steps 
            
            # 2. Agent Responds
            # brain.decide() handles autoregressive state injection (indices 800-900 in old)
            # We need to update decide() too? 
            # brain.decide() uses hardcoded indices! I need to check evolutionary_brain.py decide() method.
            # But for now, let's update this script.
            action, _ = brain.decide(full_state)
            
            step_char = ""
            decoded = decode_action(action)
            if decoded:
                step_char = decoded
            
            # Accumulate Output
            if action != 0 and step_char != "":
                agent_response += step_char
                silence_counter = 0 # Reset silence
                # print(f"Agent said: {step_char}")
            else:
                if len(agent_response) > 0:
                    silence_counter += 1
            
            # Stop Conditions
            # 1. Silence for 2 steps after starting to speak
            if silence_counter >= 2 and len(agent_response) > 0:
                break
                
            # 2. Max length reached (e.g. 3 digits)
            if len(agent_response) >= len(expected) + 1:
                break
                
            # 3. Exact match length (optimization for speed)
            if len(agent_response) == len(expected):
                pass

            time.sleep(0.05) # Faster thinking
        
        end_time = time.time()
        reaction_time = end_time - start_time
        
        if agent_response == "":
            print("Teacher: Time's up! You didn't answer.")
            agent_response = "TIMEOUT"
            action = 0 
        
        print(f"Agent Answered: '{agent_response}' (Expected: '{expected}') | RT: {reaction_time:.4f}s")
        
        # 3. Teacher Feedback
        # EXTREME MODE: We require the FULL answer now.
        if agent_response == expected:
            print("Teacher: CORRECT! (+1.0 Reward)")
            
            # Reinforce the FULL SEQUENCE
            # We must replay the correct sequence to the brain so it learns the transitions.
            # State 0 -> Action(Char 1)
            # State 1 (Prev=Char 1) -> Action(Char 2)
            # ...
            # State N (Prev=Char N) -> Action(0) [STOP]
            
            # Reconstruct State for Training
            train_state = torch.zeros(brain.input_size)
            train_state[semantic_start:semantic_end] = semantic_vector # Same Question
            train_state[context_start] = 1.0
            
            current_prev = 0
            
            for char_idx, char in enumerate(expected):
                # Set Time
                train_state[context_start + 13] = float(char_idx) / max_thinking_steps
                
                # Set Prev Action (Autoregressive)
                # Old: 800-900. New: auto_start to auto_start + 100
                train_state[auto_start:auto_start+100] = 0.0
                if current_prev > 0:
                    # Safety check
                    if current_prev < 100:
                        train_state[auto_start + current_prev] = 5.0
                    
                # Target Action
                target_action = get_action_code(char)
                
                # Store
                # brain.memory.store(train_state, action=target_action, reward=1.0) # DEPRECATED
                
                # Add to Replay Buffer for DDQN (Reflex Learning)
                brain.replay_buffer.add(
                    train_state.numpy(),
                    target_action,
                    1.0,
                    train_state.numpy(), # Next state (same for single-step association)
                    False # Not done yet
                )
                
                # Update Prev for next step
                current_prev = target_action
                
            # Teach STOP (Action 0) after sequence
            train_state[context_start + 13] = float(len(expected)) / max_thinking_steps
            train_state[auto_start:auto_start+100] = 0.0
            if current_prev > 0 and current_prev < 100:
                train_state[auto_start + current_prev] = 10.0
            
            # brain.memory.store(train_state, action=0, reward=5.0) # DEPRECATED
            brain.replay_buffer.add(
                train_state.numpy(),
                0,
                5.0,
                train_state.numpy(),
                True # Terminal
            )
            
            # brain.memory.store(train_state, action=0, reward=5.0) # REMOVED
            
            correct_streak += 1
            
            # Save on Success
            brain.save_model(model_path or f"models/saved/gen_{pop_manager.generation}_elite.pt")
            print(f"[Teacher]: Progress saved.")
        else:
            print(f"Teacher: WRONG. The answer is {expected}. (Correction)")
            
            # Correction: Teach the CORRECT SEQUENCE
            # Same logic as above, but this is the "Correction" phase
            
            train_state = torch.zeros(brain.input_size)
            train_state[512:768] = semantic_vector
            train_state[760] = 1.0
            
            current_prev = 0
            
            for char_idx, char in enumerate(expected):
                train_state[773] = float(char_idx) / max_thinking_steps
                
                train_state[800:900] = 0.0
                if current_prev > 0:
                    train_state[800 + current_prev] = 5.0
                    
                target_action = get_action_code(char)
                
                brain.memory.store(train_state, action=target_action, reward=1.0)
                current_prev = target_action
                
            # Teach STOP
            train_state[773] = float(len(expected)) / max_thinking_steps
            train_state[800:900] = 0.0
            if current_prev > 0:
                train_state[800 + current_prev] = 5.0
            brain.memory.store(train_state, action=0, reward=5.0)
            
            # NEGATIVE FEEDBACK FOR OVERSHOOT
            # Teach that any further action is BAD (-1.0)
            train_state[773] = float(len(expected) + 1) / max_thinking_steps
            train_state[800:900] = 0.0
            # We simulate one more step where the agent tries to continue
            # We reinforce STOP at Step N+1 as well to say "You should still be stopped"
            brain.memory.store(train_state, action=0, reward=2.0)
            
            correct_streak = 0
            
        total_attempts += 1
        
        # Evolution / Sleep Cycle
        if total_attempts % 50 == 0:
            print("\n[Teacher]: Nap time! Consolidating memories...")
            
            # Simple Fitness based on recent performance
            fitness = float(correct_streak)
            
            # Update Brain
            brain.mutate_adaptive(fitness/100.0)
            
            # Save
            if not model_path:
                os.makedirs("models/saved", exist_ok=True)
                model_path = f"models/saved/gen_{pop_manager.generation}_elite.pt"
                individual['model_path'] = model_path
            
            # Ensure directory exists (in case it was deleted)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
            brain.save_model(model_path)
            print(f"[Teacher]: Progress saved to {model_path}")
            print("[Teacher]: Wake up! Back to class.")
            
        # Check Graduation (Streak > 100)
        if correct_streak >= 100:
            print("\n[Teacher]: GRADUATED! 100 Correct in a row!")
            print("[Teacher]: Graduation Ceremony! Consolidating knowledge into next generation...")
            
            # Save current progress first
            brain.save_model(model_path)
            individual['genome'] = brain.genome
            
            # CRITICAL FIX: Update fitness so PopulationManager knows this is a Champion
            if len(pop_manager.population) > 0:
                pop_manager.population[0]['fitness'] = float(correct_streak)
                print(f"Population: Updated Champion Fitness to {correct_streak}")
            
            pop_manager.save()
            
            # Evolve Population
            best_genome = pop_manager.evolve()
            pop_manager.save()
            
            print(f"[Teacher]: Evolution Complete. Welcome Generation {pop_manager.generation}!")
            break

if __name__ == "__main__":
    main()
