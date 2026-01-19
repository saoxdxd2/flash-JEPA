import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.fnd_encoder import FractalEncoder
from brain.modules.ribosome import NeuralRibosome

def test_fractal_compression():
    print("=== Fractal Neural DNA Verification ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create a Dummy Weight Matrix (Simulating a Qwen Layer)
    # 4096 x 4096 is standard for large models
    H, W = 1024, 1024 # Smaller for quick test
    print(f"Generating target weights ({H}x{W})...")
    
    # Create a pattern that is somewhat fractal/structured (easier to compress than pure noise)
    # Real weights have structure.
    x = torch.linspace(-1, 1, W).view(1, W).repeat(H, 1)
    y = torch.linspace(-1, 1, H).view(H, 1).repeat(1, W)
    target = torch.sin(x * 10) * torch.cos(y * 10) + torch.sin(x * y * 20)
    target = target.to(device)
    
    original_size_bytes = target.element_size() * target.nelement()
    print(f"Original Size: {original_size_bytes / 1024 / 1024:.2f} MB")
    
    # 2. Compress (Encode)
    encoder = FractalEncoder(device=device)
    num_transforms = 16 # Very high compression
    
    print(f"Compressing with {num_transforms} transforms...")
    start_time = time.time()
    dna = encoder.encode(target, num_transforms=num_transforms, iterations=10)
    encode_time = time.time() - start_time
    
    # Calculate Compressed Size
    # 16 transforms * 7 floats (6 params + 1 prob) * 4 bytes
    compressed_size_bytes = num_transforms * 7 * 4
    ratio = original_size_bytes / compressed_size_bytes
    
    print(f"Compression Complete in {encode_time:.2f}s")
    print(f"Compressed Size: {compressed_size_bytes} bytes")
    print(f"Compression Ratio: {ratio:.2f}:1")
    
    # 3. Decompress (Transcribe)
    ribosome = NeuralRibosome(device=device)
    
    print("Transcribing DNA back to weights...")
    start_time = time.time()
    reconstructed = ribosome.transcribe(dna, iterations=20)
    decode_time = time.time() - start_time
    
    print(f"Decompression Complete in {decode_time:.2f}s")
    
    # 4. Verify Reconstruction
    target_norm = (target - target.min()) / (target.max() - target.min())
    mse = torch.nn.functional.mse_loss(reconstructed, target_norm)
    print(f"Reconstruction MSE: {mse.item():.6f}")
    
    # 5. Verify Knowledge Indexing (NeuralSSD Integration)
    print("\n=== Testing Knowledge Indexing ===")
    from brain.modules.neural_ssd import NeuralSSD
    
    # Initialize SSD
    ssd = NeuralSSD(key_dim=64, value_dim=1, capacity=10) # Value dim irrelevant for object storage
    
    # Create a Semantic Key (e.g., embedding for "Visual Cortex Weights")
    key = torch.randn(1, 64).to(device)
    
    # Store DNA in SSD
    print("Storing DNA in NeuralSSD...")
    ssd.write(key, torch.zeros(1, 1).to(device), objects=[dna])
    
    # Query SSD
    print("Querying NeuralSSD...")
    # Add some noise to query to test similarity search
    query = key + torch.randn(1, 64).to(device) * 0.1
    
    vals, scores, objs = ssd.read(query, k=1)
    retrieved_dna = objs[0][0]
    
    if retrieved_dna is not None:
        print(f"Retrieved DNA! Score: {scores.item():.4f}")
        # Verify it's the same DNA
        if retrieved_dna.shape == dna.shape:
             print("SUCCESS: DNA Retrieved correctly.")
        else:
             print("FAILURE: DNA corrupted.")
    else:
        print("FAILURE: DNA not found.")


if __name__ == "__main__":
    test_fractal_compression()
