import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import mss
from PIL import Image
import threading
import queue
import time
import timm

class PredictiveRetina(nn.Module):
    """
    Biological Foveal Retina.
    Two Streams:
    1. Foveal (High Res, Narrow) -> Ventral Stream (What)
       - Uses Context Optical Compression (timm backbone)
    2. Peripheral (Low Res, Wide) -> Dorsal Stream (Where)
    """
    def __init__(self, latent_size=256, fovea_size=64, genome=None):
        super().__init__()
        self.latent_size = latent_size
        self.fovea_size = fovea_size
        self.genome = genome
        
        # Resolution from Genome if available
        self.peripheral_res = getattr(genome, 'PERIPHERAL_RESOLUTION', 64) if genome else 64
        self.running = False
        self.input_queue = queue.Queue(maxsize=1)
        self.train_lock = threading.Lock()
        
        # Gaze State (0.0 to 1.0)
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        
        # 1. Ventral Stream (What) - Context Optical Compression
        # Using a CPU-optimized timm backbone (FastViT or MobileNetV4)
        try:
            self.foveal_encoder = timm.create_model(
                'fastvit_t8.apple_dist_in1k', 
                pretrained=True, 
                num_classes=latent_size,
                in_chans=3
            )
            print("Retina: Using FastViT-T8 for Context Optical Compression.")
        except Exception as e:
            print(f"Retina: Failed to load FastViT ({e}), falling back to MobileNetV4.")
            self.foveal_encoder = timm.create_model(
                'mobilenetv4_conv_small.e200_r224_in1k',
                pretrained=True,
                num_classes=latent_size,
                in_chans=3
            )

        # 2. Dorsal Stream (Where) - Peripheral CNN
        # Low Res Full Screen -> Spatial/Motion Map
        self.peripheral_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (self.peripheral_res // 4) * (self.peripheral_res // 4), latent_size)
        )
        
        # 3. Decoder (Reconstructs Fovea to minimize surprise)
        # We build this dynamically based on fovea_size
        self.decoder = self._build_decoder()
        
        self.criterion = nn.MSELoss()
        
        # Apply Orthogonal Initialization
        self._init_weights()
        
        # Initialize Optimizer LAST to ensure all parameters are captured
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
        # Apply Orthogonal Initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.orthogonal_(m.weight, gain=2.0) # Reduced from 10.0 to avoid saturation
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _build_decoder(self):
        """Builds a decoder that scales from 8x8 to fovea_size."""
        layers = [
            nn.Linear(self.latent_size, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8))
        ]
        
        curr_res = 8
        curr_channels = 128
        
        while curr_res < self.fovea_size:
            next_channels = max(32, curr_channels // 2)
            layers.append(nn.ConvTranspose2d(curr_channels, next_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
            curr_res *= 2
            curr_channels = next_channels
            
        layers.append(nn.Conv2d(curr_channels, 3, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def start(self):
        self.running = True
        threading.Thread(target=self._vision_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def move_eyes(self, dx, dy):
        """Saccade: Move gaze relative to current position."""
        self.gaze_x = max(0.0, min(1.0, self.gaze_x + dx))
        self.gaze_y = max(0.0, min(1.0, self.gaze_y + dy))

    def set_resolution(self, new_size):
        """Evolutionary Upgrade: Change fovea resolution."""
        if new_size == self.fovea_size: return
        print(f"Retina: Upgrading fovea resolution to {new_size}x{new_size}...")
        self.fovea_size = new_size
        
        # Rebuild decoder for new resolution
        self.decoder = self._build_decoder()
        if next(self.parameters()).is_cuda:
            self.decoder = self.decoder.cuda()
            
        # Re-init optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def set_peripheral_resolution(self, new_res):
        """Evolutionary Upgrade: Change peripheral resolution."""
        if new_res == self.peripheral_res: return
        print(f"Retina: Upgrading peripheral resolution to {new_res}x{new_res}...")
        self.peripheral_res = new_res
        
        # Rebuild peripheral encoder
        old_encoder = self.peripheral_encoder
        self.peripheral_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (self.peripheral_res // 4) * (self.peripheral_res // 4), self.latent_size)
        )
        
        if next(self.parameters()).is_cuda:
            self.peripheral_encoder = self.peripheral_encoder.cuda()
            
        # Re-init optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def resize_latent(self, new_latent_size):
        """Evolutionary Upgrade: Change latent dimension."""
        if new_latent_size == self.latent_size:
            return
            
        print(f"Retina: Resizing Latent Dim {self.latent_size} -> {new_latent_size}")
        min_out = min(self.latent_size, new_latent_size)
        
        # 1. Resize Foveal Encoder
        if hasattr(self.foveal_encoder, 'reset_classifier'):
            # timm standard way
            self.foveal_encoder.reset_classifier(num_classes=new_latent_size)
        elif hasattr(self.foveal_encoder, 'head') and hasattr(self.foveal_encoder.head, 'fc'):
            old_linear = self.foveal_encoder.head.fc
            new_linear = nn.Linear(old_linear.in_features, new_latent_size)
            with torch.no_grad():
                min_out = min(self.latent_size, new_latent_size)
                new_linear.weight[:min_out, :] = old_linear.weight[:min_out, :]
                new_linear.bias[:min_out] = old_linear.bias[:min_out]
            self.foveal_encoder.head.fc = new_linear
        else:
            # Fallback for Sequential or other models
            try:
                old_linear = self.foveal_encoder[-1]
                new_linear = nn.Linear(old_linear.in_features, new_latent_size)
                with torch.no_grad():
                    min_out = min(self.latent_size, new_latent_size)
                    new_linear.weight[:min_out, :] = old_linear.weight[:min_out, :]
                    new_linear.bias[:min_out] = old_linear.bias[:min_out]
                self.foveal_encoder[-1] = new_linear
            except Exception as e:
                print(f"Retina: Warning - Could not resize foveal_encoder head: {e}")
        
        # 2. Resize Peripheral Encoder
        old_linear = self.peripheral_encoder[-1]
        new_linear = nn.Linear(old_linear.in_features, new_latent_size)
        with torch.no_grad():
            new_linear.weight[:min_out, :] = old_linear.weight[:min_out, :]
            new_linear.bias[:min_out] = old_linear.bias[:min_out]
        self.peripheral_encoder[-1] = new_linear
        
        # 3. Resize Decoder (Linear layer at the start)
        old_linear = self.decoder[0]
        new_linear = nn.Linear(new_latent_size, old_linear.out_features)
        with torch.no_grad():
            new_linear.weight[:, :min_out] = old_linear.weight[:, :min_out]
            new_linear.bias[:] = old_linear.bias[:]
        self.decoder[0] = new_linear
        
        self.latent_size = new_latent_size
        
        # Re-initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def _vision_loop(self):
        with mss.mss() as sct:
             # Get primary monitor
            monitor = sct.monitors[1]
            w = monitor["width"]
            h = monitor["height"]
            
            while self.running:
                try:
                    # print("DEBUG: Retina Loop Tick")
                    start_time = time.time()
                    
                    # Capture Screen
                    sct_img = sct.grab(monitor)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    # img_gray = img.convert('L') # REMOVED: Keep RGB
                    
                    # 1. Extract Fovea (High Res Patch)
                    # Sync gaze with actual mouse position if available
                    import pyautogui
                    mx, my = pyautogui.position()
                    self.gaze_x = mx / w
                    self.gaze_y = my / h
                    
                    # Calculate pixel coordinates
                    cx, cy = mx, my
                    half_fovea = self.fovea_size // 2
                    
                    # Crop with bounds checking
                    left = max(0, cx - half_fovea)
                    top = max(0, cy - half_fovea)
                    right = min(w, cx + half_fovea)
                    bottom = min(h, cy + half_fovea)
                    
                    fovea_patch = img.crop((left, top, right, bottom)) # Crop RGB
                    fovea_patch = fovea_patch.resize((self.fovea_size, self.fovea_size)) # Dynamic Resolution
                    
                    # 2. Extract Peripheral (Low Res Full Screen)
                    peripheral_img = img.resize((self.peripheral_res, self.peripheral_res)) # Resize RGB
                    
                    # Convert to Tensors
                    # [H, W, C] -> [C, H, W]
                    fovea_np = np.array(fovea_patch).transpose(2, 0, 1)
                    fovea_tensor = torch.from_numpy(fovea_np).float() / 255.0
                    fovea_tensor = fovea_tensor.unsqueeze(0) # [1, 3, 64, 64]
                    
                    peripheral_np = np.array(peripheral_img).transpose(2, 0, 1)
                    peripheral_tensor = torch.from_numpy(peripheral_np).float() / 255.0
                    peripheral_tensor = peripheral_tensor.unsqueeze(0) # [1, 3, 64, 64]

                    # Online Learning Step (Thread Safe)
                    with self.train_lock:
                        self.optimizer.zero_grad()
                        
                        # Encode
                        foveal_latent = self.foveal_encoder(fovea_tensor)
                        peripheral_latent = self.peripheral_encoder(peripheral_tensor)
                        
                        # Decode (Predict Fovea from Foveal Latent)
                        reconstructed = self.decoder(foveal_latent)
                        
                        # Loss (Minimize Reconstruction Error)
                        loss = self.criterion(reconstructed, fovea_tensor)
                        loss.backward()
                        self.optimizer.step()
                    
                    surprise = loss.item()
                    
                    # Calculate Text Density (Language Instinct) on Fovea
                    # Sobel Edge Detection (Convert to Gray for this)
                    fovea_gray = fovea_patch.convert('L')
                    fovea_np_gray = np.array(fovea_gray)
                    sobel_x = cv2.Sobel(fovea_np_gray, cv2.CV_64F, 1, 0, ksize=3)
                    text_density = np.mean(np.abs(sobel_x)) / 255.0
                    
                    # Put in Queue
                    if self.input_queue.full():
                        self.input_queue.get()
                    
                    # Return: Foveal Latent (What), Peripheral Latent (Where), Surprise, Text Density, Raw Fovea (For OCR)
                    self.input_queue.put((
                        foveal_latent.detach().numpy().flatten(), 
                        peripheral_latent.detach().numpy().flatten(),
                        surprise, 
                        text_density,
                        fovea_tensor.detach().cpu().numpy() # [1, 3, H, W]
                    ))
                    
                    # Cap FPS (10Hz)
                    elapsed = time.time() - start_time
                    if elapsed < 0.1:
                        time.sleep(0.1 - elapsed)
                    
                except Exception as e:
                    print(f"Retina Error: {e}")
                    time.sleep(1)

    def get_latest_input(self):
        if not self.input_queue.empty():
            return self.input_queue.get()
        return None
    def process_image(self, img_pil):
        """
        Process an external PIL image (e.g., from Teacher or Test).
        Returns: Foveal Latent Vector (Tensor)
        """
        # Resize to Fovea Size
        fovea_patch = img_pil.resize((self.fovea_size, self.fovea_size))
        
        # Convert to Tensor
        fovea_np = np.array(fovea_patch).transpose(2, 0, 1)
        fovea_tensor = torch.from_numpy(fovea_np).float() / 255.0
        fovea_tensor = fovea_tensor.unsqueeze(0) # [1, 3, H, W]
        
        # Encode
        with torch.no_grad():
            foveal_latent = self.foveal_encoder(fovea_tensor)
            
        return foveal_latent.squeeze(0) # [256]

    def train_on_image(self, img_pil):
        """
        N2N2 Helper: Train the retina on a specific image (e.g., from Z-Image-Turbo).
        This updates the weights of the encoder/decoder.
        """
        # Resize to Fovea Size
        fovea_patch = img_pil.resize((self.fovea_size, self.fovea_size))
        
        # Convert to Tensor
        fovea_np = np.array(fovea_patch).transpose(2, 0, 1)
        fovea_tensor = torch.from_numpy(fovea_np).float() / 255.0
        fovea_tensor = fovea_tensor.unsqueeze(0) # [1, 3, H, W]
        
        # Training Step (Thread Safe)
        with self.train_lock:
            self.optimizer.zero_grad()
            
            # Encode
            foveal_latent = self.foveal_encoder(fovea_tensor)
            
            # Decode (Reconstruct)
            reconstructed = self.decoder(foveal_latent)
            
            # Loss (Minimize Reconstruction Error)
            loss = self.criterion(reconstructed, fovea_tensor)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    def train_on_target(self, img_pil, target_vec):
        """
        N2N2 Helper: Train retina to match a specific Target Vector (from Teacher).
        """
        # Resize to Fovea Size
        fovea_patch = img_pil.resize((self.fovea_size, self.fovea_size))
        
        # Convert to Tensor
        fovea_np = np.array(fovea_patch).transpose(2, 0, 1)
        fovea_tensor = torch.from_numpy(fovea_np).float() / 255.0
        fovea_tensor = fovea_tensor.unsqueeze(0) # [1, 3, H, W]
        
        # Training Step (Thread Safe)
        with self.train_lock:
            self.optimizer.zero_grad()
            
            # Encode
            foveal_latent = self.foveal_encoder(fovea_tensor)
            
            # Loss: Match the Teacher's Vector (MSE)
            # We assume target_vec is already projected to [256]
            loss = nn.MSELoss()(foveal_latent, target_vec.unsqueeze(0))
            
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
