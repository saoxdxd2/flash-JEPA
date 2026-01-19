import torch
import torch.nn as nn
import torch.optim as optim

class HamiltonianField(nn.Module):
    """
    Learns the Hamiltonian Energy Landscape H(q, p).
    The gradients of H determine the vector field (motion).
    dq/dt = dH/dp
    dp/dt = -dH/dq
    """
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.dim = dim
        # The network learns the "Shape" of the energy landscape.
        # It takes (q, p) -> outputs a single scalar Energy value (H)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.Softplus(), # Softplus ensures smooth derivatives (vital for physics)
            nn.Linear(hidden, hidden),
            nn.Tanh(),     # Tanh creates bounded "gravity wells"
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x contains [Batch, q, p] (Position and Momentum)
        # We need the gradient of H with respect to inputs to get motion vectors
        with torch.enable_grad():
            x = x.requires_grad_(True)
            H = self.net(x)
            
            # Compute gradients dH/dx
            # create_graph=True allows training the derivative (matching velocity)
            dH_dx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            
        # Symplectic Matrix J (The geometric heart of Hamiltonian mechanics)
        # splits the gradients into dq/dt and dp/dt correctly.
        # J = [[0, I], [-I, 0]]
        dH_dq, dH_dp = torch.chunk(dH_dx, 2, dim=1)
        
        # Hamilton's Equations: dq/dt = dH/dp, dp/dt = -dH/dq
        vector_field = torch.cat([dH_dp, -dH_dq], dim=1)
        
        return vector_field

def integrate_flow(model, z0, t_span, steps):
    """
    Reconstructs the memory by simulating the physics.
    z0: Initial state (The 'Key') [Batch, Dim*2]
    steps: How much memory to generate
    """
    dt = t_span / steps
    trajectory = [z0]
    z = z0
    
    for _ in range(steps):
        # Euler integration on the Symplectic Manifold
        # (For higher precision, we would use Runge-Kutta 4)
        dz_dt = model(z)
        z = z + dz_dt * dt
        trajectory.append(z)
        
    # Stack: [Steps+1, Batch, Dim*2]
    return torch.stack(trajectory, dim=0)

class HamiltonianMemory(nn.Module):
    """
    A module that wraps the Hamiltonian Field and provides
    high-level methods to 'imprint' (learn) and 'recall' (integrate) trajectories.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # input_dim is the size of the vector we want to store (q).
        # We augment it with momentum (p) of same size.
        self.field = HamiltonianField(dim=input_dim, hidden=hidden_dim)
        self.optimizer = optim.Adam(self.field.parameters(), lr=0.01)
        
    def imprint(self, trajectory, epochs=100, verbose=False):
        """
        Learns the laws of physics that generate the given trajectory.
        trajectory: [Steps, Batch, Dim] (Just Position q)
        We infer momentum p from velocity.
        """
        # 1. Infer Momentum (Velocity)
        # dq/dt ~ (q_t+1 - q_t) / dt
        # Let's assume dt=1 for simplicity or infer from data.
        q = trajectory
        v = q[1:] - q[:-1]
        
        # We need (q, p) pairs.
        # Let's say p = v (Mass = 1).
        # Data: [q_t, p_t] -> Target Velocity: [dq/dt, dp/dt]
        # We need next v to get dv/dt (acceleration).
        # So we need 3 points to get 1 training sample for (q, p) -> (v, a).
        # Or simpler: Just match the vector field to the observed flow in phase space.
        
        # Phase Space Construction:
        # State z_t = [q_t, v_t]
        # Target dz_t = [v_t, a_t] where a_t = v_t+1 - v_t
        
        # q: [Steps, Batch, Dim]
        # v: [Steps-1, Batch, Dim]
        # a: [Steps-2, Batch, Dim]
        
        # We align them:
        # z_t = [q_t, v_t] (for t=0 to Steps-3)
        # target_flow = [v_t, a_t]
        
        if q.shape[0] < 3:
            return 0.0 # Not enough data
            
        v = q[1:] - q[:-1]
        a = v[1:] - v[:-1]
        
        # Truncate to match sizes
        q_train = q[:-2]
        v_train = v[:-1]
        
        # Input State z: [Steps-2, Batch, 2*Dim]
        z = torch.cat([q_train, v_train], dim=2)
        
        # Target Flow dz: [Steps-2, Batch, 2*Dim]
        # dq/dt = v
        # dp/dt = a
        target_flow = torch.cat([v_train, a], dim=2)
        
        # Flatten Batch and Steps for training
        z_flat = z.reshape(-1, z.shape[-1])
        target_flat = target_flow.reshape(-1, target_flow.shape[-1])
        
        loss_val = 0
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            pred_flow = self.field(z_flat)
            loss = torch.mean((pred_flow - target_flat)**2)
            
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            
        if verbose:
            print(f"HAM Imprint Loss: {loss_val:.6f}")
            
        return loss_val
        
    def recall(self, start_q, start_v=None, steps=100):
        """
        Generates a trajectory from a starting point.
        """
        if start_v is None:
            start_v = torch.zeros_like(start_q)
            
        z0 = torch.cat([start_q, start_v], dim=1)
        
        # Integrate
        # We assume dt=1.0 because we trained with difference=1.0
        traj_z = integrate_flow(self.field, z0, t_span=steps, steps=steps)
        
        # Extract q (Position)
        # traj_z: [Steps+1, Batch, 2*Dim]
        traj_q = traj_z[:, :, :self.field.dim]
        
        return traj_q
