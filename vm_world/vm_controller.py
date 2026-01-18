"""
VM Controller

High-level controller for the VM world.
Coordinates sessions, communication, and lifecycle.
"""

import asyncio
import time
from typing import Optional, List
from dataclasses import dataclass

from .session_manager import SessionManager, AgentSession, get_session_manager
from .communication import CommunicationServer, get_communication_server, MessageType
from .agent_lifecycle import AgentLifecycle, get_lifecycle_manager, DeathCause


@dataclass
class VMWorldConfig:
    """Configuration for the VM world."""
    # Resource limits
    ram_gb: float = 5.0
    cpu_cores: int = 3
    storage_gb: float = 25.0
    
    # Population limits
    max_agents: int = 10
    
    # Agent settings
    max_lifespan_hours: float = 24.0
    initial_energy: float = 100.0
    energy_decay_per_second: float = 0.01
    
    # Screen
    screen_width: int = 1024
    screen_height: int = 768


class VMController:
    """
    Main controller for the Project Genesis VM world.
    
    Coordinates:
    - VM sessions for each agent
    - Communication between agents
    - Agent lifecycle (birth, life, death)
    - Evolution and reproduction
    """
    
    def __init__(self, config: VMWorldConfig = None):
        self.config = config or VMWorldConfig()
        
        # Initialize subsystems
        self.session_manager = get_session_manager()
        self.session_manager.max_sessions = self.config.max_agents
        
        self.comm_server = get_communication_server()
        self.lifecycle = get_lifecycle_manager()
        self.lifecycle.max_lifespan_seconds = self.config.max_lifespan_hours * 3600
        self.lifecycle.initial_energy = self.config.initial_energy
        
        # World state
        self._running = False
        self._tick_count = 0
        self._start_time = None
        
        # Agent brains
        self.agent_brains = {}  # agent_id -> EvolutionaryBrain
        self.agent_genomes = {}  # agent_id -> Genome
    
    def spawn_agent(self, name: str, brain, genome) -> Optional[int]:
        """
        Spawn a new agent with given brain and genome.
        
        Returns agent_id if successful, None otherwise.
        """
        record = self.lifecycle.birth_agent(name, genome)
        if record is None:
            return None
        
        agent_id = record.agent_id
        self.agent_brains[agent_id] = brain
        self.agent_genomes[agent_id] = genome
        
        return agent_id
    
    def kill_agent(self, agent_id: int, cause: DeathCause = DeathCause.MANUAL):
        """Kill an agent."""
        self.lifecycle.kill_agent(agent_id, cause)
        
        # Clean up
        if agent_id in self.agent_brains:
            del self.agent_brains[agent_id]
        if agent_id in self.agent_genomes:
            del self.agent_genomes[agent_id]
    
    def get_agent_count(self) -> int:
        """Get number of living agents."""
        return len(self.lifecycle.get_living_agents())
    
    async def run_tick(self):
        """
        Run one world tick.
        
        Each tick:
        1. Capture screen for each agent
        2. Run brain forward pass
        3. Execute actions in VM
        4. Update energy/lifecycle
        5. Check for mating
        """
        self._tick_count += 1
        
        living = self.lifecycle.get_living_agents()
        
        for record in living:
            agent_id = record.agent_id
            
            if agent_id not in self.agent_brains:
                continue
            
            brain = self.agent_brains[agent_id]
            genome = self.agent_genomes.get(agent_id)
            
            # Get session
            session = self.session_manager.get_session(agent_id)
            if session is None:
                continue
            
            try:
                # 1. Capture screen
                screen = session.get_screen()
                
                # 2. Get brain decision
                # (This would normally call brain.decide() with the screen)
                # For now, just update lifecycle
                
                # 3. Update lifecycle (energy decay)
                action_energy_cost = 0.01  # Base cost
                self.lifecycle.update_agent(agent_id, action_energy_cost)
                
                # 4. Trigger gene expression based on state
                if genome:
                    energy = self.lifecycle.get_agent_energy(agent_id)
                    if energy < 30:  # Low energy = stress
                        genome.on_stress(0.5)
                
            except Exception as e:
                print(f"Tick error for agent {agent_id}: {e}")
    
    async def run_world(self, duration_seconds: float = None):
        """
        Run the world simulation.
        
        Args:
            duration_seconds: How long to run (None = forever)
        """
        self._running = True
        self._start_time = time.time()
        
        print("🌍 Starting Project Genesis world...")
        print(f"   Resources: {self.config.ram_gb}GB RAM, {self.config.cpu_cores} cores")
        print(f"   Max agents: {self.config.max_agents}")
        
        try:
            while self._running:
                await self.run_tick()
                
                # Check duration
                if duration_seconds and (time.time() - self._start_time) > duration_seconds:
                    break
                
                # Rate limiting
                await asyncio.sleep(0.1)  # 10 ticks per second
        
        except KeyboardInterrupt:
            print("\n🛑 World stopped by user")
        
        finally:
            self._running = False
            self._print_stats()
    
    def stop_world(self):
        """Stop the world simulation."""
        self._running = False
    
    def _print_stats(self):
        """Print world statistics."""
        stats = self.lifecycle.get_population_stats()
        print("\n📊 World Statistics:")
        print(f"   Total agents born: {stats['total_born']}")
        print(f"   Currently alive: {stats['alive']}")
        print(f"   Died: {stats['dead']}")
        if stats['dead'] > 0:
            print(f"   Avg lifespan: {stats['avg_lifespan_hours']:.2f} hours")
        print(f"   Total offspring: {stats['total_offspring']}")


def create_world(config: VMWorldConfig = None) -> VMController:
    """Create a new VM world."""
    return VMController(config)
