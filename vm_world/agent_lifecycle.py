"""
Agent Lifecycle Manager

Handles birth, life, and death of agents in the VM world.
Tracks mortality-based lifespan (not generations).
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum

from .session_manager import SessionManager, AgentSession, get_session_manager
from .communication import CommunicationServer, get_communication_server


class DeathCause(Enum):
    """Possible causes of agent death."""
    STARVATION = "starvation"       # Energy depleted
    OLD_AGE = "old_age"             # Exceeded max lifespan
    KILLED = "killed"               # Killed by another agent (if enabled)
    RESOURCE_LIMIT = "resource"     # Out of memory/CPU
    MANUAL = "manual"               # Manually terminated
    

@dataclass
class AgentLifeRecord:
    """
    Record of an agent's life.
    
    Tracks from birth to death, all key events.
    """
    agent_id: int
    name: str
    genome_id: str  # Reference to genome
    
    # Parents (None for Adam/Eve)
    parent_1_id: Optional[int] = None
    parent_2_id: Optional[int] = None
    
    # Life events
    birth_time: float = field(default_factory=time.time)
    death_time: Optional[float] = None
    death_cause: Optional[DeathCause] = None
    
    # Resources consumed during lifetime
    total_energy_consumed: float = 0.0
    total_actions_taken: int = 0
    
    # Offspring produced
    offspring_ids: List[int] = field(default_factory=list)
    
    @property
    def is_alive(self) -> bool:
        return self.death_time is None
    
    @property
    def lifespan_seconds(self) -> float:
        """How long this agent lived (or has lived)."""
        end = self.death_time if self.death_time else time.time()
        return end - self.birth_time
    
    @property
    def lifespan_hours(self) -> float:
        return self.lifespan_seconds / 3600
    
    @property
    def offspring_count(self) -> int:
        return len(self.offspring_ids)


class AgentLifecycle:
    """
    Manages the complete lifecycle of agents.
    
    Responsibilities:
    - Birth: Create new agent with genome
    - Life: Monitor energy, health, aging
    - Reproduction: Handle mating and offspring
    - Death: Clean up resources
    """
    
    def __init__(
        self,
        max_lifespan_hours: float = 24.0,  # Max 24 hours per agent
        initial_energy: float = 100.0,
        energy_decay_per_second: float = 0.01,
    ):
        self.max_lifespan_seconds = max_lifespan_hours * 3600
        self.initial_energy = initial_energy
        self.energy_decay_per_second = energy_decay_per_second
        
        self.session_manager = get_session_manager()
        self.comm_server = get_communication_server()
        
        # All agents (alive and dead)
        self.life_records: Dict[int, AgentLifeRecord] = {}
        self.agent_energy: Dict[int, float] = {}
        
        self._next_agent_id = 1
    
    def birth_agent(
        self,
        name: str,
        genome,  # Genome object
        parent_1_id: Optional[int] = None,
        parent_2_id: Optional[int] = None
    ) -> Optional[AgentLifeRecord]:
        """
        Birth a new agent into the world.
        
        Creates:
        - Life record
        - RDP session
        - Registers with comm server
        """
        agent_id = self._next_agent_id
        self._next_agent_id += 1
        
        # Create session
        session = self.session_manager.create_session(name)
        if session is None:
            print(f"Cannot birth {name}: no available sessions")
            return None
        
        # Create life record
        record = AgentLifeRecord(
            agent_id=agent_id,
            name=name,
            genome_id=str(id(genome)),
            parent_1_id=parent_1_id,
            parent_2_id=parent_2_id,
            birth_time=time.time()
        )
        
        self.life_records[agent_id] = record
        self.agent_energy[agent_id] = self.initial_energy
        
        # Register with communication server
        self.comm_server.register_agent(agent_id, name)
        
        # Update parent offspring lists
        if parent_1_id and parent_1_id in self.life_records:
            self.life_records[parent_1_id].offspring_ids.append(agent_id)
        if parent_2_id and parent_2_id in self.life_records:
            self.life_records[parent_2_id].offspring_ids.append(agent_id)
        
        print(f"🎂 Agent '{name}' (ID: {agent_id}) born!")
        return record
    
    def kill_agent(self, agent_id: int, cause: DeathCause):
        """
        Kill an agent and clean up resources.
        """
        if agent_id not in self.life_records:
            return
        
        record = self.life_records[agent_id]
        if not record.is_alive:
            return  # Already dead
        
        record.death_time = time.time()
        record.death_cause = cause
        
        # Clean up session
        self.session_manager.destroy_session(agent_id)
        
        # Unregister from comm server
        self.comm_server.unregister_agent(agent_id, cause.value)
        
        # Clean up energy tracking
        if agent_id in self.agent_energy:
            del self.agent_energy[agent_id]
        
        print(f"💀 Agent '{record.name}' died: {cause.value} (lived {record.lifespan_hours:.2f} hours)")
    
    def update_agent(self, agent_id: int, energy_cost: float = 0.0):
        """
        Update agent state (call every tick).
        
        Checks for death conditions:
        - Energy depletion (starvation)
        - Age limit (old age)
        """
        if agent_id not in self.life_records:
            return
        
        record = self.life_records[agent_id]
        if not record.is_alive:
            return
        
        # Energy decay
        if agent_id in self.agent_energy:
            self.agent_energy[agent_id] -= self.energy_decay_per_second + energy_cost
            record.total_energy_consumed += self.energy_decay_per_second + energy_cost
            
            # Check starvation
            if self.agent_energy[agent_id] <= 0:
                self.kill_agent(agent_id, DeathCause.STARVATION)
                return
        
        # Check old age
        if record.lifespan_seconds > self.max_lifespan_seconds:
            self.kill_agent(agent_id, DeathCause.OLD_AGE)
            return
        
        record.total_actions_taken += 1
    
    def feed_agent(self, agent_id: int, energy: float):
        """Give energy to an agent (reward for good actions)."""
        if agent_id in self.agent_energy:
            self.agent_energy[agent_id] = min(
                self.initial_energy,  # Cap at initial energy
                self.agent_energy[agent_id] + energy
            )
    
    def get_agent_energy(self, agent_id: int) -> float:
        """Get agent's current energy."""
        return self.agent_energy.get(agent_id, 0.0)
    
    def get_living_agents(self) -> List[AgentLifeRecord]:
        """Get all living agents."""
        return [r for r in self.life_records.values() if r.is_alive]
    
    def get_dead_agents(self) -> List[AgentLifeRecord]:
        """Get all dead agents."""
        return [r for r in self.life_records.values() if not r.is_alive]
    
    def attempt_reproduction(
        self,
        parent_1_id: int,
        parent_2_id: int,
        parent_1_genome,
        parent_2_genome,
    ) -> Optional[Tuple[AgentLifeRecord, object]]:
        """
        Attempt reproduction between two agents.
        
        Returns (child_record, child_genome) if successful.
        """
        # Both parents must be alive
        if parent_1_id not in self.life_records or parent_2_id not in self.life_records:
            return None
        
        p1 = self.life_records[parent_1_id]
        p2 = self.life_records[parent_2_id]
        
        if not p1.is_alive or not p2.is_alive:
            return None
        
        # Both parents need enough energy
        min_energy = self.initial_energy * 0.3  # 30% energy required
        if (self.get_agent_energy(parent_1_id) < min_energy or 
            self.get_agent_energy(parent_2_id) < min_energy):
            return None
        
        # Create child genome through crossover
        child_genome = parent_1_genome.crossover(parent_2_genome)
        child_genome.mutate()
        
        # Name the child
        child_name = f"Child_{p1.name[:2]}{p2.name[:2]}_{random.randint(100, 999)}"
        
        # Energy cost for reproduction
        energy_cost = self.initial_energy * 0.2  # 20% of energy
        self.agent_energy[parent_1_id] -= energy_cost
        self.agent_energy[parent_2_id] -= energy_cost
        
        # Birth the child
        child_record = self.birth_agent(
            name=child_name,
            genome=child_genome,
            parent_1_id=parent_1_id,
            parent_2_id=parent_2_id
        )
        
        if child_record:
            return (child_record, child_genome)
        return None
    
    def get_population_stats(self) -> dict:
        """Get population statistics."""
        living = self.get_living_agents()
        dead = self.get_dead_agents()
        
        return {
            "total_born": len(self.life_records),
            "alive": len(living),
            "dead": len(dead),
            "avg_lifespan_hours": sum(r.lifespan_hours for r in dead) / len(dead) if dead else 0,
            "total_offspring": sum(r.offspring_count for r in self.life_records.values()),
        }


# Singleton instance
_lifecycle_manager = None

def get_lifecycle_manager() -> AgentLifecycle:
    """Get the global lifecycle manager instance."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = AgentLifecycle()
    return _lifecycle_manager
