"""
Project Genesis: Launch Script

Starts the VM world with Adam and Eve.
They have N2N2 knowledge from Qwen via gen_350_transplanted.pt

Usage:
    python scripts/genesis_world.py
"""

import asyncio
import sys
import os
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome
from vm_world.vm_controller import VMController, VMWorldConfig
from vm_world.vm_agent import VMAgent
from vm_world.session_manager import get_session_manager
from vm_world.communication import get_communication_server


def load_genesis_brain(name: str) -> tuple:
    """
    Load brain from gen_350_transplanted.pt
    This brain has N2N2 knowledge from Qwen.
    
    Returns:
        (brain, genome) tuple
    """
    model_path = "models/saved/gen_350_transplanted.pt"
    
    try:
        brain = EvolutionaryBrain.load_model(model_path)
        print(f"✅ Loaded {name} from {model_path}")
        return brain, brain.genome
    except Exception as e:
        print(f"⚠️ Could not load from {model_path}: {e}")
        print(f"   Creating fresh brain for {name}")
        genome = Genome()
        brain = EvolutionaryBrain(genome)
        return brain, genome


class GenesisWorld:
    """
    The Genesis World simulation.
    
    Contains Adam and Eve, monitors their lifecycle,
    and handles reproduction when conditions are met.
    """
    
    def __init__(self, config: VMWorldConfig = None):
        self.config = config or VMWorldConfig(
            ram_gb=5.0,
            cpu_cores=3,
            storage_gb=25.0,
            max_agents=10,
            max_lifespan_hours=24.0,
        )
        
        self.controller = VMController(self.config)
        self.session_manager = get_session_manager()
        self.comm_server = get_communication_server()
        
        # Agents
        self.agents = {}  # agent_id -> VMAgent
        self.adam = None
        self.eve = None
    
    def spawn_adam_and_eve(self):
        """Spawn the first two agents."""
        print("\n🧬 Creating Adam (♂)...")
        adam_brain, adam_genome = load_genesis_brain("Adam")
        adam_genome.species = "male"
        
        # Create session and agent
        adam_session = self.session_manager.create_session("Adam", "agent_adam")
        if adam_session:
            adam_id = self.controller.spawn_agent("Adam", adam_brain, adam_genome)
            if adam_id:
                self.adam = VMAgent(adam_id, "Adam", adam_brain, adam_session)
                self.agents[adam_id] = self.adam
                print(f"   ✅ Adam spawned (ID: {adam_id})")
        
        print("\n🧬 Creating Eve (♀)...")
        eve_brain, eve_genome = load_genesis_brain("Eve")
        eve_genome.species = "female"
        
        eve_session = self.session_manager.create_session("Eve", "agent_eve")
        if eve_session:
            eve_id = self.controller.spawn_agent("Eve", eve_brain, eve_genome)
            if eve_id:
                self.eve = VMAgent(eve_id, "Eve", eve_brain, eve_session)
                self.agents[eve_id] = self.eve
                print(f"   ✅ Eve spawned (ID: {eve_id})")
        
        self._print_gene_summary()
    
    def _print_gene_summary(self):
        """Print gene expression summary for all agents."""
        print("\n" + "=" * 50)
        print("🧬 GENE EXPRESSION SUMMARY")
        print("=" * 50)
        
        for agent_id, agent in self.agents.items():
            genome = agent.genome
            print(f"\n{agent.name} ({genome.species}):")
            print(f"  BDNF (learning):     {genome.get_expression('bdnf'):.3f}")
            print(f"  COMT (dopamine):     {genome.get_expression('comt'):.3f}")
            print(f"  5-HTTLPR (serotonin): {genome.get_expression('sert'):.3f}")
            print(f"  FKBP5 (stress):      {genome.get_expression('fkbp5'):.3f}")
            print(f"  NMDA (learning threshold): {genome.get_expression('grin2a'):.3f}")
    
    async def run_tick(self):
        """Run one simulation tick."""
        for agent_id, agent in list(self.agents.items()):
            try:
                # Run agent tick
                action, energy_cost = agent.tick()
                
                # Update lifecycle
                self.controller.lifecycle.update_agent(agent_id, energy_cost)
                
                # Check if agent died
                if not self.controller.lifecycle.life_records[agent_id].is_alive:
                    print(f"💀 {agent.name} died!")
                    del self.agents[agent_id]
                    
            except Exception as e:
                print(f"Tick error for {agent.name}: {e}")
    
    async def run(self, duration_seconds: float = None):
        """Run the Genesis simulation."""
        print("\n" + "=" * 60)
        print("🌍 PROJECT GENESIS: STARTING SIMULATION")
        print("=" * 60)
        print(f"Resources: {self.config.ram_gb}GB RAM, {self.config.cpu_cores} cores, {self.config.storage_gb}GB storage")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        tick_count = 0
        
        try:
            while True:
                await self.run_tick()
                tick_count += 1
                
                # Status update every 100 ticks
                if tick_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"⏱️ Tick {tick_count} | Elapsed: {elapsed:.1f}s | Agents: {len(self.agents)}")
                
                # Check duration
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Rate limit
                await asyncio.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Simulation stopped by user")
        
        finally:
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final statistics."""
        stats = self.controller.lifecycle.get_population_stats()
        
        print("\n" + "=" * 50)
        print("📊 FINAL STATISTICS")
        print("=" * 50)
        print(f"Total born:     {stats['total_born']}")
        print(f"Still alive:    {stats['alive']}")
        print(f"Died:           {stats['dead']}")
        if stats['dead'] > 0:
            print(f"Avg lifespan:   {stats['avg_lifespan_hours']:.2f} hours")
        print(f"Total offspring: {stats['total_offspring']}")
        
        # Agent-level stats
        for agent_id, agent in self.agents.items():
            stats = agent.get_stats()
            print(f"\n{stats['name']}:")
            print(f"  Actions: {stats['action_count']}")
            print(f"  Reward:  {stats['reward_total']:.2f}")
            print(f"  BDNF:    {stats['bdnf']:.3f}")


async def main():
    """Launch Project Genesis."""
    print("=" * 60)
    print("🌍 PROJECT GENESIS: VM-Based Artificial Life")
    print("=" * 60)
    
    # Create world
    world = GenesisWorld()
    
    # Spawn Adam and Eve
    world.spawn_adam_and_eve()
    
    if world.adam and world.eve:
        print("\n✅ Adam and Eve are alive!")
        
        # Run simulation for 30 seconds (demo)
        await world.run(duration_seconds=30)
    else:
        print("❌ Failed to spawn Adam and Eve")


if __name__ == "__main__":
    asyncio.run(main())
