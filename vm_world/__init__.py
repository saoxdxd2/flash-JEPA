"""
Project Genesis: VM World Infrastructure

This module provides the core infrastructure for running AI agents
in a shared Windows VM with RDP sessions.

Architecture:
- Single Windows VM with Remote Desktop Services
- Each agent gets their own RDP session
- Agents share: 5GB RAM, 3 cores, 25GB storage
- Communication via WebSocket server
"""

from .session_manager import SessionManager, AgentSession
from .communication import CommunicationServer
from .agent_lifecycle import AgentLifecycle
from .vm_controller import VMController
from .vm_agent import VMAgent

__all__ = [
    'SessionManager',
    'AgentSession', 
    'CommunicationServer',
    'AgentLifecycle',
    'VMController',
    'VMAgent',
]

# Resource constraints (tight for evolutionary pressure)
VM_CONFIG = {
    'ram_gb': 5,
    'cpu_cores': 3,
    'storage_gb': 25,
    'max_agents': 10,  # Maximum concurrent sessions
}
