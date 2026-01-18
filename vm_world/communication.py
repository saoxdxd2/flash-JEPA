"""
Communication Server

WebSocket-based server for inter-agent communication.
Agents can broadcast messages, send direct messages, and
receive system announcements (births, deaths).
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, Any
from enum import Enum


class MessageType(Enum):
    """Types of messages in the system."""
    BROADCAST = "broadcast"          # Message to all agents
    DIRECT = "direct"                # Message to specific agent
    BIRTH = "birth"                  # System: new agent born
    DEATH = "death"                  # System: agent died
    MATING_REQUEST = "mating_req"    # Agent requests to mate
    MATING_ACCEPT = "mating_accept"  # Agent accepts mating
    MATING_REJECT = "mating_reject"  # Agent rejects mating
    RESOURCE_SHARE = "resource"      # Share resources
    LOCATION = "location"            # Share location/state


@dataclass
class Message:
    """A message between agents or from system."""
    msg_type: MessageType
    sender_id: Optional[int]  # None for system messages
    content: Any
    timestamp: float = field(default_factory=time.time)
    target_id: Optional[int] = None  # For direct messages
    
    def to_dict(self) -> dict:
        return {
            "type": self.msg_type.value,
            "sender": self.sender_id,
            "target": self.target_id,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(
            msg_type=MessageType(data["type"]),
            sender_id=data.get("sender"),
            target_id=data.get("target"),
            content=data.get("content"),
            timestamp=data.get("timestamp", time.time())
        )


class CommunicationServer:
    """
    WebSocket server for agent communication.
    
    Features:
    - Broadcast to all agents
    - Direct messaging between agents
    - System announcements (birth/death)
    - Message history
    """
    
    def __init__(self, max_history: int = 1000):
        self.agents: Dict[int, asyncio.Queue] = {}  # agent_id -> message queue
        self.agent_names: Dict[int, str] = {}       # agent_id -> name
        self.message_history: list = []
        self.max_history = max_history
        self._running = False
        self._callbacks: Dict[MessageType, list] = {}
    
    def register_agent(self, agent_id: int, agent_name: str) -> asyncio.Queue:
        """
        Register an agent with the communication server.
        Returns a queue for receiving messages.
        """
        queue = asyncio.Queue()
        self.agents[agent_id] = queue
        self.agent_names[agent_id] = agent_name
        
        # Announce birth to all
        self._announce_birth(agent_id, agent_name)
        
        return queue
    
    def unregister_agent(self, agent_id: int, death_cause: str = "unknown"):
        """Unregister an agent (they died or disconnected)."""
        if agent_id in self.agents:
            agent_name = self.agent_names.get(agent_id, f"Agent-{agent_id}")
            
            # Announce death
            self._announce_death(agent_id, agent_name, death_cause)
            
            del self.agents[agent_id]
            if agent_id in self.agent_names:
                del self.agent_names[agent_id]
    
    async def broadcast(self, sender_id: int, content: Any):
        """Send a message to all agents."""
        message = Message(
            msg_type=MessageType.BROADCAST,
            sender_id=sender_id,
            content=content
        )
        await self._deliver_to_all(message, exclude=[sender_id])
    
    async def send_direct(self, sender_id: int, target_id: int, content: Any) -> bool:
        """
        Send a direct message to a specific agent.
        Returns True if delivered, False if target not found.
        """
        if target_id not in self.agents:
            return False
        
        message = Message(
            msg_type=MessageType.DIRECT,
            sender_id=sender_id,
            target_id=target_id,
            content=content
        )
        await self.agents[target_id].put(message)
        self._add_to_history(message)
        return True
    
    async def request_mating(self, sender_id: int, target_id: int) -> bool:
        """Send a mating request to another agent."""
        if target_id not in self.agents:
            return False
        
        message = Message(
            msg_type=MessageType.MATING_REQUEST,
            sender_id=sender_id,
            target_id=target_id,
            content={"sender_name": self.agent_names.get(sender_id)}
        )
        await self.agents[target_id].put(message)
        return True
    
    async def respond_mating(self, sender_id: int, target_id: int, accept: bool):
        """Respond to a mating request."""
        if target_id not in self.agents:
            return
        
        msg_type = MessageType.MATING_ACCEPT if accept else MessageType.MATING_REJECT
        message = Message(
            msg_type=msg_type,
            sender_id=sender_id,
            target_id=target_id,
            content={"accepted": accept}
        )
        await self.agents[target_id].put(message)
    
    def _announce_birth(self, agent_id: int, agent_name: str):
        """Announce a new agent to all."""
        message = Message(
            msg_type=MessageType.BIRTH,
            sender_id=None,  # System message
            content={
                "agent_id": agent_id,
                "name": agent_name,
                "time": time.time()
            }
        )
        asyncio.create_task(self._deliver_to_all(message, exclude=[agent_id]))
        self._trigger_callbacks(MessageType.BIRTH, message)
    
    def _announce_death(self, agent_id: int, agent_name: str, cause: str):
        """Announce agent death to all."""
        message = Message(
            msg_type=MessageType.DEATH,
            sender_id=None,
            content={
                "agent_id": agent_id,
                "name": agent_name,
                "cause": cause,
                "time": time.time()
            }
        )
        asyncio.create_task(self._deliver_to_all(message))
        self._trigger_callbacks(MessageType.DEATH, message)
    
    async def _deliver_to_all(self, message: Message, exclude: list = None):
        """Deliver message to all agents (except excluded)."""
        exclude = exclude or []
        for agent_id, queue in self.agents.items():
            if agent_id not in exclude:
                await queue.put(message)
        self._add_to_history(message)
    
    def _add_to_history(self, message: Message):
        """Add message to history."""
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
    
    def on_event(self, msg_type: MessageType, callback: Callable):
        """Register callback for message type."""
        if msg_type not in self._callbacks:
            self._callbacks[msg_type] = []
        self._callbacks[msg_type].append(callback)
    
    def _trigger_callbacks(self, msg_type: MessageType, message: Message):
        """Trigger registered callbacks."""
        for callback in self._callbacks.get(msg_type, []):
            try:
                callback(message)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_active_agents(self) -> Dict[int, str]:
        """Get all active agent IDs and names."""
        return self.agent_names.copy()
    
    def get_agent_count(self) -> int:
        """Get number of active agents."""
        return len(self.agents)


# Singleton instance
_comm_server = None

def get_communication_server() -> CommunicationServer:
    """Get the global communication server instance."""
    global _comm_server
    if _comm_server is None:
        _comm_server = CommunicationServer()
    return _comm_server
