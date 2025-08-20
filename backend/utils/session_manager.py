import asyncio
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from cachetools import TTLCache
from loguru import logger

@dataclass
class MLSession:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    task_id: Optional[str] = None
    status: str = "idle"  # idle, running, completed, failed
    agent_name: Optional[str] = None  # Add this field
    requirements: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class SessionManager:
    """High-performance session management with TTL cache"""
    
    def __init__(self, cache_ttl: int = 3600):
        self.sessions: TTLCache[str, MLSession] = TTLCache(
            maxsize=1000, 
            ttl=cache_ttl
        )
        self.user_sessions: Dict[str, str] = {}
        
    async def create_session(self, user_id: str, agent_name: Optional[str] = None) -> MLSession:
        """Create new session with optional agent_name parameter"""
        session_id = str(uuid.uuid4())
        
        session = MLSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            agent_name=agent_name  # Now accepts agent_name
        )
        
        # Store in cache and user mapping
        self.sessions[session_id] = session
        self.user_sessions[user_id] = session_id
        
        logger.info(f"Created session {session_id} for user {user_id} with agent {agent_name}")
        return session
    
    # Keep all other methods unchanged...
    async def get_session(self, session_id: str) -> Optional[MLSession]:
        """Get session by ID with O(1) complexity"""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
        return session
    
    async def get_user_session(self, user_id: str) -> Optional[MLSession]:
        """Get user's active session with O(1) complexity"""
        session_id = self.user_sessions.get(user_id)
        if session_id:
            return await self.get_session(session_id)
        return None
    
    async def update_session(self, session_id: str, **updates) -> bool:
        """Update session with O(1) complexity"""
        session = self.sessions.get(session_id)
        if not session:
            return False
            
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.last_activity = datetime.now()
        return True
    
    async def cleanup_expired_sessions(self):
        """Cleanup is handled automatically by TTL cache"""
        expired_users = []
        
        for user_id, session_id in self.user_sessions.items():
            if session_id not in self.sessions:
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.user_sessions[user_id]
            
        logger.info(f"Cleaned up {len(expired_users)} expired user sessions")

# Global session manager
session_manager = SessionManager()
