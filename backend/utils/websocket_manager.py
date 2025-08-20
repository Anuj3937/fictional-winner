import asyncio
import json
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from dataclasses import asdict
import time

class ConnectionManager:
    """High-performance WebSocket manager with minimal overhead"""
    
    def __init__(self):
        # Use sets for O(1) add/remove operations
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        
    async def connect(self, websocket: WebSocket, session_id: str) -> str:
        """Connect WebSocket with O(1) complexity"""
        await websocket.accept()
        
        connection_id = f"conn_{int(time.time() * 1000)}_{id(websocket)}"
        
        # Store mappings
        self.active_connections[connection_id] = websocket
        self.session_connections[session_id] = connection_id
        self.connection_sessions[connection_id] = session_id
        
        logger.info(f"WebSocket connected: {connection_id} for session {session_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect WebSocket with O(1) complexity"""
        if connection_id in self.active_connections:
            session_id = self.connection_sessions.get(connection_id)
            
            # Remove all mappings
            del self.active_connections[connection_id]
            del self.connection_sessions[connection_id]
            
            if session_id and session_id in self.session_connections:
                del self.session_connections[session_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific session with O(1) lookup"""
        connection_id = self.session_connections.get(session_id)
        if not connection_id:
            return False
            
        websocket = self.active_connections.get(connection_id)
        if not websocket:
            return False
            
        try:
            await websocket.send_text(json.dumps(message, default=str))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections"""
        if not self.active_connections:
            return
            
        # Use asyncio.gather for concurrent sending
        tasks = []
        for connection_id, websocket in self.active_connections.items():
            tasks.append(self._safe_send(websocket, message, connection_id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_send(self, websocket: WebSocket, message: Dict[str, Any], connection_id: str):
        """Safely send message with error handling"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send to {connection_id}: {e}")
            self.disconnect(connection_id)

# Global connection manager
connection_manager = ConnectionManager()
