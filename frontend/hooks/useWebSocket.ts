import { useState, useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { WebSocketMessage, AgentProgress, LogEntry } from '@/types';
import { apiClient } from '@/utils/api';

interface UseWebSocketReturn {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  agentProgress: AgentProgress[];
  logs: LogEntry[];
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  clearLogs: () => void;
  clearProgress: () => void;
}

export function useWebSocket(sessionId: string | null): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [agentProgress, setAgentProgress] = useState<AgentProgress[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const maxReconnectAttempts = 5;
  const reconnectAttemptRef = useRef(0);

  const connect = useCallback(() => {
    if (!sessionId || socketRef.current?.connected) return;

    setConnectionStatus('connecting');
    
    try {
      // Connect to the Socket.IO server (not WebSocket URL)
      console.log('🔌 Connecting to Socket.IO server');
      
      socketRef.current = io('http://localhost:8000', {
        transports: ['websocket', 'polling'],
        timeout: 10000,
        reconnection: true,
        reconnectionAttempts: maxReconnectAttempts,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        forceNew: true // Force new connection
      });

      socketRef.current.on('connect', () => {
        console.log('✅ Socket.IO connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttemptRef.current = 0;
        
        // Join session room after connecting
        if (sessionId) {
          socketRef.current?.emit('join_session', { session_id: sessionId });
        }
      });

      socketRef.current.on('disconnect', (reason) => {
        console.log('❌ Socket.IO disconnected:', reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
      });

      socketRef.current.on('connect_error', (error) => {
        console.error('❌ Socket.IO connection error:', error);
        setConnectionStatus('error');
        
        // Implement exponential backoff
        if (reconnectAttemptRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);
          reconnectAttemptRef.current++;
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (socketRef.current && !socketRef.current.connected) {
              socketRef.current.connect();
            }
          }, delay);
        }
      });

      // Handle session joined confirmation
      socketRef.current.on('session_joined', (data) => {
        console.log('🎉 Joined session:', data.session_id);
      });

      // Handle session updates (from backend)
      socketRef.current.on('session_update', (message: WebSocketMessage) => {
        console.log('📨 Session update:', message);
        setLastMessage(message);
        handleWebSocketMessage(message);
      });

      // Handle direct events from backend
      socketRef.current.on('connected', (data) => {
        console.log('🎉 Socket.IO session established:', data);
      });

      socketRef.current.on('pipeline_response', (data) => {
        console.log('📨 Pipeline response:', data);
        const message: WebSocketMessage = {
          type: 'pipeline_update',
          timestamp: data.timestamp || new Date().toISOString(),
          ...data
        };
        setLastMessage(message);
        handleWebSocketMessage(message);
      });

      // Generic message handler
      socketRef.current.onAny((eventName, ...args) => {
        console.log('📨 Socket.IO event:', eventName, args);
        
        // Convert Socket.IO events to WebSocket message format
        if (args[0] && typeof args[0] === 'object') {
          const message: WebSocketMessage = {
            type: eventName as any,
            timestamp: new Date().toISOString(),
            ...args
          };
          setLastMessage(message);
          handleWebSocketMessage(message);
        }
      });

    } catch (error) {
      console.error('Failed to create Socket.IO connection:', error);
      setConnectionStatus('error');
    }
  }, [sessionId]);

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'connected':
        console.log('🎉 WebSocket session established');
        break;

      case 'progress_update':
      case 'pipeline_update':
        if (message.agent) {
          setAgentProgress(prev => {
            const existingIndex = prev.findIndex(a => a.agent_name === message.agent!.agent_name);
            if (existingIndex >= 0) {
              const updated = [...prev];
              updated[existingIndex] = message.agent!;
              return updated;
            } else {
              return [...prev, message.agent!];
            }
          });
        }
        
        // Also handle stage updates without agent object
        if (message.stage && message.progress !== undefined) {
          setAgentProgress(prev => {
            const updated = [...prev];
            const agent: AgentProgress = {
              agent_name: message.stage!,
              status: 'running',
              progress: message.progress!,
              current_step: message.stage!,
              start_time: new Date().toISOString(),
              messages: []
            };
            
            const existingIndex = updated.findIndex(a => a.agent_name === message.stage!);
            if (existingIndex >= 0) {
              updated[existingIndex] = { ...updated[existingIndex], ...agent };
            } else {
              updated.push(agent);
            }
            return updated;
          });
        }
        break;

      case 'log_entry':
        if (message.log) {
          setLogs(prev => {
            const newLogs = [...prev, message.log!];
            return newLogs.slice(-100);
          });
        }
        break;

      case 'pipeline_started':
        console.log('🚀 Pipeline started:', message.task_id);
        setAgentProgress([]);
        setLogs([]);
        break;

      case 'pipeline_completed':
        console.log('✅ Pipeline completed:', message.task_id);
        // Mark all agents as completed
        setAgentProgress(prev => prev.map(agent => ({
          ...agent,
          status: 'completed',
          progress: 100
        })));
        break;

      case 'pipeline_failed':
        console.error('❌ Pipeline failed:', message.error);
        // Mark all agents as failed
        setAgentProgress(prev => prev.map(agent => ({
          ...agent,
          status: 'failed'
        })));
        break;

      case 'heartbeat':
        // Silent heartbeat handling
        break;

      default:
        console.log('📨 Unknown message type:', message.type, message);
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('pipeline_update', message);
    } else {
      console.warn('Cannot send message: Socket.IO not connected');
    }
  }, [isConnected]);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  const clearProgress = useCallback(() => {
    setAgentProgress([]);
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  // Effect to manage connection
  useEffect(() => {
    if (sessionId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [sessionId, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    connectionStatus,
    agentProgress,
    logs,
    lastMessage,
    sendMessage,
    clearLogs,
    clearProgress,
  };
}
