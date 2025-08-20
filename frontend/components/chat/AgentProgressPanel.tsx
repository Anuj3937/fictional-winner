import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronRight, 
  ChevronDown, 
  Terminal, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Activity,
  X,
  Minimize2,
  Maximize2,
  TrendingUp,
  Zap,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';
import { MLTask, AgentProgress, LogEntry } from '@/types';

interface Props {
  task: MLTask;
  agentProgress: AgentProgress[];
  logs: LogEntry[];
  showLogs: boolean;
  onToggleLogs: () => void;
  isConnected: boolean;
}

export default function AgentProgressPanel({ 
  task, 
  agentProgress, 
  logs, 
  showLogs, 
  onToggleLogs,
  isConnected 
}: Props) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [logLevel, setLogLevel] = useState<'all' | 'info' | 'warning' | 'error'>('all');

  if (!task || task.status === 'pending') return null;

  const filteredLogs = logs.filter(log => 
    logLevel === 'all' || log.level === logLevel
  );

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-400';
      case 'warning': return 'text-yellow-400';
      case 'info': return 'text-blue-400';
      case 'debug': return 'text-gray-400';
      default: return 'text-gray-300';
    }
  };

  const getLogLevelIcon = (level: string) => {
    switch (level) {
      case 'error': return <AlertCircle className="w-3 h-3" />;
      case 'warning': return <AlertTriangle className="w-3 h-3" />;
      case 'info': return <CheckCircle className="w-3 h-3" />;
      case 'debug': return <Zap className="w-3 h-3" />;
      default: return <CheckCircle className="w-3 h-3" />;
    }
  };

  return (
    <motion.div
      initial={{ x: 400, opacity: 0 }}
      animate={{ 
        x: 0, 
        opacity: 1,
        width: isMinimized ? '80px' : '480px'
      }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
      className="bg-white dark:bg-dark-800 border-l border-gray-200 dark:border-dark-700 flex flex-col shadow-xl"
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-dark-700 bg-gray-50 dark:bg-dark-750">
        <div className="flex items-center justify-between">
          {!isMinimized && (
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary-500" />
                <h3 className="font-bold text-gray-900 dark:text-gray-100">
                  Agent Pipeline
                </h3>
              </div>
              <span className="bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 px-2 py-1 rounded-full text-xs font-medium">
                {agentProgress.filter(a => a.status === 'completed').length} / {agentProgress.length}
              </span>
            </div>
          )}
          
          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className="p-2 hover:bg-gray-200 dark:hover:bg-dark-600 rounded-lg transition-colors"
              title={isMinimized ? 'Expand Panel' : 'Minimize Panel'}
            >
              {isMinimized ? (
                <Maximize2 className="w-4 h-4" />
              ) : (
                <Minimize2 className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        {!isMinimized && (
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="text-center">
              <div className="text-lg font-bold text-green-600 dark:text-green-400">
                {agentProgress.filter(a => a.status === 'completed').length}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">Completed</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                {agentProgress.filter(a => a.status === 'running').length}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">Running</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-gray-600 dark:text-gray-400">
                {agentProgress.filter(a => a.status === 'pending').length}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">Pending</div>
            </div>
          </div>
        )}
      </div>

      {!isMinimized && (
        <>
          {/* Agent Progress List */}
          <div className="flex-1 overflow-auto p-4 space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Agent Execution Status
            </h4>
            
            {agentProgress.length === 0 ? (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>Waiting for agents to start...</p>
              </div>
            ) : (
              agentProgress.map((agent) => (
                <AgentProgressItem
                  key={agent.agent_name}
                  agent={agent}
                  isExpanded={expandedAgent === agent.agent_name}
                  onToggleExpanded={() => 
                    setExpandedAgent(
                      expandedAgent === agent.agent_name ? null : agent.agent_name
                    )
                  }
                />
              ))
            )}
          </div>

          {/* Logs Section */}
          <div className="border-t border-gray-200 dark:border-dark-700 bg-gray-50 dark:bg-dark-750">
            <button
              onClick={onToggleLogs}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-dark-600 transition-colors"
            >
              <div className="flex items-center gap-3">
                <Terminal className="w-4 h-4" />
                <span className="font-semibold">Live Execution Logs</span>
                <span className="text-xs bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 px-2 py-1 rounded-full">
                  {filteredLogs.length}
                </span>
              </div>
              {showLogs ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>

            <AnimatePresence>
              {showLogs && (
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: 320 }}
                  exit={{ height: 0 }}
                  className="overflow-hidden"
                >
                  {/* Log Level Filter */}
                  <div className="px-4 py-2 border-t border-gray-200 dark:border-dark-600 bg-gray-100 dark:bg-dark-700">
                    <div className="flex gap-2">
                      {['all', 'info', 'warning', 'error'].map((level) => (
                        <button
                          key={level}
                          onClick={() => setLogLevel(level as any)}
                          className={`px-2 py-1 text-xs rounded font-medium transition-colors ${
                            logLevel === level
                              ? 'bg-primary-500 text-white'
                              : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-500'
                          }`}
                        >
                          {level.charAt(0).toUpperCase() + level.slice(1)}
                          {level !== 'all' && (
                            <span className="ml-1">
                              ({logs.filter(l => l.level === level).length})
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Log Content */}
                  <div className="h-64 overflow-auto bg-gray-900 text-green-400 font-mono text-xs">
                    <div className="p-3 space-y-1">
                      {filteredLogs.length === 0 ? (
                        <div className="text-gray-500 text-center py-8">
                          <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                          <p>No {logLevel !== 'all' ? logLevel : ''} logs yet...</p>
                        </div>
                      ) : (
                        filteredLogs.map((log, index) => (
                          <motion.div
                            key={log.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.01 }}
                            className="flex gap-2 items-start hover:bg-gray-800/50 rounded px-1 py-0.5"
                          >
                            <span className="text-gray-500 shrink-0 text-[10px] mt-0.5">
                              {new Date(log.timestamp).toLocaleTimeString()}
                            </span>
                            <span className={`shrink-0 ${getLogLevelColor(log.level)}`}>
                              {getLogLevelIcon(log.level)}
                            </span>
                            <span className="text-blue-400 shrink-0 text-[10px]">
                              [{log.agent}]
                            </span>
                            <span className="text-purple-400 shrink-0 text-[10px]">
                              {log.stage}:
                            </span>
                            <span className="text-gray-300 break-all">
                              {log.message}
                            </span>
                            {log.performance && (
                              <span className="text-yellow-400 text-[10px] ml-auto shrink-0">
                                {log.performance.memory_mb.toFixed(0)}MB
                              </span>
                            )}
                          </motion.div>
                        ))
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </>
      )}
    </motion.div>
  );
}

function AgentProgressItem({ 
  agent, 
  isExpanded, 
  onToggleExpanded 
}: { 
  agent: AgentProgress; 
  isExpanded: boolean; 
  onToggleExpanded: () => void; 
}) {
  const getStatusIcon = () => {
    switch (agent.status) {
      case 'pending': return <Clock className="w-4 h-4 text-gray-400" />;
      case 'running': return (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        >
          <RefreshCw className="w-4 h-4 text-primary-500" />
        </motion.div>
      );
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getStatusColor = () => {
    switch (agent.status) {
      case 'pending': return 'text-gray-500 border-gray-200 dark:border-gray-600';
      case 'running': return 'text-primary-600 dark:text-primary-400 border-primary-200 dark:border-primary-800 bg-primary-50/50 dark:bg-primary-900/10';
      case 'completed': return 'text-green-600 dark:text-green-400 border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-900/10';
      case 'failed': return 'text-red-600 dark:text-red-400 border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-900/10';
    }
  };

  const getProgressColor = () => {
    if (agent.quality_metrics) {
      if (agent.quality_metrics.score >= 0.8) return 'bg-green-500';
      if (agent.quality_metrics.score >= 0.6) return 'bg-yellow-500';
      return 'bg-red-500';
    }
    return 'bg-primary-500';
  };

  return (
    <div className={`border rounded-xl overflow-hidden transition-all ${getStatusColor()}`}>
      <button
        onClick={onToggleExpanded}
        className="w-full p-4 flex items-center gap-3 hover:bg-gray-50 dark:hover:bg-dark-700/50 transition-colors text-left"
      >
        {getStatusIcon()}
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-1">
            {agent.agent_name}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 truncate">
            {agent.message}
          </div>
          
          {/* Quality Score Badge */}
          {agent.quality_metrics && (
            <div className="mt-2 flex items-center gap-2">
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                agent.quality_metrics.score >= 0.8 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                agent.quality_metrics.score >= 0.6 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300' :
                'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
              }`}>
                Quality: {Math.round(agent.quality_metrics.score * 100)}%
              </span>
              {agent.quality_metrics.should_reiterate && (
                <span className="bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300 px-2 py-1 rounded-full text-xs font-medium">
                  Reiteration Required
                </span>
              )}
            </div>
          )}
        </div>
        
        <div className="text-right shrink-0">
          <div className="text-sm font-semibold">
            {agent.progress}%
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 mt-1 mx-auto" />
          ) : (
            <ChevronRight className="w-4 h-4 mt-1 mx-auto" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 bg-gray-50/50 dark:bg-dark-700/50">
              {/* Progress Bar */}
              <div className="mb-3">
                <div className="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${agent.progress}%` }}
                    transition={{ duration: 0.5 }}
                    className={`h-2 rounded-full ${getProgressColor()}`}
                  />
                </div>
              </div>
              
              {/* Stage Details */}
              {agent.stage_details && (
                <div className="mb-3 p-3 bg-white dark:bg-dark-800 rounded-lg">
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
                    Current Step: {agent.stage_details.current_step}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                    Progress: {agent.stage_details.step_progress}% 
                    ({agent.stage_details.total_steps} total steps)
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-1.5">
                    <div 
                      className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${agent.stage_details.step_progress}%` }}
                    />
                  </div>
                </div>
              )}
              
              {/* Quality Issues */}
              {agent.quality_metrics?.issues && agent.quality_metrics.issues.length > 0 && (
                <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                    <span className="text-sm font-medium text-yellow-700 dark:text-yellow-300">
                      Quality Issues Detected
                    </span>
                  </div>
                  <ul className="text-xs text-yellow-600 dark:text-yellow-400 list-disc list-inside space-y-1">
                    {agent.quality_metrics.issues.map((issue, i) => (
                      <li key={i}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Additional Details */}
              <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                <div>Status: <span className="font-medium">{agent.status}</span></div>
                <div>Started: {new Date(agent.timestamp).toLocaleString()}</div>
                {agent.status === 'completed' && (
                  <div className="text-green-600 dark:text-green-400 font-medium">
                    ✓ Completed successfully
                  </div>
                )}
                {agent.status === 'failed' && (
                  <div className="text-red-600 dark:text-red-400 font-medium">
                    ✗ Execution failed
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
