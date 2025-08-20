import { motion } from 'framer-motion';
import { Bot, User, Clock, CheckCircle, AlertCircle, Code, Download, Copy, ExternalLink } from 'lucide-react';
import { ChatMessage, MLTask } from '@/types';
import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface Props {
  message: ChatMessage;
  currentTask?: MLTask | null;
  darkMode?: boolean;
}

export default function MessageBubble({ message, currentTask, darkMode = true }: Props) {
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  
  const isUser = message.type === 'user';
  const isSystem = message.type === 'system';
  const isAgent = message.type === 'agent';

  const getMessageIcon = () => {
    if (isUser) return <User className="w-4 h-4" />;
    if (isSystem) return <Bot className="w-4 h-4 text-blue-500" />;
    if (isAgent) return <Bot className="w-4 h-4 text-purple-500" />;
    return <Bot className="w-4 h-4" />;
  };

  const getMessageStyle = () => {
    if (isUser) {
      return 'bg-gradient-to-r from-primary-500 to-primary-600 text-white ml-12 shadow-lg';
    }
    if (isSystem) {
      return 'bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 mr-12';
    }
    if (isAgent) {
      return 'bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800 mr-12';
    }
    return 'bg-white dark:bg-dark-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-dark-700 shadow-sm mr-12';
  };

  const getAvatarStyle = () => {
    if (isUser) {
      return 'bg-gradient-to-r from-primary-500 to-primary-600 text-white';
    }
    if (isSystem) {
      return 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white';
    }
    if (isAgent) {
      return 'bg-gradient-to-r from-purple-500 to-pink-500 text-white';
    }
    return 'bg-gradient-to-r from-gray-500 to-gray-600 text-white';
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const detectCodeBlocks = (content: string) => {
    const codeBlockRegex = /``````/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: content.slice(lastIndex, match.index)
        });
      }

      // Add code block
      parts.push({
        type: 'code',
        language: match[1] || 'text',
        content: match[1].trim()
      });

      lastIndex = match.index + match.length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
      parts.push({
        type: 'text',
        content: content.slice(lastIndex)
      });
    }

    return parts.length > 0 ? parts : [{ type: 'text', content }];
  };

  const renderContent = () => {
    const parts = detectCodeBlocks(message.content);

    return parts.map((part, index) => {
      if (part.type === 'code') {
        return (
          <div key={index} className="my-4">
            <div className="flex items-center justify-between bg-gray-800 dark:bg-gray-900 px-4 py-2 rounded-t-lg">
              <span className="text-sm text-gray-300 font-mono">{part.language}</span>
              <button
                onClick={() => copyToClipboard(part.content)}
                className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
              >
                {copied ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                <span className="text-sm">{copied ? 'Copied!' : 'Copy'}</span>
              </button>
            </div>
            <SyntaxHighlighter
              language={part.language}
              style={darkMode ? vscDarkPlus : oneLight}
              customStyle={{
                margin: 0,
                borderTopLeftRadius: 0,
                borderTopRightRadius: 0,
                borderBottomLeftRadius: '0.5rem',
                borderBottomRightRadius: '0.5rem'
              }}
            >
              {part.content}
            </SyntaxHighlighter>
          </div>
        );
      } else {
        return (
          <div key={index} className="whitespace-pre-wrap leading-relaxed">
            {part.content}
          </div>
        );
      }
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`max-w-4xl rounded-2xl px-6 py-4 ${getMessageStyle()}`}>
        {/* Header */}
        <div className="flex items-center gap-3 mb-3">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getAvatarStyle()}`}>
            {getMessageIcon()}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">
              {isUser ? 'You' : isSystem ? 'System' : isAgent ? message.agent_name || 'Agent' : 'ML Assistant'}
            </span>
            {message.metadata?.stage && (
              <span className="bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded-md text-xs font-medium">
                {message.metadata.stage}
              </span>
            )}
          </div>
          <span className="text-xs opacity-60 ml-auto">
            {formatTimestamp(message.timestamp)}
          </span>
        </div>

        {/* Progress Info */}
        {message.metadata?.progress !== undefined && (
          <div className="mb-3 p-3 bg-white/10 dark:bg-black/10 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Progress</span>
              <span className="text-sm">{Math.round(message.metadata.progress)}%</span>
            </div>
            <div className="w-full bg-white/20 dark:bg-black/20 rounded-full h-2">
              <div 
                className="bg-current h-2 rounded-full transition-all duration-300"
                style={{ width: `${message.metadata.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Quality Score */}
        {message.metadata?.quality_score !== undefined && (
          <div className="mb-3 p-3 bg-white/10 dark:bg-black/10 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Quality Score</span>
              <span className={`text-sm font-bold ${
                message.metadata.quality_score >= 0.8 ? 'text-green-500' :
                message.metadata.quality_score >= 0.6 ? 'text-yellow-500' : 'text-red-500'
              }`}>
                {(message.metadata.quality_score * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        )}

        {/* Performance Metrics */}
        {message.metadata?.performance && (
          <div className="mb-3 p-3 bg-white/10 dark:bg-black/10 rounded-lg">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="opacity-75">CPU:</span>
                <span className="ml-2 font-medium">{message.metadata.performance.cpu_percent.toFixed(1)}%</span>
              </div>
              <div>
                <span className="opacity-75">Memory:</span>
                <span className="ml-2 font-medium">{message.metadata.performance.memory_mb.toFixed(0)}MB</span>
              </div>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="prose prose-sm max-w-none">
          {renderContent()}
        </div>

        {/* File Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="mt-4 space-y-2">
            {message.attachments.map((file) => (
              <div key={file.id} className="flex items-center gap-3 p-3 bg-white/10 dark:bg-black/10 rounded-lg">
                <div className="w-8 h-8 bg-white/20 dark:bg-black/20 rounded-lg flex items-center justify-center">
                  <ExternalLink className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{file.name}</p>
                  <p className="text-xs opacity-75">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                {file.url && (
                  <a
                    href={file.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium hover:underline"
                  >
                    View
                  </a>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Task Status */}
        {message.task_id && currentTask && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ delay: 0.2 }}
            className="mt-4 pt-4 border-t border-current/20"
          >
            <TaskStatusDisplay task={currentTask} />
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

function TaskStatusDisplay({ task }: { task: MLTask }) {
  const getStatusIcon = () => {
    switch (task.status) {
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'running': return <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'cancelled': return <AlertCircle className="w-4 h-4 text-gray-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (task.status) {
      case 'pending': return 'text-yellow-600 dark:text-yellow-400';
      case 'running': return 'text-blue-600 dark:text-blue-400';
      case 'completed': return 'text-green-600 dark:text-green-400';
      case 'failed': return 'text-red-600 dark:text-red-400';
      case 'cancelled': return 'text-gray-600 dark:text-gray-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        {getStatusIcon()}
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-semibold ${getStatusColor()}`}>
              Task {task.status.charAt(0).toUpperCase() + task.status.slice(1)}
            </span>
            <span className="text-xs opacity-75">
              ID: {task.task_id.slice(0, 8)}...
            </span>
          </div>
          {task.current_stage && (
            <p className="text-xs opacity-75 mt-1">{task.current_stage}</p>
          )}
        </div>
        {task.progress > 0 && (
          <span className="text-sm font-medium">
            {Math.round(task.progress)}%
          </span>
        )}
      </div>

      {/* Progress Bar */}
      {task.status === 'running' && task.progress > 0 && (
        <div className="w-full bg-white/20 dark:bg-black/20 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${task.progress}%` }}
            transition={{ duration: 0.5 }}
            className="bg-current h-2 rounded-full"
          />
        </div>
      )}

      {/* Action Buttons */}
      {task.status === 'completed' && task.results && (
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-3 py-1.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-lg text-sm hover:bg-primary-200 dark:hover:bg-primary-900/50 transition-colors">
            <Code className="w-3 h-3" />
            View Code
          </button>
          <button className="flex items-center gap-2 px-3 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-lg text-sm hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors">
            <Download className="w-3 h-3" />
            Download Model
          </button>
        </div>
      )}
    </div>
  );
}
