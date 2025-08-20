import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Upload, 
  Bot, 
  User, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  FileText,
  Play,
  Pause,
  X,
  Paperclip,
  Loader2
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import MessageBubble from './MessageBubble';
import ProgressIndicator from './ProgressIndicator';
import AgentProgressPanel from './AgentProgressPanel';
import { ChatMessage, MLTask, AgentProgress, LogEntry } from '@/types';
import { useDebounce } from 'use-debounce';

interface Props {
  messages: ChatMessage[];
  onSendMessage: (message: string, files?: File[]) => void;
  currentTask: MLTask | null;
  agentProgress: AgentProgress[];
  logs: LogEntry[];
  loading: boolean;
  isConnected: boolean;
}

export default function ChatInterface({
  messages,
  onSendMessage,
  currentTask,
  agentProgress,
  logs,
  loading,
  isConnected
}: Props) {
  const [inputValue, setInputValue] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Debounce typing indicator
  const [debouncedTyping] = useDebounce(isTyping, 1000);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, agentProgress]);

  useEffect(() => {
    if (debouncedTyping) {
      setIsTyping(false);
    }
  }, [debouncedTyping]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.apache.parquet': ['.parquet']
    },
    maxSize: 500 * 1024 * 1024, // 500MB
    onDrop: (acceptedFiles, rejectedFiles) => {
      if (rejectedFiles.length > 0) {
        const error = rejectedFiles[0].errors;
        alert(`File rejected: ${error.message}`);
        return;
      }
      setUploadedFiles(prev => [...prev, ...acceptedFiles]);
    },
    noClick: true,
    noKeyboard: true
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() || uploadedFiles.length > 0) {
      onSendMessage(inputValue, uploadedFiles);
      setInputValue('');
      setUploadedFiles([]);
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
    setIsTyping(true);
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleFileInputClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setUploadedFiles(prev => [...prev, ...files]);
    e.target.value = '';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'csv':
        return '📊';
      case 'json':
        return '📋';
      case 'xlsx':
      case 'xls':
        return '📈';
      case 'parquet':
        return '🗂️';
      default:
        return '📄';
    }
  };

  return (
    <div className="flex h-full" {...getRootProps()}>
      <input {...getInputProps()} />
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".csv,.json,.xlsx,.xls,.parquet"
        onChange={handleFileInputChange}
        className="hidden"
      />
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Connection Status */}
        <div className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {isConnected ? 'Connected to ML Pipeline' : 'Disconnected'}
              </span>
              {currentTask && (
                <span className="bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 px-3 py-1 rounded-full text-xs font-medium">
                  Task: {currentTask.task_id.slice(0, 8)}...
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowLogs(!showLogs)}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                  showLogs 
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {showLogs ? 'Hide Logs' : 'Show Logs'}
              </button>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-auto p-6 space-y-6 bg-gray-50 dark:bg-dark-900">
          <AnimatePresence>
            {messages.map((message) => (
              <MessageBubble 
                key={message.id} 
                message={message}
                currentTask={currentTask}
              />
            ))}
          </AnimatePresence>
          
          {/* Agent Progress Indicator */}
          {currentTask && currentTask.status === 'running' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-center"
            >
              <ProgressIndicator 
                agentProgress={agentProgress}
                currentStage={getCurrentStage(agentProgress)}
                currentTask={currentTask}
              />
            </motion.div>
          )}

          {/* Typing Indicator */}
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 px-4 py-3 bg-white dark:bg-dark-800 rounded-2xl max-w-xs shadow-sm"
            >
              <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-primary-600 dark:text-primary-400" />
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              </div>
              <span className="text-sm text-gray-500">ML agents working...</span>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Drag & Drop Overlay */}
        {isDragActive && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-primary-500/10 border-2 border-dashed border-primary-500 rounded-lg flex items-center justify-center z-50"
          >
            <div className="text-center">
              <Upload className="w-16 h-16 text-primary-500 mx-auto mb-4" />
              <p className="text-2xl font-bold text-primary-600 dark:text-primary-400 mb-2">
                Drop your dataset files here
              </p>
              <p className="text-lg text-gray-600 dark:text-gray-300">
                Supports CSV, JSON, Excel, Parquet files
              </p>
            </div>
          </motion.div>
        )}

        {/* Input Area */}
        <div className="bg-white dark:bg-dark-800 border-t border-gray-200 dark:border-dark-700 p-6">
          {/* File Upload Preview */}
          {uploadedFiles.length > 0 && (
            <div className="mb-4">
              <div className="flex flex-wrap gap-3">
                {uploadedFiles.map((file, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 px-4 py-2 rounded-xl border border-primary-200 dark:border-primary-800"
                  >
                    <span className="text-lg mr-2">{getFileIcon(file.name)}</span>
                    <div className="flex flex-col min-w-0">
                      <span className="text-sm font-medium truncate max-w-32">{file.name}</span>
                      <span className="text-xs opacity-75">{formatFileSize(file.size)}</span>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="ml-3 text-primary-500 hover:text-primary-700 dark:hover:text-primary-300 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Message Input */}
          <form onSubmit={handleSubmit} className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder="Describe your ML project or ask a question..."
                className="w-full px-4 py-4 pr-12 border-2 border-gray-200 dark:border-dark-600 rounded-2xl resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-dark-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                rows={3}
                disabled={loading}
                style={{ minHeight: '80px' }}
              />
              
              {/* File Upload Button */}
              <button
                type="button"
                onClick={handleFileInputClick}
                className="absolute right-3 bottom-3 p-2 text-gray-400 hover:text-primary-500 transition-colors"
                title="Upload dataset"
                disabled={loading}
              >
                <Paperclip className="w-5 h-5" />
              </button>
            </div>

            <button
              type="submit"
              disabled={loading || (!inputValue.trim() && uploadedFiles.length === 0)}
              className="px-8 py-4 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-2xl font-semibold hover:from-primary-700 hover:to-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center gap-3 min-w-[120px]"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Send</span>
                </>
              )}
            </button>
          </form>

          <div className="flex items-center justify-between mt-4 text-sm text-gray-500 dark:text-gray-400">
            <span>Press Enter to send, Shift+Enter for new line</span>
            <span>Supports CSV, JSON, Excel, Parquet datasets up to 500MB</span>
          </div>
        </div>
      </div>

      {/* Agent Progress Sidebar */}
      {currentTask && (
        <AgentProgressPanel 
          task={currentTask}
          agentProgress={agentProgress}
          logs={logs}
          showLogs={showLogs}
          onToggleLogs={() => setShowLogs(!showLogs)}
          isConnected={isConnected}
        />
      )}
    </div>
  );
}

function getCurrentStage(agentProgress: AgentProgress[]): string {
  const runningAgent = agentProgress.find(agent => agent.status === 'running');
  return runningAgent?.agent_name || 'Initializing...';
}
