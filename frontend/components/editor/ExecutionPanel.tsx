import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Terminal, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Copy,
  Download,
  Trash2,
  Maximize2,
  Minimize2,
  Play,
  Square
} from 'lucide-react';
import { CodeExecutionResult } from '@/types';

interface Props {
  isExecuting: boolean;
  result: CodeExecutionResult | null;
  onClose: () => void;
  onClear: () => void;
  onRerun?: () => void;
}

export default function ExecutionPanel({ 
  isExecuting, 
  result, 
  onClose, 
  onClear, 
  onRerun 
}: Props) {
  const [isMaximized, setIsMaximized] = useState(false);
  const [copied, setCopied] = useState(false);
  const outputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new output appears
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [result?.output, result?.error]);

  const copyOutput = async () => {
    if (!result) return;
    
    const output = result.output || result.error || '';
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy output: ', err);
    }
  };

  const downloadOutput = () => {
    if (!result) return;
    
    const output = result.output || result.error || '';
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `execution_output_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getStatusIcon = () => {
    if (isExecuting) {
      return <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />;
    }
    if (!result) {
      return <Terminal className="w-4 h-4 text-gray-400" />;
    }
    if (result.success) {
      return <CheckCircle className="w-4 h-4 text-green-400" />;
    }
    return <AlertCircle className="w-4 h-4 text-red-400" />;
  };

  const getStatusText = () => {
    if (isExecuting) return 'Executing...';
    if (!result) return 'Ready';
    if (result.success) return 'Success';
    return 'Error';
  };

  const getStatusColor = () => {
    if (isExecuting) return 'text-blue-400';
    if (!result) return 'text-gray-400';
    if (result.success) return 'text-green-400';
    return 'text-red-400';
  };

  return (
    <motion.div
      initial={{ height: 0 }}
      animate={{ height: isMaximized ? '60vh' : '40vh' }}
      exit={{ height: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
      className="bg-gray-900 text-green-400 border-t border-gray-600 overflow-hidden flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3">
          {getStatusIcon()}
          <span className="font-semibold text-gray-200">Terminal</span>
          <span className={`text-sm ${getStatusColor()}`}>
            {getStatusText()}
          </span>
          
          {result?.execution_time && (
            <span className="text-xs text-gray-500">
              ({result.execution_time.toFixed(2)}s)
            </span>
          )}
          
          {result?.memory_usage && (
            <span className="text-xs text-gray-500">
              • {(result.memory_usage / 1024 / 1024).toFixed(1)}MB
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {onRerun && (
            <button
              onClick={onRerun}
              disabled={isExecuting}
              className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-50"
              title="Rerun"
            >
              <Play className="w-4 h-4" />
            </button>
          )}
          
          {result && (
            <>
              <button
                onClick={copyOutput}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
                title="Copy Output"
              >
                {copied ? <CheckCircle className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
              </button>
              
              <button
                onClick={downloadOutput}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
                title="Download Output"
              >
                <Download className="w-4 h-4" />
              </button>
              
              <button
                onClick={onClear}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-red-400 transition-colors"
                title="Clear Output"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </>
          )}
          
          <div className="w-px h-4 bg-gray-600" />
          
          <button
            onClick={() => setIsMaximized(!isMaximized)}
            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
            title={isMaximized ? 'Minimize' : 'Maximize'}
          >
            {isMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
            title="Close Terminal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div ref={outputRef} className="flex-1 overflow-auto p-4 font-mono text-sm">
        {isExecuting && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-3 text-blue-400 mb-4"
          >
            <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            <span>Executing Python code...</span>
          </motion.div>
        )}

        {result && (
          <div className="space-y-4">
            {/* Execution Info */}
            <div className="flex items-center gap-4 text-gray-400 text-xs">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {new Date().toLocaleTimeString()}
              </span>
              {result.execution_time && (
                <span>Duration: {result.execution_time.toFixed(2)}s</span>
              )}
              {result.exit_code !== undefined && (
                <span>Exit Code: {result.exit_code}</span>
              )}
            </div>

            {/* Success Output */}
            {result.success && result.output && (
              <div>
                <div className="text-green-400 text-xs mb-2 flex items-center gap-2">
                  <CheckCircle className="w-3 h-3" />
                  OUTPUT:
                </div>
                <div className="bg-gray-800 rounded-lg p-3 border-l-4 border-green-500">
                  <pre className="text-green-300 whitespace-pre-wrap overflow-x-auto">
                    {result.output}
                  </pre>
                </div>
              </div>
            )}

            {/* Error Output */}
            {!result.success && result.error && (
              <div>
                <div className="text-red-400 text-xs mb-2 flex items-center gap-2">
                  <AlertCircle className="w-3 h-3" />
                  ERROR:
                </div>
                <div className="bg-gray-800 rounded-lg p-3 border-l-4 border-red-500">
                  <pre className="text-red-300 whitespace-pre-wrap overflow-x-auto">
                    {result.error}
                  </pre>
                </div>
              </div>
            )}

            {/* Logs */}
            {result.logs && result.logs.length > 0 && (
              <div>
                <div className="text-gray-400 text-xs mb-2">LOGS:</div>
                <div className="bg-gray-800 rounded-lg p-3 space-y-1">
                  {result.logs.map((log, index) => (
                    <div key={index} className="text-gray-300 text-xs">
                      <span className="text-gray-500">[{index + 1}]</span> {log}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Success with no output */}
            {result.success && !result.output && (
              <div className="text-green-400 italic text-center py-4">
                Code executed successfully with no output
              </div>
            )}
          </div>
        )}

        {/* Ready State */}
        {!isExecuting && !result && (
          <div className="text-gray-500 italic text-center py-8">
            <Terminal className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>Ready to execute Python code...</p>
            <p className="text-sm mt-2">Click the "Run Code" button or press Ctrl+Enter</p>
          </div>
        )}
      </div>

      {/* Quick Actions Bar */}
      {result && (
        <div className="bg-gray-800 border-t border-gray-700 px-4 py-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-4 text-gray-400">
              <span>
                Status: <span className={getStatusColor()}>{getStatusText()}</span>
              </span>
              {result.output && (
                <span>Output: {result.output.split('\n').length} lines</span>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              <kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Ctrl+C</kbd>
              <span className="text-gray-500">Copy</span>
              <span className="text-gray-600">•</span>
              <kbd className="px-1 py-0.5 bg-gray-700 rounded text-xs">Ctrl+S</kbd>
              <span className="text-gray-500">Download</span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}
