import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Head from 'next/head';
import WelcomeScreen from '@/components/welcome/WelcomeScreen';
import ChatInterface from '@/components/chat/ChatInterface';
import CodeEditor from '@/components/editor/CodeEditor';
import ResultsDashboard from '@/components/dashboard/ResultsDashboard';
import Header from '@/components/layout/Header';
import Sidebar from '@/components/layout/Sidebar';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useMLPipeline } from '@/hooks/useMLPipeline';
import { ChatMessage, UIState, FileNode } from '@/types';
import { v4 as uuidv4 } from 'uuid';

export default function Home() {
  const [uiState, setUIState] = useState<UIState>({
    activeView: 'welcome',
    sidebarOpen: false,
    darkMode: true,
    currentTask: null,
    isProcessing: false,
    showLogs: false,
    selectedFile: null,
    codeEditorTheme: 'vs-dark'
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [fileTree, setFileTree] = useState<FileNode[]>([]);

  const { 
    currentTask, 
    isLoading, 
    error, 
    createTask, 
    downloadModel, 
    clearError 
  } = useMLPipeline();

  const {
    isConnected,
    agentProgress,
    logs,
    lastMessage,
    clearLogs,
    clearProgress
  } = useWebSocket(currentTask?.session_id || null);

  // Update UI state when task changes
  useEffect(() => {
    setUIState(prev => ({
      ...prev,
      currentTask,
      isProcessing: isLoading || (currentTask?.status === 'running'),
    }));
  }, [currentTask, isLoading]);

  // Handle WebSocket messages with enhanced error handling
  useEffect(() => {
    if (lastMessage) {
      console.log('Processing WebSocket message:', lastMessage);
      
      switch (lastMessage.type) {
        case 'pipeline_completed':
          // Generate file tree from results
          if (lastMessage.results?.artifacts) {
            generateFileTree(lastMessage.results.artifacts);
          } else if (lastMessage.results) {
            // Fallback file generation
            generateFallbackFileTree(lastMessage.results);
          }
          
          addSystemMessage('🎉 ML pipeline completed successfully! Your model and code are ready.');
          break;
          
        case 'pipeline_failed':
          addSystemMessage(`❌ Pipeline failed: ${lastMessage.error || 'Unknown error'}`);
          break;
          
        case 'progress_update':
        case 'pipeline_update':
          // Progress updates are handled by the WebSocket hook
          if (lastMessage.stage) {
            addSystemMessage(`🔄 ${lastMessage.stage}: ${Math.round(lastMessage.progress || 0)}% complete`);
          }
          break;

        case 'pipeline_started':
          addSystemMessage('🚀 ML pipeline started! Watch the progress below.');
          break;
      }
    }
  }, [lastMessage]);

  // Auto-switch views based on task status
  useEffect(() => {
    if (currentTask) {
      if (currentTask.status === 'running') {
        setUIState(prev => ({ ...prev, activeView: 'chat' }));
      } else if (currentTask.status === 'completed' && currentTask.results) {
        // Auto-switch to results view when completed
        if (currentTask.results.code_generation_result || fileTree.length > 0) {
          setTimeout(() => {
            setUIState(prev => ({ ...prev, activeView: 'results' }));
          }, 1000); // Small delay for better UX
        }
      }
    }
  }, [currentTask?.status, currentTask?.results, fileTree.length]);

  const addSystemMessage = (content: string) => {
    const message: ChatMessage = {
      id: uuidv4(),
      type: 'system',
      content,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, message]);
  };

  const addUserMessage = (content: string, files?: File[]) => {
    const message: ChatMessage = {
      id: uuidv4(),
      type: 'user',
      content,
      timestamp: new Date().toISOString(),
      attachments: files?.map(file => ({
        id: uuidv4(),
        name: file.name,
        size: file.size,
        type: file.type
      }))
    };
    setMessages(prev => [...prev, message]);
  };

  const addAssistantMessage = (content: string, taskId?: string) => {
    const message: ChatMessage = {
      id: uuidv4(),
      type: 'assistant',
      content,
      timestamp: new Date().toISOString(),
      task_id: taskId
    };
    setMessages(prev => [...prev, message]);
  };

  const generateFileTree = (artifacts: any): void => {
    if (!artifacts?.generated_files) {
      generateFallbackFileTree(artifacts);
      return;
    }
    
    const files: FileNode[] = [];
    const generatedFiles = artifacts.generated_files;
    
    // Create main project structure
    files.push({
      name: 'README.md',
      type: 'file',
      path: '/README.md',
      content: generateReadmeContent(),
      language: 'markdown'
    });

    if (generatedFiles.inference_code) {
      files.push({
        name: 'model_inference.py',
        type: 'file',
        path: '/model_inference.py',
        content: generatedFiles.inference_code,
        language: 'python'
      });
    }

    if (generatedFiles.streamlit_app) {
      files.push({
        name: 'streamlit_app.py',
        type: 'file',
        path: '/streamlit_app.py',
        content: generatedFiles.streamlit_app,
        language: 'python'
      });
    }

    if (generatedFiles.requirements) {
      files.push({
        name: 'requirements.txt',
        type: 'file',
        path: '/requirements.txt',
        content: generatedFiles.requirements,
        language: 'plaintext'
      });
    }

    // Add Docker files if available
    if (generatedFiles.docker_file) {
      files.push({
        name: 'Dockerfile',
        type: 'file',
        path: '/Dockerfile',
        content: generatedFiles.docker_file,
        language: 'dockerfile'
      });
    }

    // Create models folder
    files.push({
      name: 'models',
      type: 'folder',
      path: '/models',
      children: [
        {
          name: 'trained_model.pkl',
          type: 'file',
          path: '/models/trained_model.pkl',
          content: '# Binary model file - download to view\n# Model Type: ' + (artifacts.model_type || 'Unknown'),
          language: 'plaintext'
        }
      ]
    });

    setFileTree(files);
  };

  const generateFallbackFileTree = (results: any): void => {
    const files: FileNode[] = [];
    
    // Always create a README
    files.push({
      name: 'README.md',
      type: 'file',
      path: '/README.md',
      content: generateReadmeContent(),
      language: 'markdown'
    });

    // Create a sample inference script
    files.push({
      name: 'model_inference.py',
      type: 'file',
      path: '/model_inference.py',
      content: generateSampleInferenceCode(results),
      language: 'python'
    });

    // Create requirements.txt
    files.push({
      name: 'requirements.txt',
      type: 'file',
      path: '/requirements.txt',
      content: 'scikit-learn>=1.0.0\npandas>=1.3.0\nnumpy>=1.21.0\njoblib>=1.0.0\nstreamlit>=1.0.0',
      language: 'plaintext'
    });

    // Create models folder
    files.push({
      name: 'models',
      type: 'folder',
      path: '/models',
      children: [
        {
          name: 'model_info.json',
          type: 'file',
          path: '/models/model_info.json',
          content: JSON.stringify(results || { message: "Model training completed" }, null, 2),
          language: 'json'
        }
      ]
    });

    setFileTree(files);
  };

  const generateSampleInferenceCode = (results: any): string => {
    return `"""
ML Model Inference Script
Generated by ML Automation Playground

Model Performance: ${results?.model_performance?.accuracy ? 
  (results.model_performance.accuracy * 100).toFixed(2) + '%' : 'N/A'}
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
import json
from typing import Union, List, Dict, Any

class MLModelInference:
    def __init__(self, model_path: str = "models/trained_model.pkl"):
        """Initialize the ML model inference."""
        self.model = None
        self.model_info = ${JSON.stringify(results || {}, null, 8)}
        
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load model - {e}")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        # Convert input to appropriate format
        if isinstance(data, list):
            data = np.array(data).reshape(1, -1) if len(np.array(data).shape) == 1 else np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        predictions = self.model.predict(data)
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """Get prediction probabilities (for classification models)."""
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
        # Convert input to appropriate format
        if isinstance(data, list):
            data = np.array(data).reshape(1, -1) if len(np.array(data).shape) == 1 else np.array(data)
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        probabilities = self.model.predict_proba(data)
        return probabilities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return self.model_info

# Example usage
if __name__ == "__main__":
    # Initialize the model
    inference = MLModelInference()
    
    # Example prediction (replace with your actual data format)
    sample_data = [[1.0, 2.0, 3.0]]  # Replace with actual feature values
    
    try:
        predictions = inference.predict(sample_data)
        print(f"Predictions: {predictions}")
        
        # Try to get probabilities if it's a classification model
        try:
            probabilities = inference.predict_proba(sample_data)
            print(f"Probabilities: {probabilities}")
        except:
            print("Model doesn't support probability predictions (likely a regression model)")
            
    except Exception as e:
        print(f"Error making predictions: {e}")
`;
  };

  const generateReadmeContent = (): string => {
    const taskInfo = currentTask ? `
## Task Information
- **Task ID**: ${currentTask.task_id}
- **Status**: ${currentTask.status}
- **Created**: ${new Date(currentTask.created_at || '').toLocaleString()}
- **Progress**: ${Math.round(currentTask.progress || 0)}%
` : '';

    return `# ML Project - Generated by AI

## Overview
This project was automatically generated using our ML Automation Playground.
${taskInfo}
## Files
- \`model_inference.py\` - Production model inference code
- \`streamlit_app.py\` - Interactive web application
- \`requirements.txt\` - Python dependencies
- \`models/\` - Trained model files

## Quick Start

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the Streamlit app (if available):
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

3. Use the inference code:
\`\`\`python
from model_inference import MLModelInference

model = MLModelInference("models/trained_model.pkl")
result = model.predict(your_data)
\`\`\`

## Performance Metrics
${currentTask?.results?.model_performance ? `
- **Accuracy**: ${(currentTask.results.model_performance.accuracy * 100).toFixed(2)}%
- **Precision**: ${(currentTask.results.model_performance.precision * 100).toFixed(2)}%
- **Recall**: ${(currentTask.results.model_performance.recall * 100).toFixed(2)}%
` : 'Metrics will be available after model training completes.'}

## Generated on
${new Date().toLocaleString()}

---
*Generated by ML Automation Playground*
`;
  };

  const handleSendMessage = async (message: string, files?: File[]) => {
    addUserMessage(message, files);
    
    try {
      const task = await createTask({
        prompt: message, // Changed from user_prompt to match backend
        mode: files && files.length > 0 ? 'dataset_provided' : 'prompt_only',
        dataset_path: files && files.length > 0 ? files[0].name : undefined // Simplified for now
      });
      
      if (task) {
        addAssistantMessage(
          `🚀 Starting ML pipeline for your request. I'll analyze your requirements and build a complete ML solution.

**Task ID**: ${task.task_id}
**Session ID**: ${task.session_id}

You can watch the progress in real-time as our intelligent agents work together to create your model.`,
          task.task_id
        );
        
        // Switch to chat view
        setUIState(prev => ({ ...prev, activeView: 'chat' }));
        
        // Clear previous progress
        clearProgress();
        clearLogs();
      }
    } catch (error) {
      console.error('Error creating task:', error);
      addAssistantMessage(
        `❌ Sorry, I encountered an error starting the ML pipeline: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  };

  const handleFileSelect = (path: string) => {
    setUIState(prev => ({ ...prev, selectedFile: path }));
  };

  const handleCodeChange = (path: string, content: string) => {
    setFileTree(prev => updateFileContent(prev, path, content));
  };

  const updateFileContent = (nodes: FileNode[], path: string, content: string): FileNode[] => {
    return nodes.map(node => {
      if (node.path === path && node.type === 'file') {
        return { ...node, content };
      }
      if (node.children) {
        return { ...node, children: updateFileContent(node.children, path, content) };
      }
      return node;
    });
  };

  const handleViewChange = (view: UIState['activeView']) => {
    setUIState(prev => ({ ...prev, activeView: view }));
    
    if (error) {
      clearError();
    }
  };

  const toggleSidebar = () => {
    setUIState(prev => ({ ...prev, sidebarOpen: !prev.sidebarOpen }));
  };

  const toggleDarkMode = () => {
    setUIState(prev => ({ 
      ...prev, 
      darkMode: !prev.darkMode,
      codeEditorTheme: prev.darkMode ? 'light' : 'vs-dark'
    }));
  };

  return (
    <>
      <Head>
        <title>ML Automation Playground - AI-Powered Machine Learning</title>
        <meta name="description" content="Create complete ML solutions using natural language with our intelligent agent system" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className={`min-h-screen ${uiState.darkMode ? 'dark' : ''}`}>
        <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
          {/* Sidebar */}
          {uiState.sidebarOpen && (
            <Sidebar
              activeView={uiState.activeView}
              setActiveView={handleViewChange}
              currentTask={uiState.currentTask}
              onClose={() => setUIState(prev => ({ ...prev, sidebarOpen: false }))}
            />
          )}

          {/* Main Content */}
          <div className="flex-1 flex flex-col">
            {/* Header */}
            <Header
              onToggleSidebar={toggleSidebar}
              onToggleDarkMode={toggleDarkMode}
              darkMode={uiState.darkMode}
              isConnected={isConnected}
              currentTask={uiState.currentTask}
            />

            {/* Content Area */}
            <div className="flex-1 relative">
              {uiState.activeView === 'welcome' && (
                <motion.div
                  key="welcome"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full"
                >
                  <WelcomeScreen
                    onGetStarted={() => handleViewChange('chat')}
                    onSendMessage={handleSendMessage}
                  />
                </motion.div>
              )}

              {uiState.activeView === 'chat' && (
                <motion.div
                  key="chat"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full"
                >
                  <ChatInterface
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    currentTask={uiState.currentTask}
                    agentProgress={agentProgress}
                    logs={logs}
                    loading={uiState.isProcessing}
                    isConnected={isConnected}
                  />
                </motion.div>
              )}

              {uiState.activeView === 'code' && (
                <motion.div
                  key="code"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full"
                >
                  <CodeEditor
                    fileTree={fileTree}
                    selectedFile={uiState.selectedFile}
                    onFileSelect={handleFileSelect}
                    onCodeChange={handleCodeChange}
                    darkMode={uiState.darkMode}
                  />
                </motion.div>
              )}

              {uiState.activeView === 'results' && uiState.currentTask && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full"
                >
                  <ResultsDashboard
                    task={uiState.currentTask}
                    onDownloadModel={() => downloadModel(uiState.currentTask!.task_id)}
                    onViewCode={() => handleViewChange('code')}
                  />
                </motion.div>
              )}
            </div>
          </div>
        </div>

        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: uiState.darkMode ? '#1f2937' : '#ffffff',
              color: uiState.darkMode ? '#f9fafb' : '#111827',
              border: uiState.darkMode ? '1px solid #374151' : '1px solid #e5e7eb',
            },
          }}
        />
      </div>
    </>
  );
}
