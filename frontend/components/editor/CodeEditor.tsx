import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Editor from '@monaco-editor/react';
import FileExplorer from './FileExplorer';
import TabManager from './TabManager';
import ExecutionPanel from './ExecutionPanel';
import { 
  Play, 
  Square, 
  Download, 
  Save, 
  RotateCcw,
  Settings,
  Terminal,
  Maximize2,
  Minimize2,
  PanelRightClose,
  PanelRightOpen,
  FileText,
  Folder,
  Search,
  Copy,
  ExternalLink
} from 'lucide-react';
import { FileNode, CodeExecutionResult } from '@/types';
import { apiClient } from '@/utils/api';
import toast from 'react-hot-toast';

interface Props {
  fileTree: FileNode[];
  selectedFile: string | null;
  onFileSelect: (path: string) => void;
  onCodeChange: (path: string, content: string) => void;
  darkMode?: boolean;
}

interface OpenTab {
  path: string;
  name: string;
  language: string;
  content: string;
  isDirty: boolean;
  isReadonly?: boolean;
}

export default function CodeEditor({ 
  fileTree, 
  selectedFile, 
  onFileSelect, 
  onCodeChange,
  darkMode = true 
}: Props) {
  const [openTabs, setOpenTabs] = useState<OpenTab[]>([]);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<CodeExecutionResult | null>(null);
  const [showTerminal, setShowTerminal] = useState(false);
  const [showFileExplorer, setShowFileExplorer] = useState(true);
  const [editorTheme, setEditorTheme] = useState(darkMode ? 'vs-dark' : 'light');
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [editorOptions, setEditorOptions] = useState({
    fontSize: 14,
    lineNumbers: 'on' as const,
    minimap: { enabled: true },
    wordWrap: 'on' as const,
    formatOnPaste: true,
    formatOnType: true,
    autoIndent: 'advanced' as const
  });
  
  const editorRef = useRef<any>(null);

  // Auto-open main files when fileTree changes
  useEffect(() => {
    if (fileTree.length > 0 && openTabs.length === 0) {
      const findMainFiles = (nodes: FileNode[]): FileNode[] => {
        const mainFiles: FileNode[] = [];
        
        const traverse = (nodes: FileNode[]) => {
          for (const node of nodes) {
            if (node.type === 'file') {
              // Prioritize main files
              const mainFileNames = ['main.py', 'app.py', 'model_inference.py', 'streamlit_app.py', 'README.md'];
              if (mainFileNames.includes(node.name) || node.name.endsWith('.py')) {
                mainFiles.push(node);
              }
            }
            if (node.children) {
              traverse(node.children);
            }
          }
        };
        
        traverse(nodes);
        return mainFiles;
      };
      
      const mainFiles = findMainFiles(fileTree);
      if (mainFiles.length > 0) {
        // Open the first few main files
        mainFiles.slice(0, 3).forEach(file => {
          handleFileSelect(file);
        });
      }
    }
  }, [fileTree]);

  const handleFileSelect = useCallback((file: FileNode) => {
    if (file.type === 'file') {
      onFileSelect(file.path);
      
      // Check if tab is already open
      const existingTab = openTabs.find(tab => tab.path === file.path);
      if (!existingTab) {
        const newTab: OpenTab = {
          path: file.path,
          name: file.name,
          language: file.language || getLanguageFromExtension(file.name),
          content: file.content || '',
          isDirty: false,
          isReadonly: file.name === 'README.md' || file.name.endsWith('.txt')
        };
        setOpenTabs(prev => [...prev, newTab]);
      }
      setActiveTab(file.path);
    }
  }, [openTabs, onFileSelect]);

  const handleCodeChange = useCallback((value: string | undefined) => {
    if (activeTab && value !== undefined) {
      setOpenTabs(prev => prev.map(tab => 
        tab.path === activeTab 
          ? { ...tab, content: value, isDirty: true }
          : tab
      ));
      onCodeChange(activeTab, value);
    }
  }, [activeTab, onCodeChange]);

  const handleTabClose = useCallback((path: string) => {
    setOpenTabs(prev => {
      const newTabs = prev.filter(tab => tab.path !== path);
      if (activeTab === path && newTabs.length > 0) {
        setActiveTab(newTabs[newTabs.length - 1].path);
      } else if (newTabs.length === 0) {
        setActiveTab(null);
      }
      return newTabs;
    });
  }, [activeTab]);

  const handleSave = useCallback(async () => {
    if (activeTab) {
      const tab = openTabs.find(t => t.path === activeTab);
      if (tab) {
        try {
          // In a real implementation, this would save to the backend
          setOpenTabs(prev => prev.map(t => 
            t.path === activeTab ? { ...t, isDirty: false } : t
          ));
          toast.success(`Saved ${tab.name}`);
        } catch (error) {
          toast.error('Failed to save file');
        }
      }
    }
  }, [activeTab, openTabs]);

  const handleExecute = useCallback(async () => {
    if (!activeTab) return;
    
    const tab = openTabs.find(t => t.path === activeTab);
    if (!tab || tab.language !== 'python') {
      toast.error('Can only execute Python files');
      return;
    }

    setIsExecuting(true);
    setShowTerminal(true);
    
    try {
      const result = await apiClient.executeCode(tab.content, 'python');
      setExecutionResult(result);
      
      if (result.success) {
        toast.success('Code executed successfully');
      } else {
        toast.error('Code execution failed');
      }
    } catch (error) {
      const result: CodeExecutionResult = {
        success: false,
        error: error instanceof Error ? error.message : 'Execution failed'
      };
      setExecutionResult(result);
      toast.error('Execution failed');
    } finally {
      setIsExecuting(false);
    }
  }, [activeTab, openTabs]);

  const handleDownloadProject = useCallback(() => {
    try {
      const projectData = {
        files: openTabs.map(tab => ({
          path: tab.path,
          name: tab.name,
          content: tab.content,
          language: tab.language
        })),
        metadata: {
          created: new Date().toISOString(),
          fileCount: openTabs.length,
          totalSize: openTabs.reduce((size, tab) => size + tab.content.length, 0)
        }
      };
      
      const blob = new Blob([JSON.stringify(projectData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ml-project-export.json';
      a.click();
      URL.revokeObjectURL(url);
      
      toast.success('Project exported successfully');
    } catch (error) {
      toast.error('Failed to export project');
    }
  }, [openTabs]);

  const handleSearch = useCallback((query: string) => {
    if (editorRef.current && query.trim()) {
      const editor = editorRef.current;
      const model = editor.getModel();
      const matches = model.findMatches(query, false, false, false, null, false);
      
      if (matches.length > 0) {
        editor.setSelection(matches[0].range);
        editor.revealRangeInCenter(matches.range);
        toast.success(`Found ${matches.length} matches`);
      } else {
        toast.error('No matches found');
      }
    }
  }, []);

  const getLanguageFromExtension = (filename: string): string => {
    const extension = filename.split('.').pop()?.toLowerCase();
    const languageMap: Record<string, string> = {
      'py': 'python',
      'js': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'jsx': 'javascript',
      'json': 'json',
      'md': 'markdown',
      'yml': 'yaml',
      'yaml': 'yaml',
      'txt': 'plaintext',
      'csv': 'csv',
      'sql': 'sql',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'xml': 'xml',
      'dockerfile': 'dockerfile',
      'sh': 'shell',
      'bat': 'batch'
    };
    return languageMap[extension || ''] || 'plaintext';
  };

  const currentTab = openTabs.find(tab => tab.path === activeTab);

  return (
    <div className="flex h-full bg-gray-50 dark:bg-dark-900">
      {/* File Explorer */}
      <AnimatePresence>
        {showFileExplorer && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="bg-white dark:bg-dark-800 border-r border-gray-200 dark:border-dark-700"
          >
            <FileExplorer 
              files={fileTree}
              selectedFile={selectedFile}
              onFileSelect={handleFileSelect}
              searchQuery={searchQuery}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Editor Area */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFileExplorer(!showFileExplorer)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
                title="Toggle File Explorer"
              >
                {showFileExplorer ? (
                  <PanelRightClose className="w-4 h-4" />
                ) : (
                  <PanelRightOpen className="w-4 h-4" />
                )}
              </button>

              <div className="h-4 w-px bg-gray-300 dark:bg-dark-600" />

              <button
                onClick={handleExecute}
                disabled={!currentTab || currentTab.language !== 'python' || isExecuting}
                className="flex items-center gap-2 px-3 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
              >
                {isExecuting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Running...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Run Code</span>
                  </>
                )}
              </button>

              <button
                onClick={handleSave}
                disabled={!currentTab || !currentTab.isDirty}
                className="flex items-center gap-2 px-3 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
              >
                <Save className="w-4 h-4" />
                <span>Save</span>
              </button>

              <button
                onClick={() => setShowSearch(!showSearch)}
                className={`p-2 rounded-lg transition-colors ${
                  showSearch 
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                    : 'hover:bg-gray-100 dark:hover:bg-dark-700'
                }`}
                title="Search in Code"
              >
                <Search className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center gap-2">
              {/* Search Bar */}
              <AnimatePresence>
                {showSearch && (
                  <motion.div
                    initial={{ width: 0, opacity: 0 }}
                    animate={{ width: 200, opacity: 1 }}
                    exit={{ width: 0, opacity: 0 }}
                    className="flex gap-2"
                  >
                    <input
                      type="text"
                      placeholder="Search in code..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSearch(searchQuery)}
                      className="px-3 py-1 text-sm border border-gray-300 dark:border-dark-600 rounded bg-white dark:bg-dark-700"
                    />
                  </motion.div>
                )}
              </AnimatePresence>

              <button
                onClick={() => setShowTerminal(!showTerminal)}
                className={`p-2 rounded-lg transition-colors ${
                  showTerminal 
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                    : 'hover:bg-gray-100 dark:hover:bg-dark-700'
                }`}
                title="Toggle Terminal"
              >
                <Terminal className="w-4 h-4" />
              </button>

              <button
                onClick={handleDownloadProject}
                className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
                title="Download Project"
              >
                <Download className="w-4 h-4" />
              </button>

              <button
                onClick={() => setEditorTheme(editorTheme === 'vs-dark' ? 'light' : 'vs-dark')}
                className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
                title="Toggle Theme"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Tabs */}
        {openTabs.length > 0 && (
          <TabManager
            tabs={openTabs}
            activeTab={activeTab}
            onTabSelect={setActiveTab}
            onTabClose={handleTabClose}
          />
        )}

        {/* Editor */}
        <div className="flex-1 relative">
          {currentTab ? (
            <div className="h-full">
              <Editor
                height="100%"
                language={currentTab.language}
                value={currentTab.content}
                onChange={handleCodeChange}
                theme={editorTheme}
                onMount={(editor) => {
                  editorRef.current = editor;
                  
                  // Add keyboard shortcuts
                  editor.addCommand(editor.KeyMod.CtrlCmd | editor.KeyCode.KeyS, handleSave);
                  editor.addCommand(editor.KeyMod.CtrlCmd | editor.KeyCode.Enter, handleExecute);
                  editor.addCommand(editor.KeyMod.CtrlCmd | editor.KeyCode.KeyF, () => setShowSearch(true));
                }}
                options={{
                  ...editorOptions,
                  readOnly: currentTab.isReadonly,
                  automaticLayout: true,
                  scrollBeyondLastLine: false,
                  bracketPairColorization: { enabled: true },
                  suggest: {
                    showKeywords: true,
                    showSnippets: true,
                    showFunctions: true,
                    showClasses: true,
                  },
                  quickSuggestions: {
                    other: true,
                    comments: false,
                    strings: false
                  },
                  parameterHints: { enabled: true },
                  folding: true,
                  foldingStrategy: 'indentation',
                  showFoldingControls: 'always',
                  unfoldOnClickAfterEndOfLine: true,
                }}
              />
              
              {/* File Info Overlay */}
              <div className="absolute bottom-4 right-4 bg-black/70 text-white px-3 py-1 rounded-lg text-xs">
                {currentTab.language} • {currentTab.content.length} chars
                {currentTab.isDirty && (
                  <span className="text-yellow-400 ml-2">• Unsaved</span>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full bg-gray-50 dark:bg-dark-900">
              <div className="text-center">
                <div className="w-20 h-20 bg-gray-200 dark:bg-dark-700 rounded-3xl flex items-center justify-center mx-auto mb-6">
                  <FileText className="w-10 h-10 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-3">
                  No File Selected
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md">
                  Select a file from the explorer to start editing your generated ML project
                </p>
                <button
                  onClick={() => setShowFileExplorer(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors mx-auto"
                >
                  <Folder className="w-4 h-4" />
                  Open File Explorer
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Terminal/Execution Panel */}
        <AnimatePresence>
          {showTerminal && (
            <ExecutionPanel
              isExecuting={isExecuting}
              result={executionResult}
              onClose={() => setShowTerminal(false)}
              onClear={() => setExecutionResult(null)}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
