import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronRight, 
  ChevronDown, 
  File, 
  Folder, 
  FolderOpen,
  FileText,
  Code,
  Settings,
  Database,
  Image,
  Archive,
  Search,
  X,
  MoreHorizontal,
  Download,
  Copy,
  Eye
} from 'lucide-react';
import { FileNode } from '@/types';

interface Props {
  files: FileNode[];
  selectedFile: string | null;
  onFileSelect: (file: FileNode) => void;
  searchQuery?: string;
}

export default function FileExplorer({ files, selectedFile, onFileSelect, searchQuery = '' }: Props) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['/']));
  const [searchInput, setSearchInput] = useState(searchQuery);
  const [showSearch, setShowSearch] = useState(false);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; file: FileNode } | null>(null);

  // Filter files based on search
  const filteredFiles = useMemo(() => {
    if (!searchInput.trim()) return files;
    
    const filterTree = (nodes: FileNode[]): FileNode[] => {
      return nodes.reduce((acc: FileNode[], node) => {
        if (node.type === 'file' && node.name.toLowerCase().includes(searchInput.toLowerCase())) {
          acc.push(node);
        } else if (node.type === 'folder' && node.children) {
          const filteredChildren = filterTree(node.children);
          if (filteredChildren.length > 0) {
            acc.push({ ...node, children: filteredChildren });
          } else if (node.name.toLowerCase().includes(searchInput.toLowerCase())) {
            acc.push(node);
          }
        }
        return acc;
      }, []);
    };
    
    return filterTree(files);
  }, [files, searchInput]);

  // Calculate file statistics
  const fileStats = useMemo(() => {
    const calculateStats = (nodes: FileNode[]): { files: number; folders: number; totalSize: number } => {
      let files = 0;
      let folders = 0;
      let totalSize = 0;
      
      nodes.forEach(node => {
        if (node.type === 'file') {
          files++;
          totalSize += node.size || 0;
        } else if (node.type === 'folder') {
          folders++;
          if (node.children) {
            const childStats = calculateStats(node.children);
            files += childStats.files;
            folders += childStats.folders;
            totalSize += childStats.totalSize;
          }
        }
      });
      
      return { files, folders, totalSize };
    };
    
    return calculateStats(files);
  }, [files]);

  const toggleFolder = (path: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  };

  const getFileIcon = (file: FileNode) => {
    if (file.type === 'folder') {
      return expandedFolders.has(file.path) ? (
        <FolderOpen className="w-4 h-4 text-blue-500" />
      ) : (
        <Folder className="w-4 h-4 text-blue-500" />
      );
    }

    const extension = file.name.split('.').pop()?.toLowerCase();
    const iconMap: Record<string, { icon: JSX.Element; color: string }> = {
      'py': { icon: <Code className="w-4 h-4" />, color: 'text-green-500' },
      'js': { icon: <Code className="w-4 h-4" />, color: 'text-yellow-500' },
      'ts': { icon: <Code className="w-4 h-4" />, color: 'text-blue-500' },
      'tsx': { icon: <Code className="w-4 h-4" />, color: 'text-blue-600' },
      'jsx': { icon: <Code className="w-4 h-4" />, color: 'text-cyan-500' },
      'json': { icon: <Settings className="w-4 h-4" />, color: 'text-orange-500' },
      'md': { icon: <FileText className="w-4 h-4" />, color: 'text-gray-600' },
      'txt': { icon: <FileText className="w-4 h-4" />, color: 'text-gray-500' },
      'csv': { icon: <Database className="w-4 h-4" />, color: 'text-green-600' },
      'yml': { icon: <Settings className="w-4 h-4" />, color: 'text-purple-500' },
      'yaml': { icon: <Settings className="w-4 h-4" />, color: 'text-purple-500' },
      'png': { icon: <Image className="w-4 h-4" />, color: 'text-pink-500' },
      'jpg': { icon: <Image className="w-4 h-4" />, color: 'text-pink-500' },
      'jpeg': { icon: <Image className="w-4 h-4" />, color: 'text-pink-500' },
      'zip': { icon: <Archive className="w-4 h-4" />, color: 'text-gray-600' },
      'tar': { icon: <Archive className="w-4 h-4" />, color: 'text-gray-600' },
      'gz': { icon: <Archive className="w-4 h-4" />, color: 'text-gray-600' },
      'dockerfile': { icon: <Code className="w-4 h-4" />, color: 'text-indigo-500' },
      'html': { icon: <Code className="w-4 h-4" />, color: 'text-orange-600' },
      'css': { icon: <Code className="w-4 h-4" />, color: 'text-blue-400' },
      'scss': { icon: <Code className="w-4 h-4" />, color: 'text-pink-400' },
      'sql': { icon: <Database className="w-4 h-4" />, color: 'text-blue-700' },
    };

    const fileInfo = iconMap[extension || ''] || { icon: <File className="w-4 h-4" />, color: 'text-gray-400' };
    return <span className={fileInfo.color}>{fileInfo.icon}</span>;
  };

  const formatFileSize = (size?: number) => {
    if (!size) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    let fileSize = size;
    
    while (fileSize >= 1024 && i < units.length - 1) {
      fileSize /= 1024;
      i++;
    }
    
    return `${fileSize.toFixed(1)} ${units[i]}`;
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return '';
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleContextMenu = (e: React.MouseEvent, file: FileNode) => {
    e.preventDefault();
    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      file
    });
  };

  const closeContextMenu = () => {
    setContextMenu(null);
  };

  const handleCopyPath = (file: FileNode) => {
    navigator.clipboard.writeText(file.path);
    closeContextMenu();
  };

  const renderFileNode = (file: FileNode, depth: number = 0) => {
    const isExpanded = expandedFolders.has(file.path);
    const isSelected = selectedFile === file.path;

    return (
      <div key={file.path}>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          whileHover={{ backgroundColor: 'rgba(59, 130, 246, 0.05)' }}
          className={`flex items-center gap-2 px-3 py-2 cursor-pointer rounded-md mx-2 group ${
            isSelected 
              ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300' 
              : 'hover:bg-gray-100 dark:hover:bg-dark-700'
          }`}
          style={{ marginLeft: depth * 16 }}
          onClick={() => {
            if (file.type === 'folder') {
              toggleFolder(file.path);
            } else {
              onFileSelect(file);
            }
          }}
          onContextMenu={(e) => handleContextMenu(e, file)}
        >
          {file.type === 'folder' && (
            <button 
              className="p-0.5 hover:bg-gray-200 dark:hover:bg-dark-600 rounded transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                toggleFolder(file.path);
              }}
            >
              {isExpanded ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </button>
          )}
          
          {file.type === 'file' && <div className="w-4" />}
          
          {getFileIcon(file)}
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium truncate">{file.name}</span>
              {file.type === 'file' && file.size && (
                <span className="text-xs text-gray-400 ml-2 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                  {formatFileSize(file.size)}
                </span>
              )}
            </div>
            {file.modified && (
              <div className="text-xs text-gray-400 truncate">
                {formatDate(file.modified)}
              </div>
            )}
          </div>

          {/* Context menu trigger */}
          <button
            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-300 dark:hover:bg-dark-600 rounded transition-all"
            onClick={(e) => {
              e.stopPropagation();
              handleContextMenu(e, file);
            }}
          >
            <MoreHorizontal className="w-3 h-3" />
          </button>
        </motion.div>

        <AnimatePresence>
          {file.type === 'folder' && isExpanded && file.children && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              {file.children
                .sort((a, b) => {
                  // Folders first, then files
                  if (a.type === 'folder' && b.type === 'file') return -1;
                  if (a.type === 'file' && b.type === 'folder') return 1;
                  return a.name.localeCompare(b.name);
                })
                .map(child => renderFileNode(child, depth + 1))
              }
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col bg-white dark:bg-dark-800">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-dark-700">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Folder className="w-5 h-5" />
            Project Explorer
          </h2>
          <button
            onClick={() => setShowSearch(!showSearch)}
            className={`p-1.5 rounded-lg transition-colors ${
              showSearch 
                ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
                : 'hover:bg-gray-100 dark:hover:bg-dark-700'
            }`}
          >
            <Search className="w-4 h-4" />
          </button>
        </div>

        {/* Search */}
        <AnimatePresence>
          {showSearch && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="relative mb-3">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search files..."
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="w-full pl-10 pr-8 py-2 text-sm border border-gray-200 dark:border-dark-600 rounded-lg bg-gray-50 dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                {searchInput && (
                  <button
                    onClick={() => setSearchInput('')}
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* File Statistics */}
        <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
          <div className="flex justify-between">
            <span>Files:</span>
            <span className="font-medium">{fileStats.files}</span>
          </div>
          <div className="flex justify-between">
            <span>Folders:</span>
            <span className="font-medium">{fileStats.folders}</span>
          </div>
          <div className="flex justify-between">
            <span>Total Size:</span>
            <span className="font-medium">{formatFileSize(fileStats.totalSize)}</span>
          </div>
        </div>
      </div>

      {/* File Tree */}
      <div className="flex-1 overflow-auto py-2">
        {filteredFiles.length > 0 ? (
          <div className="space-y-1">
            {filteredFiles.map(file => renderFileNode(file))}
          </div>
        ) : searchInput ? (
          <div className="p-6 text-center">
            <Search className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
              No files found for "{searchInput}"
            </p>
            <button
              onClick={() => setSearchInput('')}
              className="text-sm text-primary-600 dark:text-primary-400 hover:underline"
            >
              Clear search
            </button>
          </div>
        ) : (
          <div className="p-6 text-center">
            <Folder className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
            <p className="text-sm text-gray-500 dark:text-gray-400">
              No files to display
            </p>
          </div>
        )}
      </div>

      {/* Context Menu */}
      <AnimatePresence>
        {contextMenu && (
          <>
            <div 
              className="fixed inset-0 z-40"
              onClick={closeContextMenu}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="fixed z-50 bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-700 rounded-lg shadow-lg py-2 min-w-[160px]"
              style={{
                left: contextMenu.x,
                top: contextMenu.y,
              }}
            >
              <button
                onClick={() => {
                  if (contextMenu.file.type === 'file') {
                    onFileSelect(contextMenu.file);
                  }
                  closeContextMenu();
                }}
                className="w-full px-4 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-dark-700 flex items-center gap-2"
              >
                <Eye className="w-4 h-4" />
                {contextMenu.file.type === 'file' ? 'Open' : 'Expand'}
              </button>
              
              <button
                onClick={() => handleCopyPath(contextMenu.file)}
                className="w-full px-4 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-dark-700 flex items-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Copy Path
              </button>
              
              {contextMenu.file.type === 'file' && (
                <button
                  onClick={() => {
                    // Implement download functionality
                    closeContextMenu();
                  }}
                  className="w-full px-4 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-dark-700 flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200 dark:border-dark-700 text-xs text-gray-500 dark:text-gray-400">
        <div className="flex justify-between items-center">
          <span>Generated ML Project</span>
          {searchInput ? (
            <span>{filteredFiles.length} matches</span>
          ) : (
            <span>{fileStats.files + fileStats.folders} items</span>
          )}
        </div>
      </div>
    </div>
  );
}
