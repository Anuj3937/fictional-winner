import { useState, useCallback, useRef } from 'react';
import { FileNode } from '@/types';

interface UseCodeEditorReturn {
  openFiles: FileNode[];
  activeFile: string | null;
  isDirty: boolean;
  openFile: (file: FileNode) => void;
  closeFile: (filePath: string) => void;
  updateFileContent: (filePath: string, content: string) => void;
  saveFile: (filePath: string) => Promise<boolean>;
  saveAllFiles: () => Promise<boolean>;
  getFileContent: (filePath: string) => string | null;
  setActiveFile: (filePath: string) => void;
}

export function useCodeEditor(): UseCodeEditorReturn {
  const [openFiles, setOpenFiles] = useState<FileNode[]>([]);
  const [activeFile, setActiveFileState] = useState<string | null>(null);
  const [dirtyFiles, setDirtyFiles] = useState<Set<string>>(new Set());
  const saveTimeoutRef = useRef<NodeJS.Timeout>();

  const openFile = useCallback((file: FileNode) => {
    setOpenFiles(prev => {
      const exists = prev.find(f => f.path === file.path);
      if (exists) {
        setActiveFileState(file.path);
        return prev;
      }
      return [...prev, file];
    });
    setActiveFileState(file.path);
  }, []);

  const closeFile = useCallback((filePath: string) => {
    setOpenFiles(prev => prev.filter(f => f.path !== filePath));
    setDirtyFiles(prev => {
      const newSet = new Set(prev);
      newSet.delete(filePath);
      return newSet;
    });
    
    if (activeFile === filePath) {
      setOpenFiles(prev => {
        if (prev.length > 1) {
          const index = prev.findIndex(f => f.path === filePath);
          const nextFile = prev[index + 1] || prev[index - 1];
          if (nextFile && nextFile.path !== filePath) {
            setActiveFileState(nextFile.path);
          }
        } else {
          setActiveFileState(null);
        }
        return prev.filter(f => f.path !== filePath);
      });
    }
  }, [activeFile]);

  const updateFileContent = useCallback((filePath: string, content: string) => {
    setOpenFiles(prev => prev.map(file => 
      file.path === filePath 
        ? { ...file, content }
        : file
    ));
    
    setDirtyFiles(prev => {
      const newSet = new Set(prev);
      newSet.add(filePath);
      return newSet;
    });

    // Auto-save after 2 seconds of inactivity
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    saveTimeoutRef.current = setTimeout(() => {
      saveFile(filePath);
    }, 2000);
  }, []);

  const saveFile = useCallback(async (filePath: string): Promise<boolean> => {
    try {
      // In a real implementation, this would save to the backend
      setDirtyFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(filePath);
        return newSet;
      });
      
      console.log(`Saved file: ${filePath}`);
      return true;
    } catch (error) {
      console.error(`Failed to save file ${filePath}:`, error);
      return false;
    }
  }, []);

  const saveAllFiles = useCallback(async (): Promise<boolean> => {
    try {
      const savePromises = Array.from(dirtyFiles).map(filePath => saveFile(filePath));
      const results = await Promise.all(savePromises);
      return results.every(result => result);
    } catch (error) {
      console.error('Failed to save all files:', error);
      return false;
    }
  }, [dirtyFiles, saveFile]);

  const getFileContent = useCallback((filePath: string): string | null => {
    const file = openFiles.find(f => f.path === filePath);
    return file?.content || null;
  }, [openFiles]);

  const setActiveFile = useCallback((filePath: string) => {
    const file = openFiles.find(f => f.path === filePath);
    if (file) {
      setActiveFileState(filePath);
    }
  }, [openFiles]);

  const isDirty = dirtyFiles.size > 0;

  return {
    openFiles,
    activeFile,
    isDirty,
    openFile,
    closeFile,
    updateFileContent,
    saveFile,
    saveAllFiles,
    getFileContent,
    setActiveFile
  };
}
