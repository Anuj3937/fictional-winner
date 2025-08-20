import { useState, useCallback, useRef } from 'react';
import { MLTask, DatasetInfo } from '@/types';
import { apiClient } from '@/utils/api';
import toast from 'react-hot-toast';

interface UseMLPipelineReturn {
  currentTask: MLTask | null;
  isLoading: boolean;
  error: string | null;
  createTask: (params: CreateTaskParams) => Promise<MLTask | null>;
  getTaskStatus: (taskId: string) => Promise<MLTask | null>;
  cancelTask: (taskId: string) => Promise<boolean>;
  downloadModel: (taskId: string) => Promise<boolean>;
  uploadDataset: (file: File) => Promise<DatasetInfo | null>;
  clearError: () => void;
  resetTask: () => void;
}

interface CreateTaskParams {
  prompt: string; // FIXED: Changed from user_prompt to prompt
  mode: 'prompt_only' | 'dataset_provided' | 'synthetic';
  dataset_files?: File[];
}

export function useMLPipeline(): UseMLPipelineReturn {
  const [currentTask, setCurrentTask] = useState<MLTask | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const createTask = useCallback(async (params: CreateTaskParams): Promise<MLTask | null> => {
    setIsLoading(true);
    setError(null);

    try {
      // Upload dataset files if provided
      let datasetPath: string | undefined;
      
      if (params.dataset_files && params.dataset_files.length > 0) {
        toast.loading('Uploading dataset...', { id: 'upload' });
        
        const uploadResult = await apiClient.uploadDataset(params.dataset_files[0]);
        datasetPath = uploadResult.path;
        
        toast.success(`Dataset uploaded: ${uploadResult.filename}`, { id: 'upload' });
      }

      // Create ML task - FIXED: Send 'prompt' not 'user_prompt'
      toast.loading('Starting ML pipeline...', { id: 'pipeline' });
      
      const response = await apiClient.startPipeline({
        prompt: params.prompt, // FIXED: This now matches backend
        mode: params.mode,
        dataset_path: datasetPath,
      });

      const task: MLTask = {
        task_id: response.task_id,
        session_id: response.session_id,
        user_prompt: params.prompt, // Frontend expects user_prompt
        dataset_path: datasetPath,
        mode: params.mode,
        status: 'running',
        progress: 0,
        created_at: new Date().toISOString(),
      };

      setCurrentTask(task);
      toast.success('ML pipeline started!', { id: 'pipeline' });
      
      return task;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create ML task';
      setError(errorMessage);
      toast.error(errorMessage);
      console.error('Create task error:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getTaskStatus = useCallback(async (taskId: string): Promise<MLTask | null> => {
    try {
      const task = await apiClient.getPipelineStatus(taskId);
      setCurrentTask(task);
      return task;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get task status';
      setError(errorMessage);
      console.error('Get task status error:', err);
      return null;
    }
  }, []);

  const cancelTask = useCallback(async (taskId: string): Promise<boolean> => {
    try {
      await apiClient.cancelTask(taskId);
      
      if (currentTask?.task_id === taskId) {
        setCurrentTask(prev => prev ? { ...prev, status: 'cancelled' } : null);
      }
      
      toast.success('Task cancelled successfully');
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to cancel task';
      setError(errorMessage);
      toast.error(errorMessage);
      console.error('Cancel task error:', err);
      return false;
    }
  }, [currentTask]);

  const downloadModel = useCallback(async (taskId: string): Promise<boolean> => {
    try {
      toast.loading('Preparing model download...', { id: 'download' });
      
      const blob = await apiClient.downloadModel(taskId);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `ml_model_${taskId}.pkl`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success('Model downloaded successfully!', { id: 'download' });
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to download model';
      setError(errorMessage);
      toast.error(errorMessage, { id: 'download' });
      console.error('Download model error:', err);
      return false;
    }
  }, []);

  const uploadDataset = useCallback(async (file: File): Promise<DatasetInfo | null> => {
    try {
      setIsLoading(true);
      toast.loading(`Uploading ${file.name}...`, { id: 'upload' });
      
      const result = await apiClient.uploadDataset(file);
      
      toast.success(`${file.name} uploaded successfully!`, { id: 'upload' });
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to upload dataset';
      setError(errorMessage);
      toast.error(errorMessage, { id: 'upload' });
      console.error('Upload dataset error:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const resetTask = useCallback(() => {
    setCurrentTask(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    currentTask,
    isLoading,
    error,
    createTask,
    getTaskStatus,
    cancelTask,
    downloadModel,
    uploadDataset,
    clearError,
    resetTask,
  };
}
