import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { MLTask, DatasetInfo, ApiResponse } from '@/types';

class APIClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 300000, // 5 minutes timeout for long-running operations
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for logging and auth
    this.client.interceptors.request.use(
      (config) => {
        console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        if (config.data && typeof config.data === 'object') {
          console.log('📤 Request Data:', config.data);
        }
        return config;
      },
      (error) => {
        console.error('❌ API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for logging and error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message;
        console.error('❌ API Response Error:', {
          status: error.response?.status,
          message: errorMessage,
          url: error.config?.url
        });
        
        // Transform error for better handling
        const transformedError = new Error(errorMessage);
        (transformedError as any).status = error.response?.status;
        (transformedError as any).response = error.response;
        
        return Promise.reject(transformedError);
      }
    );
  }

  // Health check
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    version: string;
    performance: {
      cpu_percent: number;
      memory_mb: number;
      uptime_seconds: number;
    };
    features: {
      google_adk_integration: boolean;
      real_time_progress: boolean;
      quality_feedback_loops: boolean;
      statistical_synthetic_data: boolean;
      parallel_model_training: boolean;
      hyperparameter_optimization: boolean;
      ensemble_creation: boolean;
    };
  }> {
    const response = await this.client.get('/api/v1/health');
    return response.data;
  }

  // Start ML pipeline - UPDATED to match backend
  async startPipeline(params: {
    prompt: string;
    mode?: string;
    dataset_path?: string;
  }): Promise<{ 
    task_id: string; 
    session_id: string; 
    status: string; 
    message: string 
  }> {
    try {
      const response = await this.client.post('/api/v1/pipeline/start', {
        prompt: params.prompt,
        mode: params.mode || 'prompt_only',
        dataset_path: params.dataset_path
      });
      
      console.log('🎯 Pipeline started successfully:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Failed to start pipeline:', error);
      throw error;
    }
  }

  // Get pipeline status
  async getPipelineStatus(taskId: string): Promise<MLTask> {
    try {
      const response = await this.client.get(`/api/v1/pipeline/status/${taskId}`);
      return response.data;
    } catch (error) {
      console.error(`❌ Failed to get status for task ${taskId}:`, error);
      throw error;
    }
  }

  // Upload dataset
  async uploadDataset(file: File, onProgress?: (progress: number) => void): Promise<DatasetInfo> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.client.post('/api/v1/dataset/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            console.log(`📤 Upload Progress: ${percentCompleted}%`);
            onProgress?.(percentCompleted);
          }
        },
      });

      return {
        filename: response.data.filename,
        saved_as: response.data.saved_as,
        path: response.data.path,
        size_mb: response.data.size_mb,
        upload_time: response.data.upload_time
      };
    } catch (error) {
      console.error('❌ Dataset upload failed:', error);
      throw error;
    }
  }

  // Download model
  async downloadModel(taskId: string): Promise<Blob> {
    try {
      const response = await this.client.get(`/api/v1/model/download/${taskId}`, {
        responseType: 'blob',
      });
      
      console.log('✅ Model download successful for task:', taskId);
      return response.data;
    } catch (error) {
      console.error(`❌ Failed to download model for task ${taskId}:`, error);
      throw error;
    }
  }

  // Cancel task
  async cancelTask(taskId: string): Promise<{ message: string }> {
    try {
      const response = await this.client.delete(`/api/v1/tasks/${taskId}`);
      console.log(`🚫 Task ${taskId} cancelled successfully`);
      return response.data;
    } catch (error) {
      console.error(`❌ Failed to cancel task ${taskId}:`, error);
      throw error;
    }
  }

  // List tasks
  async listTasks(): Promise<{ 
    active_tasks: number; 
    tasks: Array<{
      task_id: string;
      status: string;
      progress: number;
      created_at: string;
      current_stage?: string;
    }>;
  }> {
    try {
      const response = await this.client.get('/api/v1/tasks');
      return response.data;
    } catch (error) {
      console.error('❌ Failed to list tasks:', error);
      throw error;
    }
  }

  // Get system information
  async getSystemInfo(): Promise<{
    system: {
      platform: string;
      python_version: string;
      cpu_count: number;
      memory_total_gb: number;
      memory_available_gb: number;
    };
    application: {
      version: string;
      active_tasks: number;
      active_sessions: number;
      active_connections: number;
    };
    timestamp: string;
  }> {
    try {
      const response = await this.client.get('/api/v1/system/info');
      return response.data;
    } catch (error) {
      console.error('❌ Failed to get system info:', error);
      throw error;
    }
  }

  // Cleanup system
  async cleanupSystem(): Promise<{
    message: string;
    cleaned_tasks: number;
    timestamp: string;
  }> {
    try {
      const response = await this.client.post('/api/v1/system/cleanup');
      console.log('🧹 System cleanup completed:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Failed to cleanup system:', error);
      throw error;
    }
  }

  // Execute code (placeholder for future implementation)
  async executeCode(code: string, language: string = 'python'): Promise<{
    success: boolean;
    output?: string;
    error?: string;
    execution_time?: number;
  }> {
    try {
      // Check if endpoint exists
      const response = await this.client.post('/api/v1/code/execute', {
        code,
        language,
      });
      return response.data;
    } catch (error) {
      console.warn('⚠️ Code execution endpoint not available:', error);
      return {
        success: false,
        error: 'Code execution not implemented on backend',
        output: `# Code execution not available\n# Submitted code:\n${code}`
      };
    }
  }

  // Get Socket.IO connection URL (updated)
  getSocketIOURL(): string {
    return this.baseURL;
  }

  // DEPRECATED: Use getSocketIOURL instead
  getWebSocketURL(sessionId: string): string {
    console.warn('⚠️ getWebSocketURL is deprecated. Use Socket.IO connection instead.');
    const wsProtocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
    const wsBaseURL = this.baseURL.replace(/^https?/, wsProtocol);
    return `${wsBaseURL}/ws/${sessionId}`;
  }

  // Advanced pipeline operations
  async retryTask(taskId: string): Promise<{ 
    task_id: string; 
    session_id: string; 
    status: string; 
    message: string 
  }> {
    try {
      const response = await this.client.post(`/api/v1/pipeline/retry/${taskId}`);
      console.log(`🔄 Task ${taskId} retry initiated:`, response.data);
      return response.data;
    } catch (error) {
      console.error(`❌ Failed to retry task ${taskId}:`, error);
      throw error;
    }
  }

  // Get task logs
  async getTaskLogs(taskId: string): Promise<{
    logs: Array<{
      timestamp: string;
      level: string;
      message: string;
      agent?: string;
    }>;
  }> {
    try {
      const response = await this.client.get(`/api/v1/tasks/${taskId}/logs`);
      return response.data;
    } catch (error) {
      console.warn(`⚠️ Could not fetch logs for task ${taskId}:`, error);
      return { logs: [] };
    }
  }

  // Batch operations
  async batchCancelTasks(taskIds: string[]): Promise<{
    cancelled: string[];
    failed: string[];
  }> {
    const results = await Promise.allSettled(
      taskIds.map(id => this.cancelTask(id))
    );

    const cancelled: string[] = [];
    const failed: string[] = [];

    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        cancelled.push(taskIds[index]);
      } else {
        failed.push(taskIds[index]);
      }
    });

    console.log(`🚫 Batch cancel results: ${cancelled.length} cancelled, ${failed.length} failed`);
    return { cancelled, failed };
  }

  // Check API connectivity
  async checkConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      console.error('❌ API connection check failed:', error);
      return false;
    }
  }

  // Get API base URL
  getBaseURL(): string {
    return this.baseURL;
  }

  // Update base URL (useful for environment switching)
  updateBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL;
    this.client.defaults.baseURL = newBaseURL;
    console.log(`🔄 API base URL updated to: ${newBaseURL}`);
  }
}

// Create singleton instance
export const apiClient = new APIClient();

// Export individual methods for easier use
export const {
  healthCheck,
  startPipeline,
  getPipelineStatus,
  uploadDataset,
  downloadModel,
  cancelTask,
  listTasks,
  executeCode,
  getWebSocketURL,
  getSocketIOURL,
  getSystemInfo,
  cleanupSystem,
  retryTask,
  getTaskLogs,
  batchCancelTasks,
  checkConnection,
  getBaseURL,
  updateBaseURL
} = apiClient;

// Export the class for direct instantiation if needed
export default APIClient;

// Type definitions for better TypeScript support
export interface PipelineStartParams {
  prompt: string;
  mode?: 'prompt_only' | 'dataset_provided';
  dataset_path?: string;
}

export interface PipelineStartResponse {
  task_id: string;
  session_id: string;
  status: string;
  message: string;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

// Utility functions
export const createFormData = (file: File, additionalFields?: Record<string, string>): FormData => {
  const formData = new FormData();
  formData.append('file', file);
  
  if (additionalFields) {
    Object.entries(additionalFields).forEach(([key, value]) => {
      formData.append(key, value);
    });
  }
  
  return formData;
};

export const downloadFile = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

// Error handling utilities
export const isAPIError = (error: any): error is { status: number; message: string; response?: any } => {
  return error && typeof error.status === 'number' && typeof error.message === 'string';
};

export const getErrorMessage = (error: any): string => {
  if (isAPIError(error)) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'An unknown error occurred';
};

// Connection status checker
export const createConnectionMonitor = (intervalMs: number = 30000) => {
  let interval: NodeJS.Timeout;
  let isConnected = true;

  const start = (onStatusChange?: (connected: boolean) => void) => {
    interval = setInterval(async () => {
      const connected = await apiClient.checkConnection();
      if (connected !== isConnected) {
        isConnected = connected;
        onStatusChange?.(connected);
        console.log(`🔔 API connection status changed: ${connected ? 'Connected' : 'Disconnected'}`);
      }
    }, intervalMs);
  };

  const stop = () => {
    if (interval) {
      clearInterval(interval);
    }
  };

  const getStatus = () => isConnected;

  return { start, stop, getStatus };
};
