export interface MLTask {
  task_id: string;
  session_id: string;
  user_prompt: string;
  dataset_path?: string;
  mode: 'prompt_only' | 'dataset_provided' | 'synthetic';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_stage?: string;
  created_at: string;
  completed_at?: string;
  failed_at?: string;
  results?: MLTaskResults;
  performance_metrics?: PerformanceMetrics;
  error?: string;
}

export interface MLTaskResults {
  task_id: string;
  status: string;
  requirements?: any;
  data_profile?: any;
  preprocessing_result?: any;
  feature_engineering_result?: any;
  training_results?: any;
  optimization_result?: any;
  ensemble_result?: any;
  evaluation_result?: any;
  code_generation_result?: any;
  model_performance?: ModelPerformance;
  artifacts?: GeneratedArtifacts;
  execution_summary?: ExecutionSummary;
}

export interface ModelPerformance {
  best_model: {
    name: string;
    type: string;
    test_score: number;
    cv_mean: number;
    cv_std: number;
    feature_importance: Record<string, number>;
  };
  all_models: ModelResult[];
  ensemble_performance?: {
    score: number;
    improvement: number;
  };
}

export interface ModelResult {
  model_type: string;
  task_type: string;
  train_score: number;
  test_score: number;
  cv_mean: number;
  cv_std: number;
  additional_metrics: Record<string, number>;
  feature_importance: Record<string, number>;
  model_path: string;
}

export interface GeneratedArtifacts {
  project_directory: string;
  generated_files: {
    inference_code: string;
    streamlit_app: string;
    requirements: string;
    readme: string;
    docker_file?: string;
  };
  model_files: string[];
  documentation: string[];
}

export interface ExecutionSummary {
  total_execution_time: number;
  stages_completed: number;
  data_quality_iterations: number;
  model_quality_iterations: number;
  final_dataset_shape: [number, number];
  features_engineered: number;
  models_trained: number;
  optimization_applied: boolean;
  ensemble_created: boolean;
}

export interface PerformanceMetrics {
  cpu_percent: number;
  memory_mb: number;
  execution_time: number;
  timestamp: string;
}

export interface AgentProgress {
  agent_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  timestamp: string;
  stage_details?: {
    current_step: string;
    total_steps: number;
    step_progress: number;
  };
  quality_metrics?: {
    score: number;
    issues: string[];
    should_reiterate: boolean;
  };
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  agent: string;
  stage: string;
  message: string;
  details?: any;
  performance?: {
    memory_mb: number;
    execution_time: number;
  };
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system' | 'agent';
  content: string;
  timestamp: string;
  task_id?: string;
  agent_name?: string;
  metadata?: {
    progress?: number;
    stage?: string;
    quality_score?: number;
    performance?: PerformanceMetrics;
  };
  attachments?: FileAttachment[];
}

export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  url?: string;
  preview?: string;
}

export interface FileNode {
  name: string;
  type: 'file' | 'folder';
  path: string;
  content?: string;
  language?: string;
  children?: FileNode[];
  size?: number;
  modified?: string;
  isExpanded?: boolean;
  isSelected?: boolean;
  icon?: string;
}

export interface CodeExecutionResult {
  success: boolean;
  output?: string;
  error?: string;
  logs?: string[];
  execution_time?: number;
  memory_usage?: number;
  exit_code?: number;
}

export interface DatasetInfo {
  filename: string;
  path: string;
  size_mb: number;
  rows?: number;
  columns?: number;
  preview?: any[];
  dtypes?: Record<string, string>;
  quality_score?: number;
  missing_values?: Record<string, number>;
  duplicates?: number;
  memory_usage_mb?: number;
}

export interface WebSocketMessage {
  type: 'connected' | 'pipeline_started' | 'progress_update' | 'pipeline_completed' | 'pipeline_failed' | 'agent_update' | 'log_entry' | 'heartbeat';
  task_id?: string;
  session_id?: string;
  message?: string;
  data?: any;
  timestamp?: string;
  progress?: number;
  stage?: string;
  agent?: AgentProgress;
  log?: LogEntry;
  results?: MLTaskResults;
  error?: string;
}

export interface UIState {
  activeView: 'welcome' | 'chat' | 'code' | 'results';
  sidebarOpen: boolean;
  darkMode: boolean;
  currentTask: MLTask | null;
  isProcessing: boolean;
  showLogs: boolean;
  selectedFile: string | null;
  codeEditorTheme: 'vs-dark' | 'light';
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}
