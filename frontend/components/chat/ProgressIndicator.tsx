import { motion } from 'framer-motion';
import { 
  FileSearch, 
  Database, 
  Settings, 
  Zap, 
  Brain, 
  Target, 
  Layers, 
  BarChart3, 
  Package,
  CheckCircle,
  Clock,
  AlertCircle,
  TrendingUp,
  Code,
  Search
} from 'lucide-react';
import { AgentProgress, MLTask } from '@/types';

interface Props {
  agentProgress: AgentProgress[];
  currentStage: string;
  currentTask: MLTask;
}

const AGENT_STAGES = [
  { 
    name: 'Requirements Interpreter', 
    icon: FileSearch, 
    description: 'Understanding your ML requirements',
    color: 'from-blue-500 to-blue-600'
  },
  { 
    name: 'Data Acquisition', 
    icon: Database, 
    description: 'Finding and loading your dataset',
    color: 'from-green-500 to-green-600'
  },
  { 
    name: 'Synthetic Data Generator', 
    icon: Search, 
    description: 'Researching statistics and generating data',
    color: 'from-purple-500 to-purple-600'
  },
  { 
    name: 'Data Profiling', 
    icon: BarChart3, 
    description: 'Analyzing data quality and characteristics',
    color: 'from-orange-500 to-orange-600'
  },
  { 
    name: 'Preprocessing', 
    icon: Settings, 
    description: 'Cleaning and preparing your data',
    color: 'from-indigo-500 to-indigo-600'
  },
  { 
    name: 'Feature Engineering', 
    icon: Zap, 
    description: 'Creating intelligent features',
    color: 'from-yellow-500 to-yellow-600'
  },
  { 
    name: 'Model Training', 
    icon: Brain, 
    description: 'Training multiple ML models in parallel',
    color: 'from-pink-500 to-pink-600'
  },
  { 
    name: 'Hyperparameter Optimization', 
    icon: Target, 
    description: 'Optimizing model parameters',
    color: 'from-red-500 to-red-600'
  },
  { 
    name: 'Ensemble Creation', 
    icon: Layers, 
    description: 'Building ensemble models',
    color: 'from-teal-500 to-teal-600'
  },
  { 
    name: 'Model Evaluation', 
    icon: TrendingUp, 
    description: 'Evaluating model performance',
    color: 'from-cyan-500 to-cyan-600'
  },
  { 
    name: 'Code Generation', 
    icon: Package, 
    description: 'Generating production-ready code',
    color: 'from-violet-500 to-violet-600'
  }
];

export default function ProgressIndicator({ agentProgress, currentStage, currentTask }: Props) {
  const getStageIndex = (stageName: string) => {
    return AGENT_STAGES.findIndex(stage => 
      stageName.toLowerCase().includes(stage.name.toLowerCase().split(' ')[0].toLowerCase()) ||
      stage.name.toLowerCase().includes(stageName.toLowerCase().split(' '))
    );
  };

  const currentStageIndex = getStageIndex(currentStage);
  const overallProgress = currentTask?.progress || 0;

  const getAgentStatus = (stage: any, index: number) => {
    const agentInfo = agentProgress.find(agent => 
      agent.agent_name.toLowerCase().includes(stage.name.toLowerCase().split(' ')[0].toLowerCase())
    );

    if (agentInfo) {
      return {
        status: agentInfo.status,
        progress: agentInfo.progress,
        message: agentInfo.message,
        quality_metrics: agentInfo.quality_metrics
      };
    }

    if (index < currentStageIndex) {
      return { status: 'completed', progress: 100, message: 'Completed' };
    } else if (index === currentStageIndex) {
      return { status: 'running', progress: overallProgress, message: 'In progress...' };
    } else {
      return { status: 'pending', progress: 0, message: 'Waiting...' };
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white dark:bg-dark-800 rounded-2xl p-8 shadow-xl border border-gray-200 dark:border-dark-700 max-w-4xl w-full"
    >
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl flex items-center justify-center shadow-lg">
          <Brain className="w-6 h-6 text-white" />
        </div>
        <div className="flex-1">
          <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            ML Pipeline in Progress
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            {currentStage} • {Math.round(overallProgress)}% Complete
          </p>
        </div>
        <div className="text-right">
          <div className="text-3xl font-bold text-primary-600 dark:text-primary-400">
            {Math.round(overallProgress)}%
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Overall Progress
          </div>
        </div>
      </div>

      {/* Progress Steps */}
      <div className="space-y-4 mb-8">
        {AGENT_STAGES.map((stage, index) => {
          const agentStatus = getAgentStatus(stage, index);
          const Icon = stage.icon;

          return (
            <motion.div
              key={stage.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`flex items-center gap-4 p-4 rounded-xl transition-all ${
                agentStatus.status === 'running' 
                  ? 'bg-primary-50 dark:bg-primary-900/20 border-2 border-primary-200 dark:border-primary-800 shadow-md' 
                  : agentStatus.status === 'completed'
                  ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                  : 'bg-gray-50 dark:bg-dark-700 border border-gray-200 dark:border-dark-600'
              }`}
            >
              {/* Icon */}
              <div className={`relative w-12 h-12 rounded-xl flex items-center justify-center ${
                agentStatus.status === 'completed' 
                  ? 'bg-green-500 shadow-lg'
                  : agentStatus.status === 'running' 
                  ? `bg-gradient-to-r ${stage.color} shadow-lg`
                  : 'bg-gray-300 dark:bg-gray-600'
              }`}>
                {agentStatus.status === 'completed' ? (
                  <CheckCircle className="w-6 h-6 text-white" />
                ) : agentStatus.status === 'running' ? (
                  <Icon className="w-6 h-6 text-white" />
                ) : (
                  <Icon className="w-6 h-6 text-gray-500 dark:text-gray-400" />
                )}
                
                {agentStatus.status === 'running' && (
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  </div>
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-1">
                  <h4 className={`font-semibold ${
                    agentStatus.status === 'running' 
                      ? 'text-primary-700 dark:text-primary-300'
                      : agentStatus.status === 'completed'
                      ? 'text-green-700 dark:text-green-300'
                      : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {stage.name}
                  </h4>
                  
                  {agentStatus.status === 'running' && (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-4 h-4 border-2 border-primary-300 border-t-primary-600 rounded-full"
                    />
                  )}
                  
                  {agentStatus.quality_metrics && (
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                      agentStatus.quality_metrics.score >= 0.8 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                      agentStatus.quality_metrics.score >= 0.6 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300' :
                      'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                    }`}>
                      Quality: {Math.round(agentStatus.quality_metrics.score * 100)}%
                    </div>
                  )}
                </div>
                
                <p className={`text-sm mb-2 ${
                  agentStatus.status === 'running' 
                    ? 'text-primary-600 dark:text-primary-400'
                    : agentStatus.status === 'completed'
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-gray-400 dark:text-gray-500'
                }`}>
                  {agentStatus.message || stage.description}
                </p>

                {/* Progress Bar for Running Agent */}
                {agentStatus.status === 'running' && (
                  <div className="w-full bg-white/50 dark:bg-black/20 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${agentStatus.progress}%` }}
                      transition={{ duration: 0.5 }}
                      className="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full"
                    />
                  </div>
                )}

                {/* Quality Issues */}
                {agentStatus.quality_metrics?.issues && agentStatus.quality_metrics.issues.length > 0 && (
                  <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                      <span className="text-sm font-medium text-yellow-700 dark:text-yellow-300">
                        Quality Issues
                      </span>
                    </div>
                    <ul className="text-xs text-yellow-600 dark:text-yellow-400 list-disc list-inside">
                      {agentStatus.quality_metrics.issues.slice(0, 2).map((issue, i) => (
                        <li key={i}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Status Indicator */}
              <div className="text-right">
                {agentStatus.status === 'running' && (
                  <div className="text-lg font-bold text-primary-600 dark:text-primary-400">
                    {Math.round(agentStatus.progress)}%
                  </div>
                )}
                {agentStatus.status === 'completed' && (
                  <div className="text-lg font-bold text-green-600 dark:text-green-400">
                    ✓
                  </div>
                )}
                {agentStatus.status === 'pending' && (
                  <Clock className="w-5 h-5 text-gray-400" />
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Overall Progress Bar */}
      <div className="pt-6 border-t border-gray-200 dark:border-dark-700">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-3">
          <span className="font-medium">Overall Pipeline Progress</span>
          <span className="font-bold">{Math.round(overallProgress)}% Complete</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-3 shadow-inner">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${overallProgress}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="bg-gradient-to-r from-primary-500 via-purple-500 to-pink-500 h-3 rounded-full shadow-lg"
          />
        </div>
        
        {/* Time Estimation */}
        {currentTask.status === 'running' && (
          <div className="mt-3 text-center text-sm text-gray-500 dark:text-gray-400">
            <Clock className="w-4 h-4 inline mr-1" />
            Estimated completion: {Math.round((100 - overallProgress) * 0.5)} minutes remaining
          </div>
        )}
      </div>
    </motion.div>
  );
}
