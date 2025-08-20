import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  Download,
  Eye,
  Code,
  BarChart3,
  Target,
  Zap,
  Award,
  Clock,
  Database,
  Layers,
  Activity
} from 'lucide-react';
import ModelComparison from './ModelComparison';
import PerformanceCharts from './PerformanceCharts';
import QualityMetrics from './QualityMetrics';
import { MLTask } from '@/types';

interface Props {
  task: MLTask;
  onDownloadModel: () => void;
  onViewCode: () => void;
}

export default function ResultsDashboard({ task, onDownloadModel, onViewCode }: Props) {
  const [activeSection, setActiveSection] = useState<'overview' | 'models' | 'performance' | 'quality'>('overview');

  if (!task.results) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Activity className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            No Results Available
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Complete the ML pipeline to view results
          </p>
        </div>
      </div>
    );
  }

  const results = task.results;
  const modelPerformance = results.model_performance;
  const bestModel = modelPerformance?.best_model;

  return (
    <div className="h-full bg-gray-50 dark:bg-dark-900 overflow-auto">
      {/* Header */}
      <div className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
              <Award className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                ML Pipeline Results
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Task completed successfully • {new Date(task.completed_at || '').toLocaleString()}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={onViewCode}
              className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
            >
              <Code className="w-4 h-4" />
              View Code
            </button>
            <button
              onClick={onDownloadModel}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
            >
              <Download className="w-4 h-4" />
              Download Model
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 px-6">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: Eye },
            { id: 'models', label: 'Models', icon: Layers },
            { id: 'performance', label: 'Performance', icon: BarChart3 },
            { id: 'quality', label: 'Quality', icon: Target }
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSection(tab.id as any)}
                className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeSection === tab.id
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeSection === 'overview' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                    <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Best Score
                  </span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {(bestModel?.test_score * 100).toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {bestModel?.type} Model
                </p>
              </div>

              <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                    <Layers className="w-5 h-5 text-green-600 dark:text-green-400" />
                  </div>
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Models Trained
                  </span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {modelPerformance?.all_models?.length || 0}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Multiple Algorithms
                </p>
              </div>

              <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                    <Clock className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                  </div>
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Execution Time
                  </span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {results.execution_summary?.total_execution_time ? 
                    `${(results.execution_summary.total_execution_time / 60).toFixed(1)}m` : 
                    'N/A'
                  }
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Total Pipeline
                </p>
              </div>

              <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                    <Database className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                  </div>
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Dataset Size
                  </span>
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {results.execution_summary?.final_dataset_shape?.[0] || 0}
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Samples Processed
                </p>
              </div>
            </div>

            {/* Best Model Details */}
            {bestModel && (
              <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                  <Award className="w-5 h-5 text-yellow-500" />
                  Best Performing Model
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Model Information</h4>
                    <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                      <div className="flex justify-between">
                        <span>Algorithm:</span>
                        <span className="font-medium">{bestModel.type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Task Type:</span>
                        <span className="font-medium capitalize">{task.task_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Test Score:</span>
                        <span className="font-medium text-green-600 dark:text-green-400">
                          {(bestModel.test_score * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Cross Validation</h4>
                    <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                      <div className="flex justify-between">
                        <span>CV Mean:</span>
                        <span className="font-medium">{(bestModel.cv_mean * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>CV Std:</span>
                        <span className="font-medium">{(bestModel.cv_std * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Stability:</span>
                        <span className={`font-medium ${bestModel.cv_std < 0.05 ? 'text-green-600 dark:text-green-400' : 'text-yellow-600 dark:text-yellow-400'}`}>
                          {bestModel.cv_std < 0.05 ? 'High' : 'Medium'}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Feature Analysis</h4>
                    <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                      <div className="flex justify-between">
                        <span>Top Feature:</span>
                        <span className="font-medium">
                          {bestModel.feature_importance ? 
                            Object.keys(bestModel.feature_importance)[0] : 
                            'N/A'
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Importance:</span>
                        <span className="font-medium">
                          {bestModel.feature_importance ? 
                            (Object.values(bestModel.feature_importance)[0] * 100).toFixed(1) + '%' : 
                            'N/A'
                          }
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Execution Summary */}
            {results.execution_summary && (
              <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-blue-500" />
                  Pipeline Execution Summary
                </h3>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                      {results.execution_summary.stages_completed}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Stages</p>
                  </div>
                  
                  <div className="text-center p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                      {results.execution_summary.features_engineered}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Features</p>
                  </div>
                  
                  <div className="text-center p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                      {results.execution_summary.models_trained}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Models</p>
                  </div>
                  
                  <div className="text-center p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {results.execution_summary.optimization_applied ? '✓' : '✗'}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Optimized</p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {activeSection === 'models' && modelPerformance && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <ModelComparison modelPerformance={modelPerformance} />
          </motion.div>
        )}

        {activeSection === 'performance' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <PerformanceCharts task={task} />
          </motion.div>
        )}

        {activeSection === 'quality' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <QualityMetrics task={task} />
          </motion.div>
        )}
      </div>
    </div>
  );
}
