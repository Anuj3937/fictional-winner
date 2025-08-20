import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  Award, 
  Target,
  Zap,
  Activity,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { ModelPerformance } from '@/types';

interface Props {
  modelPerformance: ModelPerformance;
}

export default function ModelComparison({ modelPerformance }: Props) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'score' | 'name'>('score');

  const sortedModels = [...modelPerformance.all_models].sort((a, b) => {
    if (sortBy === 'score') {
      return b.test_score - a.test_score;
    }
    return a.model_type.localeCompare(b.model_type);
  });

  const getModelColor = (modelType: string) => {
    const colors = {
      xgboost: 'from-green-500 to-emerald-500',
      lightgbm: 'from-blue-500 to-cyan-500',
      random_forest: 'from-purple-500 to-violet-500',
      catboost: 'from-orange-500 to-amber-500',
      neural_network: 'from-pink-500 to-rose-500'
    };
    return colors[modelType as keyof typeof colors] || 'from-gray-500 to-gray-600';
  };

  const getPerformanceLevel = (score: number) => {
    if (score >= 0.9) return { level: 'Excellent', color: 'text-green-600 dark:text-green-400' };
    if (score >= 0.8) return { level: 'Good', color: 'text-blue-600 dark:text-blue-400' };
    if (score >= 0.7) return { level: 'Fair', color: 'text-yellow-600 dark:text-yellow-400' };
    return { level: 'Poor', color: 'text-red-600 dark:text-red-400' };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Model Comparison
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Performance analysis across {sortedModels.length} trained models
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'score' | 'name')}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-dark-700 text-gray-900 dark:text-gray-100"
          >
            <option value="score">Sort by Score</option>
            <option value="name">Sort by Name</option>
          </select>
        </div>
      </div>

      {/* Best Model Highlight */}
      {modelPerformance.best_model && (
        <div className="bg-gradient-to-r from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 rounded-xl p-6 border-2 border-yellow-200 dark:border-yellow-800">
          <div className="flex items-center gap-3 mb-4">
            <Award className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Champion Model: {modelPerformance.best_model.type.toUpperCase()}
            </h3>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
                {(modelPerformance.best_model.test_score * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Test Score</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
                {(modelPerformance.best_model.cv_mean * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">CV Mean</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-700 dark:text-yellow-300">
                {(modelPerformance.best_model.cv_std * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">CV Std</p>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-yellow-700 dark:text-yellow-300">
                {getPerformanceLevel(modelPerformance.best_model.test_score).level}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Performance</p>
            </div>
          </div>
        </div>
      )}

      {/* Models Grid */}
      <div className="grid gap-4">
        {sortedModels.map((model, index) => {
          const isExpanded = expandedModel === model.model_type;
          const isBest = model.model_type === modelPerformance.best_model?.type;
          const performanceLevel = getPerformanceLevel(model.test_score);

          return (
            <motion.div
              key={model.model_type}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`bg-white dark:bg-dark-800 rounded-xl shadow-sm border transition-all ${
                isBest 
                  ? 'border-yellow-300 dark:border-yellow-700 shadow-lg' 
                  : 'border-gray-200 dark:border-dark-700 hover:shadow-md'
              }`}
            >
              <div className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 bg-gradient-to-r ${getModelColor(model.model_type)} rounded-xl flex items-center justify-center shadow-lg`}>
                      <Activity className="w-6 h-6 text-white" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                          {model.model_type.toUpperCase()}
                        </h3>
                        {isBest && (
                          <Award className="w-5 h-5 text-yellow-500" />
                        )}
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                        {model.task_type} Model
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <div className={`text-2xl font-bold ${performanceLevel.color}`}>
                        {(model.test_score * 100).toFixed(1)}%
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {performanceLevel.level}
                      </p>
                    </div>

                    <button
                      onClick={() => setExpandedModel(isExpanded ? null : model.model_type)}
                      className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-5 h-5" />
                      ) : (
                        <ChevronDown className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Quick Metrics */}
                <div className="grid grid-cols-3 gap-4 mt-4">
                  <div className="text-center p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {(model.train_score * 100).toFixed(1)}%
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Train Score</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {(model.cv_mean * 100).toFixed(1)}%
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">CV Mean</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      ±{(model.cv_std * 100).toFixed(1)}%
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">CV Std</p>
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-6 pt-6 border-t border-gray-200 dark:border-dark-600"
                  >
                    <div className="grid md:grid-cols-2 gap-6">
                      {/* Additional Metrics */}
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                          Additional Metrics
                        </h4>
                        <div className="space-y-2">
                          {Object.entries(model.additional_metrics).map(([metric, value]) => (
                            <div key={metric} className="flex justify-between">
                              <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                                {metric.replace('_', ' ')}:
                              </span>
                              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                                {typeof value === 'number' ? value.toFixed(3) : value}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Top Features */}
                      {model.feature_importance && Object.keys(model.feature_importance).length > 0 && (
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                            Top Features
                          </h4>
                          <div className="space-y-2">
                            {Object.entries(model.feature_importance)
                              .slice(0, 5)
                              .map(([feature, importance]) => (
                                <div key={feature} className="flex items-center justify-between">
                                  <span className="text-sm text-gray-600 dark:text-gray-400 truncate">
                                    {feature}
                                  </span>
                                  <div className="flex items-center gap-2">
                                    <div className="w-16 bg-gray-200 dark:bg-dark-600 rounded-full h-2">
                                      <div
                                        className="bg-primary-500 h-2 rounded-full"
                                        style={{ width: `${importance * 100}%` }}
                                      />
                                    </div>
                                    <span className="text-xs font-medium text-gray-900 dark:text-gray-100 w-12 text-right">
                                      {(importance * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Ensemble Information */}
      {modelPerformance.ensemble_performance && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Ensemble Performance
            </h3>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                {(modelPerformance.ensemble_performance.score * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Ensemble Score</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                +{(modelPerformance.ensemble_performance.improvement * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Improvement</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
