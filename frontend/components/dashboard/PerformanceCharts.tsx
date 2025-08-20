import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer,
  ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { TrendingUp, BarChart3, PieChart as PieIcon, Activity, Zap } from 'lucide-react';
import { MLTask } from '@/types';

interface Props {
  task: MLTask;
}

export default function PerformanceCharts({ task }: Props) {
  const [activeChart, setActiveChart] = useState<'comparison' | 'cv' | 'features' | 'metrics'>('comparison');

  const results = task.results;
  const modelPerformance = results?.model_performance;

  if (!modelPerformance) {
    return (
      <div className="text-center py-12">
        <Activity className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600 dark:text-gray-400">No performance data available</p>
      </div>
    );
  }

  // Prepare data for charts
  const modelComparisonData = modelPerformance.all_models.map(model => ({
    name: model.model_type.toUpperCase(),
    testScore: model.test_score * 100,
    trainScore: model.train_score * 100,
    cvMean: model.cv_mean * 100,
    cvStd: model.cv_std * 100,
    overfitting: (model.train_score - model.test_score) * 100
  }));

  // Cross-validation data
  const cvData = modelPerformance.all_models.map(model => ({
    model: model.model_type.toUpperCase(),
    mean: model.cv_mean * 100,
    std: model.cv_std * 100,
    min: (model.cv_mean - model.cv_std) * 100,
    max: (model.cv_mean + model.cv_std) * 100
  }));

  // Feature importance data (from best model)
  const featureData = modelPerformance.best_model?.feature_importance 
    ? Object.entries(modelPerformance.best_model.feature_importance)
        .slice(0, 10)
        .map(([feature, importance]) => ({
          feature: feature.length > 15 ? feature.substring(0, 15) + '...' : feature,
          importance: importance * 100,
          fullFeature: feature
        }))
    : [];

  // Metrics comparison for classification/regression
  const metricsData = modelPerformance.all_models.map(model => {
    const baseMetrics = {
      name: model.model_type.toUpperCase(),
      testScore: model.test_score * 100
    };

    // Add task-specific metrics
    if (task.task_type === 'classification') {
      return {
        ...baseMetrics,
        precision: (model.additional_metrics.precision || 0) * 100,
        recall: (model.additional_metrics.recall || 0) * 100,
        f1Score: (model.additional_metrics.f1_score || 0) * 100
      };
    } else {
      return {
        ...baseMetrics,
        r2Score: (model.additional_metrics.r2_score || 0) * 100,
        mse: model.additional_metrics.mse || 0,
        mae: model.additional_metrics.mae || 0
      };
    }
  });

  const COLORS = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#EC4899'];

  const chartTabs = [
    { id: 'comparison', label: 'Model Comparison', icon: BarChart3 },
    { id: 'cv', label: 'Cross Validation', icon: TrendingUp },
    { id: 'features', label: 'Feature Importance', icon: PieIcon },
    { id: 'metrics', label: 'Detailed Metrics', icon: Activity }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
          <TrendingUp className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Performance Analytics
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Detailed performance analysis and visualization
          </p>
        </div>
      </div>

      {/* Chart Navigation */}
      <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
        <div className="flex flex-wrap gap-2 mb-6">
          {chartTabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveChart(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeChart === tab.id
                    ? 'bg-primary-500 text-white'
                    : 'bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-dark-600'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Chart Content */}
        <motion.div
          key={activeChart}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="h-96"
        >
          {activeChart === 'comparison' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Model Performance Comparison
              </h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelComparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                  />
                  <YAxis 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                    label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Legend />
                  <Bar dataKey="testScore" fill="#3B82F6" name="Test Score" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="trainScore" fill="#10B981" name="Train Score" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="cvMean" fill="#8B5CF6" name="CV Mean" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeChart === 'cv' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Cross-Validation Stability
              </h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={cvData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="model" 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                  />
                  <YAxis 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                    label={{ value: 'CV Score (%)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="mean" 
                    stroke="#3B82F6" 
                    strokeWidth={3}
                    dot={{ r: 6 }}
                    name="CV Mean"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="max" 
                    stroke="#10B981" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="CV Max"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="min" 
                    stroke="#EF4444" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="CV Min"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeChart === 'features' && featureData.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Feature Importance (Top 10)
              </h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    type="number" 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280' }}
                    label={{ value: 'Importance (%)', position: 'bottom' }}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="feature" 
                    stroke="#6B7280"
                    tick={{ fill: '#6B7280', fontSize: 12 }}
                    width={100}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                    labelFormatter={(label, payload) => payload?.[0]?.payload?.fullFeature || label}
                  />
                  <Bar dataKey="importance" fill="#8B5CF6" radius={[0, 2, 2, 0]}>
                    {featureData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeChart === 'metrics' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Detailed Performance Metrics
              </h3>
              {task.task_type === 'classification' ? (
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={metricsData}>
                    <PolarGrid stroke="#374151" />
                    <PolarAngleAxis tick={{ fill: '#6B7280', fontSize: 12 }} />
                    <PolarRadiusAxis 
                      tick={{ fill: '#6B7280', fontSize: 10 }}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F9FAFB'
                      }}
                    />
                    <Legend />
                    {metricsData.map((model, index) => (
                      <Radar
                        key={model.name}
                        name={model.name}
                        dataKey={model.name}
                        stroke={COLORS[index % COLORS.length]}
                        fill={COLORS[index % COLORS.length]}
                        fillOpacity={0.1}
                        strokeWidth={2}
                      />
                    ))}
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      type="number" 
                      dataKey="mse"
                      stroke="#6B7280"
                      tick={{ fill: '#6B7280' }}
                      label={{ value: 'MSE', position: 'bottom' }}
                    />
                    <YAxis 
                      type="number" 
                      dataKey="mae"
                      stroke="#6B7280"
                      tick={{ fill: '#6B7280' }}
                      label={{ value: 'MAE', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F9FAFB'
                      }}
                      cursor={{ strokeDasharray: '3 3' }}
                    />
                    <Legend />
                    <Scatter 
                      data={metricsData} 
                      fill="#3B82F6"
                      name="Models"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </div>
          )}
        </motion.div>
      </div>

      {/* Performance Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Best Performance</h3>
          </div>
          <div className="space-y-2">
            <div className="text-2xl font-bold text-green-700 dark:text-green-300">
              {(modelPerformance.best_model.test_score * 100).toFixed(1)}%
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {modelPerformance.best_model.type.toUpperCase()} Model
            </p>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Average Score</h3>
          </div>
          <div className="space-y-2">
            <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
              {(modelPerformance.all_models.reduce((sum, model) => sum + model.test_score, 0) / modelPerformance.all_models.length * 100).toFixed(1)}%
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Across {modelPerformance.all_models.length} Models
            </p>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Model Stability</h3>
          </div>
          <div className="space-y-2">
            <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
              {modelPerformance.best_model.cv_std < 0.05 ? 'High' : 
               modelPerformance.best_model.cv_std < 0.1 ? 'Medium' : 'Low'}
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              CV Std: {(modelPerformance.best_model.cv_std * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
