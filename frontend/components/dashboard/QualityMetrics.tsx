import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  CheckCircle, 
  AlertCircle, 
  AlertTriangle,
  Target,
  TrendingUp,
  Database,
  Layers,
  Activity,
  RefreshCw,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import { MLTask } from '@/types';

interface Props {
  task: MLTask;
}

interface QualityCheck {
  category: string;
  checks: Array<{
    name: string;
    status: 'passed' | 'warning' | 'failed';
    value: string;
    threshold?: string;
    description: string;
  }>;
}

export default function QualityMetrics({ task }: Props) {
  const [selectedCategory, setSelectedCategory] = useState<string>('overview');

  const results = task.results;

  // Generate quality assessments based on results
  const generateQualityChecks = (): QualityCheck[] => {
    const checks: QualityCheck[] = [];

    // Data Quality Checks
    const dataQuality: QualityCheck = {
      category: 'Data Quality',
      checks: []
    };

    if (results?.data_profile) {
      const profile = results.data_profile;
      
      // Missing values check
      const missingRatio = Object.values(profile.missing_percentage || {})
        .reduce((sum: number, val: any) => sum + (val || 0), 0) / 100;
      
      dataQuality.checks.push({
        name: 'Missing Values',
        status: missingRatio < 0.1 ? 'passed' : missingRatio < 0.3 ? 'warning' : 'failed',
        value: `${(missingRatio * 100).toFixed(1)}%`,
        threshold: '< 10%',
        description: 'Percentage of missing values in dataset'
      });

      // Dataset size check
      const sampleCount = profile.shape?.[0] || 0;
      dataQuality.checks.push({
        name: 'Sample Count',
        status: sampleCount >= 1000 ? 'passed' : sampleCount >= 100 ? 'warning' : 'failed',
        value: sampleCount.toLocaleString(),
        threshold: '≥ 1000',
        description: 'Number of samples in dataset'
      });

      // Feature count check
      const featureCount = profile.shape?.[1] || 0;
      dataQuality.checks.push({
        name: 'Feature Count',
        status: featureCount >= 5 ? 'passed' : featureCount >= 2 ? 'warning' : 'failed',
        value: featureCount.toString(),
        threshold: '≥ 5',
        description: 'Number of features in dataset'
      });

      // Duplicates check
      const duplicateRatio = (profile.duplicates?.percentage || 0) / 100;
      dataQuality.checks.push({
        name: 'Duplicate Records',
        status: duplicateRatio < 0.05 ? 'passed' : duplicateRatio < 0.15 ? 'warning' : 'failed',
        value: `${(duplicateRatio * 100).toFixed(1)}%`,
        threshold: '< 5%',
        description: 'Percentage of duplicate records'
      });
    }

    checks.push(dataQuality);

    // Model Quality Checks
    const modelQuality: QualityCheck = {
      category: 'Model Quality',
      checks: []
    };

    if (results?.model_performance) {
      const performance = results.model_performance;
      const bestModel = performance.best_model;

      if (bestModel) {
        // Performance check
        const threshold = task.task_type === 'classification' ? 0.8 : 0.7;
        modelQuality.checks.push({
          name: 'Model Performance',
          status: bestModel.test_score >= threshold ? 'passed' : 
                  bestModel.test_score >= (threshold - 0.1) ? 'warning' : 'failed',
          value: `${(bestModel.test_score * 100).toFixed(1)}%`,
          threshold: `≥ ${(threshold * 100)}%`,
          description: `${task.task_type === 'classification' ? 'Accuracy' : 'R² Score'} on test set`
        });

        // Cross-validation stability
        modelQuality.checks.push({
          name: 'CV Stability',
          status: bestModel.cv_std < 0.05 ? 'passed' : 
                  bestModel.cv_std < 0.1 ? 'warning' : 'failed',
          value: `${(bestModel.cv_std * 100).toFixed(2)}%`,
          threshold: '< 5%',
          description: 'Standard deviation of cross-validation scores'
        });

        // Overfitting check
        const overfitting = Math.abs((bestModel.test_score || 0) - (bestModel.cv_mean || 0));
        modelQuality.checks.push({
          name: 'Overfitting Control',
          status: overfitting < 0.05 ? 'passed' : 
                  overfitting < 0.1 ? 'warning' : 'failed',
          value: `${(overfitting * 100).toFixed(2)}%`,
          threshold: '< 5%',
          description: 'Gap between test and cross-validation scores'
        });

        // Model diversity (if multiple models)
        const modelCount = performance.all_models?.length || 0;
        modelQuality.checks.push({
          name: 'Model Diversity',
          status: modelCount >= 3 ? 'passed' : modelCount >= 2 ? 'warning' : 'failed',
          value: modelCount.toString(),
          threshold: '≥ 3',
          description: 'Number of different algorithms trained'
        });
      }
    }

    checks.push(modelQuality);

    // Pipeline Quality Checks
    const pipelineQuality: QualityCheck = {
      category: 'Pipeline Quality',
      checks: []
    };

    if (results?.execution_summary) {
      const summary = results.execution_summary;

      // Stages completion
      pipelineQuality.checks.push({
        name: 'Pipeline Stages',
        status: summary.stages_completed >= 8 ? 'passed' : 
                summary.stages_completed >= 6 ? 'warning' : 'failed',
        value: summary.stages_completed.toString(),
        threshold: '≥ 8',
        description: 'Number of pipeline stages completed successfully'
      });

      // Feature engineering
      pipelineQuality.checks.push({
        name: 'Feature Engineering',
        status: summary.features_engineered > 0 ? 'passed' : 'failed',
        value: summary.features_engineered.toString(),
        threshold: '> 0',
        description: 'Number of engineered features created'
      });

      // Optimization applied
      pipelineQuality.checks.push({
        name: 'Hyperparameter Optimization',
        status: summary.optimization_applied ? 'passed' : 'warning',
        value: summary.optimization_applied ? 'Applied' : 'Not Applied',
        threshold: 'Applied',
        description: 'Whether hyperparameter optimization was performed'
      });

      // Ensemble creation
      pipelineQuality.checks.push({
        name: 'Ensemble Models',
        status: summary.ensemble_created ? 'passed' : 'warning',
        value: summary.ensemble_created ? 'Created' : 'Not Created',
        threshold: 'Created',
        description: 'Whether ensemble models were created'
      });
    }

    checks.push(pipelineQuality);

    return checks;
  };

  const qualityChecks = generateQualityChecks();

  // Calculate overall quality score
  const calculateOverallScore = () => {
    let totalChecks = 0;
    let passedChecks = 0;

    qualityChecks.forEach(category => {
      category.checks.forEach(check => {
        totalChecks++;
        if (check.status === 'passed') passedChecks++;
        else if (check.status === 'warning') passedChecks += 0.5;
      });
    });

    return totalChecks > 0 ? (passedChecks / totalChecks) * 100 : 0;
  };

  const overallScore = calculateOverallScore();

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'failed': return <AlertCircle className="w-5 h-5 text-red-500" />;
      default: return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed': return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'warning': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'failed': return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Data Quality': return <Database className="w-5 h-5" />;
      case 'Model Quality': return <Target className="w-5 h-5" />;
      case 'Pipeline Quality': return <Activity className="w-5 h-5" />;
      default: return <CheckCircle className="w-5 h-5" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header with Overall Score */}
      <div className="bg-gradient-to-r from-white to-gray-50 dark:from-dark-800 dark:to-dark-900 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
              <Target className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                Quality Assessment
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Comprehensive quality analysis across all pipeline stages
              </p>
            </div>
          </div>

          <div className="text-center">
            <div className={`text-4xl font-bold ${
              overallScore >= 80 ? 'text-green-600 dark:text-green-400' :
              overallScore >= 60 ? 'text-yellow-600 dark:text-yellow-400' :
              'text-red-600 dark:text-red-400'
            }`}>
              {overallScore.toFixed(0)}%
            </div>
            <div className="flex items-center gap-2 mt-2">
              {overallScore >= 80 ? (
                <ThumbsUp className="w-5 h-5 text-green-500" />
              ) : (
                <ThumbsDown className="w-5 h-5 text-red-500" />
              )}
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Overall Quality
              </span>
            </div>
          </div>
        </div>

        {/* Quality Score Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
            <span>Quality Score</span>
            <span>{overallScore.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-3">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${overallScore}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
              className={`h-3 rounded-full ${
                overallScore >= 80 ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                overallScore >= 60 ? 'bg-gradient-to-r from-yellow-500 to-amber-500' :
                'bg-gradient-to-r from-red-500 to-rose-500'
              }`}
            />
          </div>
        </div>
      </div>

      {/* Category Navigation */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSelectedCategory('overview')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'overview'
              ? 'bg-primary-500 text-white'
              : 'bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-dark-600'
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          Overview
        </button>
        {qualityChecks.map((category) => (
          <button
            key={category.category}
            onClick={() => setSelectedCategory(category.category)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedCategory === category.category
                ? 'bg-primary-500 text-white'
                : 'bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-dark-600'
            }`}
          >
            {getCategoryIcon(category.category)}
            {category.category}
          </button>
        ))}
      </div>

      {/* Content */}
      <motion.div
        key={selectedCategory}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {selectedCategory === 'overview' ? (
          <div className="grid gap-6">
            {qualityChecks.map((category, categoryIndex) => (
              <div
                key={category.category}
                className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700"
              >
                <div className="flex items-center gap-3 mb-4">
                  {getCategoryIcon(category.category)}
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    {category.category}
                  </h3>
                  <div className="ml-auto">
                    {(() => {
                      const passed = category.checks.filter(c => c.status === 'passed').length;
                      const total = category.checks.length;
                      const percentage = (passed / total) * 100;
                      return (
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          percentage >= 80 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                          percentage >= 60 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300' :
                          'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                        }`}>
                          {passed}/{total} Passed
                        </span>
                      );
                    })()}
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-4">
                  {category.checks.map((check, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border ${getStatusColor(check.status)}`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{check.name}</span>
                        {getStatusIcon(check.status)}
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-lg font-bold">{check.value}</span>
                        {check.threshold && (
                          <span className="text-sm opacity-75">Target: {check.threshold}</span>
                        )}
                      </div>
                      <p className="text-sm opacity-75">{check.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white dark:bg-dark-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-dark-700">
            {(() => {
              const category = qualityChecks.find(c => c.category === selectedCategory);
              if (!category) return null;

              return (
                <div>
                  <div className="flex items-center gap-3 mb-6">
                    {getCategoryIcon(category.category)}
                                        <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                      {category.category} Details
                    </h3>
                  </div>
                  
                  <div className="space-y-4">
                    {category.checks.map((check, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className={`p-6 rounded-xl border ${getStatusColor(check.status)}`}
                      >
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-3">
                            {getStatusIcon(check.status)}
                            <h4 className="text-lg font-semibold">{check.name}</h4>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold">{check.value}</div>
                            {check.threshold && (
                              <div className="text-sm opacity-75">Target: {check.threshold}</div>
                            )}
                          </div>
                        </div>
                        <p className="opacity-75">{check.description}</p>
                        
                        {/* Status-specific recommendations */}
                        {check.status !== 'passed' && (
                          <div className="mt-4 p-3 bg-white/50 dark:bg-black/20 rounded-lg">
                            <h5 className="font-medium mb-2">Recommendations:</h5>
                            <ul className="text-sm space-y-1">
                              {getRecommendations(check.name, check.status).map((rec, i) => (
                                <li key={i} className="flex items-start gap-2">
                                  <span className="text-primary-500 mt-0.5">•</span>
                                  {rec}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </motion.div>

      {/* Quality Improvement Suggestions */}
      {overallScore < 80 && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6 border border-amber-200 dark:border-amber-800">
          <div className="flex items-center gap-3 mb-4">
            <RefreshCw className="w-6 h-6 text-amber-600 dark:text-amber-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Quality Improvement Suggestions
            </h3>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Immediate Actions</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                {overallScore < 60 && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-red-500 mt-0.5">•</span>
                      Review data quality and consider data cleaning
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-500 mt-0.5">•</span>
                      Increase dataset size if possible
                    </li>
                  </>
                )}
                {overallScore < 80 && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500 mt-0.5">•</span>
                      Apply advanced feature engineering techniques
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500 mt-0.5">•</span>
                      Run hyperparameter optimization
                    </li>
                  </>
                )}
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Long-term Improvements</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500 mt-0.5">•</span>
                  Collect more diverse training data
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500 mt-0.5">•</span>
                  Implement model ensemble strategies
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500 mt-0.5">•</span>
                  Consider advanced algorithms or neural networks
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  function getRecommendations(checkName: string, status: string): string[] {
    const recommendations: Record<string, string[]> = {
      'Missing Values': [
        'Apply advanced imputation techniques (KNN, iterative)',
        'Consider removing features with >50% missing values',
        'Collect more complete data if possible'
      ],
      'Sample Count': [
        'Collect additional training samples',
        'Use data augmentation techniques',
        'Consider synthetic data generation'
      ],
      'Feature Count': [
        'Apply feature engineering to create new features',
        'Use domain knowledge to derive meaningful features',
        'Consider feature extraction techniques'
      ],
      'Model Performance': [
        'Try different algorithms (ensemble methods)',
        'Apply hyperparameter optimization',
        'Improve feature engineering',
        'Collect more training data'
      ],
      'CV Stability': [
        'Apply regularization techniques',
        'Increase cross-validation folds',
        'Use stratified sampling',
        'Consider ensemble methods'
      ],
      'Overfitting Control': [
        'Apply regularization (L1/L2)',
        'Use early stopping',
        'Increase training data',
        'Reduce model complexity'
      ]
    };

    return recommendations[checkName] || [
      'Review the specific metric requirements',
      'Consider consulting ML best practices',
      'Iteratively improve based on results'
    ];
  }
}

