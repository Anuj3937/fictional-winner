import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Upload, 
  MessageSquare, 
  Sparkles, 
  Database,
  Code,
  BarChart3,
  Zap,
  ArrowRight,
  Lightbulb,
  Target,
  Layers
} from 'lucide-react';

interface Props {
  onGetStarted: () => void;
  onSendMessage: (message: string, files?: File[]) => void;
}

export default function WelcomeScreen({ onGetStarted, onSendMessage }: Props) {
  const [inputValue, setInputValue] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  const examplePrompts = [
    {
      title: "Customer Analytics",
      prompt: "Build a customer churn prediction model using my dataset with advanced feature engineering",
      icon: <Target className="w-5 h-5" />,
      category: "Classification"
    },
    {
      title: "Price Prediction", 
      prompt: "Create a house price regression model with automated hyperparameter optimization",
      icon: <BarChart3 className="w-5 h-5" />,
      category: "Regression"
    },
    {
      title: "Synthetic Data",
      prompt: "Generate realistic healthcare data and train a classification model for disease prediction",
      icon: <Database className="w-5 h-5" />,
      category: "Synthetic"
    },
    {
      title: "Model Comparison",
      prompt: "Compare XGBoost, LightGBM, and Random Forest for fraud detection with ensemble creation",
      icon: <Layers className="w-5 h-5" />,
      category: "Ensemble"
    },
    {
      title: "Financial Analysis",
      prompt: "Build a credit scoring model with statistical feature engineering and quality gates",
      icon: <Sparkles className="w-5 h-5" />,
      category: "Finance"
    }
  ];

  const features = [
    {
      icon: <Brain className="w-6 h-6" />,
      title: "Intelligent Agent Pipeline",
      description: "10+ specialized AI agents work together with quality feedback loops and automatic reiteration",
      highlight: "Quality Loops"
    },
    {
      icon: <Database className="w-6 h-6" />,
      title: "Smart Data Handling",
      description: "Automatic data discovery, web-researched synthetic generation, and intelligent preprocessing",
      highlight: "Web Research"
    },
    {
      icon: <Sparkles className="w-6 h-6" />,
      title: "Advanced Feature Engineering",
      description: "Research-backed feature creation using cutting-edge techniques from latest ML papers",
      highlight: "Research-Backed"
    },
    {
      icon: <Code className="w-6 h-6" />,
      title: "Production-Ready Code",
      description: "Generate complete ML pipelines with FastAPI, Streamlit UI, Docker, and deployment files",
      highlight: "Full Stack"
    },
    {
      icon: <BarChart3 className="w-6 h-6" />,
      title: "Research-Informed Models",
      description: "Uses latest algorithms and hyperparameters discovered through real-time web research",
      highlight: "Live Research"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Real-Time Execution",
      description: "Watch your ML pipeline build in real-time with live logs, progress tracking, and quality metrics",
      highlight: "Live Updates"
    }
  ];

  const handleQuickStart = (prompt: string) => {
    setInputValue(prompt);
    onSendMessage(prompt);
  };

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const csvFiles = files.filter(file => 
      file.name.endsWith('.csv') || 
      file.name.endsWith('.xlsx') || 
      file.name.endsWith('.json')
    );
    
    if (csvFiles.length > 0) {
      const prompt = `Analyze and build a machine learning model using my uploaded dataset: ${csvFiles[0].name}`;
      onSendMessage(prompt, csvFiles);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  return (
    <div 
      className="flex-1 overflow-auto relative"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {/* Drag Overlay */}
      {isDragOver && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute inset-0 bg-primary-500/10 border-2 border-dashed border-primary-500 rounded-lg flex items-center justify-center z-50"
        >
          <div className="text-center">
            <Upload className="w-16 h-16 text-primary-500 mx-auto mb-4" />
            <p className="text-2xl font-bold text-primary-600 dark:text-primary-400 mb-2">
              Drop your dataset here
            </p>
            <p className="text-lg text-gray-600 dark:text-gray-300">
              Supports CSV, JSON, Excel files
            </p>
          </div>
        </motion.div>
      )}

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="flex justify-center mb-8">
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-primary-500 via-purple-500 to-pink-500 rounded-3xl flex items-center justify-center shadow-2xl">
                <Brain className="w-12 h-12 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center animate-pulse">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div className="absolute -bottom-1 -left-1 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                <Zap className="w-3 h-3 text-white" />
              </div>
            </div>
          </div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <h1 className="text-6xl font-bold bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-6">
              Hello, Anuj! 👋
            </h1>
            <p className="text-2xl text-gray-600 dark:text-gray-300 mb-4">
              Welcome back to your <span className="font-semibold text-primary-600">ML Playground</span>
            </p>
            <p className="text-lg text-gray-500 dark:text-gray-400 max-w-4xl mx-auto leading-relaxed">
              Your intelligent ML automation platform powered by <strong>10+ specialized AI agents</strong> with 
              quality feedback loops, real-time web research, and production-ready code generation.
            </p>
          </motion.div>
        </motion.div>

        {/* Features Grid */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="group bg-white dark:bg-dark-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-dark-700 hover:shadow-xl hover:border-primary-200 dark:hover:border-primary-800 transition-all duration-300"
            >
              <div className="flex items-start justify-between mb-6">
                <div className="w-14 h-14 bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/30 dark:to-primary-800/30 rounded-xl flex items-center justify-center text-primary-600 dark:text-primary-400 group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
                <span className="bg-gradient-to-r from-green-500 to-blue-500 text-white px-3 py-1 rounded-full text-xs font-medium">
                  {feature.highlight}
                </span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-gray-100 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                {feature.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Quick Start Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="bg-gradient-to-br from-white to-gray-50 dark:from-dark-800 dark:to-dark-900 rounded-3xl p-10 shadow-xl border border-gray-200 dark:border-dark-700"
        >
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
              🚀 Start Building Your ML Solution
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              Describe your project in natural language and watch our intelligent agents build it for you
            </p>
          </div>
          
          {/* Custom Input */}
          <form onSubmit={handleCustomSubmit} className="mb-10">
            <div className="flex gap-4">
              <div className="flex-1 relative">
                <MessageSquare className="absolute left-4 top-1/2 transform -translate-y-1/2 w-6 h-6 text-gray-400" />
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Describe your ML project in natural language..."
                  className="w-full pl-16 pr-6 py-6 text-lg border-2 border-gray-200 dark:border-dark-600 rounded-2xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-dark-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                />
              </div>
              <button
                type="submit"
                disabled={!inputValue.trim()}
                className="px-10 py-6 bg-gradient-to-r from-primary-600 to-primary-500 text-white rounded-2xl font-semibold hover:from-primary-700 hover:to-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center gap-3 text-lg"
              >
                <span>Build Model</span>
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </form>

          {/* Example Prompts */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <Lightbulb className="w-6 h-6 text-amber-500" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                Try These Examples:
              </h3>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {examplePrompts.map((example, index) => (
                <motion.button
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  onClick={() => handleQuickStart(example.prompt)}
                  className="text-left p-6 rounded-xl bg-white dark:bg-dark-700 hover:bg-primary-50 dark:hover:bg-primary-900/20 border-2 border-gray-200 dark:border-dark-600 hover:border-primary-300 dark:hover:border-primary-700 transition-all group"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/30 dark:to-primary-800/30 rounded-lg flex items-center justify-center text-primary-600 dark:text-primary-400 group-hover:scale-110 transition-transform">
                      {example.icon}
                    </div>
                    <span className="bg-gray-100 dark:bg-dark-600 text-gray-600 dark:text-gray-400 px-2 py-1 rounded-md text-xs font-medium">
                      {example.category}
                    </span>
                  </div>
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                    {example.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {example.prompt}
                  </p>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Upload Section */}
          <div className="mt-10 pt-8 border-t border-gray-200 dark:border-dark-700">
            <div className="flex items-center justify-center">
              <div className="flex items-center text-gray-500 dark:text-gray-400">
                <Upload className="w-5 h-5 mr-3" />
                <span className="text-lg">
                  You can also <strong>drag & drop</strong> your dataset files anywhere on this screen
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
