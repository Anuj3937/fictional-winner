import { motion } from 'framer-motion';
import { 
  Home, 
  MessageSquare, 
  Code, 
  BarChart3, 
  X,
  Sparkles,
  History,
  Settings,
  HelpCircle,
  FileText,
  Activity
} from 'lucide-react';
import { MLTask, UIState } from '@/types';

interface Props {
  activeView: UIState['activeView'];
  setActiveView: (view: UIState['activeView']) => void;
  currentTask: MLTask | null;
  onClose: () => void;
}

export default function Sidebar({ activeView, setActiveView, currentTask, onClose }: Props) {
  const navigationItems = [
    {
      id: 'welcome' as const,
      label: 'Welcome',
      icon: Home,
      description: 'Get started with ML automation'
    },
    {
      id: 'chat' as const,
      label: 'Chat & Pipeline',
      icon: MessageSquare,
      description: 'Build ML solutions with AI chat'
    },
    {
      id: 'code' as const,
      label: 'Code Editor',
      icon: Code,
      description: 'View and edit generated code',
      disabled: !currentTask?.results?.code_generation_result
    },
    {
      id: 'results' as const,
      label: 'Results Dashboard',
      icon: BarChart3,
      description: 'Analyze model performance',
      disabled: !currentTask?.results
    }
  ];

  const quickActions = [
    {
      label: 'Recent Projects',
      icon: History,
      action: () => console.log('Recent projects')
    },
    {
      label: 'Templates',
      icon: FileText,
      action: () => console.log('Templates')
    },
    {
      label: 'Documentation',
      icon: HelpCircle,
      action: () => window.open('https://docs.example.com', '_blank')
    },
    {
      label: 'Settings',
      icon: Settings,
      action: () => console.log('Settings')
    }
  ];

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 z-40 lg:hidden"
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <motion.div
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        exit={{ x: -300 }}
        transition={{ type: "spring", damping: 25, stiffness: 200 }}
        className="fixed left-0 top-0 h-full w-80 bg-white dark:bg-dark-800 border-r border-gray-200 dark:border-dark-700 z-50 flex flex-col shadow-xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-dark-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-primary-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                ML Playground
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                AI-Powered Automation
              </p>
            </div>
          </div>
          
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Current Task Status */}
        {currentTask && (
          <div className="p-6 border-b border-gray-200 dark:border-dark-700">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-3 mb-3">
                <div className={`w-3 h-3 rounded-full ${
                  currentTask.status === 'running' ? 'bg-blue-500 animate-pulse' :
                  currentTask.status === 'completed' ? 'bg-green-500' :
                  currentTask.status === 'failed' ? 'bg-red-500' : 'bg-gray-400'
                }`} />
                <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Current Task
                </span>
              </div>
              
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                {currentTask.user_prompt.length > 100 
                  ? currentTask.user_prompt.substring(0, 100) + '...'
                  : currentTask.user_prompt
                }
              </p>
              
              <div className="flex items-center justify-between">
                <span className="text-xs text-blue-600 dark:text-blue-400 font-medium capitalize">
                  {currentTask.status}
                </span>
                {currentTask.progress > 0 && (
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    {Math.round(currentTask.progress)}%
                  </span>
                )}
              </div>
              
              {/* Progress Bar */}
              {currentTask.status === 'running' && currentTask.progress > 0 && (
                <div className="mt-2 w-full bg-white/50 dark:bg-black/20 rounded-full h-1.5">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${currentTask.progress}%` }}
                    transition={{ duration: 0.5 }}
                    className="bg-blue-500 h-1.5 rounded-full"
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex-1 p-6">
          <nav className="space-y-2">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
              Navigation
            </h3>
            
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.id;
              const isDisabled = item.disabled;
              
              return (
                <button
                  key={item.id}
                  onClick={() => !isDisabled && setActiveView(item.id)}
                  disabled={isDisabled}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all group ${
                    isActive
                      ? 'bg-primary-500 text-white shadow-lg'
                      : isDisabled
                      ? 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-700'
                  }`}
                >
                  <Icon className={`w-5 h-5 ${
                    isActive ? 'text-white' : isDisabled ? '' : 'group-hover:scale-110 transition-transform'
                  }`} />
                  
                  <div className="flex-1 text-left">
                    <div className={`font-medium ${isActive ? 'text-white' : ''}`}>
                      {item.label}
                    </div>
                    <div className={`text-xs ${
                      isActive ? 'text-white/80' : 'text-gray-500 dark:text-gray-400'
                    }`}>
                      {item.description}
                    </div>
                  </div>
                  
                  {isDisabled && (
                    <div className="w-2 h-2 bg-gray-400 rounded-full" />
                  )}
                  
                  {item.id === 'chat' && currentTask?.status === 'running' && (
                    <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Quick Actions */}
          <div className="mt-8">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
              Quick Actions
            </h3>
            
            <div className="space-y-1">
              {quickActions.map((action, index) => {
                const Icon = action.icon;
                
                return (
                  <button
                    key={index}
                    onClick={action.action}
                    className="w-full flex items-center gap-3 px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors group"
                  >
                    <Icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                    <span className="text-sm font-medium">{action.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 dark:border-dark-700">
          <div className="text-center">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              ML Automation Platform
            </div>
            <div className="text-xs text-gray-400 dark:text-gray-500">
              Version 2.0.0 • Built with AI
            </div>
          </div>
        </div>
      </motion.div>
    </>
  );
}
