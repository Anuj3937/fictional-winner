import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Menu, 
  Sun, 
  Moon, 
  Bell, 
  Settings, 
  User,
  Wifi,
  WifiOff,
  Activity,
  Zap
} from 'lucide-react';
import { MLTask } from '@/types';

interface Props {
  onToggleSidebar: () => void;
  onToggleDarkMode: () => void;
  darkMode: boolean;
  isConnected: boolean;
  currentTask: MLTask | null;
}

export default function Header({ 
  onToggleSidebar, 
  onToggleDarkMode, 
  darkMode, 
  isConnected, 
  currentTask 
}: Props) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);

  const getTaskStatusColor = () => {
    if (!currentTask) return 'text-gray-400';
    
    switch (currentTask.status) {
      case 'running': return 'text-blue-500 animate-pulse';
      case 'completed': return 'text-green-500';
      case 'failed': return 'text-red-500';
      case 'cancelled': return 'text-gray-500';
      default: return 'text-gray-400';
    }
  };

  return (
    <header className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left section */}
        <div className="flex items-center gap-4">
          <button
            onClick={onToggleSidebar}
            className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-purple-500 rounded-lg flex items-center justify-center">
              <Zap className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                ML Playground
              </h1>
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-gray-100 dark:bg-dark-700">
            {isConnected ? (
              <Wifi className="w-4 h-4 text-green-500" />
            ) : (
              <WifiOff className="w-4 h-4 text-red-500" />
            )}
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Center section - Task Status */}
        {currentTask && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-3 px-4 py-2 bg-gray-50 dark:bg-dark-700 rounded-full"
          >
            <Activity className={`w-4 h-4 ${getTaskStatusColor()}`} />
            <div className="text-sm">
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {currentTask.current_stage || 'Processing...'}
              </span>
              {currentTask.progress > 0 && (
                <span className="ml-2 text-gray-600 dark:text-gray-400">
                  {Math.round(currentTask.progress)}%
                </span>
              )}
            </div>
            
            {/* Mini Progress Bar */}
            {currentTask.status === 'running' && currentTask.progress > 0 && (
              <div className="w-20 bg-gray-200 dark:bg-dark-600 rounded-full h-1.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${currentTask.progress}%` }}
                  transition={{ duration: 0.5 }}
                  className="bg-primary-500 h-1.5 rounded-full"
                />
              </div>
            )}
          </motion.div>
        )}

        {/* Right section */}
        <div className="flex items-center gap-2">
          {/* Theme Toggle */}
          <button
            onClick={onToggleDarkMode}
            className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
            title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {darkMode ? (
              <Sun className="w-5 h-5 text-yellow-500" />
            ) : (
              <Moon className="w-5 h-5 text-gray-600" />
            )}
          </button>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors relative"
            >
              <Bell className="w-5 h-5" />
              {currentTask?.status === 'running' && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
              )}
            </button>

            {/* Notifications Dropdown */}
            {showNotifications && (
              <>
                <div 
                  className="fixed inset-0 z-40"
                  onClick={() => setShowNotifications(false)}
                />
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: -10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: -10 }}
                  className="absolute right-0 top-full mt-2 w-80 bg-white dark:bg-dark-800 rounded-xl shadow-lg border border-gray-200 dark:border-dark-700 z-50"
                >
                  <div className="p-4 border-b border-gray-200 dark:border-dark-700">
                    <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                      Notifications
                    </h3>
                  </div>
                  <div className="p-4 space-y-3 max-h-60 overflow-auto">
                    {currentTask ? (
                      <div className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <Activity className={`w-4 h-4 mt-0.5 ${getTaskStatusColor()}`} />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            ML Pipeline {currentTask.status === 'running' ? 'Running' : 'Status Update'}
                          </p>
                          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                            {currentTask.current_stage || 'Processing your request...'}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                            {new Date(currentTask.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                        <Bell className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No notifications</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              </>
            )}
          </div>

          {/* Settings */}
          <button
            className="p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 p-2 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
            >
              <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-white" />
              </div>
            </button>

            {/* User Dropdown */}
            {showUserMenu && (
              <>
                <div 
                  className="fixed inset-0 z-40"
                  onClick={() => setShowUserMenu(false)}
                />
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: -10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: -10 }}
                  className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-dark-800 rounded-xl shadow-lg border border-gray-200 dark:border-dark-700 z-50"
                >
                  <div className="p-4 border-b border-gray-200 dark:border-dark-700">
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      Anuj Kumar
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      ML Engineer
                    </p>
                  </div>
                  <div className="p-2">
                    <button className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors">
                      Profile Settings
                    </button>
                    <button className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors">
                      API Keys
                    </button>
                    <button className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors">
                      Usage Statistics
                    </button>
                    <div className="border-t border-gray-200 dark:border-dark-700 my-2" />
                    <button className="w-full text-left px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors">
                      Sign Out
                    </button>
                  </div>
                </motion.div>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
