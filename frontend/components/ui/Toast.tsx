import { forwardRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertCircle, AlertTriangle, Info, X } from 'lucide-react';
import { cn } from '@/utils/cn';

export interface ToastProps {
  id?: string;
  title?: string;
  description?: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
  dismissible?: boolean;
  onDismiss?: () => void;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

const Toast = forwardRef<HTMLDivElement, ToastProps>(({
  id,
  title,
  description,
  type = 'info',
  dismissible = true,
  onDismiss,
  action,
  className,
  ...props
}, ref) => {
  const icons = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info
  };

  const colors = {
    success: {
      bg: 'bg-green-50 dark:bg-green-900/20',
      border: 'border-green-200 dark:border-green-800',
      icon: 'text-green-500 dark:text-green-400',
      title: 'text-green-800 dark:text-green-200',
      description: 'text-green-600 dark:text-green-300'
    },
    error: {
      bg: 'bg-red-50 dark:bg-red-900/20',
      border: 'border-red-200 dark:border-red-800',
      icon: 'text-red-500 dark:text-red-400',
      title: 'text-red-800 dark:text-red-200',
      description: 'text-red-600 dark:text-red-300'
    },
    warning: {
      bg: 'bg-yellow-50 dark:bg-yellow-900/20',
      border: 'border-yellow-200 dark:border-yellow-800',
      icon: 'text-yellow-500 dark:text-yellow-400',
      title: 'text-yellow-800 dark:text-yellow-200',
      description: 'text-yellow-600 dark:text-yellow-300'
    },
    info: {
      bg: 'bg-blue-50 dark:bg-blue-900/20',
      border: 'border-blue-200 dark:border-blue-800',
      icon: 'text-blue-500 dark:text-blue-400',
      title: 'text-blue-800 dark:text-blue-200',
      description: 'text-blue-600 dark:text-blue-300'
    }
  };

  const Icon = icons[type];
  const colorScheme = colors[type];

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: -50, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -50, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "relative rounded-lg border p-4 shadow-lg backdrop-blur-sm",
        colorScheme.bg,
        colorScheme.border,
        className
      )}
      {...props}
    >
      <div className="flex items-start gap-3">
        <Icon className={cn("w-5 h-5 flex-shrink-0 mt-0.5", colorScheme.icon)} />
        
        <div className="flex-1 min-w-0">
          {title && (
            <p className={cn("text-sm font-semibold", colorScheme.title)}>
              {title}
            </p>
          )}
          
          {description && (
            <p className={cn("text-sm mt-1", colorScheme.description)}>
              {description}
            </p>
          )}
          
          {action && (
            <button
              onClick={action.onClick}
              className={cn(
                "text-sm font-medium underline underline-offset-4 mt-2 hover:no-underline transition-all",
                colorScheme.title
              )}
            >
              {action.label}
            </button>
          )}
        </div>
        
        {dismissible && (
          <button
            onClick={onDismiss}
            className={cn(
              "flex-shrink-0 p-1 hover:bg-black/5 dark:hover:bg-white/5 rounded transition-colors",
              colorScheme.icon
            )}
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </motion.div>
  );
});

Toast.displayName = "Toast";

export { Toast };

// Toast container for managing multiple toasts
export const ToastContainer = ({ 
  toasts, 
  onDismiss 
}: { 
  toasts: ToastProps[]; 
  onDismiss: (id: string) => void; 
}) => (
  <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full">
    <AnimatePresence mode="popLayout">
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          {...toast}
          onDismiss={() => onDismiss(toast.id!)}
        />
      ))}
    </AnimatePresence>
  </div>
);
