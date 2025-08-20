import { HTMLAttributes, forwardRef } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Brain, Zap } from 'lucide-react';
import { cn } from '@/utils/cn';

interface LoadingSpinnerProps extends HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'default' | 'lg' | 'xl';
  variant?: 'default' | 'dots' | 'pulse' | 'brain' | 'custom';
  text?: string;
  fullScreen?: boolean;
}

const LoadingSpinner = forwardRef<HTMLDivElement, LoadingSpinnerProps>(
  ({ 
    className, 
    size = 'default', 
    variant = 'default', 
    text,
    fullScreen = false,
    ...props 
  }, ref) => {
    const sizeClasses = {
      sm: 'w-4 h-4',
      default: 'w-6 h-6',
      lg: 'w-8 h-8',
      xl: 'w-12 h-12'
    };

    const containerClasses = cn(
      "flex flex-col items-center justify-center gap-3",
      fullScreen && "min-h-screen",
      className
    );

    const spinnerClasses = cn(
      "animate-spin text-primary-500",
      sizeClasses[size]
    );

    const renderSpinner = () => {
      switch (variant) {
        case 'dots':
          return (
            <div className="flex space-x-1">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className={cn("rounded-full bg-primary-500", {
                    'w-2 h-2': size === 'sm',
                    'w-3 h-3': size === 'default',
                    'w-4 h-4': size === 'lg',
                    'w-5 h-5': size === 'xl'
                  })}
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.7, 1, 0.7]
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    delay: i * 0.2
                  }}
                />
              ))}
            </div>
          );
        
        case 'pulse':
          return (
            <motion.div
              className={cn(
                "rounded-full bg-primary-500",
                sizeClasses[size]
              )}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.7, 1, 0.7]
              }}
              transition={{
                duration: 2,
                repeat: Infinity
              }}
            />
          );
        
        case 'brain':
          return (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className={cn("text-primary-500", sizeClasses[size])}
            >
              <Brain />
            </motion.div>
          );
        
        case 'custom':
          return (
            <div className="relative">
              <motion.div
                className={cn("absolute inset-0 rounded-full border-2 border-primary-200", sizeClasses[size])}
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              <motion.div
                className={cn(
                  "rounded-full border-2 border-transparent border-t-primary-500 border-r-primary-500",
                  sizeClasses[size]
                )}
                animate={{ rotate: 360 }}
                transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
              />
            </div>
          );
        
        default:
          return <Loader2 className={spinnerClasses} />;
      }
    };

    return (
      <div ref={ref} className={containerClasses} {...props}>
        {renderSpinner()}
        
        {text && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-sm font-medium text-gray-600 dark:text-gray-400 text-center max-w-xs"
          >
            {text}
          </motion.p>
        )}

        {variant === 'brain' && !text && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-sm font-medium text-gray-600 dark:text-gray-400 text-center"
          >
            AI is thinking...
          </motion.p>
        )}
      </div>
    );
  }
);

LoadingSpinner.displayName = "LoadingSpinner";

export { LoadingSpinner };

// Additional loading components
export const FullScreenLoader = ({ text = "Loading..." }: { text?: string }) => (
  <div className="fixed inset-0 bg-white/80 dark:bg-dark-900/80 backdrop-blur-sm z-50 flex items-center justify-center">
    <LoadingSpinner variant="brain" size="xl" text={text} />
  </div>
);

export const InlineLoader = ({ className, ...props }: LoadingSpinnerProps) => (
  <LoadingSpinner 
    className={cn("inline-flex", className)} 
    size="sm" 
    {...props} 
  />
);

export const PageLoader = ({ text = "Loading page..." }: { text?: string }) => (
  <div className="min-h-[400px] flex items-center justify-center">
    <LoadingSpinner variant="custom" size="lg" text={text} />
  </div>
);
