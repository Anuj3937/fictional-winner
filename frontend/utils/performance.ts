interface PerformanceUtils {
  measureExecutionTime: <T>(fn: () => T, label?: string) => T;
  measureAsyncExecutionTime: <T>(fn: () => Promise<T>, label?: string) => Promise<T>;
  debounce: <T extends (...args: any[]) => any>(fn: T, delay: number) => T;
  throttle: <T extends (...args: any[]) => any>(fn: T, limit: number) => T;
  memoize: <T extends (...args: any[]) => any>(fn: T) => T;
  getMemoryUsage: () => MemoryInfo | null;
  formatBytes: (bytes: number) => string;
}

class PerformanceUtilities implements PerformanceUtils {
  measureExecutionTime<T>(fn: () => T, label: string = 'Operation'): T {
    const startTime = performance.now();
    const result = fn();
    const endTime = performance.now();
    
    console.log(`${label} took ${(endTime - startTime).toFixed(2)}ms`);
    return result;
  }

  async measureAsyncExecutionTime<T>(
    fn: () => Promise<T>, 
    label: string = 'Async Operation'
  ): Promise<T> {
    const startTime = performance.now();
    const result = await fn();
    const endTime = performance.now();
    
    console.log(`${label} took ${(endTime - startTime).toFixed(2)}ms`);
    return result;
  }

  debounce<T extends (...args: any[]) => any>(fn: T, delay: number): T {
    let timeoutId: NodeJS.Timeout;
    
    return ((...args: Parameters<T>) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => fn(...args), delay);
    }) as T;
  }

  throttle<T extends (...args: any[]) => any>(fn: T, limit: number): T {
    let inThrottle: boolean;
    
    return ((...args: Parameters<T>) => {
      if (!inThrottle) {
        fn(...args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    }) as T;
  }

  memoize<T extends (...args: any[]) => any>(fn: T): T {
    const cache = new Map();
    
    return ((...args: Parameters<T>) => {
      const key = JSON.stringify(args);
      
      if (cache.has(key)) {
        return cache.get(key);
      }
      
      const result = fn(...args);
      cache.set(key, result);
      return result;
    }) as T;
  }

  getMemoryUsage(): MemoryInfo | null {
    if ('memory' in performance) {
      return (performance as any).memory as MemoryInfo;
    }
    return null;
  }

  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}

export const performanceUtils = new PerformanceUtilities();

// Additional utility functions
export const createPerformanceObserver = (callback: (entries: PerformanceObserverEntryList) => void) => {
  if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver(callback);
    observer.observe({ entryTypes: ['measure', 'navigation', 'paint'] });
    return observer;
  }
  return null;
};

export const measureComponentRender = (componentName: string) => {
  return {
    start: () => performance.mark(`${componentName}-start`),
    end: () => {
      performance.mark(`${componentName}-end`);
      performance.measure(
        `${componentName}-render`,
        `${componentName}-start`,
        `${componentName}-end`
      );
    }
  };
};
