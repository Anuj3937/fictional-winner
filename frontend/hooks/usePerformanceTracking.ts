import { useState, useEffect, useCallback, useRef } from 'react';

interface PerformanceMetrics {
  memoryUsage: number;
  renderTime: number;
  componentCount: number;
  lastUpdate: number;
  fps: number;
}

interface UsePerformanceTrackingReturn {
  metrics: PerformanceMetrics;
  startTracking: () => void;
  stopTracking: () => void;
  isTracking: boolean;
  recordRender: () => void;
  recordComponent: () => void;
}

export function usePerformanceTracking(): UsePerformanceTrackingReturn {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    memoryUsage: 0,
    renderTime: 0,
    componentCount: 0,
    lastUpdate: Date.now(),
    fps: 0
  });

  const [isTracking, setIsTracking] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout>();
  const frameRef = useRef<number>();
  const renderStartTime = useRef<number>(0);
  const fpsFrames = useRef<number[]>([]);

  const updateMemoryUsage = useCallback(() => {
    if ('memory' in performance) {
      const memInfo = (performance as any).memory;
      const memoryUsage = memInfo.usedJSHeapSize / (1024 * 1024); // MB
      
      setMetrics(prev => ({
        ...prev,
        memoryUsage,
        lastUpdate: Date.now()
      }));
    }
  }, []);

  const updateFPS = useCallback(() => {
    const now = Date.now();
    fpsFrames.current.push(now);
    
    // Keep only frames from the last second
    fpsFrames.current = fpsFrames.current.filter(time => now - time <= 1000);
    
    const fps = fpsFrames.current.length;
    
    setMetrics(prev => ({
      ...prev,
      fps
    }));

    if (isTracking) {
      frameRef.current = requestAnimationFrame(updateFPS);
    }
  }, [isTracking]);

  const startTracking = useCallback(() => {
    if (isTracking) return;
    
    setIsTracking(true);
    
    // Update memory usage every second
    intervalRef.current = setInterval(updateMemoryUsage, 1000);
    
    // Start FPS tracking
    updateFPS();
    
    // Initial memory reading
    updateMemoryUsage();
  }, [isTracking, updateMemoryUsage, updateFPS]);

  const stopTracking = useCallback(() => {
    setIsTracking(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
    }
  }, []);

  const recordRender = useCallback(() => {
    if (!isTracking) return;
    
    if (renderStartTime.current === 0) {
      renderStartTime.current = performance.now();
    } else {
      const renderTime = performance.now() - renderStartTime.current;
      renderStartTime.current = 0;
      
      setMetrics(prev => ({
        ...prev,
        renderTime
      }));
    }
  }, [isTracking]);

  const recordComponent = useCallback(() => {
    if (!isTracking) return;
    
    setMetrics(prev => ({
      ...prev,
      componentCount: prev.componentCount + 1
    }));
  }, [isTracking]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  return {
    metrics,
    startTracking,
    stopTracking,
    isTracking,
    recordRender,
    recordComponent
  };
}
