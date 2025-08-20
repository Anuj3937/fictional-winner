import psutil
import time
import asyncio
from typing import Dict, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class PerformanceMetrics:
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    execution_time: float
    timestamp: float

class PerformanceMonitor:
    """High-performance monitoring with minimal overhead"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics with O(1) complexity"""
        process = psutil.Process()
        
        return PerformanceMetrics(
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_used_mb=process.memory_info().rss / 1024 / 1024,
            execution_time=time.time() - (self.start_time or time.time()),
            timestamp=time.time()
        )
    
    def log_metrics(self, operation: str):
        """Log metrics with minimal performance impact"""
        metrics = self.get_current_metrics()
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics for memory efficiency
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
            
        logger.info(f"{operation} - CPU: {metrics.cpu_percent:.1f}%, "
                   f"Memory: {metrics.memory_used_mb:.1f}MB, "
                   f"Time: {metrics.execution_time:.2f}s")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
