"""System monitoring utilities for the chat interface."""

import psutil
import time
from typing import Dict, Any, Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class SystemMonitor:
    """Monitors system resources and GPU status."""
    
    def __init__(self):
        self.gpu_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_initialized = True
            except Exception:
                pass
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics."""
        if not self.gpu_initialized or not PYNVML_AVAILABLE:
            return {'available': False, 'error': 'GPU monitoring not available'}
        
        try:
            # Get GPU handle (assuming single GPU)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent = util.gpu
            
            # Get memory info
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = int(meminfo.used)
            memory_total = int(meminfo.total)
            memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
            
            return {
                'available': True,
                'gpu_percent': gpu_percent,
                'memory_percent': round(memory_percent, 1),
                'memory_used_gb': round(memory_used / 1024**3, 2),
                'memory_total_gb': round(memory_total / 1024**3, 2)
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0)  # Non-blocking cached value
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory.percent, 1),
                'memory_used_gb': round(memory.used / 1024**3, 2),
                'memory_total_gb': round(memory.total / 1024**3, 2),
                'available': True
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        if self.gpu_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    
    def get_combined_status(self) -> str:
        """Get formatted system status string."""
        sys_stats = self.get_system_stats()
        gpu_stats = self.get_gpu_stats()
        
        status_parts = []
        
        if sys_stats.get('available'):
            status_parts.append(f"CPU: {sys_stats['cpu_percent']}%")
            status_parts.append(f"RAM: {sys_stats['memory_percent']}%")
        
        if gpu_stats.get('available'):
            status_parts.append(f"GPU: {gpu_stats['gpu_percent']}%")
            status_parts.append(f"VRAM: {gpu_stats['memory_percent']}%")
        
        return " | ".join(status_parts) if status_parts else "System monitoring unavailable"