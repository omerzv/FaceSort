"""Enhanced logging system with performance monitoring and structured logging."""

import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from contextvars import ContextVar
import asyncio
from dataclasses import dataclass

from .config import LoggingConfig

# Context variable for request tracking
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: float
    level: str
    logger: str
    message: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        context = request_context.get({})
        
        log_data = {
            'timestamp': record.created,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context information
        if context:
            log_data.update(context)
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info'):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class PerformanceFormatter(logging.Formatter):
    """Formatter that includes performance metrics."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with performance information."""
        context = request_context.get({})
        
        # Base message
        message = super().format(record)
        
        # Add performance info if available
        perf_info = []
        if 'execution_time' in context:
            perf_info.append(f"exec_time={context['execution_time']:.3f}s")
        if 'memory_mb' in context:
            perf_info.append(f"memory={context['memory_mb']:.1f}MB")
        if 'request_id' in context:
            perf_info.append(f"req_id={context['request_id'][:8]}")
        
        if perf_info:
            message += f" [{', '.join(perf_info)}]"
        
        return message


class AsyncRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Async-aware rotating file handler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = asyncio.Lock()
    
    async def emit_async(self, record: logging.LogRecord):
        """Emit log record asynchronously."""
        try:
            async with self._lock:
                self.emit(record)
        except Exception:
            self.handleError(record)


class LoggerManager:
    """Centralized logger management with performance monitoring."""
    
    def __init__(self, config: LoggingConfig):
        """Initialize logger manager."""
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.performance_stats = {
            'total_logs': 0,
            'error_logs': 0,
            'warning_logs': 0,
            'start_time': time.time()
        }
        
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.level.upper()))
        
        # Use structured logging in production, readable in development
        if self.config.level.upper() == 'DEBUG':
            console_formatter = PerformanceFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_formatter = logging.Formatter(self.config.format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
        
        # File handler if enabled
        if self.config.file_enabled:
            self._setup_file_handler()
        
        # Performance tracking handler
        self._setup_performance_handler()
    
    def _setup_file_handler(self):
        """Setup rotating file handler."""
        logs_dir = Path("./logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Main log file
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "facesorter.log",
            maxBytes=self.config.file_max_size,
            backupCount=self.config.file_backup_count
        )
        file_handler.setLevel(getattr(logging, self.config.level.upper()))
        file_handler.setFormatter(StructuredFormatter())
        
        logging.getLogger().addHandler(file_handler)
        self.handlers['file'] = file_handler
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "errors.log",
            maxBytes=self.config.file_max_size,
            backupCount=self.config.file_backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        logging.getLogger().addHandler(error_handler)
        self.handlers['error_file'] = error_handler
    
    def _setup_performance_handler(self):
        """Setup handler for performance monitoring."""
        
        class StatsHandler(logging.Handler):
            def __init__(self, stats_dict):
                super().__init__()
                self.stats = stats_dict
            
            def emit(self, record):
                self.stats['total_logs'] += 1
                if record.levelno >= logging.ERROR:
                    self.stats['error_logs'] += 1
                elif record.levelno >= logging.WARNING:
                    self.stats['warning_logs'] += 1
        
        stats_handler = StatsHandler(self.performance_stats)
        logging.getLogger().addHandler(stats_handler)
        self.handlers['stats'] = stats_handler
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get logger instance."""
        if name is None:
            return logging.getLogger()
        
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return self.loggers[name]
    
    def set_context(self, **kwargs):
        """Set logging context."""
        context = request_context.get({}).copy()
        context.update(kwargs)
        request_context.set(context)
    
    def clear_context(self):
        """Clear logging context."""
        request_context.set({})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        uptime = time.time() - self.performance_stats['start_time']
        total_logs = self.performance_stats['total_logs']
        
        return {
            **self.performance_stats,
            'uptime_seconds': uptime,
            'logs_per_second': total_logs / uptime if uptime > 0 else 0,
            'error_rate': (self.performance_stats['error_logs'] / total_logs 
                          if total_logs > 0 else 0)
        }
    
    def create_child_logger(self, parent_name: str, child_name: str) -> logging.Logger:
        """Create child logger."""
        full_name = f"{parent_name}.{child_name}"
        return self.get_logger(full_name)


class ContextManager:
    """Context manager for logging context."""
    
    def __init__(self, **context):
        self.context = context
        self.previous_context = None
    
    def __enter__(self):
        self.previous_context = request_context.get({})
        new_context = self.previous_context.copy()
        new_context.update(self.context)
        request_context.set(new_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_context.set(self.previous_context or {})


class AsyncContextManager:
    """Async context manager for logging context."""
    
    def __init__(self, **context):
        self.context = context
        self.previous_context = None
    
    async def __aenter__(self):
        self.previous_context = request_context.get({})
        new_context = self.previous_context.copy()
        new_context.update(self.context)
        request_context.set(new_context)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        request_context.set(self.previous_context or {})


def performance_timer(func_name: str = None):
    """Decorator for performance timing."""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                context = request_context.get({})
                context['execution_time'] = execution_time
                request_context.set(context)
                
                if execution_time > 1.0:  # Log slow operations
                    logger = logging.getLogger(name)
                    logger.warning(f"Slow operation: {name} took {execution_time:.3f}s")
        
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                context = request_context.get({})
                context['execution_time'] = execution_time
                request_context.set(context)
                
                if execution_time > 1.0:  # Log slow operations
                    logger = logging.getLogger(name)
                    logger.warning(f"Slow async operation: {name} took {execution_time:.3f}s")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None


def setup_logging(config: LoggingConfig = None) -> LoggerManager:
    """Setup global logging configuration."""
    global _logger_manager
    
    if config is None:
        config = LoggingConfig()
    
    _logger_manager = LoggerManager(config)
    return _logger_manager


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance."""
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = setup_logging()
    
    return _logger_manager.get_logger(name)


def set_log_context(**kwargs):
    """Set logging context."""
    global _logger_manager
    if _logger_manager:
        _logger_manager.set_context(**kwargs)


def clear_log_context():
    """Clear logging context."""
    global _logger_manager
    if _logger_manager:
        _logger_manager.clear_context()


def log_context(**context):
    """Context manager for logging context."""
    return ContextManager(**context)


def async_log_context(**context):
    """Async context manager for logging context."""
    return AsyncContextManager(**context)


def get_logging_stats() -> Dict[str, Any]:
    """Get logging statistics."""
    global _logger_manager
    if _logger_manager:
        return _logger_manager.get_stats()
    return {}