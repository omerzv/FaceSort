"""Utility for selective output suppression."""

import sys
import os
from contextlib import contextmanager
from io import StringIO
import logging

from optimized.utils.logger import get_logger

logger = get_logger(__name__)

@contextmanager
def suppress_insightface_output():
    """Context manager to selectively suppress InsightFace verbose output while preserving important messages."""
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create string buffers to capture output
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    try:
        # Redirect streams
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        yield
        
    finally:
        # Always restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Process captured output - log important messages
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        # Filter and log important messages
        if stdout_content:
            important_stdout = _filter_important_messages(stdout_content, 'stdout')
            if important_stdout:
                logger.debug(f"InsightFace stdout: {important_stdout}")
        
        if stderr_content:
            important_stderr = _filter_important_messages(stderr_content, 'stderr')
            if important_stderr:
                # Errors should be at warning level
                logger.warning(f"InsightFace stderr: {important_stderr}")

@contextmanager
def suppress_detector_init_output():
    """Suppress output during detector initialization but preserve errors."""
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Only suppress stdout (model loading messages), keep stderr for real errors
    stdout_buffer = StringIO()
    
    try:
        sys.stdout = stdout_buffer
        yield
        
    finally:
        sys.stdout = original_stdout
        
        # Log any captured stdout at debug level
        stdout_content = stdout_buffer.getvalue()
        if stdout_content.strip():
            # Only log if there's actually content and it looks important
            if any(keyword in stdout_content.lower() for keyword in ['error', 'warning', 'failed', 'exception']):
                logger.debug(f"Detector init output: {stdout_content.strip()}")

def _filter_important_messages(content: str, stream_type: str) -> str:
    """Filter content to identify important messages worth logging."""
    if not content.strip():
        return ""
    
    lines = content.strip().split('\n')
    important_lines = []
    
    # Keywords that indicate important messages
    important_keywords = [
        'error', 'warning', 'failed', 'exception', 'critical', 
        'fatal', 'cannot', 'unable', 'missing', 'not found'
    ]
    
    # Keywords for verbose/debug messages we can ignore
    ignore_keywords = [
        'loading model', 'model loaded', 'initialize', 'set model',
        'find model', 'download', 'onnx', 'tensorrt', 'gpu memory'
    ]
    
    for line in lines:
        line_lower = line.lower()
        
        # Always include error/warning messages
        if any(keyword in line_lower for keyword in important_keywords):
            important_lines.append(line)
        # Skip verbose initialization messages
        elif any(keyword in line_lower for keyword in ignore_keywords):
            continue
        # Include anything else that's not just whitespace
        elif line.strip() and len(line.strip()) > 10:  # Ignore very short messages
            important_lines.append(line)
    
    return '\n'.join(important_lines)