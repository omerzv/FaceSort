"""Input validation utilities."""

import os
from pathlib import Path
from typing import List, Optional
import logging

from optimized.utils.logger import get_logger
from optimized.utils.constants import SUPPORTED_IMAGE_EXTENSIONS, DEFAULT_EMBEDDING_DIMENSION

logger = get_logger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_input_directory(input_dir: str) -> Path:
    """Validate input directory exists and is readable.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If directory is invalid
    """
    if not input_dir:
        raise ValidationError("Input directory cannot be empty")
    
    path = Path(input_dir)
    
    if not path.exists():
        raise ValidationError(f"Input directory does not exist: {input_dir}")
    
    if not path.is_dir():
        raise ValidationError(f"Input path is not a directory: {input_dir}")
    
    if not os.access(path, os.R_OK):
        raise ValidationError(f"Input directory is not readable: {input_dir}")
    
    return path

def validate_output_directory(output_dir: str) -> Path:
    """Validate and create output directory if needed.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If directory cannot be created or accessed
    """
    if not output_dir:
        raise ValidationError("Output directory cannot be empty")
    
    path = Path(output_dir)
    
    # Create directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ValidationError(f"Permission denied creating output directory: {output_dir}")
    except Exception as e:
        raise ValidationError(f"Failed to create output directory {output_dir}: {e}")
    
    # Check if writable
    if not os.access(path, os.W_OK):
        raise ValidationError(f"Output directory is not writable: {output_dir}")
    
    return path

def validate_image_files(image_paths: List[str]) -> List[str]:
    """Validate list of image file paths.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of valid image file paths
    """
    if not image_paths:
        logger.warning("No image paths provided")
        return []
    
    valid_paths = []
    supported_extensions = SUPPORTED_IMAGE_EXTENSIONS
    
    for path_str in image_paths:
        path = Path(path_str)
        
        # Check if file exists
        if not path.exists():
            logger.debug(f"Skipping non-existent file: {path}")
            continue
        
        # Check if it's a file
        if not path.is_file():
            logger.debug(f"Skipping non-file: {path}")
            continue
        
        # Check extension
        if path.suffix.lower() not in supported_extensions:
            logger.debug(f"Skipping unsupported file type: {path}")
            continue
        
        # Check if readable
        if not os.access(path, os.R_OK):
            logger.debug(f"Skipping unreadable file: {path}")
            continue
        
        valid_paths.append(str(path))
    
    logger.info(f"Validated {len(valid_paths)} of {len(image_paths)} image files")
    return valid_paths

def validate_cluster_id(cluster_id: int) -> bool:
    """Validate cluster ID.
    
    Args:
        cluster_id: Cluster ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(cluster_id, int) and cluster_id >= -1

def validate_embedding_dimensions(embeddings: List, expected_dim: int = DEFAULT_EMBEDDING_DIMENSION) -> bool:
    """Validate embedding dimensions are consistent.
    
    Args:
        embeddings: List of embeddings to validate
        expected_dim: Expected embedding dimension
        
    Returns:
        True if all embeddings have correct dimensions
    """
    if not embeddings:
        return True
    
    for i, embedding in enumerate(embeddings):
        if not hasattr(embedding, 'shape'):
            logger.warning(f"Embedding {i} has no shape attribute")
            return False
        
        if len(embedding.shape) != 1:
            logger.warning(f"Embedding {i} is not 1D: shape {embedding.shape}")
            return False
        
        if embedding.shape[0] != expected_dim:
            logger.warning(f"Embedding {i} has wrong dimension: {embedding.shape[0]} != {expected_dim}")
            return False
    
    return True