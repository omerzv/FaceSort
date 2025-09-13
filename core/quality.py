"""Face quality assessment functionality."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any
import logging

from optimized.utils.logger import get_logger
from optimized.utils.config import QualityConfig
from optimized.utils.quality_utils import compute_quality_score

logger = get_logger(__name__)


class QualityAssessor:
    """Handles face quality assessment using various metrics."""
    
    def __init__(self, config: QualityConfig) -> None:
        """Initialize quality assessor.
        
        Args:
            config: Quality assessment configuration
        """
        self.config = config
    
    def assess_batch(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Turbo-charged batch quality assessment with maximum vectorization."""
        if not crops:
            return np.array([]), np.array([])
        
        # TURBO: Process everything in vectorized operations
        num_crops = len(crops)
        blur_vars = np.zeros(num_crops, dtype=np.float32)
        brightnesses = np.zeros(num_crops, dtype=np.float32)
        
        # TURBO: Vectorized batch processing with optimized dtypes
        for i, crop in enumerate(crops):
            # Convert to grayscale efficiently with single dtype conversion
            if crop.dtype != np.uint8:
                gray = cv2.cvtColor((crop * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            
            # Convert to float32 once for all operations
            gray_f32 = gray.astype(np.float32)
            
            # TURBO: Fastest blur detection - keep original for consistency
            blur_vars[i] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # TURBO: Fast brightness on pre-converted float32
            brightnesses[i] = float(gray_f32.mean())
        
        return blur_vars, brightnesses
    
    def _assess_single(self, crop: np.ndarray) -> Tuple[float, float]:
        """Assess quality of a single face crop.
        
        Args:
            crop: Face crop in RGB [0,1] format
            
        Returns:
            Tuple of (blur_variance, brightness)
        """
        try:
            # Convert to uint8 BGR for OpenCV
            if crop.dtype == np.float32:
                crop_uint8 = (crop * 255).astype(np.uint8)
            else:
                crop_uint8 = crop
            
            # Convert RGB to BGR for OpenCV
            if crop_uint8.shape[2] == 3:
                crop_bgr = cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = crop_uint8
            
            # Convert to grayscale
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            
            # Compute blur variance using Laplacian
            blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Compute average brightness
            brightness = float(np.mean(gray.astype(np.float32)))
            
            return float(blur_var), brightness
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.0, 127.5  # Default values
    
    def is_high_quality(self, blur_var: float, brightness: float) -> bool:
        """Determine if a face meets high quality criteria.
        
        Args:
            blur_var: Blur variance score
            brightness: Brightness score
            
        Returns:
            True if face meets quality criteria
        """
        return (blur_var >= self.config.blur_threshold and
                self.config.brightness_min <= brightness <= self.config.brightness_max)
    
    def compute_quality_score(self, blur_var: float, brightness: float, 
                             distance: float = 0.5) -> float:
        """Compute overall quality score.
        
        Args:
            blur_var: Blur variance
            brightness: Average brightness
            distance: Distance to cluster centroid
            
        Returns:
            Quality score between 0 and 1
        """
        return compute_quality_score(blur_var, brightness, distance)

