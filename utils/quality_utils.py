"""Utility functions for face quality assessment."""

from optimized.utils.constants import (
    BLUR_NORMALIZATION_FACTOR, OPTIMAL_BRIGHTNESS, MAX_BRIGHTNESS_DEVIATION,
    BLUR_WEIGHT, BRIGHTNESS_WEIGHT, DISTANCE_WEIGHT
)

def compute_quality_score(blur_var: float, brightness: float, distance: float = 0.5) -> float:
    """Compute overall quality score for a face.
    
    Args:
        blur_var: Blur variance (higher is better)
        brightness: Average brightness (0-255, closer to 127.5 is better)
        distance: Distance to cluster centroid (lower is better)
        
    Returns:
        Quality score between 0 and 1
    """
    # Normalize blur variance (higher is better, up to a point)
    blur_score = min(1.0, blur_var / BLUR_NORMALIZATION_FACTOR) if blur_var > 0 else 0.0
    
    # Normalize brightness (closer to optimal is better)
    brightness_score = 1.0 - abs(brightness - OPTIMAL_BRIGHTNESS) / MAX_BRIGHTNESS_DEVIATION
    
    # Distance score (lower is better)
    distance_score = max(0.0, 1.0 - distance)
    
    # Combined score with weights
    return (blur_score * BLUR_WEIGHT + brightness_score * BRIGHTNESS_WEIGHT + distance_score * DISTANCE_WEIGHT)