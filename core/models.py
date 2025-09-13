"""Data models for face processing pipeline."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from optimized.utils.quality_utils import compute_quality_score


@dataclass
class FaceSample:
    """Individual face sample with metadata."""
    id: int
    image_path: str
    cluster_id: int
    embedding: np.ndarray
    pca3: np.ndarray
    d2centroid: float
    blur_var: float
    mean_gray: float
    pose_confidence: float
    crop: Optional[np.ndarray] = None
    temp_crop_path: Optional[str] = None
    
    @property
    def quality_score(self) -> float:
        """Compute quality score for the face."""
        return compute_quality_score(self.blur_var, self.mean_gray, self.d2centroid)


@dataclass
class ClusterInfo:
    """Information about a face cluster."""
    cluster_id: int
    face_count: int
    centroid: np.ndarray
    faces: List[FaceSample]
    avg_quality: float
    avg_blur: float
    avg_brightness: float
    avg_distance: float
    best_face: Optional[FaceSample] = None
    worst_face: Optional[FaceSample] = None


@dataclass
class ProcessingStats:
    """Statistics for processing pipeline."""
    total_images: int = 0
    total_faces: int = 0
    clusters_found: int = 0
    noise_faces: int = 0
    high_quality_faces: int = 0
    low_quality_faces: int = 0
    
    # Timing statistics
    total_time: float = 0.0
    detection_time: float = 0.0
    embedding_time: float = 0.0
    clustering_time: float = 0.0
    
    # Quality statistics
    avg_quality: float = 0.0
    
    @property
    def faces_per_second(self) -> float:
        """Calculate faces processed per second."""
        return self.total_faces / self.total_time if self.total_time > 0 else 0.0


@dataclass
class ProcessingResult:
    """Complete processing result."""
    faces: List[FaceSample]
    clusters: Dict[int, ClusterInfo]
    stats: ProcessingStats
    config_used: Dict[str, Any]
    noise_faces: Optional[List[FaceSample]] = None
    
    def __post_init__(self):
        """Post-initialization to set noise_faces."""
        if self.noise_faces is None:
            self.noise_faces = [face for face in self.faces if face.cluster_id == -1]