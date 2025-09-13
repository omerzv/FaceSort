"""Face embedding computation functionality with batched ONNX optimization."""

import numpy as np
from typing import List, Any, Optional, Union
import logging

from optimized.utils.logger import get_logger
from optimized.utils.constants import DEFAULT_EMBEDDING_DIMENSION

logger = get_logger(__name__)


class FaceEmbeddings:
    """Face embedding computer with batched ONNX processing."""
    
    def __init__(self, detector: Any, onnx_session: Optional[Any] = None) -> None:
        self.detector = detector
        self.session = onnx_session
        self._build_session_if_needed()
        
    def _build_session_if_needed(self) -> None:
        """Build ONNX session if not provided and detector available."""
        if self.session is None and self.detector and hasattr(self.detector, 'app'):
            try:
                # Try to extract ONNX session from InsightFace detector
                if hasattr(self.detector.app, 'models') and 'recognition' in self.detector.app.models:
                    recognition_model = self.detector.app.models['recognition']
                    if hasattr(recognition_model, 'session'):
                        self.session = recognition_model.session
                        logger.debug("Using InsightFace ONNX session for batched embeddings")
            except Exception as e:
                logger.debug(f"Could not extract ONNX session: {e}")
        
    def compute_embeddings(self, crops_or_faces: Union[List[np.ndarray], List[Any]]) -> np.ndarray:
        """Compute embeddings from face crops or face objects with batched ONNX."""
        if not crops_or_faces:
            return np.zeros((0, DEFAULT_EMBEDDING_DIMENSION), dtype=np.float32)
        
        # Check if input is face crops or face objects with validation
        first_item = crops_or_faces[0]
        if isinstance(first_item, np.ndarray):
            # Validate all items are numpy arrays
            if not all(isinstance(item, np.ndarray) for item in crops_or_faces):
                raise ValueError("Mixed types detected: all items must be numpy arrays when first item is numpy array")
            # Input is face crops - use batched ONNX
            return self._compute_embeddings_batched_onnx(crops_or_faces)
        else:
            # Input is face objects - extract existing embeddings
            return self._extract_embeddings_from_faces(crops_or_faces)
    
    def _compute_embeddings_batched_onnx(self, crops: List[np.ndarray]) -> np.ndarray:
        """Compute embeddings using batched ONNX processing."""
        if not crops or not self.session:
            # Fallback to individual processing if no ONNX session
            return self._compute_embeddings_individual(crops)
        
        try:
            # Preprocess crops to batch format (N, 3, 112, 112)
            batch = self._preprocess_crops_batch(crops)
            
            # Run batched ONNX inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            embeddings = self.session.run([output_name], {input_name: batch})[0]
            embeddings = embeddings.astype(np.float32)
            
            # L2 normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Batched ONNX embedding failed: {e}, falling back to individual processing")
            return self._compute_embeddings_individual(crops)
    
    def _preprocess_crops_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Preprocess face crops to batch format for ONNX."""
        if not crops:
            return np.zeros((0, 3, 112, 112), dtype=np.float32)
        
        batch = np.zeros((len(crops), 3, 112, 112), dtype=np.float32)
        
        for i, crop in enumerate(crops):
            if crop.shape != (112, 112, 3):
                # Ensure correct size
                import cv2
                crop = cv2.resize(crop, (112, 112))
            
            # Convert to CHW format and normalize
            if crop.dtype != np.float32:
                crop = crop.astype(np.float32)
            if crop.max() > 1.0:
                crop = crop / 255.0
            
            # Transpose from HWC to CHW
            batch[i] = crop.transpose(2, 0, 1)
        
        return batch
    
    def _compute_embeddings_individual(self, crops: List[np.ndarray]) -> np.ndarray:
        """Fallback individual embedding computation."""
        embeddings = []
        for crop in crops:
            try:
                # Use detector's app to compute individual embedding
                if self.detector and hasattr(self.detector, 'app') and self.detector.app:
                    # Convert crop back to proper format for InsightFace
                    if crop.max() <= 1.0:
                        crop_uint8 = (crop * 255).astype(np.uint8)
                    else:
                        crop_uint8 = crop.astype(np.uint8)
                    
                    # Get embedding from InsightFace
                    faces = self.detector.app.get(crop_uint8, max_num=1)
                    if faces and hasattr(faces[0], 'embedding'):
                        embeddings.append(faces[0].embedding)
                    else:
                        embeddings.append(np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32))
                else:
                    embeddings.append(np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32))
            except Exception:
                embeddings.append(np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def _extract_embeddings_from_faces(self, faces: List[Any]) -> np.ndarray:
        """Extract embeddings from face objects (legacy method)."""
        embeddings = []
        for face in faces:
            if hasattr(face, 'embedding') and face.embedding is not None:
                embeddings.append(face.embedding)
            else:
                logger.warning(f"Face {getattr(face, 'id', 'unknown')} missing embedding, using zeros")
                embeddings.append(np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
