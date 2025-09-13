"""Embedding preprocessing utilities for improved clustering."""

import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

from optimized.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingPreprocessor:
    """Handles embedding preprocessing for improved clustering."""
    
    def __init__(self, normalize: bool = True, reduce_dim: Optional[int] = None, 
                 reduction_method: str = 'pca'):
        """Initialize embedding preprocessor.
        
        Args:
            normalize: Whether to L2 normalize embeddings
            reduce_dim: Target dimensionality for reduction (None to skip)
            reduction_method: 'pca' or 'tsne'
        """
        self.normalize = normalize
        self.reduce_dim = reduce_dim
        self.reduction_method = reduction_method
        self.reducer = None
        self.fitted = False
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit preprocessor and transform embeddings.
        
        Args:
            embeddings: Input embeddings (n_samples, n_features)
            
        Returns:
            Preprocessed embeddings
        """
        if embeddings.size == 0:
            return embeddings
        
        processed = embeddings.copy()
        
        # L2 Normalization
        if self.normalize:
            processed = self._normalize_l2(processed)
            logger.debug("Applied L2 normalization to embeddings")
        
        # Dimensionality reduction
        if self.reduce_dim and self.reduce_dim < embeddings.shape[1]:
            processed = self._fit_reduce_dimensions(processed)
            logger.info(f"Reduced embedding dimensions: {embeddings.shape[1]} -> {processed.shape[1]}")
        
        self.fitted = True
        return processed
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted preprocessor.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Preprocessed embeddings
            
        Raises:
            ValueError: If preprocessor not fitted yet
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if embeddings.size == 0:
            return embeddings
        
        processed = embeddings.copy()
        
        # L2 Normalization
        if self.normalize:
            processed = self._normalize_l2(processed)
        
        # Dimensionality reduction
        if self.reducer is not None:
            processed = self.reducer.transform(processed)
        
        return processed
    
    def _normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply L2 normalization with consistent math for cosine similarity.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            L2 normalized embeddings (float32 for consistent precision)
        """
        # CRITICAL: Consistent L2 normalization for cosine similarity throughout pipeline
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12  # Small epsilon to avoid division by zero
        
        # Ensure float32 for consistent precision across all clustering operations
        normalized = (embeddings / norms).astype(np.float32)
        
        return normalized
    
    def _fit_reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit dimensionality reduction and transform embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Reduced embeddings
        """
        if self.reduction_method == 'pca':
            self.reducer = PCA(n_components=self.reduce_dim, random_state=42)
            return self.reducer.fit_transform(embeddings)
        
        elif self.reduction_method == 'tsne':
            # t-SNE doesn't have separate fit/transform, so we store the result
            self.reducer = TSNE(n_components=self.reduce_dim, random_state=42, 
                              perplexity=min(30, embeddings.shape[0] - 1))
            reduced = self.reducer.fit_transform(embeddings)
            
            # For t-SNE, we can't transform new data, so we just return the result
            logger.warning("t-SNE doesn't support transform on new data")
            return reduced
        
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio for PCA.
        
        Returns:
            Explained variance ratios or None if not PCA
        """
        if isinstance(self.reducer, PCA):
            return self.reducer.explained_variance_ratio_
        return None


def preprocess_embeddings_for_clustering(embeddings: np.ndarray, 
                                        config: Optional[dict] = None) -> Tuple[np.ndarray, EmbeddingPreprocessor]:
    """Convenience function to preprocess embeddings for clustering.
    
    Args:
        embeddings: Input embeddings
        config: Preprocessing config dict with keys:
               - normalize: bool (default True)
               - reduce_dim: int or None (default None)
               - reduction_method: str (default 'pca')
    
    Returns:
        Tuple of (processed_embeddings, preprocessor)
    """
    if config is None:
        config = {}
    
    preprocessor = EmbeddingPreprocessor(
        normalize=config.get('normalize', True),
        reduce_dim=config.get('reduce_dim', None),
        reduction_method=config.get('reduction_method', 'pca')
    )
    
    processed_embeddings = preprocessor.fit_transform(embeddings)
    
    # Log preprocessing stats
    if preprocessor.normalize:
        logger.debug("Embeddings normalized with L2 norm")
    
    if preprocessor.reduce_dim:
        variance_ratio = preprocessor.get_explained_variance_ratio()
        if variance_ratio is not None:
            total_variance = np.sum(variance_ratio)
            logger.info(f"PCA retained {total_variance:.3f} of total variance")
    
    return processed_embeddings, preprocessor