"""Face clustering: HDBSCAN → DBSCAN noise rescue → Cluster merging."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from typing import Optional, Dict, List as ListType, Tuple, Type, Any

try:
    # HDBSCAN is in the separate hdbscan package, not sklearn.cluster
    from hdbscan import HDBSCAN  # type: ignore
    HDBSCAN_AVAILABLE = True
    HDBSCAN_CLASS: Optional[Type[Any]] = HDBSCAN
except ImportError:
    HDBSCAN_CLASS = None
    HDBSCAN_AVAILABLE = False
import logging

from optimized.utils.logger import get_logger
from optimized.utils.config import ClusteringConfig
from optimized.utils.constants import MIN_CLUSTER_SIZE, NOISE_CLUSTER_ID
from optimized.utils.union_find import UnionFind
from optimized.utils.embedding_preprocessing import preprocess_embeddings_for_clustering

logger = get_logger(__name__)


class FaceClusterer:
    """Face clusterer: HDBSCAN → DBSCAN rescue → Cluster merging."""
    
    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config
        
    def cluster_faces(self, embeddings: np.ndarray) -> np.ndarray:
        """3-stage hybrid clustering: Preprocessing → HDBSCAN → DBSCAN rescue → Merge similar clusters."""
        if embeddings.size == 0:
            return np.array([], dtype=int)
            
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))
        
        # Preprocess embeddings for better clustering
        preprocessing_config = {
            'normalize': True,
            'reduce_dim': None,  # Keep full dimensionality for now
            'reduction_method': 'pca'
        }
        
        processed_embeddings, _ = preprocess_embeddings_for_clustering(
            embeddings, preprocessing_config
        )
        
        # Stage 1: HDBSCAN clustering (with fallback to DBSCAN if unavailable)
        algorithm_used = "DBSCAN (fallback)"  # Default assumption
        
        if HDBSCAN_AVAILABLE and HDBSCAN_CLASS is not None:
            try:
                # HDBSCAN parameter sanity checks
                hdbscan_min_cluster_size = max(2, self.config.hdbscan_min_cluster_size)  # Never less than 2
                hdbscan_min_samples = max(1, self.config.min_samples)
                
                # Sanity check cluster_selection_epsilon for normalized embeddings
                cluster_eps = self.config.hdbscan_cluster_selection_epsilon
                if cluster_eps > 0.5:  # Too large for cosine space
                    logger.warning(f"HDBSCAN cluster_selection_epsilon ({cluster_eps}) seems large for normalized embeddings, capping at 0.1")
                    cluster_eps = 0.1
                elif cluster_eps < 0.0:
                    cluster_eps = 0.0
                
                hdbscan = HDBSCAN_CLASS(
                    min_cluster_size=hdbscan_min_cluster_size,
                    min_samples=hdbscan_min_samples,
                    cluster_selection_epsilon=cluster_eps,
                    alpha=max(0.1, self.config.hdbscan_alpha),  # Never zero or negative
                    metric='euclidean'  # HDBSCAN works better with euclidean on normalized embeddings
                )
                
                logger.debug(f"HDBSCAN params: min_cluster_size={hdbscan_min_cluster_size}, "
                           f"min_samples={hdbscan_min_samples}, eps={cluster_eps:.3f}, alpha={self.config.hdbscan_alpha}")
                
                labels = hdbscan.fit_predict(processed_embeddings)
                algorithm_used = "HDBSCAN"  # Only set this if successful
            except Exception as e:
                logger.warning(f"HDBSCAN failed ({e!r}), falling back to DBSCAN")
                dbscan = DBSCAN(
                    eps=self.config.eps,
                    min_samples=max(1, self.config.min_samples),
                    metric='cosine'
                )
                labels = dbscan.fit_predict(processed_embeddings)
        else:
            logger.warning("HDBSCAN not available, falling back to DBSCAN for stage 1")
            dbscan = DBSCAN(
                eps=self.config.eps,
                min_samples=max(1, self.config.min_samples),
                metric='cosine'
            )
            labels = dbscan.fit_predict(processed_embeddings)
        
        initial_clusters = len(np.unique(labels[labels != -1]))
        initial_noise = np.sum(labels == -1)
        
        # Stage 2: DBSCAN noise rescue
        if initial_noise > 0:
            noise_indices = np.where(labels == -1)[0]
            noise_embeddings = processed_embeddings[noise_indices]
            
            # More permissive DBSCAN parameters for noise rescue
            rescue_eps = self.config.eps * self.config.dbscan_noise_rescue_eps_factor
            rescue_min_samples = max(1, self.config.dbscan_noise_rescue_min_samples)
            dbscan = DBSCAN(eps=rescue_eps, min_samples=rescue_min_samples, metric='cosine')
            noise_labels = dbscan.fit_predict(noise_embeddings)
            
            # Assign rescued points to new clusters
            max_existing_label = labels.max() if len(labels[labels != -1]) > 0 else -1
            rescued_count = 0
            
            for i, noise_label in enumerate(noise_labels):
                if noise_label != -1:  # Point was rescued
                    original_idx = noise_indices[i]
                    labels[original_idx] = max_existing_label + 1 + noise_label
                    rescued_count += 1
            
        
        # Stage 3: Merge similar clusters
        labels = self._merge_similar_clusters(processed_embeddings, labels)
        
        # Final cleanup: remove tiny clusters and renumber
        labels = self._cleanup_and_renumber(labels)
        
        final_clusters = len(np.unique(labels[labels != -1]))
        final_noise = np.sum(labels == -1)
        
        
        return labels
    
    def _merge_similar_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Optimized cluster merging using vectorized operations."""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != NOISE_CLUSTER_ID]  # Exclude noise
        
        if len(unique_labels) < 2:
            return labels
        
        # Calculate cluster centroids using vectorized operations with consistent normalization
        centroids = np.zeros((len(unique_labels), embeddings.shape[1]), dtype=np.float32)
        
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            # CRITICAL: Consistent L2-normalization for proper cosine similarity
            norm = np.linalg.norm(centroid) + 1e-12  # Avoid division by zero
            centroids[i] = (centroid / norm).astype(np.float32)
        
        # Vectorized similarity computation - much faster than nested loops
        similarity_matrix = cosine_similarity(centroids)
        merge_threshold = self.config.merge_similarity_threshold
        
        # Find pairs above threshold (excluding diagonal)
        similar_pairs = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                if similarity_matrix[i, j] > merge_threshold:
                    similar_pairs.append((unique_labels[i], unique_labels[j], similarity_matrix[i, j]))
        
        if not similar_pairs:
            return labels
        
        # Use Union-Find for robust transitive merging
        uf = UnionFind(unique_labels.tolist())
        
        # Union all similar pairs
        for label1, label2, similarity in similar_pairs:
            uf.union(label1, label2)
        
        # Get connected components (merged clusters)
        components = uf.get_components()
        
        # Vectorized label mapping - faster than individual lookups
        new_labels = labels.copy()
        new_label_counter = 0
        
        for root, cluster_group in components.items():
            # Map all old labels in this group to the same new label
            for old_label in cluster_group:
                new_labels[labels == old_label] = new_label_counter
            new_label_counter += 1
        
        merged_groups = len([group for group in components.values() if len(group) > 1])
        total_merged = sum(len(group) - 1 for group in components.values() if len(group) > 1)
        
        
        return new_labels
    
    def _cleanup_and_renumber(self, labels: np.ndarray) -> np.ndarray:
        """Remove tiny clusters and renumber to be consecutive."""
        # Remove clusters with only 1 face
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label != NOISE_CLUSTER_ID and count < MIN_CLUSTER_SIZE:
                labels[labels == label] = NOISE_CLUSTER_ID
        
        # Renumber clusters to be consecutive (0, 1, 2, ...)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != NOISE_CLUSTER_ID]  # Exclude noise
        
        new_labels = labels.copy()
        for i, label in enumerate(unique_labels):
            new_labels[labels == label] = i
            
        return new_labels