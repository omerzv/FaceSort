"""Face processing pipeline for clustering and analysis."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict
import numpy as np

from optimized.core.face_detection import FaceDetector
from optimized.core.face_embeddings import FaceEmbeddings
from optimized.core.face_clustering import FaceClusterer
from optimized.core.result_storage import ResultSaver
from optimized.core.models import ProcessingResult, ProcessingStats, ClusterInfo
from optimized.utils.config import AppConfig
from optimized.utils.logger import get_logger
from optimized.utils.output_suppression import suppress_insightface_output
from optimized.utils.cli_visualization import visualizer

logger = get_logger(__name__)


class FacePipeline:
    """Face processing pipeline for clustering and analysis."""
    
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = None
        self.embeddings = None
        self.clusterer = None
        self.result_saver = None
        
        # CRITICAL FIX: Config validation at startup
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate critical configuration parameters."""
        issues = []
        
        # Validate clustering parameters
        if not (0 < self.config.clustering.eps < 1):
            issues.append(f"clustering.eps ({self.config.clustering.eps}) must be between 0 and 1 (cosine distance)")
        
        if hasattr(self.config.clustering, 'merge_similarity_threshold'):
            if self.config.clustering.merge_similarity_threshold < 0.6:
                issues.append(f"clustering.merge_similarity_threshold ({self.config.clustering.merge_similarity_threshold}) should be >= 0.6 for meaningful clusters")
        
        # Validate processing parameters
        if self.config.processing.batch_size <= 0:
            issues.append(f"processing.batch_size ({self.config.processing.batch_size}) must be positive")
        
        if self.config.processing.max_workers <= 0:
            issues.append(f"processing.max_workers ({self.config.processing.max_workers}) must be positive")
        
        # Validate clustering parameters
        if self.config.clustering.min_samples <= 0:
            issues.append(f"clustering.min_samples ({self.config.clustering.min_samples}) must be positive")
        
        if issues:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Show warnings for suboptimal settings
        warnings = []
        if self.config.clustering.eps > 0.5:
            warnings.append(f"clustering.eps ({self.config.clustering.eps}) is quite high - may result in over-clustering")
        
        if hasattr(self.config.clustering, 'merge_similarity_threshold') and self.config.clustering.merge_similarity_threshold < 0.8:
            warnings.append(f"clustering.merge_similarity_threshold ({self.config.clustering.merge_similarity_threshold}) is low - may result in over-merging")
        
        for warning in warnings:
            logger.warning(warning)
        
    async def process(self, input_dir: str, output_dir: str, 
                     progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Face processing pipeline with unified progress tracking."""
        start_time = time.time()
        
        # Initialize components
        logger.debug("Initializing pipeline components...")
        self.detector = FaceDetector(self.config.processing)
        self.embeddings = FaceEmbeddings(self.detector)
        self.clusterer = FaceClusterer(self.config.clustering)
        self.result_saver = ResultSaver(output_dir)
        
        # Step 1: Find all images (avoiding duplicates)
        if progress_callback:
            progress_callback("Scanning directories", 5, 100)
        
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}  # Use set for fast lookup
        image_files = []
        
        # Find all files recursively and filter by extension
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if progress_callback:
            progress_callback("Processing photos", 10, 100)
        
        
        # Debug: Show sample paths if count seems unexpected
        if len(image_files) > 100:
            logger.debug(f"Sample image paths found:")
            for i, img_path in enumerate(sorted(image_files)[:5]):
                logger.debug(f"  {i+1}. {img_path}")
            if len(image_files) > 5:
                logger.debug(f"  ... and {len(image_files) - 5} more files")
        
        # Check for potential duplicates by grouping by filename
        from collections import Counter
        filenames = [img.name for img in image_files]
        duplicates = [name for name, count in Counter(filenames).items() if count > 1]
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate filenames: {duplicates[:3]}...")
        
        # Check directory structure
        dirs = set(img.parent for img in image_files)
        if len(dirs) > 1:
            logger.debug(f"Images found in {len(dirs)} directories:")
            for i, directory in enumerate(sorted(dirs)[:3]):
                file_count = len([f for f in image_files if f.parent == directory])
                logger.debug(f"  {directory}: {file_count} files")
            if len(dirs) > 3:
                logger.debug(f"  ... and {len(dirs) - 3} more directories")
        
        # Step 2: Detect faces using batch processing
        all_faces = []
        
        if progress_callback:
            progress_callback("Detecting faces", 20, 100)
        
        # Convert paths to strings
        image_paths = [str(img) for img in image_files]
        
        try:
            # Use selective output suppression during face detection
            with suppress_insightface_output():
                # Use the detector's batch method that returns embeddings too
                face_crops, face_paths, embeddings = self.detector.detect_batch(image_paths)
            
            # Embeddings are already computed by the detector
            if face_crops:
                
                if progress_callback:
                    progress_callback("Processing faces", 50, 100)
                
                # Create face objects with embeddings
                from optimized.core.models import FaceSample
                for i, (crop, path, embedding) in enumerate(zip(face_crops, face_paths, embeddings)):
                    # Extract original image path and face index from the path string
                    if '#' in path:
                        original_path, face_idx_str = path.rsplit('#', 1)
                        face_idx = int(face_idx_str)
                        # Create unique face ID combining global index and face index
                        unique_face_id = i * 1000 + face_idx  # Ensures uniqueness
                    else:
                        original_path = path
                        unique_face_id = i
                    
                    face = FaceSample(
                        id=unique_face_id,
                        image_path=original_path,  # Use clean image path without #index
                        cluster_id=-1,  # Will be set during clustering
                        embedding=embedding,
                        pca3=np.zeros(3),
                        d2centroid=0.0,
                        blur_var=100.0,  # Default values
                        mean_gray=128.0,
                        pose_confidence=0.8,
                        crop=crop
                    )
                    all_faces.append(face)
                
        except ImportError as e:
            logger.error(f"Required dependency missing for face detection: {e}")
            raise RuntimeError(f"Face detection initialization failed - check InsightFace installation: {e}")
        except MemoryError as e:
            logger.error(f"Out of memory during face detection: {e}")
            raise RuntimeError(f"Insufficient memory for face detection batch processing: {e}")
        except FileNotFoundError as e:
            logger.error(f"Image files not found during detection: {e}")
            # Continue with empty results instead of failing completely
            all_faces = []
            face_crops = []
            embeddings = []
        except Exception as e:
            logger.error(f"Unexpected error during face detection: {type(e).__name__}: {e}")
            # For unexpected errors, still try to continue but log the full error
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            all_faces = []
            face_crops = []
            embeddings = []
        
        
        if not all_faces:
            # Return empty result
            stats = ProcessingStats(
                total_images=len(image_files),
                total_faces=0,
                total_time=time.time() - start_time
            )
            return ProcessingResult(faces=[], clusters={}, stats=stats, config_used=self.config.__dict__)
        
        # Step 3: Compute embeddings  
        if progress_callback:
            progress_callback("Computing embeddings", 60, 100)
        
        # The face objects from the detector should already have embeddings
        embeddings_array = self.embeddings.compute_embeddings(all_faces)
        
        # Step 4: Simple clustering
        if progress_callback:
            progress_callback("Clustering faces", 70, 100)
            
        cluster_labels = self.clusterer.cluster_faces(embeddings_array)
        
        # CRITICAL FIX: Assign cluster labels back to face objects
        for face, label in zip(all_faces, cluster_labels):
            face.cluster_id = int(label)
        
        # CRITICAL FIX: Quality assessment pass - compute real quality metrics
        if progress_callback:
            progress_callback("Computing quality metrics", 80, 100)
        
        from optimized.core.quality import QualityAssessor
        qa = QualityAssessor(self.config.quality)
        
        # Extract crops for quality assessment
        crops_for_quality = []
        valid_face_indices = []
        for i, face in enumerate(all_faces):
            if hasattr(face, 'crop') and face.crop is not None:
                crops_for_quality.append(face.crop)
                valid_face_indices.append(i)
        
        if crops_for_quality:
            blur_vars, brightnesses = qa.assess_batch(crops_for_quality)
            
            # Assign real quality values back to faces
            for idx, (blur_var, brightness) in zip(valid_face_indices, zip(blur_vars, brightnesses)):
                all_faces[idx].blur_var = float(blur_var)
                all_faces[idx].mean_gray = float(brightness)
        
        # Step 5: Organize results
        clusters = {}
        for i, (face, label) in enumerate(zip(all_faces, cluster_labels)):
            if label == -1:  # Skip noise
                continue
                
            if label not in clusters:
                clusters[label] = ClusterInfo(
                    cluster_id=label,
                    face_count=0,
                    centroid=np.zeros(embeddings_array.shape[1]) if len(embeddings_array) > 0 else np.zeros(512),
                    faces=[],
                    avg_quality=0.0,
                    avg_blur=0.0,
                    avg_brightness=0.0,
                    avg_distance=0.0
                )
            
            clusters[label].faces.append(face)
            clusters[label].face_count += 1
        
        # CRITICAL FIX: Compute cluster centroids and d2centroid distances
        if progress_callback:
            progress_callback("Computing cluster statistics", 85, 100)
        
        for cluster_id, cluster_info in clusters.items():
            # Get embeddings for faces in this cluster
            cluster_embeddings = []
            cluster_face_indices = []
            
            for face in cluster_info.faces:
                face_idx = all_faces.index(face)  # Find face index in original list
                if face_idx < len(embeddings_array):
                    cluster_embeddings.append(embeddings_array[face_idx])
                    cluster_face_indices.append(face_idx)
            
            if cluster_embeddings:
                cluster_embeddings = np.array(cluster_embeddings)
                
                # Compute cluster centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                # L2 normalize centroid for cosine similarity
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 0:
                    centroid = centroid / centroid_norm
                cluster_info.centroid = centroid
                
                # Compute d2centroid for each face in cluster (cosine distance)
                distances = []
                qualities = []
                blur_vars = []
                brightnesses = []
                
                for i, face in enumerate(cluster_info.faces):
                    embedding = cluster_embeddings[i] if i < len(cluster_embeddings) else None
                    if embedding is not None:
                        # Cosine distance = 1 - cosine_similarity
                        embedding_norm = np.linalg.norm(embedding)
                        if embedding_norm > 0:
                            normalized_embedding = embedding / embedding_norm
                            cosine_sim = np.dot(normalized_embedding, centroid)
                            cosine_distance = 1 - cosine_sim
                        else:
                            cosine_distance = 1.0  # Max distance for zero embeddings
                        
                        face.d2centroid = float(cosine_distance)
                        distances.append(cosine_distance)
                    else:
                        face.d2centroid = 1.0
                        distances.append(1.0)
                    
                    # Collect quality metrics
                    qualities.append(face.quality_score)
                    blur_vars.append(face.blur_var)
                    brightnesses.append(face.mean_gray)
                
                # Update cluster aggregates
                if distances:
                    cluster_info.avg_distance = float(np.mean(distances))
                if qualities:
                    cluster_info.avg_quality = float(np.mean(qualities))
                if blur_vars:
                    cluster_info.avg_blur = float(np.mean(blur_vars))
                if brightnesses:
                    cluster_info.avg_brightness = float(np.mean(brightnesses))
                
                # Find best and worst faces by quality
                if cluster_info.faces:
                    cluster_info.best_face = max(cluster_info.faces, key=lambda f: f.quality_score)
                    cluster_info.worst_face = min(cluster_info.faces, key=lambda f: f.quality_score)
        
        # Step 6: Save results
        if progress_callback:
            progress_callback("Saving results", 90, 100)
            
        # Prepare clustering info if needed
        clustering_info = None
        if clusters:
            # Convert clusters to a simple dict for saving
            clustering_info = {
                'cluster_to_person': {},  # Could be populated by template matching
                'cluster_stats': {cid: {'face_count': cluster.face_count} for cid, cluster in clusters.items()}
            }
        
        await self.result_saver.save_faces_by_cluster_async(all_faces, clustering_info)
        
        if progress_callback:
            progress_callback("Complete!", 100, 100)
        
        # Create final stats
        processing_time = time.time() - start_time
        stats = ProcessingStats(
            total_images=len(image_files),
            total_faces=len(all_faces),
            clusters_found=len(clusters),
            total_time=processing_time
        )
        
        
        # Display results using rich CLI visualization
        try:
            visualizer.display_processing_stats(stats)
            visualizer.display_cluster_summary(clusters, top_n=10)
            visualizer.display_quality_distribution(all_faces)
            
            # Display output summary
            output_files = {
                'face images': len(all_faces),
                'cluster directories': len(clusters),
                'metadata files': 1  # results.json
            }
            visualizer.display_output_summary(str(self.result_saver.output_dir), output_files)
            
        except Exception as e:
            logger.debug(f"CLI visualization failed: {e}")
        
        # Create final result (HTML/Markdown report generation disabled)
        final_result = ProcessingResult(faces=all_faces, clusters=clusters, stats=stats, config_used=self.config.__dict__)
        
        return final_result
        
    async def cleanup(self) -> None:
        """Pipeline cleanup - release resources properly."""
        try:
            # Cleanup ONNX sessions from embeddings
            if hasattr(self, 'embeddings') and self.embeddings:
                if hasattr(self.embeddings, 'session') and self.embeddings.session:
                    # ONNX sessions are usually handled by the runtime, but we can clear the reference
                    self.embeddings.session = None
                    logger.debug("Cleared ONNX session reference")
            
            # Cleanup detector resources
            if hasattr(self, 'detector') and self.detector:
                # InsightFace detector cleanup
                if hasattr(self.detector, 'app') and self.detector.app:
                    # Clear model references
                    self.detector.app = None
                    logger.debug("Cleared InsightFace app reference")
                
                # Clear any cached data (detector doesn't currently use caching)
                    
            # Clear component references
            self.detector = None
            self.embeddings = None
            self.clusterer = None
            self.result_saver = None
            
            logger.debug("Pipeline cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")
            