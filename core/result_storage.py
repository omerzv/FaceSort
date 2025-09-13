"""Result storage with memory management and async I/O."""

import asyncio
import aiofiles
import aiofiles.os
from pathlib import Path
import json
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Union
import logging
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
import pickle
import re
import gzip
from datetime import datetime

from optimized.utils.logger import get_logger
from optimized.utils.constants import (
    DEFAULT_JPEG_QUALITY, COMPRESSION_THRESHOLD_MB, DEFAULT_MAX_MEMORY_USAGE_MB
)

logger = get_logger(__name__)


class ResultSaver:
    """Result saver with memory management and async I/O."""
    
    def __init__(self, output_dir: str, max_memory_usage: int = DEFAULT_MAX_MEMORY_USAGE_MB * 1024 * 1024) -> None:
        """Initialize result saver."""
        self.base_output_dir = Path(output_dir)
        
        # Create session-specific subdirectory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / f"session_{timestamp}"
        
        self.max_memory_usage = max_memory_usage
        self.current_memory_usage = 0
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"OptimizedResultSaver ready: {self.output_dir}")
    
    async def save_faces_by_cluster_async(self, faces: List, clustering_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Save face crops organized by cluster with person names (async version)."""
        start_time = time.time()
        
        # Group faces by cluster
        clusters = {}
        for face in faces:
            cluster_id = face.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(face)
        
        # Get cluster to person mapping from clustering info
        cluster_to_person = {}
        if clustering_info and 'cluster_to_person' in clustering_info:
            cluster_to_person = clustering_info['cluster_to_person']
        
        # Save clusters in parallel
        tasks = []
        for cluster_id, cluster_faces in clusters.items():
            if cluster_id == -1:
                task = self._save_noise_cluster_async(cluster_faces)
            else:
                # Get person name if available
                person_name = cluster_to_person.get(cluster_id, None)
                task = self._save_regular_cluster_async(cluster_id, cluster_faces, person_name)
            tasks.append(task)
        
        # Wait for all saves to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect statistics
        total_faces_saved = 0
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif isinstance(result, dict):
                total_faces_saved += result.get('faces_saved', 0)
        
        save_time = time.time() - start_time
        
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during saving")
        
        return {
            'total_faces_saved': total_faces_saved,
            'total_clusters': len(clusters),
            'save_time': save_time,
            'errors': errors
        }
    
    async def _save_regular_cluster_async(self, cluster_id: int, faces: List, person_name: Optional[str] = None) -> Dict[str, Any]:
        """Save a regular cluster asynchronously with person name if known."""
        # Create directory name - use person name directly if available
        if person_name:
            # Use just the person name as folder name
            safe_name = self._sanitize_filename(person_name)
            cluster_dir = self.output_dir / safe_name
        else:
            # For unknown persons, use numbered clusters
            cluster_dir = self.output_dir / f"unknown_person_{cluster_id:03d}"
        
        await aiofiles.os.makedirs(cluster_dir, exist_ok=True)
        
        # Optimized batching based on available memory and face count
        batch_size = min(100, max(20, len(faces) // 10))  # Larger batches for speed
        faces_saved = 0
        
        for i in range(0, len(faces), batch_size):
            batch = faces[i:i + batch_size]
            
            # Save batch
            batch_result = await self._save_face_batch_async(batch, cluster_dir)
            faces_saved += batch_result['faces_saved']
            
            # Minimal async yield - reduce I/O overhead
            if i % 200 == 0:  # Only yield every 200 faces to reduce overhead
                await asyncio.sleep(0.001)
        
        # Save cluster metadata with person name
        metadata = {
            'cluster_id': cluster_id,
            'person_name': person_name,
            'face_count': len(faces),
            'avg_quality': float(np.mean([f.quality_score for f in faces])),
            'cluster_type': 'known' if person_name else 'unknown',
            'created_at': time.time()
        }
        
        metadata_path = cluster_dir / "cluster_info.json"
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        return {'faces_saved': faces_saved, 'cluster_id': cluster_id, 'person_name': person_name}
    
    async def _save_noise_cluster_async(self, faces: List) -> Dict[str, Any]:
        """Save noise faces asynchronously."""
        noise_dir = self.output_dir / "noise"
        await aiofiles.os.makedirs(noise_dir, exist_ok=True)
        
        # Save in batches
        batch_size = 30  # Smaller batch for noise faces
        faces_saved = 0
        
        for i in range(0, len(faces), batch_size):
            batch = faces[i:i + batch_size]
            batch_result = await self._save_face_batch_async(batch, noise_dir)
            faces_saved += batch_result['faces_saved']
            
            await asyncio.sleep(0.01)
        
        return {'faces_saved': faces_saved, 'cluster_id': -1}
    
    async def _save_face_batch_async(self, faces: List, output_dir: Path) -> Dict[str, Any]:
        """Save a batch of faces with optimized filesystem writes."""
        
        def save_batch_sync(faces_batch) -> int:
            """Synchronous batch save for ThreadPoolExecutor."""
            saved_count = 0
            # Common JPEG parameters for all saves
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 92, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            
            for face in faces_batch:
                try:
                    # Sanitize filename
                    original_name = Path(face.image_path).stem
                    sanitized_name = re.sub(r'[^\w\-_\.]', '_', original_name)
                    face_filename = f"{sanitized_name}_face_{face.id}.jpg"
                    face_path = output_dir / face_filename
                    
                    # Convert crop to uint8 BGR
                    if hasattr(face, 'crop') and face.crop is not None:
                        crop = face.crop
                        if crop.max() <= 1.0:
                            crop_uint8 = (crop * 255).astype(np.uint8)
                        else:
                            crop_uint8 = crop.astype(np.uint8)
                        
                        # Convert RGB to BGR for OpenCV
                        if len(crop_uint8.shape) == 3:
                            crop_bgr = cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2BGR)
                        else:
                            crop_bgr = crop_uint8
                        
                        # Save with optimized JPEG parameters
                        from optimized.utils.silence_opencv import suppress_opencv_warnings
                        with suppress_opencv_warnings():
                            if cv2.imwrite(str(face_path), crop_bgr, jpeg_params):
                                saved_count += 1
                except Exception as e:
                    logger.debug(f"Failed to save face {face.id}: {e}")
            
            return saved_count
        
        # Use ThreadPoolExecutor for batch filesystem writes
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="FaceSaver") as executor:
            # Split into sub-batches for parallel writing
            batch_size = min(50, len(faces))  # Larger batches for efficiency
            tasks = []
            
            for i in range(0, len(faces), batch_size):
                sub_batch = faces[i:i + batch_size]
                task = loop.run_in_executor(executor, save_batch_sync, sub_batch)
                tasks.append(task)
            
            # Wait for all batch writes to complete
            results = await asyncio.gather(*tasks)
            total_saved = sum(results)
        
        return {'faces_saved': total_saved}
        
        async def save_single_face_legacy(face) -> bool:
            """Save a single face crop."""
            try:
                # Generate unique filename with path hash to prevent collisions
                source_path = Path(face.image_path)
                source_name = source_path.stem
                # Use the last part of the path + face id for uniqueness
                parent_name = source_path.parent.name if source_path.parent.name != '.' else ''
                if parent_name:
                    face_filename = f"{parent_name}_{source_name}_face{face.id}.jpg"
                else:
                    face_filename = f"{source_name}_face{face.id}.jpg"
                face_path = output_dir / face_filename
                
                # Skip if already exists (prevents duplicates)
                if face_path.exists():
                    logger.debug(f"Face already saved, skipping: {face_filename}")
                    return True
                
                # Load or use cached crop
                if hasattr(face, 'crop') and face.crop is not None:
                    crop = face.crop
                elif hasattr(face, 'temp_crop_path') and face.temp_crop_path is not None:
                    # Load from temporary crop path
                    temp_path = Path(face.temp_crop_path)
                    if temp_path.exists():
                        from optimized.utils.silence_opencv import suppress_opencv_warnings
                        
                        with suppress_opencv_warnings():
                            crop = cv2.imread(str(temp_path))
                        
                        if crop is None:
                            logger.debug(f"Failed to load crop from {temp_path}")
                            return False
                        # Convert BGR to RGB (cv2 loads as BGR)
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    else:
                        logger.debug(f"Temp crop path does not exist: {temp_path}")
                        return False
                else:
                    # If crop not in memory, we'd need to regenerate it
                    # For now, skip faces without crops
                    logger.debug(f"Skipping face {face.id} - no crop data")
                    return False
                
                # Convert crop to uint8 and BGR for saving
                if crop.dtype == np.float32:
                    crop_uint8 = (crop * 255).astype(np.uint8)
                else:
                    crop_uint8 = crop
                
                # Convert RGB to BGR for OpenCV
                if crop_uint8.shape[2] == 3:
                    crop_bgr = cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2BGR)
                else:
                    crop_bgr = crop_uint8
                
                # Save using OpenCV in executor (CPU-bound) 
                def write_image_silent(path, image, params):
                    """Wrapper to suppress OpenCV warnings in executor."""
                    from optimized.utils.silence_opencv import suppress_opencv_warnings
                    with suppress_opencv_warnings():
                        return cv2.imwrite(path, image, params)
                
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    self.executor,
                    write_image_silent,
                    str(face_path),
                    crop_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY]
                )
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to save face {face.id}: {e}")
                return False
        
        # Save all faces in batch concurrently
        tasks = [save_single_face(face) for face in faces]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful saves
        successful_saves = sum(1 for result in results if result is True)
        
        return {'faces_saved': successful_saves}
    
    async def save_results_json_async(self, results, filename: str = "results.json"):
        """Save processing results as JSON asynchronously."""
        results_path = self.output_dir / filename
        
        # Prepare serializable results
        serializable_results = self._prepare_serializable_results(results)
        
        # Compress if large
        data = json.dumps(serializable_results, indent=2)
        
        if len(data) > COMPRESSION_THRESHOLD_MB * 1024 * 1024:
            # Save compressed
            compressed_path = self.output_dir / f"{filename}.gz"
            
            def compress_and_save():
                with gzip.open(compressed_path, 'wt') as f:
                    f.write(data)
                return compressed_path
            
            loop = asyncio.get_event_loop()
            saved_path = await loop.run_in_executor(self.executor, compress_and_save)
            
            logger.info(f"Results saved compressed: {saved_path}")
        else:
            # Save uncompressed
            async with aiofiles.open(results_path, 'w') as f:
                await f.write(data)
            
            logger.info(f"Results saved: {results_path}")
    
    def _prepare_serializable_results(self, results) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        
        def convert_face(face):
            """Convert face object to dict."""
            return {
                'id': face.id,
                'image_path': face.image_path,
                'cluster_id': face.cluster_id,
                'quality_score': float(face.quality_score),
                'blur_var': float(face.blur_var),
                'mean_gray': float(face.mean_gray),
                'pose_confidence': float(face.pose_confidence),
                'd2centroid': float(face.d2centroid),
                'pca3': face.pca3.tolist() if hasattr(face, 'pca3') and face.pca3 is not None else None
            }
        
        def convert_cluster(cluster_info):
            """Convert cluster info to dict."""
            return {
                'cluster_id': cluster_info.cluster_id,
                'face_count': cluster_info.face_count,
                'avg_quality': float(cluster_info.avg_quality),
                'avg_blur': float(cluster_info.avg_blur),
                'avg_brightness': float(cluster_info.avg_brightness),
                'avg_distance': float(cluster_info.avg_distance),
                'faces': [convert_face(face) for face in cluster_info.faces],
                'best_face_id': cluster_info.best_face.id if cluster_info.best_face else None,
                'worst_face_id': cluster_info.worst_face.id if cluster_info.worst_face else None
            }
        
        return {
            'faces': [convert_face(face) for face in results.faces],
            'clusters': {
                str(cluster_id): convert_cluster(cluster_info) 
                for cluster_id, cluster_info in results.clusters.items()
            },
            'stats': {
                'total_images': results.stats.total_images,
                'total_faces': results.stats.total_faces,
                'clusters_found': results.stats.clusters_found,
                'noise_faces': results.stats.noise_faces,
                'total_time': results.stats.total_time,
                'detection_time': results.stats.detection_time,
                'embedding_time': results.stats.embedding_time,
                'clustering_time': results.stats.clustering_time,
                'avg_quality': results.stats.avg_quality,
                'faces_per_second': results.stats.faces_per_second
            },
            'config': results.config_used,
            'timestamp': time.time()
        }
    
    async def cleanup_temp_files(self):
        """Clean up temporary files."""
        temp_dir = self.output_dir / "temp_crops"
        if temp_dir.exists():
            def remove_temp():
                shutil.rmtree(temp_dir)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, remove_temp)
            
            logger.debug("Temporary files cleaned up")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        
        def calculate_stats():
            stats = {
                'output_dir': str(self.output_dir),
                'total_size_mb': 0,
                'cluster_count': 0,
                'total_files': 0,
                'file_types': {}
            }
            
            if not self.output_dir.exists():
                return stats
            
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    stats['total_size_mb'] += size / (1024 * 1024)
                    stats['total_files'] += 1
                    
                    # Count file types
                    ext = file_path.suffix.lower()
                    if ext not in stats['file_types']:
                        stats['file_types'][ext] = 0
                    stats['file_types'][ext] += 1
                
                elif file_path.is_dir() and file_path.name.startswith('cluster_'):
                    stats['cluster_count'] += 1
            
            return stats
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, calculate_stats)
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a name to be safe for filesystem use."""
        import re
        # Replace invalid characters with underscores
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove extra spaces and replace with underscores
        safe_name = re.sub(r'\s+', '_', safe_name.strip())
        # Limit length
        safe_name = safe_name[:50]
        return safe_name
    
    async def save_results_async(self, result):
        """Save complete results including faces and metadata."""
        # Get clustering info if available
        clustering_info = getattr(result, 'clustering_info', None)
        
        # Save face crops organized by cluster with person names
        await self.save_faces_by_cluster_async(result.faces, clustering_info)
        
        # Save JSON results
        await self.save_results_json_async(result)
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.cleanup_temp_files()
        self.executor.shutdown(wait=True)
        logger.debug("Storage cleanup completed")
