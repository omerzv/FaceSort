"""Face detection with memory management and async support."""

import asyncio
import cv2
import numpy as np
import insightface
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Iterator, Any
from pathlib import Path
import logging
import time
import os


from optimized.utils.logger import get_logger
from optimized.utils.config import ProcessingConfig
from optimized.utils.constants import (
    DEFAULT_DETECTION_THRESHOLD, DEFAULT_EMBEDDING_DIMENSION, SUPPORTED_IMAGE_EXTENSIONS
)
from optimized.utils.output_suppression import suppress_detector_init_output

# Optimize OpenCV for multithreaded usage
cv2.setNumThreads(0)  # Disable OpenCV internal threads to avoid oversubscription
try:
    cv2.ocl.setUseOpenCL(False)  # Disable OpenCL to prevent GPU conflicts
except AttributeError:
    pass  # Older OpenCV versions might not have this

logger = get_logger(__name__)


class FaceDetector:
    """Face detector with memory management and performance improvements."""
    
    def __init__(self, config: ProcessingConfig) -> None:
        """Initialize face detector."""
        self.config = config
        self.app = None
        self._setup_detector()
    
    def _setup_detector(self) -> None:
        """Initialize InsightFace detector with robust provider fallbacks."""
        try:
            logger.debug(f"Initializing InsightFace: {self.config.model_name}")
            
            # Build provider cascade with fallbacks
            providers = []
            if self.config.use_cuda:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            # Use selective output suppression during initialization
            with suppress_detector_init_output():
                # Initialize with provider fallbacks
                self.app = insightface.app.FaceAnalysis(
                    name=self.config.model_name,
                    providers=providers
                )
                
                ctx_id = 0 if self.config.use_cuda else -1
                try:
                    self.app.prepare(
                        ctx_id=ctx_id, 
                        det_size=self.config.det_size,
                        det_thresh=DEFAULT_DETECTION_THRESHOLD
                    )
                except Exception as gpu_error:
                    if self.config.use_cuda:
                        logger.warning(f"CUDA initialization failed: {gpu_error}, falling back to CPU")
                        # Retry with CPU only
                        self.app = insightface.app.FaceAnalysis(
                            name=self.config.model_name,
                            providers=['CPUExecutionProvider']
                        )
                        self.app.prepare(
                            ctx_id=-1, 
                            det_size=self.config.det_size,
                            det_thresh=DEFAULT_DETECTION_THRESHOLD
                        )
                    else:
                        raise
            
            # Determine actual provider used
            actual_provider = "CPU"
            if self.app and hasattr(self.app, 'models'):
                for model in self.app.models.values():
                    if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                        providers = model.session.get_providers()
                        if 'CUDAExecutionProvider' in providers:
                            actual_provider = "CUDA"
                            break
            
            logger.debug(f"FaceDetector ready ({actual_provider})")
            
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {e}")
            raise RuntimeError(f"Face detector initialization failed: {e}")
    

    def detect_batch(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[str], List[np.ndarray]]:
        """Optimized batch face detection with GPU batching."""
        if not image_paths:
            return [], [], []
        
        logger.debug(f"Processing batch: {len(image_paths)} images")
        start_time = time.time()
        
        # Load images with parallel loading
        loaded_images = self._load_images_parallel(image_paths)
        
        # Filter valid images and prepare for batch processing
        valid_images = [(bgr, path) for bgr, path in loaded_images if bgr is not None]
        
        # Report skipped files if any
        skipped_count = len(image_paths) - len(valid_images)
        if skipped_count > 0:
            logger.debug(f"Skipped {skipped_count} corrupted/unreadable images in batch")
        
        if not valid_images:
            return [], [], []
        
        # GPU batch processing - process multiple images at once
        if self.config.use_cuda and len(valid_images) > 1:
            all_crops, all_owners, all_embeddings = self._detect_batch_gpu(valid_images)
        else:
            # Fallback to sequential processing
            all_crops = []
            all_owners = []
            all_embeddings = []
            
            for bgr, path in valid_images:
                crops, owners, embeddings = self._detect_and_align(bgr, path)
                all_crops.extend(crops)
                all_owners.extend(owners)
                all_embeddings.extend(embeddings)
                
                # Clear image from memory immediately
                del bgr
        
        processing_time = time.time() - start_time
        logger.debug(f"Batch processed: {len(all_crops)} faces in {processing_time:.2f}s")
        
        return all_crops, all_owners, all_embeddings
    
    def _detect_batch_gpu(self, valid_images: List[Tuple[np.ndarray, str]]) -> Tuple[List[np.ndarray], List[str], List[np.ndarray]]:
        """GPU-optimized batch face detection processing multiple images simultaneously."""
        all_crops = []
        all_owners = []
        all_embeddings = []
        
        # Process in smaller sub-batches to manage GPU memory
        gpu_batch_size = min(8, len(valid_images))  # Conservative GPU batch size
        
        for i in range(0, len(valid_images), gpu_batch_size):
            batch = valid_images[i:i + gpu_batch_size]
            
            # Convert images to RGB for batch processing
            rgb_batch = []
            paths_batch = []
            
            for bgr, path in batch:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_batch.append(rgb)
                paths_batch.append(path)
            
            # Process batch on GPU
            try:
                for j, (rgb, path) in enumerate(zip(rgb_batch, paths_batch)):
                    # Individual detection (InsightFace doesn't support true batching)
                    faces = self.app.get(rgb, max_num=self.config.max_faces_per_image) if self.app else []
                    if faces is None:
                        faces = []
                    
                    # Process detected faces
                    for face_idx, face in enumerate(faces):
                        if hasattr(face, 'kps') and face.kps is not None:
                            try:
                                from insightface.utils import face_align
                                warped = face_align.norm_crop(
                                    rgb, face.kps, image_size=self.config.align_size
                                )
                                
                                crop = warped.astype(np.float32) / 255.0
                                all_crops.append(crop)
                                all_owners.append(f"{path}#{face_idx}")
                                all_embeddings.append(face.embedding if hasattr(face, 'embedding') else np.zeros(512, dtype=np.float32))
                                
                            except Exception:
                                continue
                
            except Exception as e:
                logger.debug(f"GPU batch processing failed: {e}, falling back to sequential")
                # Fallback to sequential processing for this batch
                for bgr, path in batch:
                    crops, owners, embeddings = self._detect_and_align(bgr, path)
                    all_crops.extend(crops)
                    all_owners.extend(owners)
                    all_embeddings.extend(embeddings)
            
            # Clear batch from memory
            for bgr, _ in batch:
                del bgr
        
        return all_crops, all_owners, all_embeddings
    
    def _detect_and_align(self, bgr_image: np.ndarray, image_path: str) -> Tuple[List[np.ndarray], List[str], List[np.ndarray]]:
        """Face detection and alignment with pre-filtering gate."""
        try:
            # Convert BGR to RGB for InsightFace
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Fast Haar cascade pre-filter to skip obvious non-face images
            preview_w = 160
            h, w = rgb.shape[:2]
            if w > preview_w:
                scale = preview_w / float(w)
                rgb_small = cv2.resize(rgb, (preview_w, max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
            else:
                rgb_small = rgb
            
            gray_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
            haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(haarcascade_path)
            haar_faces = cascade.detectMultiScale(gray_small, 1.1, 2, minSize=(24, 24))
            
            if len(haar_faces) == 0:
                # No faces detected by Haar cascade - skip expensive InsightFace detection
                return [], [], []
            
            # Detect faces with configured limit using InsightFace
            max_faces = self.config.max_faces_per_image
            if self.app is None:
                logger.warning("Face detector not initialized, skipping detection")
                faces = []
            else:
                faces = self.app.get(rgb, max_num=max_faces)
                if faces is None:
                    faces = []
            
            crops = []
            owners = []
            embeddings = []
            
            # Process detected faces with size filtering
            min_box_size = 40  # Minimum face box size in pixels
            for face_idx, face in enumerate(faces):
                if hasattr(face, 'kps') and face.kps is not None and hasattr(face, 'bbox'):
                    # Early reject tiny face boxes
                    x1, y1, x2, y2 = map(int, face.bbox[:4])
                    if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                        continue  # Skip tiny faces that won't yield good embeddings
                    
                    try:
                        from insightface.utils import face_align
                        warped = face_align.norm_crop(
                            rgb, face.kps, image_size=self.config.align_size
                        )
                        
                        # Skip quality check for speed - do it later in batch
                        crop = warped.astype(np.float32) / 255.0
                        crops.append(crop)
                        owners.append(f"{image_path}#{face_idx}")
                        # Use the embedding from the face object that was already computed
                        embeddings.append(face.embedding if hasattr(face, 'embedding') else np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32))
                        
                    except (ValueError, AttributeError, RuntimeError) as e:
                        logger.debug(f"Face processing failed: {e}")
                        continue  # Skip failed face but continue processing
                    except KeyboardInterrupt:
                        raise  # Don't catch user interrupts
                    except SystemExit:
                        raise  # Don't catch system exits
            
            return crops, owners, embeddings
            
        except Exception as e:
            # Minimal error logging
            logger.debug(f"Face detection failed: {Path(image_path).name}")
            return [], [], []
    
    def _load_images_parallel(self, image_paths: List[str]) -> List[Tuple[Optional[np.ndarray], str]]:
        """Parallel image loading with CPU core-aware sizing."""
        
        # CPU core-aware thread pool sizing
        import os
        available_cores = os.cpu_count() or 4
        optimal_workers = min(self.config.max_workers, max(2, available_cores), len(image_paths))
        max_workers = optimal_workers
        
        def load_single(path: str) -> Tuple[Optional[np.ndarray], str]:
            try:
                # Use reduced JPEG decode for large images (major speedup)
                from optimized.utils.silence_opencv import suppress_opencv_warnings
                
                path_obj = Path(path)
                suffix = path_obj.suffix.lower()
                
                with suppress_opencv_warnings():
                    try:
                        if suffix in {'.jpg', '.jpeg'}:
                            # Read file data for reduced JPEG decode
                            with open(path, 'rb') as f:
                                data = np.frombuffer(f.read(), dtype=np.uint8)
                            
                            # Skip obviously corrupted files (too small)
                            if len(data) < 100:
                                logger.debug(f"Skipping tiny/empty JPEG: {path}")
                                return None, path
                            
                            # Try reduced decode first for large JPEGs with error handling
                            img = None
                            try:
                                img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)  # ~1/2 resolution
                                if img is not None and max(img.shape[:2]) > 1600:
                                    # Very large images - decode at 1/4 resolution
                                    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)  # ~1/4 resolution
                            except Exception as decode_error:
                                logger.debug(f"JPEG decode failed for {path}: {decode_error}")
                                img = None
                            
                            # Fallback to standard decode if reduced failed
                            if img is None:
                                try:
                                    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                                except Exception as fallback_error:
                                    logger.debug(f"All JPEG decode attempts failed for {path}: {fallback_error}")
                                    bgr = None
                            else:
                                bgr = img
                        else:
                            # Non-JPEG files use standard loading with error handling
                            try:
                                bgr = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                            except Exception as read_error:
                                logger.debug(f"Image read failed for {path}: {read_error}")
                                bgr = None
                                
                    except Exception as general_error:
                        logger.debug(f"General image loading error for {path}: {general_error}")
                        bgr = None
                
                if bgr is None:
                    return None, path
                
                # Additional resize if still too large
                h, w = bgr.shape[:2]
                max_size = min(self.config.max_image_size, 1280)
                
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                return bgr, path
                
            except Exception as e:
                logger.debug(f"Image load failed {path}: {e}")
                return None, path
        
        # Use maximum workers for speed
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ImageLoader") as executor:
            future_to_path = {
                executor.submit(load_single, path): path 
                for path in image_paths
            }
            
            results = []
            for future in as_completed(future_to_path):
                try:
                    bgr, path = future.result()
                    results.append((bgr, path))
                except Exception as e:
                    path = future_to_path[future]
                    logger.warning(f"Failed to load image: {Path(path).name} ({e})")
                    results.append((None, path))
        
        # Sort to maintain order (required for consistent processing)
        path_to_result = {path: (bgr, path) for bgr, path in results}
        return [path_to_result.get(path, (None, path)) for path in image_paths]

    
    
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        
        # Force cleanup of InsightFace resources
        if self.app:
            # Clear model caches if available
            try:
                if hasattr(self.app, 'models'):
                    for model in self.app.models.values():
                        if hasattr(model, 'session'):
                            # Clear ONNX session cache
                            del model.session
            except Exception as e:
                logger.debug(f"Cleanup warning: {e}")
        
        logger.debug("Face detector cleanup completed")


def iter_image_files(root_dir: str, 
                    extensions: Optional[set] = None) -> Iterator[str]:
    """Image file iteration."""
    if extensions is None:
        extensions = SUPPORTED_IMAGE_EXTENSIONS
    
    root_path = Path(root_dir)
    if not root_path.exists():
        logger.error(f"Input directory does not exist: {root_dir}")
        return
    
    # Use pathlib for better performance
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            yield str(file_path)


def iter_image_batches(root_dir: str, batch_size: int = 32,
                      max_files: Optional[int] = None) -> Iterator[List[str]]:
    """Image batch iteration with optional file limit."""
    batch = []
    file_count = 0
    
    for image_path in iter_image_files(root_dir):
        batch.append(image_path)
        file_count += 1
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
        
        # Optional file limit for testing/development
        if max_files and file_count >= max_files:
            break
    
    # Yield remaining files
    if batch:
        yield batch


def count_images(root_dir: str) -> int:
    """Image counting."""
    supported_extensions = SUPPORTED_IMAGE_EXTENSIONS
    root_path = Path(root_dir)
    
    if not root_path.exists():
        return 0
    
    # Use generator expression for memory efficiency
    return sum(1 for f in root_path.rglob("*") 
              if f.is_file() and f.suffix.lower() in supported_extensions)