"""Enhanced configuration management with validation and environment support."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Processing configuration optimized for speed."""
    use_cuda: bool = True
    model_name: str = "buffalo_l"
    det_size: tuple = (640, 640)
    align_size: int = 112
    batch_size: int = 64  # Increased for better GPU utilization
    max_workers: int = 8  # Increased for better CPU parallelism
    use_flip_tta: bool = False  # Disabled for speed (reduces accuracy slightly)
    max_faces_per_image: int = 50
    max_image_size: int = 1280  # Reduced for faster processing
    min_blur_threshold: float = 100.0
    max_memory_mb: int = 6144  # Increased memory allowance
    max_faces_in_memory: int = 15000  # Increased for larger batches
    enable_batch_processing: bool = True
    max_cache_size: int = 2000  # Increased cache size
    gpu_memory_gb: int = 8


@dataclass
class ClusteringConfig:
    """Enhanced clustering configuration with intelligent parameter selection."""
    algorithm: str = "hybrid"  # Changed default to hybrid
    eps: float = 0.22  # Stricter eps for tighter clusters
    min_samples: int = 2  # Allow 2-face clusters as requested
    metric: str = "cosine"
    use_multipass: bool = True
    high_quality_threshold: float = 0.7
    high_pose_threshold: float = 0.8
    rescue_eps_factor: float = 1.1  # Much stricter rescue (was 1.2)
    final_eps_factor: float = 1.3  # Much stricter final clustering (was 1.6)
    use_scaling: bool = False
    max_cache_size: int = 100
    
    # Graph refinement parameters - much more conservative for strict clusters
    enable_graph_refinement: bool = True
    graph_k: int = 8  # Much smaller neighborhood for stricter clustering
    graph_mutual_k: bool = True
    rescue_tau: float = 0.90  # Very strict rescue threshold
    rescue_margin: float = 0.03  # Very tight margin
    rescue_neighbor_agree: int = 7  # Most neighbors must agree
    propagation_alpha: float = 0.08  # Minimal propagation influence
    propagation_iterations: int = 10  # Fewer iterations for stability
    merge_tau: float = 0.85  # Much stricter merge threshold
    merge_min_edges: int = 12  # Many more edges required for merge
    
    # Hybrid clustering parameters - stricter for tight clusters
    hdbscan_min_cluster_size: int = 2  # Allow 2-face clusters as requested
    hdbscan_min_samples: int = 2  # Match min_samples
    hdbscan_alpha: float = 1.5  # Much higher alpha for very strict clustering
    hdbscan_cluster_selection_epsilon: float = 0.08  # Larger epsilon for stable clusters
    dbscan_noise_rescue_eps_factor: float = 1.05  # Very conservative rescue
    dbscan_noise_rescue_min_samples: int = 2  # Separate min_samples for noise rescue (not tied to MIN_CLUSTER_SIZE)
    dbscan_noise_rescue_min_samples_factor: float = 0.9  # Higher min samples for rescue
    auto_parameter_selection: bool = True
    quality_weighted_clustering: bool = True
    
    # Cluster merging parameters
    merge_similarity_threshold: float = 0.80  # Cosine similarity threshold for cluster merging
    
    # Memory-guided clustering parameters - stricter for known faces
    similarity_threshold: float = 0.80  # Higher threshold for stricter template matching
    auto_update_memory: bool = True
    high_confidence_threshold: float = 0.88  # Higher confidence required
    template_quality_threshold: float = 0.65  # Slightly higher template quality


@dataclass
class QualityConfig:
    """Quality assessment configuration."""
    blur_threshold: float = 100.0
    brightness_min: float = 30.0
    brightness_max: float = 220.0
    contrast_threshold: float = 20.0
    enable_quality_filtering: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create config from dictionary."""
        config = cls()
        
        # Update each section
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
            else:
                logger.warning(f"Unknown config section: {section_name}")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            if hasattr(field_value, '__dataclass_fields__'):
                # Convert dataclass to dict
                result[field_name] = {
                    sub_field: getattr(field_value, sub_field)
                    for sub_field in field_value.__dataclass_fields__
                }
            else:
                result[field_name] = field_value
        
        return result
    
    def validate(self) -> bool:
        """Validate configuration."""
        errors = []
        
        # Validate processing config
        if self.processing.batch_size <= 0:
            errors.append("processing.batch_size must be positive")
        
        if self.processing.max_workers <= 0:
            errors.append("processing.max_workers must be positive")
        
        # Validate clustering config
        if self.clustering.eps <= 0:
            errors.append("clustering.eps must be positive")
        
        if self.clustering.min_samples <= 0:
            errors.append("clustering.min_samples must be positive")
        
        # Note: Path validation removed - paths are handled at runtime
        
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
        
        return True


class ConfigManager:
    """Enhanced configuration manager with environment support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager."""
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[AppConfig] = None
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        # Get the directory where this config.py file is located (utils/)
        current_dir = Path(__file__).parent.parent  # Go up two levels from utils/config.py to optimized/
        
        search_paths = [
            str(current_dir / "config.yaml"),  # Look in optimized/config.yaml first
            "./config.yaml",
            "./config/config.yaml", 
            "./facesorter.yaml",
            os.path.expanduser("~/.facesorter/config.yaml"),
            "/etc/facesorter/config.yaml"
        ]
        
        for path in search_paths:
            if Path(path).exists():
                logger.info(f"Found config file: {path}")
                return path
        
        logger.info("No config file found, using defaults")
        return None
    
    def load_config(self, config_file_path: Optional[str] = None) -> AppConfig:
        """Load configuration from file and environment with caching."""
        # Use provided path or default config path
        config_path_to_use = config_file_path or self.config_path
        
        # Return cached config if available and file hasn't changed
        if hasattr(self, '_cached_config') and hasattr(self, '_cached_config_path'):
            if (self._cached_config_path == config_path_to_use and 
                config_path_to_use and Path(config_path_to_use).exists()):
                try:
                    # Check if file was modified
                    current_mtime = Path(config_path_to_use).stat().st_mtime
                    if hasattr(self, '_cached_config_mtime') and current_mtime == self._cached_config_mtime:
                        return self._cached_config
                except OSError:
                    pass  # File might have been deleted, continue with fresh load
        
        # Start with default config
        config_data = {}
        
        # Load from file if available
        if config_path_to_use and Path(config_path_to_use).exists():
            try:
                with open(config_path_to_use, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config_data.update(file_config)
                        # Cache the modification time
                        self._cached_config_mtime = Path(config_path_to_use).stat().st_mtime
                        logger.info(f"Loaded config from {config_path_to_use}")
            except Exception as e:
                logger.error(f"Failed to load config file {config_path_to_use}: {e}")
        
        # Override with environment variables
        env_overrides = self._load_env_overrides()
        if env_overrides:
            config_data.update(env_overrides)
            logger.info(f"Applied {len(env_overrides)} environment overrides")
        
        # Create config object
        self.config = AppConfig.from_dict(config_data)
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Configuration validation failed")
        
        # Cache the config for future use
        self._cached_config = self.config
        self._cached_config_path = config_path_to_use
        
        return self.config
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        prefix = "FACESORTER_"
        
        # Map environment variables to config paths
        env_mappings = {
            f"{prefix}USE_CUDA": ("processing", "use_cuda", bool),
            f"{prefix}BATCH_SIZE": ("processing", "batch_size", int),
            f"{prefix}MAX_WORKERS": ("processing", "max_workers", int),
            f"{prefix}EPS": ("clustering", "eps", float),
            f"{prefix}MIN_SAMPLES": ("clustering", "min_samples", int),
            f"{prefix}LOG_LEVEL": ("logging", "level", str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Handle boolean conversion
                    if type_func is bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = type_func(value)
                    
                    if section not in overrides:
                        overrides[section] = {}
                    overrides[section][key] = value
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={value}: {e}")
        
        return overrides
    
    def save_config(self, config: AppConfig, path: Optional[str] = None):
        """Save configuration to file."""
        if path:
            save_path = path
        elif self.config_path:
            save_path = self.config_path
        else:
            # Default to optimized folder if no config path found
            current_dir = Path(__file__).parent.parent  # Go up to optimized/
            save_path = str(current_dir / "config.yaml")
        
        try:
            config_dict = config.to_dict()
            
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file."""
        self.config = None
        return self.load_config()


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """Get global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()


def set_config_path(path: str):
    """Set configuration file path."""
    global _config_manager
    _config_manager = ConfigManager(path)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Simple config loader function."""
    config_manager = ConfigManager(config_path)
    return config_manager.load_config()


def reload_config() -> AppConfig:
    """Reload global configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.reload_config()