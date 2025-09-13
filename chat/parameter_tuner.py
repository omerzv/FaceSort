"""Advanced parameter tuning system for the chat interface."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict

from optimized.utils.colors import *
from optimized.utils.config import AppConfig


class ParameterTuner:
    """Advanced parameter tuning and management system."""
    
    def __init__(self, chat_interface):
        self.chat = chat_interface
        
        # Parameter categories with shortcuts and descriptions
        self.parameter_categories = {
            "⚡ Core Clustering": {
                "algorithm": {
                    "shortcuts": ["algo", "alg", "a"],
                    "description": "Clustering algorithm to use",
                    "explanation": "• hybrid: Best overall - combines HDBSCAN + DBSCAN for optimal results\n    • dbscan: Fast traditional clustering, good for uniform face quality\n    • hdbscan: Handles varying densities, better for mixed photo quality",
                    "type": "choice",
                    "choices": ["hybrid", "dbscan", "hdbscan"],
                    "path": "clustering.algorithm"
                },
                "eps": {
                    "shortcuts": ["e", "epsilon"],
                    "description": "Face similarity threshold (cosine distance)",
                    "explanation": "Controls how similar faces must be to group together:\n    • 0.15-0.20: Very strict - only nearly identical faces cluster\n    • 0.22-0.28: Balanced - good for most photo collections\n    • 0.30-0.40: Loose - may group similar-looking different people\n    Lower values = tighter, more accurate clusters",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.8,
                    "path": "clustering.eps"
                },
                "min_samples": {
                    "shortcuts": ["ms", "min", "samples"],
                    "description": "Minimum faces required to form a cluster",
                    "explanation": "Sets minimum cluster size before considering faces as noise:\n    • 1: Keep all faces, even single occurrences\n    • 2-3: Good balance - small clusters but reduces noise\n    • 4+: Only keep people with many photos (strict)",
                    "type": "int",
                    "min": 1,
                    "max": 20,
                    "path": "clustering.min_samples"
                },
                "metric": {
                    "shortcuts": ["met", "mt"],
                    "description": "Distance metric for face comparison",
                    "explanation": "Method for measuring face similarity:\n    • cosine: Recommended - works best with face embeddings\n    • euclidean: Alternative - may work better with normalized data",
                    "type": "choice",
                    "choices": ["cosine", "euclidean"],
                    "path": "clustering.metric"
                }
            },
            "🎯 HDBSCAN Settings": {
                "hdbscan_min_cluster_size": {
                    "shortcuts": ["hcs", "h_size", "hsize"],
                    "description": "HDBSCAN minimum cluster size",
                    "explanation": "Smallest size cluster HDBSCAN will extract:\n    • 2-3: Allow small clusters (couples, small groups)\n    • 4-8: Medium clusters only (families, common groups)\n    • 10+: Large clusters only (very common people)",
                    "type": "int",
                    "min": 2,
                    "max": 50,
                    "path": "clustering.hdbscan_min_cluster_size"
                },
                "hdbscan_alpha": {
                    "shortcuts": ["alpha", "alp"],
                    "description": "HDBSCAN cluster stability parameter",
                    "explanation": "Controls cluster formation strictness:\n    • 0.5-1.0: More permissive, finds more clusters\n    • 1.0-2.0: Balanced stability and sensitivity\n    • 2.0+: Very strict, only highly stable clusters\n    Higher values = more conservative clustering",
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0,
                    "path": "clustering.hdbscan_alpha"
                },
                "hdbscan_cluster_selection_epsilon": {
                    "shortcuts": ["cse", "c_eps"],
                    "description": "HDBSCAN cluster selection threshold",
                    "explanation": "Distance threshold for cluster selection:\n    • 0.0: Use original HDBSCAN selection\n    • 0.05-0.15: Moderate threshold for stability\n    • 0.2+: High threshold, prefers fewer larger clusters\n    Often used to fine-tune cluster boundaries",
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "path": "clustering.hdbscan_cluster_selection_epsilon"
                }
            },
            "🔬 Advanced Parameters": {
                "merge_similarity_threshold": {
                    "shortcuts": ["merge", "mst", "m_thresh"],
                    "description": "Threshold for merging similar clusters",
                    "explanation": "Cosine similarity threshold for combining clusters:\n    • 0.90-0.95: Very conservative - keeps clusters separate\n    • 0.85-0.90: Balanced - merges very similar clusters only\n    • 0.75-0.85: Aggressive - may merge different people\n    • 0.60-0.75: Very loose - high risk of over-merging\n    Higher values = less merging, more clusters",
                    "type": "float",
                    "min": 0.5,
                    "max": 0.99,
                    "path": "clustering.merge_similarity_threshold"
                },
                "dbscan_noise_rescue_eps_factor": {
                    "shortcuts": ["rescue", "r_factor", "rf"],
                    "description": "Noise rescue sensitivity multiplier",
                    "explanation": "Multiplier for eps when rescuing noise faces:\n    • 1.0-1.1: Very strict rescue, few faces recovered\n    • 1.1-1.3: Balanced rescue approach\n    • 1.3-1.5: Generous rescue, may create false clusters\n    Helps recover borderline faces labeled as noise",
                    "type": "float",
                    "min": 1.0,
                    "max": 2.0,
                    "path": "clustering.dbscan_noise_rescue_eps_factor"
                },
                "dbscan_noise_rescue_min_samples": {
                    "shortcuts": ["r_samples", "rs"],
                    "description": "Minimum samples for noise rescue clustering",
                    "explanation": "Min samples used when rescuing noise faces:\n    • 1: Very permissive rescue, saves most noise\n    • 2-3: Balanced rescue, requires some agreement\n    • 4+: Strict rescue, only clear clusters from noise\n    Lower than main min_samples for second-chance clustering",
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "path": "clustering.dbscan_noise_rescue_min_samples"
                }
            },
            "⚙️ Processing Settings": {
                "use_cuda": {
                    "shortcuts": ["cuda", "gpu"],
                    "description": "Enable GPU acceleration for face processing",
                    "explanation": "Use GPU for faster face detection and embeddings:\n    • true: Much faster processing if NVIDIA GPU available\n    • false: CPU-only processing, slower but more compatible\n    Requires CUDA-capable GPU and proper drivers",
                    "type": "bool",
                    "path": "processing.use_cuda"
                },
                "batch_size": {
                    "shortcuts": ["batch", "bs"],
                    "description": "Number of faces processed simultaneously",
                    "explanation": "Faces processed together in each GPU/CPU batch:\n    • 8-16: Conservative, low memory usage\n    • 32-64: Balanced performance and memory\n    • 128+: High performance, needs more GPU/RAM\n    Larger batches = faster processing but more memory",
                    "type": "int",
                    "min": 4,
                    "max": 256,
                    "path": "processing.batch_size"
                },
                "max_workers": {
                    "shortcuts": ["workers", "w"],
                    "description": "Number of parallel processing threads",
                    "explanation": "CPU threads used for parallel image loading:\n    • 1-2: Conservative, low CPU usage\n    • 4-8: Good balance for most systems\n    • 12+: High parallelism for powerful CPUs\n    Should match CPU cores but not exceed them",
                    "type": "int",
                    "min": 1,
                    "max": 32,
                    "path": "processing.max_workers"
                }
            }
        }
        
        # Build reverse lookup for shortcuts
        self.shortcut_to_param = {}
        for category, params in self.parameter_categories.items():
            for param_name, param_info in params.items():
                # Add full name
                self.shortcut_to_param[param_name] = param_name
                # Add shortcuts
                for shortcut in param_info["shortcuts"]:
                    self.shortcut_to_param[shortcut] = param_name
    
    def show_all_parameters(self):
        """Show all available parameters organized by category with detailed explanations."""
        from optimized.utils.typing_animation import type_word_by_word
        
        # Build the complete output as strings first
        lines = []
        lines.append(f"\n{colored('🎛️    Available Parameters', Colors.BRIGHT_MAGENTA, bold=True)}")
        lines.append(f"{colored('═' * 80, Colors.BRIGHT_BLUE)}")
        lines.append(f"{colored('Usage: set <parameter> <value> [--cfg]', Colors.BRIGHT_YELLOW)}  {colored('(e.g., set eps 0.3 --cfg)', Colors.DIM)}")
        lines.append("")
        
        for category, params in self.parameter_categories.items():
            # Category header with separator
            lines.append(f"\n{colored('┌─ ' + category + ' ' + '─' * (75 - len(category)), Colors.BRIGHT_BLUE)}")
            lines.append(f"{colored('│', Colors.BRIGHT_BLUE)}")
            
            for param_name, param_info in params.items():
                current_value = self._get_current_value(param_info["path"])
                shortcuts_str = colored(f"[{', '.join(param_info['shortcuts'])}]", Colors.CYAN)
                
                # Parameter header line - more compact and readable
                param_header = f"{colored('│', Colors.BRIGHT_BLUE)} {colored(param_name, Colors.BRIGHT_GREEN, bold=True)}"
                param_header += f" {shortcuts_str} → {colored(str(current_value), Colors.BRIGHT_CYAN, bold=True)}"
                lines.append(param_header)
                
                # Description line
                lines.append(f"{colored('│', Colors.BRIGHT_BLUE)} {colored(param_info['description'], Colors.WHITE)}")
                
                # Show detailed explanation with better formatting
                if 'explanation' in param_info:
                    lines.append(f"{colored('│', Colors.BRIGHT_BLUE)}")
                    explanation_lines = param_info['explanation'].split('\n')
                    for line in explanation_lines:
                        if line.strip():
                            # Clean up the line and add proper indentation
                            clean_line = line.strip()
                            if clean_line.startswith('•'):
                                lines.append(f"{colored('│', Colors.BRIGHT_BLUE)}   {colored(clean_line, Colors.DIM)}")
                            else:
                                lines.append(f"{colored('│', Colors.BRIGHT_BLUE)}   {colored(clean_line, Colors.BRIGHT_WHITE)}")
                
                lines.append(f"{colored('│', Colors.BRIGHT_BLUE)}")
            
            # Category footer
            lines.append(f"{colored('└' + '─' * 79, Colors.BRIGHT_BLUE)}")
        
        # Type each line word by word
        for line in lines:
            if line.strip():  # Skip empty lines for typing
                type_word_by_word(line)
            else:
                print()  # Print empty lines instantly
    
    def set_parameter(self, param_input: str, value_str: str, save_to_config: bool = False) -> bool:
        """Set a parameter value using name or shortcut."""
        # Find parameter by name or shortcut
        param_name = self.shortcut_to_param.get(param_input.lower())
        if not param_name:
            self.chat.show_error(f"Unknown parameter: '{param_input}'. Use 'tune' to see all parameters.")
            return False
        
        # Find parameter info
        param_info = None
        for category, params in self.parameter_categories.items():
            if param_name in params:
                param_info = params[param_name]
                break
        
        if not param_info:
            self.chat.show_error(f"Parameter info not found for: {param_name}")
            return False
        
        # Validate and convert value
        try:
            value = self._validate_and_convert_value(value_str, param_info)
            if value is None:
                return False
        except ValueError as e:
            self.chat.show_error(str(e))
            return False
        
        # Set the value in config
        old_value = self._get_current_value(param_info["path"])
        success = self._set_config_value(param_info["path"], value)
        
        if success:
            self.chat.show_success(f"Set {colored(param_name, Colors.BRIGHT_GREEN)}: {colored(old_value, Colors.DIM)} → {colored(value, Colors.BRIGHT_CYAN)}")
            
            # Save to config file if requested
            if save_to_config:
                config_saved = self._save_to_config_file(param_info["path"], value)
                if config_saved:
                    self.chat.show_info(f"💾 Saved to config.yaml: {colored(param_name, Colors.BRIGHT_GREEN)} = {colored(value, Colors.BRIGHT_CYAN)}")
                else:
                    self.chat.show_warning("Failed to save to config file")
            
            # Show tips for common parameters
            self._show_parameter_tips(param_name, value)
            return True
        else:
            self.chat.show_error(f"Failed to set parameter: {param_name}")
            return False
    
    def _validate_and_convert_value(self, value_str: str, param_info: Dict) -> Any:
        """Validate and convert a parameter value."""
        param_type = param_info["type"]
        
        if param_type == "bool":
            if value_str.lower() in ["true", "1", "yes", "on", "enable"]:
                return True
            elif value_str.lower() in ["false", "0", "no", "off", "disable"]:
                return False
            else:
                raise ValueError(f"Invalid boolean value: {value_str}. Use true/false, 1/0, yes/no, on/off")
        
        elif param_type == "int":
            try:
                value = int(value_str)
                if "min" in param_info and value < param_info["min"]:
                    raise ValueError(f"Value {value} is below minimum {param_info['min']}")
                if "max" in param_info and value > param_info["max"]:
                    raise ValueError(f"Value {value} is above maximum {param_info['max']}")
                return value
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid integer value: {value_str}")
                raise e
        
        elif param_type == "float":
            try:
                value = float(value_str)
                if "min" in param_info and value < param_info["min"]:
                    raise ValueError(f"Value {value} is below minimum {param_info['min']}")
                if "max" in param_info and value > param_info["max"]:
                    raise ValueError(f"Value {value} is above maximum {param_info['max']}")
                return value
            except ValueError as e:
                if "could not convert" in str(e):
                    raise ValueError(f"Invalid float value: {value_str}")
                raise e
        
        elif param_type == "choice":
            choices = param_info["choices"]
            if value_str.lower() in [c.lower() for c in choices]:
                # Return the properly cased choice
                for choice in choices:
                    if choice.lower() == value_str.lower():
                        return choice
            else:
                raise ValueError(f"Invalid choice: {value_str}. Options: {', '.join(choices)}")
        
        else:
            return value_str
    
    def _get_current_value(self, path: str) -> Any:
        """Get current value from config using dot notation path."""
        config = self.chat.get_config()
        parts = path.split(".")
        value = config
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return "unknown"
        
        return value
    
    def _set_config_value(self, path: str, value: Any) -> bool:
        """Set config value using dot notation path."""
        try:
            config = self.chat.get_config()
            parts = path.split(".")
            target = config
            
            # Navigate to parent object
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    return False
            
            # Set the value
            final_key = parts[-1]
            if hasattr(target, final_key):
                setattr(target, final_key, value)
                return True
            
            return False
        except Exception:
            return False
    
    def _save_to_config_file(self, path: str, value: Any) -> bool:
        """Save parameter value to the existing config.yaml file."""
        try:
            config_manager = self.chat.config_manager
            
            # Get current config and update it
            current_config = config_manager.get_config()
            
            # Navigate to the correct nested location and set value
            parts = path.split(".")
            target = current_config
            
            # Navigate to parent object
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    return False
            
            # Set the value
            final_key = parts[-1]
            if hasattr(target, final_key):
                setattr(target, final_key, value)
            else:
                return False
            
            # Save the updated config to existing file (overwrites)
            config_manager.save_config(current_config)
            
            return True
            
        except Exception as e:
            import traceback
            self.chat.show_error(f"Failed to save config: {str(e)}")
            # Get logger from the standard logging system
            from optimized.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.debug(f"Config save error: {traceback.format_exc()}")
            return False
    
    def _show_parameter_tips(self, param_name: str, value: Any):
        """Show helpful tips for specific parameters."""
        tips = {
            "eps": {
                0.1: "Very strict - only very similar faces will cluster",
                0.2: "Strict clustering - good for high quality photos",
                0.3: "Moderate clustering - balanced approach",
                0.4: "Loose clustering - may group different people",
            },
            "merge_similarity_threshold": {
                0.9: "Very conservative merging - keeps clusters separate",
                0.85: "Good default - merges very similar clusters",
                0.8: "More aggressive merging",
                0.7: "Loose merging - may over-merge different people"
            }
        }
        
        if param_name in tips:
            for threshold, tip in tips[param_name].items():
                if isinstance(value, (int, float)) and abs(value - threshold) < 0.05:
                    self.chat.show_info(f"💡 Tip: {tip}")
                    break
    
    def show_current_settings(self):
        """Show current parameter values in organized format."""
        print(f"\n{colored('🎛️  Current Settings:', Colors.BRIGHT_MAGENTA)}")
        
        for category, params in self.parameter_categories.items():
            print(f"\n{colored(category, Colors.BRIGHT_BLUE)}")
            
            for param_name, param_info in params.items():
                current_value = self._get_current_value(param_info["path"])
                shortcuts = colored(f"[{param_info['shortcuts'][0]}]", Colors.DIM)
                
                print(f"  {colored(param_name, Colors.BRIGHT_GREEN)} {shortcuts}: {colored(current_value, Colors.BRIGHT_CYAN)}")
        print()