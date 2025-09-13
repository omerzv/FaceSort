"""Main entry point for the optimized face clustering system - Unified Interface."""

import asyncio
import argparse
import sys
import logging
import warnings
from pathlib import Path

# Set up proper encoding for Windows console - comprehensive fix
import os
if sys.platform.startswith('win'):
    import locale
    # Set environment variable to force UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
    
    try:
        # Try to set console to UTF-8 (Python 3.7+)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore
    except (AttributeError, ValueError):
        try:
            # Fallback for older Python versions
            import codecs
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            # Final fallback - disable all Unicode output
            import re
            # Filter out any non-ASCII characters from all output
            pass

# Add parent directory to Python path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimized.utils.config import AppConfig, ConfigManager
from optimized.utils.logger import setup_logging
from optimized.utils.colors import *
from optimized.chat.base_interface import BaseChatInterface
from optimized.chat.command_processor import CommandProcessor
from optimized.chat.tutorial_manager import TutorialManager


class FaceClusteringChat(BaseChatInterface):
    """Main face clustering chat interface."""
    
    def __init__(self):
        # Suppress logging and warnings
        logging.getLogger().setLevel(logging.CRITICAL)
        warnings.filterwarnings('ignore')
        
        # Suppress OpenCV/JPEG warnings
        from optimized.utils.silence_opencv import setup_opencv_silence
        setup_opencv_silence()
        
        super().__init__()
        
        self.command_processor = CommandProcessor(self)
        self.tutorial_manager = TutorialManager(self)
    
    def welcome(self):
        """Show welcome message."""
        self.clear_screen()
        print("Starting Face Clustering Assistant...")
        
        self.show_banner()
        self.show_assistant_message("Welcome to Face Clustering!")
        print()
        
        self.show_assistant_message("I'm here to help you organize photos by faces.")
        self.show_info("Type 'help' for commands or 'tutorial' for a guided walkthrough.")
        print()
    
    async def chat_loop(self):
        """Main chat interaction loop."""
        self.welcome()
        
        while True:
            try:
                user_input = self.show_user_prompt()
                
                # Special commands that don't go through the processor
                if user_input.lower() == 'tutorial':
                    self.tutorial_manager.show_tutorial()
                    continue
                elif user_input.lower() == 'quickstart':
                    self.tutorial_manager.show_quick_start()
                    continue
                
                # Process command
                should_continue = await self.command_processor.process_command(user_input)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n")
                self.show_assistant_message("Use 'exit' to quit.")
            except EOFError:
                break
            except Exception as e:
                self.show_error(f"Unexpected error: {str(e)}")


def setup_quiet_logging():
    """Setup minimal logging for CLI."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('optimized').setLevel(logging.WARNING)


def show_progress(stage, current=None, total=None):
    """Show colorful progress updates."""
    if current is not None and total is not None and total > 0:
        progress_bar = create_progress_bar(current, total, width=30)
        progress_text = colored(f"({current:,}/{total:,})", Colors.WHITE)
        print(f"\r   {processing(stage)}: {progress_bar} {progress_text}", end='', flush=True)
    else:
        print(f"\r   {processing(stage)}...", end='', flush=True)


async def run_cli_processing(args, config=None):
    """Run CLI-based face processing."""
    import time
    from datetime import datetime
    
    # Print beautiful header
    print(create_banner("Face Clustering System v2.0", 60))
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(error(f"Input directory '{args.input}' does not exist"))
        print(info("Tip: Provide a valid path to a directory containing images"))
        return
    
    if not any(input_path.glob("*")):
        print(error(f"Input directory '{args.input}' is empty"))
        return
    
    # Show configuration
    print(create_section_header("Configuration"))
    print(format_config("Input", args.input))
    print(format_config("Output", args.output))
    print(format_config("Config", args.config or 'default'))
    
    # Load configuration (use passed config if available, otherwise load from file)
    if config is None:
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
    
    # Override with CLI arguments
    if args.cuda is not None:
        config.processing.use_cuda = args.cuda
    if args.batch_size:
        config.processing.batch_size = args.batch_size
    if args.eps:
        config.clustering.eps = args.eps
    if args.min_samples:
        config.clustering.min_samples = args.min_samples
    
    # Show processing settings
    print(create_section_header("Processing Settings"))
    mode_text = "GPU Accelerated" if config.processing.use_cuda else "CPU Mode"
    mode_color = Colors.BRIGHT_GREEN if config.processing.use_cuda else Colors.BRIGHT_YELLOW
    print(format_config("Mode", colored(mode_text, mode_color, bold=True)))
    print(format_config("Batch Size", config.processing.batch_size))
    print(format_config("Algorithm", config.clustering.algorithm))
    print(format_config("Similarity (eps)", config.clustering.eps))
    print(format_config("Min Samples", config.clustering.min_samples))
    
    # Setup quiet logging for production
    setup_quiet_logging()
    
    # Create simple pipeline  
    from optimized.core.face_pipeline import FacePipeline
    pipeline = FacePipeline(config)
    start_time = time.time()
    
    try:
        # Process faces with clean progress
        print(create_section_header("Processing Images"))
        result = await pipeline.process(
            input_dir=args.input,
            output_dir=args.output,
            progress_callback=show_progress
        )
        
        # Show completion
        total_time = time.time() - start_time
        print(f"\n{success(f'Processing Complete! ({total_time:.1f}s)')}")
        
        # Show statistics
        print(create_section_header("Results Summary"))
        print(format_stat("Images processed", result.stats.total_images))
        print(format_stat("Faces detected", result.stats.total_faces))
        print(format_stat("Clusters created", len(result.clusters)))
        print(format_stat("Processing speed", f"{result.stats.faces_per_second:.1f} faces/sec"))
        
        if result.stats.avg_quality > 0:
            print(format_stat("Average quality", f"{result.stats.avg_quality:.3f}"))
        
        print(f"\n{info('Output saved to')}: {colored(args.output, Colors.BRIGHT_CYAN)}")
        
        # Show cluster breakdown
        if len(result.clusters) > 0:
            print(create_section_header("Top Clusters"))
            sorted_clusters = sorted(result.clusters.items(), key=lambda x: x[1].face_count, reverse=True)
            for i, (cluster_id, cluster_info) in enumerate(sorted_clusters[:5]):
                cluster_name = colored(f"Cluster {cluster_id:03d}", Colors.BRIGHT_MAGENTA)
                face_count = colored(f"{cluster_info.face_count:3d} faces", Colors.BRIGHT_GREEN)
                print(f"  - {cluster_name}: {face_count}")
            if len(sorted_clusters) > 5:
                remaining = colored(f"{len(sorted_clusters) - 5} more clusters", Colors.DIM)
                print(f"  - ... and {remaining}")
        
        print(colored("=" * 60, Colors.BRIGHT_BLUE))
        
        # Cleanup
        await pipeline.cleanup()
        
    except KeyboardInterrupt:
        print(f"\n\n{warning('Processing interrupted by user')}")
        await pipeline.cleanup()
    except Exception as e:
        print(f"\n\n{error(f'Processing failed: {str(e)}')}")
        await pipeline.cleanup()
        sys.exit(1)


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Face Clustering Assistant - Unified interface for organizing photos by faces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run without arguments for interactive chat mode, or provide input/output for direct CLI processing."
    )
    
    # Optional CLI processing arguments
    parser.add_argument('input', nargs='?', help='Input directory containing images (optional - enables CLI mode)')
    parser.add_argument('-o', '--output', help='Output directory for results (required with input)')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA acceleration')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Disable CUDA acceleration')
    parser.add_argument('--batch-size', type=int, help='Processing batch size')
    parser.add_argument('--eps', type=float, help='Clustering similarity threshold')
    parser.add_argument('--min-samples', type=int, help='Minimum samples for clustering')
    parser.add_argument('--version', action='version', version='Face Clustering Assistant v2.0')
    
    return parser


async def main_async():
    """Async main entry point."""
    chat = FaceClusteringChat()
    await chat.chat_loop()


def main():
    """Main entry point - supports both CLI and chat modes."""
    # Additional Unicode safety for Windows
    if sys.platform.startswith('win'):
        # Override all print functions to be Unicode-safe
        import builtins
        original_print = builtins.print
        def safe_print(*args, **kwargs):
            try:
                # Convert all args to safe ASCII
                safe_args = []
                for arg in args:
                    if isinstance(arg, str):
                        # Remove any Unicode characters that might cause issues
                        safe_arg = ''.join(char if ord(char) < 128 else '?' for char in arg)
                        safe_args.append(safe_arg)
                    else:
                        safe_args.append(arg)
                original_print(*safe_args, **kwargs)
            except UnicodeEncodeError:
                # Final fallback - convert everything to repr
                original_print(*[repr(arg) for arg in args], **kwargs)
        builtins.print = safe_print
    
    # Suppress OpenCV/libjpeg warnings globally
    try:
        import cv2
        # Only try to set logging if the attribute exists (newer OpenCV versions)
        if hasattr(cv2.utils, 'logging'):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass  # Ignore if cv2 not available or older version
    
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Check if CLI processing arguments are provided
        if args.input:
            if not args.output:
                print(error("Output directory (-o/--output) is required when input is provided"))
                sys.exit(1)
            # Run CLI processing mode
            asyncio.run(run_cli_processing(args))
        else:
            # Run interactive chat mode
            asyncio.run(main_async())
        
    except KeyboardInterrupt:
        print(f"\n\n{warning('Processing interrupted by user')}")
        print(colored("Goodbye!", Colors.BRIGHT_BLUE))
        sys.exit(0)
    except Exception as e:
        print(f"\n{error(f'Fatal error: {e}')}")
        sys.exit(1)


if __name__ == '__main__':
    main()