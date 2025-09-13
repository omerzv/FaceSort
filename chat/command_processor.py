"""Command processing logic for the chat interface."""

import os
import asyncio
import shlex
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from optimized.utils.colors import *
from optimized.chat.system_monitor import SystemMonitor
from optimized.chat.parameter_tuner import ParameterTuner


class CommandProcessor:
    """Processes user commands in the chat interface."""
    
    def __init__(self, chat_interface):
        self.chat = chat_interface
        self.monitor = SystemMonitor()
        self.tuner = ParameterTuner(chat_interface)
        
        # Command mapping
        self.commands = {
            'help': self.show_help,
            'config': self.show_config,
            'status': self.show_status,
            'input': self.set_input_directory,
            'output': self.set_output_directory,
            'run': self.run_clustering,
            'clear': self.chat.clear_screen,
            'exit': self.exit_application,
            'quit': self.exit_application,
            # Parameter tuning commands
            'tune': self.show_parameters,
            'set': self.set_parameter,
            'settings': self.show_current_settings,
            # Typing control commands
            'typing': self.control_typing,
        }
    
    async def process_command(self, user_input: str) -> bool:
        """Process a user command. Returns True to continue, False to exit."""
        if not user_input:
            return True
        
        parts = user_input.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.commands:
            try:
                if asyncio.iscoroutinefunction(self.commands[command]):
                    result = await self.commands[command](args)
                else:
                    result = self.commands[command](args)
                
                # Handle exit commands
                if result is False:
                    return False
                    
            except Exception as e:
                self.chat.show_error(f"Command failed: {str(e)}")
        else:
            # Try to find similar commands
            suggestions = self._find_similar_commands(command)
            if suggestions:
                self.chat.show_error(f"Unknown command: '{command}'. Did you mean: {', '.join(suggestions)}?")
            else:
                self.chat.show_error(f"Unknown command: '{command}'. Type 'help' for available commands.")
        
        return True
    
    def _find_similar_commands(self, command: str) -> list:
        """Find commands similar to the input using simple distance matching."""
        import difflib
        suggestions = difflib.get_close_matches(
            command, self.commands.keys(), 
            n=3, cutoff=0.6  # Return up to 3 suggestions with 60% similarity
        )
        return suggestions
    
    def show_help(self, args: str = "") -> None:
        """Show help information."""
        self.chat.show_help()
    
    def show_config(self, args: str = "") -> None:
        """Show current configuration."""
        config = self.chat.get_config()
        
        print(f"\n{colored('⚙️  Current Configuration:', Colors.BRIGHT_MAGENTA)}")
        print(f"  {colored('Input Directory:', Colors.BRIGHT_CYAN)} {self.chat.current_input or 'Not set'}")
        print(f"  {colored('Output Directory:', Colors.BRIGHT_CYAN)} {self.chat.current_output or 'Not set'}")
        
        # Show key processing settings
        print(f"\n{colored('🔧 Processing Settings:', Colors.BRIGHT_BLUE)}")
        print(f"  {colored('CUDA:', Colors.CYAN)} {'Enabled' if config.processing.use_cuda else 'Disabled'}")
        print(f"  {colored('Batch Size:', Colors.CYAN)} {config.processing.batch_size}")
        print(f"  {colored('Workers:', Colors.CYAN)} {config.processing.max_workers}")
        
        # Show clustering settings
        print(f"\n{colored('🎯 Clustering Settings:', Colors.BRIGHT_BLUE)}")
        print(f"  {colored('Algorithm:', Colors.CYAN)} {config.clustering.algorithm}")
        print(f"  {colored('EPS:', Colors.CYAN)} {config.clustering.eps}")
        print(f"  {colored('Min Samples:', Colors.CYAN)} {config.clustering.min_samples}")
        print()
    
    def show_status(self, args: str = "") -> None:
        """Show system status."""
        status = self.monitor.get_combined_status()
        self.chat.show_info(f"System Status: {status}")
        
        # Show directory status
        input_status = "✅ Ready" if self.chat.current_input and Path(self.chat.current_input).exists() else "❌ Not found"
        output_status = "✅ Ready" if self.chat.current_output else "❌ Not set"
        
        print(f"\n{colored('📁 Directory Status:', Colors.BRIGHT_BLUE)}")
        print(f"  {colored('Input:', Colors.CYAN)} {input_status}")
        print(f"  {colored('Output:', Colors.CYAN)} {output_status}")
        print()
    
    def set_input_directory(self, path_str: str) -> None:
        """Set input directory."""
        if not path_str:
            self.chat.show_error("Please provide a directory path. Example: input ./photos")
            return
        
        path = Path(path_str).resolve()
        if not path.exists():
            self.chat.show_error(f"Directory not found: {path}")
            return
        
        if not path.is_dir():
            self.chat.show_error(f"Path is not a directory: {path}")
            return
        
        self.chat.current_input = str(path)
        self.chat.show_success(f"Input directory set to: {path}")
    
    def set_output_directory(self, path_str: str) -> None:
        """Set output directory."""
        if not path_str:
            self.chat.show_error("Please provide a directory path. Example: output ./results")
            return
        
        path = Path(path_str).resolve()
        # Create output directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        self.chat.current_output = str(path)
        self.chat.show_success(f"Output directory set to: {path}")
    
    async def run_clustering(self, args: str = "") -> None:
        """Run the clustering process."""
        if not self.chat.current_input:
            self.chat.show_error("Please set an input directory first. Use: input <path>")
            return
        
        if not self.chat.current_output:
            self.chat.show_error("Please set an output directory first. Use: output <path>")
            return
        
        self.chat.show_info("Starting face clustering process...")
        
        # Initialize pipeline outside try block for proper cleanup
        pipeline = None
        
        try:
            # Import and run the pipeline
            from optimized.core.face_pipeline import FacePipeline
            
            config = self.chat.get_config()
            pipeline = FacePipeline(config)
            
            # Run the pipeline
            result = await pipeline.process(
                self.chat.current_input,
                self.chat.current_output,
                progress_callback=self._progress_callback
            )
            
            print()  # New line after progress bar
            self.chat.last_result = result
            self.chat.show_success("Face clustering completed!")
            self.chat.show_info(f"Results saved to: {self.chat.current_output}")
            
        except Exception as e:
            self.chat.show_error(f"Clustering failed: {str(e)}")
        finally:
            # Always cleanup pipeline resources
            if pipeline is not None:
                try:
                    await pipeline.cleanup()
                except Exception as cleanup_error:
                    # Log cleanup errors but don't show to user
                    pass
    
    def _progress_callback(self, stage: str, current: int, total: int, **kwargs):
        """Unified progress callback for clustering process."""
        # Clear the current line completely and create a unified progress bar
        cols = shutil.get_terminal_size((80, 20)).columns
        print(f"\r{' ' * (cols - 1)}", end='', flush=True)  # Clear line with spaces
        
        if total == 100:  # Using percentage-based progress
            percent = max(0, min(100, current))
            # Create visual progress bar
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"\r{colored('[~]', Colors.BRIGHT_CYAN)} {colored(stage, Colors.BRIGHT_WHITE)}: {colored(f'{bar}', Colors.BRIGHT_GREEN)} {colored(f'{percent}%', Colors.BRIGHT_CYAN)}", end='', flush=True)
        elif total > 0:
            # Traditional current/total progress
            percent = max(0, min(100, int(current / total * 100)))
            bar_length = 30
            filled_length = int(bar_length * current / total)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"\r{colored('[~]', Colors.BRIGHT_CYAN)} {colored(stage, Colors.BRIGHT_WHITE)}: {colored(f'{bar}', Colors.BRIGHT_GREEN)} {colored(f'{percent}% ({current:,}/{total:,})', Colors.BRIGHT_CYAN)}", end='', flush=True)
        else:
            # Indeterminate progress
            print(f"\r{colored('[~]', Colors.BRIGHT_CYAN)} {colored(stage, Colors.BRIGHT_WHITE)}...", end='', flush=True)
    
    def exit_application(self, args: str = "") -> bool:
        """Exit the application."""
        self.chat.show_assistant_message("Thank you for using Face Clustering Assistant!")
        return False
    
    def show_parameters(self, args: str = "") -> None:
        """Show all available parameters for tuning."""
        self.tuner.show_all_parameters()
    
    def set_parameter(self, args: str = "") -> None:
        """Set a parameter value."""
        if not args:
            self.chat.show_error("Usage: set <parameter> <value> [--cfg]")
            self.chat.show_info("Example: set eps 0.3  or  set e 0.3")
            self.chat.show_info("Add --cfg to save to config file: set eps 0.3 --cfg")
            self.chat.show_info("Use 'tune' to see all available parameters")
            return
        
        # Parse arguments and check for --cfg flag
        try:
            parts = shlex.split(args)
        except ValueError as e:
            self.chat.show_error(f"Invalid command syntax: {e}")
            return
        
        save_to_config = False
        
        if len(parts) >= 2 and parts[-1] == "--cfg":
            save_to_config = True
            parts = parts[:-1]  # Remove --cfg flag
        
        if len(parts) != 2:
            self.chat.show_error("Usage: set <parameter> <value> [--cfg]")
            return
        
        param_name, value = parts
        self.tuner.set_parameter(param_name, value, save_to_config)
    
    def show_current_settings(self, args: str = "") -> None:
        """Show current parameter settings."""
        self.tuner.show_current_settings()
    
    def control_typing(self, args: str = "") -> None:
        """Control typing animation settings."""
        from optimized.utils.typing_animation import set_typing_speed, enable_typing, get_typer, set_length_cap
        
        if not args:
            typer = get_typer()
            status = "enabled" if typer.typing_enabled else "disabled"
            speed = typer.get_speed_label()
            length_cap = typer.length_cap
            
            self.chat.show_info(f"Typing animation: {status}, speed: {speed}, length cap: {length_cap}")
            self.chat.show_info("Usage: typing on/off/slow/normal/fast/instant")
            self.chat.show_info("       typing cap <number> - set character limit for instant printing")
            return
        
        parts = args.strip().lower().split()
        
        if len(parts) == 1:
            arg = parts[0]
            if arg in ['on', 'enable', 'true']:
                enable_typing(True)
                self.chat.show_success("Typing animation enabled")
            elif arg in ['off', 'disable', 'false']:
                enable_typing(False)
                self.chat.show_success("Typing animation disabled")
            elif arg in ['slow', 'normal', 'fast', 'instant']:
                set_typing_speed(arg)
                enable_typing(True)  # Auto-enable when setting speed
                self.chat.show_success(f"Typing speed set to {arg}")
            else:
                self.chat.show_error("Invalid option. Use: on/off/slow/normal/fast/instant or cap <number>")
        
        elif len(parts) == 2 and parts[0] == 'cap':
            try:
                length_cap = int(parts[1])
                if length_cap < 50:
                    self.chat.show_error("Length cap must be at least 50 characters")
                    return
                set_length_cap(length_cap)
                self.chat.show_success(f"Length cap set to {length_cap} characters")
                self.chat.show_info("Messages longer than this will be printed instantly")
            except ValueError:
                self.chat.show_error("Invalid number. Use: typing cap <number>")
        
        else:
            self.chat.show_error("Invalid option. Use: on/off/slow/normal/fast/instant or cap <number>")
    
