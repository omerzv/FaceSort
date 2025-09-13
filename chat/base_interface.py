"""Base chat interface with core functionality."""

import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from optimized.utils.colors import *
from optimized.utils.config import ConfigManager, AppConfig


class BaseChatInterface:
    """Base chat interface with essential functionality."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config: Optional[AppConfig] = None
        self.current_input: Optional[str] = "./input_photos"
        self.current_output: Optional[str] = "./sessions"
        self.last_result = None
        self.last_session_path: Optional[str] = None
    
    def show_banner(self):
        """Show application banner."""
        print(create_banner("Face Clustering Assistant", 60))
        print()
    
    def show_assistant_message(self, message: str):
        """Show assistant message with typing animation."""
        from optimized.utils.typing_animation import type_message
        
        # Type the prefix instantly, then animate the message
        prefix = f"{colored('Assistant:', Colors.BRIGHT_GREEN)} "
        print(prefix, end='', flush=True)
        type_message(message)
    
    def show_user_prompt(self) -> str:
        """Show user input prompt."""
        try:
            return input(f"{colored('You:', Colors.BRIGHT_CYAN)} ").strip()
        except (KeyboardInterrupt, EOFError):
            return "exit"
    
    def show_error(self, message: str):
        """Show error message."""
        print(f"{colored('Error:', Colors.BRIGHT_RED)} {message}")
    
    def show_success(self, message: str):
        """Show success message with typing animation."""
        from optimized.utils.typing_animation import type_message
        
        prefix = f"{colored('Success:', Colors.BRIGHT_GREEN)} "
        print(prefix, end='', flush=True)
        type_message(message)
    
    def show_warning(self, message: str):
        """Show warning message with typing animation."""
        from optimized.utils.typing_animation import type_message
        
        prefix = f"{colored('Warning:', Colors.BRIGHT_YELLOW)} "
        print(prefix, end='', flush=True)
        type_message(message)
    
    def show_info(self, message: str):
        """Show info message with typing animation."""
        from optimized.utils.typing_animation import type_message
        
        prefix = f"{colored('Info:', Colors.BRIGHT_BLUE)} "
        print(prefix, end='', flush=True)
        type_message(message)
    
    def clear_screen(self):
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_config(self) -> AppConfig:
        """Get or load configuration."""
        if self.config is None:
            self.config = self.config_manager.get_config()
        return self.config
    
    def show_help(self):
        """Show help information."""
        print(f"\n{colored('Available Commands:', Colors.BRIGHT_MAGENTA)}")
        print(f"  {colored('config', Colors.BRIGHT_GREEN)}           - Show current settings")
        print(f"  {colored('input <path>', Colors.BRIGHT_GREEN)}     - Set photo directory")
        print(f"  {colored('output <path>', Colors.BRIGHT_GREEN)}    - Set output directory")
        print(f"  {colored('run', Colors.BRIGHT_GREEN)}             - Start face clustering")
        print(f"  {colored('status', Colors.BRIGHT_GREEN)}          - Show system status")
        print(f"  {colored('tutorial', Colors.BRIGHT_GREEN)}        - Guided walkthrough")
        print(f"  {colored('quickstart', Colors.BRIGHT_GREEN)}      - Quick start tips")
        print(f"  {colored('clear', Colors.BRIGHT_GREEN)}           - Clear the screen")
        print(f"  {colored('help', Colors.BRIGHT_GREEN)}            - Show this help")
        print(f"  {colored('exit', Colors.BRIGHT_GREEN)}            - Exit application")
        print()
        print(f"{colored('Parameter Tuning:', Colors.BRIGHT_MAGENTA)}")
        print(f"  {colored('tune', Colors.BRIGHT_GREEN)}             - Show all available parameters with explanations")
        print(f"  {colored('set <param> <value>', Colors.BRIGHT_GREEN)} - Set parameter temporarily")
        print(f"  {colored('set <param> <value> --cfg', Colors.BRIGHT_GREEN)} - Set and save to config file permanently")
        print(f"  {colored('settings', Colors.BRIGHT_GREEN)}         - Show current parameter values")
        print()
        print(f"{colored('Chat Experience:', Colors.BRIGHT_MAGENTA)}")
        print(f"  {colored('typing <option>', Colors.BRIGHT_GREEN)}    - Control typing animation")
        print(f"                         Options: on/off/slow/normal/fast/instant")
        print(f"  {colored('typing cap <num>', Colors.BRIGHT_GREEN)}    - Set character limit for instant printing")
        print()
        print(f"{colored('Examples:', Colors.BRIGHT_BLUE)}")
        print(f"  {colored('tune', Colors.CYAN)}              - View all parameters with detailed explanations")
        print(f"  {colored('set eps 0.25', Colors.CYAN)}       - Set similarity threshold (temporary)")
        print(f"  {colored('set eps 0.25 --cfg', Colors.CYAN)}  - Set and save as new default")
        print(f"  {colored('typing fast', Colors.CYAN)}        - Speed up chat typing animation")
        print(f"  {colored('typing off', Colors.CYAN)}         - Disable typing animation for instant responses")
        print(f"  {colored('typing cap 200', Colors.CYAN)}      - Long messages (>200 chars) print instantly")
        print()