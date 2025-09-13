"""Typing animation utilities for natural chat experience."""

import sys
import time
import random
import threading
from typing import Optional


class SmartTyper:
    """Smart typing animation with realistic pauses and user control."""
    
    def __init__(self, wpm: int = 550, typing_enabled: bool = True, length_cap: int = 300):
        """
        Initialize smart typer.
        
        Args:
            wpm: Words per minute typing speed (60-200 realistic range)
            typing_enabled: Whether typing animation is enabled
            length_cap: Character limit - messages longer than this are printed instantly
        """
        self.wpm = wpm
        self.typing_enabled = typing_enabled
        self.skip_requested = False
        self.base_char_delay = 60 / (wpm * 5)  # Average 5 chars per word
        self.length_cap = length_cap  # New: length cap for automatic instant printing
        
    def set_speed(self, speed: str):
        """Set typing speed: slow, normal, fast."""
        speed_map = {
            'slow': 250,
            'normal': 550, 
            'fast': 600,
            'instant': 1000  # Effectively instant
        }
        self.wpm = speed_map.get(speed.lower(), 550)
        self.base_char_delay = 60 / (self.wpm * 5)
    
    def enable_typing(self, enabled: bool = True):
        """Enable or disable typing animation."""
        self.typing_enabled = enabled
    
    def set_length_cap(self, length_cap: int):
        """Set the character limit for automatic instant printing."""
        self.length_cap = length_cap
    
    def get_speed_label(self) -> str:
        """Get the current speed label."""
        wpm_to_label = {250: "slow", 550: "normal", 600: "fast", 1000: "instant"}
        return wpm_to_label.get(self.wpm, f"{self.wpm} wpm")
    
    def _setup_skip_listener(self):
        """Setup background thread to listen for skip input."""
        self.skip_requested = False
        
        def listen_for_skip():
            try:
                # Non-blocking input check (works on most systems)
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    sys.stdin.read(1)
                    self.skip_requested = True
            except (ImportError, OSError):
                # Fallback for Windows or systems without select
                pass
        
        skip_thread = threading.Thread(target=listen_for_skip, daemon=True)
        skip_thread.start()
    
    def type_message(self, message: str, skip_technical: bool = True):
        """
        Type message with smart pauses and natural rhythm.
        
        Args:
            message: Text to type
            skip_technical: Skip typing for technical/path content
        """
        if not self.typing_enabled:
            print(message)
            return
        
        # NEW: Skip typing if message is too long
        if len(message) > self.length_cap:
            print(message)
            return
        
        # Skip typing for technical content (paths, long technical strings)
        if skip_technical and self._is_technical_content(message):
            print(message)
            return
        
        # Setup skip listener
        self._setup_skip_listener()
        
        # Split into words for natural pacing
        words = message.split(' ')
        
        try:
            for i, word in enumerate(words):
                if self.skip_requested:
                    # Print remaining message instantly
                    remaining = ' '.join(words[i:])
                    sys.stdout.write(remaining)
                    break
                
                # Type each character in the word
                for char in word:
                    if self.skip_requested:
                        break
                        
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    
                    # Variable delay based on character type
                    delay = self._get_char_delay(char)
                    time.sleep(delay)
                
                # Add space between words (except last word)
                if i < len(words) - 1 and not self.skip_requested:
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                    
                    # Natural pause between words
                    word_pause = self._get_word_pause(word, words[i + 1] if i + 1 < len(words) else '')
                    time.sleep(word_pause)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully - show remaining message
            if i < len(words) - 1:
                remaining = ' '.join(words[i:])
                sys.stdout.write(remaining)
        
        print()  # New line at end
        self.skip_requested = False
    
    def type_word_by_word(self, message: str, word_delay: float = 0.005):
        """
        Type message word by word ultra fast for big chunks of text.
        Perfect for commands like 'tune' with lots of technical content.
        
        Args:
            message: Text to type
            word_delay: Delay between words (very fast)
        """
        if not self.typing_enabled:
            print(message)
            return
        
        # NEW: Skip typing if message is too long
        if len(message) > self.length_cap:
            print(message)
            return
        
        # Setup skip listener
        self._setup_skip_listener()
        
        words = message.split()
        
        try:
            for i, word in enumerate(words):
                if self.skip_requested:
                    # Print remaining message instantly
                    remaining = ' '.join(words[i:])
                    sys.stdout.write(remaining)
                    break
                
                sys.stdout.write(word)
                sys.stdout.flush()
                
                # Add space after word (except last)
                if i < len(words) - 1:
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                    time.sleep(word_delay)
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            if i < len(words) - 1:
                remaining = ' '.join(words[i:])
                sys.stdout.write(remaining)
        
        print()  # New line at end
        self.skip_requested = False
    
    def _get_char_delay(self, char: str) -> float:
        """Get typing delay for specific character."""
        base = self.base_char_delay
        
        # Punctuation gets longer pauses
        if char in '.,!?;:':
            return base * random.uniform(2, 4)
        elif char in '()[]{}':
            return base * random.uniform(1.5, 2.5)
        elif char.isdigit():
            return base * random.uniform(1.2, 1.8)  # Numbers typed slower
        elif char.isupper():
            return base * random.uniform(1.1, 1.5)  # Capitals slightly slower
        else:
            return base * random.uniform(0.8, 1.2)  # Normal variation
    
    def _get_word_pause(self, current_word: str, next_word: str) -> float:
        """Get pause duration between words."""
        base = self.base_char_delay
        
        # Longer pauses after certain words
        if current_word.lower() in ['and', 'or', 'but', 'so', 'because', 'however']:
            return base * random.uniform(3, 6)
        elif current_word.endswith(('!', '?', '.')):
            return base * random.uniform(4, 8)
        elif current_word.endswith(','):
            return base * random.uniform(2, 4)
        elif len(current_word) > 8:  # Long words get extra pause
            return base * random.uniform(2, 4)
        else:
            return base * random.uniform(1.5, 3)
    
    def _is_technical_content(self, message: str) -> bool:
        """Check if message contains technical content that should be printed instantly."""
        technical_indicators = [
            '\\', '/', ':', 'http', 'www', '.com', '.py', '.json', '.yaml',
            '>>>', '<<<', '====', '----', '####',
            'C:\\', '/usr/', '/home/', './sessions',
            '│', '├', '└', '╭', '╮', '╯', '╰'  # Table characters
        ]
        
        # NOTE: Length check removed - now handled by length_cap in type_message()
        
        # Check for technical patterns
        for indicator in technical_indicators:
            if indicator in message:
                return True
        
        return False
    
    def print_instant(self, message: str):
        """Print message instantly without typing animation."""
        print(message)


# Global typer instance
_global_typer: Optional[SmartTyper] = None


def get_typer() -> SmartTyper:
    """Get or create global typer instance."""
    global _global_typer
    if _global_typer is None:
        _global_typer = SmartTyper()
    return _global_typer


def type_message(message: str, skip_technical: bool = True):
    """Convenience function for typing messages."""
    get_typer().type_message(message, skip_technical)


def type_word_by_word(message: str, word_delay: float = 0.005):
    """Convenience function for word-by-word typing (for big chunks)."""
    get_typer().type_word_by_word(message, word_delay)


def set_typing_speed(speed: str):
    """Set global typing speed: slow, normal, fast, instant."""
    get_typer().set_speed(speed)


def enable_typing(enabled: bool = True):
    """Enable or disable typing globally."""
    get_typer().enable_typing(enabled)


def set_length_cap(length_cap: int):
    """Set global length cap for automatic instant printing."""
    get_typer().set_length_cap(length_cap)


def print_instant(message: str):
    """Print message instantly without typing."""
    get_typer().print_instant(message)