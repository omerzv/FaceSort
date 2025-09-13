"""Tutorial manager for guiding new users."""

from optimized.utils.colors import *


class TutorialManager:
    """Manages interactive tutorials for new users."""
    
    def __init__(self, chat_interface):
        self.chat = chat_interface
    
    def show_tutorial(self):
        """Interactive tutorial for new users."""
        self.chat.show_assistant_message("Welcome to the Face Clustering Tutorial!")
        print()
        
        self.chat.show_assistant_message("This tutorial will walk you through the basic steps:")
        print()
        print(f"  {colored('Step 1:', Colors.BRIGHT_BLUE)} Check your current settings")
        print(f"  {colored('Step 2:', Colors.BRIGHT_BLUE)} Set up your photo directory")  
        print(f"  {colored('Step 3:', Colors.BRIGHT_BLUE)} Configure processing options")
        print(f"  {colored('Step 4:', Colors.BRIGHT_BLUE)} Start face clustering")
        print(f"  {colored('Step 5:', Colors.BRIGHT_BLUE)} View your results")
        print()
        
        self.chat.show_assistant_message("Here's how to get started:")
        print()
        print(f"  {colored('1. Check settings:', Colors.BRIGHT_GREEN)} Type {colored('config', Colors.BRIGHT_CYAN)} to see current configuration")
        print(f"  {colored('2. Set photo folder:', Colors.BRIGHT_GREEN)} Type {colored('input <path>', Colors.BRIGHT_CYAN)} to point to your photos")
        print(f"     Example: {colored('input ./my_photos', Colors.DIM)}")
        print(f"  {colored('3. Start clustering:', Colors.BRIGHT_GREEN)} Type {colored('run', Colors.BRIGHT_CYAN)} to begin processing")
        print(f"  {colored('4. Tune parameters:', Colors.BRIGHT_GREEN)} Type {colored('tune', Colors.BRIGHT_CYAN)} for advanced settings")
        print(f"  {colored('5. Get help:', Colors.BRIGHT_GREEN)} Type {colored('help', Colors.BRIGHT_CYAN)} anytime for commands")
        print()
        
        self.chat.show_assistant_message("Tutorial completed! You can now use any command.")
        print()
    
    def show_quick_start(self):
        """Show quick start guide."""
        print(f"\n{colored('Quick Start Guide:', Colors.BRIGHT_MAGENTA)}")
        print(f"  1. {colored('input <path>', Colors.BRIGHT_GREEN)}  - Set your photo folder")
        print(f"  2. {colored('run', Colors.BRIGHT_GREEN)}           - Start clustering")
        print(f"  3. Check results in output folder")
        print()
        
        self.chat.show_assistant_message("Ready to get started? Try typing 'input ./photos' to begin!")