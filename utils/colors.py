
import sys
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows compatibility
init(autoreset=True)

class Colors:
    """Color constants for CLI output."""
    # Basic colors
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    
    # Bright colors
    BRIGHT_RED = Fore.LIGHTRED_EX
    BRIGHT_GREEN = Fore.LIGHTGREEN_EX
    BRIGHT_YELLOW = Fore.LIGHTYELLOW_EX
    BRIGHT_BLUE = Fore.LIGHTBLUE_EX
    BRIGHT_MAGENTA = Fore.LIGHTMAGENTA_EX
    BRIGHT_CYAN = Fore.LIGHTCYAN_EX
    BRIGHT_WHITE = Fore.LIGHTWHITE_EX
    
    # Styles
    BOLD = Style.BRIGHT
    DIM = Style.DIM
    RESET = Style.RESET_ALL

class Icons:
    """ASCII icons for cross-platform compatibility."""
    # Status icons
    SUCCESS = "[+]"
    ERROR = "[X]"
    WARNING = "[!]"
    INFO = "[i]"
    PROCESSING = "[~]"
    
    # Action icons
    FOLDER = "[D]"
    FILE = "[-]"
    GEAR = "[*]"
    LIGHTNING = "[^]"
    TARGET = "[o]"
    
    # Chat interface icons
    USER = ">>>"
    ASSISTANT = "<<<"
    CONFIG = "[#]"
    INPUT_DIR = "[>>]"
    OUTPUT_DIR = "[<<]"
    GPU = "[GPU]"
    BATCH = "[###]"
    SIMILARITY = "[===]"
    SAMPLES = "[000]"
    SAVE = "[S]"
    HELP = "[?]"
    QUIT = "[Q]"
    
    # Progress
    FULL_BLOCK = "█"
    LIGHT_SHADE = "░"
    


def _safe_text(text):
    """Ensure text is safe for Windows console by removing problematic Unicode."""
    import sys
    if sys.platform.startswith('win'):
        try:
            # Try encoding with the console's encoding
            text.encode(sys.stdout.encoding or 'cp1252', errors='strict')
            return text
        except (UnicodeEncodeError, AttributeError):
            # Replace problematic characters with ASCII equivalents
            import unicodedata
            # First try to normalize and convert to ASCII
            try:
                normalized = unicodedata.normalize('NFKD', text)
                ascii_text = normalized.encode('ascii', errors='ignore').decode('ascii')
                return ascii_text if ascii_text else text.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                # Final fallback - remove all non-ASCII
                return ''.join(char if ord(char) < 128 else '?' for char in text)
    return text

def colored(text, color, bold=False):
    """Return colored text with safe encoding."""
    safe_text = _safe_text(str(text))
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{safe_text}{Colors.RESET}"

def success(text):
    """Green success message."""
    return colored(f"{Icons.SUCCESS} {text}", Colors.BRIGHT_GREEN, bold=True)

def error(text):
    """Red error message."""
    return colored(f"{Icons.ERROR} {text}", Colors.BRIGHT_RED, bold=True)

def warning(text):
    """Yellow warning message."""
    return colored(f"{Icons.WARNING} {text}", Colors.BRIGHT_YELLOW, bold=True)

def info(text):
    """Blue info message."""
    return colored(f"{Icons.INFO} {text}", Colors.BRIGHT_BLUE)

def processing(text):
    """Cyan processing message.""" 
    return colored(f"{Icons.PROCESSING} {text}", Colors.BRIGHT_CYAN)

def header(text):
    """Bold header text."""
    return colored(text, Colors.BRIGHT_WHITE, bold=True)

def create_progress_bar(current, total, width=40, fill_char="#", empty_char="-"):
    """Create a visual progress bar."""
    if total == 0:
        return f"[{empty_char * width}] 0%"
    
    percentage = min(100, int((current / total) * 100))
    filled_width = int((current / total) * width)
    empty_width = width - filled_width
    
    bar = fill_char * filled_width + empty_char * empty_width
    return f"[{colored(bar[:filled_width], Colors.BRIGHT_GREEN)}{colored(bar[filled_width:], Colors.DIM)}] {percentage}%"

def create_banner(title, width=60):
    """Create a banner with title."""
    border = "=" * width
    padding = " " * ((width - len(title) - 2) // 2)
    
    return f"""
{colored(border, Colors.BRIGHT_BLUE)}
{colored(f"|{padding}{title}{padding}|", Colors.BRIGHT_WHITE, bold=True)}
{colored(border, Colors.BRIGHT_BLUE)}"""

def create_section_header(title):
    """Create a section header."""
    return colored(f"\n> {title}", Colors.BRIGHT_YELLOW, bold=True)

def format_stat(label, value, unit=""):
    """Format a statistic line."""
    label_colored = colored(f"  - {label}:", Colors.WHITE)
    value_colored = colored(f"{value:,}{unit}", Colors.BRIGHT_GREEN, bold=True)
    return f"{label_colored} {value_colored}"

def format_config(label, value):
    """Format a configuration line.""" 
    label_colored = colored(f"{Icons.GEAR} {label}:", Colors.CYAN)
    value_colored = colored(str(value), Colors.BRIGHT_CYAN)
    return f"{label_colored} {value_colored}"

