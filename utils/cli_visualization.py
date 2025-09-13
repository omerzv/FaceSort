
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich.align import Align
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from optimized.utils.logger import get_logger

logger = get_logger(__name__)


class CLIVisualizer:
    """Rich CLI visualization for face processing results."""
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
            logger.warning("Rich not available, falling back to basic text output")
    
    def display_processing_stats(self, stats: Any) -> None:
        """Display processing statistics in a formatted table.
        
        Args:
            stats: ProcessingStats object
        """
        if not self.console:
            self._display_stats_basic(stats)
            return
        
        # Create stats table
        table = Table(title="📊 Processing Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bright_green", justify="right")
        
        # Add rows
        table.add_row("Images Processed", f"{stats.total_images:,}")
        table.add_row("Faces Detected", f"{stats.total_faces:,}")
        table.add_row("Clusters Found", f"{stats.clusters_found:,}")
        
        if hasattr(stats, 'noise_faces'):
            table.add_row("Noise Faces", f"{stats.noise_faces:,}")
        
        table.add_row("Processing Time", f"{stats.total_time:.2f}s")
        table.add_row("Processing Speed", f"{stats.faces_per_second:.1f} faces/sec")
        
        if stats.avg_quality > 0:
            table.add_row("Average Quality", f"{stats.avg_quality:.3f}")
        
        self.console.print("\n")
        self.console.print(table)
    
    def display_cluster_summary(self, clusters: Dict[int, Any], top_n: int = 10) -> None:
        """Display cluster summary in a formatted table.
        
        Args:
            clusters: Dictionary of cluster info
            top_n: Number of top clusters to display
        """
        if not clusters:
            if self.console:
                self.console.print("\n[yellow]No clusters found[/yellow]")
            else:
                print("\nNo clusters found")
            return
        
        if not self.console:
            self._display_clusters_basic(clusters, top_n)
            return
        
        # Sort clusters by face count
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].face_count, reverse=True)
        
        # Create cluster table
        table = Table(title=f"👥 Top {min(top_n, len(sorted_clusters))} Clusters", box=box.ROUNDED)
        table.add_column("Cluster ID", style="bright_magenta", justify="center")
        table.add_column("Faces", style="bright_green", justify="right")
        table.add_column("Avg Quality", style="cyan", justify="center")
        table.add_column("Quality Range", style="yellow", justify="center")
        
        for i, (cluster_id, cluster_info) in enumerate(sorted_clusters[:top_n]):
            # Calculate quality range if faces available
            if hasattr(cluster_info, 'faces') and cluster_info.faces:
                qualities = [face.quality_score for face in cluster_info.faces]
                min_qual, max_qual = min(qualities), max(qualities)
                quality_range = f"{min_qual:.2f} - {max_qual:.2f}"
            else:
                quality_range = "N/A"
            
            avg_quality = getattr(cluster_info, 'avg_quality', 0.0)
            
            table.add_row(
                f"#{cluster_id:03d}",
                f"{cluster_info.face_count}",
                f"{avg_quality:.3f}" if avg_quality > 0 else "N/A",
                quality_range
            )
        
        self.console.print("\n")
        self.console.print(table)
        
        # Show remaining clusters count
        if len(sorted_clusters) > top_n:
            remaining = len(sorted_clusters) - top_n
            self.console.print(f"\n[dim]... and {remaining} more clusters[/dim]")
    
    def display_quality_distribution(self, faces: List[Any]) -> None:
        """Display quality score distribution.
        
        Args:
            faces: List of face objects with quality scores
        """
        if not faces:
            return
        
        qualities = [face.quality_score for face in faces if hasattr(face, 'quality_score')]
        if not qualities:
            return
        
        if not self.console:
            self._display_quality_basic(qualities)
            return
        
        # Create quality bins
        high_quality = sum(1 for q in qualities if q >= 0.7)
        medium_quality = sum(1 for q in qualities if 0.4 <= q < 0.7)
        low_quality = sum(1 for q in qualities if q < 0.4)
        
        # Create quality table
        table = Table(title="📈 Quality Distribution", box=box.ROUNDED)
        table.add_column("Quality Level", style="cyan")
        table.add_column("Count", style="bright_green", justify="right")
        table.add_column("Percentage", style="yellow", justify="right")
        
        total = len(qualities)
        table.add_row("High (≥0.7)", f"{high_quality}", f"{high_quality/total*100:.1f}%")
        table.add_row("Medium (0.4-0.7)", f"{medium_quality}", f"{medium_quality/total*100:.1f}%")
        table.add_row("Low (<0.4)", f"{low_quality}", f"{low_quality/total*100:.1f}%")
        
        self.console.print("\n")
        self.console.print(table)
    
    def display_output_summary(self, output_dir: str, saved_files: Dict[str, int]) -> None:
        """Display output summary with file counts.
        
        Args:
            output_dir: Output directory path
            saved_files: Dictionary of file type -> count
        """
        if not self.console:
            self._display_output_basic(output_dir, saved_files)
            return
        
        # Create output table
        table = Table(title="💾 Output Summary", box=box.ROUNDED)
        table.add_column("File Type", style="cyan")
        table.add_column("Count", style="bright_green", justify="right")
        
        for file_type, count in saved_files.items():
            table.add_row(file_type.title(), f"{count:,}")
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print(f"\n📁 Output saved to: [bright_cyan]{output_dir}[/bright_cyan]")
    
    def create_progress_bar(self, description: str = "Processing") -> Optional[Any]:
        """Create a rich progress bar.
        
        Args:
            description: Progress bar description
            
        Returns:
            Progress object or None if rich not available
        """
        if not self.console:
            return None
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        return progress
    
    def display_error_summary(self, errors: List[str]) -> None:
        """Display error summary if any errors occurred.
        
        Args:
            errors: List of error messages
        """
        if not errors:
            return
        
        if not self.console:
            print(f"\n⚠️  {len(errors)} errors occurred during processing:")
            for error in errors[:5]:
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            return
        
        # Create error panel
        error_text = "\n".join(errors[:10])  # Show up to 10 errors
        if len(errors) > 10:
            error_text += f"\n... and {len(errors) - 10} more errors"
        
        panel = Panel(
            error_text,
            title=f"⚠️  {len(errors)} Errors Occurred",
            border_style="red"
        )
        
        self.console.print("\n")
        self.console.print(panel)
    
    # Fallback methods for when rich is not available
    def _display_stats_basic(self, stats: Any) -> None:
        """Display stats using basic text."""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Images Processed: {stats.total_images:,}")
        print(f"Faces Detected: {stats.total_faces:,}")
        print(f"Clusters Found: {stats.clusters_found:,}")
        if hasattr(stats, 'noise_faces'):
            print(f"Noise Faces: {stats.noise_faces:,}")
        print(f"Processing Time: {stats.total_time:.2f}s")
        print(f"Processing Speed: {stats.faces_per_second:.1f} faces/sec")
        if stats.avg_quality > 0:
            print(f"Average Quality: {stats.avg_quality:.3f}")
    
    def _display_clusters_basic(self, clusters: Dict[int, Any], top_n: int) -> None:
        """Display clusters using basic text."""
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].face_count, reverse=True)
        
        print(f"\nTOP {min(top_n, len(sorted_clusters))} CLUSTERS")
        print("-" * 50)
        print("ID  | Faces | Avg Quality")
        print("-" * 50)
        
        for i, (cluster_id, cluster_info) in enumerate(sorted_clusters[:top_n]):
            avg_quality = getattr(cluster_info, 'avg_quality', 0.0)
            print(f"{cluster_id:03d} | {cluster_info.face_count:5d} | {avg_quality:.3f}")
        
        if len(sorted_clusters) > top_n:
            remaining = len(sorted_clusters) - top_n
            print(f"... and {remaining} more clusters")
    
    def _display_quality_basic(self, qualities: List[float]) -> None:
        """Display quality distribution using basic text."""
        if not qualities:
            print("\nNo quality data available")
            return
            
        high_quality = sum(1 for q in qualities if q >= 0.7)
        medium_quality = sum(1 for q in qualities if 0.4 <= q < 0.7)
        low_quality = sum(1 for q in qualities if q < 0.4)
        total = len(qualities)
        
        print("\nQUALITY DISTRIBUTION")
        print("-" * 30)
        print(f"High (≥0.7):    {high_quality:5d} ({high_quality/total*100:.1f}%)")
        print(f"Medium (0.4-0.7): {medium_quality:5d} ({medium_quality/total*100:.1f}%)")
        print(f"Low (<0.4):      {low_quality:5d} ({low_quality/total*100:.1f}%)")
    
    def _display_output_basic(self, output_dir: str, saved_files: Dict[str, int]) -> None:
        """Display output summary using basic text."""
        print("\nOUTPUT SUMMARY")
        print("-" * 30)
        for file_type, count in saved_files.items():
            print(f"{file_type.title()}: {count:,}")
        print(f"\nOutput saved to: {output_dir}")


# Global visualizer instance
visualizer = CLIVisualizer()