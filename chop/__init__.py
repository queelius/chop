"""chop - Unix-philosophy image manipulation CLI with JSON piping.

Every command outputs JSON to stdout. Side effects go to stderr or filesystem.

    chop load photo.jpg | chop resize 50% | chop save out.png
"""

from chop.cli import main

__version__ = "0.5.0"
__all__ = ["main"]
