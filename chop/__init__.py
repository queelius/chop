"""chop - Unix-philosophy image manipulation CLI with JSON piping.

Supports chaining operations via JSON piping:
    chop load photo.jpg -j | chop resize 50% -j | chop save out.png
"""

from chop.cli import main

__version__ = "0.2.0"
__all__ = ["main"]
