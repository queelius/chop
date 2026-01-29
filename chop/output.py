"""Centralized output handling for chop CLI.

Implements Unix-philosophy output behavior with lazy evaluation:
- Save to file when -o flag is used
- JSON output when piped (just path + ops, no image data)
- Write PNG to stdout when on TTY with no explicit output flag
- Image is only loaded/processed (materialized) at save time
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from chop.pipeline import PipelineState


def handle_output(state: PipelineState, args: argparse.Namespace) -> None:
    """Decide output based on flags and TTY detection.

    Decision logic (in order):
        1. -j/--json flag → JSON output (force even on TTY)
        2. -o/--output FILE → Materialize and save to file
        3. stdout is TTY → Materialize and write PNG info to stderr
        4. stdout is piped → JSON output (lazy, no materialization)

    Args:
        state: Pipeline state with path and operations.
        args: Parsed command-line arguments.
    """
    from chop.pipeline import write_pipeline_output

    # 1. Explicit JSON flag (force JSON even on TTY)
    if getattr(args, "json", False):
        write_pipeline_output(state)
        return

    # 2. Save to file (materialize first)
    if getattr(args, "output", None):
        image = state.materialize()
        image.save(args.output)
        print(f"Saved to {args.output}", file=sys.stderr)
        return

    # 3. TTY detection: info if interactive, JSON if piped
    if sys.stdout.isatty():
        # Materialize and show info (pipe to imgcat for display)
        image = state.materialize()
        w, h = image.size
        print(f"Image: {w}x{h} {image.mode}", file=sys.stderr)
        print("Use -o FILE to save, or pipe to imgcat for display", file=sys.stderr)
    else:
        # 4. Piped - output JSON for chaining (no materialization)
        write_pipeline_output(state)
