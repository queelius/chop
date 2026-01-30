"""Centralized output handling for chop CLI.

Implements Unix-philosophy output behavior with lazy evaluation:
- Save to file when -o flag is used (requires bound pipeline)
- JSON output when piped (ops list, no image data)
- Pretty-print program on TTY for unbound pipelines
- Image info on TTY for bound pipelines
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
        2. -o/--output FILE → Materialize and save to file (must be bound)
        3. stdout is TTY → Show info (bound) or program listing (unbound)
        4. stdout is piped → JSON output (lazy, no materialization)

    Args:
        state: Pipeline state with operations.
        args: Parsed command-line arguments.
    """
    from chop.pipeline import write_pipeline_output

    # 1. Explicit JSON flag
    if getattr(args, "json", False):
        write_pipeline_output(state)
        return

    # 2. Save to file
    if getattr(args, "output", None):
        if not state.has_load():
            print(
                "Error: Cannot save unbound pipeline to file. "
                "Add 'chop load <file>' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        image = state.materialize()
        image.save(args.output)
        print(f"Saved to {args.output}", file=sys.stderr)
        return

    # 3. TTY detection
    if sys.stdout.isatty():
        if state.has_load():
            # Bound pipeline — materialize and show info
            image = state.materialize()
            w, h = image.size
            print(f"Image: {w}x{h} {image.mode}", file=sys.stderr)
            print("Use -o FILE to save, or pipe to another chop command", file=sys.stderr)
        else:
            # Unbound pipeline — show program listing
            print("Program:", file=sys.stderr)
            for name, op_args, kwargs in state.ops:
                parts = [name] + [str(a) for a in op_args]
                for k, v in kwargs.items():
                    parts.append(f"{k}={v}")
                print(f"  {' '.join(parts)}", file=sys.stderr)
            print("Use -j to output as JSON, or pipe to 'chop apply'", file=sys.stderr)
    else:
        # 4. Piped — output JSON for chaining
        write_pipeline_output(state)
