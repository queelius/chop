"""Centralized output handling for chop CLI.

Uniform output: always writes pipeline state as JSON to stdout.
Side effects (file saves, info messages) go to stderr or filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chop.pipeline import PipelineState


def handle_output(state: PipelineState) -> None:
    """Write pipeline state as JSON to stdout. Always."""
    from chop.pipeline import write_pipeline_output

    write_pipeline_output(state)
