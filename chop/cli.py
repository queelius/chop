#!/usr/bin/env python3
"""chop - Unix-philosophy image manipulation CLI with lazy evaluation.

Lazy pipeline: JSON carries only file path + operations list.
Image is loaded and processed only at save time.

    chop load photo.jpg | chop resize 50% | chop pad 10 | chop save out.png
         ↓                    ↓                 ↓              ↓
      {path}            +resize op          +pad op     LOAD → APPLY → SAVE

Auto-detects output context for seamless piping:
    chop load photo.jpg | chop resize 50% | chop border 5 --color red
                                           ↑ auto-outputs JSON when piped

Explicit output control:
    chop load photo.jpg | chop resize 50% -j          # Force JSON on TTY
    chop load photo.jpg | chop resize 50% -o out.png  # Save to file
    chop load photo.jpg | chop save out.png           # Save command group
"""

from __future__ import annotations

import argparse
import sys

from chop.output import handle_output
from chop.pipeline import (
    PipelineState,
    read_pipeline_input,
)


def require_pipeline_input(command: str) -> PipelineState:
    """Read pipeline state from stdin, raising if not available.

    Args:
        command: Command name for error message.

    Returns:
        PipelineState from piped input.

    Raises:
        ValueError: If no piped input is available.
    """
    state = read_pipeline_input()
    if not state:
        raise ValueError(f"{command} requires piped input (use: chop load img.png | chop {command} ...)")
    return state


def cmd_load(args: argparse.Namespace) -> PipelineState:
    """Load image from file, URL, or stdin.

    Creates a lazy pipeline state with the path - image is not loaded yet.
    """
    # Check for piped input first
    prev_state = read_pipeline_input()
    if prev_state:
        # Continue from previous state, add load as an op
        # (This is unusual but supported for re-loading)
        prev_state.add_op("load", args.source)
        return prev_state

    # Create new lazy state with path
    state = PipelineState(
        path=args.source,
        metadata={
            "original_path": args.source if args.source != "-" else "<stdin>",
        },
    )
    return state


def cmd_resize(args: argparse.Namespace) -> PipelineState:
    """Resize image (lazy - just appends operation)."""
    state = require_pipeline_input("resize")
    state.add_op("resize", args.size)
    return state


def cmd_crop(args: argparse.Namespace) -> PipelineState:
    """Crop image (lazy - just appends operation)."""
    state = require_pipeline_input("crop")
    state.add_op("crop", args.x, args.y, args.width, args.height)
    return state


def cmd_rotate(args: argparse.Namespace) -> PipelineState:
    """Rotate image (lazy - just appends operation)."""
    state = require_pipeline_input("rotate")
    state.add_op("rotate", args.degrees)
    return state


def cmd_flip(args: argparse.Namespace) -> PipelineState:
    """Flip image (lazy - just appends operation)."""
    state = require_pipeline_input("flip")
    state.add_op("flip", args.direction)
    return state


def cmd_pad(args: argparse.Namespace) -> PipelineState:
    """Add padding around image (lazy - just appends operation)."""
    state = require_pipeline_input("pad")

    # Build padding args based on how many values provided
    padding = args.padding
    if len(padding) == 1:
        state.add_op("pad", padding[0], color=args.color)
    elif len(padding) == 2:
        state.add_op("pad", padding[0], padding[1], color=args.color)
    elif len(padding) == 4:
        state.add_op("pad", padding[0], padding[1], padding[2], padding[3], color=args.color)
    else:
        raise ValueError("pad requires 1, 2, or 4 values")
    return state


def cmd_border(args: argparse.Namespace) -> PipelineState:
    """Add colored border (lazy - just appends operation)."""
    state = require_pipeline_input("border")
    state.add_op("border", args.width, color=args.color)
    return state


def cmd_fit(args: argparse.Namespace) -> PipelineState:
    """Fit image within bounds (lazy - just appends operation)."""
    state = require_pipeline_input("fit")
    state.add_op("fit", args.size)
    return state


def cmd_fill(args: argparse.Namespace) -> PipelineState:
    """Fill bounds and crop excess (lazy - just appends operation)."""
    state = require_pipeline_input("fill")
    state.add_op("fill", args.size)
    return state


def cmd_hstack(args: argparse.Namespace) -> PipelineState:
    """Stack images horizontally (lazy - just appends operation)."""
    state = require_pipeline_input("hstack")
    state.add_op("hstack", args.path, align=args.align)
    return state


def cmd_vstack(args: argparse.Namespace) -> PipelineState:
    """Stack images vertically (lazy - just appends operation)."""
    state = require_pipeline_input("vstack")
    state.add_op("vstack", args.path, align=args.align)
    return state


def cmd_overlay(args: argparse.Namespace) -> PipelineState:
    """Overlay an image on top (lazy - just appends operation)."""
    state = require_pipeline_input("overlay")
    state.add_op("overlay", args.path, args.x, args.y, opacity=args.opacity, paste=args.paste)
    return state


def cmd_tile(args: argparse.Namespace) -> PipelineState:
    """Tile image NxM times (lazy - just appends operation)."""
    state = require_pipeline_input("tile")
    state.add_op("tile", cols=args.cols, rows=args.rows)
    return state


def cmd_grid(args: argparse.Namespace) -> PipelineState:
    """Arrange images in a grid (lazy - just appends operation)."""
    state = require_pipeline_input("grid")
    state.add_op("grid", args.paths, cols=args.cols)
    return state


def cmd_apply(args: argparse.Namespace) -> PipelineState:
    """Apply a program (sequence of operations) to the pipeline.

    Programs are reusable - they contain only operations, not image paths.
    The image comes from the pipeline.
    """
    from chop.dsl import load_program, parse_program

    state = require_pipeline_input("apply")

    program_text = load_program(args.program)
    ops = parse_program(program_text)

    for name, op_args, kwargs in ops:
        state.add_op(name, *op_args, **kwargs)

    return state


def cmd_save(args: argparse.Namespace) -> None:
    """Save image to file (materializes the pipeline)."""
    state = require_pipeline_input("save")

    # Materialize the pipeline
    image = state.materialize()
    image.save(args.path)
    print(f"Saved to {args.path}", file=sys.stderr)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="chop",
        description="Unix-philosophy image manipulation CLI with lazy evaluation",
        epilog=(
            "Examples:\n"
            "  chop load photo.jpg | chop resize 50%% | chop save out.png\n"
            "  chop load photo.jpg | chop fit 800x600 -j              # Force JSON output\n"
            "  chop load photo.jpg | chop fill 100x100 -o out.png     # Save to file\n"
            "  chop load photo.jpg | chop border 5 --color red -o bordered.png\n"
            "\n"
            "Inspect pipeline JSON:\n"
            "  chop load photo.jpg | chop resize 50%% | chop pad 10 | cat\n"
            '  {"version": 2, "path": "photo.jpg", "ops": [...]}\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Parent parser for common output flags (shared by all subcommands)
    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument("-j", "--json", action="store_true", help="Force JSON output (even on TTY)")
    output_parent.add_argument("-o", "--output", type=str, help="Save to file (png, jpg, etc.)")

    # Also add to main parser for no-command case
    parser.add_argument("-j", "--json", action="store_true", help="Force JSON output (even on TTY)")
    parser.add_argument("-o", "--output", type=str, help="Save to file (png, jpg, etc.)")

    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # load
    load_parser = subparsers.add_parser("load", help="Load image from file, URL, or stdin", parents=[output_parent])
    load_parser.add_argument("source", help="File path, URL, or '-' for stdin")

    # resize
    resize_parser = subparsers.add_parser("resize", help="Resize image", parents=[output_parent])
    resize_parser.add_argument("size", help="Size: 50%%, 800x600, w800, h600")

    # crop
    crop_parser = subparsers.add_parser("crop", help="Crop image", parents=[output_parent])
    crop_parser.add_argument("x", help="Left edge (pixels or %%)")
    crop_parser.add_argument("y", help="Top edge (pixels or %%)")
    crop_parser.add_argument("width", help="Width (pixels or %%)")
    crop_parser.add_argument("height", help="Height (pixels or %%)")

    # rotate
    rotate_parser = subparsers.add_parser("rotate", help="Rotate image", parents=[output_parent])
    rotate_parser.add_argument("degrees", type=float, help="Rotation angle (counter-clockwise)")

    # flip
    flip_parser = subparsers.add_parser("flip", help="Flip image", parents=[output_parent])
    flip_parser.add_argument("direction", choices=["h", "v"], help="h=horizontal, v=vertical")

    # pad
    pad_parser = subparsers.add_parser("pad", help="Add padding around image", parents=[output_parent])
    pad_parser.add_argument(
        "padding",
        type=int,
        nargs="+",
        help="Padding: 1 value (uniform), 2 (vert horiz), or 4 (top right bottom left)",
    )
    pad_parser.add_argument(
        "--color",
        default="transparent",
        help="Padding color (name, hex, or 'transparent')",
    )

    # border
    border_parser = subparsers.add_parser("border", help="Add colored border", parents=[output_parent])
    border_parser.add_argument("width", type=int, help="Border width in pixels")
    border_parser.add_argument(
        "--color",
        default="black",
        help="Border color (name or hex, default: black)",
    )

    # fit
    fit_parser = subparsers.add_parser("fit", help="Fit within bounds, preserve aspect", parents=[output_parent])
    fit_parser.add_argument("size", help="Target size as WxH (e.g., 800x600)")

    # fill
    fill_parser = subparsers.add_parser("fill", help="Fill bounds and crop excess", parents=[output_parent])
    fill_parser.add_argument("size", help="Target size as WxH (e.g., 800x600)")

    # hstack
    hstack_parser = subparsers.add_parser("hstack", help="Stack images horizontally", parents=[output_parent])
    hstack_parser.add_argument("path", help="Image to stack on the right")
    hstack_parser.add_argument(
        "--align",
        choices=["top", "center", "bottom"],
        default="center",
        help="Vertical alignment (default: center)",
    )

    # vstack
    vstack_parser = subparsers.add_parser("vstack", help="Stack images vertically", parents=[output_parent])
    vstack_parser.add_argument("path", help="Image to stack below")
    vstack_parser.add_argument(
        "--align",
        choices=["left", "center", "right"],
        default="center",
        help="Horizontal alignment (default: center)",
    )

    # overlay
    overlay_parser = subparsers.add_parser("overlay", help="Overlay an image", parents=[output_parent])
    overlay_parser.add_argument("path", help="Image to overlay")
    overlay_parser.add_argument("x", type=int, help="X position")
    overlay_parser.add_argument("y", type=int, help="Y position")
    overlay_parser.add_argument(
        "--opacity",
        type=float,
        default=1.0,
        help="Opacity multiplier (0.0-1.0, default: 1.0)",
    )
    overlay_parser.add_argument(
        "--paste",
        action="store_true",
        help="Hard paste without alpha blending",
    )

    # tile
    tile_parser = subparsers.add_parser("tile", help="Tile image NxM times", parents=[output_parent])
    tile_parser.add_argument("cols", type=int, help="Number of columns")
    tile_parser.add_argument("rows", type=int, help="Number of rows")

    # grid
    grid_parser = subparsers.add_parser("grid", help="Arrange images in a grid", parents=[output_parent])
    grid_parser.add_argument("paths", nargs="+", help="Additional images for grid")
    grid_parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of columns (default: 2)",
    )

    # apply
    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply a program (file or inline) to the pipeline",
        parents=[output_parent],
    )
    apply_parser.add_argument(
        "program",
        help="Program: file path or inline 'op1; op2; op3'",
    )

    # save (command group for saving to file)
    save_parser = subparsers.add_parser("save", help="Save to file")
    save_parser.add_argument("path", help="Output file path")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Map commands to handlers that return PipelineState
    state_handlers = {
        "load": cmd_load,
        "resize": cmd_resize,
        "crop": cmd_crop,
        "rotate": cmd_rotate,
        "flip": cmd_flip,
        "pad": cmd_pad,
        "border": cmd_border,
        "fit": cmd_fit,
        "fill": cmd_fill,
        "hstack": cmd_hstack,
        "vstack": cmd_vstack,
        "overlay": cmd_overlay,
        "tile": cmd_tile,
        "grid": cmd_grid,
        "apply": cmd_apply,
    }

    # Terminal commands (save) don't return state
    terminal_handlers = {
        "save": cmd_save,
    }

    if not args.command:
        # No command - check for piped input
        state = read_pipeline_input()
        if state:
            handle_output(state, args)
        else:
            parser.print_help()
        return

    try:
        # Check for terminal commands first
        if args.command in terminal_handlers:
            terminal_handlers[args.command](args)
            return

        # State-returning handlers
        handler = state_handlers.get(args.command)
        if not handler:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

        state = handler(args)

        # Use centralized output handling
        handle_output(state, args)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
