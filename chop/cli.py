#!/usr/bin/env python3
"""chop - Unix-philosophy image manipulation CLI with lazy evaluation.

Multi-image composition language with labeled context and cursor semantics.
Operations are recorded (not applied) and only materialized at save time.
Every command outputs JSON to stdout. Side effects go to stderr or filesystem.

Single image:
    chop load photo.jpg | chop resize 50% | chop save out.png

Multi-image composition:
    chop load --as bg photo.jpg | chop load --as fg logo.png \
        | chop resize 50% --on fg | chop overlay bg fg | chop save out.png

Unbound programs (reusable recipes):
    chop resize 50% | chop pad 10 > recipe.json
    chop load photo.jpg | chop apply recipe.json | chop save out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chop.output import handle_output
from chop.pipeline import (
    PipelineState,
    read_pipeline_input,
)


def get_or_create_state() -> PipelineState:
    """Read pipeline state from stdin, or create a fresh one.

    Returns:
        PipelineState from piped input, or a new empty state.
    """
    state = read_pipeline_input()
    return state if state is not None else PipelineState()


# =============================================================================
# Command handlers
# =============================================================================


def cmd_load(args: argparse.Namespace) -> PipelineState:
    """Load image into the pipeline context."""
    state = get_or_create_state()
    kwargs = {}
    if args.label:
        kwargs["as"] = args.label
    state.add_op("load", args.source, **kwargs)
    return state


def cmd_canvas(args: argparse.Namespace) -> PipelineState:
    """Create a blank canvas image."""
    state = get_or_create_state()
    kwargs = {}
    if args.color != "transparent":
        kwargs["color"] = args.color
    if args.label:
        kwargs["as"] = args.label
    state.add_op("canvas", args.size, **kwargs)
    return state


def cmd_resize(args: argparse.Namespace) -> PipelineState:
    """Resize image (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("resize", args.size, **kwargs)
    return state


def cmd_crop(args: argparse.Namespace) -> PipelineState:
    """Crop image (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("crop", args.x, args.y, args.width, args.height, **kwargs)
    return state


def cmd_rotate(args: argparse.Namespace) -> PipelineState:
    """Rotate image (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("rotate", args.degrees, **kwargs)
    return state


def cmd_flip(args: argparse.Namespace) -> PipelineState:
    """Flip image (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("flip", args.direction, **kwargs)
    return state


def cmd_pad(args: argparse.Namespace) -> PipelineState:
    """Add padding around image (lazy — appends operation)."""
    state = get_or_create_state()

    kwargs = {"color": args.color}
    if args.on:
        kwargs["on"] = args.on

    padding = args.padding
    if len(padding) == 1:
        state.add_op("pad", padding[0], **kwargs)
    elif len(padding) == 2:
        state.add_op("pad", padding[0], padding[1], **kwargs)
    elif len(padding) == 4:
        state.add_op("pad", padding[0], padding[1], padding[2], padding[3], **kwargs)
    else:
        raise ValueError("pad requires 1, 2, or 4 values")
    return state


def cmd_border(args: argparse.Namespace) -> PipelineState:
    """Add colored border (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {"color": args.color}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("border", args.width, **kwargs)
    return state


def cmd_fit(args: argparse.Namespace) -> PipelineState:
    """Fit image within bounds (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("fit", args.size, **kwargs)
    return state


def cmd_fill(args: argparse.Namespace) -> PipelineState:
    """Fill bounds and crop excess (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("fill", args.size, **kwargs)
    return state


def cmd_tile(args: argparse.Namespace) -> PipelineState:
    """Tile image NxM times (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("tile", cols=args.cols, rows=args.rows, **kwargs)
    return state


def cmd_hstack(args: argparse.Namespace) -> PipelineState:
    """Stack images horizontally (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.align:
        kwargs["align"] = args.align
    if args.gap:
        kwargs["gap"] = args.gap
    if args.gap_color != "transparent":
        kwargs["gap_color"] = args.gap_color
    if args.label:
        kwargs["as"] = args.label
    state.add_op("hstack", *args.images, **kwargs)
    return state


def cmd_vstack(args: argparse.Namespace) -> PipelineState:
    """Stack images vertically (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.align:
        kwargs["align"] = args.align
    if args.gap:
        kwargs["gap"] = args.gap
    if args.gap_color != "transparent":
        kwargs["gap_color"] = args.gap_color
    if args.label:
        kwargs["as"] = args.label
    state.add_op("vstack", *args.images, **kwargs)
    return state


def cmd_overlay(args: argparse.Namespace) -> PipelineState:
    """Overlay an image on top (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.x is not None:
        kwargs["x"] = args.x
    if args.y is not None:
        kwargs["y"] = args.y
    if args.opacity != 1.0:
        kwargs["opacity"] = args.opacity
    if args.paste:
        kwargs["paste"] = True
    if args.label:
        kwargs["as"] = args.label
    state.add_op("overlay", *args.images, **kwargs)
    return state


def cmd_grid(args: argparse.Namespace) -> PipelineState:
    """Arrange images in a grid (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {"cols": args.cols}
    if args.gap:
        kwargs["gap"] = args.gap
    if args.gap_color != "transparent":
        kwargs["gap_color"] = args.gap_color
    if args.label:
        kwargs["as"] = args.label
    state.add_op("grid", *args.images, **kwargs)
    return state


def cmd_select(args: argparse.Namespace) -> PipelineState:
    """Switch cursor to a labeled image."""
    state = get_or_create_state()
    state.add_op("select", args.label)
    return state


def cmd_dup(args: argparse.Namespace) -> PipelineState:
    """Duplicate a labeled image."""
    state = get_or_create_state()
    state.add_op("dup", args.source, args.dest)
    return state


def cmd_apply(args: argparse.Namespace) -> PipelineState:
    """Apply a saved program (JSON file) to the pipeline."""
    state = get_or_create_state()
    program = PipelineState.from_json(Path(args.program).read_text())
    for name, op_args, kwargs in program.ops:
        state.add_op(name, *op_args, **kwargs)
    return state


# =============================================================================
# Color operation handlers
# =============================================================================


def cmd_brightness(args: argparse.Namespace) -> PipelineState:
    """Adjust brightness (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("brightness", args.factor, **kwargs)
    return state


def cmd_contrast(args: argparse.Namespace) -> PipelineState:
    """Adjust contrast (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("contrast", args.factor, **kwargs)
    return state


def cmd_saturation(args: argparse.Namespace) -> PipelineState:
    """Adjust saturation (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("saturation", args.factor, **kwargs)
    return state


def cmd_sharpen(args: argparse.Namespace) -> PipelineState:
    """Adjust sharpness (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("sharpen", args.factor, **kwargs)
    return state


def cmd_blur(args: argparse.Namespace) -> PipelineState:
    """Apply Gaussian blur (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("blur", args.radius, **kwargs)
    return state


def cmd_grayscale(args: argparse.Namespace) -> PipelineState:
    """Convert to grayscale (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("grayscale", **kwargs)
    return state


def cmd_invert(args: argparse.Namespace) -> PipelineState:
    """Invert colors (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("invert", **kwargs)
    return state


# =============================================================================
# Trim, colorize, opacity, background handlers
# =============================================================================


def cmd_trim(args: argparse.Namespace) -> PipelineState:
    """Auto-crop uniform borders (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    if args.fuzz:
        kwargs["fuzz"] = args.fuzz
    state.add_op("trim", **kwargs)
    return state


def cmd_colorize(args: argparse.Namespace) -> PipelineState:
    """Tint image with a color (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    if args.strength != 1.0:
        kwargs["strength"] = args.strength
    state.add_op("colorize", args.color, **kwargs)
    return state


def cmd_opacity(args: argparse.Namespace) -> PipelineState:
    """Set image opacity (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("opacity", args.factor, **kwargs)
    return state


def cmd_background(args: argparse.Namespace) -> PipelineState:
    """Flatten transparency onto solid color (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    state.add_op("background", args.color, **kwargs)
    return state


def cmd_mask(args: argparse.Namespace) -> PipelineState:
    """Apply shape mask (lazy — appends operation)."""
    state = get_or_create_state()
    kwargs = {}
    if args.on:
        kwargs["on"] = args.on
    if args.radius:
        kwargs["radius"] = args.radius
    if args.invert:
        kwargs["invert"] = True
    state.add_op("mask", args.shape, **kwargs)
    return state


# =============================================================================
# Info, save, print handlers (side effects + state passthrough)
# =============================================================================


def cmd_info(args: argparse.Namespace) -> PipelineState:
    """Show pipeline info to stderr, return state for chaining."""
    state = get_or_create_state()

    if not state.ops:
        print("Empty pipeline (no operations)", file=sys.stderr)
        return state

    if not state.has_load():
        # Unbound program — show ops summary
        print("Unbound program (no load):", file=sys.stderr)
        for name, op_args, kwargs in state.ops:
            parts = [name] + [str(a) for a in op_args]
            for k, v in kwargs.items():
                parts.append(f"{k}={v}")
            print(f"  {' '.join(parts)}", file=sys.stderr)
        return state

    # Bound pipeline — materialize and show context info
    image = state.materialize()

    load_count = sum(1 for name, _, _ in state.ops if name == "load")
    w, h = image.size
    print(f"Cursor image: {w}x{h} {image.mode}", file=sys.stderr)
    print(f"Operations: {len(state.ops)}", file=sys.stderr)
    if load_count > 1:
        print(f"Images loaded: {load_count}", file=sys.stderr)

    state.metadata.update({
        "width": w, "height": h, "mode": image.mode,
        "images_loaded": load_count,
    })
    return state


def cmd_save(args: argparse.Namespace) -> PipelineState | None:
    """Save image to file or stdout (materializes the pipeline).

    Returns state for chaining, or None for binary stdout (save -).
    """
    state = get_or_create_state()

    if not state.ops:
        raise ValueError("Nothing to save (empty pipeline)")

    if not state.has_load():
        raise ValueError(
            "Cannot save: pipeline has no load. "
            "Use 'chop load <file> | ... | chop save <output>'"
        )

    image = state.materialize()

    if args.path == "-":
        # Save to stdout — binary, no JSON after this
        fmt = args.format
        if not fmt:
            raise ValueError("--format is required when saving to stdout (e.g., --format png)")
        image.save(sys.stdout.buffer, format=fmt.upper())
        return None

    image.save(args.path)
    print(f"Saved to {args.path}", file=sys.stderr)
    return state


def cmd_print(args: argparse.Namespace) -> PipelineState:
    """Print pipeline feedback to stderr (no materialization)."""
    state = get_or_create_state()
    msg = args.message if args.message else None
    if msg:
        print(msg, file=sys.stderr)
    else:
        n = len(state.ops)
        bound = "bound" if state.has_load() else "unbound"
        print(f"Pipeline: {n} ops ({bound})", file=sys.stderr)
    return state


# =============================================================================
# Parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="chop",
        description="Unix-philosophy image manipulation CLI with multi-image composition",
        epilog=(
            "Examples:\n"
            "  # Single image pipeline\n"
            "  chop load photo.jpg | chop resize 50%% | chop save out.png\n"
            "\n"
            "  # Multi-image composition\n"
            "  chop load --as bg photo.jpg | chop load --as fg logo.png \\\n"
            "      | chop resize 50%% --on fg | chop overlay bg fg | chop save out.png\n"
            "\n"
            "  # Unbound programs (reusable recipes)\n"
            "  chop resize 50%% | chop pad 10 > recipe.json\n"
            "  chop load photo.jpg | chop apply recipe.json | chop save out.png\n"
            "\n"
            "  # Color adjustments\n"
            "  chop load photo.jpg | chop brightness 1.5 | chop contrast 1.2 | chop save out.png\n"
            "\n"
            "  # Multi-save (save is non-terminal)\n"
            "  chop load photo.jpg | chop save full.png | chop resize 50%% | chop save thumb.png\n"
            "\n"
            "  # Save to stdout\n"
            "  chop load photo.jpg | chop resize 50%% | chop save - --format png | file -\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Parent parser for --on flag (shared by all transforms)
    on_parent = argparse.ArgumentParser(add_help=False)
    on_parent.add_argument("--on", type=str, default=None, help="Target a specific labeled image instead of cursor")

    # Parent parser for --as flag (shared by load + composition)
    as_parent = argparse.ArgumentParser(add_help=False)
    as_parent.add_argument("--as", type=str, default=None, dest="label", help="Name the result image")

    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # load
    subparsers.add_parser(
        "load", help="Load image from file, URL, or stdin",
        parents=[as_parent],
    ).add_argument("source", help="File path, URL, or '-' for stdin")

    # canvas
    canvas_parser = subparsers.add_parser(
        "canvas", help="Create a blank canvas image",
        parents=[as_parent],
    )
    canvas_parser.add_argument("size", help="Canvas size as WxH (e.g., 800x600)")
    canvas_parser.add_argument("--color", default="transparent", help="Fill color (name, hex, or 'transparent')")

    # resize
    resize_parser = subparsers.add_parser("resize", help="Resize image", parents=[on_parent])
    resize_parser.add_argument("size", help="Size: 50%%, 800x600, w800, h600")

    # crop
    crop_parser = subparsers.add_parser("crop", help="Crop image", parents=[on_parent])
    crop_parser.add_argument("x", help="Left edge (pixels or %%)")
    crop_parser.add_argument("y", help="Top edge (pixels or %%)")
    crop_parser.add_argument("width", help="Width (pixels or %%)")
    crop_parser.add_argument("height", help="Height (pixels or %%)")

    # rotate
    rotate_parser = subparsers.add_parser("rotate", help="Rotate image", parents=[on_parent])
    rotate_parser.add_argument("degrees", type=float, help="Rotation angle (counter-clockwise)")

    # flip
    flip_parser = subparsers.add_parser("flip", help="Flip image", parents=[on_parent])
    flip_parser.add_argument("direction", choices=["h", "v"], help="h=horizontal, v=vertical")

    # pad
    pad_parser = subparsers.add_parser("pad", help="Add padding around image", parents=[on_parent])
    pad_parser.add_argument(
        "padding", type=int, nargs="+",
        help="Padding: 1 value (uniform), 2 (vert horiz), or 4 (top right bottom left)",
    )
    pad_parser.add_argument("--color", default="transparent", help="Padding color (name, hex, or 'transparent')")

    # border
    border_parser = subparsers.add_parser("border", help="Add colored border", parents=[on_parent])
    border_parser.add_argument("width", type=int, help="Border width in pixels")
    border_parser.add_argument("--color", default="black", help="Border color (name or hex, default: black)")

    # fit
    fit_parser = subparsers.add_parser("fit", help="Fit within bounds, preserve aspect", parents=[on_parent])
    fit_parser.add_argument("size", help="Target size as WxH (e.g., 800x600)")

    # fill
    fill_parser = subparsers.add_parser("fill", help="Fill bounds and crop excess", parents=[on_parent])
    fill_parser.add_argument("size", help="Target size as WxH (e.g., 800x600)")

    # tile
    tile_parser = subparsers.add_parser("tile", help="Tile image NxM times", parents=[on_parent])
    tile_parser.add_argument("cols", type=int, help="Number of columns")
    tile_parser.add_argument("rows", type=int, help="Number of rows")

    # --- Color operations ---

    # brightness
    brightness_parser = subparsers.add_parser("brightness", help="Adjust brightness", parents=[on_parent])
    brightness_parser.add_argument("factor", type=float, help="Factor (0=black, 1=original, 2=double)")

    # contrast
    contrast_parser = subparsers.add_parser("contrast", help="Adjust contrast", parents=[on_parent])
    contrast_parser.add_argument("factor", type=float, help="Factor (0=grey, 1=original, 2=double)")

    # saturation
    saturation_parser = subparsers.add_parser("saturation", help="Adjust color saturation", parents=[on_parent])
    saturation_parser.add_argument("factor", type=float, help="Factor (0=grayscale, 1=original, 2=double)")

    # sharpen
    sharpen_parser = subparsers.add_parser("sharpen", help="Adjust sharpness", parents=[on_parent])
    sharpen_parser.add_argument("factor", type=float, help="Factor (0=blurred, 1=original, 2=sharpened)")

    # blur
    blur_parser = subparsers.add_parser("blur", help="Apply Gaussian blur", parents=[on_parent])
    blur_parser.add_argument("radius", type=float, help="Blur radius in pixels")

    # grayscale
    subparsers.add_parser("grayscale", help="Convert to grayscale", parents=[on_parent])

    # invert
    subparsers.add_parser("invert", help="Invert colors", parents=[on_parent])

    # --- Trim, colorize, opacity, background ---

    # trim
    trim_parser = subparsers.add_parser("trim", help="Auto-crop uniform borders", parents=[on_parent])
    trim_parser.add_argument("--fuzz", type=int, default=0, help="Color tolerance 0-255 (default: 0)")

    # colorize
    colorize_parser = subparsers.add_parser("colorize", help="Tint image with a color", parents=[on_parent])
    colorize_parser.add_argument("color", help="Target color (name or hex)")
    colorize_parser.add_argument("--strength", type=float, default=1.0, help="Blend strength 0.0-1.0 (default: 1.0)")

    # opacity
    opacity_parser = subparsers.add_parser("opacity", help="Set image opacity", parents=[on_parent])
    opacity_parser.add_argument("factor", type=float, help="Opacity 0.0 (transparent) to 1.0 (opaque)")

    # background
    bg_parser = subparsers.add_parser("background", help="Flatten transparency onto solid color", parents=[on_parent])
    bg_parser.add_argument("color", help="Background color (name or hex)")

    # mask
    mask_parser = subparsers.add_parser("mask", help="Apply shape mask (roundrect, circle, ellipse)", parents=[on_parent])
    mask_parser.add_argument("shape", choices=["roundrect", "circle", "ellipse"], help="Mask shape")
    mask_parser.add_argument("radius", type=int, nargs="?", default=0, help="Corner radius for roundrect")
    mask_parser.add_argument("--invert", action="store_true", help="Invert mask (inside transparent)")

    # --- Composition operations ---

    # hstack
    hstack_parser = subparsers.add_parser(
        "hstack", help="Stack images horizontally",
        parents=[as_parent],
    )
    hstack_parser.add_argument("images", nargs="*", help="Image labels to stack (default: all)")
    hstack_parser.add_argument("--align", choices=["top", "center", "bottom"], default="center", help="Vertical alignment")
    hstack_parser.add_argument("--gap", type=int, default=0, help="Pixel spacing between images")
    hstack_parser.add_argument("--gap-color", default="transparent", help="Color for gap space (default: transparent)")

    # vstack
    vstack_parser = subparsers.add_parser(
        "vstack", help="Stack images vertically",
        parents=[as_parent],
    )
    vstack_parser.add_argument("images", nargs="*", help="Image labels to stack (default: all)")
    vstack_parser.add_argument("--align", choices=["left", "center", "right"], default="center", help="Horizontal alignment")
    vstack_parser.add_argument("--gap", type=int, default=0, help="Pixel spacing between images")
    vstack_parser.add_argument("--gap-color", default="transparent", help="Color for gap space (default: transparent)")

    # overlay
    overlay_parser = subparsers.add_parser(
        "overlay", help="Overlay images",
        parents=[as_parent],
    )
    overlay_parser.add_argument("images", nargs="*", help="Image labels: base overlay")
    overlay_parser.add_argument("-x", type=int, default=None, help="X position for overlay")
    overlay_parser.add_argument("-y", type=int, default=None, help="Y position for overlay")
    overlay_parser.add_argument("--opacity", type=float, default=1.0, help="Opacity (0.0-1.0)")
    overlay_parser.add_argument("--paste", action="store_true", help="Hard paste without alpha blending")

    # grid
    grid_parser = subparsers.add_parser(
        "grid", help="Arrange images in a grid",
        parents=[as_parent],
    )
    grid_parser.add_argument("images", nargs="*", help="Image labels for grid (default: all)")
    grid_parser.add_argument("--cols", type=int, default=2, help="Number of columns (default: 2)")
    grid_parser.add_argument("--gap", type=int, default=0, help="Pixel spacing between cells")
    grid_parser.add_argument("--gap-color", default="transparent", help="Color for gap space (default: transparent)")

    # --- Pipeline management ---

    # select
    select_parser = subparsers.add_parser("select", help="Switch cursor to a labeled image")
    select_parser.add_argument("label", help="Image label to select")

    # dup
    dup_parser = subparsers.add_parser("dup", help="Duplicate a labeled image")
    dup_parser.add_argument("source", help="Source label")
    dup_parser.add_argument("dest", help="Destination label")

    # apply
    apply_parser = subparsers.add_parser("apply", help="Apply a saved program (JSON file)")
    apply_parser.add_argument("program", help="JSON program file path")

    # info
    subparsers.add_parser("info", help="Show pipeline info (to stderr)")

    # save
    save_parser = subparsers.add_parser("save", help="Save to file or stdout")
    save_parser.add_argument("path", help="Output file path, or '-' for stdout")
    save_parser.add_argument("--format", type=str, default=None, help="Image format (required for stdout)")

    # print
    print_parser = subparsers.add_parser("print", help="Print pipeline feedback to stderr")
    print_parser.add_argument("message", nargs="?", default=None, help="Custom message (default: pipeline summary)")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Unified handler map — all return PipelineState (or None for save -)
    handlers = {
        "load": cmd_load,
        "canvas": cmd_canvas,
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
        "select": cmd_select,
        "dup": cmd_dup,
        "brightness": cmd_brightness,
        "contrast": cmd_contrast,
        "saturation": cmd_saturation,
        "sharpen": cmd_sharpen,
        "blur": cmd_blur,
        "grayscale": cmd_grayscale,
        "invert": cmd_invert,
        "trim": cmd_trim,
        "colorize": cmd_colorize,
        "opacity": cmd_opacity,
        "background": cmd_background,
        "mask": cmd_mask,
        "info": cmd_info,
        "save": cmd_save,
        "print": cmd_print,
    }

    if not args.command:
        # No command — check for piped input
        state = read_pipeline_input()
        if state:
            handle_output(state)
        else:
            parser.print_help()
        return

    try:
        handler = handlers.get(args.command)
        if not handler:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

        state = handler(args)
        if state is not None:
            handle_output(state)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
