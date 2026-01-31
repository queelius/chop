"""Pipeline utilities for lazy image processing.

Implements a multi-image composition engine with labeled context and cursor
semantics. JSON carries only an operations list — images are loaded and
processed at materialize time.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class PipelineState:
    """Multi-image pipeline state — stores operations and metadata.

    Attributes:
        ops: List of operations to apply, each as (name, args, kwargs)
        metadata: Optional metadata dict
    """

    ops: list[tuple[str, tuple, dict]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_op(self, name: str, *args, **kwargs) -> None:
        """Append operation to the list.

        Args:
            name: Operation name (e.g., "load", "resize", "hstack")
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
        """
        self.ops.append((name, args, kwargs))

    def to_json(self) -> str:
        """Serialize state to JSON string.

        Returns compact JSON with version 3 format:
        {"version": 3, "ops": [...], "metadata": {...}}
        """
        ops_data = []
        for name, args, kwargs in self.ops:
            ops_data.append([name, list(args), kwargs])

        output = {
            "version": 3,
            "ops": ops_data,
            "metadata": self.metadata,
        }
        return json.dumps(output)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineState:
        """Deserialize state from JSON string.

        Supports version 3 format only.
        """
        data = json.loads(json_str)
        version = data.get("version", 1)

        if version == 3:
            ops = []
            for op_data in data.get("ops", []):
                name = op_data[0]
                args = tuple(op_data[1]) if len(op_data) > 1 else ()
                kwargs = op_data[2] if len(op_data) > 2 else {}
                ops.append((name, args, kwargs))

            return cls(
                ops=ops,
                metadata=data.get("metadata", {}),
            )
        else:
            raise ValueError(
                f"Unsupported pipeline version: {version}. "
                "Only version 3 format is supported. "
                "Re-run with the current version of chop."
            )

    def has_load(self) -> bool:
        """Check if any op is a source (load or canvas)."""
        return any(name in ("load", "canvas") for name, _, _ in self.ops)

    def materialize(self) -> Image.Image:
        """Execute the pipeline and return the cursor image.

        Builds a labeled image context by executing ops in order:
        - load: adds image to context, sets cursor
        - select: switches cursor to a label
        - dup: copies an image to a new label
        - composition ops: combine labeled images, store result
        - transform ops: mutate the cursor (or --on target) image

        Returns:
            The image at the cursor position.

        Raises:
            ValueError: If pipeline produces no image.
        """
        from chop.operations import COMPOSITION_OPS, apply_operation

        context: dict[str, Image.Image] = {}
        cursor: str | None = None
        auto_counter = 1

        for op_name, args, kwargs in self.ops:
            kw = dict(kwargs)

            if op_name == "load":
                label = kw.pop("as", None)
                if label is None:
                    label = "img" if auto_counter == 1 else f"img{auto_counter}"
                    auto_counter += 1
                context[label] = load_image(args[0])
                cursor = label

            elif op_name == "canvas":
                from chop.operations import _parse_color

                label = kw.pop("as", None)
                if label is None:
                    label = "img" if auto_counter == 1 else f"img{auto_counter}"
                    auto_counter += 1
                size_str = args[0]
                parts = size_str.lower().split("x")
                w, h = int(parts[0]), int(parts[1])
                color = kw.pop("color", "transparent")
                fill = _parse_color(color)
                context[label] = Image.new("RGBA", (w, h), fill)
                cursor = label

            elif op_name == "select":
                if args[0] not in context:
                    raise ValueError(
                        f"Label '{args[0]}' not found. "
                        f"Available: {', '.join(context)}"
                    )
                cursor = args[0]

            elif op_name == "dup":
                if args[0] not in context:
                    raise ValueError(
                        f"Label '{args[0]}' not found. "
                        f"Available: {', '.join(context)}"
                    )
                context[args[1]] = context[args[0]].copy()

            elif op_name in COMPOSITION_OPS:
                result_label = kw.pop("as", "_")
                result = execute_composition(op_name, args, kw, context)
                context[result_label] = result
                cursor = result_label

            else:
                # Transform ops
                target = kw.pop("on", cursor)
                if target is None or target not in context:
                    raise ValueError(
                        "No current image. Use 'chop load' first "
                        "or --on <label>."
                    )
                context[target] = apply_operation(
                    context[target], op_name, *args, **kw
                )

        if cursor is None or cursor not in context:
            raise ValueError("Pipeline produced no image")
        return context[cursor]


def execute_composition(
    op_name: str,
    args: tuple,
    kwargs: dict,
    context: dict[str, Image.Image],
) -> Image.Image:
    """Execute a composition operation using labeled images from context.

    If label args are provided, use those images. If no label args,
    use all context images in insertion order (excluding '_').

    Args:
        op_name: Composition op name (hstack, vstack, overlay, grid)
        args: Label arguments from the op
        kwargs: Additional keyword arguments (align, cols, gap, etc.)
        context: The labeled image context

    Returns:
        Composed PIL Image
    """
    from chop.operations import op_grid, op_hstack, op_overlay, op_vstack

    # Resolve images from labels
    if args:
        labels = list(args)
    else:
        # All context images in insertion order, excluding '_'
        labels = [k for k in context if k != "_"]

    images = []
    for label in labels:
        if label not in context:
            raise ValueError(
                f"Label '{label}' not found. "
                f"Available: {', '.join(context)}"
            )
        images.append(context[label])

    if not images:
        raise ValueError(f"{op_name} requires at least one image")

    if op_name == "hstack":
        align = kwargs.get("align", "center")
        gap = kwargs.get("gap", 0)
        gap_color = kwargs.get("gap_color", "transparent")
        result = images[0]
        for img in images[1:]:
            result = op_hstack(result, img, align=align, gap=gap, gap_color=gap_color)
        return result

    if op_name == "vstack":
        align = kwargs.get("align", "center")
        gap = kwargs.get("gap", 0)
        gap_color = kwargs.get("gap_color", "transparent")
        result = images[0]
        for img in images[1:]:
            result = op_vstack(result, img, align=align, gap=gap, gap_color=gap_color)
        return result

    if op_name == "overlay":
        if len(images) < 2:
            raise ValueError("overlay requires at least 2 images (base, overlay)")
        base = images[0]
        overlay_img = images[1]
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        opacity = kwargs.get("opacity", 1.0)
        paste = kwargs.get("paste", False)
        return op_overlay(base, overlay_img, x=x, y=y, opacity=opacity, paste=paste)

    if op_name == "grid":
        cols = kwargs.get("cols", 2)
        gap = kwargs.get("gap", 0)
        gap_color = kwargs.get("gap_color", "transparent")
        return op_grid(images[0], images[1:], cols=cols, gap=gap, gap_color=gap_color)

    raise ValueError(f"Unknown composition operation: {op_name}")


def read_pipeline_input() -> PipelineState | None:
    """Read pipeline state from stdin if available.

    Returns:
        PipelineState if valid JSON input, None otherwise.
    """
    if sys.stdin.isatty():
        return None

    try:
        data = sys.stdin.read()
        if not data.strip():
            return None
        return PipelineState.from_json(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def write_pipeline_output(state: PipelineState) -> None:
    """Write pipeline state to stdout."""
    print(state.to_json())


def load_image(source: str) -> Image.Image:
    """Load image from file, URL, or stdin.

    Args:
        source: File path, URL, or "-" for stdin.

    Returns:
        PIL Image in RGBA mode.
    """
    if source == "-":
        image_bytes = sys.stdin.buffer.read()
        image = Image.open(io.BytesIO(image_bytes))
    elif source.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(source) as response:
            image_bytes = response.read()
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = Image.open(source)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    return image


def image_to_arrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Convert PIL Image to bitmap and colors arrays.

    Args:
        image: PIL Image in RGBA mode

    Returns:
        (bitmap, colors) tuple where:
        - bitmap is 2D float32 (H, W) with luminance 0.0-1.0
        - colors is 3D float32 (H, W, 3) with RGB 0.0-1.0
    """
    arr = np.array(image, dtype=np.float32) / 255.0

    if arr.ndim == 2:
        bitmap = arr
        colors = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        bitmap = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        colors = rgb
    elif arr.shape[2] == 3:
        bitmap = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        colors = arr
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    return bitmap.astype(np.float32), colors.astype(np.float32)
