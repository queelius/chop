# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is chop?

A Unix-philosophy image manipulation CLI with lazy evaluation, JSON piping, and multi-image composition. Operations are recorded (not applied) and only materialized at save time. Every command outputs JSON to stdout; side effects go to stderr or filesystem. Supports labeled image context, cursor semantics, and unbound programs (reusable recipes).

```bash
# Single image
chop load photo.jpg | chop resize 50% | chop pad 10 | chop save out.png

# Multi-image composition
chop load --as bg photo.jpg | chop load --as fg logo.png \
    | chop resize 50% --on fg | chop overlay bg fg | chop save out.png

# Unbound programs (reusable recipes)
chop resize 50% | chop pad 10 > recipe.json
chop load photo.jpg | chop apply recipe.json | chop save out.png

# Color adjustments
chop load photo.jpg | chop brightness 1.5 | chop grayscale | chop save out.png

# Multi-save (save is non-terminal)
chop load photo.jpg | chop save full.png | chop resize 50% | chop save thumb.png

# Sepia toning
chop load photo.jpg | chop grayscale | chop colorize '#704214' | chop save sepia.png

# Circular avatar with white background
chop load photo.jpg | chop mask circle | chop background white | chop save avatar.png

# Canvas-based composition
chop canvas 800x400 --color white --as bg \
    | chop load logo.png --as fg | chop fit 200x200 --on fg \
    | chop overlay bg fg -x 300 -y 100 | chop save composed.png

# Gapped grid
chop load a.png | chop load b.png | chop load c.png | chop load d.png \
    | chop grid --cols 2 --gap 10 --gap-color white | chop save grid.png
```

## Commands

```bash
# Run all tests (verbose by default via pyproject.toml addopts)
pytest

# Run a single test class or test
pytest tests/test_chop.py::TestOperations
pytest tests/test_chop.py::TestColorOperations
pytest tests/test_chop.py::TestMaskOperation
pytest tests/test_chop.py::TestCanvasOperation
pytest tests/test_chop.py::TestGapOnComposition
pytest tests/test_chop.py::TestNewOpsIntegration

# Install in development mode
pip install -e .
```

## Architecture

Four modules, each with a single responsibility:

- **`pipeline.py`** — Multi-image composition engine. `PipelineState` stores `ops` + `metadata` (no path). `materialize()` executes ops against a labeled context (`dict[str, Image]`) with cursor tracking. Source ops (`load`, `canvas`) add images to context and set cursor. JSON serialization is version 3 format: `{version, ops, metadata}`. `execute_composition()` resolves label args for multi-image ops.

- **`operations.py`** — Pure image functions (`Image → Image`). Transform ops registered in `OPERATIONS` dict (geometric: resize, crop, rotate, flip, fit, fill, pad, border, tile; color: brightness, contrast, saturation, sharpen, blur, grayscale, invert; trim/alpha: trim, colorize, opacity, background, mask). `COMPOSITION_OPS` set identifies multi-image ops (hstack, vstack, overlay, grid) dispatched by `execute_composition()` in pipeline.py. Composition ops support `--gap`/`--gap-color` for pixel spacing.

- **`cli.py`** — argparse-based CLI. `get_or_create_state()` reads piped input or creates fresh state. All handlers return `PipelineState` (or `None` for `save -`). Single unified `handlers` dict — no terminal/state split. Transforms have `--on` flag, source/composition ops have `--as` flag. Commands: `select`, `dup`, `info`, `print`, 7 color ops, `canvas`, `trim`, `colorize`, `opacity`, `background`, `mask`.

- **`output.py`** — Always writes JSON to stdout. One function, no flags, no TTY detection.

## Key Design Patterns

**Uniform output:** Every command outputs JSON to stdout. Side effects (file saves, info messages) go to stderr. No `-j`/`-o` flags — JSON is the only output mode. `save <file>` returns state for chaining. `save -` is the single exception (binary stdout, returns `None`).

**Multi-image context with cursor:** `materialize()` maintains a `dict[str, Image]` context and a cursor string. `load` adds images and sets cursor. `--on` targets transforms at specific labels. `--as` names load/composition results.

**Source ops (load, canvas):** `PipelineState` has no `path` field. `load` is `["load", ["photo.jpg"], {"as": "bg"}]` in the ops list. `canvas` creates a blank image: `["canvas", ["800x600"], {"color": "white", "as": "bg"}]`. Both add to context, set cursor, and support auto-labeling. This enables unbound programs (ops without load) that can be saved as JSON and applied later.

**Composition via labels:** Composition ops (hstack, vstack, overlay, grid) take label arguments referencing images in context. No label args → all context images in insertion order (excluding `_`). Result stored as `_` by default, overridable with `--as`. All support `--gap`/`--gap-color` for pixel spacing between composed images.

**Auto-labeling:** Loads without `--as` get auto-labels: `img`, `img2`, `img3`... Explicit `--as` does not consume the counter.

**Metadata enrichment:** `info` enriches `state.metadata` with `width`, `height`, `mode`, `images_loaded` — survives JSON serialization for downstream `jq` queries.

## Adding a New Transform Operation

1. Add the function to `operations.py` (signature: `(image: Image, ...) -> Image`)
2. Register it in the `OPERATIONS` dict
3. Add a subcommand + handler in `cli.py` (use `on_parent` for `--on` support)
4. Add tests in `tests/test_chop.py`

## Adding a New Source Operation

1. Add dispatch case in `materialize()` in `pipeline.py` (alongside `load` and `canvas`)
2. Update `has_load()` to recognize it as a source
3. Add a subcommand + handler in `cli.py` (use `as_parent` for `--as` support)
4. Add tests in `tests/test_chop.py`

## Adding a New Composition Operation

1. Add the function to `operations.py`
2. Add its name to `COMPOSITION_OPS` set
3. Add dispatch case in `execute_composition()` in `pipeline.py`
4. Add a subcommand + handler in `cli.py` (use `as_parent` for `--as` support)
5. Add tests in `tests/test_chop.py`

## Dependencies

Runtime: `numpy>=1.20`, `pillow>=9.0`. Python 3.10+.
