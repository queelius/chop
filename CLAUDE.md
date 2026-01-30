# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is chop?

A Unix-philosophy image manipulation CLI with lazy evaluation, JSON piping, and multi-image composition. Operations are recorded (not applied) and only materialized at save time. Supports labeled image context, cursor semantics, and unbound programs (reusable recipes).

```bash
# Single image
chop load photo.jpg | chop resize 50% | chop pad 10 | chop save out.png

# Multi-image composition
chop load --as bg photo.jpg | chop load --as fg logo.png \
    | chop resize 50% --on fg | chop overlay bg fg | chop save out.png

# Unbound programs (reusable recipes)
chop resize 50% | chop pad 10 -j > recipe.json
chop load photo.jpg | chop apply recipe.json | chop save out.png
```

## Commands

```bash
# Run all tests (verbose by default via pyproject.toml addopts)
pytest

# Run a single test class or test
pytest tests/test_chop.py::TestOperations
pytest tests/test_chop.py::TestMultiImagePipeline

# Install in development mode
pip install -e .
```

## Architecture

Four modules, each with a single responsibility:

- **`pipeline.py`** — Multi-image composition engine. `PipelineState` stores `ops` + `metadata` (no path). `materialize()` executes ops against a labeled context (`dict[str, Image]`) with cursor tracking. `load` is an op, not a field. JSON serialization is version 3 format: `{version, ops, metadata}`. `execute_composition()` resolves label args for multi-image ops.

- **`operations.py`** — Pure image functions (`Image → Image`). Transform ops registered in `OPERATIONS` dict. `COMPOSITION_OPS` set identifies multi-image ops (hstack, vstack, overlay, grid) dispatched by `execute_composition()` in pipeline.py, not through `apply_operation()`.

- **`cli.py`** — argparse-based CLI. `get_or_create_state()` reads piped input or creates fresh state. Each handler returns `PipelineState` (chainable) or `None` (terminal). Transforms have `--on` flag, load/composition have `--as` flag. New commands: `select`, `dup`, `info`.

- **`output.py`** — Output decision logic: `-j` forces JSON; `-o FILE` materializes to file (bound only); TTY shows info (bound) or program listing (unbound); piped stdout emits JSON.

## Key Design Patterns

**Multi-image context with cursor:** `materialize()` maintains a `dict[str, Image]` context and a cursor string. `load` adds images and sets cursor. `--on` targets transforms at specific labels. `--as` names load/composition results.

**Load is an op:** `PipelineState` has no `path` field. `load` is `["load", ["photo.jpg"], {"as": "bg"}]` in the ops list. This enables unbound programs (ops without load) that can be saved as JSON and applied later.

**Composition via labels:** Composition ops (hstack, vstack, overlay, grid) take label arguments referencing images in context. No label args → all context images in insertion order (excluding `_`). Result stored as `_` by default, overridable with `--as`.

**Auto-labeling:** Loads without `--as` get auto-labels: `img`, `img2`, `img3`... Explicit `--as` does not consume the counter.

**Output hierarchy:** `-j` flag → `-o` file → TTY detection → piped JSON. Unbound pipelines on TTY show program listing. Unbound + `-o` is an error.

## Adding a New Transform Operation

1. Add the function to `operations.py` (signature: `(image: Image, ...) -> Image`)
2. Register it in the `OPERATIONS` dict
3. Add a subcommand + handler in `cli.py` (use `on_parent` for `--on` support)
4. Add tests in `tests/test_chop.py`

## Adding a New Composition Operation

1. Add the function to `operations.py`
2. Add its name to `COMPOSITION_OPS` set
3. Add dispatch case in `execute_composition()` in `pipeline.py`
4. Add a subcommand + handler in `cli.py` (use `as_parent` for `--as` support)
5. Add tests in `tests/test_chop.py`

## Dependencies

Runtime: `numpy>=1.20`, `pillow>=9.0`. Python 3.10+.
