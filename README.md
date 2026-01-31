# chop

A Unix-philosophy image manipulation CLI. Every command reads JSON, writes JSON, does one thing.

```bash
# Single image
chop load photo.jpg | chop resize 50% | chop save out.png

# Color adjustments
chop load photo.jpg | chop brightness 1.5 | chop grayscale | chop save out.png

# Multi-image composition
chop load --as bg photo.jpg | chop load --as fg logo.png \
    | chop resize 50% --on fg | chop overlay bg fg | chop save out.png
```

## The Design

### Unix Philosophy

Each `chop` invocation is a filter. It reads pipeline state as JSON from stdin, appends its operation, and writes the updated state as JSON to stdout. Composition happens through pipes, not flags.

```bash
# Every intermediate step is inspectable
chop load photo.jpg | tee step1.json | chop resize 50% | tee step2.json | chop save out.png

# Integrate with jq
chop load photo.jpg | chop info | jq '.metadata.width'

# Shell scripting
for f in *.jpg; do
    chop load "$f" | chop resize 50% | chop pad 10 | chop save "thumb_$f"
done
```

Side effects — file writes, diagnostic messages — go to stderr. Stdout is reserved exclusively for JSON pipeline state. No modes, no configuration files, no global state.

### Uniform Output (SICP Principle)

Every command behaves the same way: read pipeline state from stdin, write pipeline state as JSON to stdout. No exceptions by category — `load`, `resize`, `save`, `info`, `print` all follow the same contract.

This means `save` writes a file to disk (side effect) **and** outputs JSON (so the pipeline continues). `info` prints diagnostics to stderr **and** outputs JSON. `print` writes a message to stderr **and** outputs JSON.

```bash
# Multi-save: save is non-terminal
chop load photo.jpg | chop save full.png | chop resize 50% | chop save thumb.png

# Info in the middle of a pipeline
chop load photo.jpg | chop resize 50% | chop info | chop pad 10 | chop save out.png

# Debug with print
chop load photo.jpg | chop print "before resize" | chop resize 50% | chop save out.png
```

The single exception: `save -` writes binary image data to stdout (because binary bytes and JSON can't share a stream). This is the only command that terminates the pipeline.

### Lazy Evaluation

Operations are recorded, not executed. The pipeline state is a list of `[name, args, kwargs]` triples — no pixels are touched until `save` or `info` triggers materialization.

This enables **unbound programs**: pipelines without a `load` that can be saved as JSON and applied later.

```bash
# Create a reusable recipe
chop resize 50% | chop pad 10 | chop border 2 > recipe.json

# Apply it to any image
chop load photo.jpg | chop apply recipe.json | chop save out.png
chop load other.jpg | chop apply recipe.json | chop save out2.png
```

## Installation

```bash
pip install chop
```

Development:

```bash
git clone https://github.com/spinoza/chop.git
cd chop
pip install -e .
```

Requires Python 3.10+, numpy, and Pillow.

## Usage

### Single Image Pipeline

```bash
chop load photo.jpg | chop resize 800x600 | chop save out.png
chop load photo.jpg | chop resize w800 | chop save out.png    # width-only, keep aspect
chop load photo.jpg | chop resize h600 | chop save out.png    # height-only, keep aspect
chop load photo.jpg | chop fit 800x600 | chop save out.png    # fit within bounds
chop load photo.jpg | chop fill 800x600 | chop save out.png   # fill bounds, crop excess
```

### Color Operations

All color operations take a factor argument (1.0 = original).

```bash
chop load photo.jpg | chop brightness 1.5 | chop save bright.png
chop load photo.jpg | chop contrast 1.3 | chop save crisp.png
chop load photo.jpg | chop saturation 0.5 | chop save muted.png
chop load photo.jpg | chop sharpen 2.0 | chop save sharp.png
chop load photo.jpg | chop blur 3.0 | chop save soft.png
chop load photo.jpg | chop grayscale | chop save gray.png
chop load photo.jpg | chop invert | chop save inverted.png

# Sepia toning
chop load photo.jpg | chop grayscale | chop colorize '#704214' | chop save sepia.png

# Partial color tint
chop load photo.jpg | chop colorize blue --strength 0.3 | chop save cool.png

# Set opacity (for watermarks, overlays)
chop load watermark.png | chop opacity 0.3 | chop save faded.png

# Flatten transparency onto solid color
chop load logo.png | chop background white | chop save logo.jpg

# Auto-crop uniform borders
chop load scan.png | chop trim | chop save clean.png
chop load scan.png | chop trim --fuzz 10 | chop save clean.png

# Shape masks
chop load photo.jpg | chop mask circle | chop save avatar.png
chop load photo.jpg | chop mask roundrect 20 | chop save rounded.png
chop load photo.jpg | chop mask ellipse | chop save oval.png
```

### Multi-Image Composition

Label images with `--as`, target transforms with `--on`:

```bash
# Side-by-side comparison
chop load --as left before.jpg | chop load --as right after.jpg \
    | chop hstack | chop save comparison.png

# Watermark overlay
chop load --as bg photo.jpg | chop load --as wm watermark.png \
    | chop overlay bg wm -x 10 -y 10 --opacity 0.5 | chop save watermarked.png

# Vertical stack
chop load --as a img1.jpg | chop load --as b img2.jpg \
    | chop vstack --align left | chop save stacked.png

# Grid layout
chop load --as a img1.jpg | chop load --as b img2.jpg \
    | chop load --as c img3.jpg | chop load --as d img4.jpg \
    | chop grid --cols 2 | chop save grid.png
```

Spacing between composed images:

```bash
# Gapped side-by-side
chop load --as a img1.jpg | chop load --as b img2.jpg \
    | chop hstack --gap 10 --gap-color white | chop save comparison.png

# Gapped grid
chop load a.jpg | chop load b.jpg | chop load c.jpg | chop load d.jpg \
    | chop grid --cols 2 --gap 5 --gap-color white | chop save grid.png
```

Without label arguments, composition ops use all loaded images in insertion order:

```bash
chop load a.jpg | chop load b.jpg | chop load c.jpg | chop hstack | chop save row.png
```

### Canvas (Blank Image Source)

Create blank canvases for composition backgrounds:

```bash
# Colored background with overlay
chop canvas 800x600 --color '#2d5016' --as bg \
    | chop load logo.png --as fg | chop fit 200x200 --on fg \
    | chop overlay bg fg -x 300 -y 200 | chop save composed.png

# Transparent canvas
chop canvas 400x400 --as bg | chop load icon.png --as fg \
    | chop overlay bg fg -x 100 -y 100 | chop save padded-icon.png
```

### Unbound Programs (Recipes)

A pipeline without `load` is an unbound program — a reusable recipe:

```bash
# Save a recipe
chop resize 50% | chop pad 10 | chop grayscale > recipe.json

# Apply to images
chop load photo.jpg | chop apply recipe.json | chop save out.png
```

The JSON file contains only the ops list — no image data, no paths.

### Pipeline Inspection

```bash
# info: materializes the pipeline, prints dimensions to stderr, enriches metadata
chop load photo.jpg | chop resize 50% | chop info | jq '.metadata'

# print: no materialization, just prints a message to stderr
chop load photo.jpg | chop print "checkpoint" | chop resize 50% | chop save out.png

# print without message: shows pipeline summary
chop load photo.jpg | chop resize 50% | chop print
```

### Additional Operations

```bash
chop load photo.jpg | chop crop 10 10 200 150 | chop save cropped.png    # x y w h
chop load photo.jpg | chop crop 10% 10% 80% 80% | chop save cropped.png  # percentage
chop load photo.jpg | chop rotate 90 | chop save rotated.png
chop load photo.jpg | chop flip h | chop save flipped.png                 # h or v
chop load photo.jpg | chop pad 20 | chop save padded.png                  # uniform
chop load photo.jpg | chop pad 10 20 | chop save padded.png               # vert horiz
chop load photo.jpg | chop pad 10 20 30 40 | chop save padded.png         # CSS order
chop load photo.jpg | chop border 3 --color red | chop save bordered.png
chop load photo.jpg | chop tile 3 2 | chop save tiled.png                 # cols rows
```

### Pipeline Management

```bash
# Select a different image as cursor
chop load --as a img1.jpg | chop load --as b img2.jpg | chop select a | chop save out.png

# Duplicate a labeled image
chop load --as orig photo.jpg | chop dup orig copy | chop resize 50% --on copy | chop save out.png

# Save to stdout (binary — terminates pipeline)
chop load photo.jpg | chop resize 50% | chop save - --format png > out.png
```

## Command Reference

### Source Operations

| Command | Args | Flags | Description |
|---------|------|-------|-------------|
| `load` | source | `--as` | Load image from file, URL, or stdin (`-`) |
| `canvas` | WxH | `--as`, `--color` | Create a blank canvas image |
| `apply` | program | | Apply a saved JSON recipe |

### Geometric Transforms

| Command | Args | Flags | Description |
|---------|------|-------|-------------|
| `resize` | size | `--on` | Resize (`50%`, `800x600`, `w800`, `h600`) |
| `crop` | x y w h | `--on` | Crop region (pixels or `%`) |
| `rotate` | degrees | `--on` | Rotate counter-clockwise |
| `flip` | direction | `--on` | Flip (`h` horizontal, `v` vertical) |
| `fit` | WxH | `--on` | Fit within bounds, preserve aspect ratio |
| `fill` | WxH | `--on` | Fill bounds, center-crop excess |
| `pad` | values | `--on`, `--color` | Add padding (1, 2, or 4 values, CSS order) |
| `border` | width | `--on`, `--color` | Add colored border |
| `tile` | cols rows | `--on` | Tile image NxM times |

### Color Operations

| Command | Args | Flags | Description |
|---------|------|-------|-------------|
| `brightness` | factor | `--on` | Adjust brightness (0=black, 1=original, 2=double) |
| `contrast` | factor | `--on` | Adjust contrast (0=grey, 1=original, 2=double) |
| `saturation` | factor | `--on` | Adjust saturation (0=grayscale, 1=original, 2=double) |
| `sharpen` | factor | `--on` | Adjust sharpness (0=blurred, 1=original, 2=sharp) |
| `blur` | radius | `--on` | Gaussian blur (radius in pixels) |
| `grayscale` | | `--on` | Convert to grayscale |
| `invert` | | `--on` | Invert colors |
| `colorize` | color | `--on`, `--strength` | Tint image with a color (preserves luminance) |
| `opacity` | factor | `--on` | Set uniform opacity (0.0=transparent, 1.0=opaque) |
| `background` | color | `--on` | Flatten transparency onto solid color |
| `trim` | | `--on`, `--fuzz` | Auto-crop uniform borders |
| `mask` | shape [radius] | `--on`, `--invert` | Apply shape mask (roundrect, circle, ellipse) |

### Composition

| Command | Args | Flags | Description |
|---------|------|-------|-------------|
| `hstack` | [labels...] | `--as`, `--align`, `--gap`, `--gap-color` | Stack horizontally (top/center/bottom) |
| `vstack` | [labels...] | `--as`, `--align`, `--gap`, `--gap-color` | Stack vertically (left/center/right) |
| `overlay` | [labels...] | `--as`, `-x`, `-y`, `--opacity`, `--paste` | Overlay images |
| `grid` | [labels...] | `--as`, `--cols`, `--gap`, `--gap-color` | Arrange in grid |

### Pipeline Management

| Command | Args | Flags | Description |
|---------|------|-------|-------------|
| `select` | label | | Switch cursor to a labeled image |
| `dup` | source dest | | Duplicate a labeled image |

### Output

| Command | Args | Flags | Description |
|---------|------|-------|-------------|
| `save` | path | `--format` | Save to file or stdout (`-`) |
| `info` | | | Materialize and show dimensions (stderr), enrich metadata |
| `print` | [message] | | Print message or summary to stderr (no materialization) |

## JSON Wire Format

The JSON flowing between commands:

```json
{
  "version": 3,
  "ops": [
    ["load", ["photo.jpg"], {}],
    ["resize", ["50%"], {}],
    ["pad", [10], {"color": "transparent"}]
  ],
  "metadata": {}
}
```

Each operation is a `[name, args, kwargs]` triple. The `metadata` dict is enriched by `info` with `width`, `height`, `mode`, and `images_loaded`.

No image data travels through the pipe — only the recipe. Materialization (actually loading and processing pixels) happens at `save` or `info` time.

## Architecture

Four modules:

- **`pipeline.py`** — Multi-image composition engine. `PipelineState` stores ops + metadata. `materialize()` executes ops against a labeled image context (`dict[str, Image]`) with cursor tracking. Source ops (`load`, `canvas`) add images and set cursor.
- **`operations.py`** — Pure image functions (`Image → Image`). 21 transform ops in `OPERATIONS` dict (geometric, color, trim/alpha/mask), multi-image ops in `COMPOSITION_OPS` set with `--gap`/`--gap-color` support.
- **`cli.py`** — Argparse CLI. Single `handlers` dict dispatches all 34 commands. `--on` targets transforms, `--as` names source/composition results.
- **`output.py`** — One function: write JSON to stdout. Always. 20 lines total.

## License

MIT
