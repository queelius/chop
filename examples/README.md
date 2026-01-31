# chop examples

Example images and ready-to-run chop programs.

## Images

| File | Description |
|------|-------------|
| `landscape.png` | Sunrise over the sea (640×686) |
| `flowers.png` | Sunflower close-up (480×512) |
| `cat.png` | Orange tabby cat (481×599) |
| `city.png` | Times Square at night (525×787) |
| `mountain.png` | Mount Everest (640×426) |

## Programs

Run any `.sh` script from this directory:

```bash
cd examples
bash 01-polaroid.sh
```

Each script is self-contained and produces output in `output/`.

| Script | Features demonstrated |
|--------|---------------------|
| `01-polaroid.sh` | pad, border, rotate |
| `02-contact-sheet.sh` | grid, multi-load |
| `03-diptych.sh` | hstack, labels |
| `04-thumbnail-strip.sh` | hstack, fit |
| `05-moody-city.sh` | saturation, contrast, brightness |
| `06-dreamy-flowers.sh` | blur, brightness |
| `07-multi-save-sizes.sh` | multi-save chaining |
| `08-tiled-pattern.sh` | tile |
| `09-before-after.sh` | hstack, grayscale |
| `10-recipe-reuse.sh` | apply, unbound programs |
| `11-vstack-triptych.sh` | vstack, fit |
| `12-inverted-art.sh` | invert, border |
| `13-sepia-vintage.sh` | grayscale, colorize |
| `14-circle-avatar.sh` | mask circle, background |
| `15-rounded-card.sh` | mask roundrect, pad |
| `16-gapped-comparison.sh` | hstack --gap --gap-color |
| `17-canvas-composition.sh` | canvas, overlay, mask |
| `18-watermark-opacity.sh` | opacity, overlay |
