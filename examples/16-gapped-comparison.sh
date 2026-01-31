#!/usr/bin/env bash
# Gapped comparison: side-by-side images with white spacing
# Demonstrates --gap and --gap-color on hstack

chop load mountain.png --as left \
    | chop load landscape.png --as right \
    | chop fit 300x200 --on left \
    | chop fit 300x200 --on right \
    | chop hstack left right --gap 10 --gap-color white \
    | chop border 2 --color '#cccccc' \
    | chop save output/gapped-comparison.png

echo "Created output/gapped-comparison.png" >&2
