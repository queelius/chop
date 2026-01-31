#!/usr/bin/env bash
# Tiled pattern: shrink an image and tile it into a repeating pattern
# Creates wallpaper/textile-like output from any photo

chop load flowers.png \
    | chop resize 80x80 \
    | chop tile 6 4 \
    | chop border 3 --color '#2d5016' \
    | chop save output/tiled-pattern.png

echo "Created output/tiled-pattern.png" >&2
