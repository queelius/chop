#!/usr/bin/env bash
# Before/after comparison: original vs processed, side by side
# Shows a grayscale conversion comparison

chop load --as before landscape.png \
    | chop dup before after \
    | chop grayscale --on after \
    | chop contrast 1.3 --on after \
    | chop fit 320x320 --on before \
    | chop fit 320x320 --on after \
    | chop hstack before after \
    | chop pad 8 --color '#222222' \
    | chop save output/before-after.png

echo "Created output/before-after.png" >&2
