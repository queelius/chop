#!/usr/bin/env bash
# Watermark with opacity: overlay a faded image on top of another
# Demonstrates opacity operation for watermark-style compositing

chop load landscape.png --as base \
    | chop load cat.png --as mark \
    | chop fit 100x100 --on mark \
    | chop opacity 0.25 --on mark \
    | chop overlay base mark -x 10 -y 10 \
    | chop save output/watermark-opacity.png

echo "Created output/watermark-opacity.png" >&2
