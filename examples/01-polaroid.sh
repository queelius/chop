#!/usr/bin/env bash
# Polaroid effect: white border with thicker bottom, slight rotation
# Creates a vintage instant-photo look

chop load cat.png \
    | chop fit 400x400 \
    | chop pad 20 20 60 20 --color white \
    | chop border 2 --color '#cccccc' \
    | chop rotate 3 \
    | chop save output/polaroid.png

echo "Created output/polaroid.png" >&2
