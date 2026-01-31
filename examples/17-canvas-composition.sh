#!/usr/bin/env bash
# Canvas composition: create a background canvas, overlay images onto it
# Demonstrates the canvas source operation

chop canvas 700x400 --color '#2d5016' --as bg \
    | chop load flowers.png --as fg \
    | chop fit 200x200 --on fg \
    | chop mask roundrect 15 --on fg \
    | chop overlay bg fg -x 250 -y 100 \
    | chop save output/canvas-composition.png

echo "Created output/canvas-composition.png" >&2
