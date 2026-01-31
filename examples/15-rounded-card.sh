#!/usr/bin/env bash
# Rounded card: roundrect mask with border and shadow-like padding
# Demonstrates mask roundrect with radius

chop load flowers.png \
    | chop fit 300x300 \
    | chop mask roundrect 30 \
    | chop pad 4 --color '#dddddd' \
    | chop pad 8 --color white \
    | chop save output/rounded-card.png

echo "Created output/rounded-card.png" >&2
