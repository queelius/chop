#!/usr/bin/env bash
# Diptych: two images side-by-side with a divider gap
# Pairs the landscape with the mountain for a nature theme

chop load --as left landscape.png \
    | chop load --as right mountain.png \
    | chop fit 400x400 --on left \
    | chop fit 400x400 --on right \
    | chop hstack left right --align center \
    | chop pad 15 --color white \
    | chop border 2 --color black \
    | chop save output/diptych.png

echo "Created output/diptych.png" >&2
