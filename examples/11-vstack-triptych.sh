#!/usr/bin/env bash
# Vertical triptych: three landscape slices stacked as a tall panel
# Crops wide strips from different images for an art-gallery feel

chop load --as top mountain.png \
    | chop load --as mid landscape.png \
    | chop load --as bot city.png \
    | chop fill 500x500 --on top \
    | chop crop 0 150 500 120 --on top \
    | chop fill 500x500 --on mid \
    | chop crop 0 150 500 120 --on mid \
    | chop fill 500x500 --on bot \
    | chop crop 0 150 500 120 --on bot \
    | chop vstack top mid --as upper \
    | chop vstack upper bot \
    | chop pad 12 --color white \
    | chop border 2 --color '#444444' \
    | chop save output/triptych.png

echo "Created output/triptych.png" >&2
