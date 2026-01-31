#!/usr/bin/env bash
# Thumbnail strip: resize all images to same height, stack horizontally
# Produces a filmstrip-like banner

chop load --as a landscape.png \
    | chop load --as b flowers.png \
    | chop load --as c cat.png \
    | chop load --as d city.png \
    | chop load --as e mountain.png \
    | chop fit 200x150 --on a \
    | chop fit 200x150 --on b \
    | chop fit 200x150 --on c \
    | chop fit 200x150 --on d \
    | chop fit 200x150 --on e \
    | chop hstack a b --as ab \
    | chop hstack ab c --as abc \
    | chop hstack abc d --as abcd \
    | chop hstack abcd e \
    | chop pad 5 --color black \
    | chop save output/thumbnail-strip.png

echo "Created output/thumbnail-strip.png" >&2
