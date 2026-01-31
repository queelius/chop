#!/usr/bin/env bash
# Contact sheet: grid all images into a 3x2 overview
# Great for previewing a batch of photos at a glance

chop load landscape.png \
    | chop load flowers.png \
    | chop load cat.png \
    | chop load city.png \
    | chop load mountain.png \
    | chop grid --cols 3 \
    | chop pad 10 --color '#333333' \
    | chop save output/contact-sheet.png

echo "Created output/contact-sheet.png" >&2
