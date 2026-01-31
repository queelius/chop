#!/usr/bin/env bash
# Inverted art: negative image with boosted saturation for a pop-art effect
# Creates a psychedelic, Warhol-esque look

chop load cat.png \
    | chop invert \
    | chop saturation 2.0 \
    | chop contrast 1.3 \
    | chop border 8 --color '#ff00ff' \
    | chop save output/inverted-art.png

echo "Created output/inverted-art.png" >&2
