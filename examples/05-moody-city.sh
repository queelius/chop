#!/usr/bin/env bash
# Moody city: desaturated, high-contrast noir look
# Applies color grading to make a nighttime cityscape dramatic

chop load city.png \
    | chop saturation 0.3 \
    | chop contrast 1.6 \
    | chop brightness 0.85 \
    | chop sharpen 1.5 \
    | chop save output/moody-city.png

echo "Created output/moody-city.png" >&2
