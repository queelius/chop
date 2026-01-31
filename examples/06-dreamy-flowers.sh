#!/usr/bin/env bash
# Dreamy flowers: soft blur + brightness boost for a glowy look
# Creates an ethereal, overexposed-film aesthetic

chop load flowers.png \
    | chop blur 2.5 \
    | chop brightness 1.3 \
    | chop saturation 1.4 \
    | chop contrast 0.85 \
    | chop save output/dreamy-flowers.png

echo "Created output/dreamy-flowers.png" >&2
