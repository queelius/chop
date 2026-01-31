#!/usr/bin/env bash
# Sepia vintage: grayscale + colorize for classic film look
# Demonstrates the colorize operation with strength control

chop load landscape.png \
    | chop grayscale \
    | chop colorize '#704214' \
    | chop contrast 1.2 \
    | chop save output/sepia-vintage.png

echo "Created output/sepia-vintage.png" >&2
