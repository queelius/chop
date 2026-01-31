#!/usr/bin/env bash
# Multi-save: generate multiple sizes in a single pipeline pass
# Demonstrates chop's non-terminal save — one pipeline, three outputs

chop load mountain.png \
    | chop save output/mountain-full.png \
    | chop resize 50% \
    | chop save output/mountain-medium.png \
    | chop resize 50% \
    | chop save output/mountain-thumb.png

echo "Created output/mountain-full.png, output/mountain-medium.png, output/mountain-thumb.png" >&2
