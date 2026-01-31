#!/usr/bin/env bash
# Recipe reuse: save an unbound program, then apply it to multiple images
# Demonstrates chop's "recipes" — reusable JSON transform pipelines

# Step 1: Create a reusable recipe (no load — it's unbound)
chop fit 300x300 \
    | chop pad 5 --color white \
    | chop border 1 --color black \
    | chop grayscale \
    > output/recipe-framed-bw.json

echo "Saved output/recipe-framed-bw.json" >&2

# Step 2: Apply the same recipe to different images
chop load cat.png \
    | chop apply output/recipe-framed-bw.json \
    | chop save output/cat-framed-bw.png

chop load mountain.png \
    | chop apply output/recipe-framed-bw.json \
    | chop save output/mountain-framed-bw.png

chop load flowers.png \
    | chop apply output/recipe-framed-bw.json \
    | chop save output/flowers-framed-bw.png

echo "Applied recipe to: output/cat-framed-bw.png, output/mountain-framed-bw.png, output/flowers-framed-bw.png" >&2
