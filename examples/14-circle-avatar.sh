#!/usr/bin/env bash
# Circle avatar: mask + background for profile-picture style crop
# Demonstrates mask and background operations

chop load cat.png \
    | chop fill 400x400 \
    | chop mask circle \
    | chop background white \
    | chop save output/circle-avatar.png

echo "Created output/circle-avatar.png" >&2
