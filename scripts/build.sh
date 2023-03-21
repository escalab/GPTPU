#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

sudo rm -rf build
mkdir -p build
cd build
cmake ..
make
