#!/bin/bash
# Build lbm3d-perm next to this script.  No external dependencies.
#
#   bash lbm/make_perm.sh
#
# -march=broadwell rather than -march=native: the binary runs on any x86-64 CPU
# with AVX2 and FMA, wherever it was built (a native build on an AVX-512 machine
# dies with an illegal instruction on older CPUs).  Adjust it to your hardware.
#
# Safe to run while solves are in progress: the binary is built under a temp
# name and moved into place with an atomic rename.  Running solves keep the old
# file (they hold its inode); every solve started afterwards gets the new one.
# The previous binary is kept as lbm3d-perm.prev.
set -e
cd "$(dirname "$(readlink -f "$0")")"
g++ -O3 -march=broadwell -ffast-math -o lbm3d-perm.new lbm3d-perm.cpp
[ -f lbm3d-perm ] && cp -p lbm3d-perm lbm3d-perm.prev
mv -f lbm3d-perm.new lbm3d-perm
echo "built $(pwd)/lbm3d-perm (previous kept as lbm3d-perm.prev)"
