#!/bin/bash
# Step 1: the synthetic dataset (5000 anisotropic structures, 80^3 voxels).
#
#   bash pipeline/01_data.sh              unpack the shipped structures (data/aniso3d) to DATA/aniso3d
#   bash pipeline/01_data.sh regenerate   regenerate them from scratch with the generator
#
# Both give byte-identical structures (the generator is seeded); unpacking checks
# every file against the SHA-256 recorded in data/aniso3d/structures.csv.
cd "$(dirname "$0")/.."; source pipeline/common.sh
if [ "${1:-unpack}" = regenerate ]; then
  python scripts/gen_aniso_structures3d.py --output "$DATA/aniso3d" \
      --n_samples 5000 --size 80 --seed 0 --workers "$WORKERS"
else
  python scripts/pack_structures.py unpack --packed data/aniso3d --out "$DATA/aniso3d"
fi
