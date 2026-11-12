#!/bin/bash
# Step 2: permeability tensors with the lattice Boltzmann solver (lbm/lbm3d-perm.cpp).
#
#   bash pipeline/02_lbm.sh [dataset]      default dataset: aniso3d
#
# Three solves per structure (body force along x, y, z), D3Q19 BGK, periodic.
# Expensive: about 3 CPU-hours per structure at 80^3; resumable (solved
# structures are skipped).  The tensors used in the paper are shipped in
# data/aniso3d/permeability.csv, which 01_data.sh copies to DATA/aniso3d, so
# this step is only needed to reproduce the labels.
cd "$(dirname "$0")/.."; source pipeline/common.sh
SET=${1:-aniso3d}
bash lbm/make_perm.sh
python scripts/solve_permeability3d.py --dataset "$DATA/$SET" \
    --solver lbm/lbm3d-perm --workers "$WORKERS"
