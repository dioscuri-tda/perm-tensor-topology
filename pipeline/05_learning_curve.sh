#!/bin/bash
# Step 5: learning curves (Figure 5): hold-out R^2 vs. number of training structures.
#
#   bash pipeline/05_learning_curve.sh
#
# Hold-out: splits/aniso3d_golden.csv, 999 structures (333 per anisotropy class),
# never used for training.  Training subsets of size n are drawn from the other
# 4001; within a seed every smaller subset is a prefix of the larger one and is
# balanced over the classes; 5 seeds per size.  Fixed budget of 1500 iterations.
# Output: RESULTS/aniso3d/lc_<kxx..kyz>/fixed/runs.csv, table.md and table.csv.
cd "$(dirname "$0")/.."; source pipeline/common.sh
DS=$DATA/aniso3d; F="$DESC/aniso3d/cone.csv $DESC/aniso3d/throat.csv $DESC/aniso3d/baselines.csv"
for T in K_xx k_xy k_xz k_yz; do
  OUT=$RESULTS/aniso3d/lc_$(echo "${T,,}" | tr -d _)
  python scripts/learning_curve.py --dataset "$DS" --labels "$DS/permeability.csv" \
      --descriptors $F --split splits/aniso3d_golden.csv \
      --groups baselines tda thr all baselines+thr por tpc fab fabdir --targets "$T" \
      --sizes 25 50 75 100 125 250 500 1000 2000 4000 --seeds 0 1 2 3 4 \
      --iterations 1500 --no-early-stop --output "$OUT/fixed"
  python scripts/lc_table.py "$OUT/fixed" --target "$T" --out "$OUT/table.md"
done
