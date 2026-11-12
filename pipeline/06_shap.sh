#!/bin/bash
# Step 6: SHAP analysis (Figures 6-7, Supplementary S4).
#
#   bash pipeline/06_shap.sh
#
# One CatBoost model per target trained on four of the five folds of seed 0,
# TreeSHAP values computed on the held-out fold (1000 structures).
cd "$(dirname "$0")/.."; source pipeline/common.sh
DS=$DATA/aniso3d
python scripts/shap_aniso3d.py --labels "$DS/permeability.csv" \
    --descriptors $DESC/aniso3d/cone.csv $DESC/aniso3d/throat.csv $DESC/aniso3d/baselines.csv $DESC/aniso3d/phys.csv \
    --groups thr classical classical+thr --targets k_xy k_xz k_yz K_xx \
    --seed 0 --iterations 1500 --threads "$WORKERS" --output $RESULTS/aniso3d/shap
