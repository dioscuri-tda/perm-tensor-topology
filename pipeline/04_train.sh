#!/bin/bash
# Step 4: CatBoost models for every descriptor set (Tables 1-2 and Supplementary Tables).
#
#   bash pipeline/04_train.sh [aniso3d]
#
# 5-fold cross-validation, single-target, 1500 iterations, depth 6, learning rate
# 0.05, no early stopping.  Seed 0 is the main partition; seeds 1 and 2 repeat the
# whole comparison with a different fold assignment and CatBoost random state.
# Output: RESULTS/<set>/seed<S>/<target>__<group>/metrics.json (per-fold R^2, MAE).
# Resumable: finished runs are skipped.
#
# Paper names of the descriptor sets (code group keys):
#   Stats=baselines  Phys=phys  Baselines=classical  TDA=tda  THR=thr
#   Stats+TDA=all  Stats+THR=baselines+thr  Phys+TDA=phys+tda  Phys+THR=phys+thr
#   Baselines+TDA=classical+tda  Baselines+THR=classical+thr
cd "$(dirname "$0")/.."; source pipeline/common.sh
SET=${1:-aniso3d}
DS=$DATA/$SET; F="$DESC/$SET/cone.csv $DESC/$SET/throat.csv $DESC/$SET/baselines.csv $DESC/$SET/phys.csv"
GROUPS_=${GROUPS_:-"por tpc fab fabdir baselines kc cp phys classical tda thr all baselines+thr phys+tda phys+thr classical+tda classical+thr"}
SEEDS=${SEEDS:-"0 1 2"}
T0="K_xx K_yy K_zz k_xy k_xz k_yz"
T12="K_xx k_xy k_xz k_yz"

for s in $SEEDS; do
  if [ "$s" = 0 ]; then TARGETS=$T0; else TARGETS=$T12; fi
  for g in $GROUPS_; do for t in $TARGETS; do
    out=$RESULTS/$SET/seed$s/${t}__$g
    [ -f "$out/metrics.json" ] && continue
    echo "### $SET seed $s  $t  $g  $(date -Is)"
    python scripts/train_catboost.py --dataset "$DS" --labels "$DS/permeability.csv" \
        --descriptors $F --folds 5 --seed "$s" --iterations 1500 \
        --targets "$t" --groups "$g" --output "$out"
  done; done
done

if [ "$SET" = aniso3d ]; then
  # Supplementary S6: one model for the whole tensor (MultiRMSE on standardised targets)
  for g in tda thr all baselines+thr classical+tda classical+thr; do
    out=$RESULTS/$SET/seed0/tensor3d__$g
    [ -f "$out/metrics.json" ] && continue
    python scripts/train_catboost.py --dataset "$DS" --labels "$DS/permeability.csv" \
        --descriptors $F --folds 5 --seed 0 --iterations 1500 \
        --targets tensor3d --multioutput --scale-targets --groups "$g" --output "$out"
  done
  # Figure 4 (scatter plots): out-of-fold predictions, refitted with --save-oof
  for t in k_yz K_xx; do for g in thr tda classical baselines; do
    out=$RESULTS/$SET/oof_seed0/${t}__$g
    [ -f "$out/oof.csv" ] && continue
    python scripts/train_catboost.py --dataset "$DS" --labels "$DS/permeability.csv" \
        --descriptors $F --folds 5 --seed 0 --iterations 1500 \
        --targets "$t" --groups "$g" --output "$out" --save-oof
  done; done
fi
echo "TRAINING DONE $SET  $(date -Is)"
