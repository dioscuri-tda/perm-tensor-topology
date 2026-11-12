#!/bin/bash
# Step 3: all descriptors of one dataset.
#
#   bash pipeline/03_descriptors.sh [aniso3d]
#
# Writes DESC/<set>/
#   baselines.csv  Stats: porosity profile, two-point correlation, fabric tensor, fabric profile
#   phys.csv       Phys: tensorial Kozeny-Carman and directional critical path
#   cone.csv       TDA: cone filtration -> Euler characteristic profiles (ecp_) + persistence images (ph_)
#   throat.csv     THR: throat-scale filtration -> persistence images (thr_)
# All directional descriptors use the same nine directions (common.sh).
# Needs Julia with the Ecp package for the ECP step (see README).
cd "$(dirname "$0")/.."; source pipeline/common.sh
SET=${1:-aniso3d}
DS=$DATA/$SET; OUT=$DESC/$SET; NPY=$SCRATCH/$SET/npy; FILTROOT=$SCRATCH/$SET/filt
N=$(( $(wc -l < "$DS/structures.csv") - 1 ))
mkdir -p "$OUT" "$NPY" "$FILTROOT"

# cone filtration (TDA): radius of the PCA ball, cone base radius and height, ECP grid
RADIUS=4; CONE_R=3; CONE_H=6; GRID_RES=6
# throat filtration (THR): R_cap, walk length H, phase
RCAP=10; HORIZON=6; PHASE=void
# persistence images: linear weight, 10x10; per-dimension "birth range, persistence range" and bandwidth
PH_WEIGHT=linear; PH_RES="10 10"; N_POR=80; N_TPC=41
T_RANGE_H0="0 0.9 0 0.40";  T_BW_H0=0.060
T_RANGE_H1="0 0.9 0 0.75";  T_BW_H1=0.082
T_RANGE_H2="0 0.9 0 0.50";  T_BW_H2=0.067
case $SET in
  aniso3d)
    C_RANGE_H0="0 1.25 0 1.25"; C_BW_H0=0.125
    C_RANGE_H1="0 1.25 0 0.80"; C_BW_H1=0.100
    C_RANGE_H2="0.2 1.0 0 0.06"; C_BW_H2=0.019 ;;
  *) echo "unknown dataset $SET"; exit 1 ;;
esac

ph () {   # ph <filtdir> <outdir> <r0> <b0> <r1> <b1> <r2> <b2>
  python scripts/ph3d.py --inputdir "$1" --outputdir "$2" \
      --weight "$PH_WEIGHT" --resolution $PH_RES \
      --im_range_h0 $3 --bandwidth_h0 "$4" --im_range_h1 $5 --bandwidth_h1 "$6" \
      --im_range_h2 $7 --bandwidth_h2 "$8" --workers "$WORKERS"
}

echo "=== Stats  $(date -Is) ==="
[ -f "$OUT/baselines.csv" ] || python scripts/baselines3d.py --dataset "$DS" \
    --output "$OUT/baselines.csv" --n_por "$N_POR" --n_tpc "$N_TPC" --workers "$WORKERS"
echo "=== Phys  $(date -Is) ==="
[ -f "$OUT/phys.csv" ] || python scripts/physbaselines3d.py --dataset "$DS" \
    --output "$OUT/phys.csv" --workers "$WORKERS"

CHUNK=${CHUNK:-1000}   # structures per batch; the cone filtration takes ~6 MB per structure on disk
mkdir -p "$NPY/cone" "$NPY/throat"
for D in "${DIRECTIONS[@]}"; do
  TAG=${D// /_}
  for START in $(seq 0 "$CHUNK" $((N - 1))); do
    TAKE=$CHUNK; [ $((START + TAKE)) -gt "$N" ] && TAKE=$((N - START))
    if [ ! -f "$NPY/cone/.done_a${TAG}_$START" ]; then
      echo "=== TDA (cone) [$D] $START+$TAKE  $(date -Is) ==="
      FILT=$FILTROOT/cone_a$TAG
      python scripts/filtration3d.py --dataset "$DS" --outputdir "$FILT" --direction $D \
          --radius "$RADIUS" --cone-radius "$CONE_R" --cone-height "$CONE_H" \
          --workers "$WORKERS" --start "$START" --limit "$TAKE"
      julia --project="$JLENV" --threads "$WORKERS" scripts/ecp3d.jl \
          "$FILT" "$NPY/cone/ecp/a$TAG" --grid-res "$GRID_RES"
      ph "$FILT" "$NPY/cone/ph/a$TAG" \
         "$C_RANGE_H0" $C_BW_H0 "$C_RANGE_H1" $C_BW_H1 "$C_RANGE_H2" $C_BW_H2
      rm -rf "$FILT"; touch "$NPY/cone/.done_a${TAG}_$START"
    fi
    if [ ! -f "$NPY/throat/.done_a${TAG}_$START" ]; then
      echo "=== THR (throat) [$D] $START+$TAKE  $(date -Is) ==="
      FILT=$FILTROOT/throat_a$TAG
      python scripts/filtration_throat3d.py --dataset "$DS" --outputdir "$FILT" --mode dir \
          --direction $D --rcap "$RCAP" --horizon "$HORIZON" --phase "$PHASE" \
          --workers "$WORKERS" --start "$START" --limit "$TAKE"
      ph "$FILT" "$NPY/throat/thr/a$TAG" \
         "$T_RANGE_H0" $T_BW_H0 "$T_RANGE_H1" $T_BW_H1 "$T_RANGE_H2" $T_BW_H2
      rm -rf "$FILT"; touch "$NPY/throat/.done_a${TAG}_$START"
    fi
  done
done

echo "=== collect  $(date -Is) ==="
python scripts/collect_descriptors.py --dataset "$DS" --descroot "$NPY/cone" \
    --output "$OUT/cone.csv" --kinds ecp ph
python scripts/collect_descriptors.py --dataset "$DS" --descroot "$NPY/throat" \
    --output "$OUT/throat.csv" --kinds thr
for f in baselines phys cone throat; do check_rows "$OUT/$f.csv" "$N"; done
echo "DESCRIPTORS DONE $SET  $(date -Is)"
