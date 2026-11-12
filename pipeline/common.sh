# Shared settings for the pipeline scripts.  Source it from the repository root.
set -euo pipefail
WORKERS=${WORKERS:-$(nproc)}
DATA=${DATA:-DATA}            # unpacked structures + permeability.csv per dataset
DESC=${DESC:-DESC}            # descriptor tables
RESULTS=${RESULTS:-RESULTS}   # CatBoost metrics
SCRATCH=${SCRATCH:-/tmp/ptt_scratch}   # per-structure filtrations (large, temporary)
JLENV=${JLENV:-julia}         # Julia project with the Ecp package (README)

# the nine probing directions: three axes and six signed face diagonals
DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" "1 1 0" "1 -1 0" "1 0 1" "1 0 -1" "0 1 1" "0 1 -1")

check_rows () {   # check_rows <csv> <expected data rows>
  local n=$(( $(wc -l < "$1") - 1 ))
  [ "$n" -eq "$2" ] || { echo "ABORT: $1 has $n rows, expected $2"; exit 1; }
}
