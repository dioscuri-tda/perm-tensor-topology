"""
Freeze a held-out evaluation set ("golden split") as a fixed list of sample_ids.

Rows of `permeability.csv` whose three solves all converged (conv_fx, conv_fy,
conv_fz == 1, when those columns exist) are split into a golden set of
`--n-golden` samples and a training pool. The golden set is drawn stratified by
the `stratum` column with a fixed seed, so every stratum is represented in
proportion and R^2 on it is not moved by an unbalanced draw.

Outputs (given `--output splits/aniso3d_golden`):

    splits/aniso3d_golden.csv    sample_id, stratum, split ("golden" | "pool")
    splits/aniso3d_golden.json   provenance and sha256 of the sorted golden ids

The split stores ids only, not copies of structures or labels, so
`permeability.csv` stays the single source of truth and any feature table can
be joined to it on sample_id. The CSV lists the pool rows too, recording which
rows were eligible. Consumers call `load_split`, which recomputes the hash and
refuses to run if the golden set no longer matches the manifest. Both files are
small and meant to be committed; an existing split is not overwritten without
`--force`.

Usage
-----
    python scripts/make_golden_split.py --dataset DATA/aniso3d \\
        --output splits/aniso3d_golden
"""

import argparse
import hashlib
import json
import os
from datetime import date

import numpy as np
import pandas as pd

# Recorded verbatim in the split manifest, so it is derived from the columns
# actually present rather than hard-coded.
def filter_desc(cols):
    return " and ".join(f"{c} == 1" for c in cols) or "none"


def golden_hash(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()


def load_split(csv_path, json_path=None):
    """
    Read a split back, verifying it is the one the manifest describes.

    Returns (golden_ids, pool_ids) as int arrays.
    """
    df = pd.read_csv(csv_path)
    golden = df.loc[df.split == "golden", "sample_id"].to_numpy()
    pool = df.loc[df.split == "pool", "sample_id"].to_numpy()

    if json_path is None:
        json_path = os.path.splitext(csv_path)[0] + ".json"
    if os.path.exists(json_path):
        meta = json.load(open(json_path))
        got = golden_hash(golden)
        if got != meta["sha256_golden_ids"]:
            raise SystemExit(
                f"golden set in {csv_path} does not match {json_path}\n"
                f"  manifest {meta['sha256_golden_ids']}\n"
                f"  actual   {got}\n"
                "Refusing to run: results would not be comparable with earlier ones.")
    return golden, pool


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="DATA/aniso3d")
    p.add_argument("--labels", default=None)
    p.add_argument("--output", default="splits/aniso3d_golden",
                   help="path prefix; .csv and .json are appended")
    p.add_argument("--n-golden", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing split (think hard before using this)")
    a = p.parse_args()

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    csv_out, json_out = a.output + ".csv", a.output + ".json"
    if os.path.exists(csv_out) and not a.force:
        raise SystemExit(
            f"{csv_out} already exists.\nA golden set is meant to be frozen; "
            "overwriting it invalidates every number computed against it. "
            "Pass --force only if you really mean to.")

    df = pd.read_csv(labels)
    n0 = len(df)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    df = df[["sample_id", "stratum"]].sort_values("sample_id").reset_index(drop=True)
    print(f"{n0} rows -> {len(df)} usable ({filter_desc(conv)})")

    # Stratified: off-diagonal magnitude differs strongly between strata, so an
    # unbalanced golden set would move R^2 for reasons unrelated to the model.
    rng = np.random.default_rng(a.seed)
    chosen = []
    for s, g in df.groupby("stratum"):
        k = int(round(a.n_golden * len(g) / len(df)))
        chosen.append(rng.choice(g.sample_id.to_numpy(), size=k, replace=False))
    golden = np.sort(np.concatenate(chosen))

    df["split"] = np.where(df.sample_id.isin(golden), "golden", "pool")
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    counts = {k: v.stratum.value_counts().to_dict() for k, v in df.groupby("split")}
    meta = {
        "dataset": a.dataset, "labels": labels,
        "created": date.today().isoformat(),
        "seed": a.seed, "filter": filter_desc(conv),
        "n_total": int(len(df)),
        "n_golden": int((df.split == "golden").sum()),
        "n_pool": int((df.split == "pool").sum()),
        "stratified_by": "stratum",
        "counts": counts,
        "sha256_golden_ids": golden_hash(golden),
    }
    with open(json_out, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\ngolden {meta['n_golden']}   pool {meta['n_pool']}")
    print(pd.DataFrame(counts).fillna(0).astype(int).to_string())
    print(f"\nsha256(golden ids) {meta['sha256_golden_ids'][:16]}...")
    print(f"-> {csv_out}\n-> {json_out}")
    print("\nCommit both.  They are small, and they are what makes every "
          "learning-curve number reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
