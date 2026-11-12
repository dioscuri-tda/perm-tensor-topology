r"""
Assemble per-structure descriptor .npy files into one direction-tagged table.

Reads `<descroot>/<kind>/a<tag>/<name>.npy` for every kind in `--kinds`
(default `ecp` and `ph`) and every direction directory found, and writes a
single CSV keyed by `sample_id`, with the `filename`, `stratum` and `porosity`
columns of `structures.csv` followed by the descriptors.

Column naming: `{kind}_a{tag}_{i}`, where `a{tag}` is the direction directory,
e.g. `ph_a1_-1_0_0 ... ph_a1_-1_0_299` for the persistence images along
[1,-1,0] and `ecp_a0_0_1_0 ... ecp_a0_0_1_342` for the ECP along [0,0,1].

All kinds and directions are collected in a single pass: a structure without a
descriptor in any directory is an error, so a partial table is never written,
and rerunning is idempotent. The table is also rejected if it contains NaNs.

Structures are matched to rows by `sample_id`, parsed from the leading
`sample_%06d` of the filename, never from the descriptive remainder of the
name.

Usage
-----
    python scripts/collect_descriptors.py --dataset DATA/aniso3d \
        --descroot DESC/aniso3d --output DESC/aniso3d/descriptors.csv \
        [--kinds ecp ph]
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

SAMPLE_RE = re.compile(r"^sample_(\d{6})")


def sample_id_of(fname: str) -> int:
    m = SAMPLE_RE.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"cannot parse sample_id from {fname!r}")
    return int(m.group(1))


def _tag_key(t: str):
    """
    Sort direction tags: numeric tags numerically, the rest lexically.

    3D tags are integer direction vectors (`a1_0_0`, `a1_-1_0`), on which
    `float()` raises, so they sort in a stable lexical order. A purely numeric
    tag (e.g. an angle, `a45`) sorts numerically, so `a135` comes after `a45`.
    Non-direction entries sort last.
    """
    if not t.startswith("a"):
        return (2, 0.0, "")
    try:
        return (0, float(t.lstrip("a")), "")
    except ValueError:
        return (1, 0.0, t)


def load_block(dirpath: str, kind: str, theta: str) -> pd.DataFrame:
    """All .npy under dirpath -> DataFrame indexed by sample_id."""
    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".npy"))
    if not files:
        raise SystemExit(f"no .npy files in {dirpath}")

    ids = [sample_id_of(f) for f in files]
    mat = np.stack([np.load(os.path.join(dirpath, f)).ravel() for f in files])
    cols = [f"{kind}_a{theta}_{i}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "sample_id", ids)
    return df.set_index("sample_id")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--descroot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kinds", nargs="+", default=["ecp", "ph"])
    p.add_argument("--limit", type=int, default=0,
                   help="stop at structure N (exclusive; 0 = end). Use when "
                        "the descriptors were computed for a prefix of the "
                        "dataset only, which would otherwise trip the "
                        "missing-descriptor check.")
    p.add_argument("--start", type=int, default=0,
                   help="begin at structure N (inclusive), so rows "
                        "[--start, --limit) of structures.csv are collected. "
                        "Use when the .npy files exist only for a contiguous "
                        "block of the dataset (e.g. a chunked run); collect "
                        "each block into its own CSV and concatenate them.")
    a = p.parse_args()

    base = pd.read_csv(os.path.join(a.dataset, "structures.csv")).set_index("sample_id")
    # iloc, not loc: `--start`/`--limit` are ROW offsets into structures.csv, the
    # same offsets filtration_throat3d.py and filtration3d.py slice with, so the
    # three stages agree even if sample_id were ever not equal to row number.
    if a.start or a.limit:
        base = base.iloc[a.start:(a.limit or None)]
        if base.empty:
            sys.exit(f"--start {a.start} --limit {a.limit} selects no structures")
        print(f"  collecting structures [{a.start}, {a.limit or 'end'}) "
              f"-> {len(base)} rows")
    out = base[["filename", "stratum", "porosity"]].copy()

    total = 0
    for kind in a.kinds:
        kind_dir = os.path.join(a.descroot, kind)
        if not os.path.isdir(kind_dir):
            print(f"  ! no {kind_dir}, skipping")
            continue
        for tag in sorted(os.listdir(kind_dir), key=_tag_key):
            if not tag.startswith("a"):
                continue
            theta = tag[1:]
            blk = load_block(os.path.join(kind_dir, tag), kind, theta)
            missing = out.index.difference(blk.index)
            if len(missing):
                sys.exit(f"{kind}/{tag}: {len(missing)} structures have no descriptor "
                         f"(first: {list(missing[:3])}) — rerun that stage")
            out = out.join(blk, how="left")
            total += blk.shape[1]
            print(f"  {kind}/{tag}: {blk.shape[0]} rows x {blk.shape[1]} features")

    if total == 0:
        sys.exit("no descriptors found")

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    out.reset_index().to_csv(a.output, index=False)

    n_nan = int(out.isna().sum().sum())
    print(f"\n  {len(out)} rows x {total} descriptor features -> {a.output}")
    print(f"  NaNs: {n_nan}")
    if n_nan:
        sys.exit("NaNs present — investigate before training")


if __name__ == "__main__":
    main()
