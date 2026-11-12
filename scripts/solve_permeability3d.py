"""
Run the `lbm3d-perm` solver over a structure set and collect the permeability
tensors into one table.

For every row of <dataset>/structures.csv the solver is run on the matching
.raw file. It performs three lattice Boltzmann solves of the same triply
periodic cell, forced along +x, +y and +z, each giving one column of K (see
`lbm/lbm3d-perm.cpp`). Structures are solved in parallel, one solver process
per worker.

Restartable runs
----------------
Each sample writes its own CSV, <dataset>/perm/NNNNNN.csv, so a long run can be
interrupted and resumed: rerunning solves only what is missing. The solver
writes to `<out>.tmp.<pid>` and the driver renames it into place with
`os.replace` (atomic within a filesystem) only after `valid_result()` has
checked that it holds one complete row with finite tensor components, so a
kill at any instant leaves at most a stray .tmp file. Solver exit code 3 (a
direction that did not converge) is kept and flagged in conv_fx/conv_fy/conv_fz,
not treated as a failure. `--shard I/N` splits a run over several machines.

Merged output
-------------
<dataset>/permeability.csv joins structures.csv with the solver output and adds
derived columns, and <dataset>/permeability_manifest.json records the snapshot
(row count, id range, sha256 of the ids). `k_xy`, `k_xz`, `k_yz` are the
symmetrised off-diagonals, (K_ij + K_ji)/2. K = K^T is exact for Stokes flow,
so averaging the two measurements halves the noise on the off-diagonals, and
the discarded difference is kept as a per-sample error bar:

    koff_rel_err_xy = |K_xy - K_yx| / |2 k_xy|      etc.

The merge also prints the resulting label-noise ceiling on R^2 per component.
Regress the tensor components (or log k_mean plus the deviatoric `dev_*`
columns), not eigenvector directions: eigenvectors are sign-ambiguous, swap
order when eigenvalues cross, and are undetermined for rod or plate shapes.

Usage
-----
    # audit: how many solves are complete, incomplete, missing
    python scripts/solve_permeability3d.py --dataset DATA/aniso3d --verify
    # clear anything an interrupted run left half-written
    python scripts/solve_permeability3d.py --dataset DATA/aniso3d --verify --repair
    # solve what is missing and merge; safe to run repeatedly
    python scripts/solve_permeability3d.py --dataset DATA/aniso3d --workers 14
    # rebuild permeability.csv from what is on disk, solving nothing
    python scripts/solve_permeability3d.py --dataset DATA/aniso3d --merge-only
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import koff_ceilings3d                              # noqa: E402

DEFAULT_SOLVER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "lbm", "lbm3d-perm")


# Columns every finished per-sample CSV must carry; see valid_result().
REQUIRED_COLS = ("K_xx", "K_yy", "K_zz", "k_xy", "k_xz", "k_yz",
                 "conv_fx", "conv_fy", "conv_fz")


def valid_result(path) -> bool:
    """
    True if `path` holds a complete solve, not merely a non-empty file.

    Parses the CSV and requires at least one data row with all REQUIRED_COLS
    present and finite.
    """
    try:
        if os.path.getsize(path) <= 0:
            return False
        one = pd.read_csv(path)
    except Exception:
        return False
    if len(one) < 1 or not set(REQUIRED_COLS) <= set(one.columns):
        return False
    return bool(np.isfinite(one[list(REQUIRED_COLS)].to_numpy(float)).all())


def solve_one(task):
    solver, structure, out_csv, ff, timeout, extra = task
    if os.path.exists(out_csv):
        if valid_result(out_csv):
            return out_csv, 0, "cached"
        # Present but incomplete (an interrupted run): redo it.
        os.remove(out_csv)

    # Write to a temp path and rename on success. os.replace is atomic within a
    # filesystem, so the final path only ever holds a complete solve and a kill
    # at any instant leaves at most a stray .tmp file.
    tmp = f"{out_csv}.tmp.{os.getpid()}"
    cmd = [solver, structure, tmp, "--quiet", "--ff", str(ff)] + extra
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp):
            os.remove(tmp)
        return out_csv, -1, "timeout"
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    if p.returncode not in (0, 3) or not valid_result(tmp):
        msg = (p.stderr or "").strip()[-200:] if p.returncode not in (0, 3) \
            else "solver returned 0/3 but wrote an incomplete CSV"
        if os.path.exists(tmp):
            os.remove(tmp)
        return out_csv, p.returncode or -2, msg

    os.replace(tmp, out_csv)          # atomic
    return out_csv, p.returncode, "ok"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="directory produced by gen_aniso_structures3d.py")
    ap.add_argument("--solver", default=DEFAULT_SOLVER)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--ff", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="solve only the first N")
    ap.add_argument("--timeout", type=int, default=86400,
                    help="per-structure wall-clock limit in seconds (default 24 h)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--shard", default="",
                    help="I/N: solve/verify only samples with sample_id %% N == I, and "
                         "do NOT merge (other shards may still be running); split a run "
                         "over N machines, then --merge-only once all are done")
    ap.add_argument("--interleave", default="",
                    help="solve order round-robin over this structures.csv column "
                         "(e.g. stratum), so a run cut short still covers every group; "
                         "labels and permeability.csv row order are unaffected")
    ap.add_argument("--verify", action="store_true",
                    help="audit <dataset>/perm/ and report how many solves are "
                         "complete, incomplete or missing, then exit")
    ap.add_argument("--repair", action="store_true",
                    help="with --verify, delete incomplete per-sample CSVs (and "
                         "any stray .tmp files) so the next run redoes them")
    ap.add_argument("--extra", default="",
                    help='extra flags for the solver, e.g. "--eps 5e-4"')
    args = ap.parse_args()

    struct_csv = os.path.join(args.dataset, "structures.csv")
    if not os.path.exists(struct_csv):
        sys.exit(f"no structures.csv in {args.dataset}")
    df = pd.read_csv(struct_csv)
    if args.limit:
        df = df.head(args.limit)
    shard_ids = None
    if args.shard:
        if args.merge_only:
            sys.exit("--shard and --merge-only exclude each other: merge runs over all shards")
        si, sn = (int(v) for v in args.shard.split("/"))
        assert 0 <= si < sn, f"bad --shard {args.shard}"
        df = df[df.sample_id % sn == si].reset_index(drop=True)
        shard_ids = set(df.sample_id)
        print(f"shard {si}/{sn}: {len(df)} samples")

    perm_dir = os.path.join(args.dataset, "perm")
    os.makedirs(perm_dir, exist_ok=True)
    structures_dir = os.path.join(args.dataset, "structures")

    expected = [(r.sample_id, os.path.join(perm_dir, f"{r.sample_id:06d}.csv"))
                for r in df.itertuples()]

    if args.verify:
        good = [sid for sid, pth in expected if os.path.exists(pth) and valid_result(pth)]
        bad = [sid for sid, pth in expected if os.path.exists(pth) and not valid_result(pth)]
        missing = [sid for sid, pth in expected if not os.path.exists(pth)]
        strays = [f for f in os.listdir(perm_dir) if ".tmp." in f]
        if shard_ids is not None:
            # never touch another shard's in-flight temp files on a shared disk
            strays = [f for f in strays if int(f.split(".")[0]) in shard_ids]
        print(f"{args.dataset}: {len(expected)} structures")
        print(f"  complete   {len(good)}")
        print(f"  incomplete {len(bad)}" + (f"  e.g. {bad[:5]}" if bad else ""))
        print(f"  missing    {len(missing)}"
              + (f"  first {missing[:3]} ... last {missing[-3:]}" if missing else ""))
        print(f"  stray .tmp {len(strays)}")
        if args.repair:
            for sid in bad:
                os.remove(os.path.join(perm_dir, f"{sid:06d}.csv"))
            for f in strays:
                os.remove(os.path.join(perm_dir, f))
            print(f"  repaired: removed {len(bad)} incomplete + {len(strays)} stray files")
        elif bad or strays:
            print("  rerun with --repair to clear them, then resume normally")
        return

    if not args.merge_only:
        if not os.path.exists(args.solver):
            sys.exit(f"solver not found: {args.solver}  (build it with lbm/make_perm.sh)")

        extra = args.extra.split()
        order = df
        if args.interleave:
            # i-th cube of every group before the (i+1)-th of any; stable within a group
            rank = df.groupby(args.interleave, sort=False).cumcount()
            order = df.assign(_r=rank).sort_values(["_r", args.interleave], kind="stable")
        tasks = [(args.solver,
                  os.path.join(structures_dir, r.filename),
                  os.path.join(perm_dir, f"{r.sample_id:06d}.csv"),
                  args.ff, args.timeout, extra)
                 for r in order.itertuples()]

        todo = sum(1 for _, pth in expected
                   if not (os.path.exists(pth) and valid_result(pth)))
        print(f"{len(expected)} structures: {len(expected) - todo} already solved, "
              f"{todo} to do")
        if todo == 0:
            print("nothing to solve -- merging what is on disk")

        failed, unconverged, cached = [], 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(solve_one, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="solving"):
                out, rc, msg = f.result()
                if msg == "cached":
                    cached += 1
                elif rc == 3:
                    unconverged += 1
                elif rc != 0:
                    failed.append((os.path.basename(out), rc, msg))

        print(f"cached {cached}   non-converged (kept, flagged) {unconverged}   "
              f"failed {len(failed)}")
        for name, rc, msg in failed[:10]:
            print(f"  {name}: rc={rc} {msg}")
        if shard_ids is not None:
            print("shard finished -- not merging; run --merge-only once every shard is done")
            return

    # --- merge --------------------------------------------------------------
    parts, unreadable = [], []
    for r in df.itertuples():
        pth = os.path.join(perm_dir, f"{r.sample_id:06d}.csv")
        if not os.path.exists(pth):
            continue
        if not valid_result(pth):
            # Report incomplete files rather than dropping them silently.
            unreadable.append(r.sample_id)
            continue
        one = pd.read_csv(pth)
        one["sample_id"] = r.sample_id
        parts.append(one)
    if unreadable:
        print(f"WARNING: {len(unreadable)} per-sample CSVs are incomplete and were "
              f"skipped (e.g. {unreadable[:5]}).")
        print("         Run with --verify --repair, then resume, to redo them.")

    if not parts:
        sys.exit("nothing to merge")

    perm = pd.concat(parts, ignore_index=True).drop(columns=["filename"])
    out = df.merge(perm, on="sample_id", how="inner", suffixes=("", "_solver"))

    trace = out["K_xx"] + out["K_yy"] + out["K_zz"]
    out["trace"] = trace
    out["log_k_mean"] = np.log(out["k_mean"].where(out["k_mean"] > 0))
    out["k_ratio"] = out["k1"] / out["k3"].where(out["k3"] > 0)
    out["log_k_ratio"] = np.log(out["k_ratio"].where(out["k_ratio"] > 0))

    # Per-off-diagonal label noise, and the quantity it depends on.
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        sym = out[f"k_{a}{b}"]
        asym = (out[f"K_{a}{b}"] - out[f"K_{b}{a}"]).abs()
        out[f"koff_rel_err_{a}{b}"] = asym / (2 * sym).abs()
        out[f"koff_over_trace_{a}{b}"] = sym.abs() / trace.abs()

    # Scale-free deviator of the symmetrised tensor: (Ks - k_mean I) / k_mean.
    # Five independent components (dev_zz is -(dev_xx + dev_yy)), so
    # log_k_mean plus these five is the whole tensor with scale and shape
    # separated, which is the target set the docstring above argues for.
    km = out["k_mean"].where(out["k_mean"] > 0)
    out["dev_xx"] = out["K_xx"] / km - 1.0
    out["dev_yy"] = out["K_yy"] / km - 1.0
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        out[f"dev_{a}{b}"] = out[f"k_{a}{b}"] / km

    out = out.sort_values("sample_id").reset_index(drop=True)
    out["conv_all"] = ((out.conv_fx == 1) & (out.conv_fy == 1)
                       & (out.conv_fz == 1)).astype(int)

    dest = os.path.join(args.dataset, "permeability.csv")
    out.to_csv(dest, index=False)

    # --- manifest -----------------------------------------------------------
    # The table may be merged while a run is still filling it, so every merge
    # records which snapshot it is (n_rows, id range, sha256 of the ids).
    ids = out.sample_id.to_numpy()
    meta = {
        "dataset": args.dataset,
        "labels": dest,
        "merged": datetime.now().isoformat(timespec="seconds"),
        "n_structures": int(len(df)),
        "n_rows": int(len(out)),
        "n_converged": int(out.conv_all.sum()),
        "sample_id_min": int(ids.min()),
        "sample_id_max": int(ids.max()),
        "contiguous": bool(len(ids) == ids.max() - ids.min() + 1),
        "sha256_ids": hashlib.sha256(
            ",".join(str(int(i)) for i in ids).encode()).hexdigest(),
        "strata": out.stratum.value_counts().to_dict(),
    }
    man = os.path.splitext(dest)[0] + "_manifest.json"
    with open(man, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"merged {len(out)}/{len(df)} rows -> {dest}")
    print(f"  manifest -> {man}")
    print(f"  ids {ids.min()}..{ids.max()}"
          f"{'' if meta['contiguous'] else '  (GAPS -- the solve order is not a prefix)'}")
    print(f"  strata: {meta['strata']}")

    conv = out.conv_all == 1
    print(f"all three directions converged: {conv.sum()}/{len(out)}")
    cols = (["recip_resid", "fa", "k_mean", "k_ratio"]
            + [f"koff_over_trace_{a}{b}" for a, b in (("x","y"),("x","z"),("y","z"))]
            + [f"koff_rel_err_{a}{b}" for a, b in (("x","y"),("x","z"),("y","z"))])
    print(out.loc[conv, cols].describe().T.to_string())

    # R^2 against a label this noisy is bounded by 1 - var(noise)/var(signal),
    # with the noise estimated from the reciprocity residual (metrics.py).
    print("\n  label-noise ceiling on R^2 (from K = K^T)")
    for t, c in koff_ceilings3d(out.loc[conv]).items():
        print(f"    {t}: R^2 <= {c:.6f}")


if __name__ == "__main__":
    main()
