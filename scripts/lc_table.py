r"""
Markdown tables for the learning-curve study (hold-out metric vs training size).

Reads `<dir>/runs.csv` written by learning_curve.py (one row per target, group,
size, seed; every row scored on the same frozen hold-out). Each directory is
one "arm" (e.g. fixed budget vs early stopping). For each arm and the chosen
target it prints:

1. Metric vs size: mean +- sd over seeds, one row per group.
2. Paired difference vs a reference group (default `baselines`, i.e. Stats).
   The difference is taken per (size, seed), so both sides share the training
   subset and the hold-out, then averaged; the sd and the win count are over
   seeds. The hold-out's sampling noise is common to both sides and cancels.
3. (R^2 only) Training size at which each group's mean curve reaches the
   reference group's R^2 at the largest size, interpolated linearly in log(n);
   ">N" means not reached within the sizes run.

Cells computed on fewer seeds than the most complete cell are marked `*`, so
the script is safe to run on a partial study. Group `all` is printed as
`baselines+tda` (Stats+TDA).

Outputs: markdown to stdout; with --out, also the markdown file and a
long-format <out>.csv of every run (with `arm` and `group_label` columns).

Usage
    python scripts/lc_table.py RESULTS/lc3d_kxy/fixed --target k_xy
    python scripts/lc_table.py RESULTS/lc3d_kxy/fixed RESULTS/lc3d_kxy/es \
        --target k_xy --ref baselines --out RESULTS/lc3d_kxy/table.md
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

LABELS = {"all": "baselines+tda"}
ORDER = ["por", "tpc", "fab", "fabdir", "baselines", "tda", "thr",
         "all", "baselines+tda", "baselines+thr"]


def label(g):
    return LABELS.get(g, g)


def ordered(groups):
    known = [g for g in ORDER if g in groups]
    return known + sorted(g for g in groups if g not in ORDER)


def md_table(df, index_name):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([index_name] + cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(out)


def n_to_reach(sizes, means, goal):
    """First size at which the mean curve reaches `goal`, log-linear between points."""
    if means[0] >= goal:
        return f"<={sizes[0]}"
    for (n0, r0), (n1, r1) in zip(zip(sizes, means), zip(sizes[1:], means[1:])):
        if r1 >= goal:
            f = (goal - r0) / (r1 - r0) if r1 != r0 else 1.0
            return f"{int(round(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0)))))}"
    return f">{sizes[-1]}"


def arm_report(runs, target, ref, metric):
    r = runs[runs.target == target]
    if r.empty:
        return None, f"no runs for target {target}"
    groups = ordered(r.group.unique())
    sizes = sorted(r["size"].unique())
    full = r.groupby(["group", "size"]).seed.nunique().max()
    parts = []

    # 1. metric vs size
    t1 = pd.DataFrame(index=[label(g) for g in groups], columns=sizes, dtype=object)
    means = {}
    for g in groups:
        for s in sizes:
            v = r[(r.group == g) & (r["size"] == s)][metric]
            if v.empty:
                t1.loc[label(g), s] = "-"
                continue
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            mark = "*" if len(v) < full else ""
            t1.loc[label(g), s] = f"{v.mean():.4f} ± {sd:.4f}{mark}"
            means[(g, s)] = v.mean()
    parts.append(f"**{metric.upper()} on the hold-out** (mean ± sd over seeds; "
                 f"`*` = fewer than {full} seeds)\n\n" + md_table(t1, "group \\ n"))

    # 2. paired difference vs reference
    if ref in groups:
        piv = r.pivot_table(index=["size", "seed"], columns="group", values=metric)
        sign = -1 if metric in ("mae", "mse") else 1   # positive = better than ref
        t2 = pd.DataFrame(index=[label(g) for g in groups if g != ref],
                          columns=sizes, dtype=object)
        for g in groups:
            if g == ref:
                continue
            for s in sizes:
                if s not in piv.index.get_level_values(0):
                    t2.loc[label(g), s] = "-"
                    continue
                d = sign * (piv.loc[s, g] - piv.loc[s, ref]).dropna()
                if d.empty:
                    t2.loc[label(g), s] = "-"
                    continue
                sd = d.std(ddof=1) if len(d) > 1 else float("nan")
                t2.loc[label(g), s] = (f"{d.mean():+.4f} ± {sd:.4f} "
                                       f"({int((d > 0).sum())}/{len(d)})")
        better = "higher" if sign > 0 else "lower"
        parts.append(f"**Paired Δ{metric.upper()} vs `{label(ref)}`** "
                     f"(positive = better, i.e. {better} {metric}; mean ± sd over "
                     f"seeds, (seeds won/seeds))\n\n" + md_table(t2, "group \\ n"))

        # 3. structures needed to match the reference at the largest size
        if metric == "r2" and (ref, sizes[-1]) in means:
            goal = means[(ref, sizes[-1])]
            t3 = pd.DataFrame(index=[label(g) for g in groups],
                              columns=[f"n to reach R² {goal:.4f}"], dtype=object)
            for g in groups:
                ss = [s for s in sizes if (g, s) in means]
                t3.iloc[groups.index(g), 0] = (
                    n_to_reach(ss, [means[(g, s)] for s in ss], goal) if ss else "-")
            parts.append(f"**Structures needed to match `{label(ref)}` at "
                         f"n={sizes[-1]}** (mean curve, log-linear interpolation)\n\n"
                         + md_table(t3, "group"))
    else:
        parts.append(f"(reference group `{ref}` not in these runs -- "
                     "no paired table)")
    return r, "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="learning_curve.py output dirs (one per arm)")
    p.add_argument("--target", default="k_xy")
    p.add_argument("--ref", default="baselines", help="group the others are paired against")
    p.add_argument("--metric", default="r2", choices=["r2", "mae", "mse"])
    p.add_argument("--out", default=None, help="also write the markdown here, "
                   "plus a long-format <out>.csv of every run")
    a = p.parse_args()

    blocks, longs = [], []
    for d in a.dirs:
        f = os.path.join(d, "runs.csv")
        if not os.path.exists(f):
            print(f"skip {d}: no runs.csv", file=sys.stderr)
            continue
        runs = pd.read_csv(f)
        r, text = arm_report(runs, a.target, a.ref, a.metric)
        blocks.append(f"## {d}  —  target `{a.target}`\n\n{text}")
        if r is not None:
            longs.append(r.assign(arm=os.path.basename(os.path.normpath(d)),
                                  group_label=r.group.map(label)))
    if not blocks:
        raise SystemExit("nothing to report")

    md = "\n\n".join(blocks) + "\n"
    print(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(md)
        if longs:
            pd.concat(longs).to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
