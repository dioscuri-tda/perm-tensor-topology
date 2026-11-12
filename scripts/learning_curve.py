r"""
Learning curves: hold-out accuracy as a function of the number of training
structures, per descriptor group and target.

Protocol
    * One frozen hold-out (`--split`, e.g. splits/aniso3d_golden.csv: 999
      structures, balanced over the three anisotropy strata) is never trained
      on and scores every (group, size, seed). make_golden_split.load_split
      checks its ids against the sha256 in the accompanying .json.
    * Training subsets come from the remaining pool. For each seed the pool is
      ordered by a round-robin over the strata (each stratum shuffled), and the
      size-n subset is the first n structures of that order, so subsets are
      nested (n is contained in 2n) and stratum-balanced.
    * One CatBoost fit per (target, group, size, seed); the seed sets both the
      subset order and CatBoost's random seed. Groups and prefixes are those of
      train_catboost.py.
    * Early stopping on an inner validation split (carved from the training
      subset, never from the hold-out) is ON by default. The paper uses the
      fixed-budget variant: pass --no-early-stop (1500 iterations, depth 6,
      lr 0.05, as in train_catboost.py).

The defaults of --dataset, --descriptors, --split and --targets refer to a 2D
setup and must be overridden for the 3D data (see the usage example).

Outputs (in --output)
    runs.csv       one row per (target, group, size, seed): R^2, MAE, MSE
    summary.csv    mean and sd over seeds per (target, group, size)
    metrics.json   settings and every run
Resumable: runs already present in runs.csv are skipped. Tables and paired
differences are produced by lc_table.py.

Usage (paper settings, one target)
    python scripts/learning_curve.py --dataset DATA/aniso3d \
        --descriptors DESC/aniso3d/cone.csv DESC/aniso3d/throat.csv \
                      DESC/aniso3d/baselines.csv \
        --split splits/aniso3d_golden.csv --no-early-stop \
        --groups baselines tda thr baselines+thr por tpc fab fabdir \
        --targets k_xy --sizes 50 75 100 125 250 500 1000 2000 4000 \
        --seeds 0 1 2 3 4 --output RESULTS/aniso3d/lc_kxy/fixed
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import fold_metrics, summarize                      # noqa: E402
from make_golden_split import load_split                         # noqa: E402
from train_catboost import PREFIXES, TARGET_GROUPS, select       # noqa: E402

DEFAULT_SIZES = [125, 250, 500, 1000, 2000, 3990]


# ---------------------------------------------------------------------------

def nested_order(df, pool_ids, seed):
    """
    One ordering of the pool such that every prefix is stratum-balanced.

    Round-robin over the strata, each shuffled independently, so prefixes are
    both nested and balanced.
    """
    rng = np.random.default_rng(seed)
    pool = df[df.sample_id.isin(pool_ids)]
    queues = [list(rng.permutation(g.sample_id.to_numpy()))
              for _, g in sorted(pool.groupby("stratum"))]
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop())
    return np.array(order)


def fit_catboost(Xtr, ytr, Xte, yte, a, multi=False):
    import catboost as cb

    params = dict(iterations=a.iterations, depth=a.depth, learning_rate=a.lr,
                  random_seed=a.seed, verbose=0, allow_writing_files=False)
    if multi:
        params["loss_function"] = "MultiRMSE"

    if a.early_stop and len(Xtr) >= 40:
        # Inner validation split, carved from the training subset only;
        # choosing the iteration count on the hold-out would leak.
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(Xtr))
        nval = max(10, int(round(a.val_frac * len(Xtr))))
        vi, ti = idx[:nval], idx[nval:]
        m = cb.CatBoostRegressor(**params, od_type="Iter", od_wait=50)
        m.fit(Xtr[ti], ytr[ti], eval_set=(Xtr[vi], ytr[vi]), use_best_model=True)
        best = int(m.get_best_iteration() or a.iterations)
    else:
        m = cb.CatBoostRegressor(**params)
        m.fit(Xtr, ytr)
        best = a.iterations
    return m.predict(Xte), best


def fit_cnn(df_all, train_ids, test_ids, targets, a):
    """CNN arm via train_cnn.run_fold. Not used in the paper; train_cnn.py is
    not part of this release."""
    from argparse import Namespace
    import train_cnn as T

    sub = df_all[df_all.sample_id.isin(np.concatenate([train_ids, test_ids]))]
    sub = sub.reset_index(drop=True)
    pos = {int(s): i for i, s in enumerate(sub.sample_id)}
    tr = np.array([pos[int(s)] for s in train_ids])
    te = np.array([pos[int(s)] for s in test_ids])

    X = T.load_images(a.dataset, sub)
    Y = sub[targets].to_numpy(np.float64)
    args = Namespace(backbone=a.cnn, device=a.device, seed=a.seed, epochs=a.epochs,
                     batch=a.batch, augment=a.augment, lr=a.cnn_lr, wd=a.wd,
                     pretrained=False)
    return T.run_fold(X, Y, tr, te, args, targets), Y[te]


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="DATA/aniso")
    p.add_argument("--labels", default=None)
    p.add_argument("--descriptors", nargs="+",
                   default=["DESC/aniso/descriptors.csv", "DESC/aniso/baselines.csv"])
    p.add_argument("--split", default="splits/aniso_golden.csv")
    p.add_argument("--no-golden", action="store_true",
                   help="skip the hold-out (intended for K-fold CV on each subset, but no K-fold evaluation is implemented; not used in the paper)")
    p.add_argument("--folds", type=int, default=5, help="only printed with --no-golden; unused otherwise")
    p.add_argument("--groups", nargs="+",
                   default=["tda", "baselines", "tpc", "fab", "porosity"])
    p.add_argument("--targets", nargs="+", default=["k_off", "k_xx"])
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output", default="RESULTS/learning_curve")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    p.add_argument("--pca", type=int, default=0,
                   help="reduce every group to K components first (dimensionality control)")
    p.add_argument("--multioutput", action="store_true")
    # CNN arm (not used in the paper; needs train_cnn.py, not included)
    p.add_argument("--cnn", default=None, help="CNN backbone name; switches to the CNN arm (not used in the paper)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--augment", default="roll")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cnn-lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    a = p.parse_args()
    a.seed = 0

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    df = pd.read_csv(labels)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    df = df.reset_index(drop=True)

    targets = []
    for t in a.targets:
        targets.extend(TARGET_GROUPS.get(t, [t]))
    targets = list(dict.fromkeys(targets))
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise SystemExit(f"targets not in the table: {missing}")

    if a.no_golden:
        golden, pool = np.array([]), df.sample_id.to_numpy()
        print(f"no golden set: {a.folds}-fold CV on each subset, pool {len(pool)}")
    else:
        golden, pool = load_split(a.split)
        print(f"golden {len(golden)} (frozen, verified)   pool {len(pool)}")

    if a.cnn:
        feat = None
        print(f"model: CNN {a.cnn}  augment={a.augment}  epochs={a.epochs}")
    else:
        d = df[["sample_id"]].copy()
        for f in a.descriptors:
            d = d.merge(pd.read_csv(f), on="sample_id", how="left",
                        suffixes=("", "_dup"))
        feat = d
        # LEFT join: a structure missing from a descriptor table would come
        # back as NaN, which CatBoost accepts silently. Require every feature
        # of every requested group for every hold-out and pool row.
        need = sorted({c for g in a.groups for c in select(feat, g)})
        used = feat[feat.sample_id.isin(np.concatenate([golden, pool]))]
        bad = used[need].isna().any(axis=1)
        if bad.any():
            raise SystemExit(f"{int(bad.sum())} structures lack descriptors "
                             f"(first ids {used.sample_id[bad].head().tolist()}); "
                             "rebuild the descriptor tables before training.")
        feat = feat.set_index("sample_id")
        print(f"model: CatBoost  early_stop={a.early_stop}  "
              f"iterations={a.iterations}  pca={a.pca or 'off'}")

    os.makedirs(a.output, exist_ok=True)
    runs_csv = os.path.join(a.output, "runs.csv")
    done = set()
    rows = []
    if os.path.exists(runs_csv):
        prev = pd.read_csv(runs_csv)
        rows = prev.to_dict("records")
        done = {(r["target"], r["group"], int(r["size"]), int(r["seed"])) for r in rows}
        print(f"resuming: {len(done)} runs already present")

    sizes = [s for s in a.sizes if s <= len(pool)]
    groups = [a.cnn and f"cnn_{a.cnn}" or g for g in (["_"] if a.cnn else a.groups)]
    total = len(targets) * len(groups) * len(sizes) * len(a.seeds)
    print(f"\n{total} runs: {len(targets)} targets x {len(groups)} groups "
          f"x {len(sizes)} sizes x {len(a.seeds)} seeds\n")

    n_run = 0
    for seed in a.seeds:
        order = nested_order(df, pool, seed)
        for size in sizes:
            train_ids = order[:size]
            for gi, group in enumerate(groups):
                if a.cnn:
                    pass
                else:
                    cols = select(feat, a.groups[gi])
                    if not cols:
                        continue
                for target in targets:
                    key = (target, group, size, seed)
                    if key in done:
                        continue
                    t0 = time.time()

                    if a.cnn:
                        pred, ytrue = fit_cnn(df, train_ids, golden, [target], a)
                        pred = np.asarray(pred).reshape(len(golden), -1)[:, 0]
                        ytrue = ytrue[:, 0]
                    else:
                        sub = feat
                        Xtr = sub.loc[train_ids, cols].to_numpy(np.float32)
                        Xte = sub.loc[golden, cols].to_numpy(np.float32)
                        yi = df.set_index("sample_id")[target]
                        ytr = yi.loc[train_ids].to_numpy(float)
                        ytrue = yi.loc[golden].to_numpy(float)
                        if a.pca:
                            from sklearn.decomposition import PCA
                            k = min(a.pca, Xtr.shape[1], len(Xtr) - 1)
                            pc = PCA(n_components=k, random_state=0).fit(Xtr)
                            Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
                        a.seed = seed
                        pred, best = fit_catboost(Xtr, ytr, Xte, ytrue, a)

                    m = fold_metrics(ytrue, pred)
                    rec = {"target": target, "group": group, "size": int(size),
                           "seed": int(seed), "n_eval": int(len(ytrue)),
                           "n_features": (256 * 256 if a.cnn else len(cols)),
                           "seconds": round(time.time() - t0, 1),
                           **{k2: v for k2, v in m.items() if k2 != "n_test"}}
                    if not a.cnn:
                        rec["best_iteration"] = best
                    rows.append(rec)
                    n_run += 1
                    print(f"  {target:12s} {group:10s} n={size:5d} seed={seed} "
                          f"R2 {m['r2']:7.4f}  MAE {m['mae']:.4g}  "
                          f"({rec['seconds']}s)")
                    pd.DataFrame(rows).to_csv(runs_csv, index=False)

    res = pd.DataFrame(rows)
    if res.empty:
        print("nothing to do")
        return 0

    agg = (res.groupby(["target", "group", "size"])
              .agg(r2_mean=("r2", "mean"), r2_std=("r2", lambda s: s.std(ddof=1)),
                   mae_mean=("mae", "mean"), mae_std=("mae", lambda s: s.std(ddof=1)),
                   mse_mean=("mse", "mean"), mse_std=("mse", lambda s: s.std(ddof=1)),
                   n_seeds=("seed", "nunique"))
              .reset_index())
    agg.to_csv(os.path.join(a.output, "summary.csv"), index=False)

    with open(os.path.join(a.output, "metrics.json"), "w") as fh:
        json.dump({"split": a.split if not a.no_golden else None,
                   "descriptors": None if a.cnn else a.descriptors,
                   "n_golden": int(len(golden)), "n_pool": int(len(pool)),
                   "sizes": sizes, "seeds": a.seeds, "targets": targets,
                   "groups": groups, "early_stop": bool(a.early_stop),
                   "iterations": a.iterations, "pca": a.pca,
                   "model": f"cnn_{a.cnn}" if a.cnn else "catboost",
                   "runs": rows}, fh, indent=2)

    for t in targets:
        sub = agg[agg.target == t]
        if sub.empty:
            continue
        print(f"\n=== R^2 on the golden set — target {t} ===")
        print(sub.pivot(index="group", columns="size", values="r2_mean")
                 .round(4).to_string())
    print(f"\n-> {runs_csv}\n-> {a.output}/summary.csv\n-> {a.output}/metrics.json")
    print(f"{n_run} new runs this invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
