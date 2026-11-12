r"""
CatBoost regression of the permeability tensor from descriptor groups.

Trains one CatBoost model per (descriptor group, target) under shuffled k-fold
cross-validation and reports per-fold R^2, MAE and MSE. Descriptor tables are
joined to the labels on `sample_id`; a group selects its columns by prefix.
Main groups (paper name in brackets): baselines [Stats], phys [Phys],
classical [Baselines], tda [TDA, cone filtration], thr [THR, throat
filtration], and the unions all [Stats+TDA], baselines+thr, phys+tda,
phys+thr, classical+tda, classical+thr; por, tpc, fab, fabdir are the single
Stats families. The full mapping is in the comment above PREFIXES.
Rows whose LBM solve did not converge (`conv_fx/fy/fz != 1`) are dropped.

Protocol used in the paper: 5 folds, one model per target (single-target),
1500 iterations, depth 6, learning rate 0.05, no early stopping, seeds 0, 1, 2
(the seed sets both the fold shuffle and CatBoost's random seed). Every group
is scored on the same folds for a given seed, so groups can be compared fold
by fold (see metric_compare.py and metric_compare_seeds.py).

Targets: K_xx, K_yy, K_zz (diagonal) and k_xy, k_xz, k_yz, the symmetrised
off-diagonals (mean of K_ij and K_ji). The aliases `tensor3d`, `diagonal3d`
and `offdiag3d` expand to these. Always pass --targets: the default
(`tensor invariant`) names 2D label columns that the 3D tables do not have.

Outputs (in --output)
    results.csv      one row per (group, target): mean/std of R^2, MAE, MSE
    metrics.json     the same with every per-fold value and run settings
    arguments.json   command-line arguments
    oof.csv          out-of-fold predictions (with --save-oof)

--targets tensor3d --multioutput --scale-targets fits one joint MultiRMSE model
for all six components on standardised targets (Supplementary Information,
"One model for the whole permeability tensor").

Further options not used in the paper: --group-cv (leave-one-group-out CV over a labels column),
--residual-of (fit each group to the residual of a reference group, with
cross-fitted training residuals), and row filters (--strata,
--porosity-range, --filter-column).

Run one (target, group) per output directory, as pipeline/04_train.sh does;
the per-fold values are then the first experiment of each metrics.json.

Usage
    F="DESC/aniso3d/cone.csv DESC/aniso3d/throat.csv \
       DESC/aniso3d/baselines.csv DESC/aniso3d/phys.csv"
    for t in K_xx K_yy K_zz k_xy k_xz k_yz; do
      for g in baselines classical tda thr classical+thr; do
        python scripts/train_catboost.py --dataset DATA/aniso3d \
            --descriptors $F --targets $t --groups $g \
            --folds 5 --seed 0 --output RESULTS/aniso3d/seed0/${t}__$g
      done
    done
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold, LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import fold_metrics, summarize, fmt, write_metrics_json  # noqa: E402

# Descriptor groups, matched by column prefix. Paper names <- code keys:
#   Stats            <- baselines       (por_ + tpc_ + fab_ + fabdir_)
#   Phys             <- phys            (kc_ + cp_)
#   Baselines        <- classical       (Stats + Phys)
#   TDA              <- tda             (ecp_ + ph_, cone filtration)
#   THR              <- thr             (thr_, throat filtration)
#   Stats+TDA        <- all
#   Stats+THR        <- baselines+thr
#   Phys+TDA         <- phys+tda
#   Phys+THR         <- phys+thr
#   Baselines+TDA    <- classical+tda
#   Baselines+THR    <- classical+thr
# The single prefixes (por, tpc, fab, fabdir, ...) give the separate rows of
# the Stats family. `porosity` (the scalar alone) is handled in select().
PREFIXES = {
    "ecp": ("ecp_",),
    "ph": ("ph_",),
    "tda": ("ecp_", "ph_"),
    "por": ("por_",),
    "tpc": ("tpc_",),
    "fab": ("fab_",),
    # Directional fabric profile (baselines3d.py). "fabdir_" does not match
    # "fab_", so `fab` stays the 10-feature global fabric tensor; fabdir_ has
    # to be listed explicitly in every union that should include it.
    "fabdir": ("fabdir_",),
    # Pore-connectivity features; no script in this release produces conn_
    # columns, so the conn groups below select nothing here.
    "conn": ("conn_",),
    # Throat-scale PH from the distance-transform filtration
    # (filtration_throat3d.py + ph3d.py).
    "thr": ("thr_",),
    "baselines": ("por_", "tpc_", "fab_", "fabdir_"),
    "all": ("ecp_", "ph_", "por_", "tpc_", "fab_", "fabdir_"),
    "baselines+conn": ("por_", "tpc_", "fab_", "fabdir_", "conn_"),
    "tda+conn": ("ecp_", "ph_", "conn_"),
    "all+conn": ("ecp_", "ph_", "por_", "tpc_", "fab_", "fabdir_", "conn_"),
    "tpc+thr": ("tpc_", "thr_"),
    "baselines+thr": ("por_", "tpc_", "fab_", "fabdir_", "thr_"),
    "tda+thr": ("ecp_", "ph_", "thr_"),
    "all+thr": ("ecp_", "ph_", "por_", "tpc_", "fab_", "fabdir_", "thr_"),
    # Physics baselines (physbaselines3d.py): Kozeny-Carman (`kc_`, including
    # the fabric-weighted KC tensor) and the directional critical-path radius
    # with a Katz-Thompson estimate (`cp_`).
    "kc": ("kc_",),
    "cp": ("cp_",),
    "phys": ("kc_", "cp_"),
    "phys+thr": ("kc_", "cp_", "thr_"),
    "phys+tda": ("kc_", "cp_", "ecp_", "ph_"),
    "classical": ("por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_"),
    "classical+thr": ("por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_", "thr_"),
    "classical+tda": ("por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_", "ecp_", "ph_"),
    "classical+tda+thr": ("por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_", "ecp_", "ph_", "thr_"),
    # Generator parameters (gen_ columns) as an oracle reference for
    # --residual-of. Not used in the paper; the gen_ table is not shipped.
    "gen": ("gen_",),
    "gen+classical": ("gen_", "por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_"),
}

# Target aliases. The first block and k11/k22/k12/full_tensor refer to 2D
# label columns (k_off, invariants) and are not used for the 3D data.
TARGET_GROUPS = {
    "diagonal": ["k_xx", "k_yy"],
    "offdiag": ["k_off"],
    "tensor": ["k_xx", "k_yy", "k_off"],
    "invariant": ["log_k_mean", "log_k_ratio", "cos2theta", "sin2theta"],
}
TARGET_GROUPS["all"] = TARGET_GROUPS["tensor"] + TARGET_GROUPS["invariant"]

TARGET_GROUPS["k11"] = ["k_xx"]
TARGET_GROUPS["k22"] = ["k_yy"]
TARGET_GROUPS["k12"] = ["k_off"]
TARGET_GROUPS["full_tensor"] = ["k_xx", "k_yy", "k_off"]

# 3D. The off-diagonals are the symmetrised pair averages (k_xy = mean of K_xy
# and K_yx): K = K^T holds for Stokes flow, so averaging the two measurements
# reduces label noise. The raw K_xy, K_yx, ... columns remain selectable by name.
TARGET_GROUPS["tensor3d"] = ["K_xx", "K_yy", "K_zz", "k_xy", "k_xz", "k_yz"]
TARGET_GROUPS["diagonal3d"] = ["K_xx", "K_yy", "K_zz"]
TARGET_GROUPS["offdiag3d"] = ["k_xy", "k_xz", "k_yz"]


def select(df, group):
    if group == "porosity":
        return ["porosity"]
    pref = PREFIXES[group]
    return [c for c in df.columns if c.startswith(pref)]


def resolve_targets(names, cols):
    out = []
    for n in names:
        out.extend(TARGET_GROUPS.get(n, [n]))
    missing = [t for t in out if t not in cols]
    if missing:
        raise SystemExit(f"targets not in the table: {missing}")
    return list(dict.fromkeys(out))


def koff_noise_ceiling(df):
    """
    Upper bound on achievable R^2 for the 2D target k_off, from the K = K^T
    residual. Returns None when k_off/k_xy/k_yx are absent (all 3D data).

    |k_xy - k_yx| is a direct estimate of the per-sample error on k_off, so its
    variance sets the noise floor. Halved because k_off is the mean of the two,
    which averages the error down by sqrt(2) in variance.
    """
    if not {"k_xy", "k_yx", "k_off"} <= set(df.columns):
        return None
    err = (df["k_xy"] - df["k_yx"]).abs() / 2.0
    var_noise = float((err ** 2).mean())
    var_sig = float(df["k_off"].var())
    return max(0.0, 1.0 - var_noise / var_sig) if var_sig > 0 else None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--descriptors", nargs="+", required=True)
    p.add_argument("--labels", default=None,
                   help="default: <dataset>/permeability.csv")
    p.add_argument("--groups", nargs="+",
                   default=["tda", "baselines", "fab", "porosity"])
    p.add_argument("--targets", nargs="+", default=["tensor", "invariant"])
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--threads", type=int, default=-1,
                   help="CatBoost thread_count (-1 = all cores)")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--save-oof", dest="save_oof", action="store_true",
                   help="single-target mode: also write every out-of-fold "
                        "prediction to <output>/oof.csv (for scatter plots)")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--multioutput", action="store_true",
                   help="one joint model over all targets (CatBoost MultiRMSE) "
                        "instead of one model per target")
    p.add_argument("--scale-targets", dest="scale_targets", action="store_true",
                   help="with --multioutput: z-score each target using "
                        "training-fold statistics before the joint fit and "
                        "invert on the predictions, so metrics stay in raw "
                        "units. MultiRMSE sums squared errors over targets, "
                        "so without scaling the highest-variance target "
                        "dominates the loss. No effect on single-target fits.")
    p.add_argument("--strata", nargs="+", default=None,
                   help="restrict to these strata, e.g. mod strong")
    p.add_argument("--porosity-range", dest="porosity_range", nargs=2,
                   type=float, default=None, metavar=("LO", "HI"),
                   help="restrict to porosity in [LO, HI); the model is "
                        "trained and scored inside the band. R^2 is not "
                        "comparable across bands of different width or size; "
                        "compare differences between groups within a band.")
    p.add_argument("--filter-column", dest="filter_column", nargs=3,
                   default=None, metavar=("NAME", "LO", "HI"),
                   help="restrict to rows with labels-table column NAME in "
                        "[LO, HI); the general form of --porosity-range. NAME "
                        "must not be a feature of any requested group.")
    p.add_argument("--group-cv", dest="group_cv", default=None, metavar="COLUMN",
                   help="leave-one-group-out CV over this labels-table column "
                        "instead of shuffled k-fold (not used in the paper)")
    p.add_argument("--train-per-group", dest="train_per_group", type=int, default=0,
                   metavar="M", help="with --group-cv: train on M rows per training "
                        "group (nested, seeded subsample); the test group stays whole")
    p.add_argument("--subsample-seed", dest="subsample_seed", type=int, default=0,
                   help="seed of the --train-per-group permutation")
    p.add_argument("--residual-of", dest="residual_of", default=None, metavar="GROUP",
                   help="fit every group to the residual y - f_ref(x_ref) of a reference "
                        "group (e.g. classical) instead of to y; not used in the paper")
    p.add_argument("--residual-inner-folds", dest="residual_inner_folds", type=int, default=5,
                   help="inner cross-fitting folds for the training-row residuals")
    a = p.parse_args()

    if a.residual_of and a.multioutput:
        raise SystemExit("--residual-of is single-target only")

    if a.scale_targets and not a.multioutput:
        print("  ! --scale-targets has no effect without --multioutput "
              "(single-target fits do not share a loss); ignoring.")

    import catboost as cb

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    if not os.path.exists(labels):
        raise SystemExit(f"no labels at {labels} — has the LBM run finished?")

    df = pd.read_csv(labels)
    for path in a.descriptors:
        extra = pd.read_csv(path)
        drop = [c for c in extra.columns if c != "sample_id" and c in df.columns]
        df = df.merge(extra.drop(columns=drop), on="sample_id", how="inner")

    n0 = len(df)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    if a.strata:
        df = df[df.stratum.isin(a.strata)]
    if a.porosity_range:
        lo, hi = a.porosity_range
        if "porosity" not in df.columns:
            raise SystemExit("--porosity-range needs a 'porosity' column")
        n_before = len(df)
        df = df[(df.porosity >= lo) & (df.porosity < hi)]
        print(f"porosity [{lo}, {hi}): {n_before} -> {len(df)} rows")
    if a.filter_column:
        name, lo, hi = a.filter_column[0], float(a.filter_column[1]), float(a.filter_column[2])
        if name not in df.columns:
            raise SystemExit(f"--filter-column: no '{name}' column in the labels table")
        for group in a.groups:
            if name in select(df, group):
                raise SystemExit(f"--filter-column: '{name}' is a feature of "
                                 f"group '{group}' -- filtering on it would "
                                 f"condition on a predictor")
        n_before = len(df)
        df = df[(df[name] >= lo) & (df[name] < hi)]
        print(f"{name} [{lo}, {hi}): {n_before} -> {len(df)} rows")
    if a.group_cv:
        if a.group_cv not in df.columns:
            raise SystemExit(f"--group-cv: no '{a.group_cv}' column in the labels table")
        for group in a.groups + ([a.residual_of] if a.residual_of else []):
            if a.group_cv in select(df, group):
                raise SystemExit(f"--group-cv: '{a.group_cv}' is a feature of group "
                                 f"'{group}' -- it would identify the held-out group")
    if a.train_per_group and not a.group_cv:
        raise SystemExit("--train-per-group needs --group-cv")
    if a.limit:
        df = df.head(a.limit)
    df = df.reset_index(drop=True)
    print(f"{n0} rows -> {len(df)} after convergence/stratum filtering")

    targets = resolve_targets(a.targets, df.columns)
    ceiling = koff_noise_ceiling(df)
    if ceiling is not None:
        print(f"k_off label-noise ceiling (from K=K^T): R^2 <= {ceiling:.4f}")

    os.makedirs(a.output, exist_ok=True)
    gcol = df[a.group_cv].astype(str).to_numpy() if a.group_cv else None
    if a.group_cv:
        sizes = pd.Series(gcol).value_counts().sort_index()
        print(f"leave-one-{a.group_cv}-out: {len(sizes)} folds "
              f"({', '.join(f'{k} {v}' for k, v in sizes.items())})")

    def splits(ok):
        """(fold index, train idx, test idx, held-out label) over the rows in `ok`."""
        if a.group_cv:
            g = gcol[ok]
            if a.train_per_group:
                # rank of each row inside its group's seeded permutation; the
                # same for every fold, so subsets are nested and paired
                rng = np.random.RandomState(a.subsample_seed)
                rank = np.empty(len(g), dtype=np.int64)
                for name in sorted(set(g)):
                    idx = np.flatnonzero(g == name)
                    rank[idx[rng.permutation(len(idx))]] = np.arange(len(idx))
            for f_i, (tr, te) in enumerate(LeaveOneGroupOut().split(g, groups=g)):
                if a.train_per_group:
                    tr = tr[rank[tr] < a.train_per_group]
                yield f_i, tr, te, g[te[0]]
        else:
            cv = KFold(a.folds, shuffle=True, random_state=a.seed)
            for f_i, (tr, te) in enumerate(cv.split(np.zeros(int(ok.sum())))):
                yield f_i, tr, te, None

    def fold_row(y_true, y_pred, f_i, held):
        r = fold_metrics(y_true, y_pred, fold=f_i)
        if held is not None:
            r["held_out"] = held
        return r

    def pooled(y_true, y_pred):
        """Metrics of all out-of-fold predictions together (group CV only)."""
        if not a.group_cv:
            return {}
        m = fold_metrics(np.concatenate(y_true), np.concatenate(y_pred))
        return {"r2_pooled": m["r2"], "mae_pooled": m["mae"], "mse_pooled": m["mse"]}

    def catboost():
        return cb.CatBoostRegressor(iterations=a.iterations, depth=a.depth,
                                    learning_rate=a.lr, random_seed=a.seed, thread_count=a.threads,
                                    verbose=0, allow_writing_files=False)

    # --residual-of: per (target, fold), the reference model's residuals. Test
    # rows: y - f_ref(x), f_ref fitted on the whole training fold. Training
    # rows: cross-fitted inside the training fold (inner k-fold, or inner
    # group k-fold under --group-cv), so a training residual is never the
    # in-sample residual of a model that saw that row (in-sample residuals
    # shrink with overfitting and would teach the second stage the wrong scale).
    # Computed once per target and shared by every group, so all groups are
    # scored against the same residual and pair fold-for-fold.
    resid = {}
    if a.residual_of:
        from sklearn.model_selection import GroupKFold
        rcols = select(df, a.residual_of)
        if not rcols:
            raise SystemExit(f"--residual-of: group '{a.residual_of}' has no columns")
        Xr = df[rcols].to_numpy(np.float32)
        for t in targets:
            y = df[t].to_numpy(np.float64)
            ok = np.isfinite(y)
            Xo, yo = Xr[ok], y[ok]
            t0 = time.time()
            resid[t] = []
            for f_i, tr, te, held in splits(ok):
                ref_te = catboost().fit(Xo[tr], yo[tr]).predict(Xo[te])
                ref_tr = np.empty(len(tr))
                if a.group_cv:
                    g_tr = gcol[ok][tr]
                    inner = GroupKFold(min(a.residual_inner_folds, len(set(g_tr)))).split(tr, groups=g_tr)
                else:
                    inner = KFold(a.residual_inner_folds, shuffle=True,
                                  random_state=a.seed + 1).split(tr)
                for itr, ite in inner:
                    ref_tr[ite] = catboost().fit(Xo[tr][itr], yo[tr][itr]).predict(Xo[tr][ite])
                resid[t].append((f_i, tr, te, held, yo[tr] - ref_tr, yo[te] - ref_te, ref_te))
            ref = summarize([fold_metrics(yo[r[2]], r[6]) for r in resid[t]])
            print(f"  reference {a.residual_of} -> {t}: R2 {ref['r2_mean']:.4f} "
                  f"± {ref['r2_std']:.4f}  ({time.time() - t0:.0f}s, residuals cross-fitted)")

    rows = []
    per_fold = []
    oof_rows = []

    for group in a.groups:
        cols = select(df, group)
        if not cols:
            print(f"  ! group {group}: no columns, skipping")
            continue
        X = df[cols].to_numpy(np.float32)

        if a.multioutput:
            Y = df[targets].to_numpy(np.float64)
            ok = np.isfinite(Y).all(1)
            per = {t: [] for t in targets}
            oof = {t: ([], []) for t in targets}
            t0 = time.time()
            for f_i, tr, te, held in splits(ok):
                m = cb.CatBoostRegressor(iterations=a.iterations, depth=a.depth,
                                         learning_rate=a.lr, random_seed=a.seed, thread_count=a.threads,
                                         loss_function="MultiRMSE", verbose=0,
                                         allow_writing_files=False)
                Ytr = Y[ok][tr]
                if a.scale_targets:
                    # Fitted on the training fold only, so no label information
                    # from the test rows leaks into mu/sd.
                    mu, sd = Ytr.mean(0), Ytr.std(0)
                    sd[sd == 0] = 1.0
                    Ytr = (Ytr - mu) / sd
                m.fit(X[ok][tr], Ytr)
                pred = np.atleast_2d(m.predict(X[ok][te]))
                if a.scale_targets:
                    # Back to raw units before scoring, so MAE and MSE are
                    # comparable with unscaled runs (R^2 alone is invariant
                    # under a per-target affine map).
                    pred = pred * sd + mu
                for i, t in enumerate(targets):
                    per[t].append(fold_row(Y[ok][te][:, i], pred[:, i], f_i, held))
                    oof[t][0].append(Y[ok][te][:, i])
                    oof[t][1].append(pred[:, i])
            secs = round(time.time() - t0, 1)
            for t in targets:
                summ = {**summarize(per[t]), **pooled(*oof[t])}
                rows.append({"group": group, "n_features": len(cols), "target": t,
                             "mode": "joint", **summ,
                             "n": int(ok.sum()), "seconds": secs})
                per_fold.append({"group": group, "target": t, "mode": "joint",
                                 "n_features": len(cols), "n": int(ok.sum()),
                                 "seconds": secs, "folds": per[t], **summ})
                pr = f"  pooled R2 {summ['r2_pooled']:.4f}" if a.group_cv else ""
                print(f"  {group:10s} {len(cols):5d}f  {t:14s} [joint] {fmt(summ)}{pr}")
            continue

        for t in targets:
            y = df[t].to_numpy(np.float64)
            ok = np.isfinite(y)
            folds = []
            oof = ([], [])
            t0 = time.time()
            if a.residual_of:
                # R^2 on the residual = share of what the reference leaves
                # unexplained that this group recovers; r2_total scores the
                # stacked prediction ref + residual model against y itself.
                sid = df["sample_id"].to_numpy()[ok]
                for f_i, tr, te, held, r_tr, r_te, ref_te in resid[t]:
                    pred = catboost().fit(X[ok][tr], r_tr).predict(X[ok][te])
                    row = fold_row(r_te, pred, f_i, held)
                    row["r2_ref"] = fold_metrics(y[ok][te], ref_te)["r2"]
                    row["r2_total"] = fold_metrics(y[ok][te], ref_te + pred)["r2"]
                    folds.append(row)
                    oof[0].append(r_te)
                    oof[1].append(pred)
                    oof_rows.append(pd.DataFrame({"group": group, "target": t, "fold": f_i,
                                                  "sample_id": sid[te], "y": y[ok][te],
                                                  "ref": ref_te, "resid_pred": pred}))
                secs = round(time.time() - t0, 1)
                summ = {**summarize(folds), **pooled(*oof),
                        "r2_ref_mean": float(np.mean([f["r2_ref"] for f in folds])),
                        "r2_total_mean": float(np.mean([f["r2_total"] for f in folds])),
                        "residual_of": a.residual_of}
                rows.append({"group": group, "n_features": len(cols), "target": t,
                             "mode": "residual", **summ, "n": int(ok.sum()), "seconds": secs})
                per_fold.append({"group": group, "target": t, "mode": "residual",
                                 "n_features": len(cols), "n": int(ok.sum()),
                                 "seconds": secs, "folds": folds, **summ})
                print(f"  {group:10s} {len(cols):5d}f  {t:14s} [resid of {a.residual_of}] "
                      f"{fmt(summ)}  ref {summ['r2_ref_mean']:.4f} -> total "
                      f"{summ['r2_total_mean']:.4f}  ({secs}s)")
                continue
            for f_i, tr, te, held in splits(ok):
                m = cb.CatBoostRegressor(iterations=a.iterations, depth=a.depth,
                                         learning_rate=a.lr, random_seed=a.seed, thread_count=a.threads,
                                         verbose=0, allow_writing_files=False)
                m.fit(X[ok][tr], y[ok][tr])
                pred = m.predict(X[ok][te])
                folds.append(fold_row(y[ok][te], pred, f_i, held))
                oof[0].append(y[ok][te])
                oof[1].append(pred)
                if a.save_oof:
                    oof_rows.append(pd.DataFrame({
                        "group": group, "target": t, "fold": f_i,
                        "sample_id": df["sample_id"].to_numpy()[ok][te],
                        "y": y[ok][te], "pred": pred}))
            secs = round(time.time() - t0, 1)
            summ = {**summarize(folds), **pooled(*oof)}
            rows.append({"group": group, "n_features": len(cols), "target": t,
                         "mode": "single", **summ,
                         "n": int(ok.sum()), "seconds": secs})
            per_fold.append({"group": group, "target": t, "mode": "single",
                             "n_features": len(cols), "n": int(ok.sum()),
                             "seconds": secs, "folds": folds, **summ})
            pr = f"  pooled R2 {summ['r2_pooled']:.4f}" if a.group_cv else ""
            print(f"  {group:10s} {len(cols):5d}f  {t:14s} {fmt(summ)}{pr}  ({secs}s)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.output, "results.csv"), index=False)
    if oof_rows:
        pd.concat(oof_rows).to_csv(os.path.join(a.output, "oof.csv"), index=False)
    write_metrics_json(a.output,
                       {"kind": "catboost",
                        "folds": int(len(np.unique(gcol))) if a.group_cv else a.folds,
                        "cv": f"leave-one-{a.group_cv}-out" if a.group_cv else "kfold",
                        "seed": a.seed,
                        "iterations": a.iterations, "depth": a.depth, "lr": a.lr,
                        "multioutput": bool(a.multioutput),
                        "porosity_range": a.porosity_range,
                        "train_per_group": a.train_per_group,
                        "subsample_seed": a.subsample_seed,
                        "residual_of": a.residual_of,
                        "residual_inner_folds": a.residual_inner_folds if a.residual_of else None,
                        "scale_targets": bool(a.scale_targets and a.multioutput),
                        "n_rows": len(df), "koff_ceiling": ceiling},
                       per_fold)
    with open(os.path.join(a.output, "arguments.json"), "w") as fh:
        json.dump({**vars(a), "n_rows": len(df), "koff_ceiling": ceiling}, fh, indent=2)

    print("\n=== R^2 by group and target ===")
    print(res.pivot(index="group", columns="target", values="r2_mean")
             .round(4).to_string())
    print(f"\n-> {a.output}/results.csv")


if __name__ == "__main__":
    main()
