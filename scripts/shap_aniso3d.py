#!/usr/bin/env python
r"""
SHAP attribution for the aniso3d CatBoost models.

For each (target, group) one CatBoost model is fitted on the training part of
fold 0 of KFold(5, shuffle=True, random_state=--seed), with the same settings
as train_catboost.py (1500 iterations, depth 6, lr 0.05, RMSE). Exact TreeSHAP
values (CatBoost's ShapValues) are computed on that fold's held-out rows, so
the attributions are out-of-sample; the test R^2 is printed with each fit.
Feature shares are then aggregated per descriptor family and, for THR
features, per direction and homology dimension.

THR column layout (filtration_throat3d.py + ph3d.py): thr_a<dx>_<dy>_<dz>_<idx>,
idx 0..299 = H0, H1, H2 persistence images of 10x10 pixels. gudhi flattens an
image as idx = p_row*10 + b_col (b = birth along columns, p = persistence
along rows), with pixel centres at linspace(lo, hi, 10) of the image range in
RANGES. The throat filtration is f = 1 - min(dt along +-d)/RCAP with RCAP = 10
voxels (sublevel), so a level f corresponds to throat radius r = RCAP*(1 - f):
birth radius r_b = RCAP*(1 - b), death radius r_d = RCAP*(1 - b - p). For H0,
r_b is a pore-body radius and r_d the throat that merges it.

Outputs, per <output>/<target>__<group>/ (existing outputs are skipped):
    shap.npz            test ids, feature names, SHAP matrix, X, y, prediction
    meanabs.csv         per-feature mean |SHAP|, share, family, and the
                        correlation of feature value with its SHAP value
    info.json           test R^2, timing, share per descriptor family
    thr_by_dir_dim.csv  (groups with thr_) share per direction x homology dim
    thr_pixels.csv      (groups with thr_) per-pixel shares with birth/death radii

Usage
    python scripts/shap_aniso3d.py --labels DATA/aniso3d/permeability.csv \
        --descriptors DESC/aniso3d/throat.csv DESC/aniso3d/baselines.csv \
                      DESC/aniso3d/phys.csv \
        --groups thr classical classical+thr --targets k_xy k_xz k_yz K_xx \
        --output RESULTS/aniso3d/shap
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

RCAP = 10.0
RES = 10
# Persistence-image ranges (birth lo, hi, persistence lo, hi) per homology
# dimension; must match the --im_range_h* values used with ph3d.py for thr_.
RANGES = {0: (0, .9, 0, .40), 1: (0, .9, 0, .75), 2: (0, .9, 0, .50)}

PREFIX = {
    "thr": ("thr_",),
    "classical": ("por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_"),
    "classical+thr": ("por_", "tpc_", "fab_", "fabdir_", "kc_", "cp_", "thr_"),
}
# Direction pair of each off-diagonal target (reference only; unused below).
PAIR = {"k_xy": ("a1_1_0", "a1_-1_0"), "k_xz": ("a1_0_1", "a1_0_-1"),
        "k_yz": ("a0_1_1", "a0_1_-1")}


def family(col):
    for p in ("fabdir_", "fab_", "por_", "tpc_", "cp_", "thr_"):
        if col.startswith(p):
            return p[:-1]
    if col.startswith("kc_t_"):
        return "kc_tensor"
    if col.startswith("kc_"):
        return "kc_scalar"
    return "other"


def thr_parse(col):
    """thr_a1_-1_0_123 -> ('a1_-1_0', dim, birth, pers)."""
    body = col[len("thr_"):]
    d, idx = body.rsplit("_", 1)
    idx = int(idx)
    dim, j = divmod(idx, RES * RES)
    prow, bcol = divmod(j, RES)
    b0, b1, p0, p1 = RANGES[dim]
    b = b0 + bcol * (b1 - b0) / (RES - 1)
    p = p0 + prow * (p1 - p0) / (RES - 1)
    return d, dim, b, p


def load(a):
    df = pd.read_csv(a.labels)
    df = df[(df[["conv_fx", "conv_fy", "conv_fz"]] == 1).all(axis=1)]
    for path in a.descriptors:
        extra = pd.read_csv(path)
        drop = [c for c in extra.columns if c != "sample_id" and c in df.columns]
        df = df.merge(extra.drop(columns=drop), on="sample_id", how="inner")
    return df.sort_values("sample_id").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="DATA/aniso3d/permeability.csv")
    p.add_argument("--descriptors", nargs="+", default=[
        "DESC/aniso3d_throat/descriptors.csv", "DESC/aniso3d/baselines.csv",
        "DESC/aniso3d_phys/physbaselines.csv"])
    p.add_argument("--groups", nargs="+", default=["thr", "classical", "classical+thr"])
    p.add_argument("--targets", nargs="+", default=["k_xy", "k_xz", "k_yz", "K_xx"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--output", default="RESULTS/shap_aniso3d")
    a = p.parse_args()

    df = load(a)
    print(f"{len(df)} rows after merge + convergence filter", flush=True)
    tr, te = next(KFold(5, shuffle=True, random_state=a.seed).split(df))

    for target in a.targets:
        for group in a.groups:
            out = os.path.join(a.output, f"{target}__{group}")
            if os.path.exists(os.path.join(out, "meanabs.csv")):
                print(f"skip {out} (done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            cols = [c for c in df.columns if c.startswith(PREFIX[group])]
            X = df[cols].to_numpy(np.float32)
            y = df[target].to_numpy(np.float64)
            t0 = time.time()
            m = CatBoostRegressor(iterations=a.iterations, depth=6, learning_rate=.05,
                                  loss_function="RMSE", random_seed=a.seed,
                                  thread_count=a.threads, verbose=0)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            r2 = 1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
            sv = m.get_feature_importance(Pool(X[te], y[te]), type="ShapValues",
                                          thread_count=a.threads)
            base, sv = sv[0, -1], sv[:, :-1]
            np.savez_compressed(os.path.join(out, "shap.npz"), ids=df.sample_id.to_numpy()[te],
                                cols=np.array(cols), shap=sv.astype(np.float32),
                                X=X[te], y=y[te], pred=pred, base=base)

            ma = np.abs(sv).mean(0)
            # sign: corr(feature value, SHAP); positive means "more of it raises the target"
            sgn = np.array([np.corrcoef(X[te, i], sv[:, i])[0, 1]
                            if X[te, i].std() > 0 and sv[:, i].std() > 0 else 0.0
                            for i in range(len(cols))])
            fe = pd.DataFrame({"feature": cols, "family": [family(c) for c in cols],
                               "mean_abs_shap": ma, "share": ma / ma.sum(),
                               "corr_x_shap": sgn}).sort_values("mean_abs_shap",
                                                                ascending=False)
            fe.to_csv(os.path.join(out, "meanabs.csv"), index=False)

            if any(c.startswith("thr_") for c in cols):
                t = fe[fe.feature.str.startswith("thr_")].copy()
                parsed = t.feature.map(thr_parse)
                t["dir"] = [q[0] for q in parsed]
                t["dim"] = [q[1] for q in parsed]
                t["birth"] = [q[2] for q in parsed]
                t["pers"] = [q[3] for q in parsed]
                t["r_birth"] = RCAP * (1 - t.birth)
                t["r_death"] = RCAP * (1 - t.birth - t.pers)
                t.groupby(["dir", "dim"]).share.sum().unstack().to_csv(
                    os.path.join(out, "thr_by_dir_dim.csv"))
                t.to_csv(os.path.join(out, "thr_pixels.csv"), index=False)

            info = {"target": target, "group": group, "n_features": len(cols),
                    "n_train": len(tr), "n_test": len(te), "r2_test": r2,
                    "seconds": time.time() - t0,
                    "family_share": fe.groupby("family").share.sum().to_dict()}
            json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
            print(f"{target:5s} {group:14s} R2 {r2:.4f}  {time.time()-t0:6.0f}s  "
                  + "  ".join(f"{k} {v:.2f}" for k, v in
                              sorted(info["family_share"].items(), key=lambda kv: -kv[1])),
                  flush=True)


if __name__ == "__main__":
    main()
