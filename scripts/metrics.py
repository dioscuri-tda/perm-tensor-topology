"""
Per-fold regression metrics shared by the training and solver scripts.

Every experiment records, per cross-validation fold: R^2, MAE and MSE. The
summary carries the mean and standard deviation of each across folds, and the
full per-fold detail is written to `metrics.json` next to `results.csv`, so
that every model in a comparison is scored with the same statistic computed in
the same way.

The standard deviation is the *sample* std (ddof=1). With 5 folds the
population std understates the spread by about 11%, and this number is read as
an error bar.

`koff_ceilings3d` gives the label-noise ceiling on R^2 for each off-diagonal
permeability component, estimated from the reciprocity residual K_ij - K_ji.

Usage
-----
    from metrics import fold_metrics, summarize, fmt
    folds = [fold_metrics(y_true, y_pred, fold=i)    # one dict per fold
             for i, (y_true, y_pred) in enumerate(per_fold_predictions)]
    print(fmt(summarize(folds)))
"""

import json
import os

import numpy as np

METRICS = ("r2", "mae", "mse")

__all__ = ["METRICS", "fold_metrics", "summarize", "write_metrics_json",
           "koff_ceilings3d"]


def fold_metrics(y_true, y_pred, fold=None):
    """R^2, MAE and MSE for one fold. Non-finite pairs are dropped."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]

    out = {"n_test": int(len(y_true))}
    if fold is not None:
        out = {"fold": int(fold), **out}
    if len(y_true) < 2:
        out.update({m: float("nan") for m in METRICS})
        return out

    err = y_true - y_pred
    ss = float(((y_true - y_true.mean()) ** 2).sum())
    out["r2"] = float(1.0 - (err ** 2).sum() / ss) if ss > 0 else float("nan")
    out["mae"] = float(np.abs(err).mean())
    out["mse"] = float((err ** 2).mean())
    return out


def summarize(folds):
    """Mean and sample std of each metric across folds -> flat dict."""
    out = {}
    for m in METRICS:
        v = np.array([f.get(m, np.nan) for f in folds], dtype=float)
        v = v[np.isfinite(v)]
        out[f"{m}_mean"] = float(v.mean()) if len(v) else float("nan")
        out[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


def fmt(summary):
    """One-line rendering for the console."""
    return (f"R2 {summary['r2_mean']:7.4f} ± {summary['r2_std']:.4f}   "
            f"MAE {summary['mae_mean']:.4g} ± {summary['mae_std']:.2g}   "
            f"MSE {summary['mse_mean']:.4g} ± {summary['mse_std']:.2g}")


def write_metrics_json(output_dir, meta, experiments):
    """
    Write `metrics.json`: every metric, every fold, every experiment in this run.

    `results.csv` keeps only the aggregates; this file allows per-fold analysis
    afterwards (for example a paired test between two feature groups) without
    retraining.
    """
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as fh:
        json.dump({**meta, "experiments": experiments}, fh, indent=2)
    return path


def koff_ceilings3d(df):
    """
    Upper bound on achievable R^2 for each 3D off-diagonal, from K = K^T.

    Stokes flow makes K exactly symmetric, so |K_ij - K_ji| is a direct
    estimate of the per-sample numerical error. The target k_ij is the mean of
    the two measurements, so its error is half that, and
    ceiling = 1 - mean(err^2) / var(k_ij).

    Returns {"k_xy": ceiling, ...}, or None if the columns are not 3D ones.
    """
    import numpy as np

    out = {}
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        need = {f"K_{a}{b}", f"K_{b}{a}", f"k_{a}{b}"}
        if not need <= set(df.columns):
            return None
        err = (df[f"K_{a}{b}"] - df[f"K_{b}{a}"]).abs() / 2.0
        var_sig = float(df[f"k_{a}{b}"].var())
        out[f"k_{a}{b}"] = (float(1.0 - float((err ** 2).mean()) / var_sig)
                            if var_sig > 0 else float("nan"))
    return out
