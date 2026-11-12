r"""
Non-topological directional baselines in 3D (the "Stats" descriptor family):
porosity profile, two-point correlation, fabric profile and fabric tensor.

All directional descriptors are evaluated along the same nine probing
directions as the topological ones (3 axes + 6 signed face diagonals, given as
integer vectors (h, k, l) in the solver frame), so the comparison with TDA and
THR is like for like. Every quantity is computed on the 3-torus, the domain
the LBM solves.

Descriptors
-----------
por_<tag>_<i>     Directional porosity profile: mean solid fraction in the
                  lattice planes perpendicular to d, indexed by
                  p = (h x + k y + l z) mod N. With gcd(h, k, l) = 1 this gives
                  N populated planes for every direction and wraps exactly on
                  the torus (binning the raw projection between its min and max
                  would treat the two ends of a diagonal as maximally distant,
                  although they are adjacent). Resampled to `--n_por` values.
tpc_<tag>_<i>     Directional two-point correlation S2(r) = <g(x) g(x + r d)>
                  by cumulative periodic rolls; S2(0) is the solid fraction.
                  The lag runs to floor(N / (2 |d|)), so diagonals cover fewer
                  integer steps than axes; all are resampled to `--n_tpc`.
fabdir_<tag>_<i>  Directional fabric profile: mean of (n.d)^2 over interface
                  voxels in each lattice plane, with n the interface normal
                  from periodic central differences. 1 means interfaces face
                  along d, 0 that they run parallel to it, 1/3 is isotropic.
fab_*             Fabric tensor F_ij = <n_i n_j> over interface voxels:
                  fab_xx .. fab_yz (6 components), fab_eig1..3 (eigenvalues,
                  descending) and fab_fa (fractional anisotropy).
                  No eigenvectors: they are sign-ambiguous, swap order when
                  eigenvalues cross and are degenerate for rods and plates,
                  whereas the tensor components are unambiguous and covariant.

Tags follow `tag_of`, e.g. `a1_-1_0` for [1,-1,0].

Input / output
--------------
Reads `<dataset>/structures.csv` and `<dataset>/structures/<filename>.raw`;
writes one CSV with a `sample_id` column and one column per feature.

Usage
-----
    python scripts/baselines3d.py --self-test
    python scripts/baselines3d.py --dataset DATA/aniso3d \
        --output DESC/aniso3d/baselines.csv [--n_por 80] [--n_tpc 41]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io3d import load_structure          # noqa: E402

# 3 axes + 6 signed face diagonals, as (dx, dy, dz) in the solver frame.
# The signed pairs are essential: k_xy is the contrast between [1,1,0] and
# [1,-1,0], so a set holding only one of the pair cannot express the sign of an
# off-diagonal.
DIRECTIONS = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
              (1, 1, 0), (1, -1, 0),
              (1, 0, 1), (1, 0, -1),
              (0, 1, 1), (0, 1, -1)]


def tag_of(hkl) -> str:
    """Direction tag used in column names, e.g. `a1_-1_0` for [1,-1,0]."""
    return "a" + "_".join(str(int(c)) for c in hkl)


def _plane_index(shape, hkl):
    """p = (h x + k y + l z) mod N, exact on the torus."""
    nz, ny, nx = shape
    if not (nx == ny == nz):
        raise ValueError(f"lattice-plane binning needs a cubic cell, got {shape}")
    h, k, l = hkl
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx]
    return (h * X + k * Y + l * Z) % nx, nx


def _interface_normals(solid):
    """Periodic central differences; returns (nx, ny, nz, mask) at interfaces."""
    g = solid.astype(np.float64)
    gz = 0.5 * (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0))
    gy = 0.5 * (np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1))
    gx = 0.5 * (np.roll(g, -1, axis=2) - np.roll(g, 1, axis=2))
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    m = mag > 1e-10
    return gx, gy, gz, mag, m


def porosity_profile(solid, hkl, n_out):
    """Mean solid fraction per lattice plane perpendicular to hkl."""
    p, N = _plane_index(solid.shape, hkl)
    tot = np.bincount(p.ravel(), weights=solid.ravel().astype(float), minlength=N)
    cnt = np.bincount(p.ravel(), minlength=N)
    prof = tot / np.maximum(cnt, 1)
    return np.interp(np.linspace(0, N - 1, n_out), np.arange(N), prof)


def two_point_correlation(solid, hkl, n_out):
    """S2(r) along hkl by cumulative periodic rolls, lag out to N / (2 |d|)."""
    g = solid.astype(np.float64)
    h, k, l = hkl
    N = solid.shape[2]
    max_steps = int(np.floor(N / (2.0 * np.sqrt(h * h + k * k + l * l))))
    vals = [float((g * g).mean())]
    rolled = g
    for _ in range(max_steps):
        rolled = np.roll(rolled, shift=(l, k, h), axis=(0, 1, 2))   # (z, y, x)
        vals.append(float((g * rolled).mean()))
    vals = np.asarray(vals)
    return np.interp(np.linspace(0, len(vals) - 1, n_out),
                     np.arange(len(vals)), vals)


def fabric_profile(solid, hkl, n_out, normals=None):
    """F_d(r) = mean over interface voxels in plane r of (n.d)^2."""
    gx, gy, gz, mag, m = normals if normals is not None else _interface_normals(solid)
    p, N = _plane_index(solid.shape, hkl)
    d = np.asarray(hkl, dtype=np.float64)
    d = d / np.linalg.norm(d)

    if not m.any():
        return np.full(n_out, np.nan)

    dot = (gx[m] * d[0] + gy[m] * d[1] + gz[m] * d[2]) / mag[m]
    tot = np.bincount(p[m], weights=dot * dot, minlength=N)
    cnt = np.bincount(p[m], minlength=N)
    good = cnt > 0
    prof = np.empty(N)
    prof[good] = tot[good] / cnt[good]
    if not good.all():
        # empty slabs filled by periodic interpolation between populated ones
        prof = np.interp(np.arange(N), np.flatnonzero(good), prof[good], period=N)
    return np.interp(np.linspace(0, N - 1, n_out), np.arange(N), prof)


def fabric_tensor(solid, normals=None):
    """
    F_ij = <n_i n_j> over interface voxels, plus eigenvalues and anisotropy.

    Returns (fxx, fyy, fzz, fxy, fxz, fyz, eig1 >= eig2 >= eig3, fa).
    Deliberately no eigenvectors; see the module docstring.
    """
    gx, gy, gz, mag, m = normals if normals is not None else _interface_normals(solid)
    if not m.any():
        return (np.nan,) * 10

    nx_, ny_, nz_ = gx[m] / mag[m], gy[m] / mag[m], gz[m] / mag[m]
    F = np.array([
        [(nx_ * nx_).mean(), (nx_ * ny_).mean(), (nx_ * nz_).mean()],
        [(nx_ * ny_).mean(), (ny_ * ny_).mean(), (ny_ * nz_).mean()],
        [(nx_ * nz_).mean(), (ny_ * nz_).mean(), (nz_ * nz_).mean()],
    ])
    w = np.sort(np.linalg.eigvalsh(F))[::-1]
    norm = np.linalg.norm(w)
    fa = (np.sqrt(1.5) * np.linalg.norm(w - w.mean()) / norm) if norm > 0 else 0.0
    return (F[0, 0], F[1, 1], F[2, 2], F[0, 1], F[0, 2], F[1, 2],
            w[0], w[1], w[2], float(fa))


FAB_NAMES = ["fab_xx", "fab_yy", "fab_zz", "fab_xy", "fab_xz", "fab_yz",
             "fab_eig1", "fab_eig2", "fab_eig3", "fab_fa"]


def _one(args):
    sample_id, path, n_por, n_tpc = args
    s = load_structure(path)
    normals = _interface_normals(s)
    row = {"sample_id": sample_id}
    for hkl in DIRECTIONS:
        t = tag_of(hkl)
        for i, v in enumerate(porosity_profile(s, hkl, n_por)):
            row[f"por_{t}_{i}"] = v
        for i, v in enumerate(two_point_correlation(s, hkl, n_tpc)):
            row[f"tpc_{t}_{i}"] = v
        for i, v in enumerate(fabric_profile(s, hkl, n_por, normals)):
            row[f"fabdir_{t}_{i}"] = v
    row.update(dict(zip(FAB_NAMES, fabric_tensor(s, normals))))
    return row


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    N = 40
    # plates normal to solver x: solid slabs at a lattice of x, full in y and z
    plates = np.zeros((N, N, N), dtype=np.uint8)
    for x0 in range(0, N, 10):
        plates[:, :, x0:x0 + 4] = 1

    px = porosity_profile(plates, (1, 0, 0), N)     # slabs of constant x
    py = porosity_profile(plates, (0, 1, 0), N)     # slabs of constant y
    check("porosity profile structured across the plates, flat along them",
          py.std() < 1e-12 < px.std(), f"std across={px.std():.3f} along={py.std():.2e}")

    s2x = two_point_correlation(plates, (1, 0, 0), 41)
    s2y = two_point_correlation(plates, (0, 1, 0), 41)
    check("S2(0) equals the solid fraction",
          abs(s2x[0] - plates.mean()) < 1e-12,
          f"{s2x[0]:.4f} vs {plates.mean():.4f}")
    check("S2 constant along the plates, structured across",
          s2y.std() < 1e-12 < s2x.std(), f"std across={s2x.std():.3f} along={s2y.std():.2e}")

    f = fabric_tensor(plates)
    check("fabric of x-normal plates has normals along x",
          f[0] > 0.99 and f[1] < 1e-9 and f[2] < 1e-9 and max(abs(v) for v in f[3:6]) < 1e-9,
          f"fxx={f[0]:.4f} fyy={f[1]:.2e} fzz={f[2]:.2e}")
    check("fabric of x-normal plates is maximally anisotropic",
          abs(f[9] - 1.0) < 1e-6, f"fa={f[9]:.6f}")

    fp = fabric_profile(plates, (1, 0, 0), N)
    check("fabric profile ~ 1 along the plate normal",
          np.nanmin(fp) > 0.99, f"min={np.nanmin(fp):.4f}")

    # permuting solver axes must permute the fabric tensor the same way
    swap_xz = np.ascontiguousarray(np.transpose(plates, (2, 1, 0)))
    g = fabric_tensor(swap_xz)
    check("swapping solver x and z swaps fxx and fzz",
          abs(g[2] - f[0]) < 1e-12 and abs(g[0] - f[2]) < 1e-12,
          f"fxx={g[0]:.2e} fzz={g[2]:.4f}")

    # periodicity: a diagonal profile must wrap exactly (N slabs, not 2N)
    rng = np.random.default_rng(0)
    r = (rng.random((N, N, N)) > 0.5).astype(np.uint8)
    for hkl in [(1, 1, 0), (1, -1, 0), (0, 1, -1)]:
        pr = porosity_profile(r, hkl, N)
        check(f"diagonal profile {tag_of(hkl)} wraps: mean = solid fraction",
              abs(pr.mean() - r.mean()) < 0.01,
              f"{pr.mean():.4f} vs {r.mean():.4f}")

    fr = fabric_tensor(r)
    check("fabric of an isotropic field is ~ diag(1/3)",
          max(abs(fr[i] - 1 / 3) for i in range(3)) < 0.02
          and max(abs(fr[i]) for i in range(3, 6)) < 0.02,
          f"fxx={fr[0]:.3f} fyy={fr[1]:.3f} fzz={fr[2]:.3f} fa={fr[9]:.3f}")
    fpr = fabric_profile(r, (1, 1, 0), N)
    check("fabric profile of an isotropic field is ~ 1/3",
          abs(np.nanmean(fpr) - 1 / 3) < 0.02, f"mean={np.nanmean(fpr):.4f}")

    # cyclic shift must shift the profile, not change its content
    sh = (3, -5, 7)
    pr0 = porosity_profile(r, (1, 1, 0), N)
    pr1 = porosity_profile(np.ascontiguousarray(np.roll(r, sh, axis=(0, 1, 2))),
                           (1, 1, 0), N)
    check("cyclic shift permutes the profile without changing its multiset",
          np.allclose(np.sort(pr0), np.sort(pr1)),
          f"max diff {np.abs(np.sort(pr0) - np.sort(pr1)).max():.2e}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--output")
    p.add_argument("--n_por", type=int, default=80)
    p.add_argument("--n_tpc", type=int, default=41)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.output):
        p.error("--dataset and --output are required (or use --self-test)")

    df = pd.read_csv(os.path.join(a.dataset, "structures.csv"))
    if a.limit:
        df = df.head(a.limit)
    tasks = [(r.sample_id, os.path.join(a.dataset, "structures", r.filename),
              a.n_por, a.n_tpc) for r in df.itertuples()]

    from tqdm import tqdm
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="baselines"):
            rows.append(f.result())

    out = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"  {len(out)} rows x {out.shape[1] - 1} features -> {a.output}")
    print(f"  NaNs: {int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
